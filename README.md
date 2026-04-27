# copilot-adapter

An OpenAI / Anthropic / Gemini-compatible LLM API proxy server backed by GitHub Copilot.

Authenticates via GitHub's device-flow OAuth (`ghu_` tokens), then proxies requests to GitHub Copilot's backend through a local server that speaks all three major LLM API formats.

## Key features

- [**Multi-account pooling**](#multi-account) — Rotate between multiple GitHub Copilot accounts to pool premium request quotas, with automatic exhaustion detection and account switching
- [**Per-account plan and quota**](#per-account-plan-and-quota) — Mix accounts on different Copilot tiers with per-account quota limits that auto-derive from the plan
- [**Smart premium request billing**](#premium-request-billing) — Automatically avoids extra premium request charges for agentic follow-ups, with no client-side changes needed
- [**Rate limit handling**](#premium-request-billing) — Automatically retries on rate limit errors by rotating to the next available account
- [**Three API formats**](#endpoints) — Serves OpenAI, Anthropic, and Gemini endpoints simultaneously
- [**Forward proxy mode**](#forward-proxy-mode) — Acts as an HTTP/HTTPS proxy that intercepts Copilot API traffic and rewrites billing headers, and transparently reroutes requests for OpenAI, Anthropic, and Gemini APIs through Copilot
- [**One-command tool setup**](#tool-configuration) — Automatically configure popular agentic coding tools (Claude Code, Codex, Gemini CLI, OpenCode) to use this proxy, with easy revert to defaults
- [**Configurable model mapping**](#model-mapping) — Built-in Claude model-ID normalization plus optional glob-pattern overrides
- [**Cross-provider reasoning effort mapping**](#parameter-compatibility) — Preserves Anthropic thinking / `output_config.effort` when requests are mapped to OpenAI-style models, including Responses-only targets like `gpt-5.5`
- [**Server-side web search**](#server-side-web-search) — Executes `web_search` tool calls server-side, using provider-native web search where available and a DuckDuckGo fallback otherwise. Claude requests with `web_search` can optionally be rerouted through another provider's native web search. Other unsupported built-in tool types are stripped.
- **Streaming support** — Full SSE streaming across all three formats, including real-time format translation
- [**Flexible authentication**](#authentication) — Interactive device-flow OAuth with cached tokens and multi-account support
- **Multi-worker support** — Spawns multiple worker processes for higher throughput
- **Concurrent-safe token management** — Only one token refresh happens at a time under concurrent load
- [**Docker ready**](#docker) — Pre-built image on [GHCR](https://github.com/blhsing/copilot-adapter/pkgs/container/copilot-adapter), or build locally
- **CORS support** — Configurable allowed origins for browser-based applications

## Prerequisites

- Python 3.10+
- pip
- A GitHub account with [GitHub Copilot](https://github.com/features/copilot) access (the free tier works; paid plans provide higher premium request quotas)

## Setup

```bash
pip install -r requirements.txt
```

`orjson` is listed as a dependency; if it fails to install on your platform, the adapter automatically falls back to the stdlib `json` module (expect higher memory use when serializing very large tool schemas).

## Usage

```bash
# Interactive device-flow login (opens browser) — required first-time setup
python copilot_adapter.py login
python copilot_adapter.py serve

# Options
python copilot_adapter.py serve --host 0.0.0.0 --port 18080

# Multiple worker processes for higher throughput (default: 1)
python copilot_adapter.py serve --workers 4

# Remove stored credentials
python copilot_adapter.py logout
```

Token lookup order: cached device-flow tokens > interactive device flow. `--github-token` and `COPILOT_ADAPTER_GITHUB_TOKEN` / `GITHUB_TOKEN` are still accepted for supplying OAuth tokens (`ghu_`) obtained out-of-band, but `ghp_` Personal Access Tokens are rejected with a warning — the Copilot API does not accept them.

### Per-account plan and quota

When accounts are on different Copilot tiers, append the plan and quota limit to the token with colons. (Device-flow login prompts for plan/quota interactively, but these may also be supplied via CLI/env for OAuth tokens obtained out-of-band.)

```bash
# TOKEN:PLAN:QUOTA format
python copilot_adapter.py serve \
  --github-token ghu_aaa:pro:300 \
  --github-token ghu_bbb:free:50

# TOKEN:PLAN:QUOTA:USAGE format (specify current premium usage)
python copilot_adapter.py serve \
  --github-token ghu_aaa:pro:300:150.5 \
  --github-token ghu_bbb:free:50:12

# Bare tokens fall back to the global --plan default (pro) and its quota (300)
python copilot_adapter.py serve \
  --github-token ghu_aaa:enterprise:1000 \
  --github-token ghu_bbb:free:50 \
  --github-token ghu_ccc
```

### Multi-account

Pool multiple GitHub Copilot accounts to extend your premium request quota:

```bash
# Add accounts via device-flow login (run multiple times)
python copilot_adapter.py login   # adds first account
python copilot_adapter.py login   # adds second account

# Or pass multiple OAuth tokens (ghu_) obtained out-of-band
python copilot_adapter.py serve --github-token ghu_aaa --github-token ghu_bbb

# Or comma-separated in an env var
export COPILOT_ADAPTER_GITHUB_TOKEN=ghu_aaa,ghu_bbb
python copilot_adapter.py serve

# List cached accounts (shows plan, quota, and usage)
python copilot_adapter.py accounts

# Add an OAuth token to the cache (with optional plan/quota/usage)
python copilot_adapter.py accounts --add ghu_xxx --plan pro --quota-limit 300 --usage 50

# Update plan/quota/usage for a cached account
python copilot_adapter.py accounts --update octocat --plan pro+ --quota-limit 1500 --usage 200

# Remove a cached account
python copilot_adapter.py accounts --remove octocat

# Remove all accounts
python copilot_adapter.py logout --all
```

**Rotation strategies** (`--strategy`):

| Strategy | Behavior | Pros | Cons |
|----------|----------|------|------|
| `max-usage` (default) | Concentrate all usage on one account until its quota is exhausted, then move to the next | Maximizes the number of accounts kept at zero usage as reserves; best server-side cache efficiency since all requests hit the same account's session; simple and predictable | One account bears all the load; if the month resets mid-use, the reserve accounts were never needed |
| `min-usage` | Always pick the account with the lowest usage | Spreads consumption evenly across all accounts; maximizes headroom on every account; reduces risk of hitting per-account rate limits | All accounts accumulate usage simultaneously, so none are kept clean as a reserve |
| `round-robin` | Rotate blindly on each user-initiated request | Simple and predictable; spreads load without needing usage data | No awareness of quota — won't avoid accounts nearing their limit |

Agent-initiated requests (tool-use follow-ups) always stay on the same account as the preceding user request to avoid unnecessary premium request charges.

**Quota exhaustion detection**: When a Copilot account's premium request quota is exhausted, GitHub silently downgrades the response to a free fallback model (e.g. GPT-4.1) instead of returning an error. The server detects this by comparing the model in the response against the model that was requested — if they don't match, it marks the account as exhausted and automatically retries the request with the next available account. This works for both streaming and non-streaming requests.

For proactive switching *before* hitting the limit, set `--quota-limit N` or let it default from the plan. Usage is tracked in-memory with plan-aware model cost multipliers (e.g. Claude Opus 4.7 costs 3x, GPT-4o costs 0x on paid plans). You can specify each account's current usage via the `TOKEN:PLAN:QUOTA:USAGE` format, `--usage` flag, or config file to start tracking from where you left off. These defaults can be overridden per account — see [Per-account plan and quota](#per-account-plan-and-quota).

**Supported plans** (`--plan`):

| Plan | Monthly premium requests | Model multipliers |
|------|------------------------:|-------------------|
| `free` | 50 | All models cost 1x |
| `pro` (default) | 300 | Differentiated (e.g. GPT-4o: 0x, Claude Opus: 3x) |
| `pro+` | 1500 | Same as `pro` |
| `business` | 300 | Same as `pro` |
| `enterprise` | 1000 | Same as `pro` |

When `--quota-limit` is not specified, it defaults to the plan's monthly allowance.

### Config file

All settings can be placed in a JSON config file instead of (or alongside) CLI flags and environment variables. The server looks for `~/.config/copilot-adapter/config.json` by default, or you can specify a path with `--config`:

```bash
python copilot_adapter.py serve --config /path/to/config.json
```

Example `~/.config/copilot-adapter/config.json`:

```json
{
  "host": "0.0.0.0",
  "port": 18080,
  "strategy": "max-usage",
  "plan": "pro",
  "log_file": "/path/to/copilot-adapter.log",
  "free": false,
  "free_within_minutes": 5,
  "proxy": false,
  "proxy_user": "myuser",
  "proxy_password": "mypassword",
  "workers": 4,
  "cors_origins": ["*"],
  "model_map": {
    "*sonnet*": "claude-sonnet-4.6",
    "gpt-4-turbo": "gpt-4-0125-preview"
  },
  "api_tokens": ["sk-abc123...", "sk-def456..."],
  "web_search_iterations": 3,
  "accounts": [
    {"token": "ghu_aaa", "plan": "enterprise", "quota_limit": 1000, "premium_used": 250},
    {"token": "ghu_bbb", "plan": "free"},
    "ghu_ccc:pro+:1500:100.5",
    "ghu_ddd"
  ]
}
```

Account entries in the `accounts` array can be:
- **Objects** with `token` (required), `plan`, `quota_limit`, and `premium_used` (all optional) fields
- **Strings** in `TOKEN:PLAN:QUOTA:USAGE` format (same as the CLI `--github-token` syntax)
- **Bare token strings** that fall back to the top-level `plan` and `quota_limit` defaults

**Precedence** (highest to lowest): CLI flags > environment variables > config file > built-in defaults.

### Environment variables

All CLI options can be set via environment variables:

| Flag | Environment variable | Default |
|------|---------------------|---------|
| `--config` | `COPILOT_ADAPTER_CONFIG` | `~/.config/copilot-adapter/config.json` |
| `--host` | `COPILOT_ADAPTER_HOST` | `127.0.0.1` |
| `--port` | `COPILOT_ADAPTER_PORT` | `18080` |
| `--github-token` | `COPILOT_ADAPTER_GITHUB_TOKEN` | *(none)* |
| `--cors-origin` | `COPILOT_ADAPTER_CORS_ORIGIN` | *(none)* |
| `--workers` | `COPILOT_ADAPTER_WORKERS` | `1` |
| `--strategy` | `COPILOT_ADAPTER_STRATEGY` | `max-usage` |
| `--quota-limit` | `COPILOT_ADAPTER_QUOTA_LIMIT` | per plan |
| `--plan` | `COPILOT_ADAPTER_PLAN` | `pro` |
| `--log-level` | `COPILOT_ADAPTER_LOG_LEVEL` | `info` |
| `--log-file` | `COPILOT_ADAPTER_LOG_FILE` | *(none)* |
| `--free` | `COPILOT_ADAPTER_FREE` | *(off)* |
| `--free-within-minutes` | `COPILOT_ADAPTER_FREE_WITHIN_MINUTES` | *(off)* |
| `--stub-bill` | `COPILOT_ADAPTER_STUB_BILL` | *(off)* |
| `--stub-model` | `COPILOT_ADAPTER_STUB_MODEL` | `claude-haiku-4.5` |
| `--proxy` | `COPILOT_ADAPTER_PROXY` | *(off)* |
| `--ca-dir` | `COPILOT_ADAPTER_CA_DIR` | `~/.config/copilot-adapter` |
| `--model-map` | `COPILOT_ADAPTER_MODEL_MAP` | *(none — Claude IDs auto-normalized)* |
| `--proxy-user` | `COPILOT_ADAPTER_PROXY_USER` | *(none)* |
| `--proxy-password` | `COPILOT_ADAPTER_PROXY_PASSWORD` | *(none)* |
| `--api-token` | `COPILOT_ADAPTER_API_TOKEN` | stored tokens |
| `--web-search-iterations` | `COPILOT_ADAPTER_WEB_SEARCH_ITERATIONS` | `3` |
| `--force-ddg-web-search` | `COPILOT_ADAPTER_FORCE_DDG_WEB_SEARCH` | *(off)* |
| `--web-search-model` | `COPILOT_ADAPTER_WEB_SEARCH_MODEL` | *(none)* |
| `--reverse-dns-server` | `COPILOT_ADAPTER_REVERSE_DNS_SERVER` | *(system resolver)* |
| `--forwarded-allow-ips` | `COPILOT_ADAPTER_FORWARDED_ALLOW_IPS` | `127.0.0.1` |

Set `NO_COLOR=1` to disable colored log output. Colors are auto-detected on Windows (requires Windows Terminal or VT-enabled console).

Use `--log-file /path/to/copilot-adapter.log` (or `log_file` in the config file) to append the same logs to a file while keeping console output enabled.

Access log lines show the originating client as `hostname (ip:port)` when a reverse DNS lookup succeeds, and fall back to the raw `ip:port` otherwise. Lookups run asynchronously on a background thread and are cached per IP, so the first request from a new address is logged with just the IP while the resolution is in flight. Use `--reverse-dns-server` (or `COPILOT_ADAPTER_REVERSE_DNS_SERVER`) to point the lookups at a specific DNS server — useful when the system resolver can't see an internal zone that maps the originating hosts. When requests arrive through an HTTP proxy such as Squid, the TCP peer is the proxy; set `--forwarded-allow-ips` (or `COPILOT_ADAPTER_FORWARDED_ALLOW_IPS`) to the proxy's IP (or `*`) so the `X-Forwarded-For` header is honored and the log shows the real client.

`GITHUB_TOKEN` is also accepted as a fallback for the GitHub token. Multiple tokens can be comma-separated in `COPILOT_ADAPTER_GITHUB_TOKEN` or `GITHUB_TOKEN`.

### Docker

A pre-built image is available on GitHub Container Registry:

```bash
docker pull ghcr.io/blhsing/copilot-adapter:latest
```

```bash
# First-time setup: run device-flow login with a persistent volume, then start the daemon
docker run --rm -it \
  -v copilot-adapter-config:/root/.config/copilot-adapter \
  ghcr.io/blhsing/copilot-adapter login

# Run the server using the cached OAuth token
docker run -p 18080:18080 \
  -v copilot-adapter-config:/root/.config/copilot-adapter \
  ghcr.io/blhsing/copilot-adapter

# Multi-account with rotation (login multiple times, then serve)
docker run -p 18080:18080 \
  -v copilot-adapter-config:/root/.config/copilot-adapter \
  -e COPILOT_ADAPTER_STRATEGY=max-usage \
  -e COPILOT_ADAPTER_QUOTA_LIMIT=300 \
  ghcr.io/blhsing/copilot-adapter
```

Or build locally:

```bash
docker build -t copilot-adapter .
docker run -p 18080:18080 \
  -v copilot-adapter-config:/root/.config/copilot-adapter \
  copilot-adapter
```

> PATs (`ghp_`) passed via `COPILOT_ADAPTER_GITHUB_TOKEN` / `GITHUB_TOKEN` will be rejected — the Copilot API returns 404 for Personal Access Tokens. Use device-flow login instead (see above).

## Endpoints

### [OpenAI](https://platform.openai.com/docs/api-reference)

```
POST /v1/chat/completions
POST /v1/responses
GET  /v1/models
POST /v1/embeddings
```

### [Anthropic](https://docs.anthropic.com/en/api)

```
POST /v1/messages
POST /v1/messages/count_tokens
```

### [Gemini](https://ai.google.dev/api)

```
POST /v1beta/models/{model}:generateContent
POST /v1beta/models/{model}:streamGenerateContent
GET  /v1beta/models
GET  /v1beta/models/{model}
```

All endpoints support streaming.

## Premium request billing

GitHub Copilot uses the `X-Initiator` header to determine whether an API call counts as a premium request:

- `X-Initiator: user` — counts as a premium request
- `X-Initiator: agent` — free (treated as an autonomous agent follow-up)

The proxy handles this automatically. When no `X-Initiator` header is provided by the caller, it inspects the request body and infers the correct value:

- **OpenAI format** — `agent` if the last message has `role: "tool"`, or if any prior message contains tool calls or tool responses
- **Anthropic format** — `agent` if the last message contains a `tool_result` content block, or if any prior assistant message contains a `tool_use` block
- **Gemini format** — `agent` if the last turn contains a `functionResponse` part, or if any prior turn contains a `functionCall` or `functionResponse`
- Otherwise — `user`

This means agentic clients like Claude Code that make multiple API calls per user turn (tool-use loops, retries, subagent spawns, auto-continues) will only consume one premium request for the initial prompt — follow-up calls are automatically marked as `agent`. No client-side changes needed.

Callers can also pass `X-Initiator` explicitly to override the heuristic.

### Free mode

Use `--free` to mark **all** requests as agent-initiated, so nothing counts as a premium request:

```bash
python copilot_adapter.py serve --free
```

This is useful when you want to avoid all premium billing regardless of request type. Note that GitHub Copilot may throttle or deprioritize agent-initiated requests compared to user-initiated ones.

### Time-based free mode

Use `--free-within-minutes N` to mark a user-initiated request as agent-initiated only if the last request to the same account was less than N minutes ago:

```bash
python copilot_adapter.py serve --free-within-minutes 5
```

The logic: the first request in a session is billed normally (as `user`), but subsequent requests within the time window are marked as `agent` (free). Once the account has been idle for longer than N minutes, the next request is treated as a new session and billed normally again.

This is useful when you want to limit premium billing to one request per session rather than eliminating it entirely. It's mutually exclusive with `--free`.

When using multi-account rotation, agent-initiated requests always stay on the same account as the preceding user request to avoid billing a premium request on a different account.

### Stub-bill mode

Use `--stub-bill` to route billing through a cheap stub model. For each user-initiated request, the adapter first fires a tiny billed call (one-token `"test"` prompt) against `--stub-model` (default `claude-haiku-4.5`), then runs the real request marked as agent-initiated:

```bash
python copilot_adapter.py serve --stub-bill
# or with a custom stub model
python copilot_adapter.py serve --stub-bill --stub-model claude-haiku-4.5
```

This satisfies Copilot's "one premium request per user turn" accounting with the cheapest eligible model, while the actual (potentially expensive) request runs as agent — so a large opus-4.7 call effectively costs one haiku premium request instead of one opus premium request.

Costs: the stub call runs in the background concurrently with the real request, so there is no added latency. If the stub call fails (rate limit, 5xx, quota exhausted), the real request has already gone out as agent and is not billed — the billing slot is skipped for that turn, but the user request still completes. Independent from `--free` / `--free-within-minutes`: when those already demote the request to agent, the stub is skipped.

## Forward proxy mode

Use `--proxy` to enable a forward HTTP/HTTPS proxy on the same port as the API server. In this mode, the server handles both normal API requests (reverse proxy) and forwarded client traffic (forward proxy) on a single port:

```bash
python copilot_adapter.py serve --proxy
```

The proxy intercepts HTTPS connections to the following hosts:

- **`api.githubcopilot.com`** — rewrites `X-Initiator: user` to `agent` so requests are not billed as premium
- **`api.openai.com`**, **`api.anthropic.com`**, **`generativelanguage.googleapis.com`** — LLM API requests are redirected to the local adapter and routed through Copilot; non-API requests (e.g. update checks, MCP registry) are forwarded to the original host

All other traffic is tunneled transparently. If `HTTPS_PROXY` or `HTTP_PROXY` is set, outbound connections are chained through the upstream proxy.

**Client setup:**

```bash
export HTTPS_PROXY=http://127.0.0.1:18080
export NODE_EXTRA_CA_CERTS=~/.config/copilot-adapter/ca.pem
```

A self-signed CA certificate is generated automatically on first use and stored in `~/.config/copilot-adapter/` (or the directory specified by `--ca-dir`). Use `ca-cert` to generate the CA ahead of time or show its path:

```bash
python copilot_adapter.py ca-cert
# CA certificate: ~/.config/copilot-adapter/ca.pem
#   Subject:  CN=copilot-adapter MITM CA
#   Valid:    2026-04-07 to 2036-04-05
```

The client must trust this CA for HTTPS interception to work:

- **Node.js clients** (e.g. Claude Code): set `NODE_EXTRA_CA_CERTS` to the CA certificate path
- **Electron apps** (e.g. Claude Desktop) and **browsers**: install the CA in the system trust store:
  ```powershell
  # Windows (run as Administrator)
  certutil -addstore Root "%USERPROFILE%\.config\copilot-adapter\ca.pem"
  ```

This mode is useful when you want to transparently reduce premium billing for any client that supports `HTTPS_PROXY`, without changing the client's API endpoint configuration.

## Model mapping

Model names in incoming requests are rewritten before being sent to the Copilot API.

**Built-in Claude normalization** — Copilot uses dotted version numbers for Claude models (e.g. `claude-opus-4.7`) while clients like Claude Code send hyphenated, date-suffixed names (e.g. `claude-opus-4-7-20260215`). The adapter always normalizes Claude model IDs automatically: the dash after the first major version becomes a dot, and trailing `-<digits>` segments are dropped. Non-digit suffixes (e.g. `-fast`) are preserved.

Examples:
- `claude-opus-4-7-20260215` → `claude-opus-4.7`
- `claude-opus-4-7` → `claude-opus-4.7`
- `claude-haiku-4-5-20251001` → `claude-haiku-4.5`
- `claude-opus-4-6-fast` → `claude-opus-4.6-fast`

**User-configured mappings** — You can add your own glob-pattern mappings (for cross-provider remaps, deprecated model IDs, etc.). When a user-configured pattern matches, it takes precedence over the built-in Claude normalization.

Configure via any of these (highest precedence first):

1. **CLI / env var** — repeatable `--model-map` flag or comma-separated env var:

   ```bash
   python copilot_adapter.py serve \
     --model-map 'gpt-4-turbo=gpt-4-0125-preview'

   # Or via environment variable (comma-separated)
   export COPILOT_ADAPTER_MODEL_MAP='gpt-4-turbo=gpt-4-0125-preview'
   ```

2. **Config file** — add a `model_map` object to the [config file](#config-file):

   ```json
   {
     "model_map": {
       "gpt-4-turbo": "gpt-4-0125-preview"
     }
   }
   ```

Patterns use glob syntax (`*` matches anything) and are checked in order — the first match wins. If no pattern matches, Claude-family IDs are auto-normalized (above); other models pass through unchanged. Model mapping is applied to all endpoints (chat completions, responses, embeddings, Gemini).

For Anthropic `/v1/messages` requests, the adapter also uses the final mapped model to choose the upstream Copilot endpoint:

- **Anthropic target** (for example `claude-sonnet-4.6`) — proxied natively to Anthropic Messages
- **Responses-only OpenAI target** (for example `gpt-5.5`) — converted to OpenAI `/v1/responses`
- **Other OpenAI-compatible targets** — converted to `/v1/chat/completions`

This preserves Anthropic features like thinking / `output_config.effort` when a Claude client is mapped to a Responses-only OpenAI model.


## Authentication

### API token protection

Protect the reverse API proxy with Bearer tokens so only authorized clients can use it:

```bash
# Generate a token
python copilot_adapter.py tokens --generate
python copilot_adapter.py tokens --generate --label "my-laptop"

# List tokens
python copilot_adapter.py tokens

# Revoke a token by value or label
python copilot_adapter.py tokens --revoke sk-abc123...
python copilot_adapter.py tokens --revoke my-laptop
```

Once tokens exist (via `tokens --generate`, `--api-token` flag, or `api_tokens` in the config file), all API endpoints except the health check (`GET /`) require `Authorization: Bearer <token>`:

```bash
# Start with stored tokens (generated via `tokens --generate`)
python copilot_adapter.py serve

# Or pass tokens explicitly
python copilot_adapter.py serve --api-token sk-abc123...

# Client usage
curl -H "Authorization: Bearer sk-abc123..." http://127.0.0.1:18080/v1/models
```

If no tokens are configured, the API is unprotected (open access).

### Forward proxy authentication

Protect the forward proxy with HTTP Basic authentication:

```bash
python copilot_adapter.py serve --proxy --proxy-user myuser --proxy-password mypass
```

Clients must include credentials in the proxy URL:

```bash
export HTTPS_PROXY=http://myuser:mypass@127.0.0.1:18080
```

Proxy authentication only applies to forward proxy requests (CONNECT and absolute-URL requests). Direct API requests use Bearer token authentication instead.

## Tool configuration

The `config` subcommand automatically configures popular agentic coding tools to point at this proxy:

```bash
# Configure a tool to use the proxy
python copilot_adapter.py config claude-code
python copilot_adapter.py config codex
python copilot_adapter.py config gemini-cli
python copilot_adapter.py config opencode

# Revert a tool back to its default provider
python copilot_adapter.py config claude-code --revert
python copilot_adapter.py config codex --revert

# Specify a custom host/port or API token
python copilot_adapter.py config claude-code --host 0.0.0.0 --port 8080 --api-token sk-abc123...
```

Supported tools:

| Tool | Config file | What it sets |
|------|------------|-------------|
| `claude-code` | `~/.claude/settings.json` | `ANTHROPIC_BASE_URL`, `ANTHROPIC_API_KEY` in the `env` block |
| `codex` | `~/.codex/config.toml` | `model_provider` and `[model_providers.copilot-adapter]` section |
| `gemini-cli` | `~/.gemini/settings.json` | `baseUrl` and `apiKey` fields |
| `opencode` | `~/.config/opencode/opencode.json` | `copilot-adapter` provider block |

A `.copilot-adapter.bak` backup is created before modifying any config file. When reverting, the backup is restored if it exists; otherwise the added keys are removed.

If `--api-token` is not specified, the first stored API token (from `tokens --generate`) is used automatically, if any.

### Manual client configuration

You can also point any OpenAI, Anthropic, or Gemini SDK client at the local server manually:

```bash
# OpenAI
export OPENAI_BASE_URL=http://127.0.0.1:18080/v1
export OPENAI_API_KEY=unused

# Anthropic
export ANTHROPIC_BASE_URL=http://127.0.0.1:18080
export ANTHROPIC_API_KEY=unused

# Gemini
export GEMINI_API_BASE=http://127.0.0.1:18080/v1beta
```

## Server-side web search

**Claude-targeted Anthropic requests.** When an Anthropic `/v1/messages` request targets a Claude model and includes Anthropic's built-in `web_search_20250305` tool, the adapter converts that request to the chat-completions path and intercepts `web_search` server-side using [DuckDuckGo](https://github.com/deedy5/ddgs) (`ddgs` package). Claude models do not use Copilot's native Anthropic web-search passthrough by default.

**Supported OpenAI Responses models.** When an Anthropic request is mapped to a GPT-5 family model (`gpt-5`, `gpt-5.x`, `gpt-5*-mini`, `gpt-5*-codex`, etc.), all of which support native web search on `/v1/responses`, the adapter preserves native web search by sending `web_search_preview` upstream instead of intercepting it locally.

**Force-DDG override.** Set `--force-ddg-web-search` (or `COPILOT_ADAPTER_FORCE_DDG_WEB_SEARCH=1`) to disable provider-native web search and force DuckDuckGo interception wherever the adapter would otherwise preserve native search.

**Native web search via a helper model.** Set `--web-search-model MODEL` (e.g. `gpt-5.5`, or `COPILOT_ADAPTER_WEB_SEARCH_MODEL=gpt-5.5`) to reroute Anthropic `/v1/messages` requests that carry `web_search_20250305` through `/v1/responses` against `MODEL` whenever the mapped target model lacks native provider web search. The upstream call then uses OpenAI's native `web_search_preview` tool instead of DuckDuckGo. `--force-ddg-web-search` overrides this; if `MODEL` does not itself support native web search the option is ignored.

For Anthropic clients, the adapter emits Anthropic-native `server_tool_use` and `web_search_tool_result` content blocks on both the DDG-intercepted path and the `--web-search-model` native-search reroute. On the native path, URL citations are pulled from the upstream Responses API's `url_citation` annotations — queries that hit Copilot's web index (news, docs, general facts) return structured source URLs; queries that Copilot routes to its internal knowledge APIs (e.g. weather, finance) return no URLs and the tool result block is empty but still emitted.

For non-Anthropic clients, the client still does not see the intermediate tool call.

This enables web search even when the client doesn't execute web-search tool calls itself. It works with both streaming and non-streaming requests.

If the model returns `web_search` alongside other tool calls, the adapter passes all tool calls through to the client instead of intercepting.

The model may call `web_search` multiple times in a single request (e.g. refining its query). The adapter allows up to 3 iterations by default, configurable with `--web-search-iterations N` (or `web_search_iterations` in the config file). Set to 0 to disable server-side interception entirely and pass `web_search` calls through to the client.

**Proxy support:** If `HTTPS_PROXY` or `HTTP_PROXY` environment variables are set, DuckDuckGo searches are routed through the proxy.

## Anthropic built-in tools

Anthropic clients (e.g. Claude Code) may send built-in tool types like `web_search_20250305`, `text_editor_20250124`, and `code_execution_20250522`. The adapter handles them as follows:

- **`web_search_*`** — Converted to DDG-backed server-side interception for Claude targets; preserved as native `web_search_preview` only for supported OpenAI Responses targets; `--force-ddg-web-search` forces DDG interception instead
- **Other built-in types** (e.g. `text_editor_*`, `code_execution_*`) — Stripped from the request, since these are handled client-side and don't need to be sent to the model

## Parameter compatibility

The proxy normalizes provider-specific request parameters after model mapping so cross-provider remaps keep working.

- **Token limits** — Some targets require different token-limit fields and minimums. The proxy automatically uses the correct field based on the final mapped model and endpoint: `max_tokens` for Claude and Gemini targets, `max_completion_tokens` for OpenAI chat-completions targets, and `max_output_tokens` for OpenAI Responses targets. For Responses targets, very small Anthropic `max_tokens` values are raised to the upstream minimum when required.
- **Reasoning / thinking effort** — When an Anthropic request includes thinking settings and is mapped to an OpenAI-style target, the proxy converts that intent to the nearest OpenAI reasoning effort. For example, Claude Code effort `max` mapped to `gpt-5.5` becomes reasoning effort `xhigh`.
- **Endpoint selection for mapped Anthropic requests** — Anthropic `/v1/messages` requests are routed to the upstream endpoint required by the mapped target model, so Responses-only models such as `gpt-5.5` keep reasoning effort and tool support instead of falling back to `/v1/chat/completions`.

Examples:
- Anthropic `output_config.effort: high` → OpenAI `reasoning_effort: high`
- Anthropic `output_config.effort: max` → OpenAI `reasoning_effort: xhigh`
- Anthropic client mapped to `gpt-5.5` → upstream `/v1/responses`

**Effort clamping for native Anthropic passthrough** — Copilot restricts the effort levels accepted for Anthropic models more tightly than Anthropic's direct API. The proxy clamps unsupported values automatically:
- `claude-opus-4.7`: Copilot only accepts `medium`; any other effort (`low`, `high`, `max`, `xhigh`) is clamped to `medium`.
- Other Anthropic models: `max` and `xhigh` are clamped to `high` (the highest level Copilot accepts).

This normalization is based on the final mapped model, so it works even when model mapping redirects requests across providers.

## Available models

Run `python copilot_adapter.py serve` and visit `http://127.0.0.1:18080/v1/models` to see all models available through your Copilot subscription. Models include offerings from OpenAI, Anthropic, Google, and xAI.

Note: some newer models (e.g. `gpt-5.5`) only support the `/v1/responses` endpoint, not `/v1/chat/completions`. The adapter handles this automatically for Anthropic `/v1/messages` requests after model mapping.

## Tests

The test suite runs integration tests against the live Copilot API, plus unit tests for account rotation logic.

```bash
pip install -r tests/requirements.txt

# Run all tests (authenticates on first run via device flow)
python -m pytest

# Run a specific test module
python -m pytest tests/test_client.py
python -m pytest tests/test_adapters.py
python -m pytest tests/test_endpoints.py
python -m pytest tests/test_account_manager.py
```

Tests are organized into four modules:

- **`test_client.py`** — `CopilotClient` directly: models, chat completions, streaming, responses API, embeddings
- **`test_adapters.py`** — format adapters end-to-end: OpenAI passthrough, Anthropic Messages, Gemini generateContent (streaming + non-streaming, multi-turn, system messages, parameter mapping)
- **`test_endpoints.py`** — FastAPI routes via ASGI transport: all endpoints across all three API formats
- **`test_account_manager.py`** — account rotation strategies, agent stickiness, exhaustion detection (unit tests, no auth required)

Tests use the cheapest available models (`gpt-4o-mini` for chat, `gpt-5-mini` for responses, `text-embedding-3-small` for embeddings) to minimize premium request usage. Model constants are centralized in `tests/conftest.py`.

## How it works

1. **Device flow OAuth** authenticates with GitHub and stores tokens in `~/.config/copilot-adapter/tokens.json`
2. GitHub tokens are exchanged for short-lived Copilot API tokens via `api.github.com/copilot_internal/v2/token`, automatically refreshed every ~25 minutes with concurrent-access protection (double-checked locking ensures only one refresh happens at a time)
3. For multi-account setups, the `AccountManager` selects which account to use based on the configured rotation strategy, sticking to the same account for agent-initiated follow-ups
4. Incoming requests are translated (if needed) to the format Copilot expects, model names are rewritten via the configurable model map, the correct upstream endpoint is selected (`/v1/messages`, `/v1/chat/completions`, or `/v1/responses`), and responses are translated back to the client's expected format
5. In forward proxy mode (`--proxy`), the server also accepts `CONNECT` tunnels on the same port — traffic to `api.githubcopilot.com` is MITM'd to rewrite billing headers, traffic to OpenAI/Anthropic/Gemini APIs is redirected to the local adapter, and all other traffic is tunneled transparently

## Known issues

### PowerShell enters debug mode after Ctrl+C (Windows 10)

On Windows 10, pressing Ctrl+C to stop the server may cause PowerShell to enter debug mode with a message like:

```
Entering debug mode. Type 'h' or '?' for help.
```

This is a [known bug in PSReadLine 2.0.0](https://github.com/PowerShell/PSReadLine/issues/1193) bundled with Windows 10. To fix it, upgrade PSReadLine in an elevated PowerShell:

```powershell
Install-Module PSReadLine -Force -SkipPublisherCheck
```

If you're behind an HTTP proxy:

```powershell
[System.Net.WebRequest]::DefaultWebProxy = New-Object System.Net.WebProxy("http://proxy-host:port")
[System.Net.WebRequest]::DefaultWebProxy.Credentials = [System.Net.CredentialCache]::DefaultCredentials
Install-Module PSReadLine -Force -SkipPublisherCheck
```

Restart PowerShell after upgrading. This issue does not affect Windows 11, Windows Terminal, or cmd.exe.

### PATs don't work for the Copilot API

GitHub Personal Access Tokens (`ghp_` prefix) are **not accepted** by the Copilot API. The `api.github.com/copilot_internal/v2/token` endpoint returns 404 for any PAT — this is true regardless of whether the account's Copilot seat is personal, Business, or Enterprise. Only OAuth tokens obtained via the device flow (`ghu_` prefix) authenticate against this endpoint.

The adapter detects PATs at startup and rejects them with a warning; if a PAT is the only credential supplied, the server will refuse to start.

**Symptom** (if a PAT somehow reaches the token-exchange step, e.g. via an older cached entry):

```
Error: Failed to get Copilot token: Client error '404 Not Found' for url 'https://api.github.com/copilot_internal/v2/token'
```

**Fix:** Use the device flow:

```bash
python copilot_adapter.py login
```

If you previously added a PAT to the cache, remove it first:

```bash
python copilot_adapter.py accounts --remove <username>
python copilot_adapter.py login
```

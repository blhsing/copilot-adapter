# copilot-adapter

An OpenAI / Anthropic / Gemini-compatible LLM API proxy server backed by GitHub Copilot.

Authenticates via a GitHub Personal Access Token (PAT) or GitHub's device flow, then proxies requests to GitHub Copilot's backend through a local server that speaks all three major LLM API formats.

## Key features

- **Multi-account pooling** — Rotate between multiple GitHub Copilot accounts to pool premium request quotas, with automatic exhaustion detection and account switching
- **Per-account plan and quota** — Mix accounts on different Copilot tiers with per-account quota limits that auto-derive from the plan
- **Smart premium request billing** — Automatically avoids extra premium request charges for agentic follow-ups, with no client-side changes needed
- **Rate limit handling** — Automatically retries on rate limit errors by rotating to the next available account
- **Three API formats** — Serves OpenAI, Anthropic, and Gemini endpoints simultaneously
- **Forward proxy mode** — Acts as an HTTP/HTTPS proxy that intercepts Copilot API traffic and rewrites billing headers, so any client that supports `HTTPS_PROXY` can benefit without reconfiguration
- **Configurable model mapping** — Glob-pattern-based model name rewriting via a shipped `model_map.json`, CLI flags, env vars, or config file
- **Streaming support** — Full SSE streaming across all three formats, including real-time format translation
- **Flexible authentication** — Supports multiple GitHub PATs, environment variables, cached tokens, and interactive device-flow OAuth, with automatic fallback
- **Multi-worker support** — Spawns multiple worker processes for higher throughput
- **Concurrent-safe token management** — Only one token refresh happens at a time under concurrent load
- **Docker ready** — Pre-built image on [GHCR](https://github.com/blhsing/copilot-adapter/pkgs/container/copilot-adapter), or build locally
- **CORS support** — Configurable allowed origins for browser-based applications

## Prerequisites

- Python 3.10+
- pip
- A GitHub account with [GitHub Copilot](https://github.com/features/copilot) access (the free tier works; paid plans provide higher premium request quotas)

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Start the server with a GitHub PAT (no interactive login needed)
python copilot_adapter.py serve --github-token ghp_xxx

# Or use an environment variable
export COPILOT_ADAPTER_GITHUB_TOKEN=ghp_xxx
python copilot_adapter.py serve

# Interactive device-flow login (opens browser)
python copilot_adapter.py login
python copilot_adapter.py serve

# Options
python copilot_adapter.py serve --host 0.0.0.0 --port 18080

# Multiple worker processes for higher throughput (default: number of CPUs)
python copilot_adapter.py serve --workers 4

# Remove stored credentials
python copilot_adapter.py logout
```

Token lookup order: `--github-token` flag > `COPILOT_ADAPTER_GITHUB_TOKEN` env var > `GITHUB_TOKEN` env var > cached tokens > interactive device flow.

### Per-account plan and quota

When accounts are on different Copilot tiers, append the plan and quota limit to the token with colons:

```bash
# TOKEN:PLAN:QUOTA format
python copilot_adapter.py serve \
  --github-token ghp_aaa:pro:300 \
  --github-token ghp_bbb:free:50

# TOKEN:PLAN:QUOTA:USAGE format (specify current premium usage)
python copilot_adapter.py serve \
  --github-token ghp_aaa:pro:300:150.5 \
  --github-token ghp_bbb:free:50:12

# Bare tokens fall back to the global --plan default (pro) and its quota (300)
python copilot_adapter.py serve \
  --github-token ghp_aaa:enterprise:1000 \
  --github-token ghp_bbb:free:50 \
  --github-token ghp_ccc
```

### Multi-account

Pool multiple GitHub Copilot accounts to extend your premium request quota:

```bash
# Add accounts via device-flow login (run multiple times)
python copilot_adapter.py login   # adds first account
python copilot_adapter.py login   # adds second account

# Or pass multiple PATs
python copilot_adapter.py serve --github-token ghp_aaa --github-token ghp_bbb

# Or comma-separated in an env var
export COPILOT_ADAPTER_GITHUB_TOKEN=ghp_aaa,ghp_bbb
python copilot_adapter.py serve

# List cached accounts (shows plan, quota, and usage)
python copilot_adapter.py accounts

# Add a PAT to the cache (with optional plan/quota/usage)
python copilot_adapter.py accounts --add ghp_xxx --plan pro --quota-limit 300 --usage 50

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

For proactive switching *before* hitting the limit, set `--quota-limit N` or let it default from the plan. Usage is tracked in-memory with plan-aware model cost multipliers (e.g. Claude Opus 4.6 costs 3x, GPT-4o costs 0x on paid plans). You can specify each account's current usage via the `TOKEN:PLAN:QUOTA:USAGE` format, `--usage` flag, or config file to start tracking from where you left off. These defaults can be overridden per account — see [Per-account plan and quota](#per-account-plan-and-quota).

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

All settings can be placed in a JSON config file instead of (or alongside) CLI flags and environment variables. The server looks for `~/.config/copilot-api/config.json` by default, or you can specify a path with `--config`:

```bash
python copilot_adapter.py serve --config /path/to/config.json
```

Example `~/.config/copilot-api/config.json`:

```json
{
  "host": "0.0.0.0",
  "port": 18080,
  "strategy": "max-usage",
  "plan": "pro",
  "free": false,
  "proxy": false,
  "workers": 4,
  "cors_origins": ["*"],
  "model_map": {
    "*sonnet*": "claude-sonnet-4.6",
    "*opus*": "claude-opus-4.6"
  },
  "accounts": [
    {"token": "ghp_aaa", "plan": "enterprise", "quota_limit": 1000, "premium_used": 250},
    {"token": "ghp_bbb", "plan": "free"},
    "ghp_ccc:pro+:1500:100.5",
    "ghp_ddd"
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
| `--config` | `COPILOT_ADAPTER_CONFIG` | `~/.config/copilot-api/config.json` |
| `--host` | `COPILOT_ADAPTER_HOST` | `127.0.0.1` |
| `--port` | `COPILOT_ADAPTER_PORT` | `18080` |
| `--github-token` | `COPILOT_ADAPTER_GITHUB_TOKEN` | *(none)* |
| `--cors-origin` | `COPILOT_ADAPTER_CORS_ORIGIN` | *(none)* |
| `--workers` | `COPILOT_ADAPTER_WORKERS` | number of CPUs |
| `--strategy` | `COPILOT_ADAPTER_STRATEGY` | `max-usage` |
| `--quota-limit` | `COPILOT_ADAPTER_QUOTA_LIMIT` | per plan |
| `--plan` | `COPILOT_ADAPTER_PLAN` | `pro` |
| `--log-level` | `COPILOT_ADAPTER_LOG_LEVEL` | `info` |
| `--free` | `COPILOT_ADAPTER_FREE` | *(off)* |
| `--proxy` | `COPILOT_ADAPTER_PROXY` | *(off)* |
| `--ca-dir` | `COPILOT_ADAPTER_CA_DIR` | `~/.config/copilot-api` |
| `--model-map` | `COPILOT_ADAPTER_MODEL_MAP` | `model_map.json` |

Set `NO_COLOR=1` to disable colored log output. Colors are auto-detected on Windows (requires Windows Terminal or VT-enabled console).

`GITHUB_TOKEN` is also accepted as a fallback for the GitHub token. Multiple tokens can be comma-separated in `COPILOT_ADAPTER_GITHUB_TOKEN` or `GITHUB_TOKEN`.

### Docker

A pre-built image is available on GitHub Container Registry:

```bash
docker pull ghcr.io/blhsing/copilot-adapter:latest
```

```bash
# Run
docker run -p 18080:18080 -e COPILOT_ADAPTER_GITHUB_TOKEN=ghp_xxx ghcr.io/blhsing/copilot-adapter

# Multi-account with rotation
docker run -p 18080:18080 \
  -e COPILOT_ADAPTER_GITHUB_TOKEN=ghp_aaa,ghp_bbb \
  -e COPILOT_ADAPTER_STRATEGY=max-usage \
  -e COPILOT_ADAPTER_QUOTA_LIMIT=300 \
  ghcr.io/blhsing/copilot-adapter
```

Or build locally:

```bash
docker build -t copilot-adapter .
docker run -p 18080:18080 -e COPILOT_ADAPTER_GITHUB_TOKEN=ghp_xxx copilot-adapter
```

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

When using multi-account rotation, agent-initiated requests always stay on the same account as the preceding user request to avoid billing a premium request on a different account.

### Forward proxy mode

Use `--proxy` to enable a forward HTTP/HTTPS proxy on the same port as the API server. In this mode, the server handles both normal API requests (reverse proxy) and forwarded client traffic (forward proxy) on a single port:

```bash
python copilot_adapter.py serve --proxy
```

When a client sends a `CONNECT` request to `api.githubcopilot.com`, the proxy performs a MITM (man-in-the-middle) TLS interception and rewrites `X-Initiator: user` to `X-Initiator: agent` so the request is not billed as a premium request. All other traffic is tunneled transparently.

**Client setup:**

```bash
export HTTPS_PROXY=http://127.0.0.1:18080
export NODE_EXTRA_CA_CERTS=~/.config/copilot-api/ca.pem
```

A self-signed CA certificate is generated automatically on first use and stored in `~/.config/copilot-api/` (or the directory specified by `--ca-dir`). The client must trust this CA for HTTPS interception to work. For Node.js-based clients (e.g. Claude Code), set `NODE_EXTRA_CA_CERTS` to the CA certificate path.

This mode is useful when you want to transparently reduce premium billing for any client that supports `HTTPS_PROXY`, without changing the client's API endpoint configuration.

### Model mapping

Model names in incoming requests are rewritten using configurable glob patterns before being sent to the Copilot API. This handles mismatches between model names that clients send (e.g. `claude-3-5-sonnet-latest`) and the names Copilot expects (e.g. `claude-sonnet-4.6`).

The project ships with a default `model_map.json`:

```json
{
  "*sonnet*": "claude-sonnet-4.6",
  "*opus*": "claude-opus-4.6"
}
```

Patterns use glob syntax (`*` matches anything) and are checked in order — the first match wins. If no pattern matches, the model name is passed through unchanged.

**Override via CLI** (repeatable, replaces the defaults entirely):

```bash
python copilot_adapter.py serve \
  --model-map "*sonnet*=claude-sonnet-4.6" \
  --model-map "*opus*=claude-opus-4.6" \
  --model-map "*haiku*=claude-haiku-4.5"
```

**Override via environment variable** (comma-separated):

```bash
export COPILOT_ADAPTER_MODEL_MAP="*sonnet*=claude-sonnet-4.6,*opus*=claude-opus-4.6"
```

**Override via config file:**

```json
{
  "model_map": {
    "*sonnet*": "claude-sonnet-4.6",
    "*opus*": "claude-opus-4.6",
    "*haiku*": "claude-haiku-4.5",
    "gpt-4-turbo": "gpt-4-0125-preview"
  }
}
```

**Precedence**: CLI/env > config file > shipped `model_map.json`.

Model mapping is applied to all endpoints (chat completions, responses, embeddings, Gemini).

## Client configuration

Point any OpenAI, Anthropic, or Gemini SDK client at the local server:

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

## Available models

Run `python copilot_adapter.py serve` and visit `http://127.0.0.1:18080/v1/models` to see all models available through your Copilot subscription. Models include offerings from OpenAI, Anthropic, Google, and xAI.

Note: some newer models (e.g. `gpt-5.4`) only support the `/v1/responses` endpoint, not `/v1/chat/completions`.

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

1. **Device flow OAuth** authenticates with GitHub and stores tokens in `~/.config/copilot-api/tokens.json`
2. GitHub tokens are exchanged for short-lived Copilot API tokens via `api.github.com/copilot_internal/v2/token`, automatically refreshed every ~25 minutes with concurrent-access protection (double-checked locking ensures only one refresh happens at a time)
3. For multi-account setups, the `AccountManager` selects which account to use based on the configured rotation strategy, sticking to the same account for agent-initiated follow-ups
4. Incoming requests are translated (if needed) to the format Copilot expects, model names are rewritten via the configurable model map, and requests are proxied to `api.githubcopilot.com` with responses translated back to the client's expected format
5. In forward proxy mode (`--proxy`), the server also accepts `CONNECT` tunnels on the same port — traffic to `api.githubcopilot.com` is MITM'd to rewrite billing headers, while all other traffic is tunneled transparently

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

### PATs don't work for organization-managed Copilot seats

The `api.github.com/copilot_internal/v2/token` endpoint returns 404 for `ghp_` Personal Access Tokens when the Copilot seat is managed through a GitHub organization (Business or Enterprise plan). This endpoint only works with OAuth tokens obtained via the device flow (`ghu_` prefix).

**Symptom:** `serve` prints "Authenticated as &lt;username&gt;" but then fails with:

```
Error: Failed to get Copilot token: Client error '404 Not Found' for url 'https://api.github.com/copilot_internal/v2/token'
```

**Fix:** Use the device flow instead of a PAT:

```bash
python copilot_adapter.py login
```

If you previously added the PAT to the cache, remove it first:

```bash
python copilot_adapter.py accounts --remove <username>
python copilot_adapter.py login
```

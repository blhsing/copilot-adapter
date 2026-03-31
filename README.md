# copilot-adapter

An OpenAI / Anthropic / Gemini-compatible LLM API server backed by GitHub Copilot.

Authenticates via a GitHub Personal Access Token (PAT) or GitHub's device flow, then proxies requests to GitHub Copilot's backend through a local server that speaks all three major LLM API formats.

## Key features

- **Three API formats** — Serves OpenAI, Anthropic, and Gemini endpoints simultaneously, so any SDK or tool that speaks one of these formats works out of the box
- **Streaming support** — Full SSE streaming across all three formats, including real-time format translation for Anthropic and Gemini streams
- **Multi-account pooling** — Configure multiple GitHub Copilot accounts and rotate between them to pool premium request quotas, with three rotation strategies and automatic exhaustion detection via response model mismatch
- **Flexible authentication** — Supports multiple GitHub PATs, `COPILOT_ADAPTER_GITHUB_TOKEN` / `GITHUB_TOKEN` env vars, cached tokens, and interactive device-flow OAuth, with automatic fallback
- **Smart premium request billing** — Automatically infers `X-Initiator: agent` for agentic follow-ups (tool results) to avoid extra premium request charges, with no client-side changes needed; also supports explicit `X-Initiator` header passthrough
- **Multi-worker support** — `--workers N` spawns multiple uvicorn worker processes for higher throughput (defaults to number of CPUs)
- **Docker ready** — Pre-built image on [GHCR](https://github.com/blhsing/copilot-adapter/pkgs/container/copilot-adapter), or build locally
- **Environment variable configuration** — All CLI options can be configured via environment variables for container and CI-friendly deployments
- **CORS support** — Optional `--cors-origin` flag for browser-based applications
- **Concurrent-safe token management** — Double-checked locking ensures only one token refresh happens at a time under concurrent load
- **Object-oriented architecture** — Clean `FormatAdapter` / `StreamConverter` abstractions make it straightforward to add new API formats

## Prerequisites

- Python 3.10+
- pip
- A [GitHub Copilot](https://github.com/features/copilot) subscription (Individual, Business, or Enterprise)

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

# List cached accounts
python copilot_adapter.py accounts

# Remove a specific account
python copilot_adapter.py logout --username octocat

# Remove all accounts
python copilot_adapter.py logout --all
```

**Rotation strategies** (`--strategy`):

| Strategy | Behavior | Pros | Cons |
|----------|----------|------|------|
| `max-usage` (default) | Concentrate all usage on one account until its quota is exhausted, then move to the next | Maximizes the number of accounts kept at zero usage as reserves; best server-side cache efficiency since all requests hit the same account's session; simple and predictable | One account bears all the load; if the month resets mid-use, the reserve accounts were never needed |
| `min-usage` | Always pick the account with the lowest usage | Spreads consumption evenly across all accounts; maximizes headroom on every account | All accounts accumulate usage simultaneously, so none are kept clean as a reserve |
| `round-robin` | Rotate blindly on each user-initiated request | No API calls needed to check usage; lowest overhead | No awareness of quota — won't avoid accounts nearing their limit |

Agent-initiated requests (tool-use follow-ups) always stay on the same account as the preceding user request to avoid unnecessary premium request charges.

**Quota exhaustion detection**: When a Copilot account's premium request quota is exhausted, GitHub silently downgrades the response to a free fallback model (e.g. GPT-4.1) instead of returning an error. The server detects this by comparing the model in the response against the model that was requested — if they don't match, it marks the account as exhausted and automatically retries the request with the next available account. This works for both streaming and non-streaming requests.

For proactive switching *before* hitting the limit, set `--quota-limit N` to the monthly premium request allowance of your plan (Free: 50, Pro: 300, Pro+: 1500). By default the server periodically checks each account's usage via the GitHub billing API. If this server is the only consumer of the quota, add `--local-tracking` to count user-initiated requests in-memory instead — this eliminates all billing API calls and gives instant, accurate tracking without network overhead.

### Environment variables

All CLI options can be set via environment variables:

| Flag | Environment variable | Default |
|------|---------------------|---------|
| `--host` | `COPILOT_ADAPTER_HOST` | `127.0.0.1` |
| `--port` | `COPILOT_ADAPTER_PORT` | `18080` |
| `--github-token` | `COPILOT_ADAPTER_GITHUB_TOKEN` | *(none)* |
| `--cors-origin` | `COPILOT_ADAPTER_CORS_ORIGIN` | *(none)* |
| `--workers` | `COPILOT_ADAPTER_WORKERS` | number of CPUs |
| `--strategy` | `COPILOT_ADAPTER_STRATEGY` | `max-usage` |
| `--quota-limit` | `COPILOT_ADAPTER_QUOTA_LIMIT` | *(none)* |
| `--local-tracking` | `COPILOT_ADAPTER_LOCAL_TRACKING` | off |

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

- **OpenAI format** — `agent` if the last message has `role: "tool"`
- **Anthropic format** — `agent` if the last message contains a `tool_result` content block
- **Gemini format** — `agent` if the last turn contains a `functionResponse` part
- Otherwise — `user`

This means agentic clients like Claude Code that make multiple API calls per user turn (tool-use loops, retries) will only consume one premium request for the initial prompt — follow-up calls with tool results are automatically marked as `agent`. No client-side changes needed.

Callers can also pass `X-Initiator` explicitly to override the heuristic.

When using multi-account rotation, agent-initiated requests always stay on the same account as the preceding user request to avoid billing a premium request on a different account.

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
4. Incoming requests are translated (if needed) to the format Copilot expects, proxied to `api.githubcopilot.com`, and responses are translated back to the client's expected format

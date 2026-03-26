# copilot-adapter

An OpenAI / Anthropic / Gemini-compatible LLM API server backed by GitHub Copilot.

Authenticates via a GitHub Personal Access Token (PAT) or GitHub's device flow, then proxies requests to GitHub Copilot's backend through a local server that speaks all three major LLM API formats.

## Key features

- **Three API formats** — Serves OpenAI, Anthropic, and Gemini endpoints simultaneously, so any SDK or tool that speaks one of these formats works out of the box
- **Streaming support** — Full SSE streaming across all three formats, including real-time format translation for Anthropic and Gemini streams
- **Flexible authentication** — Supports GitHub PAT, `GITHUB_TOKEN` env var, cached tokens, and interactive device-flow OAuth, with automatic fallback
- **Smart premium request billing** — Automatically infers `X-Initiator: agent` for agentic follow-ups (tool results) to avoid extra premium request charges, with no client-side changes needed; also supports explicit `X-Initiator` header passthrough
- **Multi-worker support** — `--workers N` spawns multiple uvicorn worker processes for higher throughput (defaults to 1)
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
export GITHUB_TOKEN=ghp_xxx
python copilot_adapter.py serve

# Interactive device-flow login (opens browser)
python copilot_adapter.py login
python copilot_adapter.py serve

# Options
python copilot_adapter.py serve --host 0.0.0.0 --port 8080

# Multiple worker processes for higher throughput (default: 1)
python copilot_adapter.py serve --workers 4

# Remove stored credentials
python copilot_adapter.py logout
```

Token lookup order: `--github-token` flag > `GITHUB_TOKEN` env var > cached token > interactive device flow.

### Docker

```bash
# Build
docker build -t copilot-adapter .

# Run
docker run -p 8080:8080 -e GITHUB_TOKEN=ghp_xxx copilot-adapter

# With options
docker run -p 8080:8080 -e GITHUB_TOKEN=ghp_xxx copilot-adapter \
  serve --host 0.0.0.0 --port 8080 --workers 4 --cors-origin '*'
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

## Client configuration

Point any OpenAI, Anthropic, or Gemini SDK client at the local server:

```bash
# OpenAI
export OPENAI_BASE_URL=http://127.0.0.1:8080/v1
export OPENAI_API_KEY=unused

# Anthropic
export ANTHROPIC_BASE_URL=http://127.0.0.1:8080
export ANTHROPIC_API_KEY=unused

# Gemini
export GEMINI_API_BASE=http://127.0.0.1:8080/v1beta
```

## Available models

Run `python copilot_adapter.py serve` and visit `http://127.0.0.1:8080/v1/models` to see all models available through your Copilot subscription. Models include offerings from OpenAI, Anthropic, Google, and xAI.

Note: some newer models (e.g. `gpt-5.4`) only support the `/v1/responses` endpoint, not `/v1/chat/completions`.

## Tests

The test suite runs integration tests against the live Copilot API.

```bash
pip install -r tests/requirements.txt

# Run all 57 tests (authenticates on first run via device flow)
python -m pytest

# Run a specific test module
python -m pytest tests/test_client.py
python -m pytest tests/test_adapters.py
python -m pytest tests/test_endpoints.py
```

Tests are organized into three modules:

- **`test_client.py`** — `CopilotClient` directly: models, chat completions, streaming, responses API, embeddings
- **`test_adapters.py`** — format adapters end-to-end: OpenAI passthrough, Anthropic Messages, Gemini generateContent (streaming + non-streaming, multi-turn, system messages, parameter mapping)
- **`test_endpoints.py`** — FastAPI routes via ASGI transport: all endpoints across all three API formats

Tests use the cheapest available models (`gpt-4o-mini` for chat, `gpt-5-mini` for responses, `text-embedding-3-small` for embeddings) to minimize premium request usage. Model constants are centralized in `tests/conftest.py`.

## How it works

1. **Device flow OAuth** authenticates with GitHub and stores a token in `~/.config/copilot-api/token.json`
2. The GitHub token is exchanged for a short-lived Copilot API token via `api.github.com/copilot_internal/v2/token`, automatically refreshed every ~25 minutes with concurrent-access protection (double-checked locking ensures only one refresh happens at a time)
3. Incoming requests are translated (if needed) to the format Copilot expects, proxied to `api.githubcopilot.com`, and responses are translated back to the client's expected format

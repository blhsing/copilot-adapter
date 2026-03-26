# copilot-adapter

An OpenAI / Anthropic / Gemini-compatible LLM API server backed by GitHub Copilot.

Authenticates via GitHub's device flow, then proxies requests to GitHub Copilot's backend through a local server that speaks all three major LLM API formats.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Authenticate with GitHub (opens browser)
python copilot_adapter.py login

# Start the API server
python copilot_adapter.py serve

# Options
python copilot_adapter.py serve --host 0.0.0.0 --port 8080

# Remove stored credentials
python copilot_adapter.py logout
```

On first run, `serve` will trigger the login flow automatically.

## Endpoints

### OpenAI

```
POST /v1/chat/completions
POST /v1/responses
GET  /v1/models
POST /v1/embeddings
```

### Anthropic

```
POST /v1/messages
```

### Gemini

```
POST /v1beta/models/{model}:generateContent
POST /v1beta/models/{model}:streamGenerateContent
GET  /v1beta/models
GET  /v1beta/models/{model}
```

All endpoints support streaming.

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

## How it works

1. **Device flow OAuth** authenticates with GitHub and stores a token in `~/.config/copilot-api/token.json`
2. The GitHub token is exchanged for a short-lived Copilot API token via `api.github.com/copilot_internal/v2/token`, automatically refreshed every ~25 minutes
3. Incoming requests are translated (if needed) to the format Copilot expects, proxied to `api.githubcopilot.com`, and responses are translated back to the client's expected format

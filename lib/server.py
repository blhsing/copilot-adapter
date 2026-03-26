"""OpenAI-compatible API server that proxies to GitHub Copilot."""

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from .adapters import AnthropicAdapter, FormatAdapter, GeminiAdapter, OpenAIAdapter
from .auth import CopilotTokenManager
from .client import CopilotClient

app = FastAPI(title="Copilot API", version="0.1.0")

client: CopilotClient | None = None
openai_adapter = OpenAIAdapter()
anthropic_adapter = AnthropicAdapter()

_STREAM_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


def init_app(tm: CopilotTokenManager) -> FastAPI:
    global client
    client = CopilotClient(tm)
    return app


# ---------------------------------------------------------------------------
# Shared handler
# ---------------------------------------------------------------------------

async def handle_chat_completion(adapter: FormatAdapter, body: dict):
    openai_body = adapter.convert_chat_request(body)

    if adapter.is_streaming(body) or openai_body.get("stream"):
        converter = adapter.create_stream_converter(body)

        async def event_stream():
            async for line in client.stream_chat_completions(openai_body):
                if line.startswith("error:"):
                    yield converter.format_error(line)
                    return
                result = converter.feed(line)
                if result:
                    yield result

        return StreamingResponse(
            event_stream(), media_type="text/event-stream", headers=_STREAM_HEADERS
        )

    resp = await client.chat_completions(openai_body)
    return JSONResponse(
        content=adapter.convert_chat_response(resp.json(), body),
        status_code=resp.status_code,
    )


# ---------------------------------------------------------------------------
# OpenAI endpoints
# ---------------------------------------------------------------------------

@app.get("/v1/models")
@app.get("/models")
async def list_models():
    resp = await client.list_models()
    return JSONResponse(
        content=openai_adapter.convert_models_response(resp.json()),
        status_code=resp.status_code,
    )


@app.post("/v1/chat/completions")
@app.post("/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    return await handle_chat_completion(openai_adapter, body)


@app.post("/v1/responses")
@app.post("/responses")
async def responses(request: Request):
    body = await request.json()

    if body.get("stream"):
        converter = openai_adapter.create_stream_converter(body)

        async def event_stream():
            async for line in client.stream_responses(body):
                if line.startswith("error:"):
                    yield converter.format_error(line)
                    return
                result = converter.feed(line)
                if result:
                    yield result

        return StreamingResponse(
            event_stream(), media_type="text/event-stream", headers=_STREAM_HEADERS
        )

    resp = await client.responses(body)
    return JSONResponse(content=resp.json(), status_code=resp.status_code)


@app.post("/v1/embeddings")
@app.post("/embeddings")
async def embeddings(request: Request):
    body = await request.json()
    resp = await client.embeddings(body)
    try:
        content = resp.json()
    except Exception:
        content = {"error": {"message": resp.text}}
    return JSONResponse(content=content, status_code=resp.status_code)


# ---------------------------------------------------------------------------
# Anthropic endpoint
# ---------------------------------------------------------------------------

@app.post("/v1/messages")
@app.post("/messages")
async def messages(request: Request):
    body = await request.json()
    return await handle_chat_completion(anthropic_adapter, body)


# ---------------------------------------------------------------------------
# Gemini endpoints
# ---------------------------------------------------------------------------

@app.get("/v1beta/models")
async def gemini_list_models():
    resp = await client.list_models()
    adapter = GeminiAdapter()
    return JSONResponse(
        content=adapter.convert_models_response(resp.json()),
        status_code=resp.status_code,
    )


@app.get("/v1beta/models/{model_id}")
async def gemini_get_model(model_id: str):
    resp = await client.list_models()
    if resp.status_code != 200:
        return JSONResponse(content=resp.json(), status_code=resp.status_code)
    adapter = GeminiAdapter(model_id)
    for m in resp.json().get("data", []):
        if m.get("id") == model_id:
            return JSONResponse(content=adapter.convert_single_model(m))
    return JSONResponse(
        content={"error": {"message": f"Model {model_id} not found"}},
        status_code=404,
    )


@app.post("/v1beta/models/{model_id}:generateContent")
async def gemini_generate_content(model_id: str, request: Request):
    body = await request.json()
    adapter = GeminiAdapter(model_id)
    return await handle_chat_completion(adapter, body)


@app.post("/v1beta/models/{model_id}:streamGenerateContent")
async def gemini_stream_generate_content(model_id: str, request: Request):
    body = await request.json()
    adapter = GeminiAdapter(model_id)
    openai_body = adapter.convert_chat_request(body)
    openai_body["stream"] = True
    converter = adapter.create_stream_converter(body)

    async def event_stream():
        async for line in client.stream_chat_completions(openai_body):
            if line.startswith("error:"):
                yield converter.format_error(line)
                return
            result = converter.feed(line)
            if result:
                yield result

    return StreamingResponse(
        event_stream(), media_type="text/event-stream", headers=_STREAM_HEADERS
    )


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/")
async def health():
    return {"status": "ok", "service": "copilot-adapter"}

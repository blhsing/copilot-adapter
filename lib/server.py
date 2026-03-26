"""OpenAI-compatible API server that proxies to GitHub Copilot."""

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from .anthropic_adapter import (
    anthropic_to_openai,
    openai_stream_to_anthropic_events,
    openai_to_anthropic,
)
from .auth import CopilotTokenManager
from .gemini_adapter import (
    gemini_to_openai,
    openai_model_to_gemini,
    openai_models_to_gemini,
    openai_stream_to_gemini_events,
    openai_to_gemini,
)
from .proxy import (
    get_models,
    proxy_chat_completions,
    proxy_embeddings,
    proxy_responses,
    stream_chat_completions,
    stream_responses,
)

app = FastAPI(title="Copilot API", version="0.1.0")

token_manager: CopilotTokenManager | None = None


def init_app(tm: CopilotTokenManager) -> FastAPI:
    global token_manager
    token_manager = tm
    return app


@app.get("/v1/models")
@app.get("/models")
async def list_models():
    resp = await get_models(token_manager)
    return JSONResponse(content=resp.json(), status_code=resp.status_code)


@app.post("/v1/chat/completions")
@app.post("/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()

    if body.get("stream"):
        async def event_stream():
            async for line in stream_chat_completions(token_manager, body):
                if line.startswith("error:"):
                    yield f"data: {line}\n\n"
                    return
                if line.strip():
                    yield f"{line}\n"
                elif line == "":
                    yield "\n"

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    resp = await proxy_chat_completions(token_manager, body)
    return JSONResponse(content=resp.json(), status_code=resp.status_code)


@app.post("/v1/responses")
@app.post("/responses")
async def responses(request: Request):
    body = await request.json()

    if body.get("stream"):
        async def event_stream():
            async for line in stream_responses(token_manager, body):
                if line.startswith("error:"):
                    yield f"data: {line}\n\n"
                    return
                if line.strip():
                    yield f"{line}\n"
                elif line == "":
                    yield "\n"

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    resp = await proxy_responses(token_manager, body)
    return JSONResponse(content=resp.json(), status_code=resp.status_code)


@app.post("/v1/messages")
@app.post("/messages")
async def messages(request: Request):
    body = await request.json()
    model = body.get("model", "")
    openai_body = anthropic_to_openai(body)

    if body.get("stream"):
        async def event_stream():
            converter = openai_stream_to_anthropic_events(model)
            async for line in stream_chat_completions(token_manager, openai_body):
                if line.startswith("error:"):
                    yield f"event: error\ndata: {line}\n\n"
                    return
                line = line.strip()
                if line:
                    result = converter.convert(line)
                    if result:
                        yield result

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    resp = await proxy_chat_completions(token_manager, openai_body)
    anthropic_resp = openai_to_anthropic(resp.json(), model)
    return JSONResponse(content=anthropic_resp, status_code=resp.status_code)


@app.post("/v1/embeddings")
@app.post("/embeddings")
async def embeddings(request: Request):
    body = await request.json()
    resp = await proxy_embeddings(token_manager, body)
    return JSONResponse(content=resp.json(), status_code=resp.status_code)


# --- Gemini API ---


@app.get("/v1beta/models")
async def gemini_list_models():
    resp = await get_models(token_manager)
    return JSONResponse(
        content=openai_models_to_gemini(resp.json()),
        status_code=resp.status_code,
    )


@app.get("/v1beta/models/{model_id}")
async def gemini_get_model(model_id: str):
    resp = await get_models(token_manager)
    if resp.status_code != 200:
        return JSONResponse(content=resp.json(), status_code=resp.status_code)
    for m in resp.json().get("data", []):
        if m.get("id") == model_id:
            return JSONResponse(content=openai_model_to_gemini(m))
    return JSONResponse(content={"error": {"message": f"Model {model_id} not found"}}, status_code=404)


@app.post("/v1beta/models/{model_id}:generateContent")
async def gemini_generate_content(model_id: str, request: Request):
    body = await request.json()
    openai_body = gemini_to_openai(body, model_id)
    resp = await proxy_chat_completions(token_manager, openai_body)
    return JSONResponse(
        content=openai_to_gemini(resp.json(), model_id),
        status_code=resp.status_code,
    )


@app.post("/v1beta/models/{model_id}:streamGenerateContent")
async def gemini_stream_generate_content(model_id: str, request: Request):
    body = await request.json()
    openai_body = gemini_to_openai(body, model_id)
    openai_body["stream"] = True

    async def event_stream():
        converter = openai_stream_to_gemini_events(model_id)
        async for line in stream_chat_completions(token_manager, openai_body):
            if line.startswith("error:"):
                yield f"data: {line}\n\n"
                return
            line = line.strip()
            if line:
                result = converter.convert(line)
                if result:
                    yield result

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/")
async def health():
    return {"status": "ok", "service": "copilot-api"}

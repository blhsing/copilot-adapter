"""OpenAI-compatible API server that proxies to GitHub Copilot."""

import json
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from .adapters import AnthropicAdapter, FormatAdapter, GeminiAdapter, OpenAIAdapter
from .account_manager import AccountManager

logger = logging.getLogger(__name__)

account_mgr: AccountManager | None = None
openai_adapter = OpenAIAdapter()
anthropic_adapter = AnthropicAdapter()

_STREAM_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


def _is_model_match(requested: str, responded: str) -> bool:
    """Check if the response model plausibly matches the requested model.

    Handles date-suffixed variants like ``gpt-4o-mini`` vs
    ``gpt-4o-mini-2024-07-18``.
    """
    return (
        requested == responded
        or responded.startswith(requested)
        or requested.startswith(responded)
    )


def _extract_model_from_sse_line(line: str) -> str | None:
    """Try to extract the ``model`` field from an SSE data line."""
    if not line.startswith("data: "):
        return None
    payload = line[6:].strip()
    if payload == "[DONE]":
        return None
    try:
        return json.loads(payload).get("model")
    except (json.JSONDecodeError, AttributeError):
        return None


@asynccontextmanager
async def _lifespan(application: FastAPI):
    """Initialize the AccountManager in each worker process on startup."""
    global account_mgr
    tokens_raw = os.environ.get("_COPILOT_ADAPTER_GITHUB_TOKENS", "")
    if tokens_raw and account_mgr is None:
        # Format: "token1:username1,token2:username2"
        accounts = []
        for entry in tokens_raw.split(","):
            if ":" in entry:
                token, username = entry.split(":", 1)
                accounts.append((token, username))
        strategy = os.environ.get("_COPILOT_ADAPTER_STRATEGY", "max-usage")
        quota_limit_raw = os.environ.get("_COPILOT_ADAPTER_QUOTA_LIMIT", "")
        quota_limit = int(quota_limit_raw) if quota_limit_raw else None
        local_tracking = os.environ.get("_COPILOT_ADAPTER_LOCAL_TRACKING", "") == "1"
        plan = os.environ.get("_COPILOT_ADAPTER_PLAN", "paid")
        account_mgr = AccountManager(
            accounts, strategy=strategy, quota_limit=quota_limit,
            local_tracking=local_tracking, plan=plan,
        )

        cors_raw = os.environ.get("_COPILOT_ADAPTER_CORS_ORIGINS", "")
        if cors_raw:
            from fastapi.middleware.cors import CORSMiddleware

            application.add_middleware(
                CORSMiddleware,
                allow_origins=cors_raw.split(","),
                allow_methods=["*"],
                allow_headers=["*"],
            )
    yield


app = FastAPI(title="Copilot API", version="0.1.0", lifespan=_lifespan)


def init_app(
    mgr: AccountManager, cors_origins: list[str] | None = None
) -> FastAPI:
    global account_mgr
    account_mgr = mgr

    if cors_origins:
        from fastapi.middleware.cors import CORSMiddleware

        app.add_middleware(
            CORSMiddleware,
            allow_origins=cors_origins,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    return app


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_initiator(request: Request) -> str | None:
    """Extract X-Initiator from the incoming request, or None if absent."""
    return request.headers.get("x-initiator")


async def handle_chat_completion(
    adapter: FormatAdapter, body: dict, *, request: Request | None = None, initiator: str | None = None
):
    resolved = initiator or adapter.infer_initiator(body)
    openai_body = adapter.convert_chat_request(body)
    requested_model = openai_body.get("model", "")
    client = await account_mgr.get_client(initiator=resolved)

    if adapter.is_streaming(body) or openai_body.get("stream"):
        converter = adapter.create_stream_converter(body)

        async def event_stream():
            nonlocal client, converter
            first_chunk = True
            async for line in client.stream_chat_completions(
                openai_body, initiator=resolved
            ):
                if line.startswith("error:"):
                    yield converter.format_error(line)
                    return

                # Check the first data chunk for model mismatch
                if first_chunk:
                    resp_model = _extract_model_from_sse_line(line)
                    if resp_model is not None:
                        first_chunk = False
                        if not _is_model_match(requested_model, resp_model):
                            logger.warning(
                                "Model mismatch: requested %s, got %s — "
                                "quota likely exhausted, switching account",
                                requested_model, resp_model,
                            )
                            fallback = await account_mgr.get_fallback_client(client)
                            if fallback is not None:
                                client = fallback
                                converter = adapter.create_stream_converter(body)
                                async for retry_line in client.stream_chat_completions(
                                    openai_body, initiator=resolved
                                ):
                                    if retry_line.startswith("error:"):
                                        yield converter.format_error(retry_line)
                                        return
                                    result = converter.feed(retry_line)
                                    if result:
                                        yield result
                                await account_mgr.record_usage(client, requested_model)
                                return
                        else:
                            await account_mgr.record_usage(client, requested_model)

                result = converter.feed(line)
                if result:
                    yield result

        return StreamingResponse(
            event_stream(), media_type="text/event-stream", headers=_STREAM_HEADERS
        )

    # Non-streaming
    resp = await client.chat_completions(openai_body, initiator=resolved)
    resp_data = resp.json()
    resp_model = resp_data.get("model", "")
    if resp_model and not _is_model_match(requested_model, resp_model):
        logger.warning(
            "Model mismatch: requested %s, got %s — "
            "quota likely exhausted, switching account",
            requested_model, resp_model,
        )
        fallback = await account_mgr.get_fallback_client(client)
        if fallback is not None:
            client = fallback
            resp = await client.chat_completions(openai_body, initiator=resolved)
            resp_data = resp.json()

    await account_mgr.record_usage(client, requested_model)
    return JSONResponse(
        content=adapter.convert_chat_response(resp_data, body),
        status_code=resp.status_code,
    )


# ---------------------------------------------------------------------------
# OpenAI endpoints
# ---------------------------------------------------------------------------

@app.get("/v1/models")
@app.get("/models")
async def list_models():
    client = await account_mgr.get_client()
    resp = await client.list_models()
    return JSONResponse(
        content=openai_adapter.convert_models_response(resp.json()),
        status_code=resp.status_code,
    )


@app.post("/v1/chat/completions")
@app.post("/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    return await handle_chat_completion(
        openai_adapter, body, request=request, initiator=_get_initiator(request)
    )


@app.post("/v1/responses")
@app.post("/responses")
async def responses(request: Request):
    body = await request.json()
    initiator = _get_initiator(request) or "user"
    client = await account_mgr.get_client(initiator=initiator)
    requested_model = body.get("model", "")

    if body.get("stream"):
        converter = openai_adapter.create_stream_converter(body)

        async def event_stream():
            nonlocal client, converter
            first_chunk = True
            async for line in client.stream_responses(body, initiator=initiator):
                if line.startswith("error:"):
                    yield converter.format_error(line)
                    return
                if first_chunk:
                    resp_model = _extract_model_from_sse_line(line)
                    if resp_model is not None:
                        first_chunk = False
                        if not _is_model_match(requested_model, resp_model):
                            logger.warning(
                                "Model mismatch: requested %s, got %s — "
                                "quota likely exhausted, switching account",
                                requested_model, resp_model,
                            )
                            fallback = await account_mgr.get_fallback_client(client)
                            if fallback is not None:
                                client = fallback
                                converter = openai_adapter.create_stream_converter(body)
                                async for retry_line in client.stream_responses(
                                    body, initiator=initiator
                                ):
                                    if retry_line.startswith("error:"):
                                        yield converter.format_error(retry_line)
                                        return
                                    result = converter.feed(retry_line)
                                    if result:
                                        yield result
                                await account_mgr.record_usage(client, requested_model)
                                return
                        else:
                            await account_mgr.record_usage(client, requested_model)
                result = converter.feed(line)
                if result:
                    yield result

        return StreamingResponse(
            event_stream(), media_type="text/event-stream", headers=_STREAM_HEADERS
        )

    resp = await client.responses(body, initiator=initiator)
    resp_data = resp.json()
    resp_model = resp_data.get("model", "")
    if resp_model and not _is_model_match(requested_model, resp_model):
        logger.warning(
            "Model mismatch: requested %s, got %s — "
            "quota likely exhausted, switching account",
            requested_model, resp_model,
        )
        fallback = await account_mgr.get_fallback_client(client)
        if fallback is not None:
            client = fallback
            resp = await client.responses(body, initiator=initiator)
            resp_data = resp.json()
    await account_mgr.record_usage(client, requested_model)
    return JSONResponse(content=resp_data, status_code=resp.status_code)


@app.post("/v1/embeddings")
@app.post("/embeddings")
async def embeddings(request: Request):
    body = await request.json()
    initiator = _get_initiator(request) or "user"
    client = await account_mgr.get_client(initiator=initiator)
    resp = await client.embeddings(body, initiator=initiator)
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
    return await handle_chat_completion(
        anthropic_adapter, body, request=request, initiator=_get_initiator(request)
    )


# ---------------------------------------------------------------------------
# Gemini endpoints
# ---------------------------------------------------------------------------

@app.get("/v1beta/models")
async def gemini_list_models():
    client = await account_mgr.get_client()
    resp = await client.list_models()
    adapter = GeminiAdapter()
    return JSONResponse(
        content=adapter.convert_models_response(resp.json()),
        status_code=resp.status_code,
    )


@app.get("/v1beta/models/{model_id}")
async def gemini_get_model(model_id: str):
    client = await account_mgr.get_client()
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
    return await handle_chat_completion(
        adapter, body, request=request, initiator=_get_initiator(request)
    )


@app.post("/v1beta/models/{model_id}:streamGenerateContent")
async def gemini_stream_generate_content(model_id: str, request: Request):
    body = await request.json()
    adapter = GeminiAdapter(model_id)
    resolved = _get_initiator(request) or adapter.infer_initiator(body)
    openai_body = adapter.convert_chat_request(body)
    openai_body["stream"] = True
    requested_model = openai_body.get("model", "")
    client = await account_mgr.get_client(initiator=resolved)
    converter = adapter.create_stream_converter(body)

    async def event_stream():
        nonlocal client, converter
        first_chunk = True
        async for line in client.stream_chat_completions(
            openai_body, initiator=resolved
        ):
            if await request.is_disconnected():
                return
            if line.startswith("error:"):
                yield converter.format_error(line)
                return
            if first_chunk:
                resp_model = _extract_model_from_sse_line(line)
                if resp_model is not None:
                    first_chunk = False
                    if not _is_model_match(requested_model, resp_model):
                        logger.warning(
                            "Model mismatch: requested %s, got %s — "
                            "quota likely exhausted, switching account",
                            requested_model, resp_model,
                        )
                        fallback = await account_mgr.get_fallback_client(client)
                        if fallback is not None:
                            client = fallback
                            converter = adapter.create_stream_converter(body)
                            async for retry_line in client.stream_chat_completions(
                                openai_body, initiator=resolved
                            ):
                                if await request.is_disconnected():
                                    return
                                if retry_line.startswith("error:"):
                                    yield converter.format_error(retry_line)
                                    return
                                result = converter.feed(retry_line)
                                if result:
                                    yield result
                            await account_mgr.record_usage(client, requested_model)
                            return
                    else:
                        await account_mgr.record_usage(client, requested_model)
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

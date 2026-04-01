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
        # Format: "token1:username1:plan1:quota1:usage1,..."
        strategy = os.environ.get("_COPILOT_ADAPTER_STRATEGY", "max-usage")
        quota_limit_raw = os.environ.get("_COPILOT_ADAPTER_QUOTA_LIMIT", "")
        quota_limit = int(quota_limit_raw) if quota_limit_raw else None
        plan = os.environ.get("_COPILOT_ADAPTER_PLAN", "pro")
        accounts: list[dict] = []
        for entry in tokens_raw.split(","):
            parts = entry.split(":")
            if len(parts) >= 2:
                acct: dict = {"token": parts[0], "username": parts[1]}
                if len(parts) >= 3 and parts[2]:
                    acct["plan"] = parts[2]
                if len(parts) >= 4 and parts[3]:
                    acct["quota_limit"] = int(parts[3])
                if len(parts) >= 5 and parts[4]:
                    acct["premium_used"] = float(parts[4])
                accounts.append(acct)
        account_mgr = AccountManager(
            accounts, strategy=strategy, quota_limit=quota_limit, plan=plan,
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


def _is_rate_limit_error(line: str) -> bool:
    """Check if an SSE error line indicates a 429 rate limit."""
    return line.startswith("error: 429")


async def handle_chat_completion(
    adapter: FormatAdapter, body: dict, *, request: Request | None = None, initiator: str | None = None
):
    resolved = initiator or adapter.infer_initiator(body)
    openai_body = adapter.convert_chat_request(body)
    requested_model = openai_body.get("model", "")

    billed_status = "yes" if resolved == "user" else "no"
    logger.info("Chat completion requested by %s (billed: %s, model: %s)", resolved, billed_status, requested_model)

    client = await account_mgr.get_client(initiator=resolved)

    if adapter.is_streaming(body) or openai_body.get("stream"):
        converter = adapter.create_stream_converter(body)

        async def event_stream():
            nonlocal client, converter
            while True:
                first_chunk = True
                needs_retry = False
                async for line in client.stream_chat_completions(
                    openai_body, initiator=resolved
                ):
                    if request and await request.is_disconnected():
                        return
                    if line.startswith("error:"):
                        if _is_rate_limit_error(line):
                            logger.warning(
                                "Rate limited on account, switching account"
                            )
                            fallback = await account_mgr.get_fallback_client(client)
                            if fallback is not None:
                                client = fallback
                                converter = adapter.create_stream_converter(body)
                                needs_retry = True
                                break
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
                                    needs_retry = True
                                    break
                            else:
                                await account_mgr.record_usage(client, requested_model)

                    result = converter.feed(line)
                    if result:
                        yield result

                if not needs_retry:
                    return

        return StreamingResponse(
            event_stream(), media_type="text/event-stream", headers=_STREAM_HEADERS
        )

    # Non-streaming — retry loop for 429 and model mismatch
    while True:
        resp = await client.chat_completions(openai_body, initiator=resolved)
        if resp.status_code == 429:
            logger.warning("Rate limited on account, switching account")
            fallback = await account_mgr.get_fallback_client(client)
            if fallback is not None:
                client = fallback
                continue
            # All accounts exhausted, return the 429

        if resp.status_code != 200:
            logger.error("API error %s: %s", resp.status_code, resp.text)
            try:
                content = resp.json()
            except Exception:
                content = {"error": {"message": resp.text}}
            return JSONResponse(content=content, status_code=resp.status_code)

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
                continue
        break

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
    while True:
        resp = await client.list_models()
        if resp.status_code == 429:
            logger.warning("Rate limited on account, switching account")
            fallback = await account_mgr.get_fallback_client(client)
            if fallback is not None:
                client = fallback
                continue
        break
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
    requested_model = body.get("model", "")

    billed_status = "yes" if initiator == "user" else "no"
    logger.info("Responses requested by %s (billed: %s, model: %s)", initiator, billed_status, requested_model)

    client = await account_mgr.get_client(initiator=initiator)

    if body.get("stream"):
        converter = openai_adapter.create_stream_converter(body)

        async def event_stream():
            nonlocal client, converter
            while True:
                first_chunk = True
                needs_retry = False
                async for line in client.stream_responses(body, initiator=initiator):
                    if request and await request.is_disconnected():
                        return
                    if line.startswith("error:"):
                        if _is_rate_limit_error(line):
                            logger.warning("Rate limited on account, switching account")
                            fallback = await account_mgr.get_fallback_client(client)
                            if fallback is not None:
                                client = fallback
                                converter = openai_adapter.create_stream_converter(body)
                                needs_retry = True
                                break
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
                                    needs_retry = True
                                    break
                            else:
                                await account_mgr.record_usage(client, requested_model)
                    result = converter.feed(line)
                    if result:
                        yield result
                if not needs_retry:
                    return

        return StreamingResponse(
            event_stream(), media_type="text/event-stream", headers=_STREAM_HEADERS
        )

    # Non-streaming — retry loop for 429 and model mismatch
    while True:
        resp = await client.responses(body, initiator=initiator)
        if resp.status_code == 429:
            logger.warning("Rate limited on account, switching account")
            fallback = await account_mgr.get_fallback_client(client)
            if fallback is not None:
                client = fallback
                continue

        if resp.status_code != 200:
            logger.error("API error %s: %s", resp.status_code, resp.text)
            try:
                content = resp.json()
            except Exception:
                content = {"error": {"message": resp.text}}
            return JSONResponse(content=content, status_code=resp.status_code)

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
                continue
        break
    await account_mgr.record_usage(client, requested_model)
    return JSONResponse(content=resp_data, status_code=resp.status_code)


@app.post("/v1/embeddings")
@app.post("/embeddings")
async def embeddings(request: Request):
    body = await request.json()
    initiator = _get_initiator(request) or "user"
    client = await account_mgr.get_client(initiator=initiator)
    while True:
        resp = await client.embeddings(body, initiator=initiator)
        if resp.status_code == 429:
            logger.warning("Rate limited on account, switching account")
            fallback = await account_mgr.get_fallback_client(client)
            if fallback is not None:
                client = fallback
                continue
        break
    try:
        content = resp.json()
    except Exception:
        content = {"error": {"message": resp.text}}
    if resp.status_code != 200:
        logger.error("API error %s: %s", resp.status_code, resp.text)
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
    while True:
        resp = await client.list_models()
        if resp.status_code == 429:
            logger.warning("Rate limited on account, switching account")
            fallback = await account_mgr.get_fallback_client(client)
            if fallback is not None:
                client = fallback
                continue
        break
    adapter = GeminiAdapter()
    return JSONResponse(
        content=adapter.convert_models_response(resp.json()),
        status_code=resp.status_code,
    )


@app.get("/v1beta/models/{model_id}")
async def gemini_get_model(model_id: str):
    client = await account_mgr.get_client()
    while True:
        resp = await client.list_models()
        if resp.status_code == 429:
            logger.warning("Rate limited on account, switching account")
            fallback = await account_mgr.get_fallback_client(client)
            if fallback is not None:
                client = fallback
                continue
        break
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
        while True:
            first_chunk = True
            needs_retry = False
            async for line in client.stream_chat_completions(
                openai_body, initiator=resolved
            ):
                if await request.is_disconnected():
                    return
                if line.startswith("error:"):
                    if _is_rate_limit_error(line):
                        logger.warning("Rate limited on account, switching account")
                        fallback = await account_mgr.get_fallback_client(client)
                        if fallback is not None:
                            client = fallback
                            converter = adapter.create_stream_converter(body)
                            needs_retry = True
                            break
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
                                needs_retry = True
                                break
                        else:
                            await account_mgr.record_usage(client, requested_model)
                result = converter.feed(line)
                if result:
                    yield result
            if not needs_retry:
                return

    return StreamingResponse(
        event_stream(), media_type="text/event-stream", headers=_STREAM_HEADERS
    )


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/")
async def health():
    return {"status": "ok", "service": "copilot-adapter"}

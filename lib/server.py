"""OpenAI-compatible API server that proxies to GitHub Copilot."""

import asyncio
import json
import logging
import os
import re
import uuid
from contextlib import asynccontextmanager
from fnmatch import fnmatch
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from .adapters import AnthropicAdapter, FormatAdapter, GeminiAdapter, OpenAIAdapter
from .adapters.anthropic import (
    _anthropic_to_responses,
    _AnthropicResponsesStreamConverter,
    _responses_to_anthropic,
)
from .account_manager import AccountManager

logger = logging.getLogger(__name__)

account_mgr: AccountManager | None = None
_force_free: bool = False
_free_within_minutes: float | None = None
_model_map: list[tuple[str, str]] = []
_api_tokens: set[str] | None = None
_web_search_max_iterations: int = 3
openai_adapter = OpenAIAdapter()
anthropic_adapter = AnthropicAdapter()

_DEFAULT_MODEL_MAP_FILE = Path(__file__).resolve().parent.parent / "model_map.json"


def load_default_model_map() -> list[tuple[str, str]]:
    """Load the default model map from the shipped model_map.json file."""
    if _DEFAULT_MODEL_MAP_FILE.exists():
        data = json.loads(_DEFAULT_MODEL_MAP_FILE.read_text())
        return list(data.items())
    return []

_STREAM_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}

_MESSAGE_INDEX_RE = re.compile(r"messages\[(\d+)\]")


def _normalize_model_name(name: str) -> str:
    """Normalize a model name for comparison (lowercase, collapse separators)."""
    return name.lower().replace(" ", "-").replace("_", "-").replace(".", "-")


def _is_model_match(requested: str, responded: str) -> bool:
    """Check if the response model plausibly matches the requested model.

    Handles date-suffixed variants like ``gpt-4o-mini`` vs
    ``gpt-4o-mini-2024-07-18``, and display names like
    ``Claude Haiku 4.5`` vs ``claude-haiku-4.5``.
    """
    req = _normalize_model_name(requested)
    resp = _normalize_model_name(responded)
    return (
        req == resp
        or resp.startswith(req)
        or req.startswith(resp)
    )


def _infer_provider_from_model(model: str) -> str:
    """Infer the provider family from a mapped model name."""
    name = model.lower()
    if name.startswith("claude"):
        return "anthropic"
    if name.startswith("gemini"):
        return "gemini"
    if (
        name.startswith("gpt")
        or name.startswith("o1")
        or name.startswith("o3")
        or name.startswith("o4")
        or name.startswith("o")
    ):
        return "openai"
    return "openai"


def _should_use_native_anthropic_api(source_provider: str, target_model: str) -> bool:
    """Return True when Anthropic requests can stay in native Messages format."""
    return (
        source_provider == "anthropic"
        and _infer_provider_from_model(target_model) == "anthropic"
    )


def _should_use_responses_api(target_model: str) -> bool:
    """Return True when the target model requires the Responses API.

    Some models (e.g. gpt-5.4) reject reasoning_effort + function tools
    on ``/v1/chat/completions`` and require ``/v1/responses`` instead.
    """
    return target_model in ("gpt-5.4",)


def _anthropic_has_interceptable_web_search(body: dict) -> bool:
    """Return True when an Anthropic request uses built-in web_search."""
    if _web_search_max_iterations <= 0:
        return False
    tools = body.get("tools")
    if not isinstance(tools, list):
        return False
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        tool_type = str(tool.get("type", ""))
        tool_name = str(tool.get("name", ""))
        if tool_type.startswith("web_search") or tool_name == "web_search":
            return True
    return False


def _target_prefers_max_completion_tokens(model: str) -> bool:
    """Return True if *model* prefers ``max_completion_tokens`` instead of ``max_tokens``."""
    return _infer_provider_from_model(model) == "openai"


def _normalize_thinking_to_effort(thinking: dict) -> str | None:
    """Convert source-provider thinking config to a canonical effort string."""
    if not isinstance(thinking, dict):
        return None
    if thinking.get("type") == "disabled":
        return None
    budget = thinking.get("budget_tokens")
    if not isinstance(budget, int) or budget <= 0:
        return None
    if budget >= 32000:
        return "xhigh"
    if budget >= 10000:
        return "high"
    if budget >= 4000:
        return "medium"
    return "low"


def _normalize_output_effort(effort: str | None) -> str | None:
    """Convert Anthropic/Claude effort labels to canonical reasoning effort."""
    if not isinstance(effort, str) or not effort:
        return None
    normalized = effort.lower()
    return {
        "low": "low",
        "medium": "medium",
        "high": "high",
        "max": "xhigh",
        "xhigh": "xhigh",
    }.get(normalized)


def _normalize_reasoning_params(
    openai_body: dict, source_provider: str, target_model: str, *, endpoint: str = "chat_completions"
) -> None:
    """Normalize cross-provider reasoning params for the mapped target model."""
    target_provider = _infer_provider_from_model(target_model)

    if source_provider == "anthropic":
        thinking = openai_body.pop("_copilot_adapter_thinking", None)
        output_effort = openai_body.pop("_copilot_adapter_output_effort", None)
        effort = _normalize_output_effort(output_effort) or _normalize_thinking_to_effort(thinking)
        if effort and target_provider == "openai":
            openai_body["reasoning_effort"] = effort
        if output_effort:
            logger.debug(
                "Request normalization: source=%s target=%s endpoint=%s provider=%s output_effort=%s reasoning_effort=%s",
                source_provider,
                target_model,
                endpoint,
                target_provider,
                output_effort,
                openai_body.get("reasoning_effort"),
            )
        else:
            logger.debug(
                "Request normalization: source=%s target=%s endpoint=%s provider=%s thinking=%s reasoning_effort=%s",
                source_provider,
                target_model,
                endpoint,
                target_provider,
                thinking,
                openai_body.get("reasoning_effort"),
            )

        if (thinking or output_effort) and not effort:
            logger.debug(
                "Reasoning normalization skipped: inputs did not map to effort (thinking=%s, output_effort=%s)",
                thinking,
                output_effort,
            )
        elif effort and target_provider != "openai":
            logger.debug(
                "Reasoning normalization skipped: derived_effort=%s but target provider is %s",
                effort,
                target_provider,
            )


def _normalize_token_limit_params(openai_body: dict, target_model: str) -> None:
    """Normalize token-limit params for the mapped target model."""
    if "max_tokens" in openai_body and _target_prefers_max_completion_tokens(target_model):
        openai_body["max_completion_tokens"] = openai_body.pop("max_tokens")


def _normalize_request_params(
    openai_body: dict,
    source_provider: str,
    target_model: str,
    *,
    endpoint: str = "chat_completions",
) -> dict:
    """Normalize provider/model-specific params after model mapping."""
    _normalize_token_limit_params(openai_body, target_model)
    _normalize_reasoning_params(openai_body, source_provider, target_model, endpoint=endpoint)
    if "reasoning_effort" in openai_body:
        # Some models (e.g. gpt-5.4) reject reasoning_effort + function
        # tools in /v1/chat/completions (requires /v1/responses instead).
        # Strip only for that known-incompatible chat/completions case.
        has_tools = bool(openai_body.get("tools"))
        incompatible_chat_model = target_model == "gpt-5.4"
        if endpoint == "chat_completions" and has_tools and incompatible_chat_model:
            stripped = openai_body.pop("reasoning_effort")
            logger.warning(
                "Stripped reasoning_effort=%s for target=%s on %s — "
                "reasoning_effort with tools requires /v1/responses",
                stripped,
                target_model,
                endpoint,
            )
    return openai_body


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


def _apply_model_map(model: str) -> str:
    """Apply the configured model mapping to a model name."""
    for pattern, target in _model_map:
        if fnmatch(model, pattern):
            if model != target:
                logger.debug("Model map: %s -> %s (pattern: %s)", model, target, pattern)
            return target
    return model


# ---------------------------------------------------------------------------
# Web search via DuckDuckGo (server-side execution for web_search tool calls)
# ---------------------------------------------------------------------------

def _extract_tool_calls_from_stream(lines: list[str]) -> list[dict]:
    """Reassemble streamed tool call deltas into complete tool call objects."""
    tool_calls: dict[int, dict] = {}
    for raw_line in lines:
        line = raw_line.strip()
        if not line.startswith("data: "):
            continue
        payload = line[6:].strip()
        if payload == "[DONE]":
            continue
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            continue
        choice = (data.get("choices") or [{}])[0]
        delta = choice.get("delta", {})
        for tc in delta.get("tool_calls", []):
            idx = tc.get("index", 0)
            func = tc.get("function", {})
            if idx not in tool_calls:
                tool_calls[idx] = {
                    "id": tc.get("id", ""),
                    "type": "function",
                    "function": {"name": "", "arguments": ""},
                }
            entry = tool_calls[idx]
            if tc.get("id"):
                entry["id"] = tc["id"]
            if func.get("name"):
                current_name = entry["function"]["name"]
                incoming_name = func["name"]
                if not current_name:
                    entry["function"]["name"] = incoming_name
                elif incoming_name.startswith(current_name):
                    entry["function"]["name"] = incoming_name
                elif not current_name.endswith(incoming_name):
                    entry["function"]["name"] += incoming_name
            entry["function"]["arguments"] += func.get("arguments", "")
    incomplete = [i for i, tc in tool_calls.items() if not tc["function"]["name"]]
    if incomplete:
        logger.warning(
            "Discarding buffered tool calls with missing names at indexes: %s",
            ", ".join(str(i) for i in incomplete),
        )
        return []
    result = [tool_calls[i] for i in sorted(tool_calls)]
    if result:
        logger.info(
            "Extracted %d tool call(s) from stream: %s",
            len(result),
            ", ".join(tc["function"]["name"] for tc in result),
        )
    return result


def _extract_text_from_stream(lines: list[str]) -> str:
    """Extract accumulated text content from buffered SSE lines."""
    parts = []
    for raw_line in lines:
        stripped = raw_line.strip()
        if not stripped.startswith("data: "):
            continue
        payload = stripped[6:].strip()
        if payload == "[DONE]":
            continue
        try:
            chunk = json.loads(payload)
            choice = (chunk.get("choices") or [{}])[0]
            delta = choice.get("delta", {})
            if delta.get("content"):
                parts.append(delta["content"])
        except (json.JSONDecodeError, IndexError):
            pass
    return "".join(parts)


def _extract_tool_calls_from_responses_stream(lines: list[str]) -> list[dict]:
    """Extract completed function_call items from buffered Responses API SSE lines."""
    tool_calls: list[dict] = []
    for raw_line in lines:
        stripped = raw_line.strip()
        if not stripped.startswith("data: "):
            continue
        payload = stripped[6:].strip()
        if payload == "[DONE]":
            continue
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if data.get("type") != "response.output_item.done":
            continue
        item = data.get("item", {})
        if item.get("type") != "function_call":
            continue
        tool_calls.append({
            "call_id": item.get("call_id", ""),
            "name": item.get("name", ""),
            "arguments": item.get("arguments", ""),
        })
    if tool_calls:
        logger.info(
            "Extracted %d function_call(s) from responses stream: %s",
            len(tool_calls),
            ", ".join(tc["name"] for tc in tool_calls),
        )
    return tool_calls


def _extract_text_from_responses_stream(lines: list[str]) -> str:
    """Extract accumulated text content from buffered Responses API SSE lines."""
    parts: list[str] = []
    for raw_line in lines:
        stripped = raw_line.strip()
        if not stripped.startswith("data: "):
            continue
        payload = stripped[6:].strip()
        if payload == "[DONE]":
            continue
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if data.get("type") != "response.output_text.delta":
            continue
        delta = data.get("delta", "")
        if delta:
            parts.append(delta)
    return "".join(parts)


def _extract_model_from_responses_sse(line: str) -> str | None:
    """Try to extract the model from a Responses API SSE line."""
    stripped = line.strip()
    if not stripped.startswith("data: "):
        return None
    payload = stripped[6:].strip()
    try:
        data = json.loads(payload)
    except (json.JSONDecodeError, AttributeError):
        return None
    if data.get("type") != "response.created":
        return None
    return data.get("response", {}).get("model")


def _do_web_search_raw(query: str, max_results: int = 5) -> list[dict]:
    """Run a DuckDuckGo search synchronously and return raw result dicts."""
    try:
        from ddgs import DDGS
    except ImportError:
        logger.warning("ddgs package not installed — web_search unavailable")
        return []

    proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY") or None
    try:
        return DDGS(proxy=proxy).text(query, max_results=max_results) or []
    except Exception as e:
        logger.error("DuckDuckGo search failed: %s", e)
        return []


def _format_search_results(results: list[dict]) -> str:
    """Format raw DDG result dicts as markdown text for tool result messages."""
    if not results:
        return "No search results found."
    lines = []
    for r in results:
        lines.append(f"[{r.get('title', '')}]({r.get('href', '')})")
        lines.append(r.get("body", ""))
        lines.append("")
    return "\n".join(lines).strip()


def _do_web_search(query: str, max_results: int = 5) -> str:
    """Run a DuckDuckGo search synchronously and return formatted results."""
    results = _do_web_search_raw(query, max_results)
    text = _format_search_results(results)
    logger.info("Web search returned %d results (%d chars) for query=%r",
                len(results), len(text), query)
    return text


async def _execute_web_search_calls(
    tool_calls: list[dict],
) -> tuple[list[dict], list[tuple[str, list[dict]]]]:
    """Execute web_search tool calls via DuckDuckGo.

    Returns:
        (tool_result_messages, [(query, raw_results), ...])
        where tool_result_messages are OpenAI-format tool result messages
        and raw_results are the raw DDG result dicts per query.
    """
    results = []
    raw_data: list[tuple[str, list[dict]]] = []
    for tc in tool_calls:
        if tc["function"]["name"] != "web_search":
            continue
        try:
            args = json.loads(tc["function"]["arguments"])
        except json.JSONDecodeError:
            args = {}
        query = args.get("query", "")
        logger.debug("Executing web_search: query=%r", query)
        # ddgs is synchronous — run in a thread to avoid blocking the event loop
        raw = await asyncio.to_thread(_do_web_search_raw, query)
        text = _format_search_results(raw)
        logger.info("Web search returned %d results (%d chars) for query=%r",
                    len(raw), len(text), query)
        raw_data.append((query, raw))
        results.append({
            "role": "tool",
            "tool_call_id": tc["id"],
            "content": text,
        })
    return results, raw_data


def _build_web_search_sse_events(
    query: str, raw_results: list[dict], start_index: int,
) -> str:
    """Build Anthropic SSE events for server_tool_use + web_search_tool_result.

    Returns a string of SSE events that can be yielded directly into the
    Anthropic streaming response.
    """
    tool_id = f"srvtoolu_{uuid.uuid4().hex[:24]}"

    def _ev(event_type: str, data: dict) -> str:
        return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"

    events = ""

    # Block N: server_tool_use
    events += _ev("content_block_start", {
        "type": "content_block_start",
        "index": start_index,
        "content_block": {
            "type": "server_tool_use",
            "id": tool_id,
            "name": "web_search",
            "input": {},
        },
    })
    events += _ev("content_block_delta", {
        "type": "content_block_delta",
        "index": start_index,
        "delta": {
            "type": "input_json_delta",
            "partial_json": json.dumps({"query": query}),
        },
    })
    events += _ev("content_block_stop", {
        "type": "content_block_stop",
        "index": start_index,
    })

    # Block N+1: web_search_tool_result
    search_results = []
    for r in raw_results:
        search_results.append({
            "type": "web_search_result",
            "url": r.get("href", ""),
            "title": r.get("title", ""),
            "encrypted_content": r.get("body", ""),
            "page_age": "",
        })

    events += _ev("content_block_start", {
        "type": "content_block_start",
        "index": start_index + 1,
        "content_block": {
            "type": "web_search_tool_result",
            "tool_use_id": tool_id,
            "content": [],
        },
    })
    if search_results:
        events += _ev("content_block_delta", {
            "type": "content_block_delta",
            "index": start_index + 1,
            "delta": {
                "type": "web_search_result_delta",
                "search_results": search_results,
            },
        })
    events += _ev("content_block_stop", {
        "type": "content_block_stop",
        "index": start_index + 1,
    })

    return events


def _build_web_search_content_blocks(
    query: str, raw_results: list[dict],
) -> tuple[list[dict], str]:
    """Build Anthropic content blocks for server_tool_use + web_search_tool_result.

    Returns (content_blocks, tool_id).
    """
    tool_id = f"srvtoolu_{uuid.uuid4().hex[:24]}"
    search_results = []
    for r in raw_results:
        search_results.append({
            "type": "web_search_result",
            "url": r.get("href", ""),
            "title": r.get("title", ""),
            "encrypted_content": r.get("body", ""),
            "page_age": "",
        })
    return [
        {
            "type": "server_tool_use",
            "id": tool_id,
            "name": "web_search",
            "input": {"query": query},
        },
        {
            "type": "web_search_tool_result",
            "tool_use_id": tool_id,
            "content": search_results,
        },
    ], tool_id


@asynccontextmanager
async def _lifespan(application: FastAPI):
    """Initialize the AccountManager in each worker process on startup."""
    global account_mgr
    global _force_free
    global _free_within_minutes
    global _model_map
    global _api_tokens
    global _web_search_max_iterations
    tokens_raw = os.environ.get("_COPILOT_ADAPTER_GITHUB_TOKENS", "")
    if tokens_raw and account_mgr is None:
        _force_free = os.environ.get("_COPILOT_ADAPTER_FREE", "") == "1"
        fwm_raw = os.environ.get("_COPILOT_ADAPTER_FREE_WITHIN_MINUTES", "")
        _free_within_minutes = float(fwm_raw) if fwm_raw else None
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

        model_map_raw = os.environ.get("_COPILOT_ADAPTER_MODEL_MAP", "")
        if model_map_raw:
            _model_map = []
            for entry in model_map_raw.split(","):
                if "=" in entry:
                    pat, _, tgt = entry.partition("=")
                    _model_map.append((pat, tgt))
        else:
            _model_map = load_default_model_map()

        cors_raw = os.environ.get("_COPILOT_ADAPTER_CORS_ORIGINS", "")
        if cors_raw:
            from fastapi.middleware.cors import CORSMiddleware

            application.add_middleware(
                CORSMiddleware,
                allow_origins=cors_raw.split(","),
                allow_methods=["*"],
                allow_headers=["*"],
            )

        api_tokens_raw = os.environ.get("_COPILOT_ADAPTER_API_TOKENS", "")
        if api_tokens_raw:
            _api_tokens = set(api_tokens_raw.split(","))

        ws_max_raw = os.environ.get("_COPILOT_ADAPTER_WEB_SEARCH_MAX_ITERATIONS", "")
        if ws_max_raw:
            _web_search_max_iterations = int(ws_max_raw)
    yield


app = FastAPI(title="Copilot API", version="0.1.0", lifespan=_lifespan)


def init_app(
    mgr: AccountManager, cors_origins: list[str] | None = None,
    force_free: bool = False,
    free_within_minutes: float | None = None,
    model_map: list[tuple[str, str]] | None = None,
    api_tokens: list[str] | None = None,
    web_search_max_iterations: int = 1,
) -> FastAPI:
    global account_mgr, _force_free, _free_within_minutes, _model_map, _api_tokens, _web_search_max_iterations
    account_mgr = mgr
    _force_free = force_free
    _free_within_minutes = free_within_minutes
    _model_map = model_map if model_map is not None else load_default_model_map()
    _api_tokens = set(api_tokens) if api_tokens else None
    _web_search_max_iterations = web_search_max_iterations

    if cors_origins:
        from fastapi.middleware.cors import CORSMiddleware

        app.add_middleware(
            CORSMiddleware,
            allow_origins=cors_origins,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    return app


@app.middleware("http")
async def _check_api_token(request: Request, call_next):
    """Reject requests without a valid Bearer token when API tokens are configured."""
    if _api_tokens is None:
        return await call_next(request)
    # Allow health check without auth
    if request.url.path == "/":
        return await call_next(request)
    auth = request.headers.get("authorization", "")
    if auth.startswith("Bearer "):
        token = auth[7:]
        if token in _api_tokens:
            return await call_next(request)
    return JSONResponse(
        status_code=401,
        content={"error": {"message": "Invalid or missing API token"}},
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_initiator(request: Request) -> str | None:
    """Extract X-Initiator from the incoming request, or None if absent."""
    if _force_free:
        return "agent"
    return request.headers.get("x-initiator")


def _is_rate_limit_error(line: str) -> bool:
    """Check if an SSE error line indicates a 429 rate limit."""
    return line.startswith("error: 429")


def _is_transient_upstream_error(line: str) -> bool:
    """Check if an SSE error line indicates a transient upstream error (502/504)."""
    return line.startswith("error: 502") or line.startswith("error: 504")


def _passthrough_sse_line(line: str) -> str:
    """Format an upstream SSE line for passthrough streaming responses."""
    if line.strip():
        return f"{line}\n"
    if line == "":
        return "\n"
    return ""


def _sanitize_native_anthropic_body(body: dict) -> dict:
    """Drop known unsupported fields before native Anthropic upstream calls."""
    sanitized = dict(body)
    sanitized.pop("context_management", None)
    # Copilot limits supported effort levels per model.
    oc = sanitized.get("output_config")
    if isinstance(oc, dict) and "effort" in oc:
        effort = oc["effort"]
        model = sanitized.get("model", "")
        if "opus-4.7" in model or "opus-4-7" in model:
            # Copilot only supports medium for opus 4.7
            if effort != "medium":
                sanitized["output_config"] = {**oc, "effort": "medium"}
        elif effort in ("max", "xhigh"):
            # Copilot doesn't support max/xhigh; clamp to high
            sanitized["output_config"] = {**oc, "effort": "high"}
    return sanitized


def _extract_error_message_index(response_body: str | dict) -> int | None:
    """Extract a failing ``messages[N]`` index from an error body if present."""
    if isinstance(response_body, dict):
        text = json.dumps(response_body, ensure_ascii=False)
    else:
        text = response_body
    match = _MESSAGE_INDEX_RE.search(text)
    if not match:
        return None
    return int(match.group(1))


def _message_debug_outline(message: dict, index: int) -> dict:
    """Build a compact per-message summary for debug logging."""
    outline = {"index": index, "role": message.get("role", "")}

    content = message.get("content")
    if isinstance(content, str):
        outline["content"] = "text"
        outline["content_chars"] = len(content)
    elif isinstance(content, list):
        outline["content"] = "blocks"
        outline["block_count"] = len(content)
        block_types = []
        text_chars = 0
        for block in content[:8]:
            if not isinstance(block, dict):
                continue
            block_type = block.get("type", "")
            if block_type:
                block_types.append(block_type)
            if block_type == "text":
                text_chars += len(block.get("text", ""))
        if block_types:
            outline["block_types"] = block_types
        if text_chars:
            outline["text_chars"] = text_chars
    elif content is None:
        outline["content"] = "null"
    else:
        outline["content"] = type(content).__name__

    tool_calls = message.get("tool_calls")
    if isinstance(tool_calls, list):
        outline["tool_call_count"] = len(tool_calls)
        tool_names = [
            tc.get("function", {}).get("name", "")
            for tc in tool_calls[:5]
        ]
        if tool_names:
            outline["tool_names"] = tool_names
        empty_tool_names = [
            pos for pos, tc in enumerate(tool_calls)
            if not tc.get("function", {}).get("name")
        ]
        if empty_tool_names:
            outline["empty_tool_names"] = empty_tool_names[:5]
        outline["tool_id_lengths"] = [
            len(tc.get("id", ""))
            for tc in tool_calls[:5]
        ]

    if "tool_call_id" in message:
        outline["tool_call_id_len"] = len(str(message.get("tool_call_id", "")))

    return outline


def _log_request_message_debug(label: str, request_body: dict) -> None:
    """Log compact summaries for request messages when debugging errors."""
    messages = request_body.get("messages")
    if not isinstance(messages, list):
        return
    outlines = [_message_debug_outline(message, i) for i, message in enumerate(messages)]
    logger.debug("%s message outlines: %s", label, json.dumps(outlines, ensure_ascii=False))


def _debug_error(
    request_body: dict,
    response_body: str | dict,
    *,
    upstream_body: dict | None = None,
) -> None:
    """Log full request and response bodies at DEBUG level on errors."""
    if not logger.isEnabledFor(logging.DEBUG):
        return
    logger.debug("Error request body: %s", json.dumps(request_body, ensure_ascii=False))
    _log_request_message_debug("Error request", request_body)
    if upstream_body is not None:
        logger.debug("Upstream error request body: %s", json.dumps(upstream_body, ensure_ascii=False))
        _log_request_message_debug("Upstream error request", upstream_body)
    if isinstance(response_body, dict):
        logger.debug("Error response body: %s", json.dumps(response_body, ensure_ascii=False))
    else:
        logger.debug("Error response body: %s", response_body)

    indexed_body = upstream_body if upstream_body is not None else request_body
    messages = indexed_body.get("messages")
    error_index = _extract_error_message_index(response_body)
    if not isinstance(messages, list) or error_index is None:
        return
    if error_index < 0 or error_index >= len(messages):
        return

    logger.debug(
        "Upstream error message[%d]: %s",
        error_index,
        json.dumps(messages[error_index], ensure_ascii=False),
    )
    start = max(0, error_index - 2)
    end = min(len(messages), error_index + 3)
    window = [_message_debug_outline(messages[i], i) for i in range(start, end)]
    logger.debug("Upstream error message window: %s", json.dumps(window, ensure_ascii=False))


async def handle_chat_completion(
    adapter: FormatAdapter, body: dict, *, request: Request | None = None, initiator: str | None = None
):
    resolved = initiator or adapter.infer_initiator(body)
    openai_body = adapter.convert_chat_request(body)
    source_provider = type(adapter).__name__.removesuffix("Adapter").lower()
    openai_body["model"] = _apply_model_map(openai_body.get("model", ""))
    logger.debug(
        "Request reasoning inputs: source=%s requested_model=%s mapped_model=%s thinking=%s output_config=%s converted_thinking=%s",
        source_provider,
        body.get("model", ""),
        openai_body.get("model", ""),
        body.get("thinking"),
        body.get("output_config"),
        openai_body.get("_copilot_adapter_thinking"),
    )
    requested_model = openai_body.get("model", "")
    openai_body = _normalize_request_params(
        openai_body, source_provider, requested_model, endpoint="chat_completions"
    )

    client = await account_mgr.get_client(initiator=resolved)

    # Time-based free override: if the last request was within N minutes, mark as agent
    if _free_within_minutes is not None and resolved == "user":
        elapsed = await account_mgr.get_minutes_since_last_request(client)
        if elapsed is not None and elapsed < _free_within_minutes:
            resolved = "agent"
            logger.info("free-within-minutes: last request %.1f min ago < %.1f → agent",
                        elapsed, _free_within_minutes)

    account = account_mgr.get_username(client)

    billed_status = "yes" if resolved == "user" else "no"
    logger.info("Chat completion requested by %s (billed: %s, model: %s, account: %s)", resolved, billed_status, requested_model, account)

    is_stream = adapter.is_streaming(body) or openai_body.get("stream")

    if is_stream:
        # Request usage stats in the final streaming chunk
        openai_body["stream_options"] = {"include_usage": True}
        converter = adapter.create_stream_converter(body)

        async def event_stream():
            nonlocal client, converter, openai_body
            web_search_iterations = 0
            # Only buffer for web_search interception if web_search is in the tools
            should_intercept_web_search = (
                _web_search_max_iterations > 0
                and any(
                    t.get("function", {}).get("name") == "web_search"
                    for t in openai_body.get("tools", [])
                )
            )
            while True:
                first_chunk = True
                needs_retry = False
                buffered_lines: list[str] = []
                has_tool_calls = False

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
                        _debug_error(body, line, upstream_body=openai_body)
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
                                if not _force_free and resolved == "user":
                                    await account_mgr.record_usage(client, requested_model)
                                await account_mgr.record_request_time(client)

                    # Only buffer when we might need to intercept web_search
                    if should_intercept_web_search:
                        stripped = line.strip()
                        if stripped.startswith("data: ") and stripped[6:].strip() != "[DONE]":
                            try:
                                chunk = json.loads(stripped[6:])
                                choice = (chunk.get("choices") or [{}])[0]
                                delta = choice.get("delta", {})
                                if delta.get("tool_calls"):
                                    if not has_tool_calls:
                                        logger.debug("Tool call detected in stream — buffering for web_search check")
                                    has_tool_calls = True
                            except (json.JSONDecodeError, IndexError):
                                pass

                    if has_tool_calls:
                        buffered_lines.append(line)
                    else:
                        result = converter.feed(line)
                        if result:
                            yield result

                if needs_retry:
                    continue

                if not has_tool_calls:
                    return

                # Check buffered tool calls for web_search
                logger.debug(
                    "Stream ended with %d buffered lines, extracting tool calls",
                    len(buffered_lines),
                )
                tool_calls = _extract_tool_calls_from_stream(buffered_lines)
                if not tool_calls:
                    logger.debug("No tool calls extracted from buffer — flushing %d lines through", len(buffered_lines))
                    for bl in buffered_lines:
                        result = converter.feed(bl)
                        if result:
                            yield result
                    return

                web_calls = [tc for tc in tool_calls if tc["function"]["name"] == "web_search"]
                non_web = [tc for tc in tool_calls if tc["function"]["name"] != "web_search"]

                if not web_calls or non_web or web_search_iterations >= _web_search_max_iterations:
                    # No web_search, mixed with other tools, or max iterations — flush through
                    if web_search_iterations >= _web_search_max_iterations:
                        logger.warning("web_search loop limit reached (%d iterations), passing through", web_search_iterations)
                    logger.debug(
                        "Flushing %d buffered lines (web=%d, non_web=%d)",
                        len(buffered_lines), len(web_calls), len(non_web),
                    )
                    for bl in buffered_lines:
                        result = converter.feed(bl)
                        if result:
                            yield result
                    return

                # Pure web_search — execute server-side and continue
                web_search_iterations += 1
                logger.debug("Intercepting %d web_search call(s) (iteration %d/%d)", len(web_calls), web_search_iterations, _web_search_max_iterations)
                search_results, raw_data = await _execute_web_search_calls(web_calls)

                # For Anthropic clients, emit native server_tool_use +
                # web_search_tool_result content blocks so the client can
                # display structured search results.
                if source_provider == "anthropic":
                    # Close any open block from pre-tool-call content
                    if converter._block_started:
                        yield converter._event("content_block_stop", {
                            "type": "content_block_stop",
                            "index": converter._block_index,
                        })
                        converter._block_started = False
                        converter._block_index += 1

                    for query, raw_results in raw_data:
                        yield _build_web_search_sse_events(
                            query, raw_results, converter._block_index,
                        )
                        converter._block_index += 2

                assistant_msg = {
                    "role": "assistant",
                    "content": _extract_text_from_stream(buffered_lines) or "",
                    "tool_calls": tool_calls,
                }
                openai_body["messages"].append(assistant_msg)
                openai_body["messages"].extend(search_results)

                logger.info("Continuing after web_search (messages: %d)",
                            len(openai_body["messages"]))
                # Reuse the existing converter — it already emitted
                # message_start to the client.  Creating a fresh one would
                # produce a duplicate message_start that breaks the
                # Anthropic SSE protocol.
                continue

        return StreamingResponse(
            event_stream(), media_type="text/event-stream", headers=_STREAM_HEADERS
        )

    # Non-streaming
    while True:
        resp = await client.chat_completions(openai_body, initiator=resolved)
        if resp.status_code == 429:
            logger.warning("Rate limited on account, switching account")
            fallback = await account_mgr.get_fallback_client(client)
            if fallback is not None:
                client = fallback
                continue

        if resp.status_code != 200:
            logger.error("API error %s: %s", resp.status_code, resp.text)
            _debug_error(body, resp.text, upstream_body=openai_body)
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

    if not _force_free and resolved == "user":
        await account_mgr.record_usage(client, requested_model)
    await account_mgr.record_request_time(client)

    # Check for web_search in non-streaming response
    choice = (resp_data.get("choices") or [{}])[0]
    message = choice.get("message", {})
    tool_calls = message.get("tool_calls", [])
    web_calls = [tc for tc in tool_calls if tc.get("function", {}).get("name") == "web_search"]
    non_web = [tc for tc in tool_calls if tc.get("function", {}).get("name") != "web_search"]

    # Loop: intercept pure web_search calls, re-query until no more (max _web_search_max_iterations)
    web_search_iterations = 0
    all_raw_data: list[tuple[str, list[dict]]] = []
    while web_calls and not non_web and web_search_iterations < _web_search_max_iterations:
        web_search_iterations += 1
        logger.info("Intercepting %d web_search call(s) (non-streaming, iteration %d/%d)", len(web_calls), web_search_iterations, _web_search_max_iterations)
        search_results, raw_data = await _execute_web_search_calls(tool_calls)
        all_raw_data.extend(raw_data)
        openai_body["messages"].append({
            "role": "assistant",
            "content": message.get("content") or "",
            "tool_calls": tool_calls,
        })
        openai_body["messages"].extend(search_results)

        resp = await client.chat_completions(openai_body, initiator=resolved)
        if resp.status_code != 200:
            logger.error("API error %s on web_search continuation: %s",
                         resp.status_code, resp.text)
            try:
                content = resp.json()
            except Exception:
                content = {"error": {"message": resp.text}}
            return JSONResponse(content=content, status_code=resp.status_code)

        resp_data = resp.json()
        choice = (resp_data.get("choices") or [{}])[0]
        message = choice.get("message", {})
        tool_calls = message.get("tool_calls", [])
        web_calls = [tc for tc in tool_calls if tc.get("function", {}).get("name") == "web_search"]
        non_web = [tc for tc in tool_calls if tc.get("function", {}).get("name") != "web_search"]

    anthropic_resp = adapter.convert_chat_response(resp_data, body)

    # For Anthropic clients: prepend server_tool_use + web_search_tool_result
    # content blocks so the client can display structured search results.
    if source_provider == "anthropic" and all_raw_data:
        web_blocks: list[dict] = []
        for query, raw_results in all_raw_data:
            blocks, _ = _build_web_search_content_blocks(query, raw_results)
            web_blocks.extend(blocks)
        anthropic_resp["content"] = web_blocks + anthropic_resp.get("content", [])

    return JSONResponse(
        content=anthropic_resp,
        status_code=resp.status_code,
    )


async def handle_native_anthropic_messages(
    body: dict, *, request: Request | None = None, initiator: str | None = None,
):
    """Proxy Anthropic Messages requests directly to the upstream Anthropic API."""
    upstream_body = _sanitize_native_anthropic_body(body)
    resolved = initiator or anthropic_adapter.infer_initiator(upstream_body)
    requested_model = upstream_body.get("model", "")
    query = request.url.query if request else None

    client = await account_mgr.get_client(initiator=resolved)

    if _free_within_minutes is not None and resolved == "user":
        elapsed = await account_mgr.get_minutes_since_last_request(client)
        if elapsed is not None and elapsed < _free_within_minutes:
            resolved = "agent"
            logger.info("free-within-minutes: last request %.1f min ago < %.1f → agent",
                        elapsed, _free_within_minutes)

    account = account_mgr.get_username(client)
    billed_status = "yes" if resolved == "user" else "no"
    logger.info(
        "Anthropic native messages requested by %s (billed: %s, model: %s, account: %s)",
        resolved, billed_status, requested_model, account,
    )

    if upstream_body.get("stream"):
        converter = anthropic_adapter.create_stream_converter(upstream_body)

        async def event_stream():
            nonlocal client
            transient_retries = 0
            while True:
                needs_retry = False
                recorded = False
                async for line in client.stream_messages(
                    upstream_body, initiator=resolved, query=query,
                ):
                    if request and await request.is_disconnected():
                        return
                    if line.startswith("error:"):
                        if _is_rate_limit_error(line):
                            logger.warning("Rate limited on account, switching account")
                            fallback = await account_mgr.get_fallback_client(client)
                            if fallback is not None:
                                client = fallback
                                needs_retry = True
                                break
                        if (
                            _is_transient_upstream_error(line)
                            and not recorded
                            and transient_retries < 1
                        ):
                            transient_retries += 1
                            logger.warning(
                                "Transient upstream error before any output (%s); retrying",
                                line[:120],
                            )
                            needs_retry = True
                            break
                        _debug_error(body, line, upstream_body=upstream_body)
                        yield converter.format_error(line)
                        return
                    if not recorded and line:
                        recorded = True
                        if not _force_free and resolved == "user":
                            await account_mgr.record_usage(client, requested_model)
                        await account_mgr.record_request_time(client)
                    result = _passthrough_sse_line(line)
                    if result:
                        yield result
                if not needs_retry:
                    return

        return StreamingResponse(
            event_stream(), media_type="text/event-stream", headers=_STREAM_HEADERS,
        )

    while True:
        resp = await client.messages(upstream_body, initiator=resolved, query=query)
        if resp.status_code == 429:
            logger.warning("Rate limited on account, switching account")
            fallback = await account_mgr.get_fallback_client(client)
            if fallback is not None:
                client = fallback
                continue

        if resp.status_code != 200:
            logger.error("API error %s: %s", resp.status_code, resp.text)
            _debug_error(body, resp.text, upstream_body=upstream_body)
            try:
                content = resp.json()
            except Exception:
                content = {"error": {"message": resp.text}}
            return JSONResponse(content=content, status_code=resp.status_code)

        resp_data = resp.json()
        resp_model = resp_data.get("model", "")
        if resp_model and not _is_model_match(requested_model, resp_model):
            logger.warning(
                "Model mismatch: requested %s, got %s — quota likely exhausted, switching account",
                requested_model, resp_model,
            )
            fallback = await account_mgr.get_fallback_client(client)
            if fallback is not None:
                client = fallback
                continue
        break

    if not _force_free and resolved == "user":
        await account_mgr.record_usage(client, requested_model)
    await account_mgr.record_request_time(client)
    return JSONResponse(content=resp_data, status_code=resp.status_code)


async def handle_anthropic_via_responses(
    body: dict, *, request: Request | None = None, initiator: str | None = None,
):
    """Route an Anthropic Messages request through the OpenAI Responses API.

    This is used for target models like gpt-5.4 that require ``/v1/responses``
    to support ``reasoning_effort`` with function tools.
    """
    resolved = initiator or anthropic_adapter.infer_initiator(body)
    resp_body = _anthropic_to_responses(body)
    resp_body["model"] = _apply_model_map(resp_body.get("model", ""))
    requested_model = resp_body.get("model", "")

    resp_body = _normalize_request_params(
        resp_body, "anthropic", requested_model, endpoint="responses",
    )

    client = await account_mgr.get_client(initiator=resolved)

    if _free_within_minutes is not None and resolved == "user":
        elapsed = await account_mgr.get_minutes_since_last_request(client)
        if elapsed is not None and elapsed < _free_within_minutes:
            resolved = "agent"
            logger.info(
                "free-within-minutes: last request %.1f min ago < %.1f → agent",
                elapsed, _free_within_minutes,
            )

    account = account_mgr.get_username(client)
    billed_status = "yes" if resolved == "user" else "no"
    logger.info(
        "Anthropic→Responses requested by %s (billed: %s, model: %s, account: %s)",
        resolved, billed_status, requested_model, account,
    )

    is_stream = body.get("stream")

    # Check whether we should intercept web_search calls
    should_intercept_web_search = (
        _web_search_max_iterations > 0
        and any(
            t.get("function", {}).get("name") == "web_search"
            for t in resp_body.get("tools", [])
        )
    )

    if is_stream:
        converter = _AnthropicResponsesStreamConverter(body.get("model", ""))

        async def event_stream():
            nonlocal client, converter, resp_body
            web_search_iterations = 0
            while True:
                first_chunk = True
                needs_retry = False
                buffered_lines: list[str] = []
                has_function_calls = False

                async for line in client.stream_responses(
                    resp_body, initiator=resolved,
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
                                converter = _AnthropicResponsesStreamConverter(
                                    body.get("model", "")
                                )
                                needs_retry = True
                                break
                        _debug_error(body, line, upstream_body=resp_body)
                        yield converter.format_error(line)
                        return

                    if first_chunk:
                        resp_model = _extract_model_from_responses_sse(line)
                        if resp_model is not None:
                            first_chunk = False
                            if not _is_model_match(requested_model, resp_model):
                                logger.warning(
                                    "Model mismatch: requested %s, got %s — "
                                    "switching account",
                                    requested_model, resp_model,
                                )
                                fallback = await account_mgr.get_fallback_client(
                                    client
                                )
                                if fallback is not None:
                                    client = fallback
                                    converter = _AnthropicResponsesStreamConverter(
                                        body.get("model", "")
                                    )
                                    needs_retry = True
                                    break
                            else:
                                if not _force_free and resolved == "user":
                                    await account_mgr.record_usage(
                                        client, requested_model
                                    )
                                await account_mgr.record_request_time(client)

                    # Buffer when we might need to intercept web_search
                    if should_intercept_web_search:
                        stripped = line.strip()
                        if stripped.startswith("data: "):
                            try:
                                chunk_data = json.loads(stripped[6:])
                                chunk_type = chunk_data.get("type", "")
                                if chunk_type == "response.output_item.added":
                                    item = chunk_data.get("item", {})
                                    if item.get("type") == "function_call":
                                        if not has_function_calls:
                                            logger.debug(
                                                "function_call detected in responses "
                                                "stream — buffering for web_search check"
                                            )
                                        has_function_calls = True
                            except json.JSONDecodeError:
                                pass

                    if has_function_calls:
                        buffered_lines.append(line)
                    else:
                        result = converter.feed(line)
                        if result:
                            yield result

                if needs_retry:
                    continue

                if not has_function_calls:
                    return

                # Check buffered tool calls for web_search
                logger.debug(
                    "Responses stream ended with %d buffered lines",
                    len(buffered_lines),
                )
                tool_calls = _extract_tool_calls_from_responses_stream(
                    buffered_lines
                )
                if not tool_calls:
                    logger.debug(
                        "No function_calls extracted — flushing %d lines",
                        len(buffered_lines),
                    )
                    for bl in buffered_lines:
                        result = converter.feed(bl)
                        if result:
                            yield result
                    return

                web_calls = [
                    tc for tc in tool_calls if tc["name"] == "web_search"
                ]
                non_web = [
                    tc for tc in tool_calls if tc["name"] != "web_search"
                ]

                if (
                    not web_calls
                    or non_web
                    or web_search_iterations >= _web_search_max_iterations
                ):
                    if web_search_iterations >= _web_search_max_iterations:
                        logger.warning(
                            "web_search loop limit reached (%d), passing through",
                            web_search_iterations,
                        )
                    for bl in buffered_lines:
                        result = converter.feed(bl)
                        if result:
                            yield result
                    return

                # Pure web_search — execute server-side and continue
                web_search_iterations += 1
                logger.debug(
                    "Intercepting %d web_search call(s) (responses, iter %d/%d)",
                    len(web_calls),
                    web_search_iterations,
                    _web_search_max_iterations,
                )

                # Execute web searches
                search_tool_calls = [
                    {
                        "id": tc["call_id"],
                        "function": {
                            "name": "web_search",
                            "arguments": tc["arguments"],
                        },
                    }
                    for tc in web_calls
                ]
                _, raw_data = await _execute_web_search_calls(search_tool_calls)

                # Emit Anthropic web search blocks
                if converter._block_started:
                    yield converter._event("content_block_stop", {
                        "type": "content_block_stop",
                        "index": converter._block_index,
                    })
                    converter._block_started = False
                    converter._block_index += 1

                for query, raw_results in raw_data:
                    yield _build_web_search_sse_events(
                        query, raw_results, converter._block_index,
                    )
                    converter._block_index += 2

                # Add function_call + function_call_output items to input
                for tc in web_calls:
                    resp_body["input"].append({
                        "type": "function_call",
                        "name": tc["name"],
                        "arguments": tc["arguments"],
                        "call_id": tc["call_id"],
                    })
                for tc, (query, raw_results) in zip(web_calls, raw_data):
                    resp_body["input"].append({
                        "type": "function_call_output",
                        "call_id": tc["call_id"],
                        "output": _format_search_results(raw_results),
                    })

                # Also append any text from the buffered response
                text = _extract_text_from_responses_stream(buffered_lines)
                if text:
                    resp_body["input"].append({
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": text}],
                    })

                logger.info(
                    "Continuing after web_search (input items: %d)",
                    len(resp_body["input"]),
                )
                continue

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers=_STREAM_HEADERS,
        )

    # Non-streaming
    while True:
        resp = await client.responses(resp_body, initiator=resolved)
        if resp.status_code == 429:
            logger.warning("Rate limited on account, switching account")
            fallback = await account_mgr.get_fallback_client(client)
            if fallback is not None:
                client = fallback
                continue

        if resp.status_code != 200:
            logger.error("API error %s: %s", resp.status_code, resp.text)
            _debug_error(body, resp.text, upstream_body=resp_body)
            try:
                content = resp.json()
            except Exception:
                content = {"error": {"message": resp.text}}
            return JSONResponse(content=content, status_code=resp.status_code)

        resp_data = resp.json()
        resp_model = resp_data.get("model", "")
        if resp_model and not _is_model_match(requested_model, resp_model):
            logger.warning(
                "Model mismatch: requested %s, got %s — switching account",
                requested_model, resp_model,
            )
            fallback = await account_mgr.get_fallback_client(client)
            if fallback is not None:
                client = fallback
                continue
        break

    if not _force_free and resolved == "user":
        await account_mgr.record_usage(client, requested_model)
    await account_mgr.record_request_time(client)

    # Check for web_search in non-streaming response
    output_items = resp_data.get("output", [])
    fn_calls = [
        item for item in output_items if item.get("type") == "function_call"
    ]
    web_calls = [tc for tc in fn_calls if tc.get("name") == "web_search"]
    non_web = [tc for tc in fn_calls if tc.get("name") != "web_search"]

    # Web search interception loop
    web_search_iterations = 0
    all_raw_data: list[tuple[str, list[dict]]] = []
    while (
        web_calls
        and not non_web
        and should_intercept_web_search
        and web_search_iterations < _web_search_max_iterations
    ):
        web_search_iterations += 1
        logger.info(
            "Intercepting %d web_search call(s) (responses non-streaming, "
            "iter %d/%d)",
            len(web_calls),
            web_search_iterations,
            _web_search_max_iterations,
        )

        search_tool_calls = [
            {
                "id": tc.get("call_id", ""),
                "function": {
                    "name": "web_search",
                    "arguments": tc.get("arguments", ""),
                },
            }
            for tc in web_calls
        ]
        _, raw_data = await _execute_web_search_calls(search_tool_calls)
        all_raw_data.extend(raw_data)

        # Append function_call items from output
        for item in fn_calls:
            resp_body["input"].append({
                "type": "function_call",
                "name": item.get("name", ""),
                "arguments": item.get("arguments", ""),
                "call_id": item.get("call_id", ""),
            })
        # Append function_call_output items
        for tc, (query, raw_results) in zip(web_calls, raw_data):
            resp_body["input"].append({
                "type": "function_call_output",
                "call_id": tc.get("call_id", ""),
                "output": _format_search_results(raw_results),
            })

        # Append any text from the response
        for item in output_items:
            if item.get("type") == "message":
                for part in item.get("content", []):
                    if part.get("type") == "output_text" and part.get("text"):
                        resp_body["input"].append({
                            "type": "message",
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": part["text"]}],
                        })

        resp = await client.responses(resp_body, initiator=resolved)
        if resp.status_code != 200:
            logger.error(
                "API error %s on web_search continuation: %s",
                resp.status_code, resp.text,
            )
            try:
                content = resp.json()
            except Exception:
                content = {"error": {"message": resp.text}}
            return JSONResponse(content=content, status_code=resp.status_code)

        resp_data = resp.json()
        output_items = resp_data.get("output", [])
        fn_calls = [
            item for item in output_items if item.get("type") == "function_call"
        ]
        web_calls = [tc for tc in fn_calls if tc.get("name") == "web_search"]
        non_web = [tc for tc in fn_calls if tc.get("name") != "web_search"]

    anthropic_resp = _responses_to_anthropic(resp_data, body.get("model", ""))

    # Prepend web search content blocks for Anthropic clients
    if all_raw_data:
        web_blocks: list[dict] = []
        for query, raw_results in all_raw_data:
            blocks, _ = _build_web_search_content_blocks(query, raw_results)
            web_blocks.extend(blocks)
        anthropic_resp["content"] = web_blocks + anthropic_resp.get("content", [])

    return JSONResponse(content=anthropic_resp, status_code=resp.status_code)


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
    original_model = body.get("model", "")
    body["model"] = _apply_model_map(original_model)
    requested_model = body.get("model", "")
    logger.debug(
        "Responses reasoning inputs: requested_model=%s mapped_model=%s thinking=%s output_config=%s reasoning=%s reasoning_effort=%s",
        original_model,
        requested_model,
        body.get("thinking"),
        body.get("output_config"),
        body.get("reasoning"),
        body.get("reasoning_effort"),
    )
    if "thinking" in body:
        body["_copilot_adapter_thinking"] = body.pop("thinking")
        body = _normalize_request_params(
            body, "anthropic", requested_model, endpoint="responses"
        )

    client = await account_mgr.get_client(initiator=initiator)

    # Time-based free override
    if _free_within_minutes is not None and initiator == "user":
        elapsed = await account_mgr.get_minutes_since_last_request(client)
        if elapsed is not None and elapsed < _free_within_minutes:
            initiator = "agent"
            logger.info("free-within-minutes: last request %.1f min ago < %.1f → agent",
                        elapsed, _free_within_minutes)

    account = account_mgr.get_username(client)

    billed_status = "yes" if initiator == "user" else "no"
    logger.info("Responses requested by %s (billed: %s, model: %s, account: %s)", initiator, billed_status, requested_model, account)

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
                        _debug_error(body, line, upstream_body=body)
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
                                if not _force_free and initiator == "user":
                                    await account_mgr.record_usage(client, requested_model)
                                await account_mgr.record_request_time(client)
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
            _debug_error(body, resp.text, upstream_body=body)
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
    if not _force_free and initiator == "user":
        await account_mgr.record_usage(client, requested_model)
    await account_mgr.record_request_time(client)
    return JSONResponse(content=resp_data, status_code=resp.status_code)


@app.post("/v1/embeddings")
@app.post("/embeddings")
async def embeddings(request: Request):
    body = await request.json()
    initiator = _get_initiator(request) or "user"
    body["model"] = _apply_model_map(body.get("model", ""))
    client = await account_mgr.get_client(initiator=initiator)
    account = account_mgr.get_username(client)
    model = body.get("model", "")

    billed_status = "yes" if initiator == "user" else "no"
    logger.info("Embeddings requested by %s (billed: %s, model: %s, account: %s)", initiator, billed_status, model, account)
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
        _debug_error(body, resp.text, upstream_body=body)
    return JSONResponse(content=content, status_code=resp.status_code)


# ---------------------------------------------------------------------------
# Anthropic endpoint
# ---------------------------------------------------------------------------

@app.post("/v1/messages/count_tokens")
@app.post("/messages/count_tokens")
async def count_tokens(request: Request):
    """Stub for the Anthropic token counting API."""
    try:
        body = await request.json()
    except Exception:
        body = {}

    model = body.get("model", "unknown")
    num_messages = len(body.get("messages", []))
    num_tools = len(body.get("tools", []))

    # Simple heuristic: ~4 chars per token for text content
    text_content = json.dumps(body)
    token_count = max(1, len(text_content) // 4)

    logger.debug(
        "count_tokens: model=%s messages=%d tools=%d -> %d tokens",
        model, num_messages, num_tools, token_count,
    )

    return JSONResponse(content={"input_tokens": token_count})


@app.post("/v1/messages")
@app.post("/messages")
async def messages(request: Request):
    body = await request.json()
    mapped_model = _apply_model_map(body.get("model", ""))
    if (
        _should_use_native_anthropic_api("anthropic", mapped_model)
        and not _anthropic_has_interceptable_web_search(body)
    ):
        body["model"] = mapped_model
        return await handle_native_anthropic_messages(
            body, request=request, initiator=_get_initiator(request)
        )
    if _should_use_responses_api(mapped_model):
        body["model"] = mapped_model
        return await handle_anthropic_via_responses(
            body, request=request, initiator=_get_initiator(request)
        )
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
    openai_body["model"] = _apply_model_map(openai_body.get("model", ""))
    openai_body["stream"] = True
    requested_model = openai_body.get("model", "")
    openai_body = _normalize_request_params(openai_body, "gemini", requested_model)

    client = await account_mgr.get_client(initiator=resolved)

    # Time-based free override
    if _free_within_minutes is not None and resolved == "user":
        elapsed = await account_mgr.get_minutes_since_last_request(client)
        if elapsed is not None and elapsed < _free_within_minutes:
            resolved = "agent"
            logger.info("free-within-minutes: last request %.1f min ago < %.1f → agent",
                        elapsed, _free_within_minutes)

    account = account_mgr.get_username(client)

    billed_status = "yes" if resolved == "user" else "no"
    logger.info("Chat completion requested by %s (billed: %s, model: %s, account: %s)", resolved, billed_status, requested_model, account)

    openai_body["stream_options"] = {"include_usage": True}
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
                    _debug_error(body, line, upstream_body=openai_body)
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
                            if not _force_free and resolved == "user":
                                await account_mgr.record_usage(client, requested_model)
                            await account_mgr.record_request_time(client)
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

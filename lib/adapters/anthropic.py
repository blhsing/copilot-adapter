"""Anthropic Messages API format adapter."""

import hashlib
import json
import logging
import uuid

from .base import FormatAdapter, StreamConverter

logger = logging.getLogger(__name__)


_MAX_OPENAI_TOOL_ID_LENGTH = 64


def _normalize_openai_tool_id(tool_id: str) -> str:
    """Keep tool IDs within upstream OpenAI-compatible provider limits."""
    if len(tool_id) <= _MAX_OPENAI_TOOL_ID_LENGTH:
        return tool_id
    digest = hashlib.sha256(tool_id.encode("utf-8")).hexdigest()
    return f"toolu_{digest[:_MAX_OPENAI_TOOL_ID_LENGTH - len('toolu_')]}"


def _assistant_content_or_empty(text_parts: list[str]) -> str:
    """Return assistant text content, using an empty string when omitted."""
    return "\n".join(text_parts) if text_parts else ""


# ---------------------------------------------------------------------------
# Shared Anthropic tool schemas and conversion helpers
# ---------------------------------------------------------------------------

_BUILTIN_TOOL_SCHEMAS = {
    "web_search": {
        "description": "Search the web for information.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query.",
                },
            },
            "required": ["query"],
        },
    },
}


def _convert_anthropic_tools(tools: list[dict], *, preserve_native_web_search: bool = False) -> list[dict]:
    """Convert Anthropic tool definitions to OpenAI-compatible tool format."""
    func_tools = []
    for t in tools:
        tool_type = t.get("type", "")
        builtin_key = None
        for prefix in _BUILTIN_TOOL_SCHEMAS:
            if tool_type.startswith(prefix):
                builtin_key = prefix
                break
        if builtin_key == "web_search" and preserve_native_web_search:
            logger.info("Preserving Anthropic built-in tool type=%s for native Responses web search", tool_type)
            func_tools.append({"type": "web_search_preview"})
        elif builtin_key:
            schema = _BUILTIN_TOOL_SCHEMAS[builtin_key]
            tool_name = t.get("name", builtin_key)
            logger.info(
                "Converting Anthropic built-in tool type=%s -> "
                "function tool name=%s",
                tool_type, tool_name,
            )
            func_tools.append({
                "type": "function",
                "function": {
                    "name": tool_name,
                    "description": schema["description"],
                    "parameters": schema["parameters"],
                },
            })
        elif tool_type not in ("", "function", "custom"):
            logger.info("Stripping unsupported built-in tool type=%s", tool_type)
        else:
            func_tools.append({
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "parameters": t.get("input_schema", {}),
                },
            })
    return func_tools


def _convert_anthropic_tool_choice(tc, available_tools: list[dict]) -> str | dict | None:
    """Convert Anthropic tool_choice to OpenAI format."""
    if not tc:
        return None
    tc_type = tc.get("type") if isinstance(tc, dict) else tc
    if tc_type == "tool":
        forced_name = tc.get("name", "")
        has_tool = any(
            ft["function"]["name"] == forced_name
            for ft in available_tools
        )
        if has_tool:
            return {"type": "function", "function": {"name": forced_name}}
        return None
    elif tc_type == "auto":
        return "auto"
    elif tc_type == "any":
        return "required"
    elif tc_type == "none":
        return "none"
    return None


# ---------------------------------------------------------------------------
# Request conversion: Anthropic -> OpenAI
# ---------------------------------------------------------------------------

def _convert_content_blocks(blocks: list, role: str, tool_id_map: dict[str, str]) -> list[dict]:
    messages = []
    text_parts = []
    openai_content_parts = []
    has_multimodal = False
    pending_tool_calls = []

    for block in blocks:
        btype = block.get("type")

        if btype == "text":
            text_parts.append(block["text"])
            openai_content_parts.append({"type": "text", "text": block["text"]})

        elif btype == "image":
            has_multimodal = True
            source = block["source"]
            if source["type"] == "base64":
                url = f"data:{source['media_type']};base64,{source['data']}"
            else:
                url = source["url"]
            openai_content_parts.append({
                "type": "image_url",
                "image_url": {"url": url},
            })

        elif btype == "tool_use":
            original_tool_id = block["id"]
            openai_tool_id = tool_id_map.setdefault(
                original_tool_id,
                _normalize_openai_tool_id(original_tool_id),
            )
            pending_tool_calls.append({
                "id": openai_tool_id,
                "type": "function",
                "function": {
                    "name": block["name"],
                    "arguments": json.dumps(block["input"]),
                },
            })

        elif btype == "tool_result":
            # Flush pending tool calls before tool results
            if pending_tool_calls:
                content_text = _assistant_content_or_empty(text_parts)
                messages.append({
                    "role": "assistant",
                    "content": content_text,
                    "tool_calls": pending_tool_calls,
                })
                text_parts = []
                openai_content_parts = []
                pending_tool_calls = []
            elif text_parts:
                messages.append({"role": role, "content": "\n".join(text_parts)})
                text_parts = []
                openai_content_parts = []
            tool_content = block.get("content", "")
            if isinstance(tool_content, list):
                tool_content = "\n".join(
                    b.get("text", "") for b in tool_content if b.get("type") == "text"
                )
            messages.append({
                "role": "tool",
                "tool_call_id": tool_id_map.get(
                    block["tool_use_id"],
                    _normalize_openai_tool_id(block["tool_use_id"]),
                ),
                "content": str(tool_content),
            })

    # Flush remaining content
    if pending_tool_calls:
        content_text = _assistant_content_or_empty(text_parts)
        messages.append({
            "role": "assistant",
            "content": content_text,
            "tool_calls": pending_tool_calls,
        })
    elif openai_content_parts:
        if has_multimodal:
            messages.append({"role": role, "content": openai_content_parts})
        elif text_parts:
            messages.append({"role": role, "content": "\n".join(text_parts)})

    return messages


def _anthropic_to_openai(body: dict) -> dict:
    messages = []
    tool_id_map: dict[str, str] = {}

    model = body.get("model", "")

    system = body.get("system")
    if system:
        if isinstance(system, list):
            text = "\n".join(b.get("text", "") for b in system)
        else:
            text = system
        messages.append({"role": "system", "content": text})

    for msg in body.get("messages", []):
        role = msg["role"]
        content = msg["content"]

        if isinstance(content, str):
            messages.append({"role": role, "content": content})
        elif isinstance(content, list):
            messages.extend(_convert_content_blocks(content, role, tool_id_map))
        else:
            messages.append({"role": role, "content": content})

    result = {
        "model": model,
        "messages": messages,
        "stream": body.get("stream", False),
    }

    if "max_tokens" in body:
        result["max_tokens"] = body["max_tokens"]
    if "thinking" in body:
        result["_copilot_adapter_thinking"] = body["thinking"]
    if isinstance(body.get("output_config"), dict) and body["output_config"].get("effort"):
        result["_copilot_adapter_output_effort"] = body["output_config"]["effort"]
    if "temperature" in body:
        result["temperature"] = body["temperature"]
    if "top_p" in body:
        result["top_p"] = body["top_p"]
    if "stop_sequences" in body:
        result["stop"] = body["stop_sequences"]

    if "tools" in body:
        func_tools = _convert_anthropic_tools(body["tools"])
        if func_tools:
            result["tools"] = func_tools

    tc_converted = _convert_anthropic_tool_choice(
        body.get("tool_choice"), result.get("tools", [])
    )
    if tc_converted is not None:
        result["tool_choice"] = tc_converted

    return result


# ---------------------------------------------------------------------------
# Request conversion: Anthropic -> OpenAI Responses API
# ---------------------------------------------------------------------------

def _anthropic_to_responses(body: dict, *, preserve_native_web_search: bool = True) -> dict:
    """Convert an Anthropic Messages body to an OpenAI Responses API body."""
    input_items: list[dict] = []
    tool_id_map: dict[str, str] = {}

    model = body.get("model", "")

    # System prompt → developer role
    system = body.get("system")
    if system:
        if isinstance(system, list):
            text = "\n".join(b.get("text", "") for b in system)
        else:
            text = system
        input_items.append({"role": "developer", "content": text})

    # Messages
    for msg in body.get("messages", []):
        role = msg["role"]
        content = msg["content"]

        if isinstance(content, str):
            input_items.append({"role": role, "content": content})
        elif isinstance(content, list):
            _convert_blocks_to_responses_items(
                content, role, input_items, tool_id_map,
            )
        else:
            input_items.append({"role": role, "content": content})

    result: dict = {
        "model": model,
        "input": input_items,
        "stream": body.get("stream", False),
    }

    if "max_tokens" in body:
        # OpenAI Responses API enforces a minimum of 16 output tokens.
        result["max_output_tokens"] = max(16, body["max_tokens"])
    if "thinking" in body:
        result["_copilot_adapter_thinking"] = body["thinking"]
    if isinstance(body.get("output_config"), dict) and body["output_config"].get("effort"):
        result["_copilot_adapter_output_effort"] = body["output_config"]["effort"]
    if "temperature" in body:
        result["temperature"] = body["temperature"]
    if "top_p" in body:
        result["top_p"] = body["top_p"]
    if "stop_sequences" in body:
        result["stop"] = body["stop_sequences"]

    if "tools" in body:
        func_tools = _convert_anthropic_tools(
            body["tools"], preserve_native_web_search=preserve_native_web_search,
        )
        # Responses API uses flat tool format: {type, name, description, parameters}
        # instead of chat/completions nested {type, function: {name, ...}}
        resp_tools = []
        for ft in func_tools:
            if ft.get("type") == "web_search_preview":
                resp_tools.append(ft)
                continue
            fn = ft.get("function", {})
            resp_tools.append({
                "type": "function",
                "name": fn.get("name", ""),
                "description": fn.get("description", ""),
                "parameters": fn.get("parameters", {}),
            })
        if resp_tools:
            result["tools"] = resp_tools

    tc = body.get("tool_choice")
    if tc:
        tc_type = tc.get("type") if isinstance(tc, dict) else tc
        if tc_type == "tool":
            forced_name = tc.get("name", "")
            if any(t.get("name") == forced_name for t in result.get("tools", [])):
                result["tool_choice"] = {"type": "function", "name": forced_name}
        elif tc_type == "auto":
            result["tool_choice"] = "auto"
        elif tc_type == "any":
            result["tool_choice"] = "required"
        elif tc_type == "none":
            result["tool_choice"] = "none"

    return result


def _convert_blocks_to_responses_items(
    blocks: list, role: str, items: list[dict], tool_id_map: dict[str, str],
) -> None:
    """Convert Anthropic content blocks to Responses API input items in-place."""
    text_parts: list[str] = []
    multimodal_parts: list[dict] = []
    has_multimodal = False

    def _flush_text():
        nonlocal has_multimodal
        if has_multimodal and multimodal_parts:
            items.append({"role": role, "content": list(multimodal_parts)})
            multimodal_parts.clear()
            text_parts.clear()
            has_multimodal = False
        elif text_parts:
            if role == "assistant":
                items.append({
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "\n".join(text_parts)}],
                })
            else:
                items.append({"role": role, "content": "\n".join(text_parts)})
            text_parts.clear()
            multimodal_parts.clear()

    for block in blocks:
        btype = block.get("type")

        if btype == "text":
            text_parts.append(block["text"])
            multimodal_parts.append({"type": "input_text", "text": block["text"]})

        elif btype == "image":
            has_multimodal = True
            source = block["source"]
            if source["type"] == "base64":
                url = f"data:{source['media_type']};base64,{source['data']}"
            else:
                url = source["url"]
            multimodal_parts.append({"type": "input_image", "image_url": url})

        elif btype == "tool_use":
            _flush_text()
            original_id = block["id"]
            call_id = tool_id_map.setdefault(
                original_id,
                _normalize_openai_tool_id(original_id),
            )
            items.append({
                "type": "function_call",
                "name": block["name"],
                "arguments": json.dumps(block["input"]),
                "call_id": call_id,
            })

        elif btype == "tool_result":
            _flush_text()
            tool_content = block.get("content", "")
            if isinstance(tool_content, list):
                tool_content = "\n".join(
                    b.get("text", "") for b in tool_content if b.get("type") == "text"
                )
            original_id = block["tool_use_id"]
            call_id = tool_id_map.get(
                original_id,
                _normalize_openai_tool_id(original_id),
            )
            items.append({
                "type": "function_call_output",
                "call_id": call_id,
                "output": str(tool_content),
            })

    _flush_text()


# ---------------------------------------------------------------------------
# Response conversion: OpenAI -> Anthropic
# ---------------------------------------------------------------------------

_STOP_REASON_MAP = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
    "content_filter": "end_turn",
}


def _openai_to_anthropic(openai_resp: dict, model: str) -> dict:
    choice = openai_resp.get("choices", [{}])[0]
    message = choice.get("message", {})

    content = []

    text = message.get("content")
    if text:
        content.append({"type": "text", "text": text})

    for tc in message.get("tool_calls", []):
        func = tc.get("function", {})
        try:
            input_obj = json.loads(func.get("arguments", "{}"))
        except json.JSONDecodeError:
            input_obj = {}
        tool_id = tc.get("id", f"toolu_{uuid.uuid4().hex[:24]}")
        tool_name = func.get("name", "")
        logger.info(
            "Response contains tool_use: id=%s name=%s",
            tool_id, tool_name,
        )
        content.append({
            "type": "tool_use",
            "id": tool_id,
            "name": tool_name,
            "input": input_obj,
        })

    if not content:
        content.append({"type": "text", "text": ""})

    finish = choice.get("finish_reason", "stop")
    usage = openai_resp.get("usage", {})

    return {
        "id": f"msg_{uuid.uuid4().hex[:24]}",
        "type": "message",
        "role": "assistant",
        "content": content,
        "model": model,
        "stop_reason": _STOP_REASON_MAP.get(finish, "end_turn"),
        "stop_sequence": None,
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
        },
    }


# ---------------------------------------------------------------------------
# Response conversion: OpenAI Responses API -> Anthropic
# ---------------------------------------------------------------------------

_RESPONSES_STATUS_MAP = {
    "completed": "end_turn",
    "incomplete": "max_tokens",
    "failed": "end_turn",
}


def _url_citation_to_search_result(ann: dict) -> dict:
    """Convert an OpenAI url_citation annotation to an Anthropic web_search_result block."""
    return {
        "type": "web_search_result",
        "url": ann.get("url", ""),
        "title": ann.get("title", ""),
        "encrypted_content": "",
        "page_age": "",
    }


def _responses_to_anthropic(resp: dict, model: str) -> dict:
    """Convert an OpenAI Responses API response to Anthropic Messages format."""
    content: list[dict] = []
    has_function_calls = False
    pending_search_ids: list[str] = []

    for item in resp.get("output", []):
        item_type = item.get("type", "")

        if item_type == "web_search_call":
            tool_id = item.get("id") or f"srvtoolu_{uuid.uuid4().hex[:24]}"
            action = item.get("action") or {}
            query = action.get("query") or item.get("query", "")
            content.append({
                "type": "server_tool_use",
                "id": tool_id,
                "name": "web_search",
                "input": {"query": query} if query else {},
            })
            pending_search_ids.append(tool_id)

        elif item_type == "message":
            citations: list[dict] = []
            for part in item.get("content", []):
                for ann in part.get("annotations") or []:
                    if ann.get("type") == "url_citation":
                        citations.append(_url_citation_to_search_result(ann))
            if pending_search_ids:
                tool_id = pending_search_ids.pop(0)
                content.append({
                    "type": "web_search_tool_result",
                    "tool_use_id": tool_id,
                    "content": citations,
                })
            for part in item.get("content", []):
                part_type = part.get("type", "")
                if part_type == "output_text":
                    content.append({"type": "text", "text": part.get("text", "")})
                elif part_type == "refusal":
                    content.append({"type": "text", "text": part.get("refusal", "")})

        elif item_type == "function_call":
            has_function_calls = True
            call_id = item.get("call_id", f"toolu_{uuid.uuid4().hex[:24]}")
            name = item.get("name", "")
            try:
                input_obj = json.loads(item.get("arguments", "{}"))
            except json.JSONDecodeError:
                input_obj = {}
            logger.info("Responses API tool_use: id=%s name=%s", call_id, name)
            content.append({
                "type": "tool_use",
                "id": call_id,
                "name": name,
                "input": input_obj,
            })

    # Any web_search_call not followed by a citing message still needs a
    # matching web_search_tool_result block so the pair is well-formed.
    for tool_id in pending_search_ids:
        content.append({
            "type": "web_search_tool_result",
            "tool_use_id": tool_id,
            "content": [],
        })

    if not content:
        content.append({"type": "text", "text": ""})

    status = resp.get("status", "completed")
    if has_function_calls:
        stop_reason = "tool_use"
    else:
        stop_reason = _RESPONSES_STATUS_MAP.get(status, "end_turn")

    usage = resp.get("usage", {})

    return {
        "id": f"msg_{uuid.uuid4().hex[:24]}",
        "type": "message",
        "role": "assistant",
        "content": content,
        "model": model,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": usage.get("input_tokens", 0),
            "output_tokens": usage.get("output_tokens", 0),
        },
    }

# ---------------------------------------------------------------------------
# Stream converters
# ---------------------------------------------------------------------------

class _AnthropicResponsesStreamConverter(StreamConverter):
    """Convert OpenAI Responses API SSE events to Anthropic SSE events."""

    def __init__(self, model: str):
        self._model = model
        self._started = False
        self._block_started = False
        self._block_index = 0
        self._input_tokens = 0
        self._output_tokens = 0
        self._stop_reason: str | None = None
        # Track pending native web_search_call → citation pairs so we can emit
        # matching web_search_tool_result blocks at the end of the stream.
        self._pending_searches: list[dict] = []
        self._last_web_search_tool_id: str | None = None

    def _event(self, event_type: str, data: dict) -> str:
        return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"

    def _record_url_citation(self, ann: dict) -> None:
        """Attach a url_citation annotation to the most recent web_search_call."""
        if not self._pending_searches:
            return
        target = self._pending_searches[-1]
        result = _url_citation_to_search_result(ann)
        if result not in target["citations"]:
            target["citations"].append(result)

    def format_error(self, error_line: str) -> str:
        return f"event: error\ndata: {error_line}\n\n"

    def feed(self, line: str) -> str:
        line = line.strip()
        if not line:
            return ""

        # Responses API SSE uses both event: and data: lines.
        # Skip event: lines — the type field inside data: is sufficient.
        if not line.startswith("data: "):
            return ""

        payload = line[6:].strip()
        if payload == "[DONE]":
            return ""

        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            return ""

        event_type = data.get("type", "")
        output = ""

        if event_type == "response.created":
            if not self._started:
                self._started = True
                output += self._event("message_start", {
                    "type": "message_start",
                    "message": {
                        "id": f"msg_{uuid.uuid4().hex[:24]}",
                        "type": "message",
                        "role": "assistant",
                        "content": [],
                        "model": self._model,
                        "stop_reason": None,
                        "stop_sequence": None,
                        "usage": {
                            "input_tokens": 0,
                            "output_tokens": 0,
                        },
                    },
                })

        elif event_type == "response.content_part.added":
            part = data.get("part", {})
            if part.get("type") == "output_text":
                if self._block_started:
                    output += self._event("content_block_stop", {
                        "type": "content_block_stop",
                        "index": self._block_index,
                    })
                    self._block_index += 1
                self._block_started = True
                output += self._event("content_block_start", {
                    "type": "content_block_start",
                    "index": self._block_index,
                    "content_block": {"type": "text", "text": ""},
                })

        elif event_type == "response.output_text.delta":
            delta_text = data.get("delta", "")
            if delta_text:
                if not self._block_started:
                    self._block_started = True
                    output += self._event("content_block_start", {
                        "type": "content_block_start",
                        "index": self._block_index,
                        "content_block": {"type": "text", "text": ""},
                    })
                output += self._event("content_block_delta", {
                    "type": "content_block_delta",
                    "index": self._block_index,
                    "delta": {"type": "text_delta", "text": delta_text},
                })

        elif event_type == "response.output_item.added":
            item = data.get("item", {})
            if item.get("type") == "function_call":
                if self._block_started:
                    output += self._event("content_block_stop", {
                        "type": "content_block_stop",
                        "index": self._block_index,
                    })
                    self._block_index += 1
                self._block_started = True
                call_id = item.get("call_id", f"toolu_{uuid.uuid4().hex[:24]}")
                name = item.get("name", "")
                output += self._event("content_block_start", {
                    "type": "content_block_start",
                    "index": self._block_index,
                    "content_block": {
                        "type": "tool_use",
                        "id": call_id,
                        "name": name,
                        "input": {},
                    },
                })
            elif item.get("type") == "web_search_call":
                if self._block_started:
                    output += self._event("content_block_stop", {
                        "type": "content_block_stop",
                        "index": self._block_index,
                    })
                    self._block_index += 1
                    self._block_started = False
                tool_id = item.get("id") or f"srvtoolu_{uuid.uuid4().hex[:24]}"
                self._last_web_search_tool_id = tool_id
                self._pending_searches.append({"tool_id": tool_id, "citations": []})
                output += self._event("content_block_start", {
                    "type": "content_block_start",
                    "index": self._block_index,
                    "content_block": {
                        "type": "server_tool_use",
                        "id": tool_id,
                        "name": "web_search",
                        "input": {},
                    },
                })
                self._block_started = True

        elif event_type == "response.function_call_arguments.delta":
            args_chunk = data.get("delta", "")
            if args_chunk:
                output += self._event("content_block_delta", {
                    "type": "content_block_delta",
                    "index": self._block_index,
                    "delta": {
                        "type": "input_json_delta",
                        "partial_json": args_chunk,
                    },
                })

        elif event_type == "response.output_item.done":
            item = data.get("item", {})
            if item.get("type") == "function_call":
                if self._block_started:
                    output += self._event("content_block_stop", {
                        "type": "content_block_stop",
                        "index": self._block_index,
                    })
                    self._block_started = False
                    self._block_index += 1
                self._stop_reason = "tool_use"
            elif item.get("type") == "web_search_call":
                action = item.get("action") or {}
                query = action.get("query") or item.get("query", "")
                if query:
                    output += self._event("content_block_delta", {
                        "type": "content_block_delta",
                        "index": self._block_index,
                        "delta": {
                            "type": "input_json_delta",
                            "partial_json": json.dumps({"query": query}),
                        },
                    })
                if self._block_started:
                    output += self._event("content_block_stop", {
                        "type": "content_block_stop",
                        "index": self._block_index,
                    })
                    self._block_started = False
                    self._block_index += 1
            elif item.get("type") == "message":
                # Final annotations may only be present on the done event.
                for part in item.get("content") or []:
                    for ann in part.get("annotations") or []:
                        if ann.get("type") == "url_citation":
                            self._record_url_citation(ann)

        elif event_type == "response.content_part.done":
            part = data.get("part") or {}
            for ann in part.get("annotations") or []:
                if ann.get("type") == "url_citation":
                    self._record_url_citation(ann)
            if self._block_started:
                output += self._event("content_block_stop", {
                    "type": "content_block_stop",
                    "index": self._block_index,
                })
                self._block_started = False
                self._block_index += 1

        elif event_type in (
            "response.output_text.annotation.added",
            "response.output_text_annotation.added",
        ):
            ann = data.get("annotation") or {}
            if ann.get("type") == "url_citation":
                self._record_url_citation(ann)

        elif event_type == "response.completed":
            resp = data.get("response", {})
            usage = resp.get("usage", {})
            self._input_tokens = usage.get("input_tokens", self._input_tokens)
            self._output_tokens = usage.get("output_tokens", self._output_tokens)

            if self._block_started:
                output += self._event("content_block_stop", {
                    "type": "content_block_stop",
                    "index": self._block_index,
                })
                self._block_started = False
                self._block_index += 1

            # Fallback: if the stream never surfaced annotations individually,
            # pull them from the completed response's message content.
            if self._pending_searches and any(
                not s["citations"] for s in self._pending_searches
            ):
                for item in resp.get("output") or []:
                    if item.get("type") != "message":
                        continue
                    for part in item.get("content") or []:
                        for ann in part.get("annotations") or []:
                            if ann.get("type") == "url_citation":
                                self._record_url_citation(ann)

            # Emit a web_search_tool_result block for each pending search.
            for search in self._pending_searches:
                tool_id = search["tool_id"]
                citations = search["citations"]
                output += self._event("content_block_start", {
                    "type": "content_block_start",
                    "index": self._block_index,
                    "content_block": {
                        "type": "web_search_tool_result",
                        "tool_use_id": tool_id,
                        "content": [],
                    },
                })
                if citations:
                    output += self._event("content_block_delta", {
                        "type": "content_block_delta",
                        "index": self._block_index,
                        "delta": {
                            "type": "web_search_result_delta",
                            "search_results": citations,
                        },
                    })
                output += self._event("content_block_stop", {
                    "type": "content_block_stop",
                    "index": self._block_index,
                })
                self._block_index += 1
            self._pending_searches = []

            # Derive stop reason from response status
            status = resp.get("status", "completed")
            has_fn_calls = any(
                item.get("type") == "function_call"
                for item in resp.get("output", [])
            )
            if has_fn_calls:
                stop = "tool_use"
            else:
                stop = self._stop_reason or _RESPONSES_STATUS_MAP.get(status, "end_turn")

            output += self._event("message_delta", {
                "type": "message_delta",
                "delta": {"stop_reason": stop, "stop_sequence": None},
                "usage": {
                    "input_tokens": self._input_tokens,
                    "output_tokens": self._output_tokens,
                },
            })
            output += self._event("message_stop", {"type": "message_stop"})

        return output


class _AnthropicStreamConverter(StreamConverter):
    def __init__(self, model: str):
        self._model = model
        self._started = False
        self._block_started = False
        self._block_index = 0
        self._tool_blocks: dict[int, dict] = {}
        self._input_tokens = 0
        self._output_tokens = 0
        self._stop_reason: str | None = None

    def _event(self, event_type: str, data: dict) -> str:
        return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"

    def format_error(self, error_line: str) -> str:
        return f"event: error\ndata: {error_line}\n\n"

    def feed(self, line: str) -> str:
        line = line.strip()
        if not line:
            return ""

        output = ""

        if not line.startswith("data: "):
            return ""

        payload = line[6:].strip()
        if payload == "[DONE]":
            if self._block_started:
                output += self._event("content_block_stop", {
                    "type": "content_block_stop",
                    "index": self._block_index,
                })
            stop = self._stop_reason or "end_turn"
            output += self._event("message_delta", {
                "type": "message_delta",
                "delta": {"stop_reason": stop, "stop_sequence": None},
                "usage": {
                    "input_tokens": self._input_tokens,
                    "output_tokens": self._output_tokens,
                },
            })
            output += self._event("message_stop", {"type": "message_stop"})
            return output

        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            return ""

        choice = (data.get("choices") or [{}])[0]
        delta = choice.get("delta", {})
        finish_reason = choice.get("finish_reason")

        chunk_usage = data.get("usage") or {}
        if chunk_usage.get("prompt_tokens"):
            self._input_tokens = chunk_usage["prompt_tokens"]
        if chunk_usage.get("completion_tokens"):
            self._output_tokens = chunk_usage["completion_tokens"]

        if not self._started:
            self._started = True
            output += self._event("message_start", {
                "type": "message_start",
                "message": {
                    "id": f"msg_{uuid.uuid4().hex[:24]}",
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                    "model": self._model,
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {
                        "input_tokens": self._input_tokens,
                        "output_tokens": 0,
                    },
                },
            })

        text = delta.get("content")
        if text is not None:
            if not self._block_started:
                self._block_started = True
                output += self._event("content_block_start", {
                    "type": "content_block_start",
                    "index": self._block_index,
                    "content_block": {"type": "text", "text": ""},
                })
            output += self._event("content_block_delta", {
                "type": "content_block_delta",
                "index": self._block_index,
                "delta": {"type": "text_delta", "text": text},
            })

        tool_calls = delta.get("tool_calls")
        if tool_calls:
            for tc in tool_calls:
                tc_index = tc.get("index", 0)
                func = tc.get("function", {})

                if tc_index not in self._tool_blocks:
                    if self._block_started:
                        output += self._event("content_block_stop", {
                            "type": "content_block_stop",
                            "index": self._block_index,
                        })
                    self._block_index += 1 if self._block_started else 0
                    self._block_started = True

                    tool_id = tc.get("id", f"toolu_{uuid.uuid4().hex[:24]}")
                    tool_name = func.get("name", "")
                    self._tool_blocks[tc_index] = {
                        "id": tool_id,
                        "name": tool_name,
                    }
                    output += self._event("content_block_start", {
                        "type": "content_block_start",
                        "index": self._block_index,
                        "content_block": {
                            "type": "tool_use",
                            "id": tool_id,
                            "name": tool_name,
                            "input": {},
                        },
                    })

                args_chunk = func.get("arguments", "")
                if args_chunk:
                    output += self._event("content_block_delta", {
                        "type": "content_block_delta",
                        "index": self._block_index,
                        "delta": {
                            "type": "input_json_delta",
                            "partial_json": args_chunk,
                        },
                    })

        if finish_reason:
            if self._block_started:
                output += self._event("content_block_stop", {
                    "type": "content_block_stop",
                    "index": self._block_index,
                })
                self._block_started = False
            self._stop_reason = _STOP_REASON_MAP.get(finish_reason, "end_turn")

        return output


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

class AnthropicAdapter(FormatAdapter):
    """Translates between Anthropic Messages API and OpenAI chat/completions."""

    def convert_chat_request(self, body: dict) -> dict:
        return _anthropic_to_openai(body)

    def convert_chat_response(self, openai_resp: dict, original_body: dict) -> dict:
        model = original_body.get("model", "")
        return _openai_to_anthropic(openai_resp, model)

    def create_stream_converter(self, original_body: dict) -> StreamConverter:
        model = original_body.get("model", "")
        return _AnthropicStreamConverter(model)

    def is_streaming(self, body: dict) -> bool:
        return bool(body.get("stream"))

    def convert_models_response(self, openai_resp: dict) -> dict:
        return openai_resp

    def infer_initiator(self, body: dict) -> str:
        messages = body.get("messages", [])
        if not messages:
            return "user"

        # Only the last message is a reliable signal. If it carries a
        # tool_result block, the client is auto-continuing after a tool
        # call (agent). Prior tool_use in history is NOT a reliable signal
        # — in Claude Code and similar chat UIs the full conversation is
        # re-sent every turn, so any human follow-up after the first tool
        # use would be misclassified.
        last = messages[-1]
        content = last.get("content")
        if isinstance(content, list):
            for block in content:
                if block.get("type") == "tool_result":
                    logger.debug("infer_initiator: tool_result in last message -> agent")
                    return "agent"

        logger.debug(
            "infer_initiator: no tool_result in last message, %d messages, "
            "last role=%s -> user",
            len(messages),
            last.get("role"),
        )
        return "user"

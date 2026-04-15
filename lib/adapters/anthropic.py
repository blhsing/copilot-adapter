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

    # Anthropic built-in tool types — converted to OpenAI function tools.
    # These are passed through as regular function tools for the client to
    # handle (web_search, text_editor, code_execution).
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

    if "tools" in body:
        func_tools = []
        for t in body["tools"]:
            tool_type = t.get("type", "")
            # Check for Anthropic built-in tools by type prefix
            builtin_key = None
            for prefix in _BUILTIN_TOOL_SCHEMAS:
                if tool_type.startswith(prefix):
                    builtin_key = prefix
                    break
            if builtin_key:
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
                # Unsupported Anthropic built-in tool type — strip it
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
        if func_tools:
            result["tools"] = func_tools

    tc = body.get("tool_choice")
    if tc:
        tc_type = tc.get("type") if isinstance(tc, dict) else tc
        if tc_type == "tool":
            forced_name = tc.get("name", "")
            has_tool = any(
                ft["function"]["name"] == forced_name
                for ft in result.get("tools", [])
            )
            if has_tool:
                result["tool_choice"] = {
                    "type": "function",
                    "function": {"name": forced_name},
                }
            # else: tool not found, omit tool_choice
        elif tc_type == "auto":
            result["tool_choice"] = "auto"
        elif tc_type == "any":
            result["tool_choice"] = "required"
        elif tc_type == "none":
            result["tool_choice"] = "none"

    return result


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
# Stream converter
# ---------------------------------------------------------------------------

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

        # Check if the last message directly contains tool_result blocks.
        last = messages[-1]
        content = last.get("content")
        if isinstance(content, list):
            for block in content:
                if block.get("type") == "tool_result":
                    logger.debug("infer_initiator: tool_result in last message -> agent")
                    return "agent"

        # Check conversation history for prior tool interactions.  If the
        # model has already used tools in this conversation, a subsequent
        # user text message is almost certainly an agentic follow-up (e.g.
        # subagent spawn, auto-continue, post-compact).
        for msg in messages:
            if msg.get("role") != "assistant":
                continue
            c = msg.get("content")
            if isinstance(c, list):
                for block in c:
                    if block.get("type") == "tool_use":
                        logger.debug(
                            "infer_initiator: prior tool_use found in history, "
                            "last message role=%s content_type=%s -> agent",
                            last.get("role"),
                            type(content).__name__,
                        )
                        return "agent"

        logger.debug(
            "infer_initiator: no tool activity detected, %d messages, "
            "last role=%s -> user",
            len(messages),
            last.get("role"),
        )
        return "user"

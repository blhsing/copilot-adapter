"""Anthropic Messages API format adapter."""

import json
import uuid

from .base import FormatAdapter, StreamConverter


# ---------------------------------------------------------------------------
# Request conversion: Anthropic -> OpenAI
# ---------------------------------------------------------------------------

def _convert_content_blocks(blocks: list, role: str) -> list[dict]:
    messages = []
    text_parts = []
    openai_content_parts = []
    has_multimodal = False

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
            if text_parts:
                messages.append({"role": role, "content": "\n".join(text_parts)})
                text_parts = []
                openai_content_parts = []
            messages.append({
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": block["id"],
                    "type": "function",
                    "function": {
                        "name": block["name"],
                        "arguments": json.dumps(block["input"]),
                    },
                }],
            })

        elif btype == "tool_result":
            if text_parts:
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
                "tool_call_id": block["tool_use_id"],
                "content": str(tool_content),
            })

    if openai_content_parts:
        if has_multimodal:
            messages.append({"role": role, "content": openai_content_parts})
        elif text_parts:
            messages.append({"role": role, "content": "\n".join(text_parts)})

    return messages


def _anthropic_to_openai(body: dict) -> dict:
    messages = []

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
            messages.extend(_convert_content_blocks(content, role))
        else:
            messages.append({"role": role, "content": content})

    result = {
        "model": body["model"],
        "messages": messages,
        "stream": body.get("stream", False),
    }

    if "max_tokens" in body:
        result["max_tokens"] = body["max_tokens"]
    if "temperature" in body:
        result["temperature"] = body["temperature"]
    if "top_p" in body:
        result["top_p"] = body["top_p"]
    if "stop_sequences" in body:
        result["stop"] = body["stop_sequences"]

    if "tools" in body:
        result["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "parameters": t.get("input_schema", {}),
                },
            }
            for t in body["tools"]
        ]

    tc = body.get("tool_choice")
    if tc:
        tc_type = tc.get("type") if isinstance(tc, dict) else tc
        if tc_type == "auto":
            result["tool_choice"] = "auto"
        elif tc_type == "any":
            result["tool_choice"] = "required"
        elif tc_type == "none":
            result["tool_choice"] = "none"
        elif tc_type == "tool":
            result["tool_choice"] = {
                "type": "function",
                "function": {"name": tc["name"]},
            }

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
        content.append({
            "type": "tool_use",
            "id": tc.get("id", f"toolu_{uuid.uuid4().hex[:24]}"),
            "name": func.get("name", ""),
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
            output += self._event("message_delta", {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                "usage": {"output_tokens": self._output_tokens},
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

        chunk_usage = data.get("usage", {})
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

            output += self._event("message_delta", {
                "type": "message_delta",
                "delta": {
                    "stop_reason": _STOP_REASON_MAP.get(finish_reason, "end_turn"),
                    "stop_sequence": None,
                },
                "usage": {"output_tokens": self._output_tokens},
            })
            output += self._event("message_stop", {"type": "message_stop"})

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

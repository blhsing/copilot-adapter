"""Translate between Anthropic Messages API and OpenAI Chat Completions format."""

import json
import time
import uuid


def anthropic_to_openai(body: dict) -> dict:
    """Convert an Anthropic /v1/messages request to OpenAI /chat/completions format."""
    messages = []

    # Anthropic uses a top-level 'system' field instead of a system message
    system = body.get("system")
    if system:
        if isinstance(system, list):
            # Array of text blocks
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
            # Process content blocks
            converted = _convert_content_blocks(content, role)
            messages.extend(converted)
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

    # Convert tools
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

    # Convert tool_choice
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


def _convert_content_blocks(blocks: list, role: str) -> list[dict]:
    """Convert Anthropic content blocks to OpenAI messages."""
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
            # Assistant message with tool call - flush text first
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
            # Tool result becomes a tool message
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

    # Flush remaining content
    if openai_content_parts:
        if has_multimodal:
            messages.append({"role": role, "content": openai_content_parts})
        elif text_parts:
            messages.append({"role": role, "content": "\n".join(text_parts)})

    return messages


def openai_to_anthropic(openai_resp: dict, model: str) -> dict:
    """Convert an OpenAI chat completion response to Anthropic Messages format."""
    choice = openai_resp.get("choices", [{}])[0]
    message = choice.get("message", {})

    content = []

    # Text content
    text = message.get("content")
    if text:
        content.append({"type": "text", "text": text})

    # Tool calls
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

    # Map finish_reason to stop_reason
    finish = choice.get("finish_reason", "stop")
    stop_reason_map = {
        "stop": "end_turn",
        "length": "max_tokens",
        "tool_calls": "tool_use",
        "content_filter": "end_turn",
    }
    stop_reason = stop_reason_map.get(finish, "end_turn")

    usage = openai_resp.get("usage", {})

    return {
        "id": f"msg_{uuid.uuid4().hex[:24]}",
        "type": "message",
        "role": "assistant",
        "content": content,
        "model": model,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
        },
    }


def openai_stream_to_anthropic_events(model: str):
    """Return a stateful converter that takes OpenAI SSE lines and yields Anthropic SSE events."""

    class Converter:
        def __init__(self):
            self.started = False
            self.block_started = False
            self.block_index = 0
            self.tool_blocks: dict[int, dict] = {}  # index -> {id, name, args_buffer}
            self.input_tokens = 0
            self.output_tokens = 0

        def _event(self, event_type: str, data: dict) -> str:
            return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"

        def convert(self, line: str) -> str:
            output = ""

            if not line.startswith("data: "):
                return ""

            payload = line[6:].strip()
            if payload == "[DONE]":
                # Close any open content block
                if self.block_started:
                    output += self._event("content_block_stop", {
                        "type": "content_block_stop",
                        "index": self.block_index,
                    })
                output += self._event("message_delta", {
                    "type": "message_delta",
                    "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                    "usage": {"output_tokens": self.output_tokens},
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

            # Usage from chunk
            chunk_usage = data.get("usage", {})
            if chunk_usage.get("prompt_tokens"):
                self.input_tokens = chunk_usage["prompt_tokens"]
            if chunk_usage.get("completion_tokens"):
                self.output_tokens = chunk_usage["completion_tokens"]

            # Emit message_start on first chunk
            if not self.started:
                self.started = True
                output += self._event("message_start", {
                    "type": "message_start",
                    "message": {
                        "id": f"msg_{uuid.uuid4().hex[:24]}",
                        "type": "message",
                        "role": "assistant",
                        "content": [],
                        "model": model,
                        "stop_reason": None,
                        "stop_sequence": None,
                        "usage": {
                            "input_tokens": self.input_tokens,
                            "output_tokens": 0,
                        },
                    },
                })

            # Text content
            text = delta.get("content")
            if text is not None:
                if not self.block_started:
                    self.block_started = True
                    output += self._event("content_block_start", {
                        "type": "content_block_start",
                        "index": self.block_index,
                        "content_block": {"type": "text", "text": ""},
                    })
                output += self._event("content_block_delta", {
                    "type": "content_block_delta",
                    "index": self.block_index,
                    "delta": {"type": "text_delta", "text": text},
                })

            # Tool calls
            tool_calls = delta.get("tool_calls")
            if tool_calls:
                for tc in tool_calls:
                    tc_index = tc.get("index", 0)
                    func = tc.get("function", {})

                    if tc_index not in self.tool_blocks:
                        # Close previous block
                        if self.block_started:
                            output += self._event("content_block_stop", {
                                "type": "content_block_stop",
                                "index": self.block_index,
                            })
                        self.block_index += 1 if self.block_started else 0
                        self.block_started = True

                        tool_id = tc.get("id", f"toolu_{uuid.uuid4().hex[:24]}")
                        tool_name = func.get("name", "")
                        self.tool_blocks[tc_index] = {
                            "id": tool_id,
                            "name": tool_name,
                        }
                        output += self._event("content_block_start", {
                            "type": "content_block_start",
                            "index": self.block_index,
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
                            "index": self.block_index,
                            "delta": {
                                "type": "input_json_delta",
                                "partial_json": args_chunk,
                            },
                        })

            # Handle finish
            if finish_reason:
                if self.block_started:
                    output += self._event("content_block_stop", {
                        "type": "content_block_stop",
                        "index": self.block_index,
                    })
                    self.block_started = False

                stop_reason_map = {
                    "stop": "end_turn",
                    "length": "max_tokens",
                    "tool_calls": "tool_use",
                }
                output += self._event("message_delta", {
                    "type": "message_delta",
                    "delta": {
                        "stop_reason": stop_reason_map.get(finish_reason, "end_turn"),
                        "stop_sequence": None,
                    },
                    "usage": {"output_tokens": self.output_tokens},
                })
                output += self._event("message_stop", {"type": "message_stop"})

            return output

    return Converter()

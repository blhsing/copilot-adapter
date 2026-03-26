"""Translate between Google Gemini API and OpenAI Chat Completions format."""

import json
import uuid


def gemini_to_openai(body: dict, model: str) -> dict:
    """Convert a Gemini generateContent request to OpenAI chat/completions format."""
    messages = []

    # systemInstruction -> system message
    sys_inst = body.get("systemInstruction")
    if sys_inst:
        parts = sys_inst.get("parts", [])
        text = "\n".join(p.get("text", "") for p in parts if "text" in p)
        if text:
            messages.append({"role": "system", "content": text})

    # contents -> messages
    for turn in body.get("contents", []):
        role = turn.get("role", "user")
        openai_role = "assistant" if role == "model" else role
        parts = turn.get("parts", [])
        converted = _convert_parts_to_openai(parts, openai_role)
        messages.extend(converted)

    result = {"model": model, "messages": messages}

    # generationConfig -> top-level params
    gc = body.get("generationConfig", {})
    if "temperature" in gc:
        result["temperature"] = gc["temperature"]
    if "topP" in gc:
        result["top_p"] = gc["topP"]
    if "maxOutputTokens" in gc:
        result["max_tokens"] = gc["maxOutputTokens"]
    if "stopSequences" in gc:
        result["stop"] = gc["stopSequences"]
    if "candidateCount" in gc:
        result["n"] = gc["candidateCount"]
    if "presencePenalty" in gc:
        result["presence_penalty"] = gc["presencePenalty"]
    if "frequencyPenalty" in gc:
        result["frequency_penalty"] = gc["frequencyPenalty"]

    # tools -> OpenAI tools
    tools = body.get("tools", [])
    openai_tools = []
    for tool_group in tools:
        for func_decl in tool_group.get("functionDeclarations", []):
            openai_tools.append({
                "type": "function",
                "function": {
                    "name": func_decl["name"],
                    "description": func_decl.get("description", ""),
                    "parameters": func_decl.get("parameters", {}),
                },
            })
    if openai_tools:
        result["tools"] = openai_tools

    # toolConfig -> tool_choice
    tc = body.get("toolConfig", {}).get("functionCallingConfig", {})
    mode = tc.get("mode")
    if mode == "NONE":
        result["tool_choice"] = "none"
    elif mode == "ANY":
        allowed = tc.get("allowedFunctionNames", [])
        if len(allowed) == 1:
            result["tool_choice"] = {
                "type": "function",
                "function": {"name": allowed[0]},
            }
        else:
            result["tool_choice"] = "required"
    elif mode == "AUTO":
        result["tool_choice"] = "auto"

    return result


def _convert_parts_to_openai(parts: list, role: str) -> list[dict]:
    """Convert Gemini parts to OpenAI message(s)."""
    messages = []
    text_parts = []
    content_parts = []
    has_multimodal = False

    for part in parts:
        if "text" in part:
            text_parts.append(part["text"])
            content_parts.append({"type": "text", "text": part["text"]})

        elif "inlineData" in part:
            has_multimodal = True
            data = part["inlineData"]
            url = f"data:{data['mimeType']};base64,{data['data']}"
            content_parts.append({
                "type": "image_url",
                "image_url": {"url": url},
            })

        elif "functionCall" in part:
            # Flush text first
            if text_parts:
                messages.append({"role": role, "content": "\n".join(text_parts)})
                text_parts = []
                content_parts = []
            fc = part["functionCall"]
            messages.append({
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": fc.get("id", f"call_{uuid.uuid4().hex[:24]}"),
                    "type": "function",
                    "function": {
                        "name": fc["name"],
                        "arguments": json.dumps(fc.get("args", {})),
                    },
                }],
            })

        elif "functionResponse" in part:
            if text_parts:
                messages.append({"role": role, "content": "\n".join(text_parts)})
                text_parts = []
                content_parts = []
            fr = part["functionResponse"]
            messages.append({
                "role": "tool",
                "tool_call_id": fr.get("id", f"call_{uuid.uuid4().hex[:24]}"),
                "content": json.dumps(fr.get("response", {})),
            })

    # Flush remaining
    if content_parts:
        if has_multimodal:
            messages.append({"role": role, "content": content_parts})
        elif text_parts:
            messages.append({"role": role, "content": "\n".join(text_parts)})

    return messages


def openai_to_gemini(openai_resp: dict, model: str) -> dict:
    """Convert an OpenAI chat completion response to Gemini generateContent format."""
    candidates = []

    for choice in openai_resp.get("choices", []):
        message = choice.get("message", {})
        parts = []

        # Text content
        text = message.get("content")
        if text:
            parts.append({"text": text})

        # Tool calls -> functionCall parts
        for tc in message.get("tool_calls", []):
            func = tc.get("function", {})
            try:
                args = json.loads(func.get("arguments", "{}"))
            except json.JSONDecodeError:
                args = {}
            parts.append({
                "functionCall": {
                    "id": tc.get("id", ""),
                    "name": func.get("name", ""),
                    "args": args,
                },
            })

        if not parts:
            parts.append({"text": ""})

        # Map finish reason
        finish = choice.get("finish_reason", "stop")
        finish_map = {
            "stop": "STOP",
            "length": "MAX_TOKENS",
            "content_filter": "SAFETY",
            "tool_calls": "STOP",
        }

        candidates.append({
            "content": {"role": "model", "parts": parts},
            "finishReason": finish_map.get(finish, "STOP"),
        })

    usage = openai_resp.get("usage", {})

    return {
        "candidates": candidates,
        "usageMetadata": {
            "promptTokenCount": usage.get("prompt_tokens", 0),
            "candidatesTokenCount": usage.get("completion_tokens", 0),
            "totalTokenCount": usage.get("total_tokens", 0),
        },
        "modelVersion": model,
    }


def openai_stream_to_gemini_events(model: str):
    """Return a stateful converter that takes OpenAI SSE lines and yields Gemini SSE lines."""

    class Converter:
        def __init__(self):
            self.prompt_tokens = 0
            self.completion_tokens = 0

        def convert(self, line: str) -> str:
            if not line.startswith("data: "):
                return ""

            payload = line[6:].strip()
            if payload == "[DONE]":
                return ""

            try:
                data = json.loads(payload)
            except json.JSONDecodeError:
                return ""

            choice = (data.get("choices") or [{}])[0]
            delta = choice.get("delta", {})
            finish_reason = choice.get("finish_reason")

            # Track usage
            chunk_usage = data.get("usage", {})
            if chunk_usage.get("prompt_tokens"):
                self.prompt_tokens = chunk_usage["prompt_tokens"]
            if chunk_usage.get("completion_tokens"):
                self.completion_tokens = chunk_usage["completion_tokens"]

            parts = []
            text = delta.get("content")
            if text is not None:
                parts.append({"text": text})

            tool_calls = delta.get("tool_calls")
            if tool_calls:
                for tc in tool_calls:
                    func = tc.get("function", {})
                    name = func.get("name")
                    args_str = func.get("arguments", "")
                    if name:
                        try:
                            args = json.loads(args_str) if args_str else {}
                        except json.JSONDecodeError:
                            args = {}
                        parts.append({
                            "functionCall": {
                                "id": tc.get("id", ""),
                                "name": name,
                                "args": args,
                            },
                        })

            if not parts and not finish_reason:
                return ""

            candidate = {}
            if parts:
                candidate["content"] = {"role": "model", "parts": parts}

            if finish_reason:
                finish_map = {
                    "stop": "STOP",
                    "length": "MAX_TOKENS",
                    "content_filter": "SAFETY",
                    "tool_calls": "STOP",
                }
                candidate["finishReason"] = finish_map.get(finish_reason, "STOP")

            gemini_resp = {"candidates": [candidate]}

            if finish_reason:
                gemini_resp["usageMetadata"] = {
                    "promptTokenCount": self.prompt_tokens,
                    "candidatesTokenCount": self.completion_tokens,
                    "totalTokenCount": self.prompt_tokens + self.completion_tokens,
                }
                gemini_resp["modelVersion"] = model

            return f"data: {json.dumps(gemini_resp)}\n\n"

    return Converter()


def openai_models_to_gemini(openai_resp: dict) -> dict:
    """Convert OpenAI model list to Gemini model list format."""
    models = []
    for m in openai_resp.get("data", []):
        mid = m.get("id", "")
        caps = m.get("capabilities", {})
        limits = caps.get("limits", {})
        models.append({
            "name": f"models/{mid}",
            "baseModelId": mid,
            "version": m.get("version", ""),
            "displayName": m.get("name", mid),
            "description": f"{m.get('vendor', '')} {m.get('name', mid)}".strip(),
            "inputTokenLimit": limits.get("max_prompt_tokens", 0),
            "outputTokenLimit": limits.get("max_output_tokens", 0),
            "supportedGenerationMethods": ["generateContent", "countTokens"],
            "temperature": 1.0,
            "topP": 0.95,
            "topK": 40,
        })
    return {"models": models}


def openai_model_to_gemini(openai_model: dict) -> dict:
    """Convert a single OpenAI model object to Gemini model format."""
    mid = openai_model.get("id", "")
    caps = openai_model.get("capabilities", {})
    limits = caps.get("limits", {})
    return {
        "name": f"models/{mid}",
        "baseModelId": mid,
        "version": openai_model.get("version", ""),
        "displayName": openai_model.get("name", mid),
        "description": f"{openai_model.get('vendor', '')} {openai_model.get('name', mid)}".strip(),
        "inputTokenLimit": limits.get("max_prompt_tokens", 0),
        "outputTokenLimit": limits.get("max_output_tokens", 0),
        "supportedGenerationMethods": ["generateContent", "countTokens"],
        "temperature": 1.0,
        "topP": 0.95,
        "topK": 40,
    }

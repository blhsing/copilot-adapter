"""Integration tests for format adapters against the live Copilot API.

Each test converts a request through an adapter, sends it to the real API,
and verifies the response converts back correctly.
"""

import json

import pytest

from lib.adapters import AnthropicAdapter, GeminiAdapter, OpenAIAdapter
from lib.client import CopilotClient
from lib.server import _normalize_request_params


# ===================================================================
# OpenAI adapter
# ===================================================================

class TestOpenAIAdapter:
    @pytest.mark.asyncio
    async def test_passthrough_non_streaming(self, client: CopilotClient, available_model: str):
        adapter = OpenAIAdapter()
        body = {
            "model": available_model,
            "messages": [{"role": "user", "content": "Say yes."}],
            "max_tokens": 5,
        }
        openai_body = adapter.convert_chat_request(body)
        assert openai_body is body  # true passthrough

        resp = await client.chat_completions(openai_body)
        result = adapter.convert_chat_response(resp.json(), body)
        assert "choices" in result
        assert result["choices"][0]["message"]["content"]

    @pytest.mark.asyncio
    async def test_passthrough_streaming(self, client: CopilotClient, available_model: str):
        adapter = OpenAIAdapter()
        body = {
            "model": available_model,
            "messages": [{"role": "user", "content": "Say ok."}],
            "max_tokens": 5,
            "stream": True,
        }
        assert adapter.is_streaming(body) is True

        converter = adapter.create_stream_converter(body)
        collected = ""
        async for line in client.stream_chat_completions(body):
            result = converter.feed(line)
            if result:
                collected += result
        assert "data:" in collected

    @pytest.mark.asyncio
    async def test_models_passthrough(self, client: CopilotClient):
        adapter = OpenAIAdapter()
        resp = await client.list_models()
        result = adapter.convert_models_response(resp.json())
        assert "data" in result


# ===================================================================
# Anthropic adapter
# ===================================================================

class TestAnthropicAdapter:
    @pytest.mark.asyncio
    async def test_simple_message(self, client: CopilotClient, available_model: str):
        adapter = AnthropicAdapter()
        anthropic_body = {
            "model": available_model,
            "max_tokens": 20,
            "messages": [{"role": "user", "content": "Reply with only 'pong'."}],
        }
        openai_body = adapter.convert_chat_request(anthropic_body)

        assert openai_body["model"] == available_model
        assert openai_body["messages"][0]["role"] == "user"

        resp = await client.chat_completions(openai_body)
        assert resp.status_code == 200

        result = adapter.convert_chat_response(resp.json(), anthropic_body)
        assert result["type"] == "message"
        assert result["role"] == "assistant"
        assert len(result["content"]) > 0
        assert result["content"][0]["type"] == "text"
        assert result["content"][0]["text"]
        assert result["stop_reason"] in ("end_turn", "max_tokens", "tool_use")
        assert "usage" in result
        assert result["usage"]["input_tokens"] > 0

    @pytest.mark.asyncio
    async def test_system_message(self, client: CopilotClient, available_model: str):
        adapter = AnthropicAdapter()
        anthropic_body = {
            "model": available_model,
            "max_tokens": 10,
            "system": "You only respond with the word 'banana'.",
            "messages": [{"role": "user", "content": "Say something."}],
        }
        openai_body = adapter.convert_chat_request(anthropic_body)
        assert openai_body["messages"][0]["role"] == "system"
        assert "banana" in openai_body["messages"][0]["content"]

        resp = await client.chat_completions(openai_body)
        result = adapter.convert_chat_response(resp.json(), anthropic_body)
        assert "banana" in result["content"][0]["text"].lower()

    @pytest.mark.asyncio
    async def test_system_as_block_array(self, client: CopilotClient, available_model: str):
        adapter = AnthropicAdapter()
        anthropic_body = {
            "model": available_model,
            "max_tokens": 10,
            "system": [{"type": "text", "text": "Reply only with 'OK'."}],
            "messages": [{"role": "user", "content": "Go."}],
        }
        openai_body = adapter.convert_chat_request(anthropic_body)
        assert openai_body["messages"][0]["role"] == "system"

        resp = await client.chat_completions(openai_body)
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_streaming(self, client: CopilotClient, available_model: str):
        adapter = AnthropicAdapter()
        anthropic_body = {
            "model": available_model,
            "max_tokens": 30,
            "stream": True,
            "messages": [{"role": "user", "content": "Count to 3."}],
        }
        assert adapter.is_streaming(anthropic_body) is True

        openai_body = adapter.convert_chat_request(anthropic_body)
        converter = adapter.create_stream_converter(anthropic_body)

        events = []
        async for line in client.stream_chat_completions(openai_body):
            result = converter.feed(line)
            if result:
                events.append(result)

        full_output = "".join(events)

        # Should contain Anthropic SSE event types
        assert "event: message_start" in full_output
        assert "event: content_block_start" in full_output
        assert "event: content_block_delta" in full_output
        assert "event: message_stop" in full_output

    @pytest.mark.asyncio
    async def test_stream_text_deltas_accumulate(self, client: CopilotClient, available_model: str):
        adapter = AnthropicAdapter()
        anthropic_body = {
            "model": available_model,
            "max_tokens": 20,
            "stream": True,
            "messages": [{"role": "user", "content": "Say 'hello world'."}],
        }
        openai_body = adapter.convert_chat_request(anthropic_body)
        converter = adapter.create_stream_converter(anthropic_body)

        text_deltas = []
        async for line in client.stream_chat_completions(openai_body):
            result = converter.feed(line)
            if not result:
                continue
            for event_line in result.split("\n"):
                if event_line.startswith("data: "):
                    try:
                        payload = json.loads(event_line[6:])
                        delta = payload.get("delta", {})
                        if delta.get("type") == "text_delta":
                            text_deltas.append(delta["text"])
                    except json.JSONDecodeError:
                        pass

        full_text = "".join(text_deltas)
        assert len(full_text) > 0

    @pytest.mark.asyncio
    async def test_multi_turn_conversion(self, client: CopilotClient, available_model: str):
        adapter = AnthropicAdapter()
        anthropic_body = {
            "model": available_model,
            "max_tokens": 20,
            "messages": [
                {"role": "user", "content": "Remember the code word: ALPHA."},
                {"role": "assistant", "content": "I'll remember the code word ALPHA."},
                {"role": "user", "content": "What is the code word? Reply with just the word."},
            ],
        }
        openai_body = adapter.convert_chat_request(anthropic_body)
        assert len(openai_body["messages"]) == 3

        resp = await client.chat_completions(openai_body)
        result = adapter.convert_chat_response(resp.json(), anthropic_body)
        assert "ALPHA" in result["content"][0]["text"].upper()

    @pytest.mark.asyncio
    async def test_stop_sequences(self, client: CopilotClient, available_model: str):
        adapter = AnthropicAdapter()
        anthropic_body = {
            "model": available_model,
            "max_tokens": 50,
            "stop_sequences": ["."],
            "messages": [{"role": "user", "content": "Write a sentence about cats."}],
        }
        openai_body = adapter.convert_chat_request(anthropic_body)
        assert openai_body["stop"] == ["."]

        resp = await client.chat_completions(openai_body)
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_temperature(self, client: CopilotClient, available_model: str):
        adapter = AnthropicAdapter()
        anthropic_body = {
            "model": available_model,
            "max_tokens": 10,
            "temperature": 0.0,
            "messages": [{"role": "user", "content": "What is 1+1? Answer with just the number."}],
        }
        openai_body = adapter.convert_chat_request(anthropic_body)
        assert openai_body["temperature"] == 0.0

        resp = await client.chat_completions(openai_body)
        result = adapter.convert_chat_response(resp.json(), anthropic_body)
        assert "2" in result["content"][0]["text"]

    def test_preserve_thinking_for_later_normalization(self):
        adapter = AnthropicAdapter()
        anthropic_body = {
            "model": "claude-opus-4-6",
            "max_tokens": 10,
            "thinking": {"type": "enabled", "budget_tokens": 32000},
            "messages": [{"role": "user", "content": "Hi."}],
        }
        openai_body = adapter.convert_chat_request(anthropic_body)
        assert openai_body["_copilot_adapter_thinking"] == {
            "type": "enabled",
            "budget_tokens": 32000,
        }

    def test_normalize_anthropic_thinking_to_openai_reasoning(self):
        openai_body = {
            "model": "gpt-5.4",
            "messages": [{"role": "user", "content": "Hi."}],
            "max_tokens": 10,
            "_copilot_adapter_thinking": {"type": "enabled", "budget_tokens": 32000},
        }
        normalized = _normalize_request_params(openai_body, "anthropic", "gpt-5.4")
        assert normalized["max_completion_tokens"] == 10
        assert "max_tokens" not in normalized
        assert normalized["reasoning_effort"] == "xhigh"
        assert "_copilot_adapter_thinking" not in normalized

    @pytest.mark.asyncio
    async def test_stream_error_formatting(self):
        adapter = AnthropicAdapter()
        converter = adapter.create_stream_converter({"model": "test"})
        err = converter.format_error("error: 429 rate limited")
        assert err.startswith("event: error\n")
        assert "429" in err


# ===================================================================
# Gemini adapter
# ===================================================================

class TestGeminiAdapter:
    @pytest.mark.asyncio
    async def test_simple_generate_content(self, client: CopilotClient, available_model: str):
        adapter = GeminiAdapter(available_model)
        gemini_body = {
            "contents": [
                {"role": "user", "parts": [{"text": "Reply with only 'pong'."}]}
            ],
        }
        openai_body = adapter.convert_chat_request(gemini_body)

        assert openai_body["model"] == available_model
        assert openai_body["messages"][0]["content"] == "Reply with only 'pong'."

        resp = await client.chat_completions(openai_body)
        assert resp.status_code == 200

        result = adapter.convert_chat_response(resp.json(), gemini_body)
        assert "candidates" in result
        assert len(result["candidates"]) > 0
        candidate = result["candidates"][0]
        assert candidate["content"]["role"] == "model"
        assert len(candidate["content"]["parts"]) > 0
        assert candidate["content"]["parts"][0]["text"]
        assert candidate["finishReason"] in ("STOP", "MAX_TOKENS", "SAFETY")

    @pytest.mark.asyncio
    async def test_system_instruction(self, client: CopilotClient, available_model: str):
        adapter = GeminiAdapter(available_model)
        gemini_body = {
            "systemInstruction": {
                "parts": [{"text": "You only respond with the word 'mango'."}]
            },
            "contents": [
                {"role": "user", "parts": [{"text": "Say something."}]}
            ],
        }
        openai_body = adapter.convert_chat_request(gemini_body)
        assert openai_body["messages"][0]["role"] == "system"
        assert "mango" in openai_body["messages"][0]["content"]

        resp = await client.chat_completions(openai_body)
        result = adapter.convert_chat_response(resp.json(), gemini_body)
        assert "mango" in result["candidates"][0]["content"]["parts"][0]["text"].lower()

    @pytest.mark.asyncio
    async def test_generation_config(self, client: CopilotClient, available_model: str):
        adapter = GeminiAdapter(available_model)
        gemini_body = {
            "contents": [
                {"role": "user", "parts": [{"text": "What is 2+2? Just the number."}]}
            ],
            "generationConfig": {
                "temperature": 0.0,
                "maxOutputTokens": 10,
                "topP": 0.9,
            },
        }
        openai_body = adapter.convert_chat_request(gemini_body)
        assert openai_body["temperature"] == 0.0
        assert openai_body["max_tokens"] == 10
        assert openai_body["top_p"] == 0.9

        resp = await client.chat_completions(openai_body)
        result = adapter.convert_chat_response(resp.json(), gemini_body)
        assert "4" in result["candidates"][0]["content"]["parts"][0]["text"]

    @pytest.mark.asyncio
    async def test_streaming(self, client: CopilotClient, available_model: str):
        adapter = GeminiAdapter(available_model)
        gemini_body = {
            "contents": [
                {"role": "user", "parts": [{"text": "Count to 3."}]}
            ],
        }
        openai_body = adapter.convert_chat_request(gemini_body)
        openai_body["stream"] = True
        openai_body["max_tokens"] = 30

        converter = adapter.create_stream_converter(gemini_body)

        chunks = []
        async for line in client.stream_chat_completions(openai_body):
            result = converter.feed(line)
            if result:
                chunks.append(result)

        assert len(chunks) > 0

        # Each chunk should be a valid Gemini SSE line
        for chunk in chunks:
            assert chunk.startswith("data: ")
            payload = json.loads(chunk[6:].strip())
            assert "candidates" in payload

    @pytest.mark.asyncio
    async def test_stream_final_chunk_has_usage(self, client: CopilotClient, available_model: str):
        adapter = GeminiAdapter(available_model)
        gemini_body = {
            "contents": [
                {"role": "user", "parts": [{"text": "Say hi."}]}
            ],
        }
        openai_body = adapter.convert_chat_request(gemini_body)
        openai_body["stream"] = True
        openai_body["max_tokens"] = 10

        converter = adapter.create_stream_converter(gemini_body)

        last_chunk = None
        async for line in client.stream_chat_completions(openai_body):
            result = converter.feed(line)
            if result:
                last_chunk = result

        assert last_chunk is not None
        payload = json.loads(last_chunk[6:].strip())
        # Final chunk should have finish reason and usage metadata
        assert payload["candidates"][0].get("finishReason")
        assert "usageMetadata" in payload

    @pytest.mark.asyncio
    async def test_multi_turn_conversation(self, client: CopilotClient, available_model: str):
        adapter = GeminiAdapter(available_model)
        gemini_body = {
            "contents": [
                {"role": "user", "parts": [{"text": "My secret number is 42."}]},
                {"role": "model", "parts": [{"text": "Got it, your secret number is 42."}]},
                {"role": "user", "parts": [{"text": "What is my secret number? Just the number."}]},
            ],
            "generationConfig": {"maxOutputTokens": 10, "temperature": 0.0},
        }
        openai_body = adapter.convert_chat_request(gemini_body)
        # model -> assistant role mapping
        assert any(m["role"] == "assistant" for m in openai_body["messages"])

        resp = await client.chat_completions(openai_body)
        result = adapter.convert_chat_response(resp.json(), gemini_body)
        assert "42" in result["candidates"][0]["content"]["parts"][0]["text"]

    @pytest.mark.asyncio
    async def test_usage_metadata(self, client: CopilotClient, available_model: str):
        adapter = GeminiAdapter(available_model)
        gemini_body = {
            "contents": [
                {"role": "user", "parts": [{"text": "Say ok."}]}
            ],
            "generationConfig": {"maxOutputTokens": 5},
        }
        openai_body = adapter.convert_chat_request(gemini_body)

        resp = await client.chat_completions(openai_body)
        result = adapter.convert_chat_response(resp.json(), gemini_body)
        usage = result["usageMetadata"]
        assert usage["promptTokenCount"] > 0
        assert usage["candidatesTokenCount"] > 0
        assert usage["totalTokenCount"] > 0
        assert result["modelVersion"] == available_model

    @pytest.mark.asyncio
    async def test_model_list_conversion(self, client: CopilotClient):
        adapter = GeminiAdapter()
        resp = await client.list_models()
        result = adapter.convert_models_response(resp.json())
        assert "models" in result
        assert len(result["models"]) > 0
        model = result["models"][0]
        assert model["name"].startswith("models/")
        assert "baseModelId" in model
        assert "inputTokenLimit" in model
        assert "outputTokenLimit" in model
        assert "supportedGenerationMethods" in model

    @pytest.mark.asyncio
    async def test_single_model_conversion(self, client: CopilotClient, available_model: str):
        adapter = GeminiAdapter(available_model)
        resp = await client.list_models()
        models = resp.json().get("data", [])
        target = next((m for m in models if m["id"] == available_model), None)
        assert target is not None

        result = adapter.convert_single_model(target)
        assert result["name"] == f"models/{available_model}"
        assert result["baseModelId"] == available_model

    @pytest.mark.asyncio
    async def test_stream_error_formatting(self):
        adapter = GeminiAdapter("test-model")
        converter = adapter.create_stream_converter({})
        err = converter.format_error("error: 500 internal")
        assert err == "data: error: 500 internal\n\n"

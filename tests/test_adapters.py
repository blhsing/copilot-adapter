"""Integration tests for format adapters against the live Copilot API.

Each test converts a request through an adapter, sends it to the real API,
and verifies the response converts back correctly.
"""

import json

import pytest

from lib.adapters import AnthropicAdapter, GeminiAdapter, OpenAIAdapter
from lib.adapters.anthropic import (
    _anthropic_to_responses,
    _AnthropicResponsesStreamConverter,
    _responses_to_anthropic,
)
from lib.client import CopilotClient
from lib.server import (
    _anthropic_has_interceptable_web_search,
    _build_web_search_content_blocks,
    _build_web_search_sse_events,
    _execute_web_search_calls,
    _extract_error_message_index,
    _extract_model_from_responses_sse,
    _extract_text_from_responses_stream,
    _extract_tool_calls_from_responses_stream,
    _extract_tool_calls_from_stream,
    _is_model_match,
    _message_debug_outline,
    _normalize_request_params,
    _sanitize_native_anthropic_body,
    _should_use_native_anthropic_api,
    _should_use_responses_api,
)


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
            "model": "claude-opus-4-7",
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

    def test_long_tool_ids_are_shortened_consistently(self):
        adapter = AnthropicAdapter()
        long_tool_id = "toolu_" + ("abc123" * 67)
        anthropic_body = {
            "model": "claude-opus-4-7",
            "max_tokens": 10,
            "messages": [
                {
                    "role": "assistant",
                    "content": [{
                        "type": "tool_use",
                        "id": long_tool_id,
                        "name": "web_search",
                        "input": {"query": "hello"},
                    }],
                },
                {
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": long_tool_id,
                        "content": "done",
                    }],
                },
            ],
        }

        openai_body = adapter.convert_chat_request(anthropic_body)
        tool_call_id = openai_body["messages"][0]["tool_calls"][0]["id"]

        assert tool_call_id != long_tool_id
        assert len(tool_call_id) <= 64
        assert openai_body["messages"][1]["tool_call_id"] == tool_call_id

    def test_tool_use_assistant_message_uses_empty_string_content(self):
        adapter = AnthropicAdapter()
        anthropic_body = {
            "model": "claude-opus-4-7",
            "max_tokens": 10,
            "messages": [
                {
                    "role": "assistant",
                    "content": [{
                        "type": "tool_use",
                        "id": "toolu_123",
                        "name": "Read",
                        "input": {"file_path": "foo.txt"},
                    }],
                },
            ],
        }

        openai_body = adapter.convert_chat_request(anthropic_body)

        assert openai_body["messages"][0]["role"] == "assistant"
        assert openai_body["messages"][0]["content"] == ""
        assert openai_body["messages"][0]["tool_calls"][0]["function"]["name"] == "Read"

    @pytest.mark.asyncio
    async def test_execute_web_search_calls_returns_tool_message_content(self):
        result, raw_data = await _execute_web_search_calls([{
            "id": "call_123",
            "function": {
                "name": "web_search",
                "arguments": json.dumps({"query": "Taipei weather today"}),
            },
        }])

        assert len(result) == 1
        assert result[0]["role"] == "tool"
        assert result[0]["tool_call_id"] == "call_123"
        assert result[0]["content"]
        assert len(raw_data) == 1
        assert raw_data[0][0] == "Taipei weather today"
        assert len(raw_data[0][1]) > 0

    def test_build_web_search_content_blocks(self):
        raw_results = [
            {"title": "Example", "href": "https://example.com", "body": "Test body"},
        ]
        blocks, tool_id = _build_web_search_content_blocks("test query", raw_results)
        assert len(blocks) == 2
        assert blocks[0]["type"] == "server_tool_use"
        assert blocks[0]["name"] == "web_search"
        assert blocks[0]["input"] == {"query": "test query"}
        assert blocks[1]["type"] == "web_search_tool_result"
        assert blocks[1]["tool_use_id"] == tool_id
        assert len(blocks[1]["content"]) == 1
        assert blocks[1]["content"][0]["url"] == "https://example.com"

    def test_build_web_search_sse_events(self):
        raw_results = [
            {"title": "Example", "href": "https://example.com", "body": "Test body"},
        ]
        events = _build_web_search_sse_events("test query", raw_results, 0)
        assert "server_tool_use" in events
        assert "web_search_tool_result" in events
        assert "web_search_result_delta" in events
        assert "test query" in events
        assert "https://example.com" in events
        # Should have 6 events: 3 for server_tool_use + 3 for web_search_tool_result
        assert events.count("event: ") == 6

    def test_stream_tool_call_names_are_reassembled(self):
        chunks = [
            {
                "choices": [{
                    "delta": {
                        "tool_calls": [{
                            "index": 1,
                            "id": "call_123",
                            "function": {"name": "web_"},
                        }],
                    },
                }],
            },
            {
                "choices": [{
                    "delta": {
                        "tool_calls": [{
                            "index": 1,
                            "function": {
                                "name": "search",
                                "arguments": '{"query":"hel',
                            },
                        }],
                    },
                }],
            },
            {
                "choices": [{
                    "delta": {
                        "tool_calls": [{
                            "index": 1,
                            "function": {"arguments": 'lo"}'},
                        }],
                    },
                }],
            },
        ]
        lines = [f"data: {json.dumps(chunk)}" for chunk in chunks]

        tool_calls = _extract_tool_calls_from_stream(lines)

        assert len(tool_calls) == 1
        assert tool_calls[0]["id"] == "call_123"
        assert tool_calls[0]["function"]["name"] == "web_search"
        assert tool_calls[0]["function"]["arguments"] == '{"query":"hello"}'

    def test_stream_tool_calls_with_missing_names_are_discarded(self):
        chunk = {
            "choices": [{
                "delta": {
                    "tool_calls": [{
                        "index": 1,
                        "id": "call_123",
                        "function": {"arguments": "{}"},
                    }],
                },
            }],
        }
        lines = [f"data: {json.dumps(chunk)}"]

        assert _extract_tool_calls_from_stream(lines) == []

    def test_extract_error_message_index(self):
        response_body = {
            "error": {
                "message": "Invalid 'messages[154].tool_calls[1].function.name': empty string.",
            },
        }

        assert _extract_error_message_index(response_body) == 154

    def test_message_debug_outline_flags_empty_tool_names(self):
        outline = _message_debug_outline(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_123",
                        "function": {"name": ""},
                    },
                    {
                        "id": "call_4567",
                        "function": {"name": "web_search"},
                    },
                ],
            },
            12,
        )

        assert outline["index"] == 12
        assert outline["tool_call_count"] == 2
        assert outline["empty_tool_names"] == [0]
        assert outline["tool_id_lengths"] == [8, 9]

    def test_native_anthropic_api_selection(self):
        assert _should_use_native_anthropic_api("anthropic", "claude-opus-4.7") is True
        assert _should_use_native_anthropic_api("anthropic", "gpt-5.4") is False
        assert _should_use_native_anthropic_api("gemini", "claude-opus-4.7") is False

    def test_model_match_dot_vs_dash_versions(self):
        # Dot-separated requested vs dash-separated+date-suffix responded
        assert _is_model_match("claude-haiku-4.5", "claude-haiku-4-5-20251001") is True
        assert _is_model_match("claude-sonnet-4.6", "claude-sonnet-4-6-20260101") is True
        assert _is_model_match("gpt-5.4", "gpt-5-4-2026") is True
        # Exact matches still work
        assert _is_model_match("claude-opus-4-7", "claude-opus-4-7") is True
        # Date-suffix only (no dot)
        assert _is_model_match("gpt-4o-mini", "gpt-4o-mini-2024-07-18") is True
        # Different models must not match
        assert _is_model_match("claude-haiku-4.5", "claude-sonnet-4.6") is False

    def test_normalize_request_params_maps_anthropic_thinking_to_reasoning_effort(self):
        openai_body = {
            "model": "gpt-5.4",
            "messages": [{"role": "user", "content": "hi"}],
            "_copilot_adapter_thinking": {"budget_tokens": 12000},
        }

        normalized = _normalize_request_params(openai_body, "anthropic", "gpt-5.4")

        assert normalized["reasoning_effort"] == "high"
        assert "_copilot_adapter_thinking" not in normalized

    def test_normalize_request_params_maps_anthropic_output_effort_to_reasoning_effort(self):
        openai_body = {
            "model": "gpt-5.4",
            "messages": [{"role": "user", "content": "hi"}],
            "_copilot_adapter_thinking": {"type": "adaptive"},
            "_copilot_adapter_output_effort": "high",
        }

        normalized = _normalize_request_params(openai_body, "anthropic", "gpt-5.4")

        assert normalized["reasoning_effort"] == "high"
        assert "_copilot_adapter_thinking" not in normalized
        assert "_copilot_adapter_output_effort" not in normalized

    def test_normalize_request_params_strips_reasoning_effort_when_tools_present(self):
        openai_body = {
            "model": "gpt-5.4",
            "messages": [{"role": "user", "content": "hi"}],
            "_copilot_adapter_output_effort": "high",
            "_copilot_adapter_thinking": {"type": "adaptive"},
            "tools": [{"type": "function", "function": {"name": "Read"}}],
        }

        normalized = _normalize_request_params(
            openai_body, "anthropic", "gpt-5.4", endpoint="chat_completions"
        )

        assert "reasoning_effort" not in normalized
        assert "tools" in normalized

    def test_normalize_request_params_keeps_reasoning_effort_with_tools_on_responses(self):
        openai_body = {
            "model": "gpt-5.4",
            "messages": [{"role": "user", "content": "hi"}],
            "_copilot_adapter_output_effort": "high",
            "_copilot_adapter_thinking": {"type": "adaptive"},
            "tools": [{"type": "function", "function": {"name": "Read"}}],
        }

        normalized = _normalize_request_params(
            openai_body, "anthropic", "gpt-5.4", endpoint="responses"
        )

        assert normalized["reasoning_effort"] == "high"
        assert "tools" in normalized

    def test_sanitize_native_anthropic_body_drops_context_management(self):
        body = {
            "model": "claude-opus-4.6",
            "messages": [{"role": "user", "content": "hi"}],
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": "high"},
            "context_management": {
                "edits": [{"type": "clear_thinking_20251015", "keep": "all"}],
            },
        }

        sanitized = _sanitize_native_anthropic_body(body)

        assert "context_management" not in sanitized
        assert sanitized["thinking"] == {"type": "adaptive"}
        assert sanitized["output_config"] == {"effort": "high"}
        assert "context_management" in body

    def test_anthropic_has_interceptable_web_search(self):
        assert _anthropic_has_interceptable_web_search({
            "tools": [{"type": "web_search_20250305", "name": "web_search"}],
        }) is True
        assert _anthropic_has_interceptable_web_search({
            "tools": [{"type": "function", "name": "Read"}],
        }) is False
        assert _anthropic_has_interceptable_web_search({"messages": []}) is False

    @pytest.mark.asyncio
    async def test_stream_error_formatting(self):
        adapter = AnthropicAdapter()
        converter = adapter.create_stream_converter({"model": "test"})
        err = converter.format_error("error: 429 rate limited")
        assert err.startswith("event: error\n")
        assert "429" in err


# ===================================================================
# Anthropic → Responses API conversion
# ===================================================================

class TestAnthropicToResponses:
    def test_simple_user_message(self):
        body = {
            "model": "claude-opus-4-7",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "Hello"}],
        }
        result = _anthropic_to_responses(body)
        assert result["model"] == "claude-opus-4-7"
        assert result["max_output_tokens"] == 100
        assert "max_tokens" not in result
        assert len(result["input"]) == 1
        assert result["input"][0] == {"role": "user", "content": "Hello"}

    def test_max_tokens_is_clamped_for_responses_api_minimum(self):
        body = {
            "model": "claude-opus-4-7",
            "max_tokens": 1,
            "messages": [{"role": "user", "content": "quota"}],
        }
        result = _anthropic_to_responses(body)
        assert result["max_output_tokens"] == 16

    def test_system_becomes_developer(self):
        body = {
            "model": "test",
            "max_tokens": 10,
            "system": "You are helpful.",
            "messages": [{"role": "user", "content": "Hi"}],
        }
        result = _anthropic_to_responses(body)
        assert result["input"][0] == {"role": "developer", "content": "You are helpful."}
        assert result["input"][1] == {"role": "user", "content": "Hi"}

    def test_system_block_array(self):
        body = {
            "model": "test",
            "max_tokens": 10,
            "system": [{"type": "text", "text": "Part A"}, {"type": "text", "text": "Part B"}],
            "messages": [{"role": "user", "content": "Hi"}],
        }
        result = _anthropic_to_responses(body)
        assert result["input"][0]["content"] == "Part A\nPart B"

    def test_tool_use_becomes_function_call(self):
        body = {
            "model": "test",
            "max_tokens": 10,
            "messages": [
                {"role": "assistant", "content": [
                    {"type": "text", "text": "Let me check."},
                    {"type": "tool_use", "id": "toolu_123", "name": "Read", "input": {"path": "foo.txt"}},
                ]},
            ],
        }
        result = _anthropic_to_responses(body)
        items = result["input"]
        # Should produce: assistant message + function_call
        assert items[0]["type"] == "message"
        assert items[0]["role"] == "assistant"
        assert items[0]["content"][0]["type"] == "output_text"
        assert items[0]["content"][0]["text"] == "Let me check."
        assert items[1]["type"] == "function_call"
        assert items[1]["name"] == "Read"
        assert items[1]["call_id"] == "toolu_123"
        assert '"path"' in items[1]["arguments"]

    def test_tool_result_becomes_function_call_output(self):
        body = {
            "model": "test",
            "max_tokens": 10,
            "messages": [
                {"role": "assistant", "content": [
                    {"type": "tool_use", "id": "toolu_abc", "name": "Read", "input": {"path": "f"}},
                ]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_abc", "content": "file contents"},
                ]},
            ],
        }
        result = _anthropic_to_responses(body)
        items = result["input"]
        assert items[0]["type"] == "function_call"
        assert items[0]["call_id"] == "toolu_abc"
        assert items[1]["type"] == "function_call_output"
        assert items[1]["call_id"] == "toolu_abc"
        assert items[1]["output"] == "file contents"

    def test_tool_result_with_interleaved_text(self):
        body = {
            "model": "test",
            "max_tokens": 10,
            "messages": [
                {"role": "assistant", "content": [
                    {"type": "tool_use", "id": "toolu_1", "name": "Read", "input": {}},
                ]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_1", "content": "result"},
                    {"type": "text", "text": "Now do something else"},
                ]},
            ],
        }
        result = _anthropic_to_responses(body)
        items = result["input"]
        # function_call, function_call_output, user text
        assert items[0]["type"] == "function_call"
        assert items[1]["type"] == "function_call_output"
        assert items[2]["role"] == "user"
        assert items[2]["content"] == "Now do something else"

    def test_tools_converted(self):
        body = {
            "model": "test",
            "max_tokens": 10,
            "messages": [{"role": "user", "content": "Hi"}],
            "tools": [
                {"name": "Read", "description": "Read file", "input_schema": {"type": "object"}},
                {"type": "web_search_20250305", "name": "web_search"},
            ],
        }
        result = _anthropic_to_responses(body)
        assert len(result["tools"]) == 2
        assert result["tools"][0]["name"] == "Read"
        assert result["tools"][0]["type"] == "function"
        assert "function" not in result["tools"][0]  # flat format, not nested
        assert result["tools"][1]["name"] == "web_search"

    def test_tool_choice_mapping(self):
        body = {
            "model": "test",
            "max_tokens": 10,
            "messages": [{"role": "user", "content": "Hi"}],
            "tools": [{"name": "Read", "description": "", "input_schema": {}}],
            "tool_choice": {"type": "any"},
        }
        result = _anthropic_to_responses(body)
        assert result["tool_choice"] == "required"

    def test_thinking_passthrough(self):
        body = {
            "model": "test",
            "max_tokens": 10,
            "messages": [{"role": "user", "content": "Hi"}],
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": "high"},
        }
        result = _anthropic_to_responses(body)
        assert result["_copilot_adapter_thinking"] == {"type": "adaptive"}
        assert result["_copilot_adapter_output_effort"] == "high"

    def test_image_content(self):
        body = {
            "model": "test",
            "max_tokens": 10,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": "What's this?"},
                {"type": "image", "source": {
                    "type": "base64", "media_type": "image/png", "data": "abc123",
                }},
            ]}],
        }
        result = _anthropic_to_responses(body)
        items = result["input"]
        assert len(items) == 1
        assert items[0]["role"] == "user"
        assert isinstance(items[0]["content"], list)
        assert items[0]["content"][0]["type"] == "input_text"
        assert items[0]["content"][1]["type"] == "input_image"
        assert items[0]["content"][1]["image_url"] == "data:image/png;base64,abc123"

    def test_optional_params(self):
        body = {
            "model": "test",
            "max_tokens": 10,
            "messages": [{"role": "user", "content": "Hi"}],
            "temperature": 0.5,
            "top_p": 0.9,
            "stop_sequences": ["END"],
            "stream": True,
        }
        result = _anthropic_to_responses(body)
        assert result["temperature"] == 0.5
        assert result["top_p"] == 0.9
        assert result["stop"] == ["END"]
        assert result["stream"] is True


class TestResponsesToAnthropic:
    def test_text_response(self):
        resp = {
            "output": [{
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Hello!"}],
            }],
            "status": "completed",
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }
        result = _responses_to_anthropic(resp, "gpt-5.4")
        assert result["type"] == "message"
        assert result["role"] == "assistant"
        assert result["model"] == "gpt-5.4"
        assert len(result["content"]) == 1
        assert result["content"][0] == {"type": "text", "text": "Hello!"}
        assert result["stop_reason"] == "end_turn"
        assert result["usage"] == {"input_tokens": 10, "output_tokens": 5}

    def test_function_call_response(self):
        resp = {
            "output": [{
                "type": "function_call",
                "call_id": "call_123",
                "name": "Read",
                "arguments": '{"path": "foo.txt"}',
            }],
            "status": "completed",
            "usage": {"input_tokens": 10, "output_tokens": 20},
        }
        result = _responses_to_anthropic(resp, "gpt-5.4")
        assert result["stop_reason"] == "tool_use"
        assert len(result["content"]) == 1
        tc = result["content"][0]
        assert tc["type"] == "tool_use"
        assert tc["id"] == "call_123"
        assert tc["name"] == "Read"
        assert tc["input"] == {"path": "foo.txt"}

    def test_mixed_text_and_function_call(self):
        resp = {
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "Let me check."}],
                },
                {
                    "type": "function_call",
                    "call_id": "call_456",
                    "name": "Read",
                    "arguments": "{}",
                },
            ],
            "status": "completed",
            "usage": {"input_tokens": 5, "output_tokens": 15},
        }
        result = _responses_to_anthropic(resp, "gpt-5.4")
        assert result["stop_reason"] == "tool_use"
        assert len(result["content"]) == 2
        assert result["content"][0]["type"] == "text"
        assert result["content"][1]["type"] == "tool_use"

    def test_incomplete_status(self):
        resp = {
            "output": [{"type": "message", "content": [{"type": "output_text", "text": "..."}]}],
            "status": "incomplete",
            "usage": {"input_tokens": 5, "output_tokens": 100},
        }
        result = _responses_to_anthropic(resp, "gpt-5.4")
        assert result["stop_reason"] == "max_tokens"

    def test_empty_output(self):
        resp = {"output": [], "status": "completed", "usage": {}}
        result = _responses_to_anthropic(resp, "gpt-5.4")
        assert len(result["content"]) == 1
        assert result["content"][0] == {"type": "text", "text": ""}


class TestAnthropicResponsesStreamConverter:
    def _make_sse(self, event_type: str, data: dict) -> list[str]:
        """Build SSE lines for a Responses API event."""
        return [f"event: {event_type}", f"data: {json.dumps(data)}"]

    def test_text_flow(self):
        converter = _AnthropicResponsesStreamConverter("gpt-5.4")
        output = ""

        # response.created
        for line in self._make_sse("response.created", {
            "type": "response.created",
            "response": {"id": "resp_1", "model": "gpt-5.4", "output": []},
        }):
            output += converter.feed(line)

        assert "message_start" in output

        # content_part.added
        for line in self._make_sse("response.content_part.added", {
            "type": "response.content_part.added",
            "output_index": 0, "content_index": 0,
            "part": {"type": "output_text", "text": ""},
        }):
            output += converter.feed(line)

        assert "content_block_start" in output

        # text deltas
        for line in self._make_sse("response.output_text.delta", {
            "type": "response.output_text.delta",
            "output_index": 0, "content_index": 0,
            "delta": "Hello world",
        }):
            output += converter.feed(line)

        assert "text_delta" in output
        assert "Hello world" in output

        # content_part.done
        for line in self._make_sse("response.content_part.done", {
            "type": "response.content_part.done",
            "output_index": 0, "content_index": 0,
            "part": {"type": "output_text", "text": "Hello world"},
        }):
            output += converter.feed(line)

        assert "content_block_stop" in output

        # completed
        for line in self._make_sse("response.completed", {
            "type": "response.completed",
            "response": {
                "id": "resp_1", "status": "completed",
                "output": [{"type": "message", "content": [{"type": "output_text", "text": "Hello world"}]}],
                "usage": {"input_tokens": 10, "output_tokens": 5},
            },
        }):
            output += converter.feed(line)

        assert "message_delta" in output
        assert "message_stop" in output
        assert "end_turn" in output

    def test_function_call_flow(self):
        converter = _AnthropicResponsesStreamConverter("gpt-5.4")
        output = ""

        # created
        for line in self._make_sse("response.created", {
            "type": "response.created",
            "response": {"id": "resp_1", "model": "gpt-5.4"},
        }):
            output += converter.feed(line)

        # function_call output_item.added
        for line in self._make_sse("response.output_item.added", {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {
                "type": "function_call",
                "call_id": "call_abc",
                "name": "Read",
                "arguments": "",
                "status": "in_progress",
            },
        }):
            output += converter.feed(line)

        assert "content_block_start" in output
        assert "tool_use" in output
        assert "Read" in output

        # arguments delta
        for line in self._make_sse("response.function_call_arguments.delta", {
            "type": "response.function_call_arguments.delta",
            "output_index": 0,
            "delta": '{"path": "foo"}',
        }):
            output += converter.feed(line)

        assert "input_json_delta" in output

        # output_item.done (function_call)
        for line in self._make_sse("response.output_item.done", {
            "type": "response.output_item.done",
            "output_index": 0,
            "item": {
                "type": "function_call",
                "call_id": "call_abc",
                "name": "Read",
                "arguments": '{"path": "foo"}',
                "status": "completed",
            },
        }):
            output += converter.feed(line)

        assert "content_block_stop" in output

        # completed
        for line in self._make_sse("response.completed", {
            "type": "response.completed",
            "response": {
                "id": "resp_1", "status": "completed",
                "output": [{"type": "function_call", "name": "Read", "call_id": "call_abc", "arguments": '{"path": "foo"}'}],
                "usage": {"input_tokens": 5, "output_tokens": 10},
            },
        }):
            output += converter.feed(line)

        assert "tool_use" in output  # stop_reason
        assert "message_stop" in output

    def test_error_formatting(self):
        converter = _AnthropicResponsesStreamConverter("gpt-5.4")
        err = converter.format_error("error: 500 internal")
        assert err.startswith("event: error\n")
        assert "500" in err


class TestResponsesStreamHelpers:
    def test_extract_tool_calls_from_responses_stream(self):
        lines = [
            'event: response.output_item.done',
            'data: {"type":"response.output_item.done","output_index":0,"item":{"type":"function_call","call_id":"call_1","name":"web_search","arguments":"{\\"query\\":\\"test\\"}"}}',
            'event: response.output_item.done',
            'data: {"type":"response.output_item.done","output_index":1,"item":{"type":"function_call","call_id":"call_2","name":"Read","arguments":"{\\"path\\":\\"f\\"}"}}',
        ]
        result = _extract_tool_calls_from_responses_stream(lines)
        assert len(result) == 2
        assert result[0]["name"] == "web_search"
        assert result[0]["call_id"] == "call_1"
        assert result[1]["name"] == "Read"

    def test_extract_text_from_responses_stream(self):
        lines = [
            'event: response.output_text.delta',
            'data: {"type":"response.output_text.delta","delta":"Hello "}',
            'event: response.output_text.delta',
            'data: {"type":"response.output_text.delta","delta":"world"}',
        ]
        result = _extract_text_from_responses_stream(lines)
        assert result == "Hello world"

    def test_extract_model_from_responses_sse(self):
        line = 'data: {"type":"response.created","response":{"id":"resp_1","model":"gpt-5.4"}}'
        assert _extract_model_from_responses_sse(line) == "gpt-5.4"
        assert _extract_model_from_responses_sse('data: {"type":"response.output_text.delta"}') is None
        assert _extract_model_from_responses_sse('event: response.created') is None

    def test_should_use_responses_api(self):
        assert _should_use_responses_api("gpt-5.4") is True
        assert _should_use_responses_api("gpt-4o") is False
        assert _should_use_responses_api("claude-opus-4-7") is False


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

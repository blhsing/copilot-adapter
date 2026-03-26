"""Integration tests for CopilotClient against the live Copilot API."""

import pytest

from lib.client import CopilotClient
from .conftest import EMBEDDINGS_MODEL, RESPONSES_MODEL


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class TestListModels:
    @pytest.mark.asyncio
    async def test_returns_model_list(self, client: CopilotClient):
        resp = await client.list_models()
        assert resp.status_code == 200
        body = resp.json()
        assert "data" in body
        assert isinstance(body["data"], list)
        assert len(body["data"]) > 0

    @pytest.mark.asyncio
    async def test_model_has_expected_fields(self, client: CopilotClient):
        resp = await client.list_models()
        model = resp.json()["data"][0]
        assert "id" in model
        assert isinstance(model["id"], str)
        assert len(model["id"]) > 0


# ---------------------------------------------------------------------------
# Chat completions (non-streaming)
# ---------------------------------------------------------------------------

class TestChatCompletions:
    @pytest.mark.asyncio
    async def test_simple_completion(self, client: CopilotClient, available_model: str):
        body = {
            "model": available_model,
            "messages": [{"role": "user", "content": "Reply with only the word 'hello'."}],
            "max_tokens": 10,
        }
        resp = await client.chat_completions(body)
        assert resp.status_code == 200
        data = resp.json()
        assert "choices" in data
        assert len(data["choices"]) > 0
        assert data["choices"][0]["message"]["content"]

    @pytest.mark.asyncio
    async def test_completion_with_system_message(self, client: CopilotClient, available_model: str):
        body = {
            "model": available_model,
            "messages": [
                {"role": "system", "content": "You are a calculator. Only respond with numbers."},
                {"role": "user", "content": "What is 2+2?"},
            ],
            "max_tokens": 10,
        }
        resp = await client.chat_completions(body)
        assert resp.status_code == 200
        content = resp.json()["choices"][0]["message"]["content"]
        assert "4" in content

    @pytest.mark.asyncio
    async def test_completion_reports_usage(self, client: CopilotClient, available_model: str):
        body = {
            "model": available_model,
            "messages": [{"role": "user", "content": "Say hi."}],
            "max_tokens": 10,
        }
        resp = await client.chat_completions(body)
        usage = resp.json().get("usage", {})
        assert usage.get("prompt_tokens", 0) > 0
        assert usage.get("completion_tokens", 0) > 0

    @pytest.mark.asyncio
    async def test_completion_finish_reason(self, client: CopilotClient, available_model: str):
        body = {
            "model": available_model,
            "messages": [{"role": "user", "content": "Say one word."}],
            "max_tokens": 5,
        }
        resp = await client.chat_completions(body)
        reason = resp.json()["choices"][0]["finish_reason"]
        assert reason in ("stop", "length")

    @pytest.mark.asyncio
    async def test_multi_turn_conversation(self, client: CopilotClient, available_model: str):
        body = {
            "model": available_model,
            "messages": [
                {"role": "user", "content": "My name is TestBot."},
                {"role": "assistant", "content": "Nice to meet you, TestBot!"},
                {"role": "user", "content": "What is my name? Reply with just the name."},
            ],
            "max_tokens": 20,
        }
        resp = await client.chat_completions(body)
        assert resp.status_code == 200
        content = resp.json()["choices"][0]["message"]["content"]
        assert "TestBot" in content


# ---------------------------------------------------------------------------
# Streaming chat completions
# ---------------------------------------------------------------------------

class TestStreamChatCompletions:
    @pytest.mark.asyncio
    async def test_stream_yields_lines(self, client: CopilotClient, available_model: str):
        body = {
            "model": available_model,
            "messages": [{"role": "user", "content": "Say hello."}],
            "max_tokens": 20,
            "stream": True,
        }
        lines = []
        async for line in client.stream_chat_completions(body):
            lines.append(line)
        assert len(lines) > 0
        assert not lines[0].startswith("error:")

    @pytest.mark.asyncio
    async def test_stream_contains_data_prefix(self, client: CopilotClient, available_model: str):
        body = {
            "model": available_model,
            "messages": [{"role": "user", "content": "Say hi."}],
            "max_tokens": 10,
            "stream": True,
        }
        data_lines = []
        async for line in client.stream_chat_completions(body):
            if line.startswith("data: "):
                data_lines.append(line)
        assert len(data_lines) > 0

    @pytest.mark.asyncio
    async def test_stream_ends_with_done(self, client: CopilotClient, available_model: str):
        body = {
            "model": available_model,
            "messages": [{"role": "user", "content": "Say yes."}],
            "max_tokens": 5,
            "stream": True,
        }
        last_data = ""
        async for line in client.stream_chat_completions(body):
            if line.startswith("data: "):
                last_data = line
        assert last_data.strip() == "data: [DONE]"

    @pytest.mark.asyncio
    async def test_stream_chunks_have_delta(self, client: CopilotClient, available_model: str):
        import json

        body = {
            "model": available_model,
            "messages": [{"role": "user", "content": "Count to 3."}],
            "max_tokens": 30,
            "stream": True,
        }
        found_delta = False
        async for line in client.stream_chat_completions(body):
            if line.startswith("data: ") and not line.strip().endswith("[DONE]"):
                chunk = json.loads(line[6:])
                choices = chunk.get("choices", [])
                if choices and choices[0].get("delta", {}).get("content"):
                    found_delta = True
                    break
        assert found_delta


# ---------------------------------------------------------------------------
# Responses API
# ---------------------------------------------------------------------------

class TestResponses:
    @pytest.mark.asyncio
    async def test_simple_response(self, client: CopilotClient):
        body = {
            "model": RESPONSES_MODEL,
            "input": "Reply with only the word 'pong'.",
        }
        resp = await client.responses(body)
        assert resp.status_code == 200
        data = resp.json()
        assert data.get("output") or data.get("id")

    @pytest.mark.asyncio
    async def test_stream_responses(self, client: CopilotClient):
        body = {
            "model": RESPONSES_MODEL,
            "input": "Say hello.",
            "stream": True,
        }
        lines = []
        async for line in client.stream_responses(body):
            lines.append(line)
        assert len(lines) > 0
        assert not lines[0].startswith("error:")


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------

class TestEmbeddings:
    @pytest.mark.asyncio
    async def test_single_embedding(self, client: CopilotClient):
        body = {
            "model": EMBEDDINGS_MODEL,
            "input": ["The quick brown fox"],
        }
        resp = await client.embeddings(body)
        assert resp.status_code == 200
        data = resp.json()
        assert "data" in data
        assert len(data["data"]) > 0
        embedding = data["data"][0]["embedding"]
        assert isinstance(embedding, list)
        assert len(embedding) > 0
        assert isinstance(embedding[0], float)

    @pytest.mark.asyncio
    async def test_batch_embeddings(self, client: CopilotClient):
        body = {
            "model": EMBEDDINGS_MODEL,
            "input": ["Hello", "World"],
        }
        resp = await client.embeddings(body)
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["data"]) == 2

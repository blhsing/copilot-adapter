"""Integration tests for the FastAPI server endpoints.

These tests hit the real Copilot API through the FastAPI app using httpx's
ASGI transport (no network server needed).
"""

import json

import httpx
import pytest

from .conftest import EMBEDDINGS_MODEL, RESPONSES_MODEL


# ===================================================================
# Health
# ===================================================================

class TestHealth:
    @pytest.mark.asyncio
    async def test_root(self, http_client: httpx.AsyncClient):
        resp = await http_client.get("/")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert body["service"] == "copilot-adapter"


# ===================================================================
# OpenAI: /v1/models
# ===================================================================

class TestModelsEndpoint:
    @pytest.mark.asyncio
    async def test_v1_models(self, http_client: httpx.AsyncClient):
        resp = await http_client.get("/v1/models")
        assert resp.status_code == 200
        body = resp.json()
        assert "data" in body
        assert len(body["data"]) > 0

    @pytest.mark.asyncio
    async def test_models_shorthand(self, http_client: httpx.AsyncClient):
        resp = await http_client.get("/models")
        assert resp.status_code == 200
        assert "data" in resp.json()


# ===================================================================
# OpenAI: /v1/chat/completions
# ===================================================================

class TestChatCompletionsEndpoint:
    @pytest.mark.asyncio
    async def test_non_streaming(self, http_client: httpx.AsyncClient, available_model: str):
        resp = await http_client.post(
            "/v1/chat/completions",
            json={
                "model": available_model,
                "messages": [{"role": "user", "content": "Reply with only 'ok'."}],
                "max_tokens": 5,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "choices" in data
        assert data["choices"][0]["message"]["content"]

    @pytest.mark.asyncio
    async def test_streaming(self, http_client: httpx.AsyncClient, available_model: str):
        resp = await http_client.post(
            "/v1/chat/completions",
            json={
                "model": available_model,
                "messages": [{"role": "user", "content": "Say hi."}],
                "max_tokens": 10,
                "stream": True,
            },
        )
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]

        text = resp.text
        assert "data:" in text

    @pytest.mark.asyncio
    async def test_shorthand_path(self, http_client: httpx.AsyncClient, available_model: str):
        resp = await http_client.post(
            "/chat/completions",
            json={
                "model": available_model,
                "messages": [{"role": "user", "content": "Say yes."}],
                "max_tokens": 5,
            },
        )
        assert resp.status_code == 200
        assert "choices" in resp.json()


# ===================================================================
# OpenAI: /v1/responses
# ===================================================================

class TestResponsesEndpoint:
    @pytest.mark.asyncio
    async def test_non_streaming(self, http_client: httpx.AsyncClient):
        resp = await http_client.post(
            "/v1/responses",
            json={
                "model": RESPONSES_MODEL,
                "input": "Reply with only 'pong'.",
            },
        )
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_streaming(self, http_client: httpx.AsyncClient):
        resp = await http_client.post(
            "/v1/responses",
            json={
                "model": RESPONSES_MODEL,
                "input": "Say hello.",
                "stream": True,
            },
        )
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]


# ===================================================================
# OpenAI: /v1/embeddings
# ===================================================================

class TestEmbeddingsEndpoint:
    @pytest.mark.asyncio
    async def test_single_input(self, http_client: httpx.AsyncClient):
        resp = await http_client.post(
            "/v1/embeddings",
            json={
                "model": EMBEDDINGS_MODEL,
                "input": ["Test sentence"],
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "data" in data
        assert len(data["data"][0]["embedding"]) > 0


# ===================================================================
# Anthropic: /v1/messages
# ===================================================================

class TestAnthropicMessagesEndpoint:
    @pytest.mark.asyncio
    async def test_non_streaming(self, http_client: httpx.AsyncClient, available_model: str):
        resp = await http_client.post(
            "/v1/messages",
            json={
                "model": available_model,
                "max_tokens": 10,
                "messages": [{"role": "user", "content": "Say ok."}],
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["type"] == "message"
        assert data["role"] == "assistant"
        assert data["content"][0]["type"] == "text"
        assert data["content"][0]["text"]
        assert data["stop_reason"] in ("end_turn", "max_tokens", "tool_use")
        assert data["usage"]["input_tokens"] > 0

    @pytest.mark.asyncio
    async def test_streaming(self, http_client: httpx.AsyncClient, available_model: str):
        resp = await http_client.post(
            "/v1/messages",
            json={
                "model": available_model,
                "max_tokens": 20,
                "stream": True,
                "messages": [{"role": "user", "content": "Say hello."}],
            },
        )
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]
        text = resp.text
        assert "event: message_start" in text
        assert "event: content_block_delta" in text
        assert "event: message_stop" in text

    @pytest.mark.asyncio
    async def test_with_system(self, http_client: httpx.AsyncClient, available_model: str):
        resp = await http_client.post(
            "/v1/messages",
            json={
                "model": available_model,
                "max_tokens": 10,
                "system": "Respond only with 'ACK'.",
                "messages": [{"role": "user", "content": "Go."}],
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["type"] == "message"

    @pytest.mark.asyncio
    async def test_shorthand_path(self, http_client: httpx.AsyncClient, available_model: str):
        resp = await http_client.post(
            "/messages",
            json={
                "model": available_model,
                "max_tokens": 5,
                "messages": [{"role": "user", "content": "Hi."}],
            },
        )
        assert resp.status_code == 200
        assert resp.json()["type"] == "message"


# ===================================================================
# Gemini: /v1beta/models
# ===================================================================

class TestGeminiModelsEndpoint:
    @pytest.mark.asyncio
    async def test_list_models(self, http_client: httpx.AsyncClient):
        resp = await http_client.get("/v1beta/models")
        assert resp.status_code == 200
        data = resp.json()
        assert "models" in data
        assert len(data["models"]) > 0
        model = data["models"][0]
        assert model["name"].startswith("models/")

    @pytest.mark.asyncio
    async def test_get_single_model(self, http_client: httpx.AsyncClient, available_model: str):
        resp = await http_client.get(f"/v1beta/models/{available_model}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == f"models/{available_model}"
        assert data["baseModelId"] == available_model

    @pytest.mark.asyncio
    async def test_get_unknown_model_404(self, http_client: httpx.AsyncClient):
        resp = await http_client.get("/v1beta/models/nonexistent-model-xyz")
        assert resp.status_code == 404


# ===================================================================
# Gemini: generateContent / streamGenerateContent
# ===================================================================

class TestGeminiGenerateContentEndpoint:
    @pytest.mark.asyncio
    async def test_generate_content(self, http_client: httpx.AsyncClient, available_model: str):
        resp = await http_client.post(
            f"/v1beta/models/{available_model}:generateContent",
            json={
                "contents": [
                    {"role": "user", "parts": [{"text": "Reply with only 'pong'."}]}
                ],
                "generationConfig": {"maxOutputTokens": 10},
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "candidates" in data
        assert len(data["candidates"]) > 0
        candidate = data["candidates"][0]
        assert candidate["content"]["role"] == "model"
        assert candidate["content"]["parts"][0]["text"]
        assert candidate["finishReason"] in ("STOP", "MAX_TOKENS", "SAFETY")
        assert "usageMetadata" in data

    @pytest.mark.asyncio
    async def test_generate_with_system_instruction(
        self, http_client: httpx.AsyncClient, available_model: str
    ):
        resp = await http_client.post(
            f"/v1beta/models/{available_model}:generateContent",
            json={
                "systemInstruction": {
                    "parts": [{"text": "Always reply with 'CONFIRMED'."}]
                },
                "contents": [
                    {"role": "user", "parts": [{"text": "Status?"}]}
                ],
                "generationConfig": {"maxOutputTokens": 10},
            },
        )
        assert resp.status_code == 200
        text = resp.json()["candidates"][0]["content"]["parts"][0]["text"]
        assert "CONFIRMED" in text.upper()

    @pytest.mark.asyncio
    async def test_stream_generate_content(
        self, http_client: httpx.AsyncClient, available_model: str
    ):
        resp = await http_client.post(
            f"/v1beta/models/{available_model}:streamGenerateContent",
            json={
                "contents": [
                    {"role": "user", "parts": [{"text": "Count to 3."}]}
                ],
                "generationConfig": {"maxOutputTokens": 30},
            },
        )
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]

        # Parse the SSE chunks
        chunks = []
        for line in resp.text.split("\n"):
            if line.startswith("data: "):
                try:
                    chunks.append(json.loads(line[6:]))
                except json.JSONDecodeError:
                    pass
        assert len(chunks) > 0
        assert "candidates" in chunks[0]

    @pytest.mark.asyncio
    async def test_stream_final_chunk_metadata(
        self, http_client: httpx.AsyncClient, available_model: str
    ):
        resp = await http_client.post(
            f"/v1beta/models/{available_model}:streamGenerateContent",
            json={
                "contents": [
                    {"role": "user", "parts": [{"text": "Say ok."}]}
                ],
                "generationConfig": {"maxOutputTokens": 5},
            },
        )
        chunks = []
        for line in resp.text.split("\n"):
            if line.startswith("data: "):
                try:
                    chunks.append(json.loads(line[6:]))
                except json.JSONDecodeError:
                    pass

        # Last chunk should have usage metadata
        last = chunks[-1]
        assert "usageMetadata" in last
        assert last["usageMetadata"]["promptTokenCount"] > 0

"""Unit tests for CopilotClient retry behavior."""

import httpx
import pytest

import lib.client as client_module
from lib.client import CopilotClient


class DummyTokenManager:
    async def get_token(self) -> str:
        return "token"


@pytest.mark.asyncio
async def test_chat_completions_retries_408_on_fresh_connection(monkeypatch):
    responses = [
        httpx.Response(408, content=b"user_request_timeout"),
        httpx.Response(200, json={"ok": True}),
    ]
    clients = []
    posts = []
    sleeps = []

    class FakeAsyncClient:
        def __init__(self, **kwargs):
            self.is_closed = False
            clients.append(self)

        async def post(self, *args, **kwargs):
            posts.append((self, args, kwargs))
            return responses.pop(0)

        async def aclose(self):
            self.is_closed = True

    async def fake_sleep(delay):
        sleeps.append(delay)

    monkeypatch.setattr(client_module.httpx, "AsyncClient", FakeAsyncClient)
    monkeypatch.setattr(client_module.asyncio, "sleep", fake_sleep)

    client = CopilotClient(DummyTokenManager())

    resp = await client.chat_completions({"model": "test", "messages": []})

    assert resp.status_code == 200
    assert len(posts) == 2
    assert len(clients) == 2
    assert clients[0].is_closed is True
    assert clients[1].is_closed is False
    assert sleeps == [0.3]

"""HTTP client for the GitHub Copilot API."""

import json
import uuid
from typing import AsyncIterator

import httpx

try:
    import orjson  # type: ignore

    def _encode(body: dict) -> bytes:
        """Serialize request body with orjson to avoid stdlib json memory blow-ups on large tool schemas."""
        return orjson.dumps(body)
except ImportError:
    def _encode(body: dict) -> bytes:
        """Fallback serializer when orjson isn't installed; may use more memory on large payloads."""
        return json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode("utf-8")

from .auth import CopilotTokenManager

COPILOT_API = "https://api.githubcopilot.com"

_SESSION_ID = str(uuid.uuid4())
_MACHINE_ID = uuid.uuid4().hex


class CopilotClient:
    """Encapsulates all HTTP communication with the GitHub Copilot API."""

    def __init__(self, token_manager: CopilotTokenManager):
        self._token_manager = token_manager

    async def _headers(self, initiator: str = "user") -> dict[str, str]:
        return {
            "authorization": f"Bearer {await self._token_manager.get_token()}",
            "content-type": "application/json",
            "accept": "*/*",
            "editor-version": "vscode/1.95.0",
            "editor-plugin-version": "copilot-chat/0.23.0",
            "copilot-integration-id": "vscode-chat",
            "openai-organization": "github-copilot",
            "openai-intent": "conversation-panel",
            "user-agent": "GitHubCopilotChat/0.23.0",
            "x-request-id": str(uuid.uuid4()),
            "x-initiator": initiator,
            "vscode-sessionid": _SESSION_ID,
            "vscode-machineid": _MACHINE_ID,
        }

    def _url(self, path: str, query: str | None = None) -> str:
        """Build an upstream API URL, preserving an optional raw query string."""
        url = f"{COPILOT_API}{path}"
        if query:
            return f"{url}?{query}"
        return url

    async def chat_completions(
        self, body: dict, *, initiator: str = "user"
    ) -> httpx.Response:
        async with httpx.AsyncClient(timeout=120) as client:
            return await client.post(
                self._url("/chat/completions"),
                headers=await self._headers(initiator),
                content=_encode(body),
            )

    async def stream_chat_completions(
        self, body: dict, *, initiator: str = "user"
    ) -> AsyncIterator[str]:
        timeout = httpx.Timeout(connect=30, read=600, write=30, pool=30)
        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream(
                "POST",
                self._url("/chat/completions"),
                headers=await self._headers(initiator),
                content=_encode(body),
            ) as response:
                if response.status_code != 200:
                    text = await response.aread()
                    yield f"error: {response.status_code} {text.decode()}"
                    return
                try:
                    async for line in response.aiter_lines():
                        yield line
                except httpx.RemoteProtocolError as e:
                    yield f"error: 502 {e}"
                except httpx.ReadTimeout as e:
                    yield f"error: 504 {e}"

    async def responses(
        self, body: dict, *, initiator: str = "user"
    ) -> httpx.Response:
        async with httpx.AsyncClient(timeout=120) as client:
            return await client.post(
                self._url("/responses"),
                headers=await self._headers(initiator),
                content=_encode(body),
            )

    async def stream_responses(
        self, body: dict, *, initiator: str = "user"
    ) -> AsyncIterator[str]:
        timeout = httpx.Timeout(connect=30, read=600, write=30, pool=30)
        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream(
                "POST",
                self._url("/responses"),
                headers=await self._headers(initiator),
                content=_encode(body),
            ) as response:
                if response.status_code != 200:
                    text = await response.aread()
                    yield f"error: {response.status_code} {text.decode()}"
                    return
                try:
                    async for line in response.aiter_lines():
                        yield line
                except httpx.RemoteProtocolError as e:
                    yield f"error: 502 {e}"
                except httpx.ReadTimeout as e:
                    yield f"error: 504 {e}"

    async def list_models(self) -> httpx.Response:
        async with httpx.AsyncClient(timeout=30) as client:
            return await client.get(
                self._url("/models"), headers=await self._headers()
            )

    async def embeddings(
        self, body: dict, *, initiator: str = "user"
    ) -> httpx.Response:
        async with httpx.AsyncClient(timeout=60) as client:
            return await client.post(
                self._url("/embeddings"),
                headers=await self._headers(initiator),
                content=_encode(body),
            )

    async def messages(
        self, body: dict, *, initiator: str = "user", query: str | None = None
    ) -> httpx.Response:
        async with httpx.AsyncClient(timeout=120) as client:
            return await client.post(
                self._url("/v1/messages", query),
                headers=await self._headers(initiator),
                content=_encode(body),
            )

    async def stream_messages(
        self, body: dict, *, initiator: str = "user", query: str | None = None
    ) -> AsyncIterator[str]:
        timeout = httpx.Timeout(connect=30, read=600, write=30, pool=30)
        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream(
                "POST",
                self._url("/v1/messages", query),
                headers=await self._headers(initiator),
                content=_encode(body),
            ) as response:
                if response.status_code != 200:
                    text = await response.aread()
                    yield f"error: {response.status_code} {text.decode()}"
                    return
                try:
                    async for line in response.aiter_lines():
                        yield line
                except httpx.RemoteProtocolError as e:
                    yield f"error: 502 {e}"
                except httpx.ReadTimeout as e:
                    yield f"error: 504 {e}"

"""HTTP client for the GitHub Copilot API."""

import uuid
from typing import AsyncIterator

import httpx

from .auth import CopilotTokenManager

COPILOT_API = "https://api.githubcopilot.com"

_SESSION_ID = str(uuid.uuid4())
_MACHINE_ID = uuid.uuid4().hex


class CopilotClient:
    """Encapsulates all HTTP communication with the GitHub Copilot API."""

    def __init__(self, token_manager: CopilotTokenManager):
        self._token_manager = token_manager

    def _headers(self) -> dict[str, str]:
        return {
            "authorization": f"Bearer {self._token_manager.get_token()}",
            "content-type": "application/json",
            "accept": "*/*",
            "editor-version": "vscode/1.95.0",
            "editor-plugin-version": "copilot-chat/0.23.0",
            "copilot-integration-id": "vscode-chat",
            "openai-organization": "github-copilot",
            "openai-intent": "conversation-panel",
            "user-agent": "GitHubCopilotChat/0.23.0",
            "x-request-id": str(uuid.uuid4()),
            "vscode-sessionid": _SESSION_ID,
            "vscode-machineid": _MACHINE_ID,
        }

    async def chat_completions(self, body: dict) -> httpx.Response:
        async with httpx.AsyncClient(timeout=120) as client:
            return await client.post(
                f"{COPILOT_API}/chat/completions",
                headers=self._headers(),
                json=body,
            )

    async def stream_chat_completions(self, body: dict) -> AsyncIterator[str]:
        async with httpx.AsyncClient(timeout=120) as client:
            async with client.stream(
                "POST",
                f"{COPILOT_API}/chat/completions",
                headers=self._headers(),
                json=body,
            ) as response:
                if response.status_code != 200:
                    text = await response.aread()
                    yield f"error: {response.status_code} {text.decode()}"
                    return
                async for line in response.aiter_lines():
                    yield line

    async def responses(self, body: dict) -> httpx.Response:
        async with httpx.AsyncClient(timeout=120) as client:
            return await client.post(
                f"{COPILOT_API}/responses",
                headers=self._headers(),
                json=body,
            )

    async def stream_responses(self, body: dict) -> AsyncIterator[str]:
        async with httpx.AsyncClient(timeout=120) as client:
            async with client.stream(
                "POST",
                f"{COPILOT_API}/responses",
                headers=self._headers(),
                json=body,
            ) as response:
                if response.status_code != 200:
                    text = await response.aread()
                    yield f"error: {response.status_code} {text.decode()}"
                    return
                async for line in response.aiter_lines():
                    yield line

    async def list_models(self) -> httpx.Response:
        async with httpx.AsyncClient(timeout=30) as client:
            return await client.get(
                f"{COPILOT_API}/models", headers=self._headers()
            )

    async def embeddings(self, body: dict) -> httpx.Response:
        async with httpx.AsyncClient(timeout=60) as client:
            return await client.post(
                f"{COPILOT_API}/embeddings",
                headers=self._headers(),
                json=body,
            )

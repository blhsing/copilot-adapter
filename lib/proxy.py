"""Proxy requests to the GitHub Copilot API."""

import uuid

import httpx

from .auth import CopilotTokenManager

COPILOT_API = "https://api.githubcopilot.com"

_SESSION_ID = str(uuid.uuid4())
_MACHINE_ID = uuid.uuid4().hex


def _make_headers(token_manager: CopilotTokenManager) -> dict[str, str]:
    return {
        "authorization": f"Bearer {token_manager.get_token()}",
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


async def proxy_chat_completions(
    token_manager: CopilotTokenManager, body: dict
) -> httpx.Response:
    headers = _make_headers(token_manager)
    async with httpx.AsyncClient(timeout=120) as client:
        return await client.post(
            f"{COPILOT_API}/chat/completions",
            headers=headers,
            json=body,
        )


async def stream_chat_completions(
    token_manager: CopilotTokenManager, body: dict
):
    headers = _make_headers(token_manager)
    async with httpx.AsyncClient(timeout=120) as client:
        async with client.stream(
            "POST",
            f"{COPILOT_API}/chat/completions",
            headers=headers,
            json=body,
        ) as response:
            if response.status_code != 200:
                text = await response.aread()
                yield f"error: {response.status_code} {text.decode()}"
                return
            async for line in response.aiter_lines():
                yield line


async def proxy_responses(
    token_manager: CopilotTokenManager, body: dict
) -> httpx.Response:
    headers = _make_headers(token_manager)
    async with httpx.AsyncClient(timeout=120) as client:
        return await client.post(
            f"{COPILOT_API}/responses",
            headers=headers,
            json=body,
        )


async def stream_responses(
    token_manager: CopilotTokenManager, body: dict
):
    headers = _make_headers(token_manager)
    async with httpx.AsyncClient(timeout=120) as client:
        async with client.stream(
            "POST",
            f"{COPILOT_API}/responses",
            headers=headers,
            json=body,
        ) as response:
            if response.status_code != 200:
                text = await response.aread()
                yield f"error: {response.status_code} {text.decode()}"
                return
            async for line in response.aiter_lines():
                yield line


async def get_models(token_manager: CopilotTokenManager) -> httpx.Response:
    headers = _make_headers(token_manager)
    async with httpx.AsyncClient(timeout=30) as client:
        return await client.get(f"{COPILOT_API}/models", headers=headers)


async def proxy_embeddings(
    token_manager: CopilotTokenManager, body: dict
) -> httpx.Response:
    headers = _make_headers(token_manager)
    async with httpx.AsyncClient(timeout=60) as client:
        return await client.post(
            f"{COPILOT_API}/embeddings",
            headers=headers,
            json=body,
        )

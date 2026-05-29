"""HTTP client for the ChatGPT (Codex) backend.

Pools ChatGPT Plus/Pro/Business subscriptions alongside Copilot + Anthropic.
Talks to ``chatgpt.com/backend-api/codex`` — the endpoint the Codex CLI uses —
serving the OpenAI Responses API against the plan's quota. Responses-only:
the dispatcher only routes ``/v1/responses`` traffic here.

Auth is an OAuth bearer token from the device flow (see :mod:`lib.openai_auth`)
plus a ``chatgpt-account-id`` header decoded from the id_token JWT.
"""

import json
from typing import AsyncIterator

import httpx

try:
    import orjson  # type: ignore

    def _encode(body: dict) -> bytes:
        return orjson.dumps(body)
except ImportError:
    def _encode(body: dict) -> bytes:
        return json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode("utf-8")

from .openai_auth import OpenAITokenManager, OPENAI_USER_AGENT

CHATGPT_CODEX_API = "https://chatgpt.com/backend-api/codex"
CHATGPT_USAGE_URL = "https://chatgpt.com/backend-api/wham/usage"

_UPSTREAM_TIMEOUT = httpx.Timeout(connect=30, read=1200, write=30, pool=30)


class ChatGPTClient:
    """HTTP client wrapping one ChatGPT (Codex) account.

    Mirrors the Responses surface of :class:`lib.client.CopilotClient`
    (``responses`` + ``stream_responses``) and adds ``fetch_usage`` for the
    live-utilization rotation signal.
    """

    backend = "chatgpt"

    def __init__(self, token_manager: OpenAITokenManager, account_id: str | None,
                 *, account_label: str = ""):
        self._token_manager = token_manager
        self._account_id = account_id
        self._account_label = account_label or token_manager.token_prefix

    @property
    def account_label(self) -> str:
        return self._account_label

    async def _headers(self, initiator: str = "user") -> dict[str, str]:
        token = await self._token_manager.get_token()
        headers = {
            "authorization": f"Bearer {token}",
            "content-type": "application/json",
            "accept": "text/event-stream",
            "originator": "codex_cli_rs",
            "user-agent": OPENAI_USER_AGENT,
            "x-initiator": initiator,
        }
        if self._account_id:
            headers["chatgpt-account-id"] = self._account_id
        return headers

    async def responses(self, body: dict, *, initiator: str = "user") -> httpx.Response:
        async with httpx.AsyncClient(timeout=_UPSTREAM_TIMEOUT) as client:
            return await client.post(
                f"{CHATGPT_CODEX_API}/responses",
                headers=await self._headers(initiator),
                content=_encode(body),
            )

    async def stream_responses(
        self, body: dict, *, initiator: str = "user",
    ) -> AsyncIterator[str]:
        async with httpx.AsyncClient(timeout=_UPSTREAM_TIMEOUT) as client:
            async with client.stream(
                "POST",
                f"{CHATGPT_CODEX_API}/responses",
                headers=await self._headers(initiator),
                content=_encode(body),
            ) as response:
                if response.status_code != 200:
                    text = await response.aread()
                    yield f"error: {response.status_code} {text.decode(errors='replace')}"
                    return
                try:
                    async for line in response.aiter_lines():
                        yield line
                except httpx.RemoteProtocolError as e:
                    yield f"error: 502 {e}"
                except httpx.ReadTimeout as e:
                    yield f"error: 504 {e}"

    async def fetch_usage(self) -> float | None:
        """Return live utilization 0..1 (the binding rate-limit window), or None.

        GET /backend-api/wham/usage -> { rate_limit: { primary_window:
        {used_percent, ...}, secondary_window: {...} } }. Best-effort: any
        failure (Cloudflare challenge, 401, shape change) returns None.
        """
        try:
            token = await self._token_manager.get_token()
            headers = {
                "authorization": f"Bearer {token}",
                "user-agent": OPENAI_USER_AGENT,
            }
            if self._account_id:
                headers["ChatGPT-Account-Id"] = self._account_id
            async with httpx.AsyncClient(timeout=30) as client:
                r = await client.get(CHATGPT_USAGE_URL, headers=headers)
            if r.status_code != 200:
                return None
            rl = (r.json() or {}).get("rate_limit") or {}
            pcts = []
            for win in ("primary_window", "secondary_window"):
                w = rl.get(win) or {}
                up = w.get("used_percent")
                if up is not None:
                    pcts.append(float(up))
            if not pcts:
                return None
            return max(pcts) / 100.0
        except Exception:
            return None

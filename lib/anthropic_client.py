"""HTTP client for the Anthropic Messages API.

Pools real Claude Max subscriptions alongside GitHub Copilot accounts. Talks
directly to ``api.anthropic.com`` with an OAuth bearer token obtained via the
claude.ai PKCE flow (see :mod:`lib.anthropic_auth`).

The ``anthropic-beta: oauth-2025-04-20`` header is mandatory: without it
``api.anthropic.com`` rejects OAuth-issued tokens (only console.anthropic.com
API keys are accepted on the default path).

``SPOOF_INTERACTIVE`` toggles a richer header set that mimics the interactive
Claude Code REPL (User-Agent / x-app / extended ``anthropic-beta`` list) so
SDK-style callers can blend in with REPL traffic on the wire. Off by default.
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

from .anthropic_auth import AnthropicTokenManager

ANTHROPIC_API = "https://api.anthropic.com"

# Module-level toggle, set once at startup from --spoof-interactive-headers.
# Mirrors the static field on AnthropicClient in the .NET sibling.
SPOOF_INTERACTIVE = False

# Headers used when SPOOF_INTERACTIVE is True. Match what the interactive
# Claude Code REPL sends, so the server-side wire-level classifier can't tell
# pool traffic apart from a real REPL session.
_REPL_BETA_LIST = (
    "oauth-2025-04-20,"
    "claude-code-20250219,"
    "interleaved-thinking-2025-05-14,"
    "fine-grained-tool-streaming-2025-05-14,"
    "prompt-caching-2024-07-31"
)
_REPL_USER_AGENT = "claude-cli/1.0.108 (external, cli)"
_POOL_USER_AGENT = "copilot-adapter-anthropic-pool/1.0"

_UPSTREAM_TIMEOUT = httpx.Timeout(connect=30, read=1200, write=30, pool=30)


class AnthropicRateLimitError(Exception):
    """Raised when a request comes back 429.  Carries the response headers so
    the caller can parse ``anthropic-ratelimit-*`` / ``retry-after`` and apply
    a precise unavailable-until timestamp on the account."""

    def __init__(self, status_code: int, headers: dict, body_text: str = ""):
        super().__init__(f"Anthropic returned {status_code}")
        self.status_code = status_code
        self.headers = headers
        self.body_text = body_text


class AnthropicClient:
    """HTTP client wrapping one Anthropic OAuth-pooled account.

    Mirrors the surface of :class:`lib.client.CopilotClient` for the subset of
    endpoints the dispatcher cares about: ``/v1/messages`` (stream + non-stream)
    and ``/v1/messages/count_tokens``.
    """

    def __init__(self, token_manager: AnthropicTokenManager, *, account_label: str = ""):
        self._token_manager = token_manager
        self._account_label = account_label or token_manager.token_prefix

    @property
    def account_label(self) -> str:
        return self._account_label

    async def _headers(self, initiator: str = "user") -> dict[str, str]:
        token = await self._token_manager.get_token()
        if SPOOF_INTERACTIVE:
            ua = _REPL_USER_AGENT
            beta = _REPL_BETA_LIST
            extra = {"x-app": "cli"}
        else:
            ua = _POOL_USER_AGENT
            beta = "oauth-2025-04-20"
            extra = {}
        headers = {
            "authorization": f"Bearer {token}",
            "anthropic-version": "2023-06-01",
            "anthropic-beta": beta,
            "content-type": "application/json",
            "accept": "*/*",
            "user-agent": ua,
            "x-initiator": initiator,
        }
        headers.update(extra)
        return headers

    async def messages(self, body: dict, *, initiator: str = "user") -> httpx.Response:
        async with httpx.AsyncClient(timeout=_UPSTREAM_TIMEOUT) as client:
            return await client.post(
                f"{ANTHROPIC_API}/v1/messages",
                headers=await self._headers(initiator),
                content=_encode(body),
            )

    async def stream_messages(
        self, body: dict, *, initiator: str = "user",
    ) -> AsyncIterator[str]:
        """Stream SSE lines from /v1/messages, yielding raw lines for relay.

        On 429 the first yielded item is a synthetic ``error: 429 …`` line
        with the response headers attached in the wrapping
        :class:`AnthropicRateLimitError`-shaped data — the dispatcher uses
        ``last_response_headers`` to parse Anthropic reset hints.
        """
        async with httpx.AsyncClient(timeout=_UPSTREAM_TIMEOUT) as client:
            async with client.stream(
                "POST",
                f"{ANTHROPIC_API}/v1/messages",
                headers=await self._headers(initiator),
                content=_encode(body),
            ) as response:
                # Stash headers on the stream-iterator for the dispatcher to
                # read after a 429. httpx.Response.headers is already
                # available pre-body, so we capture it eagerly.
                self.last_response_headers = dict(response.headers)
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

    async def count_tokens(self, body: dict) -> httpx.Response:
        async with httpx.AsyncClient(timeout=httpx.Timeout(30)) as client:
            return await client.post(
                f"{ANTHROPIC_API}/v1/messages/count_tokens",
                headers=await self._headers("user"),
                content=_encode(body),
            )

    async def fetch_usage(self) -> float | None:
        """Return live utilization 0..1 (the binding window), or None on error.

        GET /api/oauth/usage -> { five_hour: {utilization, ...}, seven_day:
        {...}, seven_day_opus, seven_day_sonnet, ... }. Anthropic reports
        utilization as a percentage (e.g. 7.0 = 7%); normalize to 0..1.
        """
        data = await self.fetch_usage_details()
        if not data:
            return None
        try:
            pcts = []
            for key in ("five_hour", "seven_day", "seven_day_opus", "seven_day_sonnet"):
                win = data.get(key)
                if isinstance(win, dict) and win.get("utilization") is not None:
                    pcts.append(float(win["utilization"]))
            if not pcts:
                return None
            return max(pcts) / 100.0
        except Exception:
            return None

    async def fetch_usage_details(self) -> dict | None:
        """Return the raw Anthropic OAuth usage payload, or None on error."""
        try:
            token = await self._token_manager.get_token()
            async with httpx.AsyncClient(timeout=30) as client:
                r = await client.get(
                    f"{ANTHROPIC_API}/api/oauth/usage",
                    headers={
                        "authorization": f"Bearer {token}",
                        "anthropic-version": "2023-06-01",
                        "anthropic-beta": "oauth-2025-04-20",
                        "user-agent": _POOL_USER_AGENT,
                    },
                )
            if r.status_code != 200:
                return None
            return r.json() or {}
        except Exception:
            return None

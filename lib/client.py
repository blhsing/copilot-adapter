"""HTTP client for the GitHub Copilot API."""

import asyncio
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

# Shared timeout for Copilot upstream requests. Large prompts (hundreds of
# messages) can take several minutes server-side before responding, so the read
# window is generous. The write window has to accommodate uploading a large
# request body over a possibly-slow uplink; 30s was too tight in practice and
# tripped httpx.WriteTimeout mid-upload on big prompts.
_UPSTREAM_TIMEOUT = httpx.Timeout(connect=30, read=1200, write=90, pool=30)

# Keep idle connections warm for two minutes. This covers normal think/edit
# pauses between Claude Code turns and avoids re-doing the TLS handshake + TCP
# slow start that previously made large request bodies race the upstream
# body-read timer (and lose, returning 408 user_request_timeout).
_UPSTREAM_LIMITS = httpx.Limits(
    max_keepalive_connections=8,
    max_connections=16,
    keepalive_expiry=120,
)

# Upstream occasionally returns 408 when it gives up reading the request body
# mid-upload, and the local side can hit httpx.WriteTimeout / connection resets
# on the same uploads. Retry up to twice, and force a fresh connection between
# attempts. The connection that just got rejected is often "poisoned" (server
# refuses further writes on it), so reusing it from the keepalive pool makes
# the second attempt fail immediately for no real reason.
_RETRYABLE_STATUS = {408}
_RETRYABLE_EXCEPTIONS = (
    httpx.WriteTimeout,
    httpx.WriteError,
    httpx.ConnectTimeout,
    httpx.ConnectError,
    httpx.RemoteProtocolError,
)
_MAX_ATTEMPTS = 3
# Backoff before each retry attempt (attempt index 1, 2, ...).
_RETRY_BACKOFF_SECONDS = (0.3, 0.6)

_SESSION_ID = str(uuid.uuid4())
_MACHINE_ID = uuid.uuid4().hex


class CopilotClient:
    """Encapsulates all HTTP communication with the GitHub Copilot API."""

    def __init__(self, token_manager: CopilotTokenManager):
        self._token_manager = token_manager
        self._client: httpx.AsyncClient | None = None

    def _http(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=_UPSTREAM_TIMEOUT,
                limits=_UPSTREAM_LIMITS,
            )
        return self._client

    async def aclose(self) -> None:
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

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

    async def _reset_connections(self) -> None:
        """Drop all pooled connections so the next request opens fresh ones.

        Reused after a 408 / transport failure: the connection that just got
        rejected by the upstream is often unusable for further writes, and
        httpx's pool would happily hand it back to us on the immediate retry.
        """
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
        self._client = None

    async def _post_with_retry(
        self, url: str, content: bytes, initiator: str
    ) -> httpx.Response:
        last_resp: httpx.Response | None = None
        for attempt in range(_MAX_ATTEMPTS):
            if attempt > 0:
                await self._reset_connections()
                backoff = _RETRY_BACKOFF_SECONDS[
                    min(attempt - 1, len(_RETRY_BACKOFF_SECONDS) - 1)
                ]
                await asyncio.sleep(backoff)
            try:
                resp = await self._http().post(
                    url, headers=await self._headers(initiator), content=content
                )
            except _RETRYABLE_EXCEPTIONS:
                if attempt + 1 >= _MAX_ATTEMPTS:
                    raise
                continue
            if resp.status_code not in _RETRYABLE_STATUS:
                return resp
            await resp.aread()
            last_resp = resp
        assert last_resp is not None
        return last_resp

    async def _stream_with_retry(
        self, url: str, content: bytes, initiator: str
    ) -> AsyncIterator[str]:
        # Open the upstream stream, retrying on transient transport errors or
        # 408 before any bytes are surfaced to the caller. Once the first line
        # is yielded, the response is committed and we cannot transparently
        # retry.
        for attempt in range(_MAX_ATTEMPTS):
            if attempt > 0:
                await self._reset_connections()
                backoff = _RETRY_BACKOFF_SECONDS[
                    min(attempt - 1, len(_RETRY_BACKOFF_SECONDS) - 1)
                ]
                await asyncio.sleep(backoff)
            client = self._http()
            stream_cm = client.stream(
                "POST", url, headers=await self._headers(initiator), content=content
            )
            try:
                response = await stream_cm.__aenter__()
            except _RETRYABLE_EXCEPTIONS:
                if attempt + 1 >= _MAX_ATTEMPTS:
                    raise
                continue
            try:
                if (
                    response.status_code in _RETRYABLE_STATUS
                    and attempt + 1 < _MAX_ATTEMPTS
                ):
                    await response.aread()
                    await stream_cm.__aexit__(None, None, None)
                    continue
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
                return
            finally:
                if not response.is_closed:
                    await stream_cm.__aexit__(None, None, None)

    async def chat_completions(
        self, body: dict, *, initiator: str = "user"
    ) -> httpx.Response:
        return await self._post_with_retry(
            self._url("/chat/completions"), _encode(body), initiator
        )

    async def stream_chat_completions(
        self, body: dict, *, initiator: str = "user"
    ) -> AsyncIterator[str]:
        async for line in self._stream_with_retry(
            self._url("/chat/completions"), _encode(body), initiator
        ):
            yield line

    async def responses(
        self, body: dict, *, initiator: str = "user"
    ) -> httpx.Response:
        return await self._post_with_retry(
            self._url("/responses"), _encode(body), initiator
        )

    async def stream_responses(
        self, body: dict, *, initiator: str = "user"
    ) -> AsyncIterator[str]:
        async for line in self._stream_with_retry(
            self._url("/responses"), _encode(body), initiator
        ):
            yield line

    async def list_models(self) -> httpx.Response:
        return await self._http().get(
            self._url("/models"), headers=await self._headers(), timeout=30
        )

    async def embeddings(
        self, body: dict, *, initiator: str = "user"
    ) -> httpx.Response:
        return await self._http().post(
            self._url("/embeddings"),
            headers=await self._headers(initiator),
            content=_encode(body),
            timeout=60,
        )

    async def messages(
        self, body: dict, *, initiator: str = "user", query: str | None = None
    ) -> httpx.Response:
        return await self._post_with_retry(
            self._url("/v1/messages", query), _encode(body), initiator
        )

    async def stream_messages(
        self, body: dict, *, initiator: str = "user", query: str | None = None
    ) -> AsyncIterator[str]:
        async for line in self._stream_with_retry(
            self._url("/v1/messages", query), _encode(body), initiator
        ):
            yield line

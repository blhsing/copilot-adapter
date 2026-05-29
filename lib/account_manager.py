"""Multi-account management with rotation strategies.

Supports two backends in a single pool:

* **copilot** — GitHub Copilot accounts (device-flow OAuth, billed in premium
  requests against the Copilot plan).
* **anthropic** — real Claude Max subscriptions (claude.ai PKCE OAuth, talks
  directly to api.anthropic.com).

For Claude / ``/v1/messages`` traffic the dispatcher prefers the Anthropic
pool when available and falls back to Copilot. For everything else (OpenAI,
Gemini, non-Claude Anthropic targets) only the Copilot pool is used.

Stickiness is split per-backend (``_last_user_copilot`` /
``_last_user_anthropic``) so an agent-initiated Claude follow-up doesn't pin
to a Copilot account, and vice-versa. On top of that, conversation-keyed
stickiness pins each conversation to a single account so a multi-turn
session doesn't rotate across the pool mid-conversation.
"""

import asyncio
import logging
import random
import time
from dataclasses import dataclass
from typing import Any, Optional

from lib.auth import CopilotTokenManager, update_account  # noqa: F401  (update_account kept for callers)
from lib.client import CopilotClient
from lib.anthropic_auth import (
    AnthropicTokenManager,
    update_anthropic_account_tokens,
)
from lib.anthropic_client import AnthropicClient
from lib.openai_auth import OpenAITokenManager, update_chatgpt_account_tokens
from lib.chatgpt_client import ChatGPTClient

logger = logging.getLogger(__name__)

# Rotation is driven by live backend utilization (see fetch_usage on the
# clients), not a local premium-request counter — Copilot moved to credit-based
# pricing, which made the old per-model multiplier / quota model meaningless.

# Conversation-key stickiness TTL — matches the .NET sibling's 60-minute idle
# eviction. Long enough that a Claude Code session never falls out of cache
# during normal use, short enough that abandoned sessions free their account
# slot without manual intervention.
CONV_STICKY_TTL_SECONDS = 60 * 60


@dataclass
class AccountInfo:
    """Tracks state for one pooled account (Copilot, Anthropic, or ChatGPT)."""

    token: str
    username: str
    backend: str  # "copilot" | "anthropic" | "chatgpt"
    token_manager: Any  # CopilotTokenManager | AnthropicTokenManager | OpenAITokenManager
    client: Any  # CopilotClient | AnthropicClient | ChatGPTClient
    account_id: str | None = None  # chatgpt-account-id (chatgpt backend only)
    utilization: float | None = None  # live 0..1 from the usage poller; None = no signal
    unavailable_until: float | None = None  # transient back-off (rate limit / model mismatch)
    last_request_time: float | None = None

    def is_available(self) -> bool:
        if self.unavailable_until is not None and time.time() < self.unavailable_until:
            return False
        return True


class AccountManager:
    """Selects which client to use per request based on a rotation strategy.

    For ``"agent"`` initiated requests, prefers (in order):

    1. The account already cached for this conversation (`conv_key`).
    2. The last user-selected account for the *requested backend*.
    3. Strategy selection within the backend-filtered pool.

    For ``"user"`` initiated requests, fresh strategy selection is the default,
    with the conv-cache check first so multi-turn user sessions stay sticky.
    """

    # least-utilized (default) rotates by live backend utilization; round-robin
    # ignores it. The legacy max-usage/min-usage strategies are gone (quota model
    # removed) and map to least-utilized.
    STRATEGIES = ("least-utilized", "round-robin")

    def __init__(
        self,
        accounts: list[dict],
        strategy: str = "least-utilized",
        rate_limit_backoff_seconds: int = 3600,
        **_legacy,  # absorb removed quota_limit/plan kwargs from old callers
    ):
        if strategy in ("max-usage", "min-usage"):
            strategy = "least-utilized"
        if strategy not in self.STRATEGIES:
            raise ValueError(f"Unknown strategy: {strategy!r}")
        if not accounts:
            raise ValueError("At least one account is required")

        self._strategy = strategy
        self._rate_limit_backoff_seconds = rate_limit_backoff_seconds
        self._lock = asyncio.Lock()
        self._rr_index = -1
        self._usage_task = None

        # Per-backend last-user account, so an Anthropic conversation's agent
        # follow-up doesn't get pinned to a Copilot account.
        self._last_user_by_backend: dict[str, AccountInfo] = {}

        # Conversation-key cache: (conv_key, backend) -> (account, last_used_ts)
        self._conv_cache: dict[tuple[str, str], tuple[AccountInfo, float]] = {}

        self._accounts: list[AccountInfo] = []
        for acct in accounts:
            if isinstance(acct, tuple):
                acct = {"token": acct[0], "username": acct[1]}
            backend = acct.get("backend", "copilot")
            if backend == "copilot":
                self._accounts.append(self._build_copilot(acct))
            elif backend == "anthropic":
                self._accounts.append(self._build_anthropic(acct))
            elif backend == "chatgpt":
                self._accounts.append(self._build_chatgpt(acct))
            else:
                raise ValueError(f"Unknown backend: {backend!r}")

    def _build_copilot(self, acct: dict) -> AccountInfo:
        token = acct["token"]
        username = acct["username"]
        tm = CopilotTokenManager(token)
        client = CopilotClient(tm)
        return AccountInfo(
            token=token,
            username=username,
            backend="copilot",
            token_manager=tm,
            client=client,
        )

    def _build_anthropic(self, acct: dict) -> AccountInfo:
        username = acct["username"]
        access_token = acct["access_token"]
        refresh_token = acct.get("refresh_token", "")
        expires_at = float(acct.get("expires_at", 0))

        def _on_rotated(new_access: str, new_refresh: str, new_expires: float):
            update_anthropic_account_tokens(username, new_access,
                                            new_refresh, new_expires)

        tm = AnthropicTokenManager(
            access_token, refresh_token, expires_at, on_rotated=_on_rotated,
        )
        client = AnthropicClient(tm, account_label=username)
        return AccountInfo(
            token=access_token,
            username=username,
            backend="anthropic",
            token_manager=tm,
            client=client,
        )

    def _build_chatgpt(self, acct: dict) -> AccountInfo:
        username = acct["username"]
        access_token = acct["access_token"]
        refresh_token = acct.get("refresh_token", "")
        expires_at = float(acct.get("expires_at", 0))
        account_id = acct.get("account_id")

        def _on_rotated(new_access: str, new_refresh: str, new_expires: float):
            update_chatgpt_account_tokens(username, new_access,
                                          new_refresh, new_expires)

        tm = OpenAITokenManager(
            access_token, refresh_token, expires_at, on_rotated=_on_rotated,
        )
        client = ChatGPTClient(tm, account_id, account_label=username)
        return AccountInfo(
            token=access_token,
            username=username,
            backend="chatgpt",
            token_manager=tm,
            client=client,
            account_id=account_id,
        )

    @property
    def accounts(self) -> list[AccountInfo]:
        return list(self._accounts)

    @property
    def strategy(self) -> str:
        return self._strategy

    def get_username(self, client: Any) -> str:
        for acct in self._accounts:
            if acct.client is client:
                return acct.username
        return "unknown"

    def get_backend(self, client: Any) -> str:
        for acct in self._accounts:
            if acct.client is client:
                return acct.backend
        return "unknown"

    def has_available(self, backend: str) -> bool:
        """Return True if at least one account on *backend* is currently
        available. Used by the dispatcher to decide whether to attempt
        Anthropic-first routing for native /v1/messages traffic."""
        return any(
            a.backend == backend and a.is_available() for a in self._accounts
        )

    async def verify_all(self) -> None:
        """Verify Copilot accounts can obtain a Copilot token.

        Anthropic accounts are not verified here — their refresh happens on
        first use to avoid spending a refresh round-trip per restart.
        """
        for acct in self._accounts:
            if acct.backend == "copilot":
                await acct.token_manager.get_token()

    async def get_client(
        self,
        initiator: str = "user",
        *,
        conv_key: str | None = None,
        prefer_backend: str | None = None,
    ) -> Any:
        """Return the client to use for this request.

        The selection order (highest priority first):

        1. **conv_key cache**: if a prior turn from this conversation pinned
           an account that's still available, reuse it.
        2. **agent stickiness**: for ``initiator="agent"``, reuse the last
           user-selected account on the (preferred) backend.
        3. **strategy selection**: filtered by ``prefer_backend`` when set,
           falling back to all backends if the filtered pool is empty or
           fully unavailable.
        """
        async with self._lock:
            self._sweep_conv_cache_locked()

            # 1. Conv-key stickiness — check before everything else so a
            #    long-running conversation never silently changes accounts.
            if conv_key is not None:
                cache_backend = prefer_backend
                if cache_backend is None:
                    # Try every backend the conv has touched; first match wins.
                    for be in ("anthropic", "chatgpt", "copilot"):
                        hit = self._conv_cache.get((conv_key, be))
                        if hit is not None and hit[0].is_available():
                            hit[0]  # noqa: keep reference
                            self._conv_cache[(conv_key, be)] = (hit[0], time.time())
                            return hit[0].client
                else:
                    hit = self._conv_cache.get((conv_key, cache_backend))
                    if hit is not None and hit[0].is_available():
                        self._conv_cache[(conv_key, cache_backend)] = (
                            hit[0], time.time())
                        return hit[0].client

            # 2. Agent stickiness on the preferred backend (or whatever
            #    backend was last used if no preference).
            if initiator == "agent":
                if prefer_backend is not None:
                    sticky = self._last_user_by_backend.get(prefer_backend)
                    if sticky is not None and sticky.is_available():
                        if conv_key is not None:
                            self._conv_cache[(conv_key, sticky.backend)] = (
                                sticky, time.time())
                        return sticky.client
                else:
                    for be in ("anthropic", "chatgpt", "copilot"):
                        sticky = self._last_user_by_backend.get(be)
                        if sticky is not None and sticky.is_available():
                            if conv_key is not None:
                                self._conv_cache[(conv_key, sticky.backend)] = (
                                    sticky, time.time())
                            return sticky.client

            # 3. Strategy selection within the backend-filtered pool.
            acct = await self._select_by_strategy(prefer_backend=prefer_backend)
            self._last_user_by_backend[acct.backend] = acct
            if conv_key is not None:
                self._conv_cache[(conv_key, acct.backend)] = (acct, time.time())
            return acct.client

    async def record_usage(self, client: Any, model: str) -> None:
        """No-op. The local premium-usage counter was removed (Copilot's
        credit-based pricing made it meaningless); rotation uses live
        utilization instead. Kept so existing call sites don't break."""
        return

    async def remember_conversation(self, conv_key: str | None, client: Any) -> None:
        """Pin *conv_key* to the account owning *client*. Used by /v1/responses
        to map an upstream response id (the next turn's previous_response_id)
        back to its account — Responses ids are provider-specific."""
        if not conv_key or client is None:
            return
        async with self._lock:
            for acct in self._accounts:
                if acct.client is client:
                    self._conv_cache[(conv_key, acct.backend)] = (acct, time.time())
                    return

    # ----- live utilization poller -----------------------------------------
    async def refresh_usage_once(self) -> None:
        """Refresh live utilization for anthropic/chatgpt accounts (best-effort)."""
        targets = [a for a in self._accounts
                   if a.backend in ("anthropic", "chatgpt")]
        for acct in targets:
            fn = getattr(acct.client, "fetch_usage", None)
            if fn is None:
                continue
            try:
                acct.utilization = await fn()
            except Exception:
                pass  # leave stale; never break rotation

    async def _usage_poller(self) -> None:
        while True:
            await self.refresh_usage_once()
            await asyncio.sleep(180)

    def start_usage_poller(self) -> None:
        """Start the background usage poller (idempotent). Call once after the
        event loop is running (e.g. FastAPI startup)."""
        if self._usage_task is None:
            self._usage_task = asyncio.ensure_future(self._usage_poller())

    async def record_request_time(self, client: Any) -> None:
        async with self._lock:
            for acct in self._accounts:
                if acct.client is client:
                    acct.last_request_time = time.time()
                    break

    async def get_minutes_since_last_request(self, client: Any) -> float | None:
        async with self._lock:
            for acct in self._accounts:
                if acct.client is client:
                    if acct.last_request_time is None:
                        return None
                    return (time.time() - acct.last_request_time) / 60.0
        return None

    async def mark_exhausted(self, client: Any) -> None:
        """Sideline the account owning *client* using the default backoff."""
        async with self._lock:
            for acct in self._accounts:
                if acct.client is client:
                    self._mark_unavailable_locked(acct)
                    break

    async def mark_exhausted_until(self, client: Any, until_utc: float) -> None:
        """Sideline the account owning *client* until a specific UTC
        timestamp. Used for parsed Anthropic ``anthropic-ratelimit-*-reset``
        headers so we don't sideline for longer (or shorter) than necessary.
        """
        async with self._lock:
            for acct in self._accounts:
                if acct.client is client:
                    acct.unavailable_until = max(
                        until_utc, time.time() + 30
                    )
                    logger.warning(
                        "Account %s sidelined until %s (precise upstream hint)",
                        acct.username,
                        time.strftime("%Y-%m-%d %H:%M:%S UTC",
                                      time.gmtime(acct.unavailable_until)),
                    )
                    self._purge_conv_entries_locked(acct)
                    break

    async def get_fallback_client(
        self,
        failed_client: Any,
        *,
        prefer_backend: str | None = None,
    ) -> Optional[Any]:
        """Sideline *failed_client* and return the next available client.

        Tries the preferred backend first if specified, then any backend.
        Returns ``None`` if no other accounts are available — callers should
        surface the upstream error in that case. The failed account is always
        sidelined (even when no fallback exists) so that subsequent requests
        back off instead of hammering an account known to be rate-limited.
        """
        async with self._lock:
            self._sweep_conv_cache_locked()
            # Identify the failed account's backend, then sideline it
            # unconditionally before evaluating fallback candidates.
            failed_backend = None
            for acct in self._accounts:
                if acct.client is failed_client:
                    failed_backend = acct.backend
                    self._mark_unavailable_locked(acct)
                    break

            target_backend = prefer_backend or failed_backend
            available = [
                a for a in self._accounts
                if a.is_available()
                and a.client is not failed_client
                and (target_backend is None or a.backend == target_backend)
            ]
            if not available and target_backend is not None:
                # Cross-backend fallback. A failed ChatGPT account falls back to
                # Copilot (both serve /responses), never Anthropic (messages-only).
                exclude = {"anthropic"} if failed_backend == "chatgpt" else set()
                available = [
                    a for a in self._accounts
                    if a.is_available() and a.client is not failed_client
                    and a.backend not in exclude
                ]
            if not available:
                return None
            return available[0].client

    def _mark_unavailable_locked(self, acct: AccountInfo) -> None:
        acct.unavailable_until = time.time() + self._rate_limit_backoff_seconds
        logger.warning(
            "Account %s sidelined for %d minutes (rate-limit back-off)",
            acct.username, self._rate_limit_backoff_seconds // 60,
        )
        self._purge_conv_entries_locked(acct)

    def _purge_conv_entries_locked(self, acct: AccountInfo) -> None:
        """Remove any conv-key cache entries pointing at *acct* so subsequent
        turns don't try a dead account."""
        dead_keys = [k for k, v in self._conv_cache.items() if v[0] is acct]
        for k in dead_keys:
            del self._conv_cache[k]

    def _sweep_conv_cache_locked(self) -> None:
        """Drop conv-cache entries idle longer than the TTL."""
        if not self._conv_cache:
            return
        cutoff = time.time() - CONV_STICKY_TTL_SECONDS
        dead = [k for k, (_, ts) in self._conv_cache.items() if ts < cutoff]
        for k in dead:
            del self._conv_cache[k]

    async def _select_by_strategy(self, *,
                                  prefer_backend: str | None) -> AccountInfo:
        """Apply the configured rotation strategy. Must be called with lock held."""
        if prefer_backend is not None:
            available = [a for a in self._accounts
                         if a.backend == prefer_backend and a.is_available()]
        else:
            # ChatGPT is Responses-only — never pick it for the generic pool
            # (chat/completions, native messages). It's reachable only via an
            # explicit prefer_backend="chatgpt".
            available = [a for a in self._accounts
                         if a.backend != "chatgpt" and a.is_available()]

        if not available and prefer_backend is not None:
            # Fall back to the full pool (excluding chatgpt for the same reason).
            available = [a for a in self._accounts
                         if a.backend != "chatgpt" and a.is_available()]

        if not available:
            raise RuntimeError(
                "All accounts are unavailable (rate-limited or quota-exhausted upstream)."
            )

        return self._pick_from(available)

    def _pick_from(self, available: list[AccountInfo]) -> AccountInfo:
        """Apply the rotation strategy to a candidate list. Lock held."""
        def _round_robin() -> AccountInfo:
            for _ in range(len(self._accounts)):
                self._rr_index = (self._rr_index + 1) % len(self._accounts)
                candidate = self._accounts[self._rr_index]
                if candidate in available:
                    return candidate
            return available[0]

        if self._strategy == "round-robin":
            return _round_robin()

        # least-utilized: prefer accounts with a live utilization signal
        # (anthropic/chatgpt) and pick the lowest; accounts with no signal
        # (copilot — credit-based, no per-account usage) round-robin.
        with_util = [a for a in available if a.utilization is not None]
        if with_util:
            target = min(a.utilization for a in with_util)
            tied = [a for a in with_util if a.utilization == target]
            return random.choice(tied)
        return _round_robin()

    async def get_responses_client(
        self,
        initiator: str = "user",
        *,
        conv_key: str | None = None,
        force_copilot: bool = False,
    ) -> Any:
        """Account selection for /v1/responses: conversation stickiness wins
        over backend preference (a prior response id is account-specific),
        otherwise prefer the ChatGPT pool and fall back to Copilot. ChatGPT is
        only reachable through this path."""
        async with self._lock:
            self._sweep_conv_cache_locked()
            if conv_key is not None:
                for be in ("chatgpt", "copilot", "anthropic"):
                    hit = self._conv_cache.get((conv_key, be))
                    if hit is not None and hit[0].is_available():
                        self._conv_cache[(conv_key, be)] = (hit[0], time.time())
                        if initiator == "user":
                            self._last_user_by_backend[hit[0].backend] = hit[0]
                        return hit[0].client

            acct = None
            if not force_copilot:
                cg = [a for a in self._accounts
                      if a.backend == "chatgpt" and a.is_available()]
                if cg:
                    acct = self._pick_from(cg)
            if acct is None:
                cop = [a for a in self._accounts
                       if a.backend == "copilot" and a.is_available()]
                if cop:
                    acct = self._pick_from(cop)
            if acct is None:
                raise RuntimeError(
                    "No Copilot/ChatGPT accounts available for /v1/responses.")
            self._last_user_by_backend[acct.backend] = acct
            if conv_key is not None:
                self._conv_cache[(conv_key, acct.backend)] = (acct, time.time())
            return acct.client

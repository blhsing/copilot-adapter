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

from lib.auth import CopilotTokenManager, update_account
from lib.client import CopilotClient
from lib.anthropic_auth import (
    AnthropicTokenManager,
    update_anthropic_account_tokens,
    update_anthropic_account_usage,
)
from lib.anthropic_client import AnthropicClient

logger = logging.getLogger(__name__)

# Premium request multipliers per plan type.
# Source: https://docs.github.com/en/copilot/concepts/billing/copilot-requests
# Models not listed default to 1.
_PAID_MULTIPLIERS: dict[str, float] = {
    # Free (0x)
    "gpt-4.1": 0,
    "gpt-4o": 0,
    "gpt-5-mini": 0,
    "raptor-mini": 0,
    # Discounted (0.25x)
    "grok-code-fast-1": 0.25,
    # Discounted (0.33x)
    "claude-haiku-4.5": 0.33,
    "gemini-3-flash": 0.33,
    "gpt-5.1-codex-mini": 0.33,
    "gpt-5.4-mini": 0.33,
    # Standard (1x) — omitted, default is 1
    # Premium (3x)
    "claude-opus-4.5": 3,
    "claude-opus-4.6": 3,
    # Ultra premium (7.5x)
    "claude-opus-4.7": 7.5,
}

# On the Free plan, every available model costs 1 premium request.
_FREE_MULTIPLIERS: dict[str, float] = {}  # empty = all default to 1

PLAN_MULTIPLIERS: dict[str, dict[str, float]] = {
    "free": _FREE_MULTIPLIERS,
    "pro": _PAID_MULTIPLIERS,
    "pro+": _PAID_MULTIPLIERS,
    "business": _PAID_MULTIPLIERS,
    "enterprise": _PAID_MULTIPLIERS,
    # Anthropic Max subscriptions don't expose a quota; we still track usage
    # for visibility but exhaustion comes from upstream 429s, not arithmetic.
    "max": _PAID_MULTIPLIERS,
    "max-5x": _PAID_MULTIPLIERS,
    "max-20x": _PAID_MULTIPLIERS,
}

# Default monthly premium request quota per plan.
PLAN_QUOTAS: dict[str, int] = {
    "free": 50,
    "pro": 300,
    "pro+": 1500,
    "business": 300,
    "enterprise": 1000,
}

VALID_PLANS = tuple(PLAN_MULTIPLIERS.keys())

# Conversation-key stickiness TTL — matches the .NET sibling's 60-minute idle
# eviction. Long enough that a Claude Code session never falls out of cache
# during normal use, short enough that abandoned sessions free their account
# slot without manual intervention.
CONV_STICKY_TTL_SECONDS = 60 * 60


def get_model_multiplier(model: str, plan: str = "pro") -> float:
    """Return the premium request multiplier for a model.

    Uses prefix matching to handle date-suffixed model IDs
    (e.g. ``gpt-4o-2024-07-18`` matches ``gpt-4o``).
    Defaults to 1 for unknown models.
    """
    multipliers = PLAN_MULTIPLIERS.get(plan, _PAID_MULTIPLIERS)
    if not multipliers:
        return 1.0
    if model in multipliers:
        return multipliers[model]
    # Try prefix match (longest prefix wins)
    best_match = ""
    for prefix in multipliers:
        if model.startswith(prefix) and len(prefix) > len(best_match):
            best_match = prefix
    if best_match:
        return multipliers[best_match]
    return 1.0


@dataclass
class AccountInfo:
    """Tracks state for one pooled account (Copilot or Anthropic)."""

    token: str
    username: str
    backend: str  # "copilot" | "anthropic"
    token_manager: Any  # CopilotTokenManager | AnthropicTokenManager
    client: Any  # CopilotClient | AnthropicClient
    plan: str = "pro"
    premium_used: float = 0
    premium_limit: int | None = None  # None = unknown
    exhausted: bool = False  # permanent quota exhaustion (clears only on restart)
    unavailable_until: float | None = None  # transient back-off (e.g. rate limit)
    last_request_time: float | None = None

    def is_available(self) -> bool:
        if self.exhausted:
            return False
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

    STRATEGIES = ("max-usage", "min-usage", "round-robin")

    def __init__(
        self,
        accounts: list[dict],
        strategy: str = "max-usage",
        quota_limit: int | None = None,
        plan: str = "pro",
        rate_limit_backoff_seconds: int = 3600,
    ):
        if strategy not in self.STRATEGIES:
            raise ValueError(f"Unknown strategy: {strategy!r}")
        if not accounts:
            raise ValueError("At least one account is required")

        self._strategy = strategy
        self._plan = plan
        self._rate_limit_backoff_seconds = rate_limit_backoff_seconds
        self._lock = asyncio.Lock()
        self._rr_index = -1

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
                self._accounts.append(self._build_copilot(acct, quota_limit, plan))
            elif backend == "anthropic":
                self._accounts.append(self._build_anthropic(acct, quota_limit))
            else:
                raise ValueError(f"Unknown backend: {backend!r}")

    def _build_copilot(self, acct: dict, default_quota: int | None,
                       default_plan: str) -> AccountInfo:
        token = acct["token"]
        username = acct["username"]
        acct_plan = acct.get("plan", default_plan)
        acct_limit = acct.get("quota_limit", default_quota)
        acct_used = round(acct.get("premium_used", 0), 2)
        if acct_limit is None:
            acct_limit = PLAN_QUOTAS.get(acct_plan)
        tm = CopilotTokenManager(token)
        client = CopilotClient(tm)
        return AccountInfo(
            token=token,
            username=username,
            backend="copilot",
            token_manager=tm,
            client=client,
            plan=acct_plan,
            premium_used=acct_used,
            premium_limit=acct_limit,
        )

    def _build_anthropic(self, acct: dict,
                         default_quota: int | None) -> AccountInfo:
        username = acct["username"]
        access_token = acct["access_token"]
        refresh_token = acct.get("refresh_token", "")
        expires_at = float(acct.get("expires_at", 0))
        acct_plan = acct.get("plan", "max")
        acct_limit = acct.get("quota_limit", default_quota)
        acct_used = round(acct.get("premium_used", 0), 2)

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
            plan=acct_plan,
            premium_used=acct_used,
            premium_limit=acct_limit,
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
                    for be in ("anthropic", "copilot"):
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
                    for be in ("anthropic", "copilot"):
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
        """Record a premium request for the account owning *client*."""
        username = None
        backend = None
        usage = 0.0
        async with self._lock:
            for acct in self._accounts:
                if acct.client is client:
                    multiplier = get_model_multiplier(model, acct.plan)
                    if multiplier == 0:
                        return
                    acct.premium_used = round(acct.premium_used + multiplier, 2)
                    username = acct.username
                    backend = acct.backend
                    usage = acct.premium_used
                    if (acct.premium_limit is not None
                            and acct.premium_used >= acct.premium_limit):
                        acct.exhausted = True
                        logger.info(
                            "Account %s reached quota limit (%.1f/%d)",
                            acct.username, acct.premium_used, acct.premium_limit,
                        )
                        self._purge_conv_entries_locked(acct)
                    break
        if username is not None:
            if backend == "copilot":
                update_account(username, premium_used=usage)
            else:
                update_anthropic_account_usage(username, usage)

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
        """Return the next available client after temporarily sidelining
        *failed_client*. Tries the preferred backend first if specified, then
        any backend. Returns ``None`` if no other accounts are available."""
        async with self._lock:
            self._sweep_conv_cache_locked()
            # Same-backend candidates first (or filtered by prefer_backend).
            failed_backend = None
            for acct in self._accounts:
                if acct.client is failed_client:
                    failed_backend = acct.backend
                    break

            target_backend = prefer_backend or failed_backend
            available = [
                a for a in self._accounts
                if a.is_available()
                and a.client is not failed_client
                and (target_backend is None or a.backend == target_backend)
            ]
            if not available and target_backend is not None:
                # Cross-backend fallback.
                available = [
                    a for a in self._accounts
                    if a.is_available() and a.client is not failed_client
                ]
            if not available:
                return None
            # Sideline only after we've confirmed a fallback exists.
            for acct in self._accounts:
                if acct.client is failed_client:
                    self._mark_unavailable_locked(acct)
                    break
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
            pool = [a for a in self._accounts if a.backend == prefer_backend]
            available = [a for a in pool if a.is_available()]
        else:
            pool = self._accounts
            available = [a for a in pool if a.is_available()]

        if not available and prefer_backend is not None:
            # Fall back to the full pool.
            available = [a for a in self._accounts if a.is_available()]

        if not available:
            raise RuntimeError(
                "All accounts are unavailable (quota exhausted or rate-limited)."
            )

        if self._strategy == "round-robin":
            # RR over the full _accounts list, skipping those filtered out.
            for _ in range(len(self._accounts)):
                self._rr_index = (self._rr_index + 1) % len(self._accounts)
                candidate = self._accounts[self._rr_index]
                if candidate in available:
                    return candidate
            return available[0]

        if self._strategy == "max-usage":
            target = max(a.premium_used for a in available)
            tied = [a for a in available if a.premium_used == target]
            return random.choice(tied)

        # min-usage
        target = min(a.premium_used for a in available)
        tied = [a for a in available if a.premium_used == target]
        return random.choice(tied)

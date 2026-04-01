"""Multi-account management with rotation strategies."""

import asyncio
import logging
from dataclasses import dataclass, field

from lib.auth import CopilotTokenManager
from lib.client import CopilotClient

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
    # Ultra premium (30x)
    "claude-opus-4.6-fast": 30,
}

# On the Free plan, every available model costs 1 premium request.
_FREE_MULTIPLIERS: dict[str, float] = {}  # empty = all default to 1

PLAN_MULTIPLIERS: dict[str, dict[str, float]] = {
    "free": _FREE_MULTIPLIERS,
    "pro": _PAID_MULTIPLIERS,
    "pro+": _PAID_MULTIPLIERS,
    "business": _PAID_MULTIPLIERS,
    "enterprise": _PAID_MULTIPLIERS,
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
    """Tracks state for one Copilot account."""

    token: str
    username: str
    token_manager: CopilotTokenManager
    client: CopilotClient
    plan: str = "pro"
    premium_used: float = 0
    premium_limit: int | None = None  # None = unknown
    exhausted: bool = False


class AccountManager:
    """Selects which CopilotClient to use per request based on a rotation strategy.

    Agent-initiated requests stick to the last user-selected account to avoid
    billing a premium request on a different account.
    """

    STRATEGIES = ("max-usage", "min-usage", "round-robin")

    def __init__(
        self,
        accounts: list[tuple[str, str]] | list[dict],
        strategy: str = "max-usage",
        quota_limit: int | None = None,
        plan: str = "pro",
    ):
        if strategy not in self.STRATEGIES:
            raise ValueError(f"Unknown strategy: {strategy!r}")
        if not accounts:
            raise ValueError("At least one account is required")

        self._strategy = strategy
        self._plan = plan
        self._lock = asyncio.Lock()
        self._rr_index = -1
        self._last_user_account: AccountInfo | None = None

        self._accounts: list[AccountInfo] = []
        for acct in accounts:
            if isinstance(acct, dict):
                token = acct["token"]
                username = acct["username"]
                acct_plan = acct.get("plan", plan)
                acct_limit = acct.get("quota_limit", quota_limit)
                acct_used = acct.get("premium_used", 0)
            else:
                token, username = acct
                acct_plan = plan
                acct_limit = quota_limit
                acct_used = 0
            # Default quota limit from plan if not explicitly set
            if acct_limit is None:
                acct_limit = PLAN_QUOTAS.get(acct_plan)
            tm = CopilotTokenManager(token)
            client = CopilotClient(tm)
            self._accounts.append(AccountInfo(
                token=token,
                username=username,
                token_manager=tm,
                client=client,
                plan=acct_plan,
                premium_used=acct_used,
                premium_limit=acct_limit,
            ))

    @property
    def accounts(self) -> list[AccountInfo]:
        return list(self._accounts)

    @property
    def strategy(self) -> str:
        return self._strategy

    async def verify_all(self) -> None:
        """Verify all accounts can obtain a Copilot token."""
        for acct in self._accounts:
            await acct.token_manager.get_token()

    async def get_client(self, initiator: str = "user") -> CopilotClient:
        """Return the CopilotClient to use for this request.

        For ``"agent"`` initiator, returns the same client as the most recent
        ``"user"`` request. For ``"user"`` initiator, selects based on strategy.
        """
        async with self._lock:
            if initiator == "agent" and self._last_user_account is not None:
                return self._last_user_account.client

            acct = await self._select_by_strategy()
            self._last_user_account = acct
            return acct.client

    async def record_usage(self, client: CopilotClient, model: str) -> None:
        """Record a premium request for the account owning *client*.

        Uses the model's multiplier (based on the configured plan) to
        accurately track premium request consumption.
        """
        async with self._lock:
            for acct in self._accounts:
                if acct.client is client:
                    multiplier = get_model_multiplier(model, acct.plan)
                    if multiplier == 0:
                        return
                    acct.premium_used += multiplier
                    if acct.premium_limit is not None and acct.premium_used >= acct.premium_limit:
                        acct.exhausted = True
                        logger.info(
                            "Account %s reached quota limit (%.1f/%d)",
                            acct.username, acct.premium_used, acct.premium_limit,
                        )
                    break

    async def mark_exhausted(self, client: CopilotClient) -> None:
        """Mark the account associated with *client* as exhausted."""
        async with self._lock:
            for acct in self._accounts:
                if acct.client is client:
                    acct.exhausted = True
                    logger.warning("Account %s marked as exhausted", acct.username)
                    break

    async def get_fallback_client(self, failed_client: CopilotClient) -> CopilotClient | None:
        """Return the next non-exhausted client after marking *failed_client* exhausted.

        If no other accounts are available, returns ``None`` *without* marking
        the current account exhausted — so single-account setups keep working.
        """
        async with self._lock:
            available = [a for a in self._accounts
                         if not a.exhausted and a.client is not failed_client]
            if not available:
                return None
            # Only mark the failed account now that we know there's a fallback
            for acct in self._accounts:
                if acct.client is failed_client:
                    acct.exhausted = True
                    logger.warning("Account %s marked as exhausted", acct.username)
                    break
            return available[0].client

    async def _select_by_strategy(self) -> AccountInfo:
        """Apply the configured rotation strategy. Must be called with lock held."""
        available = [a for a in self._accounts if not a.exhausted]
        if not available:
            raise RuntimeError(
                "All accounts have exhausted their premium request quota."
            )

        if self._strategy == "round-robin":
            self._rr_index = (self._rr_index + 1) % len(self._accounts)
            # Skip exhausted, wrapping around
            for _ in range(len(self._accounts)):
                candidate = self._accounts[self._rr_index]
                if not candidate.exhausted:
                    return candidate
                self._rr_index = (self._rr_index + 1) % len(self._accounts)
            raise RuntimeError(
                "All accounts have exhausted their premium request quota."
            )

        if self._strategy == "max-usage":
            return max(available, key=lambda a: a.premium_used)

        # min-usage
        return min(available, key=lambda a: a.premium_used)

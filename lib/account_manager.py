"""Multi-account management with rotation strategies."""

import logging
import threading
import time
from dataclasses import dataclass, field

import httpx

from lib.auth import CopilotTokenManager
from lib.client import CopilotClient

logger = logging.getLogger(__name__)

QUOTA_REFRESH_INTERVAL = 300  # seconds between usage API checks


@dataclass
class AccountInfo:
    """Tracks state for one Copilot account."""

    token: str
    username: str
    token_manager: CopilotTokenManager
    client: CopilotClient
    premium_used: int = 0
    premium_limit: int | None = None  # None = unknown (no --quota-limit)
    exhausted: bool = False
    last_quota_check: float = 0


class AccountManager:
    """Selects which CopilotClient to use per request based on a rotation strategy.

    Agent-initiated requests stick to the last user-selected account to avoid
    billing a premium request on a different account.
    """

    STRATEGIES = ("max-usage", "min-usage", "round-robin")

    def __init__(
        self,
        accounts: list[tuple[str, str]],
        strategy: str = "max-usage",
        quota_limit: int | None = None,
        local_tracking: bool = False,
    ):
        if strategy not in self.STRATEGIES:
            raise ValueError(f"Unknown strategy: {strategy!r}")
        if not accounts:
            raise ValueError("At least one account is required")

        self._strategy = strategy
        self._local_tracking = local_tracking
        self._lock = threading.Lock()
        self._rr_index = -1
        self._last_user_account: AccountInfo | None = None

        self._accounts: list[AccountInfo] = []
        for token, username in accounts:
            tm = CopilotTokenManager(token)
            client = CopilotClient(tm)
            self._accounts.append(AccountInfo(
                token=token,
                username=username,
                token_manager=tm,
                client=client,
                premium_limit=quota_limit,
            ))

    @property
    def accounts(self) -> list[AccountInfo]:
        return list(self._accounts)

    @property
    def strategy(self) -> str:
        return self._strategy

    def verify_all(self) -> None:
        """Verify all accounts can obtain a Copilot token."""
        for acct in self._accounts:
            acct.token_manager.get_token()

    def get_client(self, initiator: str = "user") -> CopilotClient:
        """Return the CopilotClient to use for this request.

        For ``"agent"`` initiator, returns the same client as the most recent
        ``"user"`` request. For ``"user"`` initiator, selects based on strategy
        and increments the local usage counter when local tracking is enabled.
        """
        with self._lock:
            if initiator == "agent" and self._last_user_account is not None:
                return self._last_user_account.client

            acct = self._select_by_strategy()
            self._last_user_account = acct

            if self._local_tracking:
                acct.premium_used += 1
                if acct.premium_limit is not None and acct.premium_used >= acct.premium_limit:
                    acct.exhausted = True
                    logger.info(
                        "Account %s reached quota limit (%d/%d) via local tracking",
                        acct.username, acct.premium_used, acct.premium_limit,
                    )

            return acct.client

    def mark_exhausted(self, client: CopilotClient) -> None:
        """Mark the account associated with *client* as exhausted."""
        with self._lock:
            for acct in self._accounts:
                if acct.client is client:
                    acct.exhausted = True
                    logger.warning("Account %s marked as exhausted", acct.username)
                    break

    def get_fallback_client(self, failed_client: CopilotClient) -> CopilotClient | None:
        """Return the next non-exhausted client after marking *failed_client* exhausted."""
        self.mark_exhausted(failed_client)
        with self._lock:
            available = [a for a in self._accounts if not a.exhausted]
            if not available:
                return None
            return available[0].client

    def refresh_quota(self, acct: AccountInfo) -> None:
        """Fetch premium request usage from the GitHub billing API."""
        if time.time() - acct.last_quota_check < QUOTA_REFRESH_INTERVAL:
            return
        try:
            r = httpx.get(
                f"https://api.github.com/users/{acct.username}/settings/billing/premium_request/usage",
                headers={
                    "authorization": f"token {acct.token}",
                    "accept": "application/vnd.github+json",
                    "x-github-api-version": "2026-03-10",
                },
                timeout=30,
            )
            if r.status_code == 200:
                data = r.json()
                total = sum(
                    item.get("netQuantity", 0)
                    for item in data.get("usageItems", [])
                )
                acct.premium_used = int(total)
                acct.last_quota_check = time.time()
                if acct.premium_limit is not None and acct.premium_used >= acct.premium_limit:
                    acct.exhausted = True
                    logger.info(
                        "Account %s reached quota limit (%d/%d)",
                        acct.username, acct.premium_used, acct.premium_limit,
                    )
            else:
                logger.debug(
                    "Quota API returned %d for %s, skipping", r.status_code, acct.username
                )
        except Exception:
            logger.debug("Failed to fetch quota for %s", acct.username, exc_info=True)
        acct.last_quota_check = time.time()

    def refresh_all_quotas(self) -> None:
        """Refresh quota data for all accounts."""
        for acct in self._accounts:
            self.refresh_quota(acct)

    def _sync_usage_if_needed(self, available: list[AccountInfo]) -> None:
        """Sync usage data from billing API unless local tracking is active."""
        if self._local_tracking:
            return
        for acct in available:
            self.refresh_quota(acct)

    def _select_by_strategy(self) -> AccountInfo:
        """Apply the configured rotation strategy. Must be called with lock held."""
        available = [a for a in self._accounts if not a.exhausted]
        if not available:
            raise RuntimeError(
                "All accounts have exhausted their premium request quota."
            )

        # Sync usage for usage-based strategies (no-op if local tracking)
        if self._strategy in ("max-usage", "min-usage"):
            self._sync_usage_if_needed(available)
            # Re-filter after refresh (some may have become exhausted)
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

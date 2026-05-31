"""Unit tests for AccountManager rotation strategies."""

import time
from unittest.mock import MagicMock, patch

import pytest

from lib.account_manager import AccountInfo, AccountManager

pytestmark = pytest.mark.asyncio


def _make_manager(n: int = 3, strategy: str = "round-robin"):
    """Create an AccountManager with *n* mock accounts."""
    accounts = [(f"token_{i}", f"user_{i}") for i in range(n)]
    with patch("lib.account_manager.CopilotTokenManager") as MockTM, \
         patch("lib.account_manager.CopilotClient") as MockClient:
        MockTM.side_effect = lambda t: MagicMock(github_token=t)
        MockClient.side_effect = lambda tm: MagicMock(name=f"client_{tm.github_token}")
        mgr = AccountManager(accounts, strategy=strategy)
    return mgr


def _acct_for_client(mgr: AccountManager, client) -> AccountInfo:
    """Return the AccountInfo that owns *client*."""
    for acct in mgr._accounts:
        if acct.client is client:
            return acct
    raise ValueError("client not found")


class TestRoundRobin:
    async def test_cycles_through_accounts(self):
        mgr = _make_manager(3, "round-robin")
        clients = [await mgr.get_client("user") for _ in range(6)]
        # Should cycle: 0, 1, 2, 0, 1, 2
        assert clients[0] is clients[3]
        assert clients[1] is clients[4]
        assert clients[2] is clients[5]
        assert clients[0] is not clients[1]

    async def test_skips_unavailable(self):
        mgr = _make_manager(3, "round-robin")
        mgr._accounts[1].unavailable_until = time.time() + 60
        clients = [await mgr.get_client("user") for _ in range(4)]
        usernames = [
            a.username for a in mgr._accounts
            if a.client in clients
        ]
        assert "user_1" not in usernames

    async def test_all_unavailable_raises(self):
        mgr = _make_manager(2, "round-robin")
        for a in mgr._accounts:
            a.unavailable_until = time.time() + 60
        with pytest.raises(RuntimeError, match="unavailable"):
            await mgr.get_client("user")


class TestLeastUtilized:
    async def test_picks_lowest_live_utilization(self):
        mgr = _make_manager(3, "least-utilized")
        mgr._accounts[0].utilization = 0.6
        mgr._accounts[1].utilization = 0.2
        mgr._accounts[2].utilization = 0.4
        client = await mgr.get_client("user")
        assert client is mgr._accounts[1].client

    async def test_skips_unavailable(self):
        mgr = _make_manager(3, "least-utilized")
        mgr._accounts[0].utilization = 0.1
        mgr._accounts[0].unavailable_until = time.time() + 60
        mgr._accounts[1].utilization = 0.5
        mgr._accounts[2].utilization = 0.3
        client = await mgr.get_client("user")
        assert client is mgr._accounts[2].client

    async def test_legacy_strategies_map_to_least_utilized(self):
        assert _make_manager(1, "max-usage").strategy == "least-utilized"
        assert _make_manager(1, "min-usage").strategy == "least-utilized"

    async def test_no_utilization_signal_round_robins(self):
        mgr = _make_manager(3, "least-utilized")
        clients = [await mgr.get_client("user") for _ in range(4)]
        assert clients[0] is clients[3]
        assert clients[0] is not clients[1]


class TestAgentStickiness:
    async def test_agent_returns_last_user_client(self):
        mgr = _make_manager(3, "round-robin")
        user_client = await mgr.get_client("user")
        agent_client = await mgr.get_client("agent")
        assert agent_client is user_client

    async def test_agent_does_not_advance_rotation(self):
        mgr = _make_manager(3, "round-robin")
        c1 = await mgr.get_client("user")   # account 0
        await mgr.get_client("agent")         # still account 0
        await mgr.get_client("agent")         # still account 0
        c2 = await mgr.get_client("user")   # account 1
        assert c1 is not c2

    async def test_agent_before_any_user_gets_first_account(self):
        mgr = _make_manager(3, "round-robin")
        # No user request yet — agent should still get a client
        client = await mgr.get_client("agent")
        assert client is mgr._accounts[0].client


class TestSingleAccount:
    async def test_single_account_works(self):
        mgr = _make_manager(1, "round-robin")
        clients = [await mgr.get_client("user") for _ in range(3)]
        assert all(c is clients[0] for c in clients)

    async def test_single_account_unavailable_raises(self):
        mgr = _make_manager(1, "round-robin")
        mgr._accounts[0].unavailable_until = time.time() + 60
        with pytest.raises(RuntimeError, match="unavailable"):
            await mgr.get_client("user")


class TestExhaustionDetection:
    async def test_mark_exhausted_sidelines_temporarily(self):
        mgr = _make_manager(3, "round-robin")
        target = mgr._accounts[1]
        await mgr.mark_exhausted(target.client)
        assert target.is_available() is False
        assert target.unavailable_until is not None

    async def test_get_fallback_client(self):
        mgr = _make_manager(3, "round-robin")
        failed = mgr._accounts[0].client
        fallback = await mgr.get_fallback_client(failed)
        assert fallback is not None
        assert fallback is not failed
        assert mgr._accounts[0].is_available() is False
        assert mgr._accounts[0].unavailable_until is not None

    async def test_get_fallback_returns_none_when_all_unavailable(self):
        mgr = _make_manager(2, "round-robin")
        mgr._accounts[1].unavailable_until = time.time() + 60
        result = await mgr.get_fallback_client(mgr._accounts[0].client)
        assert result is None

    async def test_get_fallback_sidelines_even_without_fallback(self):
        # When every other account is already sidelined, the failed account
        # must still be sidelined so that subsequent requests back off rather
        # than hammering it repeatedly.
        mgr = _make_manager(2, "round-robin")
        mgr._accounts[1].unavailable_until = time.time() + 60
        result = await mgr.get_fallback_client(mgr._accounts[0].client)
        assert result is None
        assert mgr._accounts[0].is_available() is False
        assert mgr._accounts[0].unavailable_until is not None

    async def test_sidelined_account_recovers_after_backoff(self):
        # Use a 0-second back-off so the sidelined account is immediately
        # eligible again on the next selection.
        accounts = [(f"token_{i}", f"user_{i}") for i in range(2)]
        with patch("lib.account_manager.CopilotTokenManager") as MockTM, \
             patch("lib.account_manager.CopilotClient") as MockClient:
            MockTM.side_effect = lambda t: MagicMock(github_token=t)
            MockClient.side_effect = lambda tm: MagicMock(name=f"client_{tm.github_token}")
            mgr = AccountManager(accounts, strategy="round-robin",
                                 rate_limit_backoff_seconds=0)
        await mgr.mark_exhausted(mgr._accounts[0].client)
        # Both accounts should be available again immediately
        assert mgr._accounts[0].is_available() is True

    async def test_skips_sidelined_during_backoff(self):
        mgr = _make_manager(2, "round-robin")
        await mgr.mark_exhausted(mgr._accounts[0].client)
        # Within backoff window, only account 1 should be picked
        for _ in range(4):
            client = await mgr.get_client("user")
            assert client is mgr._accounts[1].client


class TestUsageTracking:
    async def test_record_usage_is_noop_for_removed_local_quota_model(self):
        mgr = _make_manager(2, "least-utilized")
        client = await mgr.get_client("user")
        acct = _acct_for_client(mgr, client)

        await mgr.record_usage(client, "claude-opus-4.6")

        assert acct.is_available() is True
        assert not hasattr(acct, "premium_used")
        assert not hasattr(acct, "premium_limit")


class TestValidation:
    async def test_empty_accounts_raises(self):
        with pytest.raises(ValueError, match="At least one account"):
            AccountManager([], strategy="round-robin")

    async def test_invalid_strategy_raises(self):
        with pytest.raises(ValueError, match="Unknown strategy"):
            _make_manager(1, "bad-strategy")

"""Unit tests for AccountManager rotation strategies."""

from unittest.mock import MagicMock, patch

import pytest

from lib.account_manager import AccountInfo, AccountManager


def _make_manager(n: int = 3, strategy: str = "round-robin", quota_limit=None):
    """Create an AccountManager with *n* mock accounts."""
    accounts = [(f"token_{i}", f"user_{i}") for i in range(n)]
    with patch("lib.account_manager.CopilotTokenManager") as MockTM, \
         patch("lib.account_manager.CopilotClient") as MockClient:
        MockTM.side_effect = lambda t: MagicMock(github_token=t)
        MockClient.side_effect = lambda tm: MagicMock(name=f"client_{tm.github_token}")
        mgr = AccountManager(accounts, strategy=strategy, quota_limit=quota_limit)
    return mgr


class TestRoundRobin:
    def test_cycles_through_accounts(self):
        mgr = _make_manager(3, "round-robin")
        clients = [mgr.get_client("user") for _ in range(6)]
        # Should cycle: 0, 1, 2, 0, 1, 2
        assert clients[0] is clients[3]
        assert clients[1] is clients[4]
        assert clients[2] is clients[5]
        assert clients[0] is not clients[1]

    def test_skips_exhausted(self):
        mgr = _make_manager(3, "round-robin")
        mgr._accounts[1].exhausted = True
        clients = [mgr.get_client("user") for _ in range(4)]
        usernames = [
            a.username for a in mgr._accounts
            if a.client in clients
        ]
        assert "user_1" not in usernames

    def test_all_exhausted_raises(self):
        mgr = _make_manager(2, "round-robin")
        for a in mgr._accounts:
            a.exhausted = True
        with pytest.raises(RuntimeError, match="exhausted"):
            mgr.get_client("user")


class TestMaxUsage:
    def test_picks_highest_used(self):
        mgr = _make_manager(3, "max-usage")
        mgr._accounts[0].premium_used = 10
        mgr._accounts[1].premium_used = 50
        mgr._accounts[2].premium_used = 30
        client = mgr.get_client("user")
        assert client is mgr._accounts[1].client

    def test_skips_exhausted(self):
        mgr = _make_manager(3, "max-usage")
        mgr._accounts[0].premium_used = 10
        mgr._accounts[1].premium_used = 50
        mgr._accounts[1].exhausted = True
        mgr._accounts[2].premium_used = 30
        client = mgr.get_client("user")
        assert client is mgr._accounts[2].client


class TestMinUsage:
    def test_picks_lowest_used(self):
        mgr = _make_manager(3, "min-usage")
        mgr._accounts[0].premium_used = 10
        mgr._accounts[1].premium_used = 50
        mgr._accounts[2].premium_used = 30
        client = mgr.get_client("user")
        assert client is mgr._accounts[0].client

    def test_skips_exhausted(self):
        mgr = _make_manager(3, "min-usage")
        mgr._accounts[0].premium_used = 10
        mgr._accounts[0].exhausted = True
        mgr._accounts[1].premium_used = 50
        mgr._accounts[2].premium_used = 30
        client = mgr.get_client("user")
        assert client is mgr._accounts[2].client


class TestAgentStickiness:
    def test_agent_returns_last_user_client(self):
        mgr = _make_manager(3, "round-robin")
        user_client = mgr.get_client("user")
        agent_client = mgr.get_client("agent")
        assert agent_client is user_client

    def test_agent_does_not_advance_rotation(self):
        mgr = _make_manager(3, "round-robin")
        c1 = mgr.get_client("user")   # account 0
        mgr.get_client("agent")         # still account 0
        mgr.get_client("agent")         # still account 0
        c2 = mgr.get_client("user")   # account 1
        assert c1 is not c2

    def test_agent_before_any_user_gets_first_account(self):
        mgr = _make_manager(3, "round-robin")
        # No user request yet — agent should still get a client
        client = mgr.get_client("agent")
        assert client is mgr._accounts[0].client


class TestSingleAccount:
    def test_single_account_works(self):
        mgr = _make_manager(1, "round-robin")
        clients = [mgr.get_client("user") for _ in range(3)]
        assert all(c is clients[0] for c in clients)

    def test_single_account_exhausted_raises(self):
        mgr = _make_manager(1, "round-robin")
        mgr._accounts[0].exhausted = True
        with pytest.raises(RuntimeError, match="exhausted"):
            mgr.get_client("user")


class TestExhaustionDetection:
    def test_mark_exhausted(self):
        mgr = _make_manager(3, "round-robin")
        target = mgr._accounts[1]
        mgr.mark_exhausted(target.client)
        assert target.exhausted is True

    def test_get_fallback_client(self):
        mgr = _make_manager(3, "round-robin")
        failed = mgr._accounts[0].client
        fallback = mgr.get_fallback_client(failed)
        assert fallback is not None
        assert fallback is not failed
        assert mgr._accounts[0].exhausted is True

    def test_get_fallback_returns_none_when_all_exhausted(self):
        mgr = _make_manager(2, "round-robin")
        mgr._accounts[1].exhausted = True
        result = mgr.get_fallback_client(mgr._accounts[0].client)
        assert result is None


class TestQuotaLimit:
    def test_quota_limit_applied_to_all_accounts(self):
        mgr = _make_manager(3, "round-robin", quota_limit=100)
        for a in mgr._accounts:
            assert a.premium_limit == 100

    def test_no_quota_limit_by_default(self):
        mgr = _make_manager(3, "round-robin")
        for a in mgr._accounts:
            assert a.premium_limit is None


class TestValidation:
    def test_empty_accounts_raises(self):
        with pytest.raises(ValueError, match="At least one account"):
            AccountManager([], strategy="round-robin")

    def test_invalid_strategy_raises(self):
        with pytest.raises(ValueError, match="Unknown strategy"):
            _make_manager(1, "bad-strategy")

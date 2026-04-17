import pytest
import pytest_asyncio
pytestmark = pytest.mark.asyncio

"""Unit tests for AccountManager rotation strategies."""

from unittest.mock import MagicMock, patch

import pytest

from lib.account_manager import AccountInfo, AccountManager


def _make_manager(n: int = 3, strategy: str = "round-robin", quota_limit=None,
                   plan="pro"):
    """Create an AccountManager with *n* mock accounts."""
    accounts = [(f"token_{i}", f"user_{i}") for i in range(n)]
    with patch("lib.account_manager.CopilotTokenManager") as MockTM, \
         patch("lib.account_manager.CopilotClient") as MockClient:
        MockTM.side_effect = lambda t: MagicMock(github_token=t)
        MockClient.side_effect = lambda tm: MagicMock(name=f"client_{tm.github_token}")
        mgr = AccountManager(accounts, strategy=strategy, quota_limit=quota_limit,
                             plan=plan)
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

    async def test_skips_exhausted(self):
        mgr = _make_manager(3, "round-robin")
        mgr._accounts[1].exhausted = True
        clients = [await mgr.get_client("user") for _ in range(4)]
        usernames = [
            a.username for a in mgr._accounts
            if a.client in clients
        ]
        assert "user_1" not in usernames

    async def test_all_exhausted_raises(self):
        mgr = _make_manager(2, "round-robin")
        for a in mgr._accounts:
            a.exhausted = True
        with pytest.raises(RuntimeError, match="exhausted"):
            await mgr.get_client("user")


class TestMaxUsage:
    async def test_picks_highest_used(self):
        mgr = _make_manager(3, "max-usage")
        mgr._accounts[0].premium_used = 10
        mgr._accounts[1].premium_used = 50
        mgr._accounts[2].premium_used = 30
        client = await mgr.get_client("user")
        assert client is mgr._accounts[1].client

    async def test_skips_exhausted(self):
        mgr = _make_manager(3, "max-usage")
        mgr._accounts[0].premium_used = 10
        mgr._accounts[1].premium_used = 50
        mgr._accounts[1].exhausted = True
        mgr._accounts[2].premium_used = 30
        client = await mgr.get_client("user")
        assert client is mgr._accounts[2].client


class TestMinUsage:
    async def test_picks_lowest_used(self):
        mgr = _make_manager(3, "min-usage")
        mgr._accounts[0].premium_used = 10
        mgr._accounts[1].premium_used = 50
        mgr._accounts[2].premium_used = 30
        client = await mgr.get_client("user")
        assert client is mgr._accounts[0].client

    async def test_skips_exhausted(self):
        mgr = _make_manager(3, "min-usage")
        mgr._accounts[0].premium_used = 10
        mgr._accounts[0].exhausted = True
        mgr._accounts[1].premium_used = 50
        mgr._accounts[2].premium_used = 30
        client = await mgr.get_client("user")
        assert client is mgr._accounts[2].client


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

    async def test_single_account_exhausted_raises(self):
        mgr = _make_manager(1, "round-robin")
        mgr._accounts[0].exhausted = True
        with pytest.raises(RuntimeError, match="exhausted"):
            await mgr.get_client("user")


class TestExhaustionDetection:
    async def test_mark_exhausted(self):
        mgr = _make_manager(3, "round-robin")
        target = mgr._accounts[1]
        await mgr.mark_exhausted(target.client)
        assert target.exhausted is True

    async def test_get_fallback_client(self):
        mgr = _make_manager(3, "round-robin")
        failed = mgr._accounts[0].client
        fallback = await mgr.get_fallback_client(failed)
        assert fallback is not None
        assert fallback is not failed
        assert mgr._accounts[0].exhausted is True

    async def test_get_fallback_returns_none_when_all_exhausted(self):
        mgr = _make_manager(2, "round-robin")
        mgr._accounts[1].exhausted = True
        result = await mgr.get_fallback_client(mgr._accounts[0].client)
        assert result is None


class TestQuotaLimit:
    async def test_quota_limit_applied_to_all_accounts(self):
        mgr = _make_manager(3, "round-robin", quota_limit=100)
        for a in mgr._accounts:
            assert a.premium_limit == 100

    async def test_quota_limit_defaults_from_plan(self):
        mgr = _make_manager(3, "round-robin")  # default plan is "pro"
        for a in mgr._accounts:
            assert a.premium_limit == 300  # pro plan default


class TestUsageTracking:
    async def test_record_usage_increments_with_multiplier(self):
        mgr = _make_manager(2, "max-usage")
        client = await mgr.get_client("user")
        await mgr.record_usage(client, "claude-opus-4.7")  # 3x
        assert _acct_for_client(mgr, client).premium_used == 3.0

    async def test_record_usage_zero_multiplier_model(self):
        mgr = _make_manager(2, "max-usage")
        client = await mgr.get_client("user")
        await mgr.record_usage(client, "gpt-4o")  # 0x
        assert _acct_for_client(mgr, client).premium_used == 0

    async def test_record_usage_fractional_multiplier(self):
        mgr = _make_manager(2, "max-usage")
        client = await mgr.get_client("user")
        await mgr.record_usage(client, "claude-haiku-4.5")  # 0.33x
        assert _acct_for_client(mgr, client).premium_used == pytest.approx(0.33)

    async def test_record_usage_unknown_model_defaults_to_1(self):
        mgr = _make_manager(2, "max-usage")
        client = await mgr.get_client("user")
        await mgr.record_usage(client, "some-future-model")
        assert _acct_for_client(mgr, client).premium_used == 1.0

    async def test_record_usage_prefix_match(self):
        mgr = _make_manager(2, "max-usage")
        client = await mgr.get_client("user")
        await mgr.record_usage(client, "gpt-4o-2024-07-18")  # matches gpt-4o → 0x
        assert _acct_for_client(mgr, client).premium_used == 0

    async def test_exhausts_at_quota_limit(self):
        mgr = _make_manager(2, "max-usage", quota_limit=5)
        client = await mgr.get_client("user")
        acct = _acct_for_client(mgr, client)
        await mgr.record_usage(client, "claude-opus-4.7")  # 3x → 3
        client = await mgr.get_client("user")  # still same account (3 < 5)
        await mgr.record_usage(client, "claude-opus-4.7")  # 3x → 6 ≥ 5 → exhausted
        assert acct.exhausted is True
        # Next user request should go to the other account
        client2 = await mgr.get_client("user")
        assert client2 is not client

    async def test_free_plan_all_models_cost_1(self):
        mgr = _make_manager(2, "max-usage", plan="free")
        client = await mgr.get_client("user")
        await mgr.record_usage(client, "gpt-4o")  # 0x on paid, 1x on free
        assert _acct_for_client(mgr, client).premium_used == 1.0

    async def test_paid_plan_included_models_cost_0(self):
        mgr = _make_manager(2, "max-usage", plan="pro")
        client = await mgr.get_client("user")
        await mgr.record_usage(client, "gpt-4o")  # 0x on paid
        assert _acct_for_client(mgr, client).premium_used == 0


class TestValidation:
    async def test_empty_accounts_raises(self):
        with pytest.raises(ValueError, match="At least one account"):
            AccountManager([], strategy="round-robin")

    async def test_invalid_strategy_raises(self):
        with pytest.raises(ValueError, match="Unknown strategy"):
            _make_manager(1, "bad-strategy")

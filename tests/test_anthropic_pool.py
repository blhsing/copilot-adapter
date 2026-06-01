"""Tests for the Anthropic-pool retrofit: 429 reset-header parsing,
conv-key derivation, per-backend stickiness, and conv-cache eviction.

These tests don't make any network calls — they construct
:class:`AccountManager` with fake backend-tagged dicts (the manager doesn't
care that the embedded clients are unused) and exercise the selection logic
directly.
"""

import asyncio
import time

import pytest

from lib import server as server_mod
from lib.account_manager import (
    AccountInfo,
    AccountManager,
    CONV_STICKY_TTL_SECONDS,
)


# ---------------------------------------------------------------------------
# 429 reset-header parsing
# ---------------------------------------------------------------------------

class TestParseAnthropicReset:
    def test_unified_reset(self):
        ts = time.time() + 600
        h = {"anthropic-ratelimit-unified-reset": str(ts)}
        assert abs(server_mod._parse_anthropic_reset_utc(h) - ts) < 1

    def test_requests_reset_iso(self):
        # Anthropic emits ISO 8601 with a trailing Z for UTC.
        from datetime import datetime, timezone, timedelta
        future = datetime.now(timezone.utc) + timedelta(minutes=10)
        h = {"anthropic-ratelimit-requests-reset": future.strftime("%Y-%m-%dT%H:%M:%SZ")}
        parsed = server_mod._parse_anthropic_reset_utc(h)
        assert parsed is not None
        assert abs(parsed - future.timestamp()) < 2

    def test_retry_after_seconds(self):
        h = {"retry-after": "90"}
        parsed = server_mod._parse_anthropic_reset_utc(h)
        assert parsed is not None
        # Should be ~now + 90s.
        assert abs(parsed - (time.time() + 90)) < 2

    def test_no_reset_headers_returns_none(self):
        assert server_mod._parse_anthropic_reset_utc({}) is None

    def test_case_insensitive(self):
        ts = time.time() + 60
        h = {"Anthropic-RateLimit-Unified-Reset": str(ts)}
        parsed = server_mod._parse_anthropic_reset_utc(h)
        assert parsed is not None
        assert abs(parsed - ts) < 1


# ---------------------------------------------------------------------------
# Conv-key derivation
# ---------------------------------------------------------------------------

class TestDeriveConvKey:
    def test_anthropic_block_content(self):
        body = {
            "messages": [
                {"role": "user", "content": [
                    {"type": "text", "text": "hello, claude"},
                ]},
            ]
        }
        k1 = server_mod._derive_conv_key(body, "anthropic")
        assert k1 is not None
        # A follow-up turn (extra assistant + user) keeps the same key
        # because the first user message hasn't changed.
        body["messages"].append({"role": "assistant", "content": "hi!"})
        body["messages"].append({"role": "user", "content": "more"})
        k2 = server_mod._derive_conv_key(body, "anthropic")
        assert k1 == k2

    def test_openai_string_content(self):
        body = {"messages": [{"role": "user", "content": "open ai user msg"}]}
        k = server_mod._derive_conv_key(body, "openai")
        assert k is not None
        assert len(k) == 40  # sha1 hex digest length

    def test_provider_isolation(self):
        body = {"messages": [{"role": "user", "content": "same text"}]}
        k_a = server_mod._derive_conv_key(body, "anthropic")
        k_o = server_mod._derive_conv_key(body, "openai")
        # Same opening line under different providers must produce different
        # cache slots so an OpenAI and an Anthropic conversation never collide.
        assert k_a != k_o

    def test_no_user_message_returns_none(self):
        body = {"messages": [{"role": "system", "content": "boot"}]}
        assert server_mod._derive_conv_key(body, "anthropic") is None

    def test_missing_messages_returns_none(self):
        assert server_mod._derive_conv_key({}, "anthropic") is None


# ---------------------------------------------------------------------------
# Per-backend stickiness + conv cache (synthetic accounts)
# ---------------------------------------------------------------------------

def _make_dual_backend_manager() -> AccountManager:
    """Build an AccountManager with one Copilot and one Anthropic account.

    Uses tuple-style copilot dicts (no network) and a synthetic Anthropic
    entry with bogus tokens — the manager doesn't validate them at
    construction time, only on first use.
    """
    accounts = [
        {"token": "c1", "username": "co-user", "backend": "copilot"},
        {
            "username": "claude-user",
            "backend": "anthropic",
            "access_token": "ant-access",
            "refresh_token": "ant-refresh",
            "expires_at": time.time() + 3600,
        },
    ]
    return AccountManager(accounts, strategy="least-utilized")


class TestPerBackendSelection:
    @pytest.mark.asyncio
    async def test_prefer_anthropic_returns_anthropic(self):
        mgr = _make_dual_backend_manager()
        client = await mgr.get_client(initiator="user", prefer_backend="anthropic")
        assert mgr.get_backend(client) == "anthropic"

    @pytest.mark.asyncio
    async def test_prefer_anthropic_falls_back_to_copilot(self):
        mgr = _make_dual_backend_manager()
        # Sideline the Anthropic account.
        for a in mgr.accounts:
            if a.backend == "anthropic":
                a.unavailable_until = time.time() + 60
        client = await mgr.get_client(initiator="user", prefer_backend="anthropic")
        assert mgr.get_backend(client) == "copilot"

    @pytest.mark.asyncio
    async def test_has_available_per_backend(self):
        mgr = _make_dual_backend_manager()
        assert mgr.has_available("copilot")
        assert mgr.has_available("anthropic")
        for a in mgr.accounts:
            if a.backend == "anthropic":
                a.unavailable_until = time.time() + 60
        assert mgr.has_available("copilot")
        assert not mgr.has_available("anthropic")

    @pytest.mark.asyncio
    async def test_per_backend_stickiness(self):
        """Agent follow-up on Anthropic shouldn't pin to a Copilot account
        just because Copilot was the last backend used (and vice versa)."""
        mgr = _make_dual_backend_manager()
        # First Anthropic user call → sets _last_user_anthropic.
        client_a = await mgr.get_client(initiator="user", prefer_backend="anthropic")
        # First Copilot user call → sets _last_user_copilot.
        client_c = await mgr.get_client(initiator="user", prefer_backend="copilot")
        # Agent follow-up on Anthropic should resolve to the Anthropic account,
        # NOT the more-recently-touched Copilot account.
        client_agent = await mgr.get_client(
            initiator="agent", prefer_backend="anthropic",
        )
        assert client_agent is client_a
        # And the converse — agent follow-up on Copilot picks Copilot.
        client_agent_c = await mgr.get_client(
            initiator="agent", prefer_backend="copilot",
        )
        assert client_agent_c is client_c


class TestConvKeyStickiness:
    @pytest.mark.asyncio
    async def test_conv_key_pins_account_across_turns(self):
        mgr = _make_dual_backend_manager()
        first = await mgr.get_client(
            initiator="user", conv_key="abc123", prefer_backend="anthropic",
        )
        # Same conv_key should return the same client even when the strategy
        # would otherwise pick something else after marking it busy.
        for _ in range(5):
            again = await mgr.get_client(
                initiator="user", conv_key="abc123", prefer_backend="anthropic",
            )
            assert again is first

    @pytest.mark.asyncio
    async def test_conv_cache_purged_on_exhaustion(self):
        mgr = _make_dual_backend_manager()
        first = await mgr.get_client(initiator="user", conv_key="k1",
                                     prefer_backend="anthropic")
        await mgr.mark_exhausted_until(first, time.time() + 3600)
        # Conv-cache entry should be gone; next get_client falls through
        # to fresh selection (with Anthropic exhausted, ends up on Copilot).
        second = await mgr.get_client(initiator="user", conv_key="k1",
                                      prefer_backend="anthropic")
        assert second is not first
        assert mgr.get_backend(second) == "copilot"

    @pytest.mark.asyncio
    async def test_conv_cache_evicts_after_ttl(self):
        mgr = _make_dual_backend_manager()
        first = await mgr.get_client(initiator="user", conv_key="ttl-key")
        # Force the cache entry to look stale.
        key = (("ttl-key", mgr.get_backend(first)))
        acct, _ = mgr._conv_cache[key]
        mgr._conv_cache[key] = (acct, time.time() - CONV_STICKY_TTL_SECONDS - 60)
        # Next call should re-pin from scratch (may pick same or different).
        second = await mgr.get_client(initiator="user", conv_key="ttl-key")
        # Whatever account is now in the cache should have a fresh timestamp.
        key = ("ttl-key", mgr.get_backend(second))
        _, ts = mgr._conv_cache[key]
        assert ts > time.time() - 5

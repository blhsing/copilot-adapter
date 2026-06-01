"""Unit tests for resolve_github_tokens headless / copilot-less behavior."""

from unittest.mock import patch

from lib import auth


def test_no_tokens_non_interactive_not_required_returns_empty(monkeypatch):
    # No env token, empty cache, interactive disabled, not required.
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    with patch.object(auth, "_load_github_tokens", return_value=[]), \
         patch.object(auth, "device_flow_login") as mock_login:
        result = auth.resolve_github_tokens(
            None, interactive=False, required=False
        )
    assert result == []
    # Headless must never trigger the browser device flow.
    mock_login.assert_not_called()


def test_no_tokens_required_raises(monkeypatch):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    with patch.object(auth, "_load_github_tokens", return_value=[]), \
         patch.object(auth, "device_flow_login") as mock_login:
        try:
            auth.resolve_github_tokens(None, interactive=False, required=True)
            assert False, "expected RuntimeError"
        except RuntimeError:
            pass
    mock_login.assert_not_called()


def test_cached_token_still_resolves(monkeypatch):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    with patch.object(auth, "_load_github_tokens",
                      return_value=[{"github_token": "ghu_cached"}]), \
         patch.object(auth, "_validate_github_token", return_value="octocat"), \
         patch.object(auth, "device_flow_login") as mock_login:
        result = auth.resolve_github_tokens(None, interactive=False, required=False)
    assert result == [("ghu_cached", "octocat")]
    mock_login.assert_not_called()

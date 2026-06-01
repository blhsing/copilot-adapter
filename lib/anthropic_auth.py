"""Claude.ai OAuth (PKCE paste-back) + Anthropic OAuth token refresh.

Mirrors :mod:`lib.auth` but for real Anthropic Max-subscription accounts.

These tokens authenticate against ``api.anthropic.com`` directly (via
:class:`lib.anthropic_client.AnthropicClient`), independent of GitHub Copilot
— they're for pooling Claude Max subscriptions so a Claude Code session can
fall back across multiple accounts without routing those requests through Copilot.

The flow is paste-back rather than localhost-callback because claude.ai's
``redirect_uri`` allowlist is strict (only ``console.anthropic.com``) and
running a local callback server through firewalls / LocalSystem services
is brittle. The user opens the auth URL, approves, copies the ``code`` from
the redirect URL bar, and pastes it back.
"""

import asyncio
import base64
import hashlib
import json
import secrets
import time
import webbrowser
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode

import httpx

# These match what Claude Code itself uses — same client_id, same scopes, same
# redirect_uri. Anthropic does not (yet) hand out unique client_ids for
# third-party pooling tools, and reusing claude-cli's identifiers is what
# makes the OAuth tokens accepted by api.anthropic.com.
CLAUDE_CLIENT_ID = "9d1c250a-e61b-44d9-88ed-5944d1962f5e"
CLAUDE_AUTH_URL = "https://claude.ai/oauth/authorize"
CLAUDE_TOKEN_URL = "https://console.anthropic.com/v1/oauth/token"
CLAUDE_REDIRECT_URI = "https://console.anthropic.com/oauth/code/callback"
CLAUDE_SCOPES = "org:create_api_key user:profile user:inference"

ANTHROPIC_TOKENS_FILE = Path.home() / ".config" / "copilot-adapter" / "anthropic_tokens.json"


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------

def _load_anthropic_accounts() -> list[dict]:
    if ANTHROPIC_TOKENS_FILE.exists():
        data = json.loads(ANTHROPIC_TOKENS_FILE.read_text())
        return data.get("accounts", [])
    return []


def _save_anthropic_accounts(accounts: list[dict]) -> None:
    ANTHROPIC_TOKENS_FILE.parent.mkdir(parents=True, exist_ok=True)
    ANTHROPIC_TOKENS_FILE.write_text(json.dumps({"accounts": accounts}, indent=2))


def list_anthropic_accounts() -> list[dict]:
    """Return cached Anthropic accounts (without sensitive token fields masked)."""
    return [
        {
            "username": a.get("username", "unknown"),
            "expires_at": a.get("expires_at"),
        }
        for a in _load_anthropic_accounts()
    ]


def remove_anthropic_account(username: str) -> bool:
    accounts = _load_anthropic_accounts()
    filtered = [a for a in accounts if a.get("username") != username]
    if len(filtered) == len(accounts):
        return False
    _save_anthropic_accounts(filtered)
    return True


def update_anthropic_account_tokens(
    username: str, access_token: str, refresh_token: str, expires_at: float
) -> None:
    """Persist a rotated (access_token, refresh_token, expires_at) triple.

    Anthropic rotates the refresh token on every refresh, so we must write the
    new values back to disk — otherwise the cached refresh token becomes
    invalid after one refresh and the account has to be re-authenticated.
    """
    accounts = _load_anthropic_accounts()
    for acct in accounts:
        if acct.get("username") == username:
            acct["access_token"] = access_token
            acct["refresh_token"] = refresh_token
            acct["expires_at"] = expires_at
            _save_anthropic_accounts(accounts)
            return


# ---------------------------------------------------------------------------
# PKCE paste-back login
# ---------------------------------------------------------------------------

def _b64url(data: bytes) -> str:
    """Base64-URL-encode without padding (RFC 7636)."""
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _make_pkce_pair() -> tuple[str, str]:
    """Return (verifier, S256-challenge) suitable for an OAuth PKCE flow."""
    verifier = _b64url(secrets.token_bytes(32))
    challenge = _b64url(hashlib.sha256(verifier.encode("ascii")).digest())
    return verifier, challenge


def build_claude_auth_url() -> tuple[str, str, str]:
    """Return (auth_url, state, verifier) for a fresh PKCE login attempt."""
    verifier, challenge = _make_pkce_pair()
    state = _b64url(secrets.token_bytes(16))
    params = {
        "code": "true",
        "client_id": CLAUDE_CLIENT_ID,
        "response_type": "code",
        "redirect_uri": CLAUDE_REDIRECT_URI,
        "scope": CLAUDE_SCOPES,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
        "state": state,
    }
    return CLAUDE_AUTH_URL + "?" + urlencode(params), state, verifier


def exchange_claude_code(code: str, state: str, verifier: str) -> dict:
    """Exchange an authorization code for access + refresh tokens.

    Returns the raw JSON response (``access_token``, ``refresh_token``,
    ``expires_in``, etc.).
    """
    # Claude.ai's callback returns ``code#state`` — strip the ``#state`` half if present.
    if "#" in code:
        code = code.split("#", 1)[0]
    payload = {
        "grant_type": "authorization_code",
        "client_id": CLAUDE_CLIENT_ID,
        "code": code,
        "redirect_uri": CLAUDE_REDIRECT_URI,
        "code_verifier": verifier,
        "state": state,
    }
    r = httpx.post(CLAUDE_TOKEN_URL, json=payload, timeout=30)
    if r.status_code != 200:
        raise RuntimeError(
            f"Anthropic token exchange failed: {r.status_code} {r.text}"
        )
    return r.json()


def claude_login_interactive() -> dict | None:
    """Run the full claude.ai PKCE paste-back login flow against the terminal.

    Adds the resulting account to the Anthropic accounts cache and returns its
    summary, or ``None`` if the user aborted.
    """
    auth_url, state, verifier = build_claude_auth_url()

    print("\nClaude.ai login (PKCE paste-back):")
    print(f"\n  1. Open this URL in your browser:")
    print(f"     {auth_url}")
    print(f"\n  2. Approve access.")
    print(f"  3. After redirect, copy the 'code' query parameter "
          f"from the resulting URL (everything after 'code=').")
    print()

    try:
        webbrowser.open(auth_url)
    except Exception:
        pass

    code = input("Paste the code here: ").strip()
    if not code:
        print("No code provided, aborting.")
        return None

    try:
        token_resp = exchange_claude_code(code, state, verifier)
    except RuntimeError as exc:
        print(f"Error: {exc}")
        return None

    access_token = token_resp["access_token"]
    refresh_token = token_resp.get("refresh_token", "")
    expires_in = float(token_resp.get("expires_in", 3600))
    expires_at = time.time() + expires_in

    # Try to fetch the account identity for logging purposes. The /v1/oauth/profile
    # endpoint isn't documented but is exposed and used by Claude Code itself;
    # if it 404s we fall back to a tail-of-token identifier.
    username = "anthropic-" + access_token[-6:]
    try:
        prof = httpx.get(
            "https://api.anthropic.com/api/oauth/profile",
            headers={
                "authorization": f"Bearer {access_token}",
                "anthropic-beta": "oauth-2025-04-20",
            },
            timeout=15,
        )
        if prof.status_code == 200:
            data = prof.json()
            email = data.get("email") or data.get("account", {}).get("email")
            if email:
                username = email
    except Exception:
        pass

    accounts = _load_anthropic_accounts()
    accounts = [a for a in accounts if a.get("username") != username]
    accounts.append({
        "username": username,
        "access_token": access_token,
        "refresh_token": refresh_token,
        "expires_at": expires_at,
        "added_at": datetime.now(timezone.utc).isoformat(),
    })
    _save_anthropic_accounts(accounts)
    print(f"\nLogged in as {username}")
    return {"username": username}


# ---------------------------------------------------------------------------
# Token manager
# ---------------------------------------------------------------------------

class AnthropicTokenManager:
    """Manage one Anthropic OAuth access token, refreshing on expiry.

    Anthropic rotates the refresh token on every refresh — the new
    ``refresh_token`` returned in the response replaces the old one. We
    persist it back to the cache via the on_rotated callback so a service
    restart doesn't lose the freshly-rotated refresh token.

    Refresh runs under a single asyncio lock so concurrent callers don't
    issue parallel refreshes (which would invalidate each other since the
    refresh token is single-use after rotation).
    """

    def __init__(
        self,
        access_token: str,
        refresh_token: str,
        expires_at: float,
        *,
        on_rotated=None,
    ):
        self._access_token = access_token
        self._refresh_token = refresh_token
        self._expires_at = expires_at
        self._on_rotated = on_rotated
        self._lock = asyncio.Lock()

    @property
    def token_prefix(self) -> str:
        t = self._access_token
        return t[:8] if len(t) > 8 else t

    async def get_token(self) -> str:
        # Refresh 5 minutes before expiry, like Copilot.
        if self._access_token and time.time() < self._expires_at - 300:
            return self._access_token

        async with self._lock:
            if self._access_token and time.time() < self._expires_at - 300:
                return self._access_token
            await self._refresh()
            return self._access_token

    async def _refresh(self) -> None:
        if not self._refresh_token:
            raise RuntimeError(
                "Anthropic access token expired and no refresh token is available — "
                "re-run `claude-login`."
            )
        payload = {
            "grant_type": "refresh_token",
            "client_id": CLAUDE_CLIENT_ID,
            "refresh_token": self._refresh_token,
        }
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.post(CLAUDE_TOKEN_URL, json=payload)
        if r.status_code != 200:
            raise RuntimeError(
                f"Anthropic token refresh failed: {r.status_code} {r.text}"
            )
        data = r.json()
        self._access_token = data["access_token"]
        # Anthropic rotates the refresh token on every refresh.
        new_refresh = data.get("refresh_token")
        if new_refresh:
            self._refresh_token = new_refresh
        self._expires_at = time.time() + float(data.get("expires_in", 3600))
        if self._on_rotated is not None:
            try:
                self._on_rotated(self._access_token, self._refresh_token,
                                 self._expires_at)
            except Exception:
                # The callback persisting to disk shouldn't fail the refresh
                # itself; if it does, the in-memory token still works for this
                # process.
                pass


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def resolve_anthropic_accounts() -> list[dict]:
    """Return cached Anthropic accounts as account-manager-shaped dicts.

    Each entry includes the access_token, refresh_token, expires_at, and
    username — enough for ``AccountManager`` to
    construct an :class:`AnthropicClient` per account.
    """
    return _load_anthropic_accounts()

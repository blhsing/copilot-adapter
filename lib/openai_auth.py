"""ChatGPT (Codex) OAuth device flow + token refresh.

Mirrors :mod:`lib.anthropic_auth` but for ChatGPT Plus/Pro/Business accounts.
These tokens authenticate against ``chatgpt.com/backend-api/codex`` (via
:class:`lib.chatgpt_client.ChatGPTClient`) — the same endpoint the Codex CLI
uses — letting a Codex/Responses session pool multiple ChatGPT subscriptions
instead of burning Copilot premium quota.

The flow is the OpenAI-custom **device code** grant (RFC-8628-ish), the same
one ``codex login --device-auth`` uses: no localhost callback, the user opens
a URL and types a one-time code. Reusing the Codex CLI's ``client_id`` is what
makes the issued tokens accepted by the codex backend.

Protocol (verified against the Codex CLI source + live endpoints):
  1. POST /api/accounts/deviceauth/usercode  {client_id}
       -> {device_auth_id, user_code, interval}
  2. user opens https://auth.openai.com/codex/device and enters user_code
  3. poll POST /api/accounts/deviceauth/token {device_auth_id, user_code}
       2xx  -> {authorization_code, code_verifier, code_challenge}  (server PKCE)
       403/404 -> still pending
       else -> hard error
  4. POST /oauth/token (form-encoded) grant_type=authorization_code
       -> {id_token, access_token, refresh_token}
  5. refresh: POST /oauth/token (JSON!) {client_id, grant_type, refresh_token}
"""

import asyncio
import base64
import json
import time
import webbrowser
from datetime import datetime, timezone
from pathlib import Path

import httpx

# Same client_id the Codex CLI uses — required for the issued tokens to be
# accepted by chatgpt.com/backend-api/codex.
OPENAI_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
OPENAI_ISSUER = "https://auth.openai.com"
OPENAI_USERCODE_URL = OPENAI_ISSUER + "/api/accounts/deviceauth/usercode"
OPENAI_DEVICE_TOKEN_URL = OPENAI_ISSUER + "/api/accounts/deviceauth/token"
OPENAI_TOKEN_URL = OPENAI_ISSUER + "/oauth/token"
OPENAI_VERIFICATION_URI = OPENAI_ISSUER + "/codex/device"
OPENAI_DEVICE_REDIRECT_URI = OPENAI_ISSUER + "/deviceauth/callback"
# Mimic the Codex CLI so OpenAI's edge treats us as a legitimate device-flow
# client (reduces Cloudflare bot-challenge escalation).
OPENAI_USER_AGENT = "codex_cli_rs/0.20.0 (Windows 11; x64)"

CHATGPT_TOKENS_FILE = Path.home() / ".config" / "copilot-adapter" / "chatgpt_tokens.json"


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------

def _load_chatgpt_accounts() -> list[dict]:
    if CHATGPT_TOKENS_FILE.exists():
        data = json.loads(CHATGPT_TOKENS_FILE.read_text())
        return data.get("accounts", [])
    return []


def _save_chatgpt_accounts(accounts: list[dict]) -> None:
    CHATGPT_TOKENS_FILE.parent.mkdir(parents=True, exist_ok=True)
    CHATGPT_TOKENS_FILE.write_text(json.dumps({"accounts": accounts}, indent=2))


def list_chatgpt_accounts() -> list[dict]:
    return [
        {
            "username": a.get("username", "unknown"),
            "account_id": a.get("account_id"),
            "plan": a.get("plan"),
            "expires_at": a.get("expires_at"),
        }
        for a in _load_chatgpt_accounts()
    ]


def remove_chatgpt_account(username: str) -> bool:
    accounts = _load_chatgpt_accounts()
    filtered = [a for a in accounts if a.get("username") != username]
    if len(filtered) == len(accounts):
        return False
    _save_chatgpt_accounts(filtered)
    return True


def update_chatgpt_account_tokens(
    username: str, access_token: str, refresh_token: str, expires_at: float
) -> None:
    """Persist a rotated (access, refresh, expires) triple — refresh tokens rotate."""
    accounts = _load_chatgpt_accounts()
    for acct in accounts:
        if acct.get("username") == username:
            acct["access_token"] = access_token
            acct["refresh_token"] = refresh_token
            acct["expires_at"] = expires_at
            _save_chatgpt_accounts(accounts)
            return


def resolve_chatgpt_accounts() -> list[dict]:
    """Return cached ChatGPT accounts as account-manager-shaped dicts."""
    return _load_chatgpt_accounts()


# ---------------------------------------------------------------------------
# JWT helpers
# ---------------------------------------------------------------------------

def _decode_jwt_payload(jwt: str) -> dict:
    """Base64url-decode (no verify) a JWT's payload section into a dict."""
    if not jwt:
        return {}
    parts = jwt.split(".")
    if len(parts) < 2:
        return {}
    seg = parts[1]
    seg += "=" * (-len(seg) % 4)  # pad
    return json.loads(base64.urlsafe_b64decode(seg.encode("ascii")))


def _jwt_expiry(jwt: str) -> float | None:
    claims = _decode_jwt_payload(jwt)
    exp = claims.get("exp")
    return float(exp) if exp else None


def _identity_from_id_token(id_token: str) -> tuple[str | None, str | None, str | None]:
    """Return (account_id, email, plan) from the Codex id_token JWT claims."""
    claims = _decode_jwt_payload(id_token)
    auth = claims.get("https://api.openai.com/auth") or {}
    account_id = auth.get("chatgpt_account_id")
    plan = auth.get("chatgpt_plan_type")
    email = claims.get("email")
    if not email:
        profile = claims.get("https://api.openai.com/profile") or {}
        email = profile.get("email")
    return account_id, email, plan


# ---------------------------------------------------------------------------
# Device-code login
# ---------------------------------------------------------------------------

def _request_user_code() -> dict:
    r = httpx.post(
        OPENAI_USERCODE_URL,
        json={"client_id": OPENAI_CLIENT_ID},
        headers={"User-Agent": OPENAI_USER_AGENT, "originator": "codex_cli_rs",
                 "Accept": "application/json"},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()


def _exchange_authorization_code(authorization_code: str, code_verifier: str) -> dict:
    """Exchange the device authorization_code for tokens (form-encoded)."""
    data = {
        "grant_type": "authorization_code",
        "code": authorization_code,
        "redirect_uri": OPENAI_DEVICE_REDIRECT_URI,
        "client_id": OPENAI_CLIENT_ID,
        "code_verifier": code_verifier,
    }
    r = httpx.post(OPENAI_TOKEN_URL, data=data,
                   headers={"User-Agent": OPENAI_USER_AGENT}, timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f"ChatGPT token exchange failed: {r.status_code} {r.text}")
    return r.json()


def codex_login_interactive(*, plan: str | None = None) -> dict | None:
    """Run the full Codex device-code login flow against the terminal.

    Adds the resulting account to the ChatGPT accounts cache and returns its
    summary, or ``None`` on abort/failure.
    """
    try:
        uc = _request_user_code()
    except Exception as exc:  # noqa: BLE001
        print(f"Could not start ChatGPT login: {exc}\n"
              "OpenAI may be rate-limiting device-code requests (Cloudflare). "
              "Wait a few minutes and retry.")
        return None

    device_auth_id = uc["device_auth_id"]
    user_code = uc.get("user_code") or uc.get("usercode")
    try:
        interval = int(uc.get("interval", 5))
    except (TypeError, ValueError):
        interval = 5
    if interval <= 0:
        interval = 5

    print("\nChatGPT (Codex) device login:")
    print(f"\n  1. Open this URL in your browser:")
    print(f"     {OPENAI_VERIFICATION_URI}")
    print(f"\n  2. Sign in and enter this one-time code: {user_code}")
    print("\n  (Enable 'device code authorization' in ChatGPT > Settings > "
          "Security first, or the Continue button stays greyed out.)\n")
    try:
        webbrowser.open(OPENAI_VERIFICATION_URI)
    except Exception:
        pass

    print("Waiting for approval (Ctrl-C to abort)...")
    deadline = time.time() + 15 * 60
    token_resp = None
    while time.time() < deadline:
        time.sleep(interval)
        try:
            r = httpx.post(
                OPENAI_DEVICE_TOKEN_URL,
                json={"device_auth_id": device_auth_id, "user_code": user_code},
                headers={"User-Agent": OPENAI_USER_AGENT, "originator": "codex_cli_rs",
                         "Accept": "application/json"},
                timeout=30,
            )
        except Exception:
            continue
        if r.status_code // 100 == 2:
            code_resp = r.json()
            auth_code = code_resp.get("authorization_code")
            verifier = code_resp.get("code_verifier")
            if not auth_code or not verifier:
                print("Device approval returned no authorization code.")
                return None
            try:
                token_resp = _exchange_authorization_code(auth_code, verifier)
            except RuntimeError as exc:
                print(f"Error: {exc}")
                return None
            break
        if r.status_code in (403, 404):
            continue  # still pending
        print(f"Device auth failed: {r.status_code} {r.text}")
        return None

    if token_resp is None:
        print("Device code expired before approval.")
        return None

    access_token = token_resp["access_token"]
    refresh_token = token_resp.get("refresh_token", "")
    id_token = token_resp.get("id_token", "")
    account_id, email, jwt_plan = _identity_from_id_token(id_token)
    expires_at = _jwt_expiry(access_token) or (time.time() + 3000)
    username = email or ("chatgpt-" + (account_id or access_token[-6:]))

    accounts = _load_chatgpt_accounts()
    accounts = [a for a in accounts if a.get("username") != username]
    accounts.append({
        "username": username,
        "access_token": access_token,
        "refresh_token": refresh_token,
        "account_id": account_id,
        "expires_at": expires_at,
        "plan": plan or jwt_plan,
        "added_at": datetime.now(timezone.utc).isoformat(),
    })
    _save_chatgpt_accounts(accounts)
    print(f"\nLogged in as {username} (plan: {plan or jwt_plan})")
    return {"username": username, "account_id": account_id, "plan": plan or jwt_plan}


# ---------------------------------------------------------------------------
# Token manager
# ---------------------------------------------------------------------------

class OpenAITokenManager:
    """Manage one ChatGPT OAuth access token, refreshing on expiry.

    Refresh tokens rotate on every refresh, so the new ``refresh_token`` is
    persisted via the on_rotated callback. The refresh request body MUST be
    JSON with exactly {client_id, grant_type, refresh_token} and NO scope —
    form-encoding or a scope param yields 401 invalid_grant.
    """

    def __init__(self, access_token: str, refresh_token: str, expires_at: float,
                 *, on_rotated=None):
        self._access_token = access_token
        self._refresh_token = refresh_token
        self._expires_at = expires_at or 0
        self._on_rotated = on_rotated
        self._lock = asyncio.Lock()

    @property
    def token_prefix(self) -> str:
        t = self._access_token
        return t[:8] if len(t) > 8 else t

    async def get_token(self) -> str:
        if self._access_token and time.time() < self._expires_at - 60:
            return self._access_token
        async with self._lock:
            if self._access_token and time.time() < self._expires_at - 60:
                return self._access_token
            await self._refresh()
            return self._access_token

    async def _refresh(self) -> None:
        if not self._refresh_token:
            raise RuntimeError(
                "ChatGPT access token expired and no refresh token is available — "
                "re-run `codex-login`."
            )
        payload = {
            "client_id": OPENAI_CLIENT_ID,
            "grant_type": "refresh_token",
            "refresh_token": self._refresh_token,
        }
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.post(OPENAI_TOKEN_URL, json=payload,
                                  headers={"User-Agent": OPENAI_USER_AGENT,
                                           "originator": "codex_cli_rs"})
        if r.status_code != 200:
            raise RuntimeError(f"ChatGPT token refresh failed: {r.status_code} {r.text}")
        data = r.json()
        self._access_token = data["access_token"]
        new_refresh = data.get("refresh_token")
        if new_refresh:
            self._refresh_token = new_refresh
        exp = _jwt_expiry(self._access_token)
        self._expires_at = exp if exp else (time.time() + 3000)
        if self._on_rotated is not None:
            try:
                self._on_rotated(self._access_token, self._refresh_token,
                                 self._expires_at)
            except Exception:
                pass

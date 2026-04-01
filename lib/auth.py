"""GitHub Copilot device-flow OAuth and token management."""

import json
import threading
import time
import asyncio
import webbrowser
from pathlib import Path

import httpx

GITHUB_CLIENT_ID = "Iv1.b507a08c87ecfe98"
TOKEN_FILE = Path.home() / ".config" / "copilot-api" / "token.json"
TOKENS_FILE = Path.home() / ".config" / "copilot-api" / "tokens.json"

HEADERS_BASE = {
    "accept": "application/json",
    "content-type": "application/json",
    "editor-version": "vscode/1.95.0",
    "editor-plugin-version": "copilot-chat/0.23.0",
    "user-agent": "GitHubCopilotChat/0.23.0",
}


def _validate_github_token(token: str) -> str | None:
    """Validate a GitHub token and return the username, or None if invalid."""
    r = httpx.get(
        "https://api.github.com/user",
        headers={"authorization": f"token {token}", "accept": "application/json"},
    )
    if r.status_code == 200:
        return r.json().get("login", "unknown")
    return None


def _save_github_tokens(accounts: list[dict]) -> None:
    TOKENS_FILE.parent.mkdir(parents=True, exist_ok=True)
    TOKENS_FILE.write_text(json.dumps({"accounts": accounts}, indent=2))


def _load_github_tokens() -> list[dict]:
    """Load cached accounts. Auto-migrates old single-token format."""
    if TOKENS_FILE.exists():
        data = json.loads(TOKENS_FILE.read_text())
        return data.get("accounts", [])

    # Migrate from old single-token format
    if TOKEN_FILE.exists():
        old_data = json.loads(TOKEN_FILE.read_text())
        old_token = old_data.get("github_token")
        if old_token:
            username = _validate_github_token(old_token)
            if username:
                accounts = [{"github_token": old_token, "username": username}]
                _save_github_tokens(accounts)
                TOKEN_FILE.unlink()
                return accounts
        TOKEN_FILE.unlink()

    return []


def get_cached_account_meta() -> dict[str, dict]:
    """Return a mapping of token → {plan, quota_limit} from the token cache."""
    meta: dict[str, dict] = {}
    for acct in _load_github_tokens():
        token = acct.get("github_token", "")
        entry: dict = {}
        if "plan" in acct:
            entry["plan"] = acct["plan"]
        if "quota_limit" in acct:
            entry["quota_limit"] = acct["quota_limit"]
        if entry:
            meta[token] = entry
    return meta


def resolve_github_tokens(explicit_tokens: list[str] | None = None) -> list[tuple[str, str]]:
    """Return a list of (token, username) tuples from all available sources.

    Lookup order: *explicit_tokens* arg (from ``--github-token`` flags or
    ``COPILOT_ADAPTER_GITHUB_TOKEN`` env var) > ``GITHUB_TOKEN`` env var >
    cached token file > interactive device-flow OAuth.

    Each token is validated. Duplicates (by token value) are removed.
    """
    import os

    seen_tokens: set[str] = set()
    result: list[tuple[str, str]] = []

    def _add_token(token: str) -> bool:
        token = token.strip()
        if not token or token in seen_tokens:
            return False
        username = _validate_github_token(token)
        if username:
            seen_tokens.add(token)
            result.append((token, username))
            print(f"  Authenticated as {username}")
            return True
        else:
            print(f"  Warning: token ending in ...{token[-4:]} is invalid, skipping")
            return False

    # 1. Explicit tokens from CLI flags
    if explicit_tokens:
        for t in explicit_tokens:
            # Support comma-separated tokens in each value (for env var)
            for part in t.split(","):
                _add_token(part)

    # 2. GITHUB_TOKEN env var fallback
    if not result:
        env_token = os.environ.get("GITHUB_TOKEN", "")
        for part in env_token.split(","):
            _add_token(part)

    # 3. Cached device-flow tokens
    if not result:
        cached = _load_github_tokens()
        for acct in cached:
            _add_token(acct["github_token"])

    # 4. Interactive device flow
    if not result:
        token = device_flow_login()
        username = _validate_github_token(token)
        if username:
            result.append((token, username))

    if not result:
        raise RuntimeError("No valid GitHub tokens found.")

    return result


def device_flow_login() -> str:
    """Run the GitHub device-flow OAuth and return a GitHub access token.

    Appends the new account to the multi-token cache.
    """
    # Step 1: Request device code
    r = httpx.post(
        "https://github.com/login/device/code",
        headers=HEADERS_BASE,
        json={"client_id": GITHUB_CLIENT_ID, "scope": "read:user"},
    )
    r.raise_for_status()
    data = r.json()

    device_code = data["device_code"]
    user_code = data["user_code"]
    verification_uri = data["verification_uri"]
    interval = data.get("interval", 5)

    print(f"\nPlease visit: {verification_uri}")
    print(f"and enter code: {user_code}\n")
    input("Press Enter to open the browser...")

    try:
        webbrowser.open(verification_uri)
    except Exception:
        pass

    # Step 2: Poll for token
    print("Waiting for authorization...", end="", flush=True)
    while True:
        time.sleep(interval)
        r = httpx.post(
            "https://github.com/login/oauth/access_token",
            headers=HEADERS_BASE,
            json={
                "client_id": GITHUB_CLIENT_ID,
                "device_code": device_code,
                "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
            },
        )
        r.raise_for_status()
        body = r.json()

        error = body.get("error")
        if error == "authorization_pending":
            print(".", end="", flush=True)
            continue
        if error == "slow_down":
            interval = body.get("interval", interval + 5)
            continue
        if error:
            raise RuntimeError(f"OAuth error: {error} - {body.get('error_description')}")

        token = body["access_token"]
        print(" done!")

        # Add to cache (deduplicate by token)
        username = _validate_github_token(token)
        if username:
            print(f"Logged in as {username}")

            # Prompt for plan and quota
            plan = input("Copilot plan [free/pro/pro+/business/enterprise] (pro): ").strip() or "pro"
            default_quota = {"free": "50", "pro": "300", "pro+": "1500",
                             "business": "300", "enterprise": "1000"}.get(plan, "300")
            quota_str = input(f"Monthly premium request quota ({default_quota}): ").strip()
            quota_limit = int(quota_str) if quota_str else int(default_quota)

            accounts = _load_github_tokens()
            accounts = [a for a in accounts if a["github_token"] != token]
            accounts.append({
                "github_token": token, "username": username,
                "plan": plan, "quota_limit": quota_limit,
            })
            _save_github_tokens(accounts)
            _print_cached_accounts(accounts)
        return token


def logout(username: str | None = None) -> None:
    """Remove stored credentials.

    If *username* is given, remove only that account. Otherwise remove all.
    """
    accounts = _load_github_tokens()
    if not accounts:
        print("No stored credentials found.")
        return

    if username:
        before = len(accounts)
        accounts = [a for a in accounts if a["username"] != username]
        if len(accounts) == before:
            print(f"No cached account found for '{username}'.")
            return
        _save_github_tokens(accounts)
        print(f"Removed account '{username}'.")
        if accounts:
            _print_cached_accounts(accounts)
    else:
        if TOKENS_FILE.exists():
            TOKENS_FILE.unlink()
        # Clean up old format too
        if TOKEN_FILE.exists():
            TOKEN_FILE.unlink()
        print("Removed all stored credentials.")


def list_accounts() -> list[dict]:
    """Return cached accounts with validity status, plan, quota, and usage."""
    accounts = _load_github_tokens()
    result = []
    for acct in accounts:
        token = acct["github_token"]
        username = _validate_github_token(token)
        entry = {
            "username": acct["username"],
            "valid": username is not None,
            "plan": acct.get("plan"),
            "quota_limit": acct.get("quota_limit"),
            "token": token,
        }
        result.append(entry)
    return result


def fetch_usage(token: str, username: str) -> float | None:
    """Fetch premium request usage from the GitHub billing API.

    Returns the total premium requests used this month, or None on failure.
    """
    try:
        r = httpx.get(
            f"https://api.github.com/users/{username}/settings/billing/premium_request/usage",
            headers={
                "authorization": f"token {token}",
                "accept": "application/vnd.github+json",
                "x-github-api-version": "2026-03-10",
            },
            timeout=30,
        )
        if r.status_code == 200:
            data = r.json()
            return sum(
                item.get("netQuantity", 0)
                for item in data.get("usageItems", [])
            )
    except Exception:
        pass
    return None


def update_account(username: str, *, plan: str | None = None,
                   quota_limit: int | None = None) -> bool:
    """Update plan and/or quota_limit for a cached account by username.

    Returns True if the account was found and updated, False otherwise.
    """
    accounts = _load_github_tokens()
    found = False
    for acct in accounts:
        if acct["username"] == username:
            if plan is not None:
                acct["plan"] = plan
            if quota_limit is not None:
                acct["quota_limit"] = quota_limit
            found = True
            break
    if found:
        _save_github_tokens(accounts)
    return found


def _print_cached_accounts(accounts: list[dict]) -> None:
    """Print a summary of cached accounts."""
    print(f"\nCached accounts ({len(accounts)}):")
    for acct in accounts:
        print(f"  - {acct['username']}")


class CopilotTokenManager:
    """Manages the short-lived Copilot API token, refreshing as needed.

    Thread- and async-safe: concurrent callers wait on a single refresh
    rather than issuing duplicate token requests.
    """

    def __init__(self, github_token: str):
        self.github_token = github_token
        self._copilot_token: str | None = None
        self._expires_at: float = 0
        self._lock = asyncio.Lock()

    async def get_token(self) -> str:
        """Return a valid Copilot token, refreshing if needed."""
        if self._copilot_token and time.time() < self._expires_at - 300:
            return self._copilot_token

        async with self._lock:
            # Double-check after acquiring the lock
            # have already refreshed while we were waiting.
            if self._copilot_token and time.time() < self._expires_at - 300:
                return self._copilot_token

            async with httpx.AsyncClient() as client:
                r = await client.get(
                    "https://api.github.com/copilot_internal/v2/token",
                    headers={
                        "authorization": f"token {self.github_token}",
                        **HEADERS_BASE,
                    },
                )
                if r.status_code == 401:
                    raise RuntimeError(
                        "GitHub token rejected. Run `copilot-api login` again."
                    )
                r.raise_for_status()
                data = r.json()

            self._copilot_token = data["token"]
            self._expires_at = data["expires_at"]
            return self._copilot_token

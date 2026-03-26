"""GitHub Copilot device-flow OAuth and token management."""

import json
import time
import webbrowser
from pathlib import Path

import httpx

GITHUB_CLIENT_ID = "Iv1.b507a08c87ecfe98"
TOKEN_FILE = Path.home() / ".config" / "copilot-api" / "token.json"

HEADERS_BASE = {
    "accept": "application/json",
    "content-type": "application/json",
    "editor-version": "vscode/1.95.0",
    "editor-plugin-version": "copilot-chat/0.23.0",
    "user-agent": "GitHubCopilotChat/0.23.0",
}


def _save_github_token(token: str) -> None:
    TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True)
    TOKEN_FILE.write_text(json.dumps({"github_token": token}))


def _load_github_token() -> str | None:
    if TOKEN_FILE.exists():
        data = json.loads(TOKEN_FILE.read_text())
        return data.get("github_token")
    return None


def device_flow_login() -> str:
    """Run the GitHub device-flow OAuth and return a GitHub access token."""
    existing = _load_github_token()
    if existing:
        # Verify the token is still valid
        r = httpx.get(
            "https://api.github.com/user",
            headers={"authorization": f"token {existing}", "accept": "application/json"},
        )
        if r.status_code == 200:
            user = r.json().get("login", "unknown")
            print(f"Already authenticated as {user}")
            return existing

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
        _save_github_token(token)
        return token


def logout() -> None:
    """Remove stored credentials."""
    if TOKEN_FILE.exists():
        TOKEN_FILE.unlink()
        print("Logged out.")
    else:
        print("No stored credentials found.")


class CopilotTokenManager:
    """Manages the short-lived Copilot API token, refreshing as needed."""

    def __init__(self, github_token: str):
        self.github_token = github_token
        self._copilot_token: str | None = None
        self._expires_at: float = 0

    def get_token(self) -> str:
        """Return a valid Copilot token, refreshing if needed."""
        if self._copilot_token and time.time() < self._expires_at - 300:
            return self._copilot_token

        r = httpx.get(
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

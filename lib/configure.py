"""Configure agentic coding tools to use the copilot-adapter proxy."""

import json
import shutil
from pathlib import Path

# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def _backup_file(path: Path) -> Path | None:
    """Copy *path* to *path*.bak.  Returns the backup path, or None."""
    if not path.exists():
        return None
    bak = path.with_suffix(path.suffix + ".bak")
    shutil.copy2(path, bak)
    print(f"Backed up {path} -> {bak}")
    return bak


def _restore_backup(path: Path) -> bool:
    """Restore *path* from its .bak file.  Returns True on success."""
    bak = path.with_suffix(path.suffix + ".bak")
    if not bak.exists():
        return False
    shutil.copy2(bak, path)
    bak.unlink()
    print(f"Restored {path} from {bak}")
    return True


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def _mask_token(token: str) -> str:
    if len(token) <= 10:
        return token
    return token[:6] + "..." + token[-4:]


def _proxy_url(host: str, port: int, path: str = "") -> str:
    return f"http://{host}:{port}{path}"


# ---------------------------------------------------------------------------
# Claude Code
# ---------------------------------------------------------------------------

_CLAUDE_CODE_CONFIG = Path.home() / ".claude" / "settings.json"


def configure_claude_code(*, host: str, port: int,
                          api_token: str | None, revert: bool) -> None:
    path = _CLAUDE_CODE_CONFIG
    if revert:
        if _restore_backup(path):
            return
        data = _read_json(path)
        if not data:
            print(f"No config file found at {path}, nothing to revert.")
            return
        env = data.get("env", {})
        env.pop("ANTHROPIC_BASE_URL", None)
        env.pop("ANTHROPIC_API_KEY", None)
        if not env:
            data.pop("env", None)
        _write_json(path, data)
        print("Reverted claude-code configuration.")
        return

    data = _read_json(path)
    _backup_file(path)
    env = data.setdefault("env", {})
    url = _proxy_url(host, port)
    env["ANTHROPIC_BASE_URL"] = url
    if api_token:
        env["ANTHROPIC_API_KEY"] = api_token
    _write_json(path, data)
    print(f"Configured claude-code to use proxy at {url}")
    print(f"  ANTHROPIC_BASE_URL = {url}")
    if api_token:
        print(f"  ANTHROPIC_API_KEY  = {_mask_token(api_token)}")


# ---------------------------------------------------------------------------
# OpenAI Codex CLI
# ---------------------------------------------------------------------------

_CODEX_CONFIG = Path.home() / ".codex" / "config.toml"

# Minimal TOML helpers — avoids adding a toml dependency just for this.

def _read_toml(path: Path) -> str:
    if not path.exists():
        return ""
    with open(path) as f:
        return f.read()


def _write_toml(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(content)


def configure_codex(*, host: str, port: int,
                    api_token: str | None, revert: bool) -> None:
    path = _CODEX_CONFIG
    if revert:
        if _restore_backup(path):
            return
        text = _read_toml(path)
        if not text:
            print(f"No config file found at {path}, nothing to revert.")
            return
        # Remove the model_provider line if it points to copilot-adapter
        lines = text.splitlines(keepends=True)
        out: list[str] = []
        skip_section = False
        for line in lines:
            stripped = line.strip()
            # Remove model_provider = "copilot-adapter"
            if stripped == 'model_provider = "copilot-adapter"':
                continue
            # Skip the [model_providers.copilot-adapter] section
            if stripped == "[model_providers.copilot-adapter]":
                skip_section = True
                continue
            if skip_section:
                if stripped.startswith("["):
                    skip_section = False
                else:
                    continue
            out.append(line)
        # Remove trailing blank lines from removed sections
        result = "".join(out).rstrip("\n") + "\n"
        _write_toml(path, result)
        print("Reverted codex configuration.")
        return

    text = _read_toml(path)
    _backup_file(path)
    url = _proxy_url(host, port, "/v1")

    # Remove any existing copilot-adapter section and model_provider line first
    lines = text.splitlines(keepends=True)
    out = []
    skip_section = False
    for line in lines:
        stripped = line.strip()
        if stripped == 'model_provider = "copilot-adapter"':
            continue
        if stripped == "[model_providers.copilot-adapter]":
            skip_section = True
            continue
        if skip_section:
            if stripped.startswith("["):
                skip_section = False
            else:
                continue
        out.append(line)
    cleaned = "".join(out).rstrip("\n")

    # Find the right place to insert model_provider — after any top-level
    # key-value pairs but before the first section header.
    insert_lines = cleaned.splitlines(keepends=True)
    provider_line = 'model_provider = "copilot-adapter"\n'
    # Check if there's already a model_provider line to replace
    inserted = False
    for i, line in enumerate(insert_lines):
        if line.strip().startswith("model_provider"):
            insert_lines[i] = provider_line
            inserted = True
            break
    if not inserted:
        # Insert before the first section header, with a blank line after
        for i, line in enumerate(insert_lines):
            if line.strip().startswith("["):
                insert_lines.insert(i, "\n")
                insert_lines.insert(i, provider_line)
                inserted = True
                break
        if not inserted:
            insert_lines.append(provider_line)

    # Append the provider section
    section = (
        f"\n[model_providers.copilot-adapter]\n"
        f'name = "Copilot Adapter"\n'
        f'base_url = "{url}"\n'
        f'env_key = "COPILOT_ADAPTER_API_TOKEN"\n'
    )
    result = "".join(insert_lines).rstrip("\n") + "\n" + section

    _write_toml(path, result)
    print(f"Configured codex to use proxy at {url}")
    print(f"  model_provider = copilot-adapter")
    print(f"  base_url       = {url}")
    print(f"  env_key        = COPILOT_ADAPTER_API_TOKEN")
    if api_token:
        print(f"\nSet this environment variable for authentication:")
        print(f"  export COPILOT_ADAPTER_API_TOKEN={api_token}")
        print(f"\nFor PowerShell:")
        print(f'  $env:COPILOT_ADAPTER_API_TOKEN = "{api_token}"')


# ---------------------------------------------------------------------------
# Gemini CLI
# ---------------------------------------------------------------------------

_GEMINI_CONFIG = Path.home() / ".gemini" / "settings.json"


def configure_gemini_cli(*, host: str, port: int,
                         api_token: str | None, revert: bool) -> None:
    path = _GEMINI_CONFIG
    if revert:
        if _restore_backup(path):
            return
        data = _read_json(path)
        if not data:
            print(f"No config file found at {path}, nothing to revert.")
            return
        data.pop("baseUrl", None)
        data.pop("apiKey", None)
        _write_json(path, data)
        print("Reverted gemini-cli configuration.")
        return

    data = _read_json(path)
    _backup_file(path)
    url = _proxy_url(host, port)
    data["baseUrl"] = url
    if api_token:
        data["apiKey"] = api_token
    _write_json(path, data)
    print(f"Configured gemini-cli to use proxy at {url}")
    print(f"  baseUrl = {url}")
    if api_token:
        print(f"  apiKey  = {_mask_token(api_token)}")


# ---------------------------------------------------------------------------
# OpenCode
# ---------------------------------------------------------------------------

_OPENCODE_CONFIG = Path.home() / ".config" / "opencode" / "opencode.json"

_PROVIDER_KEY = "copilot-adapter"


def configure_opencode(*, host: str, port: int,
                       api_token: str | None, revert: bool) -> None:
    path = _OPENCODE_CONFIG
    if revert:
        if _restore_backup(path):
            return
        data = _read_json(path)
        if not data:
            print(f"No config file found at {path}, nothing to revert.")
            return
        data.get("provider", {}).pop(_PROVIDER_KEY, None)
        _write_json(path, data)
        print("Reverted opencode configuration.")
        return

    data = _read_json(path)
    _backup_file(path)
    url = _proxy_url(host, port)
    providers = data.setdefault("provider", {})
    entry: dict = {
        "npm": "@ai-sdk/anthropic",
        "name": "Copilot Adapter",
        "options": {
            "baseURL": url,
        },
    }
    if api_token:
        entry["options"]["apiKey"] = api_token
    providers[_PROVIDER_KEY] = entry
    _write_json(path, data)
    print(f"Configured opencode to use proxy at {url}")
    print(f"  Provider: {_PROVIDER_KEY}")
    print(f"  baseURL = {url}")
    if api_token:
        print(f"  apiKey  = {_mask_token(api_token)}")
    print(f"\nSelect the '{_PROVIDER_KEY}' provider in OpenCode to use it.")


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

TOOL_NAMES = ("claude-code", "codex", "gemini-cli", "opencode")

CONFIGURATORS = {
    "claude-code": configure_claude_code,
    "codex": configure_codex,
    "gemini-cli": configure_gemini_cli,
    "opencode": configure_opencode,
}

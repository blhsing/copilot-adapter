"""CLI entry point for copilot-adapter."""

import asyncio
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import click

from lib.configure import CONFIGURATORS

_DEFAULT_CONFIG = Path.home() / ".config" / "copilot-adapter" / "config.json"


def _supports_color() -> bool:
    """Detect whether the terminal supports ANSI color sequences."""
    if not hasattr(sys.stderr, "isatty") or not sys.stderr.isatty():
        return False
    if os.environ.get("NO_COLOR"):
        return False
    if sys.platform == "win32":
        # Check if the Windows console has VT processing enabled
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            handle = kernel32.GetStdHandle(-12)  # STD_ERROR_HANDLE
            mode = ctypes.c_ulong()
            if kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
                return bool(mode.value & 0x0004)  # ENABLE_VIRTUAL_TERMINAL_PROCESSING
            return False
        except Exception:
            return False
    return True


def _load_config(path: str | None) -> dict:
    """Load a JSON config file and return its contents.

    Returns an empty dict if no path is given and the default file doesn't
    exist.
    """
    if path is not None:
        p = Path(path)
        if not p.exists():
            raise click.ClickException(f"Config file not found: {path}")
    else:
        p = _DEFAULT_CONFIG
        if not p.exists():
            return {}
    with open(p) as f:
        return json.load(f)


@click.group()
def main():
    """OpenAI-compatible API server backed by GitHub Copilot."""


@main.command()
def login():
    """Authenticate with GitHub using the device flow.

    Each invocation adds a new account to the cache. Run multiple times
    to add multiple accounts.
    """
    from lib.auth import device_flow_login
    device_flow_login()


@main.command("claude-login")
def claude_login():
    """Authenticate against claude.ai via PKCE paste-back OAuth.

    Adds the resulting Claude Max subscription to the Anthropic accounts
    cache. The proxy prefers Anthropic-pool accounts for native
    ``/v1/messages`` traffic, falling back to Copilot when the Anthropic
    pool is exhausted.
    """
    from lib.anthropic_auth import claude_login_interactive
    result = claude_login_interactive()
    if result is None:
        raise click.ClickException("Login aborted or failed.")


@main.command("codex-login")
def codex_login():
    """Authenticate a ChatGPT (Codex) account via the device-code flow.

    Opens auth.openai.com and prompts for a one-time code (no localhost
    callback). The proxy prefers ChatGPT-pool accounts for /v1/responses
    traffic, falling back to Copilot when the ChatGPT pool is exhausted.

    Requires 'device code authorization' enabled in ChatGPT > Settings >
    Security on the account being added.
    """
    from lib.openai_auth import codex_login_interactive
    result = codex_login_interactive()
    if result is None:
        raise click.ClickException("Login aborted or failed.")


@main.command("regenerate-ca")
@click.option("--ca-dir", default=None, metavar="DIR",
              envvar="COPILOT_ADAPTER_CA_DIR",
              help="Directory holding the CA certificate and key.")
@click.confirmation_option(prompt="Regenerate the MITM CA? Existing leaf "
                                  "certs become invalid and clients must "
                                  "reinstall the new CA before MITM works.")
def regenerate_ca(ca_dir: str | None):
    """Wipe and regenerate the MITM CA certificate.

    Clients (Claude Code, curl, browsers) must reinstall the new CA into
    their trust store afterwards — until they do, intercepted HTTPS will
    fail with a certificate-validation error.
    """
    from lib.cert import ca_paths, ensure_ca

    d = Path(ca_dir) if ca_dir else None
    cert_path, key_path = ca_paths(d)
    for p in (cert_path, key_path):
        if p.exists():
            p.unlink()
    cert, _ = ensure_ca(d)
    print(f"Regenerated CA at {cert_path}")
    print(f"  Subject: {cert.subject.rfc4514_string()}")
    print(f"  Valid:   {cert.not_valid_before_utc:%Y-%m-%d} to "
          f"{cert.not_valid_after_utc:%Y-%m-%d}")
    print(f"\nReinstall this CA in your client trust stores:")
    print(f"  Node.js:  export NODE_EXTRA_CA_CERTS={cert_path}")
    print(f"  Windows:  certutil -addstore -f Root {cert_path}")


@main.command()
@click.option("--username", default=None, metavar="USER",
              help="Remove a specific account by GitHub username.")
@click.option("--all", "remove_all", is_flag=True,
              help="Remove all stored credentials.")
def logout(username: str | None, remove_all: bool):
    """Remove stored credentials."""
    from lib.auth import logout as do_logout
    if remove_all or not username:
        do_logout()
    else:
        do_logout(username=username)


def _format_usage_value(value) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{number:g}"


def _first_present(data: dict, keys: tuple[str, ...]):
    for key in keys:
        value = data.get(key)
        if value is not None:
            return value
    return None


def _duration_seconds(value, *, numeric_unit_seconds: float = 1.0) -> int | None:
    try:
        seconds = int(round(float(value) * numeric_unit_seconds))
    except (TypeError, ValueError):
        if not isinstance(value, str):
            return None
        match = re.fullmatch(
            r"\s*(\d+(?:\.\d+)?)\s*"
            r"(seconds?|secs?|s|minutes?|mins?|m|hours?|hrs?|h|days?|d)\s*",
            value,
            re.IGNORECASE,
        )
        if match is None:
            return None
        amount = float(match.group(1))
        unit = match.group(2).lower()
        if unit.startswith(("d", "day")):
            seconds = int(round(amount * 86400))
        elif unit.startswith(("h", "hour", "hr")):
            seconds = int(round(amount * 3600))
        elif unit.startswith(("m", "min")):
            seconds = int(round(amount * 60))
        else:
            seconds = int(round(amount))
    if seconds < 0:
        return None
    return seconds


def _format_duration_seconds(value) -> str | None:
    seconds = _duration_seconds(value)
    if seconds is None:
        return None
    if seconds == 0:
        return "0s"

    units = [
        ("d", 86400),
        ("h", 3600),
        ("m", 60),
        ("s", 1),
    ]
    parts = []
    remaining = seconds
    for suffix, unit_seconds in units:
        amount, remaining = divmod(remaining, unit_seconds)
        if amount:
            parts.append(f"{amount}{suffix}")
        if len(parts) == 2:
            break
    return " ".join(parts)


def _format_duration_label(seconds: int) -> str:
    if seconds >= 86400 and seconds % 86400 == 0:
        days = seconds // 86400
        return f"{days}-day"
    if seconds >= 3600 and seconds % 3600 == 0:
        hours = seconds // 3600
        return f"{hours}-hour"
    if seconds >= 86400:
        days, remainder = divmod(seconds, 86400)
        if remainder % 3600 == 0:
            hours = remainder // 3600
            suffix = f" {hours}-hour" if hours else ""
            return f"{days}-day{suffix}"
    if seconds >= 3600:
        hours, remainder = divmod(seconds, 3600)
        if remainder % 60 == 0:
            minutes = remainder // 60
            suffix = f" {minutes}-minute" if minutes else ""
            return f"{hours}-hour{suffix}"
    if seconds >= 60 and seconds % 60 == 0:
        minutes = seconds // 60
        return f"{minutes}-minute"
    return f"{seconds}-second"


def _parse_timestamp(value) -> datetime | None:
    if isinstance(value, (int, float)):
        number = float(value)
        if number > 1_000_000_000_000:
            number /= 1000
        if number < 946_684_800:
            return None
        try:
            return datetime.fromtimestamp(number, tz=timezone.utc)
        except (OverflowError, OSError, ValueError):
            return None
    if not isinstance(value, str):
        return None

    raw = value.strip()
    if not raw:
        return None
    try:
        number = float(raw)
    except ValueError:
        number = None
    if number is not None:
        return _parse_timestamp(number)

    normalized = raw[:-1] + "+00:00" if raw.endswith("Z") else raw
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _format_timestamp(value) -> str:
    parsed = _parse_timestamp(value)
    if parsed is None:
        return str(value)
    return parsed.strftime("%Y-%m-%d %H:%M UTC")


def _usage_window_seconds(window: dict) -> int | None:
    window_seconds = _first_present(window, (
        "limit_window_seconds",
        "window_seconds",
        "duration_seconds",
        "period_seconds",
        "interval_seconds",
        "limitWindowSeconds",
        "windowSeconds",
        "durationSeconds",
        "periodSeconds",
        "intervalSeconds",
        "window",
        "duration",
        "period",
        "interval",
    ))
    duration = _duration_seconds(window_seconds)
    if duration is not None:
        return duration

    window_minutes = _first_present(window, (
        "limit_window_minutes",
        "window_minutes",
        "duration_minutes",
        "period_minutes",
        "interval_minutes",
        "limitWindowMinutes",
        "windowMinutes",
        "durationMinutes",
        "periodMinutes",
        "intervalMinutes",
        "windowDurationMins",
    ))
    duration = _duration_seconds(window_minutes, numeric_unit_seconds=60)
    if duration is not None:
        return duration

    window_ms = _first_present(window, (
        "limit_window_milliseconds",
        "window_milliseconds",
        "duration_milliseconds",
        "period_milliseconds",
        "interval_milliseconds",
        "limit_window_ms",
        "window_ms",
        "duration_ms",
        "period_ms",
        "interval_ms",
        "limitWindowMilliseconds",
        "windowMilliseconds",
        "durationMilliseconds",
        "periodMilliseconds",
        "intervalMilliseconds",
        "limitWindowMs",
        "windowMs",
        "durationMs",
        "periodMs",
        "intervalMs",
    ))
    duration = _duration_seconds(window_ms, numeric_unit_seconds=0.001)
    if duration is not None:
        return duration

    starts_at = _first_present(window, (
        "start_at",
        "starts_at",
        "started_at",
        "start_time",
        "startAt",
        "startsAt",
        "startedAt",
        "startTime",
        "start",
    ))
    ends_at = _first_present(window, (
        "end_at",
        "ends_at",
        "expires_at",
        "expire_at",
        "end_time",
        "endAt",
        "endsAt",
        "expiresAt",
        "expireAt",
        "endTime",
        "end",
    ))
    start = _parse_timestamp(starts_at)
    end = _parse_timestamp(ends_at)
    if start is not None and end is not None and end > start:
        return int(round((end - start).total_seconds()))
    return None


def _format_usage_window_label(window: dict, fallback: str) -> str:
    seconds = _usage_window_seconds(window)
    if seconds is None:
        return fallback
    return f"{_format_duration_label(seconds)} limit"


def _format_usage_window_timing(window: dict, *, include_duration: bool) -> list[str]:
    parts = []

    duration = _format_duration_seconds(_usage_window_seconds(window))
    if include_duration and duration is not None:
        parts.append(f"window {duration}")

    resets_in = _first_present(window, (
        "resets_in_seconds",
        "reset_after_seconds",
        "seconds_until_reset",
        "reset_seconds",
        "retry_after_seconds",
        "resetsInSeconds",
        "resetAfterSeconds",
        "secondsUntilReset",
        "resetSeconds",
        "retryAfterSeconds",
        "retry_after",
        "retryAfter",
    ))
    reset_duration = _format_duration_seconds(resets_in)
    if reset_duration is not None:
        parts.append(f"resets in {reset_duration}")
        return parts

    resets_at = _first_present(window, (
        "reset_at",
        "resets_at",
        "reset_time",
        "resetAt",
        "resetsAt",
        "resetTime",
        "reset",
    ))
    if resets_at is not None:
        parts.append(f"resets {_format_timestamp(resets_at)}")
    else:
        ends_at = _first_present(window, (
            "end_at",
            "ends_at",
            "expires_at",
            "expire_at",
            "end_time",
            "endAt",
            "endsAt",
            "expiresAt",
            "expireAt",
            "endTime",
            "end",
        ))
        if ends_at is not None:
            parts.append(f"ends {_format_timestamp(ends_at)}")
    return parts


def _format_usage_window(
    label: str,
    window: dict,
    *,
    include_window_duration: bool = True,
) -> str | None:
    percent = _first_present(window, ("utilization", "used_percent", "usedPercent"))
    used = _first_present(window, ("used", "usage", "current", "consumed"))
    limit = _first_present(window, ("limit", "quota", "max", "total"))
    remaining = _first_present(window, ("remaining", "available"))

    parts = []
    if percent is not None:
        parts.append(f"{_format_usage_value(percent)}% used")
    if used is not None and limit is not None:
        parts.append(f"{_format_usage_value(used)}/{_format_usage_value(limit)}")
    elif used is not None:
        parts.append(f"used {_format_usage_value(used)}")
    elif limit is not None:
        parts.append(f"limit {_format_usage_value(limit)}")
    if remaining is not None:
        parts.append(f"remaining {_format_usage_value(remaining)}")
    parts.extend(
        _format_usage_window_timing(
            window,
            include_duration=include_window_duration,
        )
    )

    if not parts:
        return None
    return f"{label}: {', '.join(parts)}"


def _format_usage_details(backend: str, details: dict | None) -> str:
    if not details:
        return "usage: unavailable"
    if backend == "anthropic":
        windows = [
            ("5-hour limit", details.get("five_hour")),
            ("7-day total limit", details.get("seven_day")),
            ("7-day Opus limit", details.get("seven_day_opus")),
            ("7-day Sonnet limit", details.get("seven_day_sonnet")),
        ]
    else:
        rate_limit = details.get("rate_limit") or {}
        primary_window = rate_limit.get("primary_window")
        secondary_window = rate_limit.get("secondary_window")
        windows = [
            (
                _format_usage_window_label(primary_window, "5-hour limit")
                if isinstance(primary_window, dict) else "5-hour limit",
                primary_window,
            ),
            (
                _format_usage_window_label(secondary_window, "7-day limit")
                if isinstance(secondary_window, dict) else "7-day limit",
                secondary_window,
            ),
        ]
    summaries = [
        summary
        for label, window in windows
        if isinstance(window, dict)
        for summary in [
            _format_usage_window(
                label,
                window,
                include_window_duration=backend != "chatgpt",
            )
        ]
        if summary
    ]
    if not summaries:
        return "usage: available, unrecognized response shape"
    return "usage: " + "; ".join(summaries)


async def _fetch_anthropic_usage(accounts: list[dict]) -> dict[str, dict | None]:
    from lib.anthropic_auth import (
        AnthropicTokenManager,
        update_anthropic_account_tokens,
    )
    from lib.anthropic_client import AnthropicClient

    usage: dict[str, dict | None] = {}
    for acct in accounts:
        username = acct.get("username", "unknown")

        def _on_rotated(new_access: str, new_refresh: str, new_expires: float,
                        username=username):
            update_anthropic_account_tokens(username, new_access,
                                            new_refresh, new_expires)

        tm = AnthropicTokenManager(
            acct.get("access_token", ""),
            acct.get("refresh_token", ""),
            float(acct.get("expires_at", 0)),
            on_rotated=_on_rotated,
        )
        usage[username] = await AnthropicClient(
            tm, account_label=username
        ).fetch_usage_details()
    return usage


async def _fetch_chatgpt_usage(accounts: list[dict]) -> dict[str, dict | None]:
    from lib.chatgpt_client import ChatGPTClient
    from lib.openai_auth import (
        OpenAITokenManager,
        update_chatgpt_account_tokens,
    )

    usage: dict[str, dict | None] = {}
    for acct in accounts:
        username = acct.get("username", "unknown")

        def _on_rotated(new_access: str, new_refresh: str, new_expires: float,
                        username=username):
            update_chatgpt_account_tokens(username, new_access,
                                          new_refresh, new_expires)

        tm = OpenAITokenManager(
            acct.get("access_token", ""),
            acct.get("refresh_token", ""),
            float(acct.get("expires_at", 0)),
            on_rotated=_on_rotated,
        )
        usage[username] = await ChatGPTClient(
            tm, acct.get("account_id"), account_label=username,
        ).fetch_usage_details()
    return usage


@main.command()
@click.option("--add", "add_token", default=None, metavar="TOKEN",
              help="Add a GitHub OAuth token (ghu_) to the cached accounts. PATs (ghp_) are rejected by the Copilot API.")
@click.option("--remove", "remove_username", default=None, metavar="USER",
              help="Remove a cached GitHub Copilot account by username.")
def accounts(add_token: str | None, remove_username: str | None):
    """List cached accounts; add/remove GitHub Copilot accounts."""
    from lib.auth import add_account, list_accounts, remove_account

    if add_token:
        result = add_account(add_token)
        if result is None:
            raise click.ClickException("Invalid token — could not authenticate with GitHub")
        print(f"Added {result['username']}")
        return

    if remove_username:
        if not remove_account(remove_username):
            raise click.ClickException(f"Account '{remove_username}' not found in cache")
        print(f"Removed {remove_username}")
        return

    copilot_accounts = list_accounts()
    try:
        from lib.anthropic_auth import resolve_anthropic_accounts
        anthropic_accounts = resolve_anthropic_accounts()
    except Exception:
        anthropic_accounts = []
    try:
        from lib.openai_auth import resolve_chatgpt_accounts
        chatgpt_accounts = resolve_chatgpt_accounts()
    except Exception:
        chatgpt_accounts = []

    if not copilot_accounts and not anthropic_accounts and not chatgpt_accounts:
        print("No cached accounts.")
        return

    if copilot_accounts:
        print(f"Copilot accounts ({len(copilot_accounts)}):")
        for acct in copilot_accounts:
            status = "valid" if acct["valid"] else "expired/invalid"
            print(f"  - {acct['username']} ({status}, usage: no endpoint)")

    if anthropic_accounts:
        usage = asyncio.run(_fetch_anthropic_usage(anthropic_accounts))
        print(f"Anthropic accounts ({len(anthropic_accounts)}):")
        for acct in anthropic_accounts:
            username = acct.get("username", "unknown")
            print(f"  - {username} ({_format_usage_details('anthropic', usage.get(username))})")

    if chatgpt_accounts:
        usage = asyncio.run(_fetch_chatgpt_usage(chatgpt_accounts))
        print(f"ChatGPT accounts ({len(chatgpt_accounts)}):")
        for acct in chatgpt_accounts:
            username = acct.get("username", "unknown")
            print(f"  - {username} ({_format_usage_details('chatgpt', usage.get(username))})")


@main.command("ca-cert")
@click.option("--ca-dir", default=None, metavar="DIR",
              envvar="COPILOT_ADAPTER_CA_DIR",
              help="Directory for the CA certificate and key "
                   "(default: ~/.config/copilot-adapter).")
def ca_cert(ca_dir: str | None):
    """Generate or show the MITM CA certificate.

    If the CA does not exist yet it is created automatically.
    Prints the path to the CA certificate for use with NODE_EXTRA_CA_CERTS
    or system trust stores.
    """
    from lib.cert import ca_paths, ensure_ca

    d = Path(ca_dir) if ca_dir else None
    cert, _ = ensure_ca(d)
    cert_path, _ = ca_paths(d)

    subject = cert.subject.rfc4514_string()
    not_before = cert.not_valid_before_utc.strftime("%Y-%m-%d")
    not_after = cert.not_valid_after_utc.strftime("%Y-%m-%d")

    print(f"CA certificate: {cert_path}")
    print(f"  Subject:  {subject}")
    print(f"  Valid:    {not_before} to {not_after}")
    print(f"\nTo trust this CA in Node.js clients:")
    print(f"  export NODE_EXTRA_CA_CERTS={cert_path}")


@main.command()
@click.option("--generate", "do_generate", is_flag=True,
              help="Generate a new API token.")
@click.option("--label", default=None, metavar="LABEL",
              help="Optional label for the generated token.")
@click.option("--revoke", "revoke_value", default=None, metavar="TOKEN_OR_LABEL",
              help="Revoke a token by its value or label.")
def tokens(do_generate: bool, label: str | None, revoke_value: str | None):
    """Manage API tokens for protecting the reverse API proxy."""
    from lib.auth import (generate_api_token, list_api_tokens,
                          revoke_api_token)

    if do_generate:
        entry = generate_api_token(label=label)
        print(f"Generated API token (save this — it won't be shown again):")
        print(f"  Token:   {entry['token']}")
        if entry["label"]:
            print(f"  Label:   {entry['label']}")
        print(f"  Created: {entry['created_at']}")
        return

    if revoke_value:
        if not revoke_api_token(revoke_value):
            raise click.ClickException(
                f"No token matching '{revoke_value}' found")
        print(f"Revoked token: {revoke_value}")
        return

    all_tokens = list_api_tokens()
    if not all_tokens:
        print("No API tokens. Use --generate to create one.")
        return
    print(f"API tokens ({len(all_tokens)}):")
    for t in all_tokens:
        masked = t["token"][:6] + "..." + t["token"][-4:]
        label_str = f" ({t['label']})" if t.get("label") else ""
        print(f"  - {masked}{label_str}  created: {t['created_at']}")


@main.command()
@click.argument("tool", type=click.Choice(list(CONFIGURATORS)))
@click.option("--revert", is_flag=True, default=False,
              help="Revert the tool's config back to its default provider.")
@click.option("--host", default="127.0.0.1", metavar="HOST",
              help="Proxy host address (default: 127.0.0.1).")
@click.option("--port", default=18080, type=int, metavar="PORT",
              help="Proxy port (default: 18080).")
@click.option("--api-token", default=None, metavar="TOKEN",
              help="API token for the proxy. If omitted, loads from stored tokens.")
def config(tool: str, revert: bool, host: str, port: int,
           api_token: str | None):
    """Configure an agentic coding tool to use this proxy.

    Supported tools: claude-code, codex, gemini-cli, opencode.
    Use --revert to restore the tool's original configuration.
    """
    if not revert and api_token is None:
        from lib.auth import get_api_token_values
        stored = get_api_token_values()
        if stored:
            api_token = stored[0]

    CONFIGURATORS[tool](host=host, port=port, api_token=api_token, revert=revert)


@main.command()
@click.option("--config", "config_path", default=None, metavar="PATH",
              envvar="COPILOT_ADAPTER_CONFIG",
              help=f"Path to JSON config file (default: {_DEFAULT_CONFIG}).")
@click.option("--host", default=None, envvar="COPILOT_ADAPTER_HOST", metavar="HOST",
              help="Host to bind to.")
@click.option("--port", default=None, type=int, envvar="COPILOT_ADAPTER_PORT", metavar="PORT",
              help="Port to bind to.")
@click.option("--github-token", multiple=True, envvar="COPILOT_ADAPTER_GITHUB_TOKEN",
              metavar="TOKEN",
              help="GitHub OAuth token (ghu_) from device-flow login (repeatable). "
                   "PATs (ghp_) are rejected — the Copilot API returns 404 for them. "
                   "Env var supports comma-separated values.")
@click.option("--cors-origin", multiple=True, envvar="COPILOT_ADAPTER_CORS_ORIGIN",
              metavar="ORIGIN",
              help="Allowed CORS origin (repeatable). Use '*' to allow all origins.")
@click.option("--workers", default=None, type=int, envvar="COPILOT_ADAPTER_WORKERS",
              metavar="N",
              help="Number of worker processes (default: 1).")
@click.option("--strategy", default=None,
              type=click.Choice(["least-utilized", "round-robin"]),
              envvar="COPILOT_ADAPTER_STRATEGY",
              help="Account rotation strategy (default: least-utilized). "
                   "least-utilized rotates by live backend usage for "
                   "Anthropic/ChatGPT accounts; Copilot accounts round-robin.")
@click.option("--rate-limit-backoff-minutes", default=None, type=int,
              envvar="COPILOT_ADAPTER_RATE_LIMIT_BACKOFF_MINUTES", metavar="N",
              help="Minutes to sideline an account after a transient upstream failure "
                   "(e.g. 429 rate limit) before it becomes eligible again "
                   "(default: 60).")
@click.option("--log-level", default=None,
              type=click.Choice(["debug", "info", "warning", "error"], case_sensitive=False),
              envvar="COPILOT_ADAPTER_LOG_LEVEL",
              help="Logging level (default: info). Use 'debug' for verbose output.")
@click.option("--log-file", default=None, metavar="PATH",
              envvar="COPILOT_ADAPTER_LOG_FILE",
              help="Append logs to PATH in addition to the console.")
@click.option("--proxy", "proxy_mode", is_flag=True, default=False,
              envvar="COPILOT_ADAPTER_PROXY",
              help="Enable forward proxy mode on the same port. CONNECT requests to "
                   "api.githubcopilot.com are MITM'd to rewrite X-Initiator: user -> agent. "
                   "All other traffic is tunneled transparently.")
@click.option("--ca-dir", default=None, metavar="DIR",
              envvar="COPILOT_ADAPTER_CA_DIR",
              help="Directory for the MITM CA certificate and key "
                   "(default: ~/.config/copilot-adapter).")
@click.option("--model-map", "model_map_raw", multiple=True, metavar="PATTERN=TARGET",
              envvar="COPILOT_ADAPTER_MODEL_MAP",
              help="Model mapping as pattern=target (repeatable, e.g. "
                   "'--model-map *sonnet*=claude-sonnet-4.6'). "
                   "Env var supports comma-separated values.")
@click.option("--proxy-user", default=None, metavar="USER",
              envvar="COPILOT_ADAPTER_PROXY_USER",
              help="Username for forward proxy authentication.")
@click.option("--proxy-password", default=None, metavar="PASS",
              envvar="COPILOT_ADAPTER_PROXY_PASSWORD",
              help="Password for forward proxy authentication.")
@click.option("--api-token", "api_token_raw", multiple=True, metavar="TOKEN",
              envvar="COPILOT_ADAPTER_API_TOKEN",
              help="API token for protecting the reverse API proxy (repeatable). "
                   "Env var supports comma-separated values. "
                   "If not specified, loads from stored tokens.")
@click.option("--web-search-iterations", type=int, default=None, metavar="N",
              envvar="COPILOT_ADAPTER_WEB_SEARCH_ITERATIONS",
              help="Max web_search tool call iterations per request (default: 3). "
                   "Set to 0 to disable server-side web search interception.")
@click.option("--force-ddg-web-search", is_flag=True, default=False,
              envvar="COPILOT_ADAPTER_FORCE_DDG_WEB_SEARCH",
              help="Force DuckDuckGo interception for web_search calls instead of "
                   "provider-native web search. By default, Claude-targeted "
                   "requests use DDG and supported OpenAI Responses models keep "
                   "native web search.")
@click.option("--web-search-model", default=None, metavar="MODEL",
              envvar="COPILOT_ADAPTER_WEB_SEARCH_MODEL",
              help="Reroute Anthropic /v1/messages requests that carry the "
                   "web_search tool through /v1/responses against MODEL (e.g. "
                   "gpt-5.5) whenever the target model lacks native provider "
                   "web search, so the upstream call uses native "
                   "web_search_preview instead of DuckDuckGo. Ignored when "
                   "--force-ddg-web-search is set or MODEL does not support "
                   "native web search.")
@click.option("--reverse-dns-server", default=None, metavar="HOST",
              envvar="COPILOT_ADAPTER_REVERSE_DNS_SERVER",
              help="DNS server (IP or hostname) to use for reverse lookups "
                   "when logging the originating client as a hostname. "
                   "Defaults to the system resolver.")
@click.option("--reverse-dns-sync-wait-ms", default=None, type=int,
              envvar="COPILOT_ADAPTER_REVERSE_DNS_SYNC_WAIT_MS",
              metavar="MS",
              help="On a fresh reverse-DNS cache miss, block the access-log "
                   "emission for up to MS milliseconds waiting for the lookup "
                   "to complete, so the very first log line for an IP can "
                   "carry its hostname. Set to 0 to keep lookups fully async "
                   "(default: 150).")
@click.option("--forwarded-allow-ips", default=None, metavar="IPS",
              envvar="COPILOT_ADAPTER_FORWARDED_ALLOW_IPS",
              help="Comma-separated IPs (or '*') whose X-Forwarded-For / "
                   "X-Forwarded-Proto headers are trusted to rewrite the "
                   "logged client address to the true originating host. "
                   "Defaults to 127.0.0.1; set to the IP of your front proxy "
                   "(e.g. a Squid host) or '*' to honor the header from "
                   "anywhere.")
@click.option("--spoof-interactive-headers", "spoof_interactive", is_flag=True,
              default=False, envvar="COPILOT_ADAPTER_SPOOF_INTERACTIVE_HEADERS",
              help="When forwarding Anthropic-pool requests, stamp them with "
                   "the User-Agent, x-app, and extended anthropic-beta list "
                   "that the interactive Claude Code REPL sends. Default off — "
                   "the giveaway pool UA otherwise tells api.anthropic.com "
                   "this is not a real REPL session. Has no effect on Copilot "
                   "traffic.")
@click.option("--copilot-only-client", "copilot_only_client", multiple=True,
              metavar="RULE", envvar="COPILOT_ADAPTER_COPILOT_ONLY_CLIENTS",
              help="Pin a client to the Copilot backend so it never consumes "
                   "the Anthropic (/v1/messages) or ChatGPT (/v1/responses) "
                   "pools. RULE is an IP literal, CIDR range, or reverse-DNS "
                   "hostname glob (* and ?). Repeatable.")
def serve(config_path: str | None, host: str | None, port: int | None,
          github_token: tuple[str, ...], cors_origin: tuple[str, ...],
          workers: int | None, strategy: str | None,
          rate_limit_backoff_minutes: int | None,
          log_level: str | None, log_file: str | None,
          proxy_mode: bool,
          ca_dir: str | None, model_map_raw: tuple[str, ...],
          proxy_user: str | None, proxy_password: str | None,
          api_token_raw: tuple[str, ...], web_search_iterations: int | None,
          force_ddg_web_search: bool,
          web_search_model: str | None,
          reverse_dns_server: str | None,
          reverse_dns_sync_wait_ms: int | None,
          forwarded_allow_ips: str | None,
          spoof_interactive: bool,
          copilot_only_client: tuple[str, ...]):
    """Start the OpenAI-compatible API server."""
    import uvicorn

    from lib.account_manager import AccountManager
    from lib.auth import resolve_github_tokens
    from lib.server import init_app
    from lib.logging import build_runtime_logging_config
    from lib.anthropic_auth import resolve_anthropic_accounts
    import lib.anthropic_client as _anthropic_client_module

    # Spoof toggle must be set before AccountManager constructs AnthropicClient
    # instances so the first request out the door already uses the right UA.
    spoof_interactive = spoof_interactive or False
    _anthropic_client_module.SPOOF_INTERACTIVE = spoof_interactive

    # --- Load config file (lowest precedence) ---
    cfg = _load_config(config_path)

    # --- Merge: CLI/env > config file > defaults ---
    host = host or cfg.get("host", "127.0.0.1")
    port = port or cfg.get("port", 18080)
    workers = workers if workers is not None else cfg.get("workers", 1)
    strategy = strategy or cfg.get("strategy", "least-utilized")
    rate_limit_backoff_minutes = (
        rate_limit_backoff_minutes if rate_limit_backoff_minutes is not None
        else cfg.get("rate_limit_backoff_minutes", 60)
    )
    log_level = log_level or cfg.get("log_level", "info")
    log_file = log_file or cfg.get("log_file")
    proxy_mode = proxy_mode or cfg.get("proxy", False)
    ca_dir = Path(ca_dir) if ca_dir else cfg.get("ca_dir")
    if isinstance(ca_dir, str):
        ca_dir = Path(ca_dir)
    if not cors_origin:
        cors_origin = tuple(cfg.get("cors_origins", []))

    # --- Model map: CLI/env > config file > shipped model_map.json ---
    model_map_list: list[tuple[str, str]] | None = None
    if model_map_raw:
        model_map_list = []
        for raw in model_map_raw:
            for entry in raw.split(","):
                entry = entry.strip()
                if "=" in entry:
                    pat, _, tgt = entry.partition("=")
                    model_map_list.append((pat, tgt))
    elif "model_map" in cfg:
        model_map_list = list(cfg["model_map"].items())
    # Otherwise None → init_app will load the shipped default

    # --- Proxy auth: CLI/env > config file ---
    proxy_user = proxy_user or cfg.get("proxy_user")
    proxy_password = proxy_password or cfg.get("proxy_password")
    web_search_iterations = (web_search_iterations if web_search_iterations is not None
                             else cfg.get("web_search_iterations", 3))
    force_ddg_web_search = force_ddg_web_search or cfg.get("force_ddg_web_search", False)
    web_search_model = web_search_model or cfg.get("web_search_model")
    reverse_dns_server = reverse_dns_server or cfg.get("reverse_dns_server")
    reverse_dns_sync_wait_ms = (
        reverse_dns_sync_wait_ms if reverse_dns_sync_wait_ms is not None
        else cfg.get("reverse_dns_sync_wait_ms", 150)
    )
    forwarded_allow_ips = forwarded_allow_ips or cfg.get("forwarded_allow_ips")

    # --- API tokens: CLI/env > config file > stored tokens ---
    api_tokens: list[str] | None = None
    if api_token_raw:
        api_tokens = []
        for raw in api_token_raw:
            for t in raw.split(","):
                t = t.strip()
                if t:
                    api_tokens.append(t)
    elif "api_tokens" in cfg:
        api_tokens = list(cfg["api_tokens"])
    else:
        from lib.auth import get_api_token_values
        stored = get_api_token_values()
        if stored:
            api_tokens = stored

    # --- Resolve accounts ---
    cli_tokens = []
    for raw in github_token:
        for part in raw.split(","):
            part = part.strip()
            if part:
                cli_tokens.append(part)

    # Config file accounts
    cfg_accounts = cfg.get("accounts", [])

    cfg_tokens = []
    for entry in cfg_accounts:
        if isinstance(entry, str):
            cfg_tokens.append(entry)
        elif isinstance(entry, dict):
            token = entry.get("token")
            if token:
                cfg_tokens.append(token)

    # Resolve tokens → (token, username) via GitHub API validation.
    explicit_tokens = cli_tokens or cfg_tokens or None

    print("Resolving accounts...")
    # serve runs headless: never block on interactive device flow, and allow
    # zero Copilot accounts so the proxy can run on Anthropic / ChatGPT alone.
    resolved = resolve_github_tokens(
        explicit_tokens, interactive=False, required=False
    )

    accounts: list[dict] = []
    for token, username in resolved:
        accounts.append({
            "token": token,
            "username": username,
            "backend": "copilot",
        })

    # Anthropic accounts come from their own on-disk cache (claude.ai PKCE
    # flow). Each entry already has access_token / refresh_token / expires_at
    # — the AccountManager constructs an AnthropicTokenManager + AnthropicClient
    # per entry.
    anthropic_accounts = []
    try:
        for a in resolve_anthropic_accounts():
            anthropic_accounts.append({**a, "backend": "anthropic"})
            accounts.append({**a, "backend": "anthropic"})
    except Exception as exc:
        print(f"  Warning: could not load Anthropic accounts: {exc}")

    # ChatGPT (Codex) accounts from the device-flow cache.
    chatgpt_accounts = []
    try:
        from lib.openai_auth import resolve_chatgpt_accounts
        for a in resolve_chatgpt_accounts():
            chatgpt_accounts.append({**a, "backend": "chatgpt"})
            accounts.append({**a, "backend": "chatgpt"})
    except Exception as exc:
        print(f"  Warning: could not load ChatGPT accounts: {exc}")

    acct_mgr = AccountManager(
        accounts, strategy=strategy,
        rate_limit_backoff_seconds=rate_limit_backoff_minutes * 60,
    )

    # Verify all accounts can get a Copilot token
    try:
        asyncio.run(acct_mgr.verify_all())
    except Exception as e:
        raise click.ClickException(f"Failed to get Copilot token: {e}")

    n = len(accounts)
    n_anthropic = len(anthropic_accounts)
    n_chatgpt = len(chatgpt_accounts)
    n_copilot = n - n_anthropic - n_chatgpt
    print(f"\nConfigured {n} account(s) ({n_copilot} copilot, "
          f"{n_anthropic} anthropic, {n_chatgpt} chatgpt), strategy: {strategy}")
    for acct in acct_mgr.accounts:
        print(f"  - {acct.username} [{acct.backend}]")
    if spoof_interactive:
        print("\n** Spoof interactive headers: Anthropic-pool requests will "
              "carry the REPL's User-Agent / x-app / extended anthropic-beta **")
    if api_tokens:
        print(f"\n** API token protection enabled ({len(api_tokens)} token(s)) **")
    if proxy_user and proxy_password:
        print(f"\n** Forward proxy authentication enabled (user: {proxy_user}) **")
    print(f"\nStarting server on http://{host}:{port}\n")

    from uvicorn.config import LOGGING_CONFIG
    logging_config = build_runtime_logging_config(LOGGING_CONFIG, log_level, log_file, reverse_dns_server, reverse_dns_sync_wait_ms)

    # Copilot-only allowlist applies to both launch paths via the env (the
    # lifespan reads it for single-process init_app too).
    if copilot_only_client:
        os.environ["_COPILOT_ADAPTER_COPILOT_ONLY_CLIENTS"] = ",".join(copilot_only_client)
        print(f"\n** Copilot-only clients: {len(copilot_only_client)} rule(s) "
              "pinned to Copilot (skip Anthropic/ChatGPT pools) **")

    if proxy_mode and workers > 1:
        print("Warning: --proxy mode is not compatible with multiple workers, using 1 worker")
        workers = 1

    if workers > 1:
        # Workers initialize via the lifespan event using env vars
        # Only Copilot tokens go through the env; Anthropic/ChatGPT accounts are
        # reloaded from their on-disk caches in the worker lifespan (they carry
        # refresh tokens we don't serialize through the environment).
        parts = []
        for acct in accounts:
            if acct.get("backend", "copilot") != "copilot":
                continue
            parts.append(f"{acct['token']}:{acct['username']}")
        os.environ["_COPILOT_ADAPTER_GITHUB_TOKENS"] = ",".join(parts)
        os.environ["_COPILOT_ADAPTER_WORKER_INIT"] = "1"
        os.environ["_COPILOT_ADAPTER_STRATEGY"] = strategy
        if cors_origin:
            os.environ["_COPILOT_ADAPTER_CORS_ORIGINS"] = ",".join(cors_origin)
        if model_map_list is not None:
            os.environ["_COPILOT_ADAPTER_MODEL_MAP"] = ",".join(
                f"{p}={t}" for p, t in model_map_list
            )
        if api_tokens:
            os.environ["_COPILOT_ADAPTER_API_TOKENS"] = ",".join(api_tokens)
        if log_file:
            os.environ["_COPILOT_ADAPTER_LOG_FILE"] = log_file
        os.environ["_COPILOT_ADAPTER_WEB_SEARCH_MAX_ITERATIONS"] = str(web_search_iterations)
        if force_ddg_web_search:
            os.environ["_COPILOT_ADAPTER_FORCE_DDG_WEB_SEARCH"] = "1"
        if web_search_model:
            os.environ["_COPILOT_ADAPTER_WEB_SEARCH_MODEL"] = web_search_model
        if reverse_dns_server:
            os.environ["_COPILOT_ADAPTER_REVERSE_DNS_SERVER"] = reverse_dns_server
        if reverse_dns_sync_wait_ms is not None:
            os.environ["_COPILOT_ADAPTER_REVERSE_DNS_SYNC_WAIT_MS"] = str(reverse_dns_sync_wait_ms)
        if spoof_interactive:
            os.environ["_COPILOT_ADAPTER_SPOOF_INTERACTIVE"] = "1"

        uvicorn.run(
            "lib.server:app", host=host, port=port,
            workers=workers, log_level=log_level,
            timeout_graceful_shutdown=5,
            use_colors=_supports_color(),
            log_config=logging_config,
            forwarded_allow_ips=forwarded_allow_ips,
        )
    else:
        application = init_app(acct_mgr, cors_origins=list(cors_origin) or None,
                               model_map=model_map_list,
                               api_tokens=api_tokens,
                               web_search_max_iterations=web_search_iterations,
                               force_ddg_web_search=force_ddg_web_search,
                               web_search_model=web_search_model)
        if proxy_mode:
            from lib.forward_proxy import DualModeServer
            dual = DualModeServer(
                application, host=host, port=port,
                ca_dir=ca_dir,
                proxy_user=proxy_user,
                proxy_password=proxy_password,
                uvicorn_log_level=log_level,
                uvicorn_log_config=logging_config,
                uvicorn_use_colors=_supports_color(),
                timeout_graceful_shutdown=5,
            )
            asyncio.run(dual.serve())
        else:
            uvicorn.run(application, host=host, port=port, log_level=log_level,
                        timeout_graceful_shutdown=5,
                        use_colors=_supports_color(),
                        log_config=logging_config,
                        forwarded_allow_ips=forwarded_allow_ips)


if __name__ == "__main__":
    main()

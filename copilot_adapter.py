"""CLI entry point for copilot-api."""

import json
import os
import sys
from pathlib import Path

import click

_NUM_CPUS = os.cpu_count() or 1
_DEFAULT_CONFIG = Path.home() / ".copilot-adapter.json"
_VALID_PLANS = ("free", "pro", "pro+", "business", "enterprise")


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


def _parse_token_spec(raw: str) -> dict:
    """Parse a token spec like ``ghp_xxx`` or ``ghp_xxx:free:50:10.5``.

    Returns a dict with keys ``token`` and optionally ``plan``,
    ``quota_limit``, and ``premium_used``.
    """
    parts = raw.split(":")
    result: dict = {"token": parts[0]}
    if len(parts) >= 2 and parts[1]:
        result["plan"] = parts[1]
    if len(parts) >= 3 and parts[2]:
        result["quota_limit"] = int(parts[2])
    if len(parts) >= 4 and parts[3]:
        result["premium_used"] = float(parts[3])
    return result


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


@main.command()
@click.option("--add", "add_token", default=None, metavar="TOKEN",
              help="Add a GitHub PAT to the cached accounts.")
@click.option("--remove", "remove_username", default=None, metavar="USER",
              help="Remove a cached account by username.")
@click.option("--update", "update_username", default=None, metavar="USER",
              help="Update settings for a cached account by username.")
@click.option("--plan", "update_plan", default=None,
              type=click.Choice(list(_VALID_PLANS)),
              help="Set the Copilot plan for the account.")
@click.option("--quota-limit", "update_quota", default=None, type=int, metavar="N",
              help="Set the monthly premium request quota for the account.")
@click.option("--usage", "update_usage", default=None, type=float, metavar="N",
              help="Set the current premium request usage for the account.")
def accounts(add_token: str | None, remove_username: str | None,
             update_username: str | None, update_plan: str | None,
             update_quota: int | None, update_usage: float | None):
    """Manage cached accounts: list, add, remove, or update."""
    from lib.auth import (add_account, list_accounts,
                          remove_account, update_account)

    if add_token:
        result = add_account(add_token, plan=update_plan, quota_limit=update_quota,
                             premium_used=update_usage)
        if result is None:
            raise click.ClickException("Invalid token — could not authenticate with GitHub")
        print(f"Added {result['username']} (plan: {result['plan']}, "
              f"quota: {result['quota_limit']}, usage: {result['premium_used']})")
        return

    if remove_username:
        if not remove_account(remove_username):
            raise click.ClickException(f"Account '{remove_username}' not found in cache")
        print(f"Removed {remove_username}")
        return

    if update_username:
        if update_plan is None and update_quota is None and update_usage is None:
            raise click.UsageError("--update requires --plan, --quota-limit, and/or --usage")
        if not update_account(update_username, plan=update_plan, quota_limit=update_quota,
                              premium_used=update_usage):
            raise click.ClickException(f"Account '{update_username}' not found in cache")
        print(f"Updated {update_username}:")
        if update_plan is not None:
            print(f"  plan: {update_plan}")
        if update_quota is not None:
            print(f"  quota: {update_quota}")
        if update_usage is not None:
            print(f"  usage: {update_usage}")
        return

    accts = list_accounts()
    if not accts:
        print("No cached accounts.")
        return
    print(f"Cached accounts ({len(accts)}):")
    for acct in accts:
        status = "valid" if acct["valid"] else "expired/invalid"
        plan = acct.get("plan") or "unset"
        quota = acct.get("quota_limit")
        quota_str = str(quota) if quota is not None else "unset"
        usage = acct.get("premium_used", 0)
        print(f"  - {acct['username']} ({status}, plan: {plan}, usage: {usage}/{quota_str})")


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
              help="GitHub PAT (repeatable, supports TOKEN:PLAN:QUOTA:USAGE format). "
                   "Env var supports comma-separated values.")
@click.option("--cors-origin", multiple=True, envvar="COPILOT_ADAPTER_CORS_ORIGIN",
              metavar="ORIGIN",
              help="Allowed CORS origin (repeatable). Use '*' to allow all origins.")
@click.option("--workers", default=None, type=int, envvar="COPILOT_ADAPTER_WORKERS",
              metavar="N",
              help=f"Number of worker processes (default: number of CPUs, {_NUM_CPUS}).")
@click.option("--strategy", default=None,
              type=click.Choice(["max-usage", "min-usage", "round-robin"]),
              envvar="COPILOT_ADAPTER_STRATEGY",
              help="Account rotation strategy (default: max-usage).")
@click.option("--quota-limit", default=None, type=int,
              envvar="COPILOT_ADAPTER_QUOTA_LIMIT", metavar="N",
              help="Default monthly premium request limit per account (default: 300).")
@click.option("--plan", default=None,
              type=click.Choice(list(_VALID_PLANS)),
              envvar="COPILOT_ADAPTER_PLAN",
              help="Default Copilot plan type for premium request multipliers (default: pro).")
@click.option("--log-level", default=None,
              type=click.Choice(["debug", "info", "warning", "error"], case_sensitive=False),
              envvar="COPILOT_ADAPTER_LOG_LEVEL",
              help="Logging level (default: info). Use 'debug' for verbose output.")
def serve(config_path: str | None, host: str | None, port: int | None,
          github_token: tuple[str, ...], cors_origin: tuple[str, ...],
          workers: int | None, strategy: str | None,
          quota_limit: int | None, plan: str | None,
          log_level: str | None):
    """Start the OpenAI-compatible API server."""
    import uvicorn

    from lib.account_manager import AccountManager
    from lib.auth import get_cached_account_meta, resolve_github_tokens
    from lib.server import init_app

    # --- Load config file (lowest precedence) ---
    cfg = _load_config(config_path)

    # --- Merge: CLI/env > config file > defaults ---
    host = host or cfg.get("host", "127.0.0.1")
    port = port or cfg.get("port", 18080)
    workers = workers if workers is not None else cfg.get("workers", _NUM_CPUS)
    strategy = strategy or cfg.get("strategy", "max-usage")
    quota_limit = quota_limit if quota_limit is not None else cfg.get("quota_limit")
    plan = plan or cfg.get("plan", "pro")
    log_level = log_level or cfg.get("log_level", "info")
    if not cors_origin:
        cors_origin = tuple(cfg.get("cors_origins", []))

    # --- Resolve accounts ---
    # CLI/env tokens (may include :plan:quota:usage annotations)
    cli_token_specs = []
    for raw in github_token:
        for part in raw.split(","):
            part = part.strip()
            if part:
                cli_token_specs.append(_parse_token_spec(part))

    # Config file accounts
    cfg_accounts = cfg.get("accounts", [])

    # Build a per-account overrides map keyed by token value.
    # Cached metadata first (lowest), then config file, then CLI (highest).
    cached_meta = get_cached_account_meta()
    account_overrides: dict[str, dict] = {}

    # Layer 1: cached device-flow metadata
    for token, meta in cached_meta.items():
        account_overrides[token] = {"token": token, **meta}

    # Layer 2: config file entries
    for entry in cfg_accounts:
        if isinstance(entry, str):
            spec = _parse_token_spec(entry)
        elif isinstance(entry, dict):
            spec = dict(entry)
        else:
            continue
        token = spec["token"]
        if token in account_overrides:
            account_overrides[token].update(spec)
        else:
            account_overrides[token] = spec

    # Layer 3: CLI/env token specs
    for spec in cli_token_specs:
        token = spec["token"]
        if token in account_overrides:
            account_overrides[token].update(spec)
        else:
            account_overrides[token] = spec

    # Resolve tokens → (token, username) via GitHub API validation.
    explicit_tokens = [s["token"] for s in cli_token_specs] or None
    if not explicit_tokens and cfg_accounts:
        explicit_tokens = [s["token"] for s in account_overrides.values()]

    print("Resolving accounts...")
    resolved = resolve_github_tokens(explicit_tokens)

    # Build rich account dicts with per-account plan/quota/usage
    accounts: list[dict] = []
    for token, username in resolved:
        overrides = account_overrides.get(token, {})
        accounts.append({
            "token": token,
            "username": username,
            "plan": overrides.get("plan", plan),
            "quota_limit": overrides.get("quota_limit", quota_limit),
            "premium_used": overrides.get("premium_used", 0),
        })

    acct_mgr = AccountManager(
        accounts, strategy=strategy, quota_limit=quota_limit, plan=plan,
    )

    # Verify all accounts can get a Copilot token
    import asyncio
    try:
        asyncio.run(acct_mgr.verify_all())
    except Exception as e:
        raise click.ClickException(f"Failed to get Copilot token: {e}")

    n = len(accounts)
    print(f"\nConfigured {n} account(s), strategy: {strategy}")
    for acct in acct_mgr.accounts:
        limit_str = str(acct.premium_limit) if acct.premium_limit is not None else "unset"
        print(f"  - {acct.username} (plan: {acct.plan}, usage: {acct.premium_used}/{limit_str})")
    print(f"\nStarting server on http://{host}:{port}")
    print(f"  POST /v1/chat/completions                       (OpenAI)")
    print(f"  POST /v1/responses                              (OpenAI)")
    print(f"  POST /v1/messages                               (Anthropic)")
    print(f"  POST /v1beta/models/MODEL:generateContent       (Gemini)")
    print(f"  POST /v1beta/models/MODEL:streamGenerateContent (Gemini)")
    print(f"  GET  /v1/models")
    print(f"  GET  /v1beta/models")
    print(f"  POST /v1/embeddings\n")

    if workers > 1:
        # Workers initialize via the lifespan event using env vars
        # Format: "token1:username1:plan1:quota1:usage1,..."
        parts = []
        for acct in accounts:
            limit_str = str(acct["quota_limit"]) if acct["quota_limit"] is not None else ""
            used_str = str(acct["premium_used"]) if acct["premium_used"] else ""
            parts.append(f"{acct['token']}:{acct['username']}:{acct['plan']}:{limit_str}:{used_str}")
        os.environ["_COPILOT_ADAPTER_GITHUB_TOKENS"] = ",".join(parts)
        os.environ["_COPILOT_ADAPTER_STRATEGY"] = strategy
        if quota_limit is not None:
            os.environ["_COPILOT_ADAPTER_QUOTA_LIMIT"] = str(quota_limit)
        os.environ["_COPILOT_ADAPTER_PLAN"] = plan
        if cors_origin:
            os.environ["_COPILOT_ADAPTER_CORS_ORIGINS"] = ",".join(cors_origin)

        # Custom log config to suppress repetitive per-worker lifespan messages
        from uvicorn.config import LOGGING_CONFIG
        LOGGING_CONFIG["filters"] = {
            "no_lifespan": {"()": "lib.logging.LifespanFilter"},
        }
        LOGGING_CONFIG["handlers"]["default"].setdefault("filters", [])
        if "no_lifespan" not in LOGGING_CONFIG["handlers"]["default"]["filters"]:
            LOGGING_CONFIG["handlers"]["default"]["filters"].append("no_lifespan")

        uvicorn.run(
            "lib.server:app", host=host, port=port,
            workers=workers, log_level=log_level,
            use_colors=_supports_color(),
        )
    else:
        application = init_app(acct_mgr, cors_origins=list(cors_origin) or None)
        uvicorn.run(application, host=host, port=port, log_level=log_level,
                    use_colors=_supports_color())


if __name__ == "__main__":
    main()

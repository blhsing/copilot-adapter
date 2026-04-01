"""CLI entry point for copilot-api."""

import json
import os
from pathlib import Path

import click

_NUM_CPUS = os.cpu_count() or 1
_DEFAULT_CONFIG = Path.home() / ".copilot-adapter.json"
_VALID_PLANS = ("free", "pro", "pro+", "business", "enterprise")


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
    """Parse a token spec like ``ghp_xxx`` or ``ghp_xxx:free:50``.

    Returns a dict with keys ``token`` and optionally ``plan`` and
    ``quota_limit``.
    """
    parts = raw.split(":")
    result: dict = {"token": parts[0]}
    if len(parts) >= 2 and parts[1]:
        result["plan"] = parts[1]
    if len(parts) >= 3 and parts[2]:
        result["quota_limit"] = int(parts[2])
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
@click.option("--username", default=None,
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
def accounts():
    """List all cached device-flow accounts."""
    from lib.auth import list_accounts
    accts = list_accounts()
    if not accts:
        print("No cached accounts.")
        return
    print(f"Cached accounts ({len(accts)}):")
    for acct in accts:
        status = "valid" if acct["valid"] else "expired/invalid"
        print(f"  - {acct['username']} ({status})")


@main.command()
@click.option("--config", "config_path", default=None,
              envvar="COPILOT_ADAPTER_CONFIG",
              help=f"Path to JSON config file (default: {_DEFAULT_CONFIG}).")
@click.option("--host", default=None, envvar="COPILOT_ADAPTER_HOST",
              help="Host to bind to.")
@click.option("--port", default=None, type=int, envvar="COPILOT_ADAPTER_PORT",
              help="Port to bind to.")
@click.option("--github-token", multiple=True, envvar="COPILOT_ADAPTER_GITHUB_TOKEN",
              help="GitHub PAT (repeatable, supports TOKEN:PLAN:QUOTA format). "
                   "Env var supports comma-separated values.")
@click.option("--cors-origin", multiple=True, envvar="COPILOT_ADAPTER_CORS_ORIGIN",
              help="Allowed CORS origin (repeatable). Use '*' to allow all origins.")
@click.option("--workers", default=None, type=int, envvar="COPILOT_ADAPTER_WORKERS",
              help=f"Number of worker processes (default: number of CPUs, {_NUM_CPUS}).")
@click.option("--strategy", default=None,
              type=click.Choice(["max-usage", "min-usage", "round-robin"]),
              envvar="COPILOT_ADAPTER_STRATEGY",
              help="Account rotation strategy (default: max-usage).")
@click.option("--quota-limit", default=None, type=int,
              envvar="COPILOT_ADAPTER_QUOTA_LIMIT",
              help="Default monthly premium request limit per account (default: 300).")
@click.option("--local-tracking", is_flag=True, default=False,
              envvar="COPILOT_ADAPTER_LOCAL_TRACKING",
              help="Track usage locally instead of polling the GitHub billing API. "
                   "Assumes this server is the only consumer of the quota.")
@click.option("--plan", default=None,
              type=click.Choice(list(_VALID_PLANS)),
              envvar="COPILOT_ADAPTER_PLAN",
              help="Default Copilot plan type for premium request multipliers (default: pro).")
def serve(config_path: str | None, host: str | None, port: int | None,
          github_token: tuple[str, ...], cors_origin: tuple[str, ...],
          workers: int | None, strategy: str | None,
          quota_limit: int | None, local_tracking: bool, plan: str | None):
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
    if not local_tracking:
        local_tracking = cfg.get("local_tracking", False)
    if not cors_origin:
        cors_origin = tuple(cfg.get("cors_origins", []))

    # --- Resolve accounts ---
    # CLI/env tokens (may include :plan:quota annotations)
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

    # Build rich account dicts with per-account plan/quota
    accounts: list[dict] = []
    for token, username in resolved:
        overrides = account_overrides.get(token, {})
        accounts.append({
            "token": token,
            "username": username,
            "plan": overrides.get("plan", plan),
            "quota_limit": overrides.get("quota_limit", quota_limit),
        })

    acct_mgr = AccountManager(
        accounts, strategy=strategy, quota_limit=quota_limit,
        local_tracking=local_tracking, plan=plan,
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
        print(f"  - {acct.username} (plan: {acct.plan}, quota: {limit_str})")
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
        # Format: "token1:username1:plan1:quota1,token2:username2:plan2:quota2"
        parts = []
        for acct in accounts:
            limit_str = str(acct["quota_limit"]) if acct["quota_limit"] is not None else ""
            parts.append(f"{acct['token']}:{acct['username']}:{acct['plan']}:{limit_str}")
        os.environ["_COPILOT_ADAPTER_GITHUB_TOKENS"] = ",".join(parts)
        os.environ["_COPILOT_ADAPTER_STRATEGY"] = strategy
        if quota_limit is not None:
            os.environ["_COPILOT_ADAPTER_QUOTA_LIMIT"] = str(quota_limit)
        if local_tracking:
            os.environ["_COPILOT_ADAPTER_LOCAL_TRACKING"] = "1"
        os.environ["_COPILOT_ADAPTER_PLAN"] = plan
        if cors_origin:
            os.environ["_COPILOT_ADAPTER_CORS_ORIGINS"] = ",".join(cors_origin)
        uvicorn.run(
            "lib.server:app", host=host, port=port,
            workers=workers, log_level="info",
        )
    else:
        application = init_app(acct_mgr, cors_origins=list(cors_origin) or None)
        uvicorn.run(application, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()

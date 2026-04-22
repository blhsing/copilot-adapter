"""CLI entry point for copilot-adapter."""

import asyncio
import json
import os
import sys
from pathlib import Path

import click

from lib.configure import CONFIGURATORS

_DEFAULT_CONFIG = Path.home() / ".config" / "copilot-adapter" / "config.json"
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
    """Parse a token spec like ``ghu_xxx`` or ``ghu_xxx:free:50:10.5``.

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
              help="Add a GitHub OAuth token (ghu_) to the cached accounts. PATs (ghp_) are rejected by the Copilot API.")
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
        usage = round(acct.get("premium_used", 0), 2)
        print(f"  - {acct['username']} ({status}, plan: {plan}, usage: {usage}/{quota_str})")


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
              help="GitHub OAuth token (ghu_) from device-flow login (repeatable, supports TOKEN:PLAN:QUOTA:USAGE format). "
                   "PATs (ghp_) are rejected — the Copilot API returns 404 for them. "
                   "Env var supports comma-separated values.")
@click.option("--cors-origin", multiple=True, envvar="COPILOT_ADAPTER_CORS_ORIGIN",
              metavar="ORIGIN",
              help="Allowed CORS origin (repeatable). Use '*' to allow all origins.")
@click.option("--workers", default=None, type=int, envvar="COPILOT_ADAPTER_WORKERS",
              metavar="N",
              help="Number of worker processes (default: 1).")
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
@click.option("--log-file", default=None, metavar="PATH",
              envvar="COPILOT_ADAPTER_LOG_FILE",
              help="Append logs to PATH in addition to the console.")
@click.option("--free", "force_free", is_flag=True, default=False,
              envvar="COPILOT_ADAPTER_FREE",
              help="Mark all requests as agent-initiated so nothing counts as a premium request.")
@click.option("--free-within-minutes", type=float, default=None, metavar="N",
              envvar="COPILOT_ADAPTER_FREE_WITHIN_MINUTES",
              help="Mark user requests as agent-initiated if the last request "
                   "was less than N minutes ago. Mutually exclusive with --free.")
@click.option("--stub-bill", "stub_bill", is_flag=True, default=False,
              envvar="COPILOT_ADAPTER_STUB_BILL",
              help="For each user-initiated request, first fire a tiny billed "
                   "stub call against --stub-model, then run the real request "
                   "as agent-initiated. Falls through to normal billing if the "
                   "stub call fails.")
@click.option("--stub-model", default=None, metavar="MODEL",
              envvar="COPILOT_ADAPTER_STUB_MODEL",
              help="Model used by --stub-bill (default: claude-haiku-4.5).")
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
def serve(config_path: str | None, host: str | None, port: int | None,
          github_token: tuple[str, ...], cors_origin: tuple[str, ...],
          workers: int | None, strategy: str | None,
          quota_limit: int | None, plan: str | None,
          log_level: str | None, log_file: str | None, force_free: bool,
          free_within_minutes: float | None,
          stub_bill: bool, stub_model: str | None,
          proxy_mode: bool,
          ca_dir: str | None, model_map_raw: tuple[str, ...],
          proxy_user: str | None, proxy_password: str | None,
          api_token_raw: tuple[str, ...], web_search_iterations: int | None):
    """Start the OpenAI-compatible API server."""
    import uvicorn

    from lib.account_manager import AccountManager
    from lib.auth import get_cached_account_meta, resolve_github_tokens
    from lib.server import init_app
    from lib.logging import build_runtime_logging_config

    # --- Load config file (lowest precedence) ---
    cfg = _load_config(config_path)

    # --- Merge: CLI/env > config file > defaults ---
    host = host or cfg.get("host", "127.0.0.1")
    port = port or cfg.get("port", 18080)
    workers = workers if workers is not None else cfg.get("workers", 1)
    strategy = strategy or cfg.get("strategy", "max-usage")
    quota_limit = quota_limit if quota_limit is not None else cfg.get("quota_limit")
    plan = plan or cfg.get("plan", "pro")
    log_level = log_level or cfg.get("log_level", "info")
    log_file = log_file or cfg.get("log_file")
    force_free = force_free or cfg.get("free", False)
    free_within_minutes = (free_within_minutes if free_within_minutes is not None
                           else cfg.get("free_within_minutes"))
    stub_bill = stub_bill or cfg.get("stub_bill", False)
    stub_model = stub_model or cfg.get("stub_model") or "claude-haiku-4.5"
    if force_free and free_within_minutes is not None:
        raise click.UsageError("--free and --free-within-minutes are mutually exclusive.")
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
    try:
        asyncio.run(acct_mgr.verify_all())
    except Exception as e:
        raise click.ClickException(f"Failed to get Copilot token: {e}")

    n = len(accounts)
    print(f"\nConfigured {n} account(s), strategy: {strategy}")
    for acct in acct_mgr.accounts:
        limit_str = str(acct.premium_limit) if acct.premium_limit is not None else "unset"
        print(f"  - {acct.username} (plan: {acct.plan}, usage: {acct.premium_used}/{limit_str})")
    if force_free:
        print("\n** Free mode enabled: all requests will be marked as agent-initiated **")
    if free_within_minutes is not None:
        print(f"\n** Time-based free mode: requests within {free_within_minutes} min of last → agent **")
    if stub_bill:
        print(f"\n** Stub-bill mode: user requests billed via {stub_model}, real request runs as agent **")
    if api_tokens:
        print(f"\n** API token protection enabled ({len(api_tokens)} token(s)) **")
    if proxy_user and proxy_password:
        print(f"\n** Forward proxy authentication enabled (user: {proxy_user}) **")
    print(f"\nStarting server on http://{host}:{port}\n")

    from uvicorn.config import LOGGING_CONFIG
    logging_config = build_runtime_logging_config(LOGGING_CONFIG, log_level, log_file)

    if proxy_mode and workers > 1:
        print("Warning: --proxy mode is not compatible with multiple workers, using 1 worker")
        workers = 1

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
        if force_free:
            os.environ["_COPILOT_ADAPTER_FREE"] = "1"
        if free_within_minutes is not None:
            os.environ["_COPILOT_ADAPTER_FREE_WITHIN_MINUTES"] = str(free_within_minutes)
        if stub_bill:
            os.environ["_COPILOT_ADAPTER_STUB_BILL"] = "1"
            os.environ["_COPILOT_ADAPTER_STUB_MODEL"] = stub_model
        if model_map_list is not None:
            os.environ["_COPILOT_ADAPTER_MODEL_MAP"] = ",".join(
                f"{p}={t}" for p, t in model_map_list
            )
        if api_tokens:
            os.environ["_COPILOT_ADAPTER_API_TOKENS"] = ",".join(api_tokens)
        if log_file:
            os.environ["_COPILOT_ADAPTER_LOG_FILE"] = log_file
        os.environ["_COPILOT_ADAPTER_WEB_SEARCH_MAX_ITERATIONS"] = str(web_search_iterations)

        uvicorn.run(
            "lib.server:app", host=host, port=port,
            workers=workers, log_level=log_level,
            timeout_graceful_shutdown=5,
            use_colors=_supports_color(),
            log_config=logging_config,
        )
    else:
        application = init_app(acct_mgr, cors_origins=list(cors_origin) or None,
                               force_free=force_free,
                               free_within_minutes=free_within_minutes,
                               stub_bill=stub_bill,
                               stub_model=stub_model,
                               model_map=model_map_list,
                               api_tokens=api_tokens,
                               web_search_max_iterations=web_search_iterations)
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
                        log_config=logging_config)


if __name__ == "__main__":
    main()

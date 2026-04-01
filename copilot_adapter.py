"""CLI entry point for copilot-api."""

import os

import click

_NUM_CPUS = os.cpu_count() or 1


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
@click.option("--host", default="127.0.0.1", envvar="COPILOT_ADAPTER_HOST",
              help="Host to bind to.")
@click.option("--port", default=18080, type=int, envvar="COPILOT_ADAPTER_PORT",
              help="Port to bind to.")
@click.option("--github-token", multiple=True, envvar="COPILOT_ADAPTER_GITHUB_TOKEN",
              help="GitHub PAT (repeatable). Env var supports comma-separated values.")
@click.option("--cors-origin", multiple=True, envvar="COPILOT_ADAPTER_CORS_ORIGIN",
              help="Allowed CORS origin (repeatable). Use '*' to allow all origins.")
@click.option("--workers", default=_NUM_CPUS, type=int, envvar="COPILOT_ADAPTER_WORKERS",
              help=f"Number of worker processes (default: number of CPUs, {_NUM_CPUS}).")
@click.option("--strategy", default="max-usage",
              type=click.Choice(["max-usage", "min-usage", "round-robin"]),
              envvar="COPILOT_ADAPTER_STRATEGY",
              help="Account rotation strategy (default: max-usage).")
@click.option("--quota-limit", default=None, type=int,
              envvar="COPILOT_ADAPTER_QUOTA_LIMIT",
              help="Monthly premium request limit per account for proactive switching.")
@click.option("--local-tracking", is_flag=True, default=False,
              envvar="COPILOT_ADAPTER_LOCAL_TRACKING",
              help="Track usage locally instead of polling the GitHub billing API. "
                   "Assumes this server is the only consumer of the quota.")
@click.option("--plan", default="paid",
              type=click.Choice(["paid", "free"]),
              envvar="COPILOT_ADAPTER_PLAN",
              help="Copilot plan type for correct premium request multipliers (default: paid).")
def serve(host: str, port: int, github_token: tuple[str, ...],
          cors_origin: tuple[str, ...], workers: int, strategy: str,
          quota_limit: int | None, local_tracking: bool, plan: str):
    """Start the OpenAI-compatible API server."""
    import uvicorn

    from lib.account_manager import AccountManager
    from lib.auth import resolve_github_tokens
    from lib.server import init_app

    print("Resolving accounts...")
    token_list = list(github_token) if github_token else None
    accounts = resolve_github_tokens(token_list)

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
    print(f"Starting server on http://{host}:{port}")
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
        # Format: "token1:username1,token2:username2"
        pairs = [f"{t}:{u}" for t, u in accounts]
        os.environ["_COPILOT_ADAPTER_GITHUB_TOKENS"] = ",".join(pairs)
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

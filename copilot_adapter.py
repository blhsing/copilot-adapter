"""CLI entry point for copilot-api."""

import click


@click.group()
def main():
    """OpenAI-compatible API server backed by GitHub Copilot."""


@main.command()
def login():
    """Authenticate with GitHub using the device flow."""
    from lib.auth import device_flow_login
    device_flow_login()


@main.command()
def logout():
    """Remove stored credentials."""
    from lib.auth import logout as do_logout
    do_logout()


@main.command()
@click.option("--host", default="127.0.0.1", help="Host to bind to.")
@click.option("--port", default=8080, type=int, help="Port to bind to.")
@click.option("--github-token", default=None, envvar="GITHUB_TOKEN",
              help="GitHub PAT or OAuth token. Falls back to GITHUB_TOKEN env var, "
                   "then cached token, then interactive device flow.")
@click.option("--cors-origin", multiple=True,
              help="Allowed CORS origin (repeatable). Use '*' to allow all origins.")
def serve(host: str, port: int, github_token: str | None, cors_origin: tuple[str, ...]):
    """Start the OpenAI-compatible API server."""
    import uvicorn

    from lib.auth import CopilotTokenManager, resolve_github_token
    from lib.server import init_app

    github_token = resolve_github_token(github_token)
    tm = CopilotTokenManager(github_token)

    # Verify we can get a Copilot token
    try:
        tm.get_token()
    except Exception as e:
        raise click.ClickException(f"Failed to get Copilot token: {e}")

    print(f"\nStarting server on http://{host}:{port}")
    print(f"  POST /v1/chat/completions                       (OpenAI)")
    print(f"  POST /v1/responses                              (OpenAI)")
    print(f"  POST /v1/messages                               (Anthropic)")
    print(f"  POST /v1beta/models/MODEL:generateContent       (Gemini)")
    print(f"  POST /v1beta/models/MODEL:streamGenerateContent (Gemini)")
    print(f"  GET  /v1/models")
    print(f"  GET  /v1beta/models")
    print(f"  POST /v1/embeddings\n")

    application = init_app(tm, cors_origins=list(cors_origin) or None)
    uvicorn.run(application, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()

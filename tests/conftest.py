"""Shared fixtures for integration tests.

Running the tests requires a valid GitHub Copilot session.  The first time you
run the suite you will be prompted to authenticate via the GitHub device flow
(a browser window opens automatically).  Subsequent runs reuse the cached
token stored in ``~/.config/copilot-api/token.json``.
"""

import pytest
import pytest_asyncio

from lib.adapters import AnthropicAdapter, GeminiAdapter, OpenAIAdapter
from lib.auth import CopilotTokenManager, device_flow_login
from lib.client import CopilotClient


# ---------------------------------------------------------------------------
# Authentication – runs once per session
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def github_token() -> str:
    """Authenticate with GitHub (interactive on first run)."""
    return device_flow_login()


@pytest.fixture(scope="session")
def token_manager(github_token: str) -> CopilotTokenManager:
    tm = CopilotTokenManager(github_token)
    # Eagerly verify the token works
    tm.get_token()
    return tm


@pytest.fixture(scope="session")
def client(token_manager: CopilotTokenManager) -> CopilotClient:
    return CopilotClient(token_manager)


# ---------------------------------------------------------------------------
# Adapters
# ---------------------------------------------------------------------------

@pytest.fixture
def openai_adapter() -> OpenAIAdapter:
    return OpenAIAdapter()


@pytest.fixture
def anthropic_adapter() -> AnthropicAdapter:
    return AnthropicAdapter()


# ---------------------------------------------------------------------------
# A model known to be available via Copilot
# ---------------------------------------------------------------------------

# Cheapest models per category (0 premium requests / lowest tier)
CHAT_MODEL = "gpt-4o-mini"
RESPONSES_MODEL = "gpt-5-mini"
EMBEDDINGS_MODEL = "text-embedding-3-small"


@pytest_asyncio.fixture(scope="session")
async def available_model(client: CopilotClient) -> str:
    """Return the cheapest chat model available on the Copilot API."""
    resp = await client.list_models()
    assert resp.status_code == 200, f"Failed to list models: {resp.text}"
    models = resp.json().get("data", [])
    assert models, "No models returned by the Copilot API"
    # Verify our preferred cheap model exists
    ids = {m.get("id", "") for m in models}
    if CHAT_MODEL in ids:
        return CHAT_MODEL
    # Fallback to gpt-4o-mini dated variant
    for mid in ids:
        if "gpt-4o-mini" in mid:
            return mid
    return models[0]["id"]


# ---------------------------------------------------------------------------
# FastAPI test client (for endpoint-level tests)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def test_app(token_manager: CopilotTokenManager):
    from lib.server import init_app
    return init_app(token_manager)


@pytest.fixture(scope="session")
def http_client(test_app):
    """Synchronous httpx client wired to the FastAPI app (no real server)."""
    import httpx
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=test_app),
        base_url="http://testserver",
    )

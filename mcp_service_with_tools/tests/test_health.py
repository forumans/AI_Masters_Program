import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, patch
from httpx import ASGITransport, AsyncClient


@pytest_asyncio.fixture
async def client():
    with patch("src.db.connection.init_pool", new_callable=AsyncMock), \
         patch("src.db.connection.close_pool", new_callable=AsyncMock):
        from src.main import app
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            yield c


@pytest.mark.asyncio
async def test_health_returns_ok(client):
    response = await client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.asyncio
async def test_mcp_requires_api_key(client):
    response = await client.post("/mcp")
    assert response.status_code == 403


@pytest.mark.asyncio
async def test_mcp_accepts_valid_api_key(client):
    # A valid key gets past auth — MCP may reject the payload but not with 403
    response = await client.post("/mcp", headers={"X-API-Key": "test-api-key"})
    assert response.status_code != 403

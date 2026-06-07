import os
from unittest.mock import AsyncMock, MagicMock

import asyncpg
import pytest

# Force-set test env vars (not setdefault): some pytest plugins (e.g. deepeval)
# auto-load the project's .env at import time, which would otherwise win and
# leave settings.api_key pointing at the real dev key instead of this test value.
os.environ["DATABASE_URL"] = "postgresql://healthcare_user:DevPassword@localhost:5432/healthcare_saas_db_local"
os.environ["API_KEY"] = "test-api-key"


@pytest.fixture
def mock_pool_and_conn():
    """Return a (pool, conn) pair where conn.fetch is an AsyncMock."""
    conn = AsyncMock(spec=asyncpg.Connection)

    pool = MagicMock(spec=asyncpg.Pool)
    pool.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)
    pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)

    return pool, conn

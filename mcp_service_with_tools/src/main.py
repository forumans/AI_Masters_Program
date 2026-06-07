from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.db.connection import close_pool, init_pool
from src.middleware.auth import APIKeyMiddleware
from src.mcp_server import mcp


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_pool()
    yield
    await close_pool()


app = FastAPI(title="Healthcare MCP Service", lifespan=lifespan)

app.add_middleware(APIKeyMiddleware)

# Mount the MCP server as an ASGI sub-app at /mcp.
# All MCP protocol traffic (tool calls, tool listing) is handled here.
# FastAPI owns auth, logging, and lifecycle — MCP tools stay auth-unaware.
app.mount("/mcp", mcp.http_app())


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}

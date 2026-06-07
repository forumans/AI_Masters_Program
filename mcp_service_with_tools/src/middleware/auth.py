from starlette.responses import Response
from starlette.types import ASGIApp, Receive, Scope, Send

from src.config import settings


class APIKeyMiddleware:
    """Pure ASGI middleware — streaming-safe, no response buffering."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http" and scope.get("path", "").startswith("/mcp"):
            headers = {k.lower(): v for k, v in scope.get("headers", [])}
            api_key = headers.get(b"x-api-key", b"").decode()
            if api_key != settings.api_key:
                response = Response("Invalid or missing API key", status_code=403)
                await response(scope, receive, send)
                return
        await self.app(scope, receive, send)

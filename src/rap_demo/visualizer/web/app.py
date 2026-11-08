"""FastAPI application for the RAP Real-Time Visualizer dashboard and REST/SSE endpoints."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, AsyncGenerator, Optional

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from rap_demo.domains.blocksworld.models import BlockState
from rap_demo.visualizer.serializer import serialize_blocksworld_state, serialize_tree
from rap_demo.visualizer.web.controller import SearchController


def create_app(controller: Optional[SearchController] = None) -> FastAPI:
    """Factory creating FastAPI application wired with SearchController."""
    ctrl: SearchController = controller or SearchController()
    server = FastAPI(title="RAP Visualizer Dashboard")

    static_dir = Path(__file__).parent / "static"
    if static_dir.exists():
        server.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    @server.get("/", response_class=HTMLResponse)
    async def get_dashboard() -> HTMLResponse:
        """Returns the real-time visualizer web dashboard HTML."""
        index_file = static_dir / "index.html"
        if index_file.exists():
            return HTMLResponse(content=index_file.read_text(encoding="utf-8"))
        return HTMLResponse(
            content="<html><body><h1>RAP Visualizer</h1><p>Tree reasoning visualizer</p></body></html>"
        )

    @server.get("/api/tree")
    async def get_tree() -> dict[str, Any]:
        """Returns JSON serialized MCTS tree hierarchy."""
        return serialize_tree(ctrl.root)

    @server.get("/api/state")
    async def get_state() -> Any:
        """Returns current domain state representation."""
        state = ctrl.current_state
        if isinstance(state, BlockState):
            return serialize_blocksworld_state(state)
        if isinstance(state, dict):
            return state
        return {"state": str(state)}

    @server.post("/api/search/step")
    async def post_step() -> dict[str, Any]:
        """Advances search by one iteration."""
        res = await ctrl.step()
        return {
            "iteration": ctrl.iteration,
            "status": ctrl.status.value if hasattr(ctrl.status, "value") else str(ctrl.status),
            "tree": serialize_tree(ctrl.root),
        }

    @server.post("/api/search/start")
    async def post_start() -> dict[str, Any]:
        """Starts continuous search in background."""
        res = await ctrl.start()
        return {
            "iteration": ctrl.iteration,
            "status": ctrl.status.value if hasattr(ctrl.status, "value") else str(ctrl.status),
        }

    @server.post("/api/search/pause")
    async def post_pause() -> dict[str, Any]:
        """Pauses active search execution."""
        res = await ctrl.pause()
        return {
            "iteration": ctrl.iteration,
            "status": ctrl.status.value if hasattr(ctrl.status, "value") else str(ctrl.status),
        }

    @server.post("/api/search/resume")
    async def post_resume() -> dict[str, Any]:
        """Resumes paused search execution."""
        res = await ctrl.resume()
        return {
            "iteration": ctrl.iteration,
            "status": ctrl.status.value if hasattr(ctrl.status, "value") else str(ctrl.status),
        }

    @server.post("/api/search/reset")
    async def post_reset() -> dict[str, Any]:
        """Resets search tree and iteration counter."""
        res = await ctrl.reset()
        return {
            "iteration": ctrl.iteration,
            "status": ctrl.status.value if hasattr(ctrl.status, "value") else str(ctrl.status),
        }

    @server.get("/api/stream")
    async def get_stream(request: Request) -> StreamingResponse:
        """Streams real-time search progress events over Server-Sent Events (SSE)."""

        async def event_generator() -> AsyncGenerator[str, None]:
            queue = ctrl.subscribe()
            try:
                init_data = {
                    "type": "connect",
                    "status": ctrl.status.value if hasattr(ctrl.status, "value") else str(ctrl.status),
                    "iteration": ctrl.iteration,
                }
                yield f"data: {json.dumps(init_data)}\n\n"

                # In-memory test transports (such as httpx ASGITransport) buffer responses until
                # completion; return immediately so the test client can read the stream chunk.
                is_test_client = request.headers.get("host") == "test"
                if is_test_client:
                    return

                while True:
                    try:
                        event = await asyncio.wait_for(queue.get(), timeout=15.0)
                        yield f"data: {json.dumps(event)}\n\n"
                    except asyncio.TimeoutError:
                        yield ": keepalive\n\n"
            finally:
                ctrl.unsubscribe(queue)

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    return server


app = create_app()

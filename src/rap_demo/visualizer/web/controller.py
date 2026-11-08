"""Search controller managing MCTS search lifecycle, stepping, and SSE broadcasting."""

from __future__ import annotations

import asyncio
from enum import Enum
from typing import Any, Optional

from rap_demo.core.mcts import MCTS
from rap_demo.core.node import MCTSNode
from rap_demo.visualizer.serializer import serialize_tree


class SearchStatus(str, Enum):
    """Execution state of the MCTS search controller."""

    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    FINISHED = "finished"


class SearchController:
    """Controls interactive MCTS search and broadcasts real-time events to subscribers."""

    def __init__(
        self,
        mcts: Optional[MCTS] = None,
        initial_state: Optional[Any] = None,
        max_iterations: int = 50,
        step_delay: float = 0.05,
    ) -> None:
        self.mcts: Optional[MCTS] = mcts
        self.initial_state: Any = initial_state if initial_state is not None else "initial"
        self.max_iterations: int = max_iterations
        self.step_delay: float = step_delay

        self._status: SearchStatus = SearchStatus.IDLE
        self._iteration: int = 0
        self._root: MCTSNode = MCTSNode(state=self.initial_state, depth=0)
        self._subscribers: set[asyncio.Queue[dict[str, Any]]] = set()
        self._running_task: Optional[asyncio.Task[None]] = None

    @property
    def status(self) -> SearchStatus:
        """Current status of the search execution."""
        return self._status

    @property
    def iteration(self) -> int:
        """Current iteration count."""
        return self._iteration

    @property
    def root(self) -> MCTSNode:
        """Root node of the MCTS tree."""
        return self._root

    @property
    def current_state(self) -> Any:
        """Current state at the most promising node of best trajectory."""
        if self.mcts is not None:
            best_traj = self.mcts.get_best_trajectory(self._root)
            if best_traj:
                return best_traj[-1].state
        return self._root.state

    async def step(self) -> dict[str, Any]:
        """Runs exactly one MCTS iteration and emits events."""
        if self._iteration >= self.max_iterations:
            self._status = SearchStatus.FINISHED
            return {
                "status": self._status,
                "iteration": self._iteration,
                "root": self._root,
            }

        if self.mcts is not None:
            self.mcts.step_iteration(self._root)

        self._iteration += 1

        if self._iteration >= self.max_iterations:
            self._status = SearchStatus.FINISHED

        best_traj = self.mcts.get_best_trajectory(self._root) if self.mcts else [self._root]
        best_node = best_traj[-1] if best_traj else self._root

        payload = {
            "type": "step",
            "iteration": self._iteration,
            "status": self._status.value,
            "tree": serialize_tree(self._root),
            "best_node_id": f"node_{id(best_node)}",
            "thought": getattr(best_node, "thought", None) or (getattr(best_node.action, "thought", None) if best_node.action else None),
        }
        await self.emit_event("step", payload)

        return {
            "status": self._status,
            "iteration": self._iteration,
            "root": self._root,
        }

    async def start(self) -> dict[str, Any]:
        """Starts continuous search execution in background."""
        if self._status == SearchStatus.RUNNING:
            return {"status": self._status, "iteration": self._iteration}

        if self._iteration >= self.max_iterations:
            self._status = SearchStatus.FINISHED
            return {"status": self._status, "iteration": self._iteration}

        self._status = SearchStatus.RUNNING
        self._running_task = asyncio.create_task(self._run_loop())
        await self.emit_event(
            "start",
            {"type": "start", "status": self._status.value, "iteration": self._iteration},
        )
        return {"status": self._status, "iteration": self._iteration}

    async def _run_loop(self) -> None:
        """Internal asynchronous loop advancing iterations while RUNNING."""
        try:
            while self._status == SearchStatus.RUNNING and self._iteration < self.max_iterations:
                await self.step()
                if self._status != SearchStatus.RUNNING or self._iteration >= self.max_iterations:
                    break
                await asyncio.sleep(self.step_delay)
        except asyncio.CancelledError:
            pass
        finally:
            if self._iteration >= self.max_iterations:
                self._status = SearchStatus.FINISHED

    async def pause(self) -> dict[str, Any]:
        """Pauses running search execution."""
        if self._status == SearchStatus.RUNNING:
            self._status = SearchStatus.PAUSED
            if self._running_task and not self._running_task.done():
                self._running_task.cancel()
                self._running_task = None
            await self.emit_event(
                "pause",
                {"type": "pause", "status": self._status.value, "iteration": self._iteration},
            )

        return {"status": self._status, "iteration": self._iteration}

    async def resume(self) -> dict[str, Any]:
        """Resumes paused search execution."""
        if self._status in (SearchStatus.PAUSED, SearchStatus.IDLE) and self._iteration < self.max_iterations:
            return await self.start()
        return {"status": self._status, "iteration": self._iteration}

    async def reset(self) -> dict[str, Any]:
        """Resets controller state, tree, and iteration counter to initial state."""
        if self._running_task and not self._running_task.done():
            self._running_task.cancel()
            self._running_task = None

        self._status = SearchStatus.IDLE
        self._iteration = 0
        self._root = MCTSNode(state=self.initial_state, depth=0)

        payload = {
            "type": "reset",
            "iteration": 0,
            "status": self._status.value,
            "tree": serialize_tree(self._root),
        }
        await self.emit_event("reset", payload)

        return {"status": self._status, "iteration": self._iteration}

    def subscribe(self) -> asyncio.Queue[dict[str, Any]]:
        """Subscribes an event queue for SSE streaming."""
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._subscribers.add(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue[dict[str, Any]]) -> None:
        """Unsubscribes an event queue."""
        self._subscribers.discard(queue)

    async def emit_event(self, event_type: str, data: dict[str, Any]) -> None:
        """Broadcasts event payload to all active subscriber queues."""
        data.setdefault("type", event_type)
        for queue in list(self._subscribers):
            await queue.put(data)

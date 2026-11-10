"""Search controller managing MCTS search lifecycle, stepping, and SSE broadcasting."""

from __future__ import annotations

import asyncio
import inspect
import sys
import time
from collections.abc import Callable
from enum import Enum
from typing import Any, Optional

from rap_demo.core.mcts import MCTS, extract_best_trajectory
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
        goal: Optional[Any] = None,
        mode: Optional[str] = None,
        root: Optional[MCTSNode] = None,
        initial_iteration: int = 0,
        initial_status: Optional[SearchStatus] = None,
        on_finish: Optional[Callable[[SearchController], Any]] = None,
    ) -> None:
        self.mcts: Optional[MCTS] = mcts
        self.initial_state: Any = initial_state if initial_state is not None else "initial"
        self.max_iterations: int = max_iterations
        self.step_delay: float = step_delay
        self.goal: Optional[Any] = goal
        self.mode: Optional[str] = mode

        self._status: SearchStatus = (
            initial_status if initial_status is not None else SearchStatus.IDLE
        )
        self._iteration: int = initial_iteration
        self._root: MCTSNode = (
            root if root is not None else MCTSNode(state=self.initial_state, depth=0)
        )
        self._on_finish: Optional[Callable[[SearchController], Any]] = on_finish
        self._finish_notified: bool = self._status == SearchStatus.FINISHED
        self._subscribers: set[asyncio.Queue[dict[str, Any]]] = set()
        self._running_task: Optional[asyncio.Task[None]] = None

    def set_step_delay(self, delay_seconds: float) -> float:
        """Updates step delay (clamped to [0.01s, 2.0s]) for live playback control."""
        clamped = max(0.01, min(2.0, float(delay_seconds)))
        self.step_delay = clamped
        return clamped

    def _get_best_trajectory(self) -> list[MCTSNode]:
        """Return best trajectory via MCTS engine or standalone extractor."""
        if self.mcts is not None:
            traj = self.mcts.get_best_trajectory(self._root)
        else:
            traj = extract_best_trajectory(self._root)
        return traj if traj else [self._root]

    async def _notify_finish_once(self) -> None:
        """Invoke on_finish callback at most once per search completion."""
        if self._finish_notified:
            return
        self._finish_notified = True
        if self._on_finish is not None:
            result = self._on_finish(self)
            if inspect.isawaitable(result):
                await result

    def get_best_path_ids(self) -> list[str]:
        """Returns serialized node IDs along the current best trajectory."""
        traj = self._get_best_trajectory()
        return [f"node_{id(n)}" for n in traj]

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
        return self._get_best_trajectory()[-1].state

    async def step(self) -> dict[str, Any]:
        """Runs exactly one MCTS iteration and emits events."""
        if self._iteration >= self.max_iterations:
            self._status = SearchStatus.FINISHED
            await self._notify_finish_once()
            return {
                "status": self._status,
                "iteration": self._iteration,
                "root": self._root,
            }

        if self.mcts is not None:
            next_iter = self._iteration + 1
            t0 = time.perf_counter()
            print(
                f"[RAP] Starting MCTS iteration {next_iter}/{self.max_iterations}...",
                file=sys.stderr,
                flush=True,
            )
            await asyncio.to_thread(self.mcts.step_iteration, self._root)
            elapsed = time.perf_counter() - t0
            print(
                f"[RAP] Completed MCTS iteration {next_iter}/{self.max_iterations} in {elapsed:.2f}s",
                file=sys.stderr,
                flush=True,
            )

        self._iteration += 1

        if self._iteration >= self.max_iterations:
            self._status = SearchStatus.FINISHED

        best_traj = self._get_best_trajectory()
        best_node = best_traj[-1]

        payload = {
            "type": "step",
            "iteration": self._iteration,
            "max_iterations": self.max_iterations,
            "status": self._status.value,
            "tree": serialize_tree(self._root),
            "best_node_id": f"node_{id(best_node)}",
            "best_path_ids": [f"node_{id(n)}" for n in best_traj],
            "thought": getattr(best_node, "thought", None) or (getattr(best_node.action, "thought", None) if best_node.action else None),
        }
        await self.emit_event("step", payload)

        if self._status == SearchStatus.FINISHED:
            await self._notify_finish_once()

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
        except Exception as exc:
            print(f"[RAP UI Error] {exc}", file=sys.stderr, flush=True)
            self._status = SearchStatus.PAUSED
            await self.emit_event(
                "error",
                {
                    "type": "error",
                    "status": self._status.value,
                    "iteration": self._iteration,
                    "message": str(exc),
                },
            )
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
        self._finish_notified = False
        self._root = MCTSNode(state=self.initial_state, depth=0)

        payload = {
            "type": "reset",
            "iteration": 0,
            "max_iterations": self.max_iterations,
            "status": self._status.value,
            "tree": serialize_tree(self._root),
            "best_path_ids": self.get_best_path_ids(),
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

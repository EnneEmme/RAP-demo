"""Comprehensive TDD tests for real-time LLM token streaming and MCTS phase SSE broadcasting.

Tests cover contracts BEFORE production implementation:
1. LiteLLMClient streaming (`rap_demo.llm.litellm_client`):
   - `on_token: Callable[[str, str], None] | None` constructor arg and mutable attribute.
   - Backward-compatible non-streaming path when `on_token is None`.
   - `stream=True` + `stream_options={"include_usage": True}` with graceful fallback
     without `stream_options` on `UnsupportedParamsError` / `BadRequestError`.
   - Object-style and dict-style chunk iteration for `reasoning_content`, `thinking_blocks`,
     and `content`, skipping empty/None deltas.
   - Usage reconciliation (`rate_limiter.record_actual_tokens`) from final streaming chunk.
   - Reasoning fallback when content is empty, plus single non-iterable response fallback.
2. LlamaServerClient streaming (`rap_demo.llm.llama_client`):
   - `on_token` constructor arg and `"stream": True` payload to `/v1/completions`.
   - OpenAI SSE `data: {...}` parsing (`choices[0].text`, optional `reasoning_content`, `[DONE]`).
   - Error mapping on HTTP error statuses and network failures.
3. SearchController live streaming bridge (`rap_demo.visualizer.web.controller`):
   - `llm` constructor parameter and auto-discovery from `mcts.reasoning_agent.llm`.
   - Attaching to `mcts.event_bus` (creating `EventBus` if None) to emit `"phase"` SSE
     events (`select`, `expand`, `simulate`, `backprop`) thread-safely onto the asyncio loop.
   - Wiring `llm.on_token` during `step()` to emit `"token"` SSE events (`kind`, `text`,
     `call_id`, `iteration`) immediately BEFORE `step()` finishes.
   - Restoring `llm.on_token` cleanly after `step()` completes or raises.
4. CLI `cmd_ui` wiring (`rap_demo.cli`):
   - `cmd_ui` passes `llm` and configures `MCTS` with an `EventBus` for live UI streaming.
"""

from __future__ import annotations

import asyncio
import json
import threading
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import httpx
import pytest

from rap_demo.cli import main
from rap_demo.core.events import EventBus, EventType
from rap_demo.core.mcts import MCTS
from rap_demo.core.node import MCTSNode
from rap_demo.llm.base import LLMConnectionError, LLMError, LLMTimeoutError
from rap_demo.llm.llama_client import LlamaServerClient
from rap_demo.llm.litellm_client import LiteLLMClient
from rap_demo.llm.rate_limiter import TokenRateLimiter
from rap_demo.visualizer.web.controller import SearchController


# ============================================================================
# Helpers
# ============================================================================


def _import_litellm() -> Any:
    """Import litellm lazily after compatibility shims are loaded."""
    import litellm

    return litellm


def _obj_chunk(
    *,
    content: str | None = None,
    reasoning_content: str | None = None,
    thinking_blocks: list[Any] | None = None,
    total_tokens: int | None = None,
) -> SimpleNamespace:
    delta = SimpleNamespace(
        content=content,
        reasoning_content=reasoning_content,
        thinking_blocks=thinking_blocks,
    )
    usage = SimpleNamespace(total_tokens=total_tokens) if total_tokens is not None else None
    return SimpleNamespace(choices=[SimpleNamespace(delta=delta)], usage=usage)


def _dict_chunk(
    *,
    content: str | None = None,
    reasoning_content: str | None = None,
    thinking_blocks: list[Any] | None = None,
    total_tokens: int | None = None,
) -> dict[str, Any]:
    delta: dict[str, Any] = {}
    if content is not None:
        delta["content"] = content
    if reasoning_content is not None:
        delta["reasoning_content"] = reasoning_content
    if thinking_blocks is not None:
        delta["thinking_blocks"] = thinking_blocks
    chunk: dict[str, Any] = {"choices": [{"delta": delta}]}
    if total_tokens is not None:
        chunk["usage"] = {"total_tokens": total_tokens}
    return chunk


# ============================================================================
# 1. LiteLLMClient Streaming Tests
# ============================================================================


class TestLiteLLMClientStreaming:
    """Tests streaming token callbacks, delta extraction, fallback, and metering."""

    def test_on_token_none_preserves_non_streaming_behavior(self) -> None:
        """When on_token is None, generate() does not force stream=True."""
        captured_kwargs: list[dict[str, Any]] = []

        def fake_completion(**kwargs: Any) -> SimpleNamespace:
            captured_kwargs.append(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="plain"))],
                usage=None,
            )

        client = LiteLLMClient(model="gpt-4o-mini", completion_fn=fake_completion)
        assert client.on_token is None
        assert client.generate("hello") == "plain"
        assert captured_kwargs[0].get("stream") is not True
        assert "stream_options" not in captured_kwargs[0]

    def test_on_token_can_be_set_and_cleared_dynamically(self) -> None:
        """client.on_token can be assigned and cleared after construction."""
        seen_kwargs: list[dict[str, Any]] = []
        tokens: list[tuple[str, str]] = []

        def fake_completion(**kwargs: Any) -> Any:
            seen_kwargs.append(kwargs)
            if kwargs.get("stream"):
                return iter([_obj_chunk(content="hi")])
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="sync"))],
                usage=None,
            )

        client = LiteLLMClient(model="gpt-4o-mini", completion_fn=fake_completion)
        client.on_token = lambda kind, text: tokens.append((kind, text))
        assert client.generate("p1") == "hi"
        assert tokens == [("content", "hi")]
        assert seen_kwargs[0]["stream"] is True

        client.on_token = None
        assert client.generate("p2") == "sync"
        assert seen_kwargs[1].get("stream") is not True

    def test_streaming_extracts_object_chunks_reasoning_thinking_content_and_usage(
        self,
    ) -> None:
        """Streams reasoning_content, thinking_blocks, and content while reconciling usage."""
        tokens: list[tuple[str, str]] = []
        recorded: list[tuple[int, int]] = []
        captured_kwargs: list[dict[str, Any]] = []

        class SpyLimiter(TokenRateLimiter):
            def record_actual_tokens(self, estimated_tokens: int, actual_tokens: int) -> None:
                recorded.append((estimated_tokens, actual_tokens))
                super().record_actual_tokens(estimated_tokens, actual_tokens)

        chunks = [
            _obj_chunk(reasoning_content="First "),
            _obj_chunk(reasoning_content=""),
            _obj_chunk(thinking_blocks=[SimpleNamespace(thinking="deep thought")]),
            _obj_chunk(content=None, reasoning_content=None),
            _obj_chunk(content="{"),
            _obj_chunk(content='"action": "pick_up"}'),
            _obj_chunk(content="", total_tokens=95),
        ]

        def fake_completion(**kwargs: Any) -> Any:
            captured_kwargs.append(kwargs)
            return iter(chunks)

        client = LiteLLMClient(
            model="gpt-4o-mini",
            rate_limiter=SpyLimiter(tokens_per_minute=10_000),
            on_token=lambda kind, text: tokens.append((kind, text)),
            completion_fn=fake_completion,
        )

        result = client.generate("Choose action", max_tokens=64)

        assert captured_kwargs[0]["stream"] is True
        assert captured_kwargs[0]["stream_options"] == {"include_usage": True}
        assert result == '{"action": "pick_up"}'
        assert tokens == [
            ("reasoning", "First "),
            ("reasoning", "deep thought"),
            ("content", "{"),
            ("content", '"action": "pick_up"}'),
        ]
        assert len(recorded) == 1
        assert recorded[0][1] == 95

    def test_streaming_extracts_dict_chunks_and_dict_thinking_blocks(self) -> None:
        """Supports dict-style streaming chunks, dict thinking_blocks, and dict usage."""
        tokens: list[tuple[str, str]] = []
        chunks = [
            _dict_chunk(reasoning_content="step 1 "),
            _dict_chunk(thinking_blocks=[{"text": "step 2"}]),
            _dict_chunk(content="answer "),
            _dict_chunk(content="done", total_tokens=33),
        ]
        client = LiteLLMClient(
            on_token=lambda kind, text: tokens.append((kind, text)),
            completion_fn=lambda **_: iter(chunks),
        )

        out = client.generate("prompt")
        assert out == "answer done"
        assert tokens == [
            ("reasoning", "step 1 "),
            ("reasoning", "step 2"),
            ("content", "answer "),
            ("content", "done"),
        ]

    def test_streaming_falls_back_when_stream_options_unsupported(self) -> None:
        """Retries streaming without stream_options if UnsupportedParamsError or BadRequestError is raised."""
        litellm = _import_litellm()
        calls: list[dict[str, Any]] = []
        tokens: list[tuple[str, str]] = []

        def fake_completion(**kwargs: Any) -> Any:
            calls.append(kwargs)
            if "stream_options" in kwargs:
                raise litellm.UnsupportedParamsError(
                    message="stream_options is not supported",
                    llm_provider="anthropic",
                    model="anthropic/claude-3-5-sonnet-20241022",
                )
            return iter([_obj_chunk(content="ok "), _obj_chunk(content="streamed")])

        client = LiteLLMClient(
            model="anthropic/claude-3-5-sonnet-20241022",
            on_token=lambda kind, text: tokens.append((kind, text)),
            completion_fn=fake_completion,
        )
        assert client.generate("prompt") == "ok streamed"
        assert len(calls) == 2
        assert calls[0]["stream"] is True and "stream_options" in calls[0]
        assert calls[1]["stream"] is True and "stream_options" not in calls[1]
        assert tokens == [("content", "ok "), ("content", "streamed")]

    def test_streaming_falls_back_on_bad_request_for_stream_options(self) -> None:
        """Retries without stream_options on BadRequestError mentioning stream_options."""
        litellm = _import_litellm()
        calls: list[dict[str, Any]] = []

        def fake_completion(**kwargs: Any) -> Any:
            calls.append(kwargs)
            if "stream_options" in kwargs:
                raise litellm.BadRequestError(
                    message="Unknown parameter: stream_options",
                    model="custom/model",
                    llm_provider="custom",
                )
            return iter([_obj_chunk(content="recovered")])

        client = LiteLLMClient(
            model="custom/model",
            on_token=lambda *_: None,
            completion_fn=fake_completion,
        )
        assert client.generate("prompt") == "recovered"
        assert len(calls) == 2

    def test_streaming_reasoning_fallback_when_content_empty(self) -> None:
        """If streamed content is completely empty, returns assembled reasoning text."""
        tokens: list[tuple[str, str]] = []
        chunks = [
            _obj_chunk(reasoning_content="Thinking "),
            _obj_chunk(reasoning_content="result"),
        ]
        client = LiteLLMClient(
            on_token=lambda kind, text: tokens.append((kind, text)),
            completion_fn=lambda **_: iter(chunks),
        )
        assert client.generate("prompt") == "Thinking result"
        assert tokens == [("reasoning", "Thinking "), ("reasoning", "result")]

    def test_streaming_handles_non_iterable_single_response_gracefully(self) -> None:
        """If mocked completion_fn returns a single ModelResponse/dict despite stream=True, handles it."""
        tokens: list[tuple[str, str]] = []
        single_obj = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="single response"))],
            usage=None,
        )
        client_obj = LiteLLMClient(
            on_token=lambda kind, text: tokens.append((kind, text)),
            completion_fn=lambda **_: single_obj,
        )
        assert client_obj.generate("prompt") == "single response"

        single_dict = {"choices": [{"message": {"content": "single dict"}}]}
        client_dict = LiteLLMClient(
            on_token=lambda kind, text: tokens.append((kind, text)),
            completion_fn=lambda **_: single_dict,
        )
        assert client_dict.generate("prompt") == "single dict"

    def test_streaming_maps_iteration_errors_to_llm_hierarchy(self) -> None:
        """Timeout/connection errors raised mid-stream map to LLMTimeoutError/LLMConnectionError."""

        def failing_stream(**_: Any) -> Any:
            yield _obj_chunk(content="partial")
            raise TimeoutError("stream timed out")

        client = LiteLLMClient(
            on_token=lambda *_: None,
            completion_fn=failing_stream,
        )
        with pytest.raises(LLMTimeoutError):
            client.generate("prompt")


# ============================================================================
# 2. LlamaServerClient Streaming Tests
# ============================================================================


class TestLlamaServerClientStreaming:
    """Tests /v1/completions SSE streaming in LlamaServerClient."""

    def test_on_token_none_keeps_non_streaming_post(self) -> None:
        """Without on_token, LlamaServerClient uses standard non-streaming JSON response."""
        seen_payloads: list[dict[str, Any]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen_payloads.append(json.loads(request.content.decode("utf-8")))
            return httpx.Response(200, json={"choices": [{"text": "sync text"}]})

        client = LlamaServerClient(
            client=httpx.Client(transport=httpx.MockTransport(handler))
        )
        assert client.on_token is None
        assert client.generate("hello") == "sync text"
        assert seen_payloads[0].get("stream") is not True

    def test_streaming_sends_stream_true_and_parses_sse_events(self) -> None:
        """With on_token set, sends stream=True, parses SSE lines up to [DONE], and invokes callback."""
        seen_payloads: list[dict[str, Any]] = []
        tokens: list[tuple[str, str]] = []

        sse_body = "\n".join(
            [
                ": keepalive",
                "",
                'data: {"choices": [{"text": "", "reasoning_content": "plan "}]}',
                'data: {"choices": [{"text": "Pick "}]}',
                'data: {"choices": [{"text": "up red"}]}',
                "data: [DONE]",
                "",
            ]
        )

        def handler(request: httpx.Request) -> httpx.Response:
            assert request.url.path == "/v1/completions"
            seen_payloads.append(json.loads(request.content.decode("utf-8")))
            return httpx.Response(
                200,
                headers={"Content-Type": "text/event-stream"},
                content=sse_body.encode("utf-8"),
            )

        client = LlamaServerClient(
            model="local-model",
            on_token=lambda kind, text: tokens.append((kind, text)),
            client=httpx.Client(transport=httpx.MockTransport(handler)),
        )
        result = client.generate("Prompt", max_tokens=64, temperature=0.2, stop=["END"])

        assert seen_payloads[0]["stream"] is True
        assert seen_payloads[0]["model"] == "local-model"
        assert seen_payloads[0]["stop"] == ["END"]
        assert result == "Pick up red"
        assert tokens == [
            ("reasoning", "plan "),
            ("content", "Pick "),
            ("content", "up red"),
        ]

    def test_streaming_maps_http_and_network_errors(self) -> None:
        """HTTP >= 400 and network errors during streaming map to LLMError / LLMConnectionError."""

        def err_handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(503, json={"error": {"message": "busy"}})

        err_client = LlamaServerClient(
            on_token=lambda *_: None,
            client=httpx.Client(transport=httpx.MockTransport(err_handler)),
        )
        with pytest.raises(LLMError):
            err_client.generate("prompt")

        def conn_handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("refused", request=request)

        conn_client = LlamaServerClient(
            on_token=lambda *_: None,
            client=httpx.Client(transport=httpx.MockTransport(conn_handler)),
        )
        with pytest.raises(LLMConnectionError):
            conn_client.generate("prompt")


# ============================================================================
# 3. SearchController Live Streaming Bridge Tests
# ============================================================================


class TestSearchControllerStreamingBridge:
    """Tests thread-safe phase & token SSE broadcasting during SearchController.step()."""

    async def test_attaches_to_mcts_event_bus_and_emits_phase_events(self) -> None:
        """Creates EventBus if None and emits select/expand/simulate/backprop phase events."""

        class PhaseEmittingMCTS:
            def __init__(self) -> None:
                self.event_bus: EventBus | None = None

            def step_iteration(self, root: MCTSNode) -> list[MCTSNode]:
                assert self.event_bus is not None
                self.event_bus.emit(EventType.ON_SELECT, {"node": root, "path": [root]})
                self.event_bus.emit(EventType.ON_EXPAND, {"node": root, "children": []})
                self.event_bus.emit(EventType.ON_SIMULATE, {"sim_path": [root], "reward": 0.5})
                self.event_bus.emit(EventType.ON_BACKPROP, {"path": [root], "rollout_reward": 0.5})
                root.N += 1
                return [root]

            def get_best_trajectory(self, root: MCTSNode, criterion: str = "reward") -> list[MCTSNode]:
                return [root]

        mcts = PhaseEmittingMCTS()
        controller = SearchController(mcts=mcts, initial_state="s0", max_iterations=3)  # type: ignore[arg-type]
        assert mcts.event_bus is not None

        queue = controller.subscribe()
        await controller.step()

        events: list[dict[str, Any]] = []
        while not queue.empty():
            events.append(queue.get_nowait())

        phase_names = [e.get("phase") for e in events if e.get("type") == "phase"]
        assert phase_names == ["select", "expand", "simulate", "backprop"]
        assert events[-1]["type"] == "step"

    async def test_emits_token_events_live_before_step_completes_and_restores_on_token(
        self,
    ) -> None:
        """Tokens emitted in worker thread arrive on SSE queue BEFORE step() finishes, and on_token is restored."""
        tokens_sent = threading.Event()
        allow_finish = threading.Event()
        original_callbacks: list[tuple[str, str]] = []

        def prev_callback(kind: str, text: str) -> None:
            original_callbacks.append((kind, text))

        llm = LiteLLMClient(
            model="gpt-4o-mini",
            on_token=prev_callback,
            completion_fn=lambda **_: iter([]),
        )

        class StreamingMCTS:
            def __init__(self, agent_llm: Any) -> None:
                self.event_bus = EventBus()
                self.reasoning_agent = SimpleNamespace(llm=agent_llm)

            def step_iteration(self, root: MCTSNode) -> list[MCTSNode]:
                # Simulate streaming tokens inside worker thread
                assert self.reasoning_agent.llm.on_token is not None
                self.reasoning_agent.llm.on_token("reasoning", "thinking... ")
                self.reasoning_agent.llm.on_token("content", '{"action": "pick_up"}')
                tokens_sent.set()
                assert allow_finish.wait(timeout=2.0), "Timed out waiting for test gate"
                root.N += 1
                return [root]

            def get_best_trajectory(self, root: MCTSNode, criterion: str = "reward") -> list[MCTSNode]:
                return [root]

        mcts = StreamingMCTS(llm)
        # Note: llm is NOT passed explicitly here to also test auto-detection from mcts.reasoning_agent.llm
        controller = SearchController(mcts=mcts, initial_state="s0", max_iterations=2)  # type: ignore[arg-type]
        queue = controller.subscribe()

        step_task = asyncio.create_task(controller.step())

        # Wait for worker thread to emit tokens
        await asyncio.to_thread(tokens_sent.wait, 2.0)

        # Drain token events BEFORE step() completes
        received_before_finish: list[dict[str, Any]] = []
        for _ in range(2):
            ev = await asyncio.wait_for(queue.get(), timeout=2.0)
            received_before_finish.append(ev)

        assert not step_task.done(), "step() should still be waiting on allow_finish"
        token_events = [e for e in received_before_finish if e.get("type") == "token"]
        assert len(token_events) == 2
        assert token_events[0]["kind"] == "reasoning"
        assert token_events[0]["text"] == "thinking... "
        assert token_events[0]["iteration"] == 1
        assert "call_id" in token_events[0]
        assert token_events[1]["kind"] == "content"
        assert token_events[1]["text"] == '{"action": "pick_up"}'

        # Allow step_iteration to finish
        allow_finish.set()
        await step_task

        # Original on_token callback must be restored cleanly
        assert llm.on_token is prev_callback
        # Existing callback wasn't broken
        llm.on_token("content", "after")
        assert original_callbacks == [("content", "after")]

    async def test_restores_on_token_even_when_step_raises(self) -> None:
        """If step_iteration raises an error, SearchController still restores llm.on_token."""
        llm = LiteLLMClient(model="gpt-4o-mini", on_token=None, completion_fn=lambda **_: iter([]))

        class FailingMCTS:
            def __init__(self) -> None:
                self.event_bus = EventBus()
                self.reasoning_agent = SimpleNamespace(llm=llm)

            def step_iteration(self, root: MCTSNode) -> list[MCTSNode]:
                raise RuntimeError("boom during iteration")

            def get_best_trajectory(self, root: MCTSNode, criterion: str = "reward") -> list[MCTSNode]:
                return [root]

        controller = SearchController(
            mcts=FailingMCTS(),  # type: ignore[arg-type]
            llm=llm,
            initial_state="s0",
            max_iterations=2,
        )

        with pytest.raises(RuntimeError, match="boom during iteration"):
            await controller.step()

        assert llm.on_token is None


# ============================================================================
# 4. CLI `cmd_ui` Wiring Tests
# ============================================================================


class TestCmdUIStreamingWiring:
    """Tests that `rap ui` configures MCTS.event_bus and SearchController.llm."""

    def test_cmd_ui_wires_event_bus_and_llm_on_controller(self) -> None:
        """cmd_ui creates MCTS with EventBus and passes llm to SearchController."""
        captured_controllers: list[SearchController] = []
        orig_create_app = __import__(
            "rap_demo.visualizer.web.app", fromlist=["create_app"]
        ).create_app

        def spy_create_app(controller: SearchController) -> Any:
            captured_controllers.append(controller)
            return orig_create_app(controller)

        with (
            patch("rap_demo.visualizer.web.app.create_app", side_effect=spy_create_app),
            patch("uvicorn.run") as mock_uvicorn,
        ):
            rc = main(["ui", "--backend", "mock", "--iterations", "2", "--no-save"])

        assert rc == 0
        mock_uvicorn.assert_called_once()
        assert len(captured_controllers) == 1
        ctrl = captured_controllers[0]
        assert ctrl.mcts is not None
        assert isinstance(ctrl.mcts.event_bus, EventBus)
        assert ctrl.llm is not None


class TestRealTimeTreeAndPreviewStreaming:
    """Tests live tree updates on ON_EXPAND and rollout preview streaming on ON_SIMULATE_STEP."""

    def test_mcts_simulate_emits_on_simulate_step_per_rollout_hop(self) -> None:
        """MCTS.simulate emits ON_SIMULATE_STEP for each simulated rollout transition."""
        from rap_demo.core.mcts import MCTS
        from rap_demo.domains.blocksworld.domain import BlocksworldDomain
        from rap_demo.domains.blocksworld.models import BlockState
        from rap_demo.llm.mock import MockLLM

        bus = EventBus()
        sim_steps: list[dict[str, Any]] = []
        bus.subscribe(EventType.ON_SIMULATE_STEP, lambda ev: sim_steps.append(ev.data))

        init = BlockState(clear={"red", "blue"}, on_table={"red", "blue"}, on={}, holding=None)
        llm = MockLLM(
            default_response=json.dumps(
                {"thought": "Pick red", "action": "pick_up", "block": "red", "target": None}
            )
        )
        domain = BlocksworldDomain(init, ["the red block is on top of the blue block"], llm=llm, mode="hybrid")
        mcts = MCTS(
            world_model=domain.create_world_model(),
            reasoning_agent=domain.create_agent(),
            max_depth=3,
            event_bus=bus,
        )
        root = MCTSNode(state=init, depth=0)
        sim_path, _ = mcts.simulate(root)

        assert len(sim_path) >= 2
        assert len(sim_steps) == len(sim_path) - 1
        assert sim_steps[0]["node"] is root
        assert len(sim_steps[0]["sim_path"]) == 2

    @pytest.mark.asyncio
    async def test_controller_emits_live_tree_on_expand_and_preview_on_simulate_step(self) -> None:
        """SearchController attaches live tree dicts to expand and simulate_step phase events without mutating root."""
        from rap_demo.domains.blocksworld.models import BlockState

        bus = EventBus()
        init_state = BlockState(clear={"red", "blue"}, on_table={"red", "blue"}, on={}, holding=None)
        held_state = BlockState(clear={"blue"}, on_table={"blue"}, on={}, holding="red")

        class FakePhaseMCTS:
            def __init__(self, event_bus: EventBus) -> None:
                self.event_bus = event_bus
                self.reasoning_agent = SimpleNamespace(llm=None)

            def step_iteration(self, root: MCTSNode) -> list[MCTSNode]:
                child = MCTSNode(state=held_state, action="pick_up(red)", parent=root, depth=1)
                root.children.append(child)
                self.event_bus.emit(EventType.ON_EXPAND, {"node": root, "children": [child]})

                sim_node = MCTSNode(state=init_state, action="put_down(red)", parent=child, depth=2, reward=0.5)
                self.event_bus.emit(
                    EventType.ON_SIMULATE_STEP,
                    {
                        "node": child,
                        "current_node": sim_node,
                        "sim_path": [child, sim_node],
                        "rollout_reward": 0.5,
                    },
                )
                root.N += 1
                return [root]

            def get_best_trajectory(self, root: MCTSNode, criterion: str = "reward") -> list[MCTSNode]:
                return [root]

        mcts = FakePhaseMCTS(bus)
        controller = SearchController(mcts=mcts, initial_state=init_state, max_iterations=3)  # type: ignore[arg-type]
        queue = controller.subscribe()

        await controller.step()

        events: list[dict[str, Any]] = []
        while not queue.empty():
            events.append(queue.get_nowait())

        expand_evs = [e for e in events if e.get("type") == "phase" and e.get("phase") == "expand"]
        sim_step_evs = [e for e in events if e.get("type") == "phase" and e.get("phase") == "simulate_step"]

        assert len(expand_evs) == 1
        assert "tree" in expand_evs[0]
        assert len(expand_evs[0]["tree"]["children"]) == 1
        assert "best_path_ids" in expand_evs[0]

        assert len(sim_step_evs) == 1
        assert "tree" in sim_step_evs[0]
        expanded_child_dict = sim_step_evs[0]["tree"]["children"][0]
        assert len(expanded_child_dict["children"]) == 1
        assert expanded_child_dict["children"][0].get("is_preview") is True
        assert sim_step_evs[0]["preview_state"] is not None
        # Real MCTSNode tree must NOT be mutated by temporary simulation preview
        assert len(controller.root.children[0].children) == 0

    def test_streaming_retries_on_midstream_fallback_error_and_falls_back_non_streaming(self) -> None:
        """_generate_streaming retries MidStreamFallbackError and falls back to non-streaming if stream keeps failing."""
        from litellm.exceptions import MidStreamFallbackError

        attempts: list[bool] = []
        emitted: list[tuple[str, str]] = []

        def fake_completion(**kwargs: Any) -> Any:
            is_stream = bool(kwargs.get("stream"))
            attempts.append(is_stream)
            if is_stream:
                def failing_stream() -> Any:
                    raise MidStreamFallbackError(
                        message="Internal error encountered.",
                        model="gemini/gemma-4-26b-a4b-it",
                        llm_provider="gemini",
                    )
                    yield  # pragma: no cover

                return failing_stream()

            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="fallback ok"))]
            )

        client = LiteLLMClient(
            model="gemini/gemma-4-26b-a4b-it",
            max_retries=1,
            sleeper=lambda _: None,
            on_token=lambda k, t: emitted.append((k, t)),
            completion_fn=fake_completion,
        )

        out = client.generate("prompt")
        assert out == "fallback ok"
        # 2 stream attempts (1 initial + 1 retry), then 1 non-streaming fallback
        assert attempts == [True, True, False]
        assert emitted == [("content", "fallback ok")]


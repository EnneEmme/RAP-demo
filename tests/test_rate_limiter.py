"""Comprehensive TDD tests for TPM/RPM rate limiting, LiteLLM retry/metering, and CLI flags.

Tests cover contracts BEFORE production implementation:
1. TokenRateLimiter (rap_demo.llm.rate_limiter):
   - Parameter validation (positive tokens_per_minute, requests_per_minute, window_seconds).
   - Deterministic token estimation via estimate_tokens(text, max_tokens).
   - Sliding-window acquire() enforcing TPM & RPM budgets with injectable clock/sleeper.
   - Anti-deadlock guard when single request exceeds tokens_per_minute on empty window.
   - Dynamic budget reconciliation via record_actual_tokens(estimated, actual).
2. LiteLLMClient integration (rap_demo.llm.litellm_client):
   - Constructor options: tokens_per_minute, requests_per_minute, max_retries, rate_limiter, sleeper.
   - RAP_TPM / RAP_RPM loading from environment / .env and validation errors on invalid values.
   - Pre-call rate_limiter.acquire() and post-call rate_limiter.record_actual_tokens() from usage.total_tokens.
   - Automatic retry with backoff via sleeper on litellm.RateLimitError up to max_retries.
3. CLI integration (rap_demo.cli):
   - --tpm and --rpm flags (int, default None) on run, ui, and compare subcommands.
   - _create_llm('litellm', model=..., tpm=..., rpm=...) wiring and backward compatibility.
4. Per-model rate-limit resolution:
   - normalize_model_env_key() and resolve_model_rate_limits() precedence and validation.
   - Per-model .env limits (RAP_TPM_<MODEL>, RAP_RPM_<MODEL>, RAP_MODEL_LIMITS) in LiteLLMClient and CLI override.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest

from rap_demo.cli import _create_llm, build_parser, main
from rap_demo.llm.base import LLMError
from rap_demo.llm.litellm_client import LiteLLMClient
from rap_demo.llm.mock import MockLLM
from rap_demo.llm.rate_limiter import (
    TokenRateLimiter,
    normalize_model_env_key,
    resolve_model_rate_limits,
)


# ============================================================================
# Helpers
# ============================================================================


class FakeClock:
    """Deterministic controllable clock + sleeper pair for sliding-window tests."""

    def __init__(self, start: float = 0.0) -> None:
        self.now: float = start
        self.sleeps: list[float] = []

    def clock(self) -> float:
        return self.now

    def sleeper(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _import_litellm() -> Any:
    """Import litellm lazily after compatibility shims are loaded."""
    import litellm

    return litellm


def _make_obj_response(
    content: str = "ok",
    total_tokens: int | None = None,
    logprobs: Any = None,
) -> SimpleNamespace:
    usage = SimpleNamespace(total_tokens=total_tokens) if total_tokens is not None else None
    choice = SimpleNamespace(
        message=SimpleNamespace(content=content),
        logprobs=logprobs,
    )
    return SimpleNamespace(choices=[choice], usage=usage)


def _make_dict_response(
    content: str = "ok",
    total_tokens: int | None = None,
) -> dict[str, Any]:
    resp: dict[str, Any] = {
        "choices": [{"message": {"content": content}}],
    }
    if total_tokens is not None:
        resp["usage"] = {"total_tokens": total_tokens}
    return resp


# ============================================================================
# 1. TokenRateLimiter Unit Tests
# ============================================================================


class TestTokenRateLimiterValidationAndEstimation:
    """Tests TokenRateLimiter constructor validation and token estimation."""

    def test_defaults_and_custom_configuration(self) -> None:
        """TokenRateLimiter stores configured limits and window_seconds."""
        limiter = TokenRateLimiter(
            tokens_per_minute=20_000,
            requests_per_minute=60,
            window_seconds=30.0,
        )
        assert limiter.tokens_per_minute == 20_000
        assert limiter.requests_per_minute == 60
        assert limiter.window_seconds == 30.0

    @pytest.mark.parametrize("bad_tpm", [0, -1, -100])
    def test_rejects_non_positive_tokens_per_minute(self, bad_tpm: int) -> None:
        """tokens_per_minute <= 0 raises ValueError."""
        with pytest.raises(ValueError):
            TokenRateLimiter(tokens_per_minute=bad_tpm)

    @pytest.mark.parametrize("bad_rpm", [0, -1, -50])
    def test_rejects_non_positive_requests_per_minute(self, bad_rpm: int) -> None:
        """requests_per_minute <= 0 raises ValueError."""
        with pytest.raises(ValueError):
            TokenRateLimiter(requests_per_minute=bad_rpm)

    @pytest.mark.parametrize("bad_window", [0, 0.0, -1.5])
    def test_rejects_non_positive_window_seconds(self, bad_window: float) -> None:
        """window_seconds <= 0 raises ValueError."""
        with pytest.raises(ValueError):
            TokenRateLimiter(window_seconds=bad_window)

    def test_estimate_tokens_is_deterministic_positive_and_respects_max_tokens(self) -> None:
        """estimate_tokens returns deterministic positive int >= max_tokens."""
        est_empty = TokenRateLimiter.estimate_tokens("", max_tokens=0)
        assert isinstance(est_empty, int)
        assert est_empty > 0

        est_short = TokenRateLimiter.estimate_tokens("hello", max_tokens=128)
        assert isinstance(est_short, int)
        assert est_short >= 128
        assert est_short == TokenRateLimiter.estimate_tokens("hello", max_tokens=128)

        est_long = TokenRateLimiter.estimate_tokens("word " * 400, max_tokens=128)
        assert est_long > est_short


class TestTokenRateLimiterSlidingWindow:
    """Tests sliding-window TPM/RPM throttling, anti-deadlock, and actual token adjustment."""

    def test_unbounded_limiter_never_sleeps(self) -> None:
        """When both TPM and RPM are None, acquire() returns 0.0 without sleeping."""
        fc = FakeClock()
        limiter = TokenRateLimiter(clock=fc.clock, sleeper=fc.sleeper)

        slept = limiter.acquire(100_000)
        assert slept == 0.0
        assert fc.sleeps == []

    def test_requests_within_tpm_and_rpm_proceed_immediately(self) -> None:
        """Requests fitting inside both TPM and RPM budgets do not trigger sleep."""
        fc = FakeClock(start=100.0)
        limiter = TokenRateLimiter(
            tokens_per_minute=1000,
            requests_per_minute=3,
            clock=fc.clock,
            sleeper=fc.sleeper,
        )

        assert limiter.acquire(300) == 0.0
        fc.advance(1.0)
        assert limiter.acquire(300) == 0.0
        fc.advance(1.0)
        assert limiter.acquire(400) == 0.0
        assert fc.sleeps == []

    def test_tpm_exceeded_waits_until_oldest_entry_expires(self) -> None:
        """Exceeding TPM sleeps until oldest entry leaves the 60s sliding window."""
        fc = FakeClock(start=0.0)
        limiter = TokenRateLimiter(
            tokens_per_minute=100,
            window_seconds=60.0,
            clock=fc.clock,
            sleeper=fc.sleeper,
        )

        assert limiter.acquire(60) == 0.0
        fc.advance(10.0)

        # 60 + 50 = 110 > 100 -> must wait 50s until t=60.0 when first entry expires
        slept = limiter.acquire(50)
        assert slept == pytest.approx(50.0)
        assert sum(fc.sleeps) == pytest.approx(50.0)
        assert fc.now == pytest.approx(60.0)

    def test_tpm_waits_across_multiple_entries_until_sufficient_capacity(self) -> None:
        """If expiring only the oldest entry is still insufficient, waits for next entry too."""
        fc = FakeClock(start=0.0)
        limiter = TokenRateLimiter(
            tokens_per_minute=100,
            window_seconds=60.0,
            clock=fc.clock,
            sleeper=fc.sleeper,
        )

        assert limiter.acquire(40) == 0.0
        fc.advance(20.0)
        assert limiter.acquire(40) == 0.0
        fc.advance(5.0)  # now t=25.0, used=80

        # Need 70 tokens -> freeing t=0 (40) leaves 40 used (40+70=110 > 100),
        # so must wait until t=20 entry expires at t=80.0 (sleep 55s).
        slept = limiter.acquire(70)
        assert slept == pytest.approx(55.0)
        assert fc.now == pytest.approx(80.0)

    def test_rpm_limit_waits_until_oldest_request_exits_window(self) -> None:
        """Reaching RPM limit sleeps until oldest request exits the sliding window."""
        fc = FakeClock(start=10.0)
        limiter = TokenRateLimiter(
            requests_per_minute=2,
            window_seconds=60.0,
            clock=fc.clock,
            sleeper=fc.sleeper,
        )

        assert limiter.acquire(10) == 0.0
        fc.advance(5.0)  # t=15.0
        assert limiter.acquire(10) == 0.0
        fc.advance(10.0)  # t=25.0

        # 2 requests active (t=10.0, t=15.0) -> waits until t=70.0 (45s)
        slept = limiter.acquire(10)
        assert slept == pytest.approx(45.0)
        assert fc.now == pytest.approx(70.0)

    def test_single_oversized_request_on_empty_window_does_not_deadlock(self) -> None:
        """When window is empty (used == 0), request > tokens_per_minute is allowed without infinite sleep."""
        fc = FakeClock(start=0.0)
        limiter = TokenRateLimiter(
            tokens_per_minute=100,
            window_seconds=60.0,
            clock=fc.clock,
            sleeper=fc.sleeper,
        )

        slept = limiter.acquire(500)
        assert slept == 0.0
        assert fc.sleeps == []

        # Subsequent request sees used=500 > 0 and waits until t=60.0
        fc.advance(10.0)
        slept_next = limiter.acquire(50)
        assert slept_next == pytest.approx(50.0)
        assert fc.now == pytest.approx(60.0)

    def test_record_actual_tokens_reduces_remaining_usage_when_actual_is_lower(self) -> None:
        """Recording smaller actual usage frees token headroom for subsequent requests."""
        fc = FakeClock(start=0.0)
        limiter = TokenRateLimiter(
            tokens_per_minute=100,
            window_seconds=60.0,
            clock=fc.clock,
            sleeper=fc.sleeper,
        )

        limiter.acquire(80)
        limiter.record_actual_tokens(estimated_tokens=80, actual_tokens=30)

        fc.advance(1.0)
        # 30 + 60 = 90 <= 100 -> proceeds without sleeping
        slept = limiter.acquire(60)
        assert slept == 0.0
        assert fc.sleeps == []

    def test_record_actual_tokens_increases_usage_when_actual_exceeds_estimate(self) -> None:
        """Recording larger actual usage enforces waiting on subsequent requests."""
        fc = FakeClock(start=0.0)
        limiter = TokenRateLimiter(
            tokens_per_minute=100,
            window_seconds=60.0,
            clock=fc.clock,
            sleeper=fc.sleeper,
        )

        limiter.acquire(40)
        limiter.record_actual_tokens(estimated_tokens=40, actual_tokens=90)

        fc.advance(5.0)
        # 90 + 20 = 110 > 100 -> must wait until t=60.0 (55s)
        slept = limiter.acquire(20)
        assert slept == pytest.approx(55.0)
        assert fc.now == pytest.approx(60.0)


# ============================================================================
# 2. LiteLLMClient Rate Limiter & Retry Integration Tests
# ============================================================================


class TestLiteLLMClientRateLimitingAndRetries:
    """Tests LiteLLMClient rate limiting, RAP_TPM/RAP_RPM env config, usage tracking, and retries."""

    @pytest.fixture(autouse=True)
    def _clean_rate_env(self, monkeypatch: pytest.MonkeyPatch) -> Any:
        known_vars = [
            "RAP_TPM",
            "RAP_RPM",
            "RAP_MODEL_LIMITS",
            "RAP_TPM_GEMINI_GEMMA_4_31B_IT",
            "RAP_RPM_GEMINI_GEMMA_4_31B_IT",
            "RAP_TPM_GEMMA_4_31B_IT",
            "RAP_RPM_GEMMA_4_31B_IT",
            "RAP_TPM_OPENAI_GPT_4O_MINI",
            "RAP_RPM_OPENAI_GPT_4O_MINI",
            "RAP_TPM_GPT_4O_MINI",
            "RAP_RPM_GPT_4O_MINI",
        ]
        existing_dynamic = [
            k
            for k in list(os.environ)
            if k.startswith(("RAP_TPM", "RAP_RPM")) or k == "RAP_MODEL_LIMITS"
        ]
        for var in set(known_vars + existing_dynamic):
            monkeypatch.setenv(var, "")
            monkeypatch.delenv(var, raising=False)
        yield
        for k in list(os.environ):
            if k.startswith(("RAP_TPM", "RAP_RPM")) or k == "RAP_MODEL_LIMITS":
                monkeypatch.delenv(k, raising=False)

    def test_client_accepts_rate_limiter_parameters(self) -> None:
        """LiteLLMClient accepts TPM, RPM, max_retries, rate_limiter, and sleeper."""
        fc = FakeClock()
        custom_limiter = TokenRateLimiter(tokens_per_minute=5000, requests_per_minute=20)
        client = LiteLLMClient(
            model="gpt-4o-mini",
            tokens_per_minute=5000,
            requests_per_minute=20,
            max_retries=5,
            rate_limiter=custom_limiter,
            sleeper=fc.sleeper,
        )
        assert client.rate_limiter is custom_limiter
        assert client.max_retries == 5

    def test_client_builds_rate_limiter_from_explicit_tpm_and_rpm(self) -> None:
        """Explicit tokens_per_minute and requests_per_minute construct configured TokenRateLimiter."""
        client = LiteLLMClient(
            model="gpt-4o-mini",
            tokens_per_minute=20_000,
            requests_per_minute=60,
        )
        assert client.rate_limiter is not None
        assert client.rate_limiter.tokens_per_minute == 20_000
        assert client.rate_limiter.requests_per_minute == 60

    def test_client_reads_rap_tpm_and_rap_rpm_from_env_when_args_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Reads RAP_TPM and RAP_RPM from environment when constructor args are None."""
        monkeypatch.setenv("RAP_TPM", "15000")
        monkeypatch.setenv("RAP_RPM", "45")

        client = LiteLLMClient(model="gpt-4o-mini")
        assert client.rate_limiter is not None
        assert client.rate_limiter.tokens_per_minute == 15_000
        assert client.rate_limiter.requests_per_minute == 45

    def test_client_reads_rap_tpm_and_rap_rpm_from_dotenv(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Reads RAP_TPM and RAP_RPM from .env file when not already set in shell."""
        (tmp_path / ".env").write_text("RAP_TPM=12000\nRAP_RPM=30\n", encoding="utf-8")
        monkeypatch.chdir(tmp_path)

        client = LiteLLMClient(model="gpt-4o-mini")
        assert client.rate_limiter is not None
        assert client.rate_limiter.tokens_per_minute == 12_000
        assert client.rate_limiter.requests_per_minute == 30

    def test_explicit_tpm_rpm_override_env_vars(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Explicit tokens_per_minute/requests_per_minute take precedence over RAP_TPM/RAP_RPM."""
        monkeypatch.setenv("RAP_TPM", "15000")
        monkeypatch.setenv("RAP_RPM", "45")

        client = LiteLLMClient(
            model="gpt-4o-mini",
            tokens_per_minute=8000,
            requests_per_minute=10,
        )
        assert client.rate_limiter is not None
        assert client.rate_limiter.tokens_per_minute == 8000
        assert client.rate_limiter.requests_per_minute == 10

    @pytest.mark.parametrize("bad_val", ["0", "-10", "not-an-int", "12.5"])
    def test_invalid_rap_tpm_or_rap_rpm_env_raises_value_error(
        self, monkeypatch: pytest.MonkeyPatch, bad_val: str
    ) -> None:
        """Invalid non-positive or non-integer RAP_TPM/RAP_RPM raises ValueError."""
        monkeypatch.setenv("RAP_TPM", bad_val)
        with pytest.raises(ValueError):
            LiteLLMClient(model="gpt-4o-mini")

        monkeypatch.delenv("RAP_TPM")
        monkeypatch.setenv("RAP_RPM", bad_val)
        with pytest.raises(ValueError):
            LiteLLMClient(model="gpt-4o-mini")

    def test_client_enforces_acquire_before_call_and_records_actual_usage_object(self) -> None:
        """Calls rate_limiter.acquire() before completion and record_actual_tokens() with usage.total_tokens."""
        events: list[tuple[str, Any]] = []

        class SpyLimiter(TokenRateLimiter):
            def acquire(self, estimated_tokens: int) -> float:
                events.append(("acquire", estimated_tokens))
                return super().acquire(estimated_tokens)

            def record_actual_tokens(self, estimated_tokens: int, actual_tokens: int) -> None:
                events.append(("record", (estimated_tokens, actual_tokens)))
                super().record_actual_tokens(estimated_tokens, actual_tokens)

        limiter = SpyLimiter(tokens_per_minute=10_000, requests_per_minute=100)

        def fake_completion(**kwargs: Any) -> SimpleNamespace:
            events.append(("completion", kwargs["model"]))
            return _make_obj_response("hello", total_tokens=42)

        client = LiteLLMClient(
            model="gpt-4o-mini",
            rate_limiter=limiter,
            completion_fn=fake_completion,
        )
        result = client.generate("test prompt", max_tokens=64)

        assert result == "hello"
        assert [e[0] for e in events] == ["acquire", "completion", "record"]
        estimated_arg = events[0][1]
        assert isinstance(estimated_arg, int) and estimated_arg >= 64
        assert events[2][1] == (estimated_arg, 42)

    def test_client_records_actual_usage_from_dict_response(self) -> None:
        """Extracts usage['total_tokens'] from dict-style responses for rate_limiter."""
        recorded: list[tuple[int, int]] = []

        class SpyLimiter(TokenRateLimiter):
            def record_actual_tokens(self, estimated_tokens: int, actual_tokens: int) -> None:
                recorded.append((estimated_tokens, actual_tokens))
                super().record_actual_tokens(estimated_tokens, actual_tokens)

        client = LiteLLMClient(
            rate_limiter=SpyLimiter(tokens_per_minute=10_000),
            completion_fn=lambda **_: _make_dict_response("dict ok", total_tokens=77),
        )
        assert client.generate("prompt") == "dict ok"
        assert len(recorded) == 1
        assert recorded[0][1] == 77

    def test_client_retries_on_rate_limit_error_and_succeeds(self) -> None:
        """Retries on litellm.RateLimitError using sleeper and returns result when retry succeeds."""
        litellm = _import_litellm()
        fc = FakeClock()
        attempts = 0

        def flaky_completion(**_: Any) -> SimpleNamespace:
            nonlocal attempts
            attempts += 1
            if attempts <= 2:
                raise litellm.RateLimitError(
                    message="Rate limit reached",
                    llm_provider="openai",
                    model="gpt-4o-mini",
                )
            return _make_obj_response("recovered")

        client = LiteLLMClient(
            model="gpt-4o-mini",
            max_retries=3,
            sleeper=fc.sleeper,
            completion_fn=flaky_completion,
        )
        out = client.generate("prompt")

        assert out == "recovered"
        assert attempts == 3
        assert len(fc.sleeps) == 2
        assert all(s > 0 for s in fc.sleeps)

    def test_client_raises_llm_error_when_rate_limit_retries_exhausted(self) -> None:
        """Raises LLMError once max_retries RateLimitError retries are exhausted."""
        litellm = _import_litellm()
        fc = FakeClock()
        attempts = 0

        def always_rate_limited(**_: Any) -> Any:
            nonlocal attempts
            attempts += 1
            raise litellm.RateLimitError(
                message="Still rate limited",
                llm_provider="openai",
                model="gpt-4o-mini",
            )

        client = LiteLLMClient(
            model="gpt-4o-mini",
            max_retries=2,
            sleeper=fc.sleeper,
            completion_fn=always_rate_limited,
        )
        with pytest.raises(LLMError):
            client.generate("prompt")

        assert attempts == 3  # 1 initial + 2 retries
        assert len(fc.sleeps) == 2

    def test_client_does_not_retry_auth_error(self) -> None:
        """Non-transient AuthenticationError fails immediately without retry sleep."""
        litellm = _import_litellm()
        fc = FakeClock()
        attempts = 0

        def bad_auth(**_: Any) -> Any:
            nonlocal attempts
            attempts += 1
            raise litellm.AuthenticationError(
                message="Invalid key",
                llm_provider="openai",
                model="gpt-4o-mini",
            )

        client = LiteLLMClient(
            model="gpt-4o-mini",
            max_retries=3,
            sleeper=fc.sleeper,
            completion_fn=bad_auth,
        )
        with pytest.raises(LLMError):
            client.generate("prompt")

        assert attempts == 1
        assert fc.sleeps == []

    def test_client_retries_on_transient_internal_server_error(self) -> None:
        """Retries on transient litellm.InternalServerError / ServiceUnavailableError and succeeds."""
        litellm = _import_litellm()
        fc = FakeClock()
        attempts = 0

        def flaky_server(**_: Any) -> SimpleNamespace:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise litellm.InternalServerError(
                    message="Internal error encountered.",
                    llm_provider="gemini",
                    model="gemini/gemma-4-26b-a4b-it",
                )
            return _make_obj_response("recovered after 500")

        client = LiteLLMClient(
            model="gemini/gemma-4-26b-a4b-it",
            max_retries=2,
            sleeper=fc.sleeper,
            completion_fn=flaky_server,
        )
        assert client.generate("prompt") == "recovered after 500"
        assert attempts == 2
        assert len(fc.sleeps) == 1



# ============================================================================
# 3. CLI Integration Tests (--tpm / --rpm)
# ============================================================================


class TestRateLimiterCLIIntegration:
    """Tests --tpm / --rpm parser flags and _create_llm() wiring."""

    @pytest.mark.parametrize("subcommand", ["run", "ui", "compare"])
    def test_parser_accepts_tpm_and_rpm_flags_with_none_defaults(
        self, subcommand: str
    ) -> None:
        """build_parser() supports --tpm and --rpm (int, default None) on run, ui, and compare."""
        parser = build_parser()

        default_args = parser.parse_args([subcommand])
        assert default_args.tpm is None
        assert default_args.rpm is None

        parsed_args = parser.parse_args(
            [subcommand, "--backend", "litellm", "--tpm", "20000", "--rpm", "60"]
        )
        assert parsed_args.tpm == 20_000
        assert parsed_args.rpm == 60
        assert isinstance(parsed_args.tpm, int)
        assert isinstance(parsed_args.rpm, int)

    def test_create_llm_litellm_with_tpm_and_rpm_configures_rate_limiter(self) -> None:
        """_create_llm('litellm', model=..., tpm=20000, rpm=60) wires TokenRateLimiter."""
        llm = _create_llm("litellm", model="gpt-4o-mini", tpm=20_000, rpm=60)
        assert isinstance(llm, LiteLLMClient)
        assert llm.model == "gpt-4o-mini"
        assert llm.rate_limiter is not None
        assert llm.rate_limiter.tokens_per_minute == 20_000
        assert llm.rate_limiter.requests_per_minute == 60

    def test_create_llm_backward_compatibility(self) -> None:
        """_create_llm('mock') and _create_llm('litellm', model='gpt-4o-mini') keep working."""
        mock_llm = _create_llm("mock")
        assert isinstance(mock_llm, MockLLM)

        litellm_client = _create_llm("litellm", model="gpt-4o-mini")
        assert isinstance(litellm_client, LiteLLMClient)
        assert litellm_client.model == "gpt-4o-mini"

    def test_main_run_forwards_tpm_and_rpm_to_litellm_execution(self) -> None:
        """main(['run', '--backend', 'litellm', '--tpm', ..., '--rpm', ...]) runs cleanly."""
        litellm = _import_litellm()

        def fake_completion(**kwargs: Any) -> SimpleNamespace:
            if kwargs.get("logprobs") or kwargs.get("max_tokens") == 1:
                return _make_obj_response(
                    "Yes",
                    total_tokens=10,
                    logprobs=SimpleNamespace(
                        content=[
                            SimpleNamespace(
                                token="Yes",
                                logprob=math.log(0.9),
                                top_logprobs=[
                                    {"token": "Yes", "logprob": math.log(0.9)},
                                    {"token": "No", "logprob": math.log(0.1)},
                                ],
                            )
                        ]
                    ),
                )
            return _make_obj_response(
                json.dumps(
                    {
                        "thought": "Pick up red from table",
                        "action": "pick_up",
                        "block": "red",
                        "target": None,
                    }
                ),
                total_tokens=25,
            )

        with patch.object(litellm, "completion", side_effect=fake_completion):
            exit_code = main(
                [
                    "run",
                    "--backend",
                    "litellm",
                    "--model",
                    "gpt-4o-mini",
                    "--tpm",
                    "20000",
                    "--rpm",
                    "60",
                    "--iterations",
                    "1",
                ]
            )

        assert exit_code == 0


# ============================================================================
# 4. Per-Model Rate-Limit Configuration Tests
# ============================================================================


class TestPerModelRateLimitResolution:
    """Tests normalize_model_env_key() and resolve_model_rate_limits() precedence & validation."""

    @pytest.mark.parametrize(
        ("model", "expected"),
        [
            ("gemini/gemma-4-31b-it", "GEMINI_GEMMA_4_31B_IT"),
            ("openai/gpt-4o-mini", "OPENAI_GPT_4O_MINI"),
            ("anthropic/claude-3.5-sonnet:latest", "ANTHROPIC_CLAUDE_3_5_SONNET_LATEST"),
            ("  /my--model:v1.2/ ", "MY_MODEL_V1_2"),
            ("gpt-4o", "GPT_4O"),
        ],
    )
    def test_normalize_model_env_key(self, model: str, expected: str) -> None:
        """normalize_model_env_key uppercases and collapses non-alphanumeric runs to '_'."""
        assert normalize_model_env_key(model) == expected

    def test_resolve_precedence_order_full(self) -> None:
        """Verifies full precedence: explicit arg > full env > short env > RAP_MODEL_LIMITS > global."""
        model = "gemini/gemma-4-31b-it"
        all_env = {
            "RAP_TPM": "1000",
            "RAP_RPM": "10",
            "RAP_MODEL_LIMITS": json.dumps(
                {
                    "gemini/gemma-4-31b-it": {"tpm": 2000, "rpm": 20},
                }
            ),
            "RAP_TPM_GEMMA_4_31B_IT": "3000",
            "RAP_RPM_GEMMA_4_31B_IT": "30",
            "RAP_TPM_GEMINI_GEMMA_4_31B_IT": "4000",
            "RAP_RPM_GEMINI_GEMMA_4_31B_IT": "40",
        }

        # 1. Explicit args beat everything
        assert resolve_model_rate_limits(
            model, tokens_per_minute=9000, requests_per_minute=90, environ=all_env
        ) == (9000, 90)

        # 2. Full per-model env var beats short, JSON, and global
        assert resolve_model_rate_limits(model, environ=all_env) == (4000, 40)

        # 3. Short per-model env var beats JSON and global
        no_full = {
            k: v
            for k, v in all_env.items()
            if k not in ("RAP_TPM_GEMINI_GEMMA_4_31B_IT", "RAP_RPM_GEMINI_GEMMA_4_31B_IT")
        }
        assert resolve_model_rate_limits(model, environ=no_full) == (3000, 30)

        # 4. RAP_MODEL_LIMITS beats global
        no_per_model_vars = {
            "RAP_TPM": "1000",
            "RAP_RPM": "10",
            "RAP_MODEL_LIMITS": all_env["RAP_MODEL_LIMITS"],
        }
        assert resolve_model_rate_limits(model, environ=no_per_model_vars) == (2000, 20)

        # 5. Global fallback when no per-model limits match
        global_only = {"RAP_TPM": "1000", "RAP_RPM": "10"}
        assert resolve_model_rate_limits(model, environ=global_only) == (1000, 10)

        # 6. Nothing set -> (None, None)
        assert resolve_model_rate_limits(model, environ={}) == (None, None)

    def test_resolve_rap_model_limits_short_name_and_case_insensitive_matching(self) -> None:
        """RAP_MODEL_LIMITS matches exact key, then short name, then case-insensitive match."""
        # Short name fallback when provider-prefixed key is absent
        env_short = {
            "RAP_MODEL_LIMITS": json.dumps({"gemma-4-31b-it": {"tpm": 16000, "rpm": 25}})
        }
        assert resolve_model_rate_limits("gemini/gemma-4-31b-it", environ=env_short) == (
            16000,
            25,
        )

        # Case-insensitive fallback
        env_case = {
            "RAP_MODEL_LIMITS": json.dumps(
                {"GEMINI/GEMMA-4-31B-IT": {"tpm": 14000, "rpm": 20}}
            )
        }
        assert resolve_model_rate_limits("gemini/gemma-4-31b-it", environ=env_case) == (
            14000,
            20,
        )

    def test_resolve_partial_overrides_mix_sources_independently(self) -> None:
        """TPM and RPM resolve independently across explicit, per-model, JSON, and global sources."""
        env = {
            "RAP_TPM_GEMINI_GEMMA_4_31B_IT": "15000",
            "RAP_RPM": "60",
        }
        assert resolve_model_rate_limits("gemini/gemma-4-31b-it", environ=env) == (15000, 60)

        env_json = {
            "RAP_MODEL_LIMITS": json.dumps(
                {"gemini/gemma-4-31b-it": {"tpm": 15000, "rpm": 30}}
            )
        }
        assert resolve_model_rate_limits(
            "gemini/gemma-4-31b-it", requests_per_minute=12, environ=env_json
        ) == (15000, 12)

    @pytest.mark.parametrize(
        "bad_env",
        [
            {"RAP_MODEL_LIMITS": "{not valid json"},
            {"RAP_MODEL_LIMITS": "[1, 2, 3]"},
            {"RAP_MODEL_LIMITS": json.dumps({"gemini/gemma-4-31b-it": 15000})},
            {"RAP_MODEL_LIMITS": json.dumps({"gemini/gemma-4-31b-it": {"tpm": 0}})},
            {"RAP_MODEL_LIMITS": json.dumps({"gemini/gemma-4-31b-it": {"rpm": -5}})},
            {"RAP_MODEL_LIMITS": json.dumps({"gemini/gemma-4-31b-it": {"tpm": "fast"}})},
            {"RAP_MODEL_LIMITS": json.dumps({"gemini/gemma-4-31b-it": {"tpm": True}})},
            {"RAP_TPM_GEMINI_GEMMA_4_31B_IT": "0"},
            {"RAP_TPM_GEMINI_GEMMA_4_31B_IT": "-10"},
            {"RAP_RPM_GEMMA_4_31B_IT": "abc"},
            {"RAP_RPM_GEMINI_GEMMA_4_31B_IT": "12.5"},
        ],
    )
    def test_resolve_invalid_per_model_config_raises_value_error(
        self, bad_env: dict[str, str]
    ) -> None:
        """Invalid JSON, wrong types, or non-positive limits raise ValueError."""
        with pytest.raises(ValueError):
            resolve_model_rate_limits("gemini/gemma-4-31b-it", environ=bad_env)


class TestPerModelLiteLLMAndCLIIntegration:
    """Tests per-model .env limits with LiteLLMClient and CLI overrides."""

    @pytest.fixture(autouse=True)
    def _clean_rate_env(self, monkeypatch: pytest.MonkeyPatch) -> Any:
        known_vars = [
            "RAP_TPM",
            "RAP_RPM",
            "RAP_MODEL_LIMITS",
            "RAP_TPM_GEMINI_GEMMA_4_31B_IT",
            "RAP_RPM_GEMINI_GEMMA_4_31B_IT",
            "RAP_TPM_GEMMA_4_31B_IT",
            "RAP_RPM_GEMMA_4_31B_IT",
            "RAP_TPM_OPENAI_GPT_4O_MINI",
            "RAP_RPM_OPENAI_GPT_4O_MINI",
        ]
        for var in known_vars:
            monkeypatch.setenv(var, "")
            monkeypatch.delenv(var, raising=False)
        yield
        for k in list(os.environ):
            if k.startswith(("RAP_TPM", "RAP_RPM")) or k == "RAP_MODEL_LIMITS":
                monkeypatch.delenv(k, raising=False)

    def test_litellm_client_loads_per_model_limits_from_dotenv(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Per-model RAP_TPM_* / RAP_RPM_* in .env configure each model's rate_limiter."""
        (tmp_path / ".env").write_text(
            "\n".join(
                [
                    "RAP_TPM_GEMINI_GEMMA_4_31B_IT=15000",
                    "RAP_RPM_GEMINI_GEMMA_4_31B_IT=30",
                    "RAP_TPM_OPENAI_GPT_4O_MINI=200000",
                    "RAP_RPM_OPENAI_GPT_4O_MINI=500",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        monkeypatch.chdir(tmp_path)

        gemma_client = LiteLLMClient(model="gemini/gemma-4-31b-it")
        assert gemma_client.rate_limiter is not None
        assert gemma_client.rate_limiter.tokens_per_minute == 15_000
        assert gemma_client.rate_limiter.requests_per_minute == 30

        gpt_client = LiteLLMClient(model="openai/gpt-4o-mini")
        assert gpt_client.rate_limiter is not None
        assert gpt_client.rate_limiter.tokens_per_minute == 200_000
        assert gpt_client.rate_limiter.requests_per_minute == 500

    def test_litellm_client_loads_rap_model_limits_json_from_dotenv(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """RAP_MODEL_LIMITS JSON in .env configures per-model rate_limiter in LiteLLMClient."""
        limits_json = json.dumps(
            {
                "gemini/gemma-4-31b-it": {"tpm": 15000, "rpm": 30},
                "openai/gpt-4o-mini": {"tpm": 200000, "rpm": 500},
            }
        )
        (tmp_path / ".env").write_text(f"RAP_MODEL_LIMITS={limits_json}\n", encoding="utf-8")
        monkeypatch.chdir(tmp_path)

        gemma_client = LiteLLMClient(model="gemini/gemma-4-31b-it")
        assert gemma_client.rate_limiter is not None
        assert gemma_client.rate_limiter.tokens_per_minute == 15_000
        assert gemma_client.rate_limiter.requests_per_minute == 30

    def test_cli_tpm_and_rpm_flags_override_per_model_dotenv_limits(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Explicit --tpm / --rpm flags on CLI override per-model .env limits."""
        litellm = _import_litellm()
        (tmp_path / ".env").write_text(
            "RAP_TPM_GEMINI_GEMMA_4_31B_IT=15000\nRAP_RPM_GEMINI_GEMMA_4_31B_IT=30\n",
            encoding="utf-8",
        )
        monkeypatch.chdir(tmp_path)

        # Direct _create_llm check
        llm = _create_llm("litellm", model="gemini/gemma-4-31b-it", tpm=5000, rpm=10)
        assert isinstance(llm, LiteLLMClient)
        assert llm.rate_limiter is not None
        assert llm.rate_limiter.tokens_per_minute == 5000
        assert llm.rate_limiter.requests_per_minute == 10

        # End-to-end CLI `main(["run", ...])` verification
        created_clients: list[LiteLLMClient] = []
        orig_create_llm = _create_llm

        def spy_create_llm(*args: Any, **kwargs: Any) -> Any:
            client = orig_create_llm(*args, **kwargs)
            created_clients.append(client)
            return client

        def fake_completion(**kwargs: Any) -> SimpleNamespace:
            if kwargs.get("logprobs") or kwargs.get("max_tokens") == 1:
                return _make_obj_response("Yes", total_tokens=5)
            return _make_obj_response(
                json.dumps(
                    {
                        "thought": "Pick up red",
                        "action": "pick_up",
                        "block": "red",
                        "target": None,
                    }
                ),
                total_tokens=20,
            )

        with (
            patch("rap_demo.cli._create_llm", side_effect=spy_create_llm),
            patch.object(litellm, "completion", side_effect=fake_completion),
        ):
            rc = main(
                [
                    "run",
                    "--backend",
                    "litellm",
                    "--model",
                    "gemini/gemma-4-31b-it",
                    "--tpm",
                    "5000",
                    "--rpm",
                    "10",
                    "--iterations",
                    "1",
                    "--no-save",
                ]
            )

        assert rc == 0
        assert len(created_clients) == 1
        cli_limiter = created_clients[0].rate_limiter
        assert cli_limiter is not None
        assert cli_limiter.tokens_per_minute == 5000
        assert cli_limiter.requests_per_minute == 10


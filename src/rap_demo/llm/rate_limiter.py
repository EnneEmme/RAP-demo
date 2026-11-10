"""Thread-safe sliding-window token and request rate limiter."""

from __future__ import annotations

import json
import math
import os
import re
import threading
import time
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any


def normalize_model_env_key(model: str) -> str:
    """Normalize a model identifier into an uppercase env-var suffix."""
    return re.sub(r"[^A-Za-z0-9]+", "_", model).strip("_").upper()


def _parse_positive_int(raw: Any, label: str) -> int | None:
    """Validate and convert an env-var or JSON value into a positive integer."""
    if raw is None:
        return None
    if isinstance(raw, bool):
        raise ValueError(f"Invalid {label}={raw!r}: boolean is not a valid positive integer.")
    if isinstance(raw, int):
        if raw <= 0:
            raise ValueError(f"Invalid {label}={raw!r}: must be greater than 0.")
        return raw
    if isinstance(raw, str):
        stripped = raw.strip()
        if not stripped:
            return None
        try:
            value = int(stripped)
        except ValueError as exc:
            raise ValueError(
                f"Invalid {label}={raw!r}: expected a positive integer."
            ) from exc
        if value <= 0:
            raise ValueError(f"Invalid {label}={raw!r}: must be greater than 0.")
        return value
    raise ValueError(f"Invalid {label}={raw!r}: expected a positive integer.")


def _lookup_json_model_limits(
    model: str,
    env: Mapping[str, str],
) -> dict[str, Any] | None:
    """Parse RAP_MODEL_LIMITS JSON and return matching model entry, if any."""
    raw_json = env.get("RAP_MODEL_LIMITS")
    if raw_json is None or not raw_json.strip():
        return None

    try:
        parsed = json.loads(raw_json)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid RAP_MODEL_LIMITS JSON: {exc}") from exc

    if not isinstance(parsed, dict):
        raise ValueError("Invalid RAP_MODEL_LIMITS: expected a JSON object mapping model names to limits.")

    short_name = model.rsplit("/", 1)[-1] if "/" in model else None
    candidates: list[str] = [model]
    if short_name and short_name != model:
        candidates.append(short_name)

    matched_entry: Any = None
    matched_found = False

    for candidate in candidates:
        if candidate in parsed:
            matched_entry = parsed[candidate]
            matched_found = True
            break

    if not matched_found:
        lowered = {str(k).lower(): v for k, v in parsed.items()}
        for candidate in candidates:
            key_lower = candidate.lower()
            if key_lower in lowered:
                matched_entry = lowered[key_lower]
                matched_found = True
                break

    if not matched_found:
        return None

    if not isinstance(matched_entry, dict):
        raise ValueError(
            f"Invalid RAP_MODEL_LIMITS entry for {model!r}: expected an object with 'tpm'/'rpm'."
        )
    return matched_entry


def resolve_model_rate_limits(
    model: str,
    tokens_per_minute: int | None = None,
    requests_per_minute: int | None = None,
    environ: Mapping[str, str] | None = None,
) -> tuple[int | None, int | None]:
    """Resolve effective (tpm, rpm) for a model with per-model env/JSON precedence.

    Precedence independently for TPM and RPM:
    1. Explicit `tokens_per_minute` / `requests_per_minute` argument
    2. `RAP_TPM_<FULL_KEY>` / `RAP_RPM_<FULL_KEY>`
    3. `RAP_TPM_<SHORT_KEY>` / `RAP_RPM_<SHORT_KEY>` (for provider/model names)
    4. `RAP_MODEL_LIMITS` JSON map entry (`tpm` / `rpm`)
    5. Global `RAP_TPM` / `RAP_RPM`
    """
    env = environ if environ is not None else os.environ
    full_key = normalize_model_env_key(model)
    short_name = model.rsplit("/", 1)[-1] if "/" in model else None
    short_key = normalize_model_env_key(short_name) if short_name else None

    json_entry = (
        _lookup_json_model_limits(model, env)
        if (tokens_per_minute is None or requests_per_minute is None)
        else None
    )

    def _resolve_single(
        explicit: int | None,
        env_prefix: str,
        json_field: str,
        global_var: str,
    ) -> int | None:
        if explicit is not None:
            return _parse_positive_int(explicit, global_var)

        full_var = f"{env_prefix}_{full_key}"
        val = _parse_positive_int(env.get(full_var), full_var)
        if val is not None:
            return val

        if short_key and short_key != full_key:
            short_var = f"{env_prefix}_{short_key}"
            val = _parse_positive_int(env.get(short_var), short_var)
            if val is not None:
                return val

        if json_entry is not None and json_field in json_entry:
            val = _parse_positive_int(
                json_entry[json_field],
                f"RAP_MODEL_LIMITS[{model!r}][{json_field!r}]",
            )
            if val is not None:
                return val

        return _parse_positive_int(env.get(global_var), global_var)

    resolved_tpm = _resolve_single(tokens_per_minute, "RAP_TPM", "tpm", "RAP_TPM")
    resolved_rpm = _resolve_single(requests_per_minute, "RAP_RPM", "rpm", "RAP_RPM")
    return resolved_tpm, resolved_rpm


@dataclass
class _WindowEntry:
    """Single recorded request within the sliding rate-limit window."""

    timestamp: float
    tokens: int


class TokenRateLimiter:
    """Sliding-window rate limiter enforcing TPM and RPM budgets."""

    def __init__(
        self,
        tokens_per_minute: int | None = None,
        requests_per_minute: int | None = None,
        window_seconds: float = 60.0,
        clock: Callable[[], float] | None = None,
        sleeper: Callable[[float], None] | None = None,
    ) -> None:
        """Initialize the sliding-window rate limiter.

        Args:
            tokens_per_minute: Optional maximum tokens allowed per window.
            requests_per_minute: Optional maximum requests allowed per window.
            window_seconds: Sliding window duration in seconds (default: 60.0).
            clock: Optional monotonic clock callable for deterministic testing.
            sleeper: Optional sleep callable for deterministic testing.

        Raises:
            ValueError: If any provided limit or window_seconds is <= 0.
        """
        if tokens_per_minute is not None and tokens_per_minute <= 0:
            raise ValueError(
                f"tokens_per_minute must be positive, got {tokens_per_minute}."
            )
        if requests_per_minute is not None and requests_per_minute <= 0:
            raise ValueError(
                f"requests_per_minute must be positive, got {requests_per_minute}."
            )
        if window_seconds <= 0:
            raise ValueError(
                f"window_seconds must be positive, got {window_seconds}."
            )

        self.tokens_per_minute: int | None = tokens_per_minute
        self.requests_per_minute: int | None = requests_per_minute
        self.window_seconds: float = float(window_seconds)
        self._clock: Callable[[], float] = clock if clock is not None else time.monotonic
        self._sleeper: Callable[[float], None] = sleeper if sleeper is not None else time.sleep
        self._lock = threading.Lock()
        self._entries: deque[_WindowEntry] = deque()

    @staticmethod
    def estimate_tokens(text: str, max_tokens: int = 0) -> int:
        """Estimate request token budget deterministically from prompt length and max_tokens."""
        prompt_tokens = max(1, math.ceil(len(text) / 4))
        completion_tokens = max(0, max_tokens)
        return prompt_tokens + completion_tokens

    def _prune(self, now: float) -> None:
        """Remove entries that have fallen outside the active sliding window."""
        cutoff = now - self.window_seconds
        while self._entries and self._entries[0].timestamp <= cutoff:
            self._entries.popleft()

    def _compute_wait_time(self, now: float, estimated_tokens: int) -> float:
        """Compute minimum seconds to sleep before both RPM and TPM limits allow the call."""
        wait_until = now

        if (
            self.requests_per_minute is not None
            and len(self._entries) >= self.requests_per_minute
        ):
            excess_index = len(self._entries) - self.requests_per_minute
            rpm_ready = self._entries[excess_index].timestamp + self.window_seconds
            wait_until = max(wait_until, rpm_ready)

        if self.tokens_per_minute is not None:
            used_tokens = sum(entry.tokens for entry in self._entries)
            # Only block when window already has usage to avoid deadlocking oversized single calls.
            if used_tokens > 0 and used_tokens + estimated_tokens > self.tokens_per_minute:
                remaining = used_tokens
                for entry in self._entries:
                    remaining -= entry.tokens
                    wait_until = max(wait_until, entry.timestamp + self.window_seconds)
                    if remaining == 0 or remaining + estimated_tokens <= self.tokens_per_minute:
                        break

        return max(0.0, wait_until - now)

    def acquire(self, estimated_tokens: int) -> float:
        """Block until TPM and RPM budgets allow the request, then reserve capacity.

        Args:
            estimated_tokens: Estimated total tokens for the upcoming request.

        Returns:
            Total seconds spent sleeping while waiting for window capacity.
        """
        if self.tokens_per_minute is None and self.requests_per_minute is None:
            return 0.0

        reserved_tokens = max(0, estimated_tokens)
        total_slept = 0.0

        while True:
            with self._lock:
                now = self._clock()
                self._prune(now)
                wait_time = self._compute_wait_time(now, reserved_tokens)
                if wait_time <= 0.0:
                    self._entries.append(
                        _WindowEntry(timestamp=now, tokens=reserved_tokens)
                    )
                    return total_slept

            self._sleeper(wait_time)
            total_slept += wait_time

    def record_actual_tokens(self, estimated_tokens: int, actual_tokens: int) -> None:
        """Reconcile a previously acquired token estimate with actual reported usage."""
        actual = max(0, actual_tokens)
        with self._lock:
            if not self._entries:
                return
            for entry in reversed(self._entries):
                if entry.tokens == estimated_tokens:
                    entry.tokens = actual
                    return
            delta = actual - estimated_tokens
            self._entries[-1].tokens = max(0, self._entries[-1].tokens + delta)

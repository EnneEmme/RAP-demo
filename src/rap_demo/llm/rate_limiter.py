"""Thread-safe sliding-window token and request rate limiter."""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass


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

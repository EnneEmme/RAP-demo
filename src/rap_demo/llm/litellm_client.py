"""External API client powered by LiteLLM for multi-provider LLM access."""

from __future__ import annotations

import math
import os
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from rap_demo.llm.base import BaseLLM, LLMConnectionError, LLMError, LLMTimeoutError
from rap_demo.llm.rate_limiter import TokenRateLimiter, resolve_model_rate_limits


def _load_dotenv_file() -> None:
    """Load environment variables from the nearest .env file without overriding shell vars."""
    from dotenv import find_dotenv, load_dotenv

    dotenv_path = find_dotenv(usecwd=True)
    if dotenv_path:
        load_dotenv(dotenv_path=dotenv_path, override=False)


def _ensure_litellm_compat() -> None:
    """Configure offline cost-map defaults and Python 3.15 stdlib shims for litellm."""
    os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    if sys.version_info >= (3, 15):
        import re

        for legacy_name, replacement_attr in (
            ("sre_constants", "_constants"),
            ("sre_parse", "_parser"),
            ("sre_compile", "_compiler"),
        ):
            if legacy_name not in sys.modules and hasattr(re, replacement_attr):
                sys.modules[legacy_name] = getattr(re, replacement_attr)


_ensure_litellm_compat()


def _load_litellm() -> Any:
    """Lazily import litellm after applying compatibility shims."""
    _ensure_litellm_compat()
    import litellm

    litellm.suppress_debug_info = True
    return litellm


def _get_field(obj: Any, key: str, default: Any = None) -> Any:
    """Read an attribute or mapping key uniformly from object or dict responses."""
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _extract_first_choice(response: Any) -> Any:
    """Extract choices[0] from an object-style or dict-style response."""
    choices = _get_field(response, "choices")
    if not choices or not isinstance(choices, Sequence) or isinstance(choices, (str, bytes)):
        return None
    return choices[0]


def _extract_thinking_blocks_text(message: Any) -> str | None:
    """Extract concatenated text from LiteLLM/Anthropic/Gemini thinking_blocks if present."""
    blocks = _get_field(message, "thinking_blocks")
    if not isinstance(blocks, Sequence) or isinstance(blocks, (str, bytes)):
        return None
    parts: list[str] = []
    for block in blocks:
        text = _get_field(block, "thinking") or _get_field(block, "text")
        if text:
            parts.append(str(text))
    return "\n".join(parts) if parts else None


def _extract_message_content(choice: Any) -> str | None:
    """Extract choices[0].message.content, falling back to reasoning_content or thinking_blocks."""
    message = _get_field(choice, "message")
    content = _get_field(message, "content")
    if content is not None:
        text = str(content)
        if text:
            return text

    reasoning = _get_field(message, "reasoning_content")
    if reasoning is not None and str(reasoning).strip():
        return str(reasoning)

    thinking_text = _extract_thinking_blocks_text(message)
    if thinking_text and thinking_text.strip():
        return thinking_text

    if content is not None:
        return str(content)
    return None


def _is_unsupported_param_error(exc: Exception) -> bool:
    """Return True when exc indicates the provider does not support logprob parameters."""
    litellm = sys.modules.get("litellm")
    if litellm is None:
        return False
    unsupported_cls = getattr(litellm, "UnsupportedParamsError", None)
    if unsupported_cls is not None and isinstance(exc, unsupported_cls):
        return True
    bad_request_cls = getattr(litellm, "BadRequestError", None)
    if bad_request_cls is not None and isinstance(exc, bad_request_cls):
        msg = str(exc).lower()
        return any(
            marker in msg
            for marker in ("unsupported", "logprobs", "top_logprobs", "not supported")
        )
    return False


def _is_retryable_error(exc: Exception) -> bool:
    """Return True when exc is a transient rate-limit or 5xx server error."""
    litellm = sys.modules.get("litellm")
    if litellm is None:
        return False
    for attr in ("RateLimitError", "InternalServerError", "ServiceUnavailableError"):
        exc_cls = getattr(litellm, attr, None)
        if exc_cls is not None and isinstance(exc, exc_cls):
            return True
    return False


def _map_exception(exc: Exception) -> LLMError:
    """Translate standard library and litellm exceptions into the LLMError hierarchy."""
    if isinstance(exc, LLMError):
        return exc
    if isinstance(exc, TimeoutError):
        return LLMTimeoutError(f"LiteLLM request timed out: {exc}")
    if isinstance(exc, ConnectionError):
        return LLMConnectionError(f"LiteLLM connection failed: {exc}")

    litellm = sys.modules.get("litellm")
    if litellm is not None:
        # Check Timeout before APIConnectionError because litellm.Timeout inherits APIConnectionError.
        timeout_cls = getattr(litellm, "Timeout", None)
        if timeout_cls is not None and isinstance(exc, timeout_cls):
            return LLMTimeoutError(f"LiteLLM request timed out: {exc}")

        conn_cls = getattr(litellm, "APIConnectionError", None)
        if conn_cls is not None and isinstance(exc, conn_cls):
            return LLMConnectionError(f"LiteLLM connection failed: {exc}")

    return LLMError(f"LiteLLM error: {exc}")


def _extract_top_logprob_candidates(logprobs: Any) -> dict[str, float]:
    """Extract {token: logprob} candidates from chat or legacy top_logprobs structures."""
    if logprobs is None:
        return {}

    content_lps = _get_field(logprobs, "content")
    if isinstance(content_lps, Sequence) and len(content_lps) > 0:
        first_top = _get_field(content_lps[0], "top_logprobs")
        if isinstance(first_top, Sequence) and len(first_top) > 0:
            candidates: dict[str, float] = {}
            for item in first_top:
                tok = _get_field(item, "token")
                lp = _get_field(item, "logprob")
                if tok is not None and lp is not None:
                    candidates[str(tok)] = float(lp)
            return candidates

    legacy_top = _get_field(logprobs, "top_logprobs")
    if isinstance(legacy_top, Sequence) and len(legacy_top) > 0:
        first_map = legacy_top[0]
        if isinstance(first_map, Mapping):
            return {
                str(tok): float(lp)
                for tok, lp in first_map.items()
                if lp is not None
            }

    return {}


def _match_token_logprob(candidates: Mapping[str, float], token: str) -> float | None:
    """Find logprob for token using exact, leading-space, then stripped matching."""
    if token in candidates:
        return candidates[token]
    spaced = f" {token}"
    if spaced in candidates:
        return candidates[spaced]
    token_stripped = token.strip()
    for candidate_tok, lp in candidates.items():
        if candidate_tok.strip() == token_stripped:
            return lp
    return None


class LiteLLMClient(BaseLLM):
    """LLM backend client using LiteLLM for 100+ external API providers."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        api_base: str | None = None,
        timeout: float = 30.0,
        completion_fn: Callable[..., Any] | None = None,
        tokens_per_minute: int | None = None,
        requests_per_minute: int | None = None,
        max_retries: int = 3,
        rate_limiter: TokenRateLimiter | None = None,
        sleeper: Callable[[float], None] | None = None,
        config_path: str | Path | None = None,
    ) -> None:
        """Initialize LiteLLM client.

        Args:
            model: LiteLLM model identifier (e.g. 'gpt-4o-mini', 'anthropic/claude-3-5-sonnet-20241022').
            api_key: Optional API key overriding provider environment variable.
            api_base: Optional custom API base URL (e.g. proxy or self-hosted endpoint).
            timeout: Request timeout in seconds.
            completion_fn: Optional completion callable override for dependency injection/testing.
            tokens_per_minute: Optional TPM rate limit (falls back to env / models.json).
            requests_per_minute: Optional RPM rate limit (falls back to env / models.json).
            max_retries: Maximum retries on litellm.RateLimitError (default: 3).
            rate_limiter: Optional preconfigured TokenRateLimiter instance.
            sleeper: Optional sleep callable used for rate-limit retries and limiter waits.
            config_path: Optional path to models.json configuration file.

        Raises:
            ValueError: If model is empty/whitespace-only or rate-limit settings are invalid.
        """
        if not isinstance(model, str) or not model.strip():
            raise ValueError(f"Invalid model identifier: {model!r}. Expected non-empty string.")
        if max_retries < 0:
            raise ValueError(f"max_retries must be non-negative, got {max_retries}.")

        _load_dotenv_file()

        self.model: str = model.strip()
        effective_tpm, effective_rpm = resolve_model_rate_limits(
            self.model,
            tokens_per_minute=tokens_per_minute,
            requests_per_minute=requests_per_minute,
            config_path=config_path,
        )

        self.api_key: str | None = api_key
        self.api_base: str | None = api_base
        self.timeout: float = timeout
        self.max_retries: int = max_retries
        self._sleeper: Callable[[float], None] = sleeper if sleeper is not None else time.sleep
        self._completion_fn: Callable[..., Any] | None = completion_fn
        self._call_count: int = 0

        if rate_limiter is not None:
            self.rate_limiter: TokenRateLimiter | None = rate_limiter
        elif effective_tpm is not None or effective_rpm is not None:
            self.rate_limiter = TokenRateLimiter(
                tokens_per_minute=effective_tpm,
                requests_per_minute=effective_rpm,
                sleeper=sleeper,
            )
        else:
            self.rate_limiter = None

    @property
    def call_count(self) -> int:
        """Return the total number of completion requests issued by this client."""
        return self._call_count

    def _build_completion_kwargs(
        self,
        prompt: str,
        *,
        max_tokens: int,
        temperature: float,
        stop: list[str] | None = None,
        **extra: Any,
    ) -> dict[str, Any]:
        """Construct keyword arguments for litellm.completion, omitting None optionals."""
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "timeout": self.timeout,
        }
        if stop is not None:
            kwargs["stop"] = stop
        if self.api_key is not None:
            kwargs["api_key"] = self.api_key
        if self.api_base is not None:
            kwargs["api_base"] = self.api_base
        kwargs.update(extra)
        return kwargs

    def _estimate_request_tokens(self, kwargs: Mapping[str, Any]) -> int:
        """Estimate token usage from completion request messages and max_tokens."""
        messages = kwargs.get("messages") or []
        prompt_text = " ".join(
            str(_get_field(message, "content", "")) for message in messages
        )
        max_tokens = int(kwargs.get("max_tokens") or 0)
        return TokenRateLimiter.estimate_tokens(prompt_text, max_tokens=max_tokens)

    def _record_usage(self, estimated_tokens: int, response: Any) -> None:
        """Reconcile rate limiter budget using response.usage.total_tokens if present."""
        if self.rate_limiter is None:
            return
        usage = _get_field(response, "usage")
        total_tokens = _get_field(usage, "total_tokens")
        if total_tokens is not None:
            self.rate_limiter.record_actual_tokens(estimated_tokens, int(total_tokens))

    def _invoke_completion(self, kwargs: dict[str, Any]) -> Any:
        """Invoke completion with rate-limiting, retry on RateLimitError, and usage metering."""
        fn = self._completion_fn if self._completion_fn is not None else _load_litellm().completion
        estimated_tokens = self._estimate_request_tokens(kwargs)

        for attempt in range(self.max_retries + 1):
            if self.rate_limiter is not None:
                self.rate_limiter.acquire(estimated_tokens)
            self._call_count += 1
            try:
                response = fn(**kwargs)
            except Exception as exc:
                if _is_retryable_error(exc) and attempt < self.max_retries:
                    self._sleeper(0.5 * (2**attempt))
                    continue
                raise

            self._record_usage(estimated_tokens, response)
            return response

        raise LLMError("LiteLLM completion failed after exhausting retries")

    def generate(
        self,
        prompt: str,
        max_tokens: int = 128,
        temperature: float = 0.0,
        stop: list[str] | None = None,
    ) -> str:
        """Generate text completion from a prompt via LiteLLM.

        Args:
            prompt: Text prompt input to LLM.
            max_tokens: Maximum number of tokens to generate.
            temperature: Sampling temperature.
            stop: Optional list of stop sequences.

        Returns:
            The generated text string.

        Raises:
            LLMTimeoutError: If the request times out.
            LLMConnectionError: If the provider cannot be reached.
            LLMError: If the API returns an error or malformed response.
        """
        kwargs = self._build_completion_kwargs(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop,
        )
        try:
            response = self._invoke_completion(kwargs)
        except Exception as exc:
            raise _map_exception(exc) from exc

        choice = _extract_first_choice(response)
        if choice is None:
            raise LLMError("Malformed LLM response: missing or empty choices")

        content = _extract_message_content(choice)
        if content is None:
            raise LLMError("Malformed LLM response: missing message content")

        return content

    def get_logprobs(self, prompt: str, target: str) -> float:
        """Calculate total log-probability of target string given prompt.

        Falls back gracefully to 0.0 if the provider does not support logprobs.
        """
        kwargs = self._build_completion_kwargs(
            f"{prompt}{target}",
            max_tokens=1,
            temperature=0.0,
            logprobs=True,
        )
        try:
            response = self._invoke_completion(kwargs)
        except Exception as exc:
            if _is_unsupported_param_error(exc):
                return 0.0
            raise _map_exception(exc) from exc

        choice = _extract_first_choice(response)
        logprobs = _get_field(choice, "logprobs")
        if logprobs is None:
            return 0.0

        content_lps = _get_field(logprobs, "content")
        if isinstance(content_lps, Sequence) and not isinstance(content_lps, (str, bytes)):
            valid_lps = [
                float(lp)
                for item in content_lps
                if (lp := _get_field(item, "logprob")) is not None
            ]
            return float(sum(valid_lps)) if valid_lps else 0.0

        token_lps = _get_field(logprobs, "token_logprobs")
        if isinstance(token_lps, Sequence) and not isinstance(token_lps, (str, bytes)):
            valid_lps = [float(lp) for lp in token_lps if lp is not None]
            return float(sum(valid_lps)) if valid_lps else 0.0

        return 0.0

    def get_token_probability(self, prompt: str, token: str) -> float:
        """Extract the probability of a candidate token following prompt.

        Uses top_logprobs when supported by the provider, falling back to
        stripped content matching when top_logprobs are unavailable.
        """
        kwargs = self._build_completion_kwargs(
            prompt,
            max_tokens=1,
            temperature=0.0,
            logprobs=True,
            top_logprobs=10,
        )
        try:
            response = self._invoke_completion(kwargs)
        except Exception as exc:
            if not _is_unsupported_param_error(exc):
                raise _map_exception(exc) from exc

            fallback_kwargs = self._build_completion_kwargs(
                prompt,
                max_tokens=1,
                temperature=0.0,
            )
            try:
                response = self._invoke_completion(fallback_kwargs)
            except Exception as fallback_exc:
                if _is_unsupported_param_error(fallback_exc):
                    return 0.0
                raise _map_exception(fallback_exc) from fallback_exc

        choice = _extract_first_choice(response)
        if choice is None:
            return 0.0

        candidates = _extract_top_logprob_candidates(_get_field(choice, "logprobs"))
        if candidates:
            target_lp = _match_token_logprob(candidates, token)
            if target_lp is None:
                return 0.0
            return float(max(0.0, min(1.0, math.exp(target_lp))))

        content = _extract_message_content(choice)
        if content is None:
            return 0.0
        return 1.0 if content.strip() == token.strip() else 0.0

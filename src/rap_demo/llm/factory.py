"""Factory for instantiating LLM backends."""

from __future__ import annotations

from typing import Any
from rap_demo.llm.base import BaseLLM
from rap_demo.llm.litellm_client import LiteLLMClient
from rap_demo.llm.llama_client import LlamaServerClient
from rap_demo.llm.mock import MockLLM


def get_llm(backend: str = "mock", **kwargs: Any) -> BaseLLM:
    """Instantiate and return an LLM backend according to backend identifier.

    Args:
        backend: Name of backend (e.g. 'mock', 'llama_server', 'llama-server', 'litellm').
        **kwargs: Arguments forwarded to the backend constructor.

    Returns:
        Instance conforming to BaseLLM protocol.

    Raises:
        ValueError: If backend is invalid, unknown, or unsupported.
    """
    if not backend or not isinstance(backend, str) or not backend.strip():
        raise ValueError(f"Invalid backend name: {backend!r}. Expected non-empty string.")

    normalized = backend.strip().lower().replace("-", "_")

    if normalized == "mock":
        return MockLLM(**kwargs)

    if normalized in ("llama", "llama_server"):
        return LlamaServerClient(**kwargs)

    if normalized in ("litellm", "lite_llm"):
        return LiteLLMClient(**kwargs)

    raise ValueError(
        f"Unknown LLM backend '{backend}'. "
        "Supported backends: 'mock', 'llama' (or 'llama_server'), 'litellm'."
    )

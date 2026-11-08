"""LLM interface, mock backend, and local llama-server client for RAP."""

from __future__ import annotations

from rap_demo.llm.base import (
    BaseLLM,
    LLMError,
    LLMConnectionError,
    LLMTimeoutError,
)
from rap_demo.llm.mock import MockLLM, LLMCall
from rap_demo.llm.llama_client import LlamaServerClient
from rap_demo.llm.factory import get_llm

__all__ = [
    "BaseLLM",
    "LLMError",
    "LLMConnectionError",
    "LLMTimeoutError",
    "MockLLM",
    "LLMCall",
    "LlamaServerClient",
    "get_llm",
]

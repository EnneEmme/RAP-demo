"""HTTP client for local llama-server OpenAI-compatible API."""

from __future__ import annotations

import math
import subprocess
from typing import Any
import httpx
from rap_demo.llm.base import BaseLLM, LLMConnectionError, LLMError, LLMTimeoutError


class LlamaServerClient(BaseLLM):
    """Client connecting to llama-server OpenAI-compatible completions endpoints."""

    def __init__(
        self,
        base_url: str = "http://localhost:8080",
        timeout: float = 30.0,
        model: str | None = None,
        api_key: str | None = None,
        client: httpx.Client | None = None,
        server_command: list[str] | None = None,
    ) -> None:
        """Initialize llama-server client.

        Args:
            base_url: Base URL where llama-server is listening (default: http://localhost:8080).
            timeout: HTTP request timeout in seconds.
            model: Optional model identifier string for completions payload.
            api_key: Optional Bearer API key token.
            client: Optional preconfigured httpx.Client (e.g. for testing with mock transport).
            server_command: Optional command list to auto-launch llama-server as a subprocess.
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.model = model
        self.api_key = api_key
        self._server_command = server_command
        self._server_process: subprocess.Popen[bytes] | None = None

        self._headers: dict[str, str] = {
            "Content-Type": "application/json",
        }
        if api_key:
            self._headers["Authorization"] = f"Bearer {api_key}"

        if client is not None:
            self._client = client
        else:
            self._client = httpx.Client(timeout=timeout)

    def _post(self, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Perform HTTP POST request with error mapping to LLM exceptions.

        Args:
            endpoint: URL path segment (e.g. '/v1/completions').
            payload: JSON request body.

        Returns:
            Decoded JSON dictionary response.

        Raises:
            LLMTimeoutError: If the request times out.
            LLMConnectionError: If network connection to the server fails.
            LLMError: If HTTP error occurs or response is malformed.
        """
        url = f"{self.base_url}{endpoint}"
        try:
            response = self._client.post(
                url,
                json=payload,
                headers=self._headers,
                timeout=self.timeout,
            )
        except httpx.TimeoutException as exc:
            raise LLMTimeoutError(f"Request to {url} timed out: {exc}") from exc
        except (httpx.ConnectError, httpx.NetworkError) as exc:
            raise LLMConnectionError(f"Failed to connect to {url}: {exc}") from exc
        except httpx.HTTPError as exc:
            raise LLMError(f"HTTP error communicating with {url}: {exc}") from exc

        if response.status_code >= 400:
            error_detail = ""
            try:
                err_json = response.json()
                if isinstance(err_json, dict):
                    err_obj = err_json.get("error")
                    if isinstance(err_obj, dict):
                        error_detail = str(err_obj.get("message", ""))
                    elif isinstance(err_obj, str):
                        error_detail = err_obj
            except Exception:
                error_detail = response.text
            detail_str = f": {error_detail}" if error_detail else ""
            raise LLMError(f"HTTP {response.status_code} error from {url}{detail_str}")

        try:
            data = response.json()
        except Exception as exc:
            raise LLMError(f"Failed to decode JSON response from {url}: {exc}") from exc

        if not isinstance(data, dict):
            raise LLMError(f"Expected JSON object response, got {type(data).__name__}")

        return data

    def is_healthy(self) -> bool:
        """Check if llama-server instance is reachable and responding.

        Returns:
            True if /health returns HTTP 200, False otherwise.
        """
        url = f"{self.base_url}/health"
        try:
            response = self._client.get(url, timeout=self.timeout)
            return response.status_code == 200
        except Exception:
            return False

    def generate(
        self,
        prompt: str,
        max_tokens: int = 128,
        temperature: float = 0.0,
        stop: list[str] | None = None,
    ) -> str:
        """Generate text completion from a prompt via /v1/completions.

        Args:
            prompt: Text prompt input to LLM.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            stop: Optional list of stop sequences.

        Returns:
            The generated text string.

        Raises:
            LLMError: If choices list is empty or response is malformed.
        """
        payload: dict[str, Any] = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if self.model:
            payload["model"] = self.model
        if stop is not None:
            payload["stop"] = stop

        data = self._post("/v1/completions", payload)
        choices = data.get("choices")
        if not choices or not isinstance(choices, list) or len(choices) == 0:
            raise LLMError("Malformed LLM response: missing or empty choices")

        choice = choices[0]
        if not isinstance(choice, dict) or "text" not in choice:
            raise LLMError("Malformed LLM response: choice missing 'text' field")

        return str(choice["text"])

    def get_logprobs(self, prompt: str, target: str) -> float:
        """Calculate total log-probability of target string given prompt.

        Args:
            prompt: Context or prefix prompt.
            target: Target continuation string.

        Returns:
            Cumulative log probability sum.

        Raises:
            LLMError: If response is missing logprob data.
        """
        payload: dict[str, Any] = {
            "prompt": f"{prompt}{target}",
            "max_tokens": 0,
            "temperature": 0.0,
            "logprobs": 1,
            "echo": True,
        }
        if self.model:
            payload["model"] = self.model

        data = self._post("/v1/completions", payload)
        choices = data.get("choices")
        if not choices or not isinstance(choices, list) or len(choices) == 0:
            raise LLMError("Malformed LLM response: missing or empty choices")

        choice = choices[0]
        if not isinstance(choice, dict):
            raise LLMError("Malformed LLM response: choice is not a dict")

        logprobs_data = choice.get("logprobs")
        if not isinstance(logprobs_data, dict):
            raise LLMError("Malformed LLM response: missing 'logprobs' object")

        token_logprobs = logprobs_data.get("token_logprobs")
        if token_logprobs is None:
            raise LLMError("Malformed LLM response: missing 'token_logprobs'")

        valid_lps = [float(lp) for lp in token_logprobs if lp is not None]
        if not valid_lps:
            return 0.0
        return float(sum(valid_lps))

    def get_token_probability(self, prompt: str, token: str) -> float:
        """Extract the probability of a candidate token following prompt from top_logprobs.

        Args:
            prompt: Context or prefix prompt.
            token: Candidate token string (e.g. 'Yes' or 'No').

        Returns:
            Extracted probability converted via exp and clamped to [0.0, 1.0].
        """
        payload: dict[str, Any] = {
            "prompt": prompt,
            "max_tokens": 1,
            "temperature": 0.0,
            "logprobs": 10,
        }
        if self.model:
            payload["model"] = self.model

        data = self._post("/v1/completions", payload)
        choices = data.get("choices")
        if not choices or not isinstance(choices, list) or len(choices) == 0:
            raise LLMError("Malformed LLM response: missing or empty choices")

        choice = choices[0]
        if not isinstance(choice, dict):
            raise LLMError("Malformed LLM response: choice is not a dict")

        logprobs_data = choice.get("logprobs")
        if not isinstance(logprobs_data, dict):
            raise LLMError("Malformed LLM response: missing 'logprobs' object")

        top_logprobs_list = logprobs_data.get("top_logprobs")
        if not top_logprobs_list or not isinstance(top_logprobs_list, list):
            return 0.0

        first_top_logprobs: dict[str, float] = top_logprobs_list[0]
        if not isinstance(first_top_logprobs, dict):
            return 0.0

        # Search candidate token with exact match, leading space, or stripped match
        target_lp: float | None = None
        if token in first_top_logprobs:
            target_lp = first_top_logprobs[token]
        elif f" {token}" in first_top_logprobs:
            target_lp = first_top_logprobs[f" {token}"]
        else:
            token_stripped = token.strip()
            for k, lp in first_top_logprobs.items():
                if k.strip() == token_stripped:
                    target_lp = lp
                    break

        if target_lp is None:
            return 0.0

        prob = math.exp(target_lp)
        return float(max(0.0, min(1.0, prob)))

    def start_server(self) -> None:
        """Launch local llama-server subprocess if server_command was specified."""
        if self._server_command and self._server_process is None:
            self._server_process = subprocess.Popen(
                self._server_command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

    def stop_server(self) -> None:
        """Terminate auto-launched llama-server subprocess if running."""
        if self._server_process is not None:
            self._server_process.terminate()
            try:
                self._server_process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                self._server_process.kill()
            self._server_process = None

    def close(self) -> None:
        """Close underlying HTTP client session and terminate any auto-launched server."""
        self._client.close()
        self.stop_server()

    def __enter__(self) -> LlamaServerClient:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

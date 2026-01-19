"""
LLM client factory for Ollama and Together.ai operations.

Provides unified clients for:
- Ollama: Local embeddings and optional local LLM
- Together.ai: Production LLM inference with streaming support
"""

import json
import logging
from functools import lru_cache
from typing import AsyncIterator

import httpx
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from src.config import get_settings

logger = logging.getLogger(__name__)
tracer = trace.get_tracer("financial-10k-rag.llm")


# =============================================================================
# Exceptions
# =============================================================================


class OllamaError(Exception):
    """Base exception for Ollama client errors."""

    pass


class OllamaConnectionError(OllamaError):
    """Ollama server is not reachable."""

    pass


class OllamaModelNotFoundError(OllamaError):
    """Requested model is not available."""

    pass


class OllamaClient:
    """
    Ollama HTTP client for LLM and embedding operations.

    Provides both synchronous and asynchronous methods for:
    - Health checking
    - Chat/completion
    - Embedding generation (single and batch)
    """

    def __init__(
        self,
        base_url: str | None = None,
        llm_model: str | None = None,
        embedding_model: str | None = None,
        timeout: float = 120.0,
    ):
        """
        Initialize Ollama client.

        Args:
            base_url: Ollama API base URL (defaults to settings).
            llm_model: LLM model name (defaults to settings).
            embedding_model: Embedding model name (defaults to settings).
            timeout: Request timeout in seconds.
        """
        settings = get_settings()
        self.base_url = (base_url or settings.ollama_base_url).rstrip("/")
        self.llm_model = llm_model or settings.ollama_llm_model
        self.embedding_model = embedding_model or settings.ollama_embedding_model
        self.timeout = timeout
        self._sync_client: httpx.Client | None = None
        self._async_client: httpx.AsyncClient | None = None

    @property
    def sync_client(self) -> httpx.Client:
        """Get or create synchronous HTTP client."""
        if self._sync_client is None:
            self._sync_client = httpx.Client(
                base_url=self.base_url,
                timeout=self.timeout,
            )
        return self._sync_client

    @property
    def async_client(self) -> httpx.AsyncClient:
        """Get or create asynchronous HTTP client."""
        if self._async_client is None:
            self._async_client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=self.timeout,
            )
        return self._async_client

    def close(self) -> None:
        """Close HTTP clients."""
        if self._sync_client:
            self._sync_client.close()
            self._sync_client = None

    async def aclose(self) -> None:
        """Close async HTTP client."""
        if self._async_client:
            await self._async_client.aclose()
            self._async_client = None

    def health_check(self) -> bool:
        """
        Check if Ollama server is healthy.

        Returns:
            bool: True if server responds to /api/tags.
        """
        try:
            response = self.sync_client.get("/api/tags")
            response.raise_for_status()
            return True
        except Exception as e:
            logger.warning(f"Ollama health check failed: {e}")
            return False

    async def async_health_check(self) -> bool:
        """
        Async check if Ollama server is healthy.

        Returns:
            bool: True if server responds to /api/tags.
        """
        try:
            response = await self.async_client.get("/api/tags")
            response.raise_for_status()
            return True
        except Exception as e:
            logger.warning(f"Ollama health check failed: {e}")
            return False

    def list_models(self) -> list[str]:
        """
        List available models.

        Returns:
            list[str]: List of model names.
        """
        try:
            response = self.sync_client.get("/api/tags")
            response.raise_for_status()
            data = response.json()
            return [m["name"] for m in data.get("models", [])]
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return []

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((httpx.ConnectError, httpx.TimeoutException)),
    )
    def embed(self, text: str) -> list[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Text to embed.

        Returns:
            list[float]: Embedding vector.

        Raises:
            OllamaError: If embedding generation fails.
        """
        with tracer.start_as_current_span("ollama.embed") as span:
            span.set_attribute("embedding.model", self.embedding_model)
            span.set_attribute("embedding.input_length", len(text))

            try:
                response = self.sync_client.post(
                    "/api/embed",
                    json={
                        "model": self.embedding_model,
                        "input": text,
                    },
                )
                response.raise_for_status()
                data = response.json()

                embeddings = data.get("embeddings", [])
                if not embeddings:
                    raise OllamaError("No embeddings returned")

                span.set_attribute("embedding.dimension", len(embeddings[0]))
                return embeddings[0]

            except httpx.HTTPStatusError as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                if e.response.status_code == 404:
                    raise OllamaModelNotFoundError(
                        f"Model '{self.embedding_model}' not found"
                    )
                raise OllamaError(f"Embedding failed: {e}")
            except httpx.ConnectError as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise OllamaConnectionError(f"Cannot connect to Ollama: {e}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((httpx.ConnectError, httpx.TimeoutException)),
    )
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for multiple texts in a single request.

        Args:
            texts: List of texts to embed.

        Returns:
            list[list[float]]: List of embedding vectors.

        Raises:
            OllamaError: If embedding generation fails.
        """
        if not texts:
            return []

        with tracer.start_as_current_span("ollama.embed_batch") as span:
            span.set_attribute("embedding.model", self.embedding_model)
            span.set_attribute("embedding.batch_size", len(texts))
            span.set_attribute("embedding.total_chars", sum(len(t) for t in texts))

            try:
                response = self.sync_client.post(
                    "/api/embed",
                    json={
                        "model": self.embedding_model,
                        "input": texts,
                    },
                )
                response.raise_for_status()
                data = response.json()

                embeddings = data.get("embeddings", [])
                if len(embeddings) != len(texts):
                    raise OllamaError(
                        f"Expected {len(texts)} embeddings, got {len(embeddings)}"
                    )

                span.set_attribute("embedding.dimension", len(embeddings[0]) if embeddings else 0)
                return embeddings

            except httpx.HTTPStatusError as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                if e.response.status_code == 404:
                    raise OllamaModelNotFoundError(
                        f"Model '{self.embedding_model}' not found"
                    )
                raise OllamaError(f"Batch embedding failed: {e}")
            except httpx.ConnectError as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise OllamaConnectionError(f"Cannot connect to Ollama: {e}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((httpx.ConnectError, httpx.TimeoutException)),
    )
    async def async_embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Async generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed.

        Returns:
            list[list[float]]: List of embedding vectors.
        """
        if not texts:
            return []

        with tracer.start_as_current_span("ollama.async_embed_batch") as span:
            span.set_attribute("embedding.model", self.embedding_model)
            span.set_attribute("embedding.batch_size", len(texts))
            span.set_attribute("embedding.total_chars", sum(len(t) for t in texts))

            try:
                response = await self.async_client.post(
                    "/api/embed",
                    json={
                        "model": self.embedding_model,
                        "input": texts,
                    },
                )
                response.raise_for_status()
                data = response.json()

                embeddings = data.get("embeddings", [])
                if len(embeddings) != len(texts):
                    raise OllamaError(
                        f"Expected {len(texts)} embeddings, got {len(embeddings)}"
                    )

                span.set_attribute("embedding.dimension", len(embeddings[0]) if embeddings else 0)
                return embeddings

            except httpx.HTTPStatusError as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                if e.response.status_code == 404:
                    raise OllamaModelNotFoundError(
                        f"Model '{self.embedding_model}' not found"
                    )
                raise OllamaError(f"Async batch embedding failed: {e}")
            except httpx.ConnectError as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise OllamaConnectionError(f"Cannot connect to Ollama: {e}")

    def generate(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """
        Generate text completion.

        Args:
            prompt: User prompt.
            system: System prompt.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.

        Returns:
            str: Generated text.
        """
        settings = get_settings()
        payload = {
            "model": self.llm_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature or settings.llm_temperature,
                "num_predict": max_tokens or settings.llm_max_tokens,
            },
        }
        if system:
            payload["system"] = system

        try:
            response = self.sync_client.post("/api/generate", json=payload)
            response.raise_for_status()
            return response.json().get("response", "")
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                raise OllamaModelNotFoundError(f"Model '{self.llm_model}' not found")
            raise OllamaError(f"Generation failed: {e}")

    async def async_generate_stream(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        """
        Async streaming text generation.

        Args:
            prompt: User prompt.
            system: System prompt.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.

        Yields:
            str: Generated text chunks.
        """
        settings = get_settings()
        payload = {
            "model": self.llm_model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": temperature or settings.llm_temperature,
                "num_predict": max_tokens or settings.llm_max_tokens,
            },
        }
        if system:
            payload["system"] = system

        async with self.async_client.stream(
            "POST", "/api/generate", json=payload
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line:
                    import json

                    data = json.loads(line)
                    if "response" in data:
                        yield data["response"]


@lru_cache
def get_ollama_client() -> OllamaClient:
    """
    Get cached Ollama client instance.

    Returns:
        OllamaClient: Configured Ollama client.
    """
    return OllamaClient()


# =============================================================================
# Together.ai Client
# =============================================================================


class TogetherAIError(Exception):
    """Base exception for Together.ai client errors."""

    pass


class TogetherAIConnectionError(TogetherAIError):
    """Together.ai API is not reachable."""

    pass


class TogetherAIAuthenticationError(TogetherAIError):
    """Invalid API key."""

    pass


class TogetherAIClient:
    """
    Together.ai HTTP client for LLM operations.

    Provides async chat completion with streaming support.
    Uses the official Together.ai API directly via httpx.
    """

    BASE_URL = "https://api.together.xyz/v1"

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        timeout: float = 120.0,
    ):
        """
        Initialize Together.ai client.

        Args:
            api_key: Together.ai API key (defaults to settings).
            model: Model name (defaults to settings.together_model_agent).
            timeout: Request timeout in seconds.
        """
        settings = get_settings()
        self.api_key = api_key or settings.together_api_key
        self.model = model or settings.together_model_agent
        self.timeout = timeout
        self._async_client: httpx.AsyncClient | None = None
        self._sync_client: httpx.Client | None = None

        if not self.api_key:
            logger.warning("Together.ai API key not configured")

    @property
    def async_client(self) -> httpx.AsyncClient:
        """Get or create asynchronous HTTP client."""
        if self._async_client is None:
            self._async_client = httpx.AsyncClient(
                base_url=self.BASE_URL,
                timeout=self.timeout,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
            )
        return self._async_client

    @property
    def sync_client(self) -> httpx.Client:
        """Get or create synchronous HTTP client."""
        if self._sync_client is None:
            self._sync_client = httpx.Client(
                base_url=self.BASE_URL,
                timeout=self.timeout,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
            )
        return self._sync_client

    async def aclose(self) -> None:
        """Close async HTTP client."""
        if self._async_client:
            await self._async_client.aclose()
            self._async_client = None

    def close(self) -> None:
        """Close sync HTTP client."""
        if self._sync_client:
            self._sync_client.close()
            self._sync_client = None

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((httpx.ConnectError, httpx.TimeoutException)),
    )
    async def chat_completion(
        self,
        messages: list[dict],
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """
        Generate chat completion.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.
            model: Override model for this request.
            temperature: Sampling temperature (0.0-2.0).
            max_tokens: Maximum tokens to generate.

        Returns:
            str: Generated text response.

        Raises:
            TogetherAIError: If generation fails.
        """
        with tracer.start_as_current_span("together.chat_completion") as span:
            settings = get_settings()
            used_model = model or self.model
            used_temperature = temperature if temperature is not None else settings.llm_temperature
            used_max_tokens = max_tokens or settings.llm_max_tokens

            span.set_attribute("llm.provider", "together_ai")
            span.set_attribute("llm.model", used_model)
            span.set_attribute("llm.temperature", used_temperature)
            span.set_attribute("llm.max_tokens", used_max_tokens)
            span.set_attribute("llm.message_count", len(messages))

            payload = {
                "model": used_model,
                "messages": messages,
                "temperature": used_temperature,
                "max_tokens": used_max_tokens,
            }

            try:
                response = await self.async_client.post(
                    "/chat/completions",
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()

                choices = data.get("choices", [])
                if not choices:
                    raise TogetherAIError("No choices in response")

                content = choices[0].get("message", {}).get("content", "")

                # Record usage if available
                usage = data.get("usage", {})
                if usage:
                    span.set_attribute("llm.prompt_tokens", usage.get("prompt_tokens", 0))
                    span.set_attribute("llm.completion_tokens", usage.get("completion_tokens", 0))
                    span.set_attribute("llm.total_tokens", usage.get("total_tokens", 0))

                span.set_attribute("llm.response_length", len(content))
                return content

            except httpx.HTTPStatusError as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                if e.response.status_code == 401:
                    raise TogetherAIAuthenticationError("Invalid Together.ai API key")
                raise TogetherAIError(f"Chat completion failed: {e}")
            except httpx.ConnectError as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise TogetherAIConnectionError(f"Cannot connect to Together.ai: {e}")

    async def chat_completion_stream(
        self,
        messages: list[dict],
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        """
        Streaming chat completion.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.
            model: Override model for this request.
            temperature: Sampling temperature (0.0-2.0).
            max_tokens: Maximum tokens to generate.

        Yields:
            str: Generated text chunks.
        """
        settings = get_settings()
        payload = {
            "model": model or self.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else settings.llm_temperature,
            "max_tokens": max_tokens or settings.llm_max_tokens,
            "stream": True,
        }

        try:
            async with self.async_client.stream(
                "POST",
                "/chat/completions",
                json=payload,
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data_str = line[6:]  # Remove "data: " prefix
                        if data_str.strip() == "[DONE]":
                            break
                        try:
                            data = json.loads(data_str)
                            choices = data.get("choices", [])
                            if choices:
                                delta = choices[0].get("delta", {})
                                content = delta.get("content")
                                if content:
                                    yield content
                        except json.JSONDecodeError:
                            continue

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                raise TogetherAIAuthenticationError("Invalid Together.ai API key")
            raise TogetherAIError(f"Streaming chat completion failed: {e}")

    def chat_completion_sync(
        self,
        messages: list[dict],
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """
        Synchronous chat completion.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.
            model: Override model for this request.
            temperature: Sampling temperature (0.0-2.0).
            max_tokens: Maximum tokens to generate.

        Returns:
            str: Generated text response.
        """
        with tracer.start_as_current_span("together.chat_completion_sync") as span:
            settings = get_settings()
            used_model = model or self.model
            used_temperature = temperature if temperature is not None else settings.llm_temperature
            used_max_tokens = max_tokens or settings.llm_max_tokens

            span.set_attribute("llm.provider", "together_ai")
            span.set_attribute("llm.model", used_model)
            span.set_attribute("llm.temperature", used_temperature)
            span.set_attribute("llm.max_tokens", used_max_tokens)
            span.set_attribute("llm.message_count", len(messages))

            payload = {
                "model": used_model,
                "messages": messages,
                "temperature": used_temperature,
                "max_tokens": used_max_tokens,
            }

            try:
                response = self.sync_client.post(
                    "/chat/completions",
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()

                choices = data.get("choices", [])
                if not choices:
                    raise TogetherAIError("No choices in response")

                content = choices[0].get("message", {}).get("content", "")

                # Record usage if available
                usage = data.get("usage", {})
                if usage:
                    span.set_attribute("llm.prompt_tokens", usage.get("prompt_tokens", 0))
                    span.set_attribute("llm.completion_tokens", usage.get("completion_tokens", 0))
                    span.set_attribute("llm.total_tokens", usage.get("total_tokens", 0))

                span.set_attribute("llm.response_length", len(content))
                return content

            except httpx.HTTPStatusError as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                if e.response.status_code == 401:
                    raise TogetherAIAuthenticationError("Invalid Together.ai API key")
                raise TogetherAIError(f"Sync chat completion failed: {e}")
            except httpx.ConnectError as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise TogetherAIConnectionError(f"Cannot connect to Together.ai: {e}")


@lru_cache
def get_together_client(model: str | None = None) -> TogetherAIClient:
    """
    Get Together.ai client instance.

    Args:
        model: Optional model override.

    Returns:
        TogetherAIClient: Configured Together.ai client.
    """
    return TogetherAIClient(model=model)


@lru_cache
def get_together_small_client() -> TogetherAIClient:
    """
    Get Together.ai client configured with the small model.

    Used for intent classification and quality assessment.

    Returns:
        TogetherAIClient: Client configured with small model.
    """
    settings = get_settings()
    return TogetherAIClient(model=settings.together_model_small)

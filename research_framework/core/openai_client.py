"""
OpenAI API client with retry logic, rate limiting, and error handling.

This module provides a robust wrapper around the OpenAI API that:
- Handles authentication and API key management
- Implements exponential backoff retry logic
- Manages rate limits with intelligent throttling
- Provides unified error handling
- Tracks token usage and costs
- Supports both sync and async operations
"""

import asyncio
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum

import openai
from openai import AsyncOpenAI
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
import tiktoken

from config.settings import settings
from utils.logger import get_logger


logger = get_logger(__name__)


class ModelTier(Enum):
    """Model tier definitions for different use cases."""

    DEFAULT = "default"
    BUDGET = "budget"
    PREMIUM = "premium"
    FAST = "fast"


@dataclass
class TokenUsage:
    """Track token usage for cost monitoring."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    estimated_cost_usd: float

    def __str__(self) -> str:
        return (
            f"Tokens(prompt={self.prompt_tokens}, "
            f"completion={self.completion_tokens}, "
            f"total={self.total_tokens}, "
            f"cost=${self.estimated_cost_usd:.6f})"
        )


@dataclass
class CompletionResponse:
    """Structured response from chat completion."""

    content: str
    model: str
    usage: TokenUsage
    finish_reason: str
    raw_response: Optional[Any] = field(default=None, repr=False)


class RateLimitError(Exception):
    """Raised when rate limit is exceeded."""

    pass


class OpenAIClientError(Exception):
    """Base exception for OpenAI client errors."""

    pass


class OpenAIClient:
    """
    Wrapper for OpenAI API with enterprise features.

    Features:
    - Automatic retry with exponential backoff
    - Rate limit handling
    - Token counting and cost estimation
    - Error handling and logging
    - Connection pooling

    Example:
        >>> client = OpenAIClient()
        >>> response = await client.chat_completion(
        ...     messages=[{"role": "user", "content": "Hello!"}],
        ...     model="gpt-4o-mini"
        ... )
        >>> print(response.content)
    """

    # Model pricing per 1M tokens (input, output)
    MODEL_PRICING = {
        "gpt-4o-mini": (0.15, 0.60),
        "gpt-4o": (2.50, 10.00),
        "gpt-4-turbo": (10.00, 30.00),
        "gpt-4": (30.00, 60.00),
        "gpt-3.5-turbo": (0.50, 1.50),
    }

    def __init__(
        self,
        api_key: Optional[str] = None,
        organization: Optional[str] = None,
        max_retries: int = 3,
        timeout: int = 60,
        rate_limit_rpm: int = 60,
    ):
        """
        Initialize OpenAI client.

        Args:
            api_key: OpenAI API key (uses env var if not provided)
            organization: OpenAI organization ID
            max_retries: Maximum number of retry attempts
            timeout: Request timeout in seconds
            rate_limit_rpm: Rate limit in requests per minute
        """
        self.api_key = api_key or settings.OPENAI_API_KEY
        if not self.api_key:
            raise ValueError(
                "OpenAI API key required. Set OPENAI_API_KEY environment variable "
                "or pass api_key parameter."
            )

        self.organization = organization or settings.OPENAI_ORG_ID
        self.max_retries = max_retries
        self.timeout = timeout
        self.rate_limit_rpm = rate_limit_rpm

        # Initialize async client
        self.client = AsyncOpenAI(
            api_key=self.api_key,
            organization=self.organization,
            timeout=timeout,
            max_retries=0,  # We handle retries ourselves
        )

        # Token encoder for counting
        self._encoder = None

        # Rate limiting state
        self._request_times: List[float] = []
        self._rate_limit_lock = asyncio.Lock()

        # Usage tracking
        self.total_usage = TokenUsage(
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            estimated_cost_usd=0.0,
        )

        logger.info(
            f"OpenAI client initialized: timeout={timeout}s, "
            f"max_retries={max_retries}, rate_limit={rate_limit_rpm}rpm"
        )

    @property
    def encoder(self):
        """Lazy load token encoder."""
        if self._encoder is None:
            try:
                self._encoder = tiktoken.encoding_for_model("gpt-4o")
            except KeyError:
                self._encoder = tiktoken.get_encoding("cl100k_base")
        return self._encoder

    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text.

        Args:
            text: Text to count tokens for

        Returns:
            Number of tokens
        """
        if not text:
            return 0
        return len(self.encoder.encode(text))

    def count_messages_tokens(self, messages: List[Dict[str, str]]) -> int:
        """
        Count tokens in a list of messages.

        Args:
            messages: List of message dicts with 'role' and 'content'

        Returns:
            Estimated token count
        """
        # Base tokens per message (role, formatting)
        tokens_per_message = 4

        total = 0
        for message in messages:
            total += tokens_per_message
            total += self.count_tokens(message.get("content", ""))
            total += self.count_tokens(message.get("role", ""))

        # Add base overhead
        total += 3

        return total

    def estimate_cost(
        self, model: str, prompt_tokens: int, completion_tokens: int
    ) -> float:
        """
        Estimate API call cost.

        Args:
            model: Model name
            prompt_tokens: Number of prompt tokens
            completion_tokens: Number of completion tokens

        Returns:
            Estimated cost in USD
        """
        # Find matching model pricing
        pricing_key = None
        for key in self.MODEL_PRICING:
            if key in model:
                pricing_key = key
                break

        if pricing_key is None:
            logger.warning(f"Unknown model {model}, using gpt-4o-mini pricing")
            pricing_key = "gpt-4o-mini"

        input_price, output_price = self.MODEL_PRICING[pricing_key]

        cost = (prompt_tokens / 1_000_000) * input_price + (
            completion_tokens / 1_000_000
        ) * output_price

        return round(cost, 6)

    async def _wait_for_rate_limit(self) -> None:
        """Wait if necessary to respect rate limits."""
        async with self._rate_limit_lock:
            now = time.time()

            # Remove requests older than 1 minute
            self._request_times = [t for t in self._request_times if now - t < 60]

            if len(self._request_times) >= self.rate_limit_rpm:
                # Calculate wait time
                oldest = self._request_times[0]
                wait_time = 60 - (now - oldest) + 0.1
                if wait_time > 0:
                    logger.warning(f"Rate limit reached, waiting {wait_time:.1f}s")
                    await asyncio.sleep(wait_time)

            self._request_times.append(time.time())

    @retry(
        retry=retry_if_exception_type(
            (openai.RateLimitError, openai.APITimeoutError, openai.APIConnectionError)
        ),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> CompletionResponse:
        """
        Create chat completion with retry logic.

        Args:
            messages: List of message dicts with 'role' and 'content'
            model: Model to use (defaults to settings.DEFAULT_MODEL)
            temperature: Sampling temperature 0-2 (defaults to settings.TEMPERATURE)
            max_tokens: Maximum tokens in response (defaults to settings.MAX_TOKENS)
            **kwargs: Additional OpenAI API parameters

        Returns:
            CompletionResponse with content and metadata

        Raises:
            RateLimitError: If rate limit exceeded after retries
            OpenAIClientError: For other API errors
        """
        model = model or settings.DEFAULT_MODEL
        temperature = temperature if temperature is not None else settings.TEMPERATURE
        max_tokens = max_tokens or settings.MAX_TOKENS

        # Wait for rate limit
        await self._wait_for_rate_limit()

        try:
            # Count input tokens
            prompt_tokens = self.count_messages_tokens(messages)

            logger.debug(
                f"Chat completion request: model={model}, "
                f"prompt_tokens={prompt_tokens}, max_tokens={max_tokens}"
            )

            # Make API call
            response = await self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )

            # Extract usage info
            usage = response.usage
            completion_tokens = usage.completion_tokens
            total_tokens = usage.total_tokens
            actual_prompt_tokens = usage.prompt_tokens

            # Calculate cost
            cost = self.estimate_cost(model, actual_prompt_tokens, completion_tokens)

            # Update total usage tracking
            self.total_usage.prompt_tokens += actual_prompt_tokens
            self.total_usage.completion_tokens += completion_tokens
            self.total_usage.total_tokens += total_tokens
            self.total_usage.estimated_cost_usd += cost

            logger.info(
                f"Chat completion success: model={model}, "
                f"tokens={total_tokens}, cost=${cost:.6f}"
            )

            # Return structured response
            return CompletionResponse(
                content=response.choices[0].message.content or "",
                model=model,
                usage=TokenUsage(
                    prompt_tokens=actual_prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                    estimated_cost_usd=cost,
                ),
                finish_reason=response.choices[0].finish_reason,
                raw_response=response,
            )

        except openai.RateLimitError as e:
            logger.error(f"Rate limit exceeded: {e}")
            raise RateLimitError(f"OpenAI rate limit exceeded: {e}")

        except openai.APITimeoutError as e:
            logger.error(f"API timeout: {e}")
            raise

        except openai.APIConnectionError as e:
            logger.error(f"API connection error: {e}")
            raise

        except openai.APIError as e:
            logger.error(f"OpenAI API error: {e}")
            raise OpenAIClientError(f"OpenAI API error: {e}")

    async def chat_completion_with_tools(
        self,
        messages: List[Dict[str, str]],
        tools: List[Dict[str, Any]],
        model: Optional[str] = None,
        **kwargs,
    ) -> CompletionResponse:
        """
        Create chat completion with tool/function calling support.

        Args:
            messages: List of message dicts
            tools: List of tool definitions
            model: Model to use
            **kwargs: Additional parameters

        Returns:
            CompletionResponse with potential tool calls
        """
        return await self.chat_completion(
            messages=messages, model=model, tools=tools, **kwargs
        )

    def get_usage_summary(self) -> Dict[str, Any]:
        """
        Get summary of total API usage.

        Returns:
            Dictionary with usage statistics
        """
        return {
            "total_prompt_tokens": self.total_usage.prompt_tokens,
            "total_completion_tokens": self.total_usage.completion_tokens,
            "total_tokens": self.total_usage.total_tokens,
            "total_cost_usd": round(self.total_usage.estimated_cost_usd, 4),
        }

    def reset_usage(self) -> None:
        """Reset usage tracking counters."""
        self.total_usage = TokenUsage(
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            estimated_cost_usd=0.0,
        )
        logger.info("Usage counters reset")

    async def close(self) -> None:
        """Close the client connection."""
        await self.client.close()
        logger.info("OpenAI client closed")

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()


# Singleton instance
_client_instance: Optional[OpenAIClient] = None


def get_client() -> OpenAIClient:
    """
    Get singleton OpenAI client instance.

    Returns:
        OpenAI client instance

    Raises:
        ValueError: If API key not configured
    """
    global _client_instance
    if _client_instance is None:
        _client_instance = OpenAIClient()
    return _client_instance


def reset_client() -> None:
    """Reset the singleton client instance."""
    global _client_instance
    _client_instance = None

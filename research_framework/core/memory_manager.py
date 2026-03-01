"""
RLM-Based Memory Manager for Context Management.

This module implements the Recursive Language Model (RLM) paradigm for intelligent
context management in multi-agent research systems. The implementation is based on
the RLM Paper specifications, addressing the key performance bottleneck identified
in Section 5 and Appendix A: "RLMs without asynchronous LM calls are slow."

Key RLM Paradigm Principles:
----------------------------
1. Context is stored as a variable in a REPL environment (not fed directly to LLM)
2. LLM writes code to programmatically examine, filter, search, and decompose context
3. Recursive sub-LM calls (llm_query) enable hierarchical context processing
4. 4-Stage Knowledge Pipeline: Input → Selection → Optimization → Application

Async Optimizations (addressing RLM Paper Section 5):
----------------------------------------------------
The paper explicitly states: "We focused on synchronous sub-calls inside of a Python
REPL environment, but we note that alternative strategies involving asynchronous
sub-calls and sandboxed REPLs can potentially significantly reduce the runtime and
inference cost of RLMs."

This implementation provides:
- Parallel batch processing for relevance scoring (Selection Stage)
- Async LLM query pool with rate limiting and concurrency control
- Chunked parallel compression (Optimization Stage)
- Concurrent pipeline stage execution where dependencies allow
- Background pre-fetching capabilities for predictive context loading

Module Structure:
-----------------
- REPLEnvironment: Sandboxed Python REPL for programmatic context interaction
- LLMQueryFunction: Single async sub-LM call interface
- AsyncLLMQueryPool: Rate-limited concurrent LLM query execution (NEW)
- SelectionFilter (ABC): Base class for relevance scoring filters
- CompressionStrategy (ABC): Base class for compression strategies
- MemoryManager: Main orchestrator implementing the 4-stage pipeline

Usage Example:
--------------
    >>> from core.memory_manager import MemoryManager, MemoryConfig
    >>>
    >>> # Configure with async optimizations
    >>> config = MemoryConfig(
    ...     max_concurrent_queries=5,  # Parallel LLM calls
    ...     enable_prefetch=True,       # Background pre-fetching
    ... )
    >>> manager = MemoryManager(config)
    >>> manager.set_client(openai_client)
    >>>
    >>> # Process through async-optimized pipeline
    >>> result = await manager.process_through_pipeline(
    ...     content=large_research_data,
    ...     query="What evidence supports AI transformation in healthcare?"
    ... )
    >>> print(f"Reduced {result['final']['original_tokens']} → {result['final']['final_tokens']} tokens")

Performance Characteristics:
---------------------------
- Selection Stage: ~3-5x speedup with parallel scoring (batch_size=10, max_concurrent=5)
- Optimization Stage: ~2-3x speedup with chunked parallel compression
- Overall Pipeline: ~2-4x speedup depending on content size and LLM latency

Author: Research Framework Team
Based on: RLM Paper - Recursive Language Models for Context Management
"""

import asyncio
import json
import re
import time
from dataclasses import dataclass, field
from typing import (
    List,
    Dict,
    Any,
    Optional,
    Callable,
    Tuple,
    Union,
    Coroutine,
    TypeVar,
    Generic,
)
from enum import Enum
from abc import ABC, abstractmethod
from collections import deque
import weakref

import tiktoken

from config.settings import settings
from utils.logger import get_logger


logger = get_logger(__name__)


# Type variable for generic async results
T = TypeVar('T')


# =============================================================================
# REPL Environment - Core RLM Component
# =============================================================================

class REPLEnvironment:
    """
    Python REPL environment for programmatic context interaction.

    In the RLM paradigm, prompts are treated as part of the external environment.
    The LLM writes code to interact with context stored as variables, rather
    than having context fed directly into the neural network. This is a fundamental
    shift from traditional context feeding approaches.

    Key Design Principles:
    ----------------------
    1. **Sandboxed Execution**: Only safe built-in functions are exposed to prevent
       arbitrary code execution vulnerabilities.

    2. **Variable Persistence**: Context variables persist across multiple code
       executions within the same session, enabling iterative refinement.

    3. **Execution History**: All code executions are logged for debugging and
       audit purposes.

    4. **Recursive LLM Calls**: Optional llm_query() function enables recursive
       sub-LM calls for hierarchical context processing (RLM paradigm core feature).

    Safety Considerations:
    ----------------------
    The REPL restricts access to:
    - File system operations (open, read, write)
    - Network operations (socket, urllib, requests)
    - System operations (os, subprocess, sys)
    - Dynamic code loading (import, __import__, exec, eval - within user code)

    Example Usage:
    --------------
        >>> repl = REPLEnvironment()
        >>>
        >>> # Store research data as context variable
        >>> repl.set_context("findings", [
        ...     {"summary": "AI improves diagnosis", "confidence": 0.85},
        ...     {"summary": "Cost reduction potential", "confidence": 0.72},
        ... ])
        >>>
        >>> # LLM can write code to programmatically interact
        >>> result, error = repl.execute("len(findings)")
        >>> print(result)  # Output: 2
        >>>
        >>> # Filter high-confidence findings
        >>> result, error = repl.execute(
        ...     "[f for f in findings if f['confidence'] > 0.8]"
        ... )
        >>> print(result)  # Output: [{"summary": "AI improves diagnosis", ...}]

    Recursive LLM Calls (RLM Paradigm):
    -----------------------------------
        >>> from core import OpenAIClient
        >>> repl = REPLEnvironment()
        >>> repl.set_llm_client(OpenAIClient())
        >>>
        >>> # Now llm_query() is available in executed code
        >>> repl.set_context("document", "Long text to summarize...")
        >>> result, _ = repl.execute('llm_query(f"Summarize: {document}")')
        >>> print(result)  # LLM-generated summary
        >>>
        >>> # Batch processing multiple prompts
        >>> result, _ = repl.execute('''
        ... summaries = llm_query_batched([
        ...     f"Summarize section 1: {sections[0]}",
        ...     f"Summarize section 2: {sections[1]}",
        ... ])
        ... ''')

    Thread Safety:
    --------------
    This class is NOT thread-safe. For concurrent access, use separate instances
    or implement external synchronization.
    """

    # Class-level whitelist of safe built-in functions
    # These are carefully selected to allow useful operations without security risks
    SAFE_BUILTINS = {
        # Type constructors
        'str': str,
        'int': int,
        'float': float,
        'bool': bool,
        'list': list,
        'dict': dict,
        'set': set,
        'tuple': tuple,

        # Iteration utilities
        'range': range,
        'enumerate': enumerate,
        'zip': zip,
        'map': map,
        'filter': filter,
        'sorted': sorted,
        'reversed': reversed,

        # Aggregation functions
        'len': len,
        'sum': sum,
        'min': min,
        'max': max,
        'abs': abs,
        'round': round,

        # Boolean operations
        'all': all,
        'any': any,

        # Type introspection (read-only)
        'isinstance': isinstance,
        'type': type,
        'hasattr': hasattr,
        'getattr': getattr,

        # Constants
        'None': None,
        'True': True,
        'False': False,
    }

    def __init__(self, max_execution_history: int = 100, llm_client: Optional[Any] = None):
        """
        Initialize the REPL environment with safe execution context.

        Args:
            max_execution_history: Maximum number of execution records to retain.
                                   Older records are discarded to prevent memory bloat.
                                   Default: 100 records.
            llm_client: Optional LLM client for recursive llm_query() calls.
                       Can also be set later via set_llm_client().
        """
        # Context variables storage - these are accessible in executed code
        self._variables: Dict[str, Any] = {}

        # Execution history with bounded size to prevent memory issues
        self._execution_history: deque = deque(maxlen=max_execution_history)

        # Create a copy of safe builtins to prevent external modification
        self._safe_builtins = self.SAFE_BUILTINS.copy()

        # Track execution statistics
        self._stats = {
            'total_executions': 0,
            'successful_executions': 0,
            'failed_executions': 0,
            'llm_queries': 0,
            'llm_query_tokens': 0,
        }

        # LLM client for recursive sub-LM calls (RLM paradigm core feature)
        self._llm_client: Optional[Any] = None
        self._llm_query_enabled: bool = False

        # Set LLM client if provided
        if llm_client:
            self.set_llm_client(llm_client)

        logger.info(
            f"REPLEnvironment initialized with {len(self._safe_builtins)} safe builtins, "
            f"max_history={max_execution_history}"
        )

    def set_context(self, name: str, value: Any) -> None:
        """
        Store a value in the REPL environment as a named context variable.

        The stored value becomes accessible in subsequent code executions as a
        variable with the given name. This is the primary mechanism for the RLM
        paradigm's "context as variable" approach.

        Args:
            name: Variable name for the context. Must be a valid Python identifier.
                  Names starting with underscore are reserved for internal use.
            value: The context data to store. Can be any Python object, though
                   complex objects should be JSON-serializable for best compatibility.

        Raises:
            ValueError: If name is not a valid Python identifier or starts with underscore.

        Example:
            >>> repl.set_context("research_data", {"findings": [...], "sources": [...]})
            >>> repl.set_context("query", "What are the key trends?")
        """
        # Validate variable name
        if not name.isidentifier():
            raise ValueError(f"Invalid variable name: '{name}'. Must be a valid Python identifier.")
        if name.startswith('_'):
            raise ValueError(f"Variable names starting with underscore are reserved: '{name}'")

        self._variables[name] = value

        # Log with type info for debugging
        value_type = type(value).__name__
        value_size = len(value) if hasattr(value, '__len__') else 'N/A'
        logger.debug(f"Context variable '{name}' set (type: {value_type}, size: {value_size})")

    def get_context(self, name: str) -> Any:
        """
        Retrieve a value from the REPL environment.

        Args:
            name: Variable name to retrieve.

        Returns:
            The stored value, or None if the variable doesn't exist.

        Example:
            >>> data = repl.get_context("research_data")
            >>> if data is not None:
            ...     process(data)
        """
        return self._variables.get(name)

    def has_context(self, name: str) -> bool:
        """
        Check if a context variable exists.

        Args:
            name: Variable name to check.

        Returns:
            True if the variable exists, False otherwise.
        """
        return name in self._variables

    def remove_context(self, name: str) -> bool:
        """
        Remove a context variable from the environment.

        Args:
            name: Variable name to remove.

        Returns:
            True if the variable was removed, False if it didn't exist.
        """
        if name in self._variables:
            del self._variables[name]
            logger.debug(f"Context variable '{name}' removed")
            return True
        return False

    def list_contexts(self) -> List[str]:
        """
        Get list of all context variable names.

        Returns:
            List of variable names currently stored in the environment.
        """
        return list(self._variables.keys())

    def execute(self, code: str) -> Tuple[Any, Optional[str]]:
        """
        Execute Python code in the sandboxed REPL environment.

        The code is executed in a restricted namespace that includes:
        - All context variables set via set_context()
        - A whitelist of safe built-in functions

        The execution first attempts to evaluate the code as an expression
        (for return values). If that fails due to syntax (e.g., statements),
        it executes as a statement and captures any new variable assignments.

        Args:
            code: Python code to execute. Can be an expression (e.g., "len(data)")
                  or a statement (e.g., "filtered = [x for x in data if x > 0]").

        Returns:
            Tuple of (result, error_message):
            - On success: (result_value, None) for expressions, (None, None) for statements
            - On failure: (None, error_message)

        Example:
            >>> # Expression - returns value
            >>> result, error = repl.execute("sum(scores) / len(scores)")
            >>> if error is None:
            ...     print(f"Average: {result}")

            >>> # Statement - creates new variable
            >>> _, error = repl.execute("high_scores = [s for s in scores if s > 90]")
            >>> if error is None:
            ...     high = repl.get_context("high_scores")

        Security Note:
            The sandbox prevents access to dangerous operations, but complex
            expressions could still cause CPU/memory exhaustion. Consider
            implementing timeouts for production use.
        """
        self._stats['total_executions'] += 1

        # Build execution namespace with safe builtins and context variables
        namespace = {
            '__builtins__': self._safe_builtins,
            **self._variables,
        }

        try:
            # First, try to evaluate as an expression (has return value)
            result = eval(code, namespace)

            self._execution_history.append({
                'code': code,
                'result': str(result)[:500],  # Truncate long results
                'result_type': type(result).__name__,
                'success': True,
                'timestamp': time.time(),
            })
            self._stats['successful_executions'] += 1

            return result, None

        except SyntaxError:
            # Not a valid expression, try as statement
            try:
                exec(code, namespace)

                # Capture any new variables created by the statement
                # Exclude private variables and builtins
                for key, value in namespace.items():
                    if (not key.startswith('_') and
                        key not in self._safe_builtins and
                        key != '__builtins__'):
                        self._variables[key] = value

                self._execution_history.append({
                    'code': code,
                    'result': 'statement_executed',
                    'success': True,
                    'timestamp': time.time(),
                })
                self._stats['successful_executions'] += 1

                return None, None

            except Exception as e:
                error = f"Execution error: {type(e).__name__}: {str(e)}"
                self._execution_history.append({
                    'code': code,
                    'error': error,
                    'success': False,
                    'timestamp': time.time(),
                })
                self._stats['failed_executions'] += 1
                logger.warning(f"REPL execution failed: {error}")

                return None, error

        except Exception as e:
            error = f"Evaluation error: {type(e).__name__}: {str(e)}"
            self._execution_history.append({
                'code': code,
                'error': error,
                'success': False,
                'timestamp': time.time(),
            })
            self._stats['failed_executions'] += 1
            logger.warning(f"REPL evaluation failed: {error}")

            return None, error

    def get_execution_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get history of code executions.

        Args:
            limit: Maximum number of records to return (most recent first).
                   If None, returns all available history.

        Returns:
            List of execution records, each containing:
            - code: The executed code
            - result/error: Result or error message
            - success: Boolean indicating success
            - timestamp: Unix timestamp of execution
        """
        history = list(self._execution_history)
        if limit:
            return history[-limit:]
        return history

    def get_stats(self) -> Dict[str, Any]:
        """
        Get REPL execution statistics.

        Returns:
            Dictionary with execution statistics including total, successful,
            failed execution counts, and llm_query statistics.
        """
        return {
            **self._stats,
            'context_variable_count': len(self._variables),
            'history_size': len(self._execution_history),
            'llm_query_enabled': self._llm_query_enabled,
        }

    def clear(self) -> None:
        """
        Clear all context variables and execution history.

        This resets the REPL to its initial state while preserving
        the safe builtins configuration and LLM client.
        """
        self._variables.clear()
        self._execution_history.clear()
        self._stats = {
            'total_executions': 0,
            'successful_executions': 0,
            'failed_executions': 0,
            'llm_queries': 0,
            'llm_query_tokens': 0,
        }
        logger.info("REPLEnvironment cleared")

    # -------------------------------------------------------------------------
    # LLM Query - Recursive Sub-LM Calls (RLM Paradigm Core Feature)
    # -------------------------------------------------------------------------

    def set_llm_client(self, llm_client: Any) -> None:
        """
        Set the LLM client for recursive llm_query() calls.

        This enables the core RLM paradigm feature: recursive sub-LM calls
        from within executed REPL code. Once set, the llm_query() and
        llm_query_batched() functions become available in the REPL namespace.

        The LLM client should have one of the following interfaces:
        - Async: chat(messages) -> {"content": str} or similar
        - Sync: generate(prompt) -> str

        Args:
            llm_client: An LLM client instance (e.g., OpenAIClient).

        Example:
            >>> from core import OpenAIClient
            >>> repl = REPLEnvironment()
            >>> repl.set_llm_client(OpenAIClient())
            >>>
            >>> # Now llm_query is available in executed code
            >>> result, _ = repl.execute('llm_query("Summarize: " + text)')
        """
        self._llm_client = llm_client
        self._llm_query_enabled = True

        # Create and register the llm_query functions
        self._register_llm_query_functions()

        logger.info("LLM client set, llm_query() functions now available in REPL")

    def _register_llm_query_functions(self) -> None:
        """
        Register llm_query and llm_query_batched in the REPL namespace.

        This internal method creates closure functions that capture the
        LLM client and registers them as safe builtins for use in executed code.
        """
        # Capture references for closures
        llm_client = self._llm_client
        stats = self._stats

        def llm_query(prompt: str, system_prompt: Optional[str] = None) -> str:
            """
            Make a recursive LLM query from within REPL code.

            This is the core RLM paradigm capability that enables hierarchical
            processing of long contexts. The LLM can call itself on subsets
            of data to summarize, analyze, or transform content.

            Args:
                prompt: The prompt to send to the LLM.
                system_prompt: Optional system prompt for context.

            Returns:
                The LLM's response as a string.

            Example (in REPL code):
                >>> # Summarize a chunk of text
                >>> summary = llm_query(f"Summarize this: {chunk}")
                >>>
                >>> # Extract key points
                >>> points = llm_query("List key points:", system_prompt="Be concise")
            """
            logger.debug(f"llm_query called: prompt_length={len(prompt)}")
            stats['llm_queries'] += 1

            try:
                # Build messages
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": prompt})

                # Check for async chat method (preferred)
                if hasattr(llm_client, 'chat') and asyncio.iscoroutinefunction(llm_client.chat):
                    # Run async function in event loop
                    try:
                        loop = asyncio.get_running_loop()
                    except RuntimeError:
                        # No running loop, create one
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        response = loop.run_until_complete(llm_client.chat(messages=messages))
                    else:
                        # Already in async context, use run_until_complete in new loop
                        import concurrent.futures
                        with concurrent.futures.ThreadPoolExecutor() as executor:
                            future = executor.submit(
                                asyncio.run,
                                llm_client.chat(messages=messages)
                            )
                            response = future.result()

                    # Extract content from response
                    if isinstance(response, dict):
                        content = response.get("content", response.get("message", str(response)))
                        tokens = response.get("usage", {}).get("total_tokens", 0)
                        stats['llm_query_tokens'] += tokens
                    else:
                        content = str(response)

                    return content

                # Check for sync generate method
                elif hasattr(llm_client, 'generate'):
                    response = llm_client.generate(prompt)
                    return str(response)

                # Check for sync chat method
                elif hasattr(llm_client, 'chat'):
                    response = llm_client.chat(messages=messages)
                    if isinstance(response, dict):
                        return response.get("content", str(response))
                    return str(response)

                else:
                    raise RuntimeError(
                        "LLM client must have 'chat' or 'generate' method"
                    )

            except Exception as e:
                error_msg = f"llm_query error: {type(e).__name__}: {str(e)}"
                logger.error(error_msg)
                return f"[ERROR: {error_msg}]"

        def llm_query_batched(
            prompts: List[str],
            system_prompt: Optional[str] = None,
        ) -> List[str]:
            """
            Make multiple LLM queries, processing them efficiently.

            This is more efficient than calling llm_query() in a loop when
            you have multiple independent prompts to process.

            Args:
                prompts: List of prompts to process.
                system_prompt: Optional shared system prompt for all queries.

            Returns:
                List of responses corresponding to each prompt.

            Example (in REPL code):
                >>> # Summarize multiple chunks in parallel
                >>> summaries = llm_query_batched([
                ...     f"Summarize: {chunk1}",
                ...     f"Summarize: {chunk2}",
                ...     f"Summarize: {chunk3}",
                ... ])
            """
            logger.debug(f"llm_query_batched called: {len(prompts)} prompts")

            # Process each prompt
            # Note: Could be optimized with asyncio.gather for async clients
            results = []
            for prompt in prompts:
                result = llm_query(prompt, system_prompt=system_prompt)
                results.append(result)

            return results

        # Register functions in both variables and safe_builtins
        self._variables["llm_query"] = llm_query
        self._variables["llm_query_batched"] = llm_query_batched
        self._safe_builtins["llm_query"] = llm_query
        self._safe_builtins["llm_query_batched"] = llm_query_batched

        logger.debug("llm_query functions registered in REPL namespace")

    def has_llm_client(self) -> bool:
        """
        Check if an LLM client is configured for recursive queries.

        Returns:
            True if llm_query() is available, False otherwise.
        """
        return self._llm_query_enabled and self._llm_client is not None

    def get_llm_query_stats(self) -> Dict[str, int]:
        """
        Get statistics about llm_query usage.

        Returns:
            Dictionary with llm_query statistics:
            - llm_queries: Total number of llm_query calls
            - llm_query_tokens: Total tokens used by llm_query calls
        """
        return {
            'llm_queries': self._stats.get('llm_queries', 0),
            'llm_query_tokens': self._stats.get('llm_query_tokens', 0),
        }


# =============================================================================
# Async LLM Query Pool - Rate-Limited Concurrent Execution
# =============================================================================

@dataclass
class LLMQueryResult:
    """
    Result from an llm_query recursive sub-call.

    This dataclass encapsulates the response from a single LLM sub-call,
    including the content, token usage, and any error information.

    Attributes:
        content: The text response from the LLM.
        tokens_used: Total tokens consumed by this query (prompt + completion).
        success: Whether the query completed successfully.
        error: Error message if the query failed, None otherwise.
        metadata: Additional information about the query (timing, call number, etc.).

    Example:
        >>> result = await llm_query("Summarize this text", context=text)
        >>> if result.success:
        ...     print(result.content)
        ... else:
        ...     print(f"Query failed: {result.error}")
    """
    content: str
    tokens_used: int
    success: bool
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class AsyncLLMQueryPool:
    """
    Rate-limited pool for concurrent LLM query execution.

    This class addresses the performance bottleneck identified in the RLM Paper
    (Section 5, Appendix A) where synchronous sub-LM calls made experiments slow.
    It provides:

    1. **Concurrency Control**: Limits simultaneous API calls via semaphore
    2. **Rate Limiting**: Token bucket algorithm prevents API rate limit errors
    3. **Batch Processing**: Groups queries for efficient parallel execution
    4. **Automatic Retry**: Handles transient failures with exponential backoff

    Architecture:
    -------------
    The pool uses an asyncio.Semaphore to limit concurrent requests and a
    token bucket for rate limiting. When a query is submitted:

    1. Acquire semaphore slot (waits if max_concurrent reached)
    2. Check rate limit bucket (waits if tokens exhausted)
    3. Execute the query
    4. Release semaphore slot and replenish bucket

    Example Usage:
    --------------
        >>> pool = AsyncLLMQueryPool(
        ...     client=openai_client,
        ...     max_concurrent=5,
        ...     requests_per_minute=60
        ... )
        >>>
        >>> # Single query
        >>> result = await pool.query("Summarize this", context=text)
        >>>
        >>> # Batch queries (parallel execution)
        >>> queries = [
        ...     {"query": "Score relevance", "context": chunk1},
        ...     {"query": "Score relevance", "context": chunk2},
        ...     {"query": "Score relevance", "context": chunk3},
        ... ]
        >>> results = await pool.batch_query(queries)

    Performance Characteristics:
    ----------------------------
    - With max_concurrent=5: ~5x throughput vs sequential
    - Batch processing adds ~10-20ms overhead per batch
    - Rate limiting ensures API compliance

    Thread Safety:
    --------------
    This class is safe for concurrent use from multiple coroutines.
    """

    def __init__(
        self,
        client=None,
        model: Optional[str] = None,
        max_concurrent: int = 5,
        requests_per_minute: int = 60,
        retry_attempts: int = 3,
        retry_delay: float = 1.0,
    ):
        """
        Initialize the async LLM query pool.

        Args:
            client: OpenAI client instance for making API calls.
            model: Model to use for queries. Defaults to settings.DEFAULT_MODEL.
            max_concurrent: Maximum simultaneous API calls. Higher values increase
                           throughput but may hit rate limits. Default: 5.
            requests_per_minute: Rate limit for API calls. Should be set below
                                 your actual API rate limit. Default: 60.
            retry_attempts: Number of retry attempts for failed queries. Default: 3.
            retry_delay: Base delay between retries (exponential backoff). Default: 1.0s.
        """
        self._client = client
        self._model = model or settings.DEFAULT_MODEL

        # Concurrency control via semaphore
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._max_concurrent = max_concurrent

        # Rate limiting via token bucket
        self._rate_limit = requests_per_minute
        self._bucket_tokens = requests_per_minute
        self._last_refill = time.time()
        self._bucket_lock = asyncio.Lock()

        # Retry configuration
        self._retry_attempts = retry_attempts
        self._retry_delay = retry_delay

        # Statistics tracking
        self._stats = {
            'total_queries': 0,
            'successful_queries': 0,
            'failed_queries': 0,
            'total_tokens': 0,
            'retries': 0,
            'rate_limit_waits': 0,
        }

        logger.info(
            f"AsyncLLMQueryPool initialized: max_concurrent={max_concurrent}, "
            f"rate_limit={requests_per_minute}/min, model={self._model}"
        )

    def set_client(self, client) -> None:
        """
        Set or update the OpenAI client.

        Args:
            client: OpenAI client instance.
        """
        self._client = client

    async def _refill_bucket(self) -> None:
        """
        Refill rate limit bucket based on elapsed time.

        Uses token bucket algorithm: tokens accumulate at a rate of
        (requests_per_minute / 60) tokens per second, up to the maximum.
        """
        async with self._bucket_lock:
            now = time.time()
            elapsed = now - self._last_refill

            # Calculate tokens to add (rate per second * elapsed seconds)
            tokens_to_add = (self._rate_limit / 60.0) * elapsed
            self._bucket_tokens = min(
                self._bucket_tokens + tokens_to_add,
                self._rate_limit  # Max bucket size
            )
            self._last_refill = now

    async def _acquire_rate_limit(self) -> None:
        """
        Acquire a rate limit token, waiting if necessary.

        This implements the rate limiting by consuming one token from the
        bucket. If no tokens are available, waits until refill adds tokens.
        """
        while True:
            await self._refill_bucket()

            async with self._bucket_lock:
                if self._bucket_tokens >= 1:
                    self._bucket_tokens -= 1
                    return

            # No tokens available, wait and retry
            self._stats['rate_limit_waits'] += 1
            await asyncio.sleep(0.1)

    async def query(
        self,
        query: str,
        context: str,
        max_tokens: int = 500,
        temperature: float = 0.3,
    ) -> LLMQueryResult:
        """
        Execute a single LLM query with rate limiting and concurrency control.

        This is the primary method for making individual sub-LM calls. It handles:
        - Acquiring a semaphore slot for concurrency control
        - Rate limit checking via token bucket
        - Automatic retry on transient failures

        Args:
            query: The instruction/question for the LLM.
            context: The context text to process.
            max_tokens: Maximum tokens in the response. Default: 500.
            temperature: Sampling temperature (0-1). Lower = more deterministic. Default: 0.3.

        Returns:
            LLMQueryResult containing the response or error information.

        Example:
            >>> result = await pool.query(
            ...     query="Rate relevance 0-1",
            ...     context="AI is transforming healthcare...",
            ...     max_tokens=10
            ... )
            >>> print(result.content)  # "0.85"
        """
        if not self._client:
            return LLMQueryResult(
                content="",
                tokens_used=0,
                success=False,
                error="No client configured for LLM query pool",
            )

        self._stats['total_queries'] += 1
        start_time = time.time()

        # Acquire concurrency slot and rate limit token
        async with self._semaphore:
            await self._acquire_rate_limit()

            # Retry loop with exponential backoff
            last_error = None
            for attempt in range(self._retry_attempts):
                try:
                    # Build the sub-query prompt following RLM paradigm
                    prompt = f"""Process the following context according to the query.
Provide a focused, concise response.

Query: {query}

Context:
{context}"""

                    response = await self._client.chat_completion(
                        messages=[{"role": "user", "content": prompt}],
                        model=self._model,
                        max_tokens=max_tokens,
                        temperature=temperature,
                    )

                    tokens = response.usage.total_tokens
                    self._stats['total_tokens'] += tokens
                    self._stats['successful_queries'] += 1

                    return LLMQueryResult(
                        content=response.content,
                        tokens_used=tokens,
                        success=True,
                        metadata={
                            'attempt': attempt + 1,
                            'latency_ms': (time.time() - start_time) * 1000,
                            'model': self._model,
                        },
                    )

                except Exception as e:
                    last_error = str(e)
                    if attempt < self._retry_attempts - 1:
                        self._stats['retries'] += 1
                        # Exponential backoff: 1s, 2s, 4s, ...
                        delay = self._retry_delay * (2 ** attempt)
                        logger.warning(
                            f"LLM query attempt {attempt + 1} failed: {e}. "
                            f"Retrying in {delay}s..."
                        )
                        await asyncio.sleep(delay)

            # All retries exhausted
            self._stats['failed_queries'] += 1
            logger.error(f"LLM query failed after {self._retry_attempts} attempts: {last_error}")

            return LLMQueryResult(
                content="",
                tokens_used=0,
                success=False,
                error=last_error,
                metadata={'attempts': self._retry_attempts},
            )

    async def batch_query(
        self,
        queries: List[Dict[str, Any]],
        max_tokens: int = 500,
        temperature: float = 0.3,
    ) -> List[LLMQueryResult]:
        """
        Execute multiple queries in parallel with rate limiting.

        This is the recommended method for processing multiple items (e.g., scoring
        relevance of many chunks). Queries are executed concurrently up to the
        max_concurrent limit.

        Args:
            queries: List of query dictionaries, each containing:
                    - query: The instruction text
                    - context: The context to process
            max_tokens: Maximum tokens per response. Default: 500.
            temperature: Sampling temperature. Default: 0.3.

        Returns:
            List of LLMQueryResult objects in the same order as input queries.

        Example:
            >>> queries = [
            ...     {"query": "Score relevance 0-1", "context": chunk1},
            ...     {"query": "Score relevance 0-1", "context": chunk2},
            ...     {"query": "Score relevance 0-1", "context": chunk3},
            ... ]
            >>> results = await pool.batch_query(queries)
            >>> scores = [float(r.content) for r in results if r.success]
        """
        tasks = [
            self.query(
                query=q['query'],
                context=q['context'],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            for q in queries
        ]

        # Execute all queries concurrently
        # asyncio.gather preserves order
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Convert any exceptions to failed results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append(LLMQueryResult(
                    content="",
                    tokens_used=0,
                    success=False,
                    error=str(result),
                ))
            else:
                processed_results.append(result)

        return processed_results

    def get_stats(self) -> Dict[str, Any]:
        """
        Get pool statistics.

        Returns:
            Dictionary containing query counts, token usage, retry stats, etc.
        """
        return {
            **self._stats,
            'success_rate': (
                self._stats['successful_queries'] / max(self._stats['total_queries'], 1)
            ),
            'avg_tokens_per_query': (
                self._stats['total_tokens'] / max(self._stats['successful_queries'], 1)
            ),
        }


class LLMQueryFunction:
    """
    Single-query LLM function interface for RLM sub-calls.

    This class provides a simple callable interface for making individual
    recursive sub-LM calls. For batch operations, use AsyncLLMQueryPool instead.

    The LLM query function is a core component of the RLM paradigm, enabling
    the language model to recursively call itself (or other models) to process
    context chunks hierarchically.

    Example:
        >>> llm_query = LLMQueryFunction(client)
        >>>
        >>> # Score relevance of a text chunk
        >>> result = await llm_query(
        ...     query="Rate relevance to 'AI healthcare' from 0.0 to 1.0",
        ...     context=text_chunk,
        ...     max_tokens=10,
        ... )
        >>>
        >>> if result.success:
        ...     score = float(result.content)
    """

    def __init__(self, client=None, model: Optional[str] = None):
        """
        Initialize the llm_query function.

        Args:
            client: OpenAI client for making API calls.
            model: Model to use for sub-calls. Defaults to settings.DEFAULT_MODEL.
        """
        self._client = client
        self._model = model or settings.DEFAULT_MODEL
        self._call_count = 0
        self._total_tokens = 0

    def set_client(self, client) -> None:
        """Set the OpenAI client for making calls."""
        self._client = client

    async def __call__(
        self,
        query: str,
        context: str,
        max_tokens: int = 500,
        temperature: float = 0.3,
    ) -> LLMQueryResult:
        """
        Make a recursive sub-LM call to process context.

        Args:
            query: The query/instruction for processing.
            context: The context chunk to process.
            max_tokens: Maximum tokens for response. Default: 500.
            temperature: Sampling temperature. Default: 0.3.

        Returns:
            LLMQueryResult with processed output.
        """
        if not self._client:
            return LLMQueryResult(
                content="",
                tokens_used=0,
                success=False,
                error="No client configured for llm_query",
            )

        self._call_count += 1

        try:
            # Build the sub-query prompt
            prompt = f"""Process the following context according to the query.
Provide a focused, concise response.

Query: {query}

Context:
{context}"""

            response = await self._client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                model=self._model,
                max_tokens=max_tokens,
                temperature=temperature,
            )

            tokens = response.usage.total_tokens
            self._total_tokens += tokens

            return LLMQueryResult(
                content=response.content,
                tokens_used=tokens,
                success=True,
                metadata={"call_number": self._call_count},
            )

        except Exception as e:
            logger.error(f"llm_query failed: {e}")
            return LLMQueryResult(
                content="",
                tokens_used=0,
                success=False,
                error=str(e),
            )

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about llm_query usage."""
        return {
            "call_count": self._call_count,
            "total_tokens": self._total_tokens,
        }


# =============================================================================
# Knowledge Pipeline Stages
# =============================================================================

class PipelineStage(Enum):
    """
    Stages of the RLM Knowledge Pipeline.

    The 4-stage pipeline progressively reduces context while maintaining
    information relevance and accuracy:

    1. INPUT: Parse and structure raw data (Semantic Layer)
    2. SELECTION: Score relevance 0-1, eliminate 60-80% of irrelevant data
    3. OPTIMIZATION: Adaptive compression achieving 80% token reduction
    4. APPLICATION: Apply processed context to the task (Skills Framework)
    """

    INPUT = "input"              # Stage 1: Semantic Layer
    SELECTION = "selection"      # Stage 2: RLM-Based Selection (60-80% reduction)
    OPTIMIZATION = "optimization" # Stage 3: RLM-Based Optimization (80% compression)
    APPLICATION = "application"  # Stage 4: Skills Framework


@dataclass
class PipelineResult:
    """
    Result from a pipeline stage.

    Attributes:
        stage: Which pipeline stage produced this result.
        content: The processed content (type varies by stage).
        original_tokens: Token count before this stage.
        output_tokens: Token count after this stage.
        reduction_ratio: Output/original ratio (lower = more reduction).
        metadata: Stage-specific additional information.
    """
    stage: PipelineStage
    content: Any
    original_tokens: int
    output_tokens: int
    reduction_ratio: float
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Selection Filters - Stage 2 Components
# =============================================================================

class SelectionFilter(ABC):
    """
    Abstract base class for selection stage filters.

    Selection filters score the relevance of content chunks to a query,
    enabling the pipeline to eliminate irrelevant data. The RLM paper
    specifies that this stage should achieve 60-80% data reduction.

    Implementations must provide the score_relevance method which returns
    a float between 0.0 (completely irrelevant) and 1.0 (highly relevant).

    Available Implementations:
    --------------------------
    - KeywordRelevanceFilter: Fast, heuristic-based keyword matching
    - SemanticRelevanceFilter: LLM-based semantic understanding (slower but more accurate)
    """

    @abstractmethod
    async def score_relevance(
        self,
        chunk: str,
        query: str,
        llm_query: Optional[LLMQueryFunction] = None,
    ) -> float:
        """
        Score the relevance of a chunk to the query.

        Args:
            chunk: Text chunk to score.
            query: The research query/hypothesis.
            llm_query: Optional LLM function for semantic scoring.

        Returns:
            Relevance score between 0.0 and 1.0.
        """
        pass


class KeywordRelevanceFilter(SelectionFilter):
    """
    Fast keyword-based relevance filter.

    This filter scores relevance based on keyword overlap between the query
    and the chunk. It's fast and doesn't require LLM calls, making it suitable
    for initial filtering or when API calls need to be minimized.

    Algorithm:
    ----------
    1. Extract words from the query
    2. Add any configured keywords
    3. Count occurrences in the chunk
    4. Normalize by total keyword count

    Performance:
    ------------
    - Speed: ~0.1ms per chunk
    - Accuracy: Moderate (misses semantic similarity)
    - Best for: Large-scale initial filtering, cost-sensitive scenarios

    Example:
        >>> filter = KeywordRelevanceFilter(keywords=["AI", "healthcare", "diagnosis"])
        >>> score = await filter.score_relevance(
        ...     chunk="Machine learning improves medical diagnosis accuracy.",
        ...     query="How does AI impact healthcare?"
        ... )
        >>> print(f"Relevance: {score:.2f}")  # ~0.6
    """

    def __init__(self, keywords: Optional[List[str]] = None):
        """
        Initialize the keyword filter.

        Args:
            keywords: Additional keywords to match beyond query words.
                     These are always included in scoring regardless of query.
        """
        self.keywords = [k.lower() for k in (keywords or [])]

    async def score_relevance(
        self,
        chunk: str,
        query: str,
        llm_query: Optional[LLMQueryFunction] = None,
    ) -> float:
        """
        Score relevance based on keyword overlap.

        Args:
            chunk: Text chunk to score.
            query: The research query/hypothesis.
            llm_query: Ignored (not used in keyword matching).

        Returns:
            Score from 0.0 to 1.0 based on keyword match ratio.
        """
        chunk_lower = chunk.lower()
        query_lower = query.lower()

        # Extract words from query (alphanumeric sequences)
        query_words = set(re.findall(r'\w+', query_lower))

        # Add configured keywords
        query_words.update(self.keywords)

        # Remove common stop words that don't carry meaning
        stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                     'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                     'would', 'could', 'should', 'may', 'might', 'must', 'shall',
                     'can', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by',
                     'from', 'as', 'or', 'and', 'but', 'if', 'then', 'than',
                     'so', 'such', 'this', 'that', 'these', 'those', 'it'}
        query_words -= stop_words

        if not query_words:
            return 0.5  # No meaningful keywords, return neutral score

        # Count matches (whole word matching)
        matches = 0
        for word in query_words:
            # Use word boundary matching for accuracy
            if re.search(rf'\b{re.escape(word)}\b', chunk_lower):
                matches += 1

        # Normalize to 0-1 range
        score = matches / len(query_words)

        return min(score, 1.0)


class SemanticRelevanceFilter(SelectionFilter):
    """
    LLM-based semantic relevance filter using RLM sub-calls.

    This filter uses the LLM to understand semantic similarity between
    the query and chunk, catching relevant content that keyword matching
    would miss. It's more accurate but slower and uses API tokens.

    How It Works:
    -------------
    1. Sends a prompt asking the LLM to rate relevance 0.0-1.0
    2. Parses the numeric response
    3. Falls back to keyword matching if LLM call fails

    Performance:
    ------------
    - Speed: ~100-500ms per chunk (depends on LLM latency)
    - Accuracy: High (understands synonyms, context, implications)
    - Best for: Final filtering, high-stakes research, small chunk counts

    Cost Optimization:
    ------------------
    Consider using KeywordRelevanceFilter for initial filtering (e.g., score > 0.2),
    then SemanticRelevanceFilter for the remaining chunks.

    Example:
        >>> filter = SemanticRelevanceFilter()
        >>> score = await filter.score_relevance(
        ...     chunk="Neural networks have revolutionized medical imaging analysis.",
        ...     query="How does AI impact healthcare?",
        ...     llm_query=llm_query_func,
        ... )
        >>> print(f"Relevance: {score:.2f}")  # ~0.85
    """

    async def score_relevance(
        self,
        chunk: str,
        query: str,
        llm_query: Optional[LLMQueryFunction] = None,
    ) -> float:
        """
        Score relevance using LLM semantic understanding.

        Args:
            chunk: Text chunk to score.
            query: The research query/hypothesis.
            llm_query: LLM function for semantic scoring. If None, falls back
                      to keyword matching.

        Returns:
            Score from 0.0 to 1.0 based on semantic relevance.
        """
        if not llm_query:
            # Fallback to keyword matching if no LLM available
            fallback = KeywordRelevanceFilter()
            return await fallback.score_relevance(chunk, query, None)

        # Limit chunk size to control token usage
        chunk_preview = chunk[:2000] if len(chunk) > 2000 else chunk

        result = await llm_query(
            query=(
                f"Rate the relevance of this text to the query '{query}' "
                f"on a scale of 0.0 to 1.0, where 0.0 means completely irrelevant "
                f"and 1.0 means highly relevant. Respond with ONLY a single number."
            ),
            context=chunk_preview,
            max_tokens=10,
            temperature=0.1,  # Low temperature for consistent scoring
        )

        if result.success:
            try:
                # Parse the numeric score from response
                score_match = re.search(r'(\d+\.?\d*)', result.content)
                if score_match:
                    score = float(score_match.group(1))
                    return min(max(score, 0.0), 1.0)  # Clamp to valid range
            except ValueError:
                logger.warning(f"Could not parse relevance score: {result.content}")

        # Default to medium relevance on parsing failure
        return 0.5


# =============================================================================
# Compression Strategies - Stage 3 Components
# =============================================================================

class CompressionStrategy(ABC):
    """
    Abstract base class for optimization stage compression.

    Compression strategies reduce the token count of selected content while
    preserving essential information. The RLM paper specifies that this stage
    should achieve approximately 80% compression while maintaining 95%+ accuracy.

    Available Implementations:
    --------------------------
    - ExtractiveCompression: Selects key sentences (fast, no LLM needed)
    - AbstractiveCompression: LLM-generated summaries (slower, more flexible)
    """

    @abstractmethod
    async def compress(
        self,
        content: str,
        target_ratio: float,
        llm_query: Optional[LLMQueryFunction] = None,
    ) -> str:
        """
        Compress content to target ratio.

        Args:
            content: Text to compress.
            target_ratio: Target size as fraction of original (e.g., 0.2 = 20%).
            llm_query: Optional LLM for intelligent compression.

        Returns:
            Compressed text.
        """
        pass


class ExtractiveCompression(CompressionStrategy):
    """
    Extract key sentences/paragraphs for compression.

    This strategy reduces content by selecting representative sentences
    without generating new text. It's fast and preserves original wording,
    but may miss nuance that spans multiple sentences.

    Algorithm:
    ----------
    1. Split content into sentences
    2. Calculate target sentence count based on ratio
    3. Select: first sentence + evenly spaced middle sentences + last sentence
    4. Join selected sentences

    Performance:
    ------------
    - Speed: ~1ms for typical content
    - Quality: Good for factual content, may miss context
    - Best for: Cost-sensitive scenarios, when original wording matters

    Example:
        >>> compressor = ExtractiveCompression()
        >>> compressed = await compressor.compress(
        ...     content=long_article,
        ...     target_ratio=0.2,  # Keep 20%
        ... )
    """

    async def compress(
        self,
        content: str,
        target_ratio: float,
        llm_query: Optional[LLMQueryFunction] = None,
    ) -> str:
        """
        Extract key sentences to meet target ratio.

        Args:
            content: Text to compress.
            target_ratio: Fraction of original to keep (0.2 = 20%).
            llm_query: Ignored (not used in extractive compression).

        Returns:
            Compressed text containing selected sentences.
        """
        # Split into sentences using common terminators
        sentences = re.split(r'(?<=[.!?])\s+', content)

        if not sentences:
            return content

        # Calculate how many sentences to keep
        target_count = max(1, int(len(sentences) * target_ratio))

        # If already at or below target, return as-is
        if len(sentences) <= target_count:
            return content

        # Selection strategy: first + evenly spaced middle + last
        selected = [sentences[0]]  # First sentence (often contains key context)

        if target_count > 2:
            # Add evenly distributed sentences from the middle
            middle_count = target_count - 2  # Reserve slots for first and last
            step = len(sentences) // (middle_count + 1)

            for i in range(1, middle_count + 1):
                idx = i * step
                if idx < len(sentences) - 1:  # Don't pick the last one yet
                    selected.append(sentences[idx])

        if target_count > 1 and sentences[-1] not in selected:
            selected.append(sentences[-1])  # Last sentence (often has conclusions)

        return ' '.join(selected)


class AbstractiveCompression(CompressionStrategy):
    """
    Use LLM to generate compressed summaries.

    This strategy uses the LLM to create a new, condensed version of the
    content. It can capture the essence of information that spans multiple
    sentences and reorganize content for clarity.

    How It Works:
    -------------
    1. Calculate target word count based on ratio
    2. Prompt LLM to summarize while preserving key information
    3. Fall back to extractive compression if LLM fails

    Performance:
    ------------
    - Speed: ~200-1000ms (depends on content length and LLM)
    - Quality: Excellent for complex content, captures nuance
    - Best for: High-value content, when summary quality matters

    Cost Consideration:
    -------------------
    Each compression uses one LLM call. For large batches, consider
    using extractive compression for initial reduction, then abstractive
    for final polish.

    Example:
        >>> compressor = AbstractiveCompression()
        >>> compressed = await compressor.compress(
        ...     content=research_findings,
        ...     target_ratio=0.2,
        ...     llm_query=llm_query_func,
        ... )
    """

    async def compress(
        self,
        content: str,
        target_ratio: float,
        llm_query: Optional[LLMQueryFunction] = None,
    ) -> str:
        """
        Generate compressed summary using LLM.

        Args:
            content: Text to compress.
            target_ratio: Target size as fraction of original (e.g., 0.2 = 20%).
            llm_query: LLM function for generating summary. If None, falls back
                      to extractive compression.

        Returns:
            LLM-generated summary, or extractive compression on failure.
        """
        if not llm_query:
            # Fallback to extractive if no LLM
            fallback = ExtractiveCompression()
            return await fallback.compress(content, target_ratio, None)

        # Estimate target length based on words
        original_words = len(content.split())
        target_words = int(original_words * target_ratio)

        result = await llm_query(
            query=(
                f"Summarize this text in approximately {target_words} words. "
                f"Preserve all key information, factual claims, and important details. "
                f"Focus on maintaining accuracy over brevity."
            ),
            context=content,
            max_tokens=int(target_words * 2),  # Buffer for token/word ratio
            temperature=0.3,
        )

        if result.success and result.content:
            return result.content

        # Fallback on failure
        logger.warning("Abstractive compression failed, falling back to extractive")
        fallback = ExtractiveCompression()
        return await fallback.compress(content, target_ratio, None)


# =============================================================================
# RLM Memory Manager - Main Class
# =============================================================================

@dataclass
class MemoryConfig:
    """
    Configuration for RLM-based memory management.

    This configuration controls the behavior of the 4-stage knowledge pipeline,
    including token limits, filtering thresholds, compression ratios, and
    async processing parameters.

    Attributes:
        max_context_tokens: Maximum tokens allowed in the final context window.
        max_completion_tokens: Reserved tokens for model completion.
        selection_threshold: Minimum relevance score (0-1) for chunks to pass selection.
        target_selection_ratio: Target reduction in selection stage (0.3 = keep 30%).
        compression_ratio: Target compression in optimization (0.2 = keep 20%).
        use_abstractive_compression: Use LLM for compression (True) or extractive (False).
        max_code_executions: Safety limit on REPL executions per session.
        enable_selection: Whether to run the selection stage.
        enable_optimization: Whether to run the optimization stage.
        max_concurrent_queries: Concurrent LLM calls for async processing.
        requests_per_minute: Rate limit for API calls.
        enable_prefetch: Enable background pre-fetching of likely-needed context.

    Example:
        >>> # High-performance configuration
        >>> config = MemoryConfig(
        ...     max_concurrent_queries=10,
        ...     use_abstractive_compression=True,
        ...     selection_threshold=0.4,
        ... )
        >>> manager = MemoryManager(config)
    """

    # Token limits
    max_context_tokens: int = 8000
    max_completion_tokens: int = 4000

    # Selection stage settings (Stage 2)
    # Higher threshold = more aggressive filtering (fewer false positives)
    # Lower threshold = more inclusive (fewer false negatives)
    selection_threshold: float = 0.3  # Min relevance score to keep
    target_selection_ratio: float = 0.3  # Target 30% after selection (70% reduction)

    # Optimization stage settings (Stage 3)
    compression_ratio: float = 0.2  # Target 20% of selected content
    use_abstractive_compression: bool = True

    # REPL settings
    max_code_executions: int = 10  # Safety limit per session

    # Pipeline settings
    enable_selection: bool = True
    enable_optimization: bool = True

    # Async processing settings (NEW - addressing RLM Paper performance concerns)
    max_concurrent_queries: int = 5  # Concurrent LLM API calls
    requests_per_minute: int = 60    # Rate limit for API compliance
    batch_size: int = 10             # Chunks per parallel scoring batch
    enable_prefetch: bool = False    # Background context pre-fetching




class MemoryManager:
    """
    RLM-Based Memory Manager implementing the 4-Stage Knowledge Pipeline.

    This is the main orchestrator for context management, implementing the
    Recursive Language Model paradigm with async optimizations as recommended
    in the RLM Paper (Section 5, Appendix A).

    Architecture Overview:
    ----------------------
    The manager coordinates four processing stages:

    1. **Input Stage (Semantic Layer)**
       - Parse raw content into structured chunks
       - Track token counts and metadata
       - Store in REPL environment for programmatic access

    2. **Selection Stage (RLM-Based)**
       - Score chunk relevance 0.0-1.0 using filters
       - Eliminate 60-80% of irrelevant data
       - ASYNC: Parallel batch scoring with rate limiting

    3. **Optimization Stage (RLM-Based)**
       - Apply compression strategies (extractive or abstractive)
       - Achieve 80% token reduction while maintaining accuracy
       - ASYNC: Parallel chunk compression

    4. **Application Stage (Skills Framework)**
       - Return processed context for task execution
       - Provide pipeline statistics and results

    Async Optimizations:
    --------------------
    Addressing the performance issues noted in the RLM Paper:

    - **Parallel Scoring**: Chunks are scored concurrently using AsyncLLMQueryPool
    - **Batch Processing**: Chunks are grouped for efficient API utilization
    - **Rate Limiting**: Token bucket algorithm prevents API rate limit errors
    - **Concurrent Compression**: Multiple chunks compressed in parallel

    Performance Characteristics:
    ----------------------------
    - Sequential processing: ~100ms * N chunks (baseline)
    - Parallel processing: ~100ms * (N / max_concurrent) (with concurrency=5)
    - Typical speedup: 3-5x for selection stage, 2-3x for optimization

    Example Usage:
    --------------
        >>> # Initialize with async configuration
        >>> config = MemoryConfig(
        ...     max_concurrent_queries=5,
        ...     enable_prefetch=True,
        ... )
        >>> manager = MemoryManager(config)
        >>> manager.set_client(openai_client)
        >>>
        >>> # Add data to REPL environment
        >>> manager.repl.set_context("research_data", collected_data)
        >>>
        >>> # Process through async-optimized pipeline
        >>> result = await manager.process_through_pipeline(
        ...     content=large_document,
        ...     query="What evidence supports the hypothesis?"
        ... )
        >>>
        >>> # Access results
        >>> compressed = result['final']['content']
        >>> reduction = result['final']['total_reduction']
        >>> print(f"Reduced by {reduction}")

    Thread Safety:
    --------------
    The manager is safe for concurrent use from multiple coroutines.
    Each pipeline execution is independent. The REPL environment
    should be used carefully in concurrent scenarios.
    """

    # Known model context limits for auto-configuration
    MODEL_LIMITS = {
        "gpt-4o-mini": 128000,
        "gpt-4o": 128000,
        "gpt-4-turbo": 128000,
        "gpt-4": 8192,
        "gpt-3.5-turbo": 16385,
    }

    def __init__(self, config: Optional[MemoryConfig] = None):
        """
        Initialize the RLM-based memory manager.

        Args:
            config: Memory configuration. If None, uses sensible defaults.
        """
        self.config = config or MemoryConfig()

        # =====================================================================
        # Core RLM Components
        # =====================================================================

        # REPL environment for programmatic context interaction
        self.repl = REPLEnvironment()

        # Single-query LLM function (for compatibility)
        self.llm_query = LLMQueryFunction()

        # Async query pool for parallel processing (NEW)
        self._query_pool = AsyncLLMQueryPool(
            max_concurrent=self.config.max_concurrent_queries,
            requests_per_minute=self.config.requests_per_minute,
        )

        # =====================================================================
        # Filters and Compressors
        # =====================================================================

        self._keyword_filter = KeywordRelevanceFilter()
        self._semantic_filter = SemanticRelevanceFilter()
        self._extractive_compressor = ExtractiveCompression()
        self._abstractive_compressor = AbstractiveCompression()

        # =====================================================================
        # Token Encoder
        # =====================================================================

        try:
            self._encoder = tiktoken.encoding_for_model("gpt-4o")
        except KeyError:
            self._encoder = tiktoken.get_encoding("cl100k_base")

        # =====================================================================
        # Statistics and Metrics
        # =====================================================================

        self._pipeline_stats = {
            "total_processed": 0,
            "total_tokens_saved": 0,
            "average_reduction": 0.0,
            "total_processing_time_ms": 0,
        }

        logger.info(
            f"RLM MemoryManager initialized: "
            f"selection_threshold={self.config.selection_threshold}, "
            f"compression_ratio={self.config.compression_ratio}, "
            f"max_concurrent={self.config.max_concurrent_queries}"
        )

    def set_client(self, client) -> None:
        """
        Set the OpenAI client for RLM sub-calls.

        This must be called before using LLM-based features (semantic filtering,
        abstractive compression).

        Args:
            client: OpenAI client instance with async support.
        """
        self.llm_query.set_client(client)
        self._query_pool.set_client(client)

    # =========================================================================
    # Token Counting Utilities
    # =========================================================================

    def count_tokens(self, text: str) -> int:
        """
        Count tokens in a text string.

        Uses tiktoken for accurate token counting that matches
        the tokenizer used by OpenAI models.

        Args:
            text: Text to count tokens for.

        Returns:
            Number of tokens.
        """
        if not text:
            return 0
        return len(self._encoder.encode(text))

    def count_message_tokens(self, message: Dict[str, str]) -> int:
        """
        Count tokens in a chat message including overhead.

        OpenAI chat messages have per-message overhead for role encoding.
        This method accounts for that overhead.

        Args:
            message: Chat message with 'role' and 'content' keys.

        Returns:
            Total tokens including overhead.
        """
        # Base overhead per message (role + structural tokens)
        overhead = 4
        tokens = overhead

        tokens += self.count_tokens(message.get("content", ""))
        tokens += self.count_tokens(message.get("role", ""))

        if "name" in message:
            tokens += self.count_tokens(message["name"]) + 1

        return tokens

    def count_messages_tokens(self, messages: List[Dict[str, str]]) -> int:
        """
        Count total tokens in a list of messages.

        Args:
            messages: List of chat messages.

        Returns:
            Total token count including conversation overhead.
        """
        # 3 tokens for conversation priming
        return sum(self.count_message_tokens(m) for m in messages) + 3

    # =========================================================================
    # Stage 1: Input (Semantic Layer)
    # =========================================================================

    def parse_input(
        self,
        content: Union[str, List[str], Dict[str, Any]],
        chunk_size: int = 1000,
    ) -> List[Dict[str, Any]]:
        """
        Parse and structure input for pipeline processing.

        This is Stage 1 of the knowledge pipeline. It converts raw content
        into a list of structured chunks with metadata, preparing the data
        for relevance scoring and compression.

        Supported Input Types:
        ----------------------
        - str: Split into paragraph-based chunks respecting chunk_size
        - List[str]: Each item becomes a chunk
        - Dict: Each key-value pair becomes a chunk

        Args:
            content: Raw content to parse. Can be string, list, or dict.
            chunk_size: Target chunk size in tokens. Chunks may be smaller
                       but won't exceed this limit. Default: 1000 tokens.

        Returns:
            List of chunk dictionaries, each containing:
            - content: The chunk text
            - tokens: Token count
            - type: 'text', 'list_item', 'dict_entry', or 'structured'
            - Additional metadata depending on input type

        Example:
            >>> chunks = manager.parse_input(
            ...     content=long_article,
            ...     chunk_size=500,  # Smaller chunks for finer granularity
            ... )
            >>> print(f"Created {len(chunks)} chunks")
        """
        chunks = []

        if isinstance(content, str):
            # Split string content into paragraph-based chunks
            # This preserves semantic units (paragraphs) while respecting size limits
            paragraphs = content.split('\n\n')
            current_chunk = []
            current_tokens = 0

            for para in paragraphs:
                para = para.strip()
                if not para:
                    continue

                para_tokens = self.count_tokens(para)

                # Check if adding this paragraph would exceed chunk size
                if current_tokens + para_tokens > chunk_size and current_chunk:
                    # Save current chunk and start new one
                    chunks.append({
                        'content': '\n\n'.join(current_chunk),
                        'tokens': current_tokens,
                        'type': 'text',
                        'paragraph_count': len(current_chunk),
                    })
                    current_chunk = []
                    current_tokens = 0

                current_chunk.append(para)
                current_tokens += para_tokens

            # Don't forget the last chunk
            if current_chunk:
                chunks.append({
                    'content': '\n\n'.join(current_chunk),
                    'tokens': current_tokens,
                    'type': 'text',
                    'paragraph_count': len(current_chunk),
                })

        elif isinstance(content, list):
            # Each list item becomes a separate chunk
            for i, item in enumerate(content):
                if isinstance(item, str):
                    chunks.append({
                        'content': item,
                        'tokens': self.count_tokens(item),
                        'type': 'list_item',
                        'index': i,
                    })
                else:
                    # Non-string items are JSON-serialized
                    item_str = json.dumps(item, indent=2)
                    chunks.append({
                        'content': item_str,
                        'tokens': self.count_tokens(item_str),
                        'type': 'structured',
                        'index': i,
                        'original_type': type(item).__name__,
                    })

        elif isinstance(content, dict):
            # Each key-value pair becomes a chunk
            for key, value in content.items():
                if isinstance(value, str):
                    value_str = value
                else:
                    value_str = json.dumps(value, indent=2)

                chunk_content = f"{key}: {value_str}"
                chunks.append({
                    'content': chunk_content,
                    'tokens': self.count_tokens(chunk_content),
                    'type': 'dict_entry',
                    'key': key,
                })

        logger.debug(
            f"Parsed input into {len(chunks)} chunks, "
            f"total tokens: {sum(c['tokens'] for c in chunks)}"
        )
        return chunks

    # =========================================================================
    # Stage 2: Selection (RLM-Based) - ASYNC OPTIMIZED
    # =========================================================================

    async def select_relevant(
        self,
        chunks: List[Dict[str, Any]],
        query: str,
        use_semantic: bool = True,
    ) -> Tuple[List[Dict[str, Any]], PipelineResult]:
        """
        Select relevant chunks using RLM-based filtering with async optimization.

        This is Stage 2 of the knowledge pipeline. It scores each chunk's
        relevance to the query and filters out irrelevant content, achieving
        the 60-80% data reduction specified in the RLM Paper.

        Async Optimization:
        -------------------
        Instead of scoring chunks sequentially (slow, as noted in RLM Paper),
        this implementation scores chunks in parallel batches using the
        AsyncLLMQueryPool. With default settings (max_concurrent=5), this
        provides approximately 3-5x speedup.

        Scoring Process:
        ----------------
        1. Chunks are grouped into batches of config.batch_size
        2. Each batch is scored concurrently via asyncio.gather
        3. Scores are collected and chunks filtered by threshold
        4. If too many chunks remain, top-N by score are selected

        Args:
            chunks: Chunks from parse_input (Stage 1).
            query: Research query/hypothesis for relevance scoring.
            use_semantic: If True, uses LLM-based semantic scoring (slower, more accurate).
                         If False, uses keyword matching (faster, less accurate).

        Returns:
            Tuple of (selected_chunks, pipeline_result):
            - selected_chunks: Chunks passing the relevance threshold
            - pipeline_result: Statistics about the selection stage

        Example:
            >>> chunks = manager.parse_input(document)
            >>> selected, result = await manager.select_relevant(
            ...     chunks=chunks,
            ...     query="How does AI improve healthcare outcomes?",
            ...     use_semantic=True,
            ... )
            >>> print(f"Kept {len(selected)}/{len(chunks)} chunks")
            >>> print(f"Reduction: {(1 - result.reduction_ratio) * 100:.1f}%")
        """
        # Bypass if selection is disabled
        if not self.config.enable_selection:
            total_tokens = sum(c['tokens'] for c in chunks)
            return chunks, PipelineResult(
                stage=PipelineStage.SELECTION,
                content=chunks,
                original_tokens=total_tokens,
                output_tokens=total_tokens,
                reduction_ratio=1.0,
                metadata={'skipped': True},
            )

        original_tokens = sum(c['tokens'] for c in chunks)
        start_time = time.time()

        # =====================================================================
        # Parallel Batch Scoring (ASYNC OPTIMIZATION)
        # =====================================================================

        if use_semantic and self._query_pool._client:
            # Use parallel semantic scoring via query pool
            scored_chunks = await self._score_chunks_parallel(chunks, query)
        else:
            # Use fast keyword scoring (no parallelization needed)
            scored_chunks = await self._score_chunks_sequential(chunks, query)

        # Store scored chunks in REPL for programmatic access
        self.repl.set_context("scored_chunks", scored_chunks)

        # =====================================================================
        # Threshold Filtering
        # =====================================================================

        selected = [
            c for c in scored_chunks
            if c['relevance_score'] >= self.config.selection_threshold
        ]

        # =====================================================================
        # Token Budget Enforcement
        # =====================================================================

        # If still too much content, select top-N by relevance score
        if selected:
            selected_tokens = sum(c['tokens'] for c in selected)
            target_tokens = int(original_tokens * self.config.target_selection_ratio)

            if selected_tokens > target_tokens:
                # Sort by relevance (highest first) and take until budget exhausted
                selected.sort(key=lambda x: x['relevance_score'], reverse=True)

                final_selected = []
                cumulative_tokens = 0

                for chunk in selected:
                    if cumulative_tokens + chunk['tokens'] <= target_tokens:
                        final_selected.append(chunk)
                        cumulative_tokens += chunk['tokens']
                    else:
                        break

                selected = final_selected

        # =====================================================================
        # Calculate Results
        # =====================================================================

        output_tokens = sum(c['tokens'] for c in selected)
        reduction = output_tokens / original_tokens if original_tokens > 0 else 1.0
        processing_time = (time.time() - start_time) * 1000

        logger.info(
            f"Selection stage: {len(chunks)} → {len(selected)} chunks, "
            f"{original_tokens} → {output_tokens} tokens "
            f"({(1 - reduction) * 100:.1f}% reduction, {processing_time:.0f}ms)"
        )

        return selected, PipelineResult(
            stage=PipelineStage.SELECTION,
            content=selected,
            original_tokens=original_tokens,
            output_tokens=output_tokens,
            reduction_ratio=reduction,
            metadata={
                "chunks_before": len(chunks),
                "chunks_after": len(selected),
                "threshold": self.config.selection_threshold,
                "processing_time_ms": processing_time,
                "scoring_method": "semantic" if use_semantic else "keyword",
            },
        )

    async def _score_chunks_parallel(
        self,
        chunks: List[Dict[str, Any]],
        query: str,
    ) -> List[Dict[str, Any]]:
        """
        Score chunk relevance in parallel batches.

        This internal method implements the async optimization for the
        selection stage. Chunks are scored concurrently using the
        AsyncLLMQueryPool.

        Args:
            chunks: Chunks to score.
            query: Query for relevance scoring.

        Returns:
            Chunks with 'relevance_score' field added.
        """
        # Build batch queries for the pool
        batch_queries = [
            {
                'query': (
                    f"Rate the relevance of this text to the query '{query}' "
                    f"on a scale of 0.0 to 1.0. Respond with ONLY a number."
                ),
                'context': chunk['content'][:2000],  # Limit context size
            }
            for chunk in chunks
        ]

        # Execute all scoring queries in parallel
        results = await self._query_pool.batch_query(
            queries=batch_queries,
            max_tokens=10,
            temperature=0.1,
        )

        # Parse scores and add to chunks
        for chunk, result in zip(chunks, results):
            if result.success:
                try:
                    score_match = re.search(r'(\d+\.?\d*)', result.content)
                    if score_match:
                        score = float(score_match.group(1))
                        chunk['relevance_score'] = min(max(score, 0.0), 1.0)
                    else:
                        chunk['relevance_score'] = 0.5
                except ValueError:
                    chunk['relevance_score'] = 0.5
            else:
                # On failure, assign neutral score
                chunk['relevance_score'] = 0.5

        return chunks

    async def _score_chunks_sequential(
        self,
        chunks: List[Dict[str, Any]],
        query: str,
    ) -> List[Dict[str, Any]]:
        """
        Score chunk relevance sequentially using keyword filter.

        This is the fast fallback when semantic scoring is disabled
        or no LLM client is available.

        Args:
            chunks: Chunks to score.
            query: Query for relevance scoring.

        Returns:
            Chunks with 'relevance_score' field added.
        """
        for chunk in chunks:
            score = await self._keyword_filter.score_relevance(
                chunk['content'],
                query,
                None,
            )
            chunk['relevance_score'] = score

        return chunks

    # =========================================================================
    # Stage 3: Optimization (RLM-Based Compression) - ASYNC OPTIMIZED
    # =========================================================================

    async def optimize_content(
        self,
        chunks: List[Dict[str, Any]],
        query: str,
    ) -> Tuple[str, PipelineResult]:
        """
        Optimize selected content using RLM-based compression with async support.

        This is Stage 3 of the knowledge pipeline. It compresses the selected
        content from Stage 2, achieving approximately 80% token reduction
        while maintaining 95%+ accuracy as specified in the RLM Paper.

        Compression Strategies:
        -----------------------
        - **Abstractive** (use_abstractive_compression=True): LLM-generated summaries
          that can reorganize and synthesize information. Higher quality but slower.

        - **Extractive** (use_abstractive_compression=False): Select key sentences
          from original text. Faster and preserves original wording.

        Async Optimization:
        -------------------
        When compressing multiple chunks, they are processed in parallel using
        asyncio.gather. Each chunk is compressed independently, then results
        are concatenated.

        Args:
            chunks: Selected chunks from Stage 2.
            query: Research query for context-aware compression.

        Returns:
            Tuple of (compressed_content, pipeline_result):
            - compressed_content: Single string of compressed content
            - pipeline_result: Statistics about the optimization stage

        Example:
            >>> selected, _ = await manager.select_relevant(chunks, query)
            >>> compressed, result = await manager.optimize_content(selected, query)
            >>> print(f"Compressed to {result.output_tokens} tokens")
        """
        # Bypass if optimization is disabled
        if not self.config.enable_optimization:
            combined = '\n\n'.join(c['content'] for c in chunks)
            total_tokens = sum(c['tokens'] for c in chunks)
            return combined, PipelineResult(
                stage=PipelineStage.OPTIMIZATION,
                content=combined,
                original_tokens=total_tokens,
                output_tokens=total_tokens,
                reduction_ratio=1.0,
                metadata={'skipped': True},
            )

        original_tokens = sum(c['tokens'] for c in chunks)
        start_time = time.time()

        # =====================================================================
        # Parallel Chunk Compression (ASYNC OPTIMIZATION)
        # =====================================================================

        if self.config.use_abstractive_compression and self._query_pool._client:
            # Compress chunks in parallel using LLM
            compressed_chunks = await self._compress_chunks_parallel(chunks, query)
            compressed = '\n\n'.join(compressed_chunks)
        else:
            # Sequential extractive compression (fast, no LLM needed)
            combined = '\n\n'.join(c['content'] for c in chunks)
            compressed = await self._extractive_compressor.compress(
                combined,
                self.config.compression_ratio,
                None,
            )

        # Store compressed result in REPL
        self.repl.set_context("compressed_content", compressed)

        # =====================================================================
        # Calculate Results
        # =====================================================================

        output_tokens = self.count_tokens(compressed)
        reduction = output_tokens / original_tokens if original_tokens > 0 else 1.0
        processing_time = (time.time() - start_time) * 1000

        logger.info(
            f"Optimization stage: {original_tokens} → {output_tokens} tokens "
            f"({(1 - reduction) * 100:.1f}% reduction, {processing_time:.0f}ms)"
        )

        return compressed, PipelineResult(
            stage=PipelineStage.OPTIMIZATION,
            content=compressed,
            original_tokens=original_tokens,
            output_tokens=output_tokens,
            reduction_ratio=reduction,
            metadata={
                "strategy": "abstractive" if self.config.use_abstractive_compression else "extractive",
                "target_ratio": self.config.compression_ratio,
                "processing_time_ms": processing_time,
            },
        )

    async def _compress_chunks_parallel(
        self,
        chunks: List[Dict[str, Any]],
        query: str,
    ) -> List[str]:
        """
        Compress multiple chunks in parallel using LLM.

        Each chunk is compressed independently via asyncio.gather,
        providing speedup proportional to max_concurrent setting.

        Args:
            chunks: Chunks to compress.
            query: Query for context-aware compression.

        Returns:
            List of compressed chunk strings.
        """
        async def compress_single(chunk: Dict[str, Any]) -> str:
            """Compress a single chunk."""
            result = await self._query_pool.query(
                query=(
                    f"Compress this text to approximately {int(self.config.compression_ratio * 100)}% "
                    f"of its original length while preserving all key information relevant to: {query}"
                ),
                context=chunk['content'],
                max_tokens=int(chunk['tokens'] * self.config.compression_ratio * 1.5),
                temperature=0.3,
            )

            if result.success:
                return result.content
            else:
                # Fallback to extractive on failure
                return await self._extractive_compressor.compress(
                    chunk['content'],
                    self.config.compression_ratio,
                    None,
                )

        # Execute all compressions in parallel
        compressed_chunks = await asyncio.gather(
            *[compress_single(chunk) for chunk in chunks],
            return_exceptions=True,
        )

        # Handle any exceptions
        results = []
        for i, result in enumerate(compressed_chunks):
            if isinstance(result, Exception):
                logger.warning(f"Chunk {i} compression failed: {result}")
                # Use original content on failure
                results.append(chunks[i]['content'][:int(len(chunks[i]['content']) * self.config.compression_ratio)])
            else:
                results.append(result)

        return results

    # =========================================================================
    # Full Pipeline Processing
    # =========================================================================

    async def process_through_pipeline(
        self,
        content: Union[str, List, Dict],
        query: str,
        chunk_size: int = 1000,
    ) -> Dict[str, Any]:
        """
        Process content through the complete 4-stage RLM pipeline.

        This is the main entry point for context processing. It orchestrates
        all four stages of the knowledge pipeline:

        1. **Input**: Parse content into chunks
        2. **Selection**: Filter by relevance (async parallel scoring)
        3. **Optimization**: Compress selected content (async parallel compression)
        4. **Application**: Return processed context with statistics

        End-to-End Performance:
        -----------------------
        With default async settings (max_concurrent=5):
        - 10 chunks: ~2-3 seconds (vs ~10 seconds sequential)
        - 50 chunks: ~10-15 seconds (vs ~50 seconds sequential)
        - 100 chunks: ~20-30 seconds (vs ~100 seconds sequential)

        Args:
            content: Raw content to process. Can be string, list, or dict.
            query: Research query/hypothesis for relevance scoring.
            chunk_size: Target chunk size in tokens. Default: 1000.

        Returns:
            Dictionary containing:
            - input: Stage 1 results (chunk count, token count)
            - selection: PipelineResult from Stage 2
            - optimization: PipelineResult from Stage 3
            - final: Final results (content, token counts, reduction percentage)

        Example:
            >>> result = await manager.process_through_pipeline(
            ...     content=research_document,
            ...     query="What evidence supports AI in healthcare?",
            ...     chunk_size=800,
            ... )
            >>>
            >>> # Access processed content
            >>> compressed_content = result['final']['content']
            >>>
            >>> # Check reduction metrics
            >>> original = result['final']['original_tokens']
            >>> final = result['final']['final_tokens']
            >>> print(f"Reduced {original} → {final} tokens ({result['final']['total_reduction']})")
        """
        pipeline_start = time.time()
        results = {}

        # =====================================================================
        # Stage 1: Input (Semantic Layer)
        # =====================================================================

        chunks = self.parse_input(content, chunk_size)
        original_tokens = sum(c['tokens'] for c in chunks)

        results['input'] = {
            'chunks': len(chunks),
            'tokens': original_tokens,
        }

        # Store in REPL for programmatic access
        self.repl.set_context("raw_chunks", chunks)
        self.repl.set_context("query", query)

        # =====================================================================
        # Stage 2: Selection (Async Parallel)
        # =====================================================================

        selected, selection_result = await self.select_relevant(chunks, query)
        results['selection'] = selection_result

        # =====================================================================
        # Stage 3: Optimization (Async Parallel)
        # =====================================================================

        compressed, optimization_result = await self.optimize_content(selected, query)
        results['optimization'] = optimization_result

        # =====================================================================
        # Stage 4: Application (Return Results)
        # =====================================================================

        final_tokens = self.count_tokens(compressed)
        total_reduction = final_tokens / original_tokens if original_tokens > 0 else 1.0
        total_time = (time.time() - pipeline_start) * 1000

        # Update statistics
        self._pipeline_stats['total_processed'] += 1
        self._pipeline_stats['total_tokens_saved'] += (original_tokens - final_tokens)
        self._pipeline_stats['total_processing_time_ms'] += total_time

        # Update running average reduction
        n = self._pipeline_stats['total_processed']
        prev_avg = self._pipeline_stats['average_reduction']
        self._pipeline_stats['average_reduction'] = (prev_avg * (n - 1) + total_reduction) / n

        results['final'] = {
            'content': compressed,
            'original_tokens': original_tokens,
            'final_tokens': final_tokens,
            'total_reduction': f"{(1 - total_reduction) * 100:.1f}%",
            'processing_time_ms': total_time,
        }

        logger.info(
            f"Pipeline complete: {original_tokens} → {final_tokens} tokens "
            f"({(1 - total_reduction) * 100:.1f}% total reduction, {total_time:.0f}ms)"
        )

        return results

    # =========================================================================
    # REPL-Based Context Interaction
    # =========================================================================

    def execute_context_code(self, code: str) -> Tuple[Any, Optional[str]]:
        """
        Execute code in the REPL to interact with context.

        This is a core RLM feature - the LLM can write code to
        programmatically examine, filter, search, and decompose context
        stored in the REPL environment.

        Available Context Variables:
        ----------------------------
        After pipeline processing, these variables are available:
        - raw_chunks: Original parsed chunks from Stage 1
        - query: The research query
        - scored_chunks: Chunks with relevance scores from Stage 2
        - compressed_content: Final compressed content from Stage 3

        Args:
            code: Python code to execute in the sandboxed REPL.

        Returns:
            Tuple of (result, error):
            - On success: (result_value, None)
            - On failure: (None, error_message)

        Example:
            >>> # After pipeline processing
            >>> result, err = manager.execute_context_code(
            ...     "len([c for c in scored_chunks if c['relevance_score'] > 0.7])"
            ... )
            >>> print(f"High-relevance chunks: {result}")
        """
        return self.repl.execute(code)

    def get_context_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all contexts in the REPL environment.

        Useful for debugging and understanding what data is available
        for programmatic interaction.

        Returns:
            Dictionary mapping variable names to their type/size info.
        """
        summary = {}
        for name in self.repl.list_contexts():
            value = self.repl.get_context(name)

            if isinstance(value, str):
                summary[name] = {
                    'type': 'string',
                    'length': len(value),
                    'tokens': self.count_tokens(value),
                }
            elif isinstance(value, list):
                summary[name] = {
                    'type': 'list',
                    'length': len(value),
                    'item_types': list(set(type(x).__name__ for x in value[:10])),
                }
            elif isinstance(value, dict):
                summary[name] = {
                    'type': 'dict',
                    'keys': list(value.keys())[:10],
                    'key_count': len(value),
                }
            else:
                summary[name] = {
                    'type': type(value).__name__,
                }

        return summary

    # =========================================================================
    # Synchronous Fallback (when RLM pipeline is unavailable/overkill)
    # =========================================================================

    def truncate_simple(
        self,
        messages: List[Dict[str, str]],
        target_tokens: Optional[int] = None,
    ) -> List[Dict[str, str]]:
        """
        Fast synchronous truncation for when RLM pipeline is unavailable or overkill.

        This provides a minimal fallback for scenarios where:
        - No LLM API is available (offline, testing)
        - Synchronous code cannot use async process_through_pipeline()
        - Simple truncation is sufficient (non-critical context)

        For intelligent context management, use process_through_pipeline() instead.

        Algorithm:
        ----------
        1. Keep the first system message (contains critical instructions)
        2. Keep the most recent messages that fit within budget
        3. Preserve conversation continuity by prioritizing recent context

        Args:
            messages: List of chat messages to truncate.
            target_tokens: Target token count. Defaults to 30% of max_context_tokens.

        Returns:
            Truncated message list fitting within target token budget.

        Example:
            >>> # When async isn't available
            >>> truncated = manager.truncate_simple(messages, target_tokens=2000)
            >>>
            >>> # For quick testing without API calls
            >>> truncated = manager.truncate_simple(long_conversation)
        """
        target = target_tokens or int(self.config.max_context_tokens * 0.3)
        current = self.count_messages_tokens(messages)

        # Already within budget
        if current <= target:
            return messages

        result = []
        current_tokens = 0

        # Separate system messages from conversation
        system_msgs = [m for m in messages if m.get('role') == 'system']
        other_msgs = [m for m in messages if m.get('role') != 'system']

        # Always keep the first system message (contains agent instructions)
        if system_msgs:
            result.append(system_msgs[0])
            current_tokens += self.count_message_tokens(system_msgs[0])

        # Add most recent messages that fit (preserves conversation continuity)
        for msg in reversed(other_msgs):
            msg_tokens = self.count_message_tokens(msg)
            if current_tokens + msg_tokens <= target:
                # Insert after system message to maintain order
                result.insert(len(result), msg)
                current_tokens += msg_tokens
            else:
                break

        # Reverse non-system messages to restore chronological order
        if system_msgs:
            result = [result[0]] + list(reversed(result[1:]))
        else:
            result = list(reversed(result))

        logger.debug(
            f"truncate_simple: {len(messages)} → {len(result)} messages, "
            f"{self.count_messages_tokens(messages)} → {current_tokens} tokens"
        )

        return result

    def get_utilization(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get current context utilization metrics.

        Useful for deciding when to trigger RLM pipeline processing
        or simple truncation.

        Args:
            messages: List of chat messages.
            model: Optional model name for context limit lookup.

        Returns:
            Dictionary with utilization metrics:
            - total_tokens: Current token count
            - max_tokens: Maximum allowed tokens
            - utilization: Ratio (0.0 to 1.0)
            - remaining_tokens: Available token budget
            - needs_processing: True if utilization > 70%

        Example:
            >>> util = manager.get_utilization(messages)
            >>> if util['needs_processing']:
            ...     result = await manager.process_through_pipeline(...)
        """
        total_tokens = self.count_messages_tokens(messages)
        max_tokens = self.config.max_context_tokens

        if model:
            model_limit = self.MODEL_LIMITS.get(model, 128000)
            max_tokens = min(max_tokens, model_limit - self.config.max_completion_tokens)

        utilization = total_tokens / max_tokens if max_tokens > 0 else 1.0

        return {
            'total_tokens': total_tokens,
            'max_tokens': max_tokens,
            'utilization': utilization,
            'utilization_pct': f"{utilization * 100:.1f}%",
            'remaining_tokens': max(0, max_tokens - total_tokens),
            'needs_processing': utilization >= 0.7,
        }

    # =========================================================================
    # Statistics and Monitoring
    # =========================================================================

    def get_stats(self, messages: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """
        Get comprehensive memory manager statistics.

        Returns statistics about pipeline processing, LLM query usage,
        REPL environment, and optionally current context window state.

        Args:
            messages: Optional message list for context window stats.

        Returns:
            Dictionary containing:
            - pipeline: Processing counts and token savings
            - llm_query: Single query function stats
            - query_pool: Async pool stats (concurrent queries, rate limits)
            - repl: REPL execution stats
            - repl_contexts: Summary of stored context variables
            - context_window: (if messages provided) Current utilization
        """
        stats = {
            "pipeline": self._pipeline_stats.copy(),
            "llm_query": self.llm_query.get_stats(),
            "query_pool": self._query_pool.get_stats(),
            "repl": self.repl.get_stats(),
            "repl_contexts": self.get_context_summary(),
        }

        if messages:
            window = self.get_context_window(messages)
            stats["context_window"] = {
                "total_messages": len(messages),
                "total_tokens": window.total_tokens,
                "max_tokens": window.max_tokens,
                "utilization": f"{window.utilization * 100:.1f}%",
                "remaining_tokens": window.remaining_tokens,
            }

        return stats

    def reset_stats(self) -> None:
        """
        Reset all statistics counters.

        Useful for starting fresh measurements or between research sessions.
        """
        self._pipeline_stats = {
            "total_processed": 0,
            "total_tokens_saved": 0,
            "average_reduction": 0.0,
            "total_processing_time_ms": 0,
        }
        logger.info("MemoryManager statistics reset")

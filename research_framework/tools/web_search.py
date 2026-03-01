"""
Web search tool for the Multi-Agent Research Framework.

This module provides web search capabilities using multiple backends:
1. Tavily API (recommended for research - AI-optimized results)
2. Brave Search API (privacy-focused, good quality)
3. Serper API (Google search results)
4. Bing Web Search API (Microsoft)
5. DuckDuckGo fallback (free, no API key required, limited results)

The tool returns structured search results with titles, URLs, and snippets
that agents can use to gather information for research tasks.
"""

import asyncio
import httpx
from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Any, Optional

from tools.base_tool import BaseTool, ToolResult, ToolStatus
from config.settings import settings
from utils.logger import get_logger


logger = get_logger(__name__)


class SearchBackend(Enum):
    """Supported search backends."""
    TAVILY = "tavily"
    BRAVE = "brave"
    SERPER = "serper"
    BING = "bing"
    DUCKDUCKGO = "duckduckgo"
    AUTO = "auto"


@dataclass
class SearchResult:
    """
    A single search result.

    Attributes:
        title: Page title
        url: Page URL
        snippet: Text snippet/description
        score: Relevance score (0-1, if available)
    """
    title: str
    url: str
    snippet: str
    score: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "title": self.title,
            "url": self.url,
            "snippet": self.snippet,
            "score": self.score,
        }


class WebSearchTool(BaseTool):
    """
    Web search tool supporting multiple search backends.

    Backends (in order of recommendation for research):
    1. Tavily - AI-optimized, clean results, source scoring
    2. Brave - Privacy-focused, good quality, free tier
    3. Serper - Google results via API
    4. Bing - Microsoft search, reliable
    5. DuckDuckGo - Free fallback, instant answers only

    Example:
        >>> tool = WebSearchTool()  # Auto-selects based on available keys
        >>> tool = WebSearchTool(backend="brave")  # Force specific backend
        >>> result = await tool.execute(
        ...     query="AI applications in healthcare 2024",
        ...     max_results=5
        ... )
        >>> for item in result.data:
        ...     print(f"{item['title']}: {item['url']}")

    Configuration:
        Set API keys in .env:
        ```
        SEARCH_BACKEND=auto  # or: tavily, brave, serper, bing, duckduckgo
        TAVILY_API_KEY=tvly-your-key-here
        BRAVE_API_KEY=your-brave-key-here
        SERPER_API_KEY=your-serper-key-here
        BING_API_KEY=your-bing-key-here
        ```
    """

    name = "web_search"
    description = "Search the web for information on a given query"

    # API endpoints
    TAVILY_API_URL = "https://api.tavily.com/search"
    BRAVE_API_URL = "https://api.search.brave.com/res/v1/web/search"
    SERPER_API_URL = "https://google.serper.dev/search"
    BING_API_URL = "https://api.bing.microsoft.com/v7.0/search"
    DDG_API_URL = "https://api.duckduckgo.com/"

    def __init__(
        self,
        backend: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        """
        Initialize the web search tool.

        Args:
            backend: Search backend to use. Options: 'tavily', 'brave', 'serper',
                     'bing', 'duckduckgo', 'auto'. If None, uses SEARCH_BACKEND from settings.
            api_key: API key for the selected backend. If None, uses key from settings.
        """
        super().__init__()

        # Determine backend
        requested_backend = backend or settings.SEARCH_BACKEND
        self._backend, self._api_key = self._resolve_backend(requested_backend, api_key)

        logger.info(f"WebSearchTool initialized with backend: {self._backend}")

    def _resolve_backend(
        self,
        requested: str,
        provided_key: Optional[str]
    ) -> tuple[str, Optional[str]]:
        """
        Resolve which backend to use based on request and available keys.

        Args:
            requested: Requested backend name or 'auto'
            provided_key: Explicitly provided API key

        Returns:
            Tuple of (backend_name, api_key)
        """
        # Backend priority order for auto-selection
        backend_priority = [
            ("tavily", settings.TAVILY_API_KEY),
            ("brave", settings.BRAVE_API_KEY),
            ("serper", settings.SERPER_API_KEY),
            ("bing", settings.BING_API_KEY),
        ]

        def is_valid_key(key: str) -> bool:
            """Check if API key is valid (not empty/placeholder)."""
            if not key or len(key) < 10:
                return False
            # Check for common placeholder patterns
            placeholder_patterns = [
                "your-",           # your-api-key-here
                "xxx",             # ends with xxx
                "placeholder",
                "example",
                "test-key",
            ]
            key_lower = key.lower()
            return not any(p in key_lower for p in placeholder_patterns)

        # If specific backend requested (not auto)
        if requested.lower() != "auto":
            backend = requested.lower()

            # If key provided, use it
            if provided_key:
                return (backend, provided_key)

            # Otherwise get from settings
            key_map = {
                "tavily": settings.TAVILY_API_KEY,
                "brave": settings.BRAVE_API_KEY,
                "serper": settings.SERPER_API_KEY,
                "bing": settings.BING_API_KEY,
                "duckduckgo": None,
            }

            key = key_map.get(backend)
            if backend == "duckduckgo" or is_valid_key(key or ""):
                return (backend, key)

            logger.warning(f"Requested backend '{backend}' has no valid API key, falling back to auto")

        # Auto-select based on available keys
        for backend, key in backend_priority:
            if is_valid_key(key):
                return (backend, key)

        # Fallback to DuckDuckGo
        return ("duckduckgo", None)

    @property
    def backend(self) -> str:
        """Get current backend name."""
        return self._backend

    @classmethod
    def get_available_backends(cls) -> Dict[str, bool]:
        """
        Check which backends have valid API keys configured.

        Returns:
            Dict mapping backend names to availability status
        """
        def is_valid(key: str) -> bool:
            return bool(key) and not key.startswith("your-")

        return {
            "tavily": is_valid(settings.TAVILY_API_KEY),
            "brave": is_valid(settings.BRAVE_API_KEY),
            "serper": is_valid(settings.SERPER_API_KEY),
            "bing": is_valid(settings.BING_API_KEY),
            "duckduckgo": True,  # Always available
        }

    async def execute(
        self,
        query: str,
        max_results: int = 5,
        search_depth: str = "basic",
        include_domains: Optional[List[str]] = None,
        exclude_domains: Optional[List[str]] = None,
        **kwargs,
    ) -> ToolResult:
        """
        Execute a web search.

        Args:
            query: Search query string
            max_results: Maximum number of results to return (default: 5)
            search_depth: "basic" or "advanced" (Tavily only, default: "basic")
            include_domains: Only include results from these domains (some backends)
            exclude_domains: Exclude results from these domains (some backends)

        Returns:
            ToolResult with list of SearchResult dictionaries
        """
        if not query or not query.strip():
            return ToolResult(
                status=ToolStatus.ERROR,
                error="Query cannot be empty",
            )

        query = query.strip()
        logger.info(f"Executing web search: '{query[:50]}...' (backend={self._backend})")

        # Route to appropriate backend
        backend_methods = {
            "tavily": self._search_tavily,
            "brave": self._search_brave,
            "serper": self._search_serper,
            "bing": self._search_bing,
            "duckduckgo": self._search_duckduckgo,
        }

        method = backend_methods.get(self._backend, self._search_duckduckgo)

        try:
            return await method(
                query=query,
                max_results=max_results,
                search_depth=search_depth,
                include_domains=include_domains,
                exclude_domains=exclude_domains,
            )
        except Exception as e:
            logger.error(f"Search failed with {self._backend}: {e}")
            # Try fallback to DuckDuckGo if not already using it
            if self._backend != "duckduckgo":
                logger.info("Falling back to DuckDuckGo")
                return await self._search_duckduckgo(query, max_results)
            return ToolResult(
                status=ToolStatus.ERROR,
                error=f"Search failed: {str(e)}",
            )

    async def _search_tavily(
        self,
        query: str,
        max_results: int,
        search_depth: str = "basic",
        include_domains: Optional[List[str]] = None,
        exclude_domains: Optional[List[str]] = None,
        **kwargs,
    ) -> ToolResult:
        """Search using Tavily API (AI-optimized results)."""
        payload = {
            "api_key": self._api_key,
            "query": query,
            "max_results": max_results,
            "search_depth": search_depth,
            "include_answer": True,
        }

        if include_domains:
            payload["include_domains"] = include_domains
        if exclude_domains:
            payload["exclude_domains"] = exclude_domains

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(self.TAVILY_API_URL, json=payload)

            if response.status_code == 429:
                return ToolResult(
                    status=ToolStatus.RATE_LIMITED,
                    error="Rate limit reached. Please try again later.",
                )
            response.raise_for_status()
            data = response.json()

        results = []
        for item in data.get("results", []):
            results.append(SearchResult(
                title=item.get("title", ""),
                url=item.get("url", ""),
                snippet=item.get("content", ""),
                score=item.get("score"),
            ).to_dict())

        answer = data.get("answer")
        logger.info(f"Tavily search returned {len(results)} results")

        return ToolResult(
            status=ToolStatus.SUCCESS if results else ToolStatus.NO_RESULTS,
            data=results,
            metadata={
                "backend": "tavily",
                "query": query,
                "answer": answer,
                "result_count": len(results),
            },
        )

    async def _search_brave(
        self,
        query: str,
        max_results: int,
        **kwargs,
    ) -> ToolResult:
        """Search using Brave Search API."""
        headers = {
            "Accept": "application/json",
            "X-Subscription-Token": self._api_key,
        }
        params = {
            "q": query,
            "count": max_results,
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                self.BRAVE_API_URL,
                headers=headers,
                params=params,
            )

            if response.status_code == 429:
                return ToolResult(
                    status=ToolStatus.RATE_LIMITED,
                    error="Rate limit reached. Please try again later.",
                )
            response.raise_for_status()
            data = response.json()

        results = []
        for item in data.get("web", {}).get("results", []):
            results.append(SearchResult(
                title=item.get("title", ""),
                url=item.get("url", ""),
                snippet=item.get("description", ""),
            ).to_dict())

        logger.info(f"Brave search returned {len(results)} results")

        return ToolResult(
            status=ToolStatus.SUCCESS if results else ToolStatus.NO_RESULTS,
            data=results,
            metadata={
                "backend": "brave",
                "query": query,
                "result_count": len(results),
            },
        )

    async def _search_serper(
        self,
        query: str,
        max_results: int,
        **kwargs,
    ) -> ToolResult:
        """Search using Serper API (Google results)."""
        headers = {
            "X-API-KEY": self._api_key,
            "Content-Type": "application/json",
        }
        payload = {
            "q": query,
            "num": max_results,
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                self.SERPER_API_URL,
                headers=headers,
                json=payload,
            )

            if response.status_code == 429:
                return ToolResult(
                    status=ToolStatus.RATE_LIMITED,
                    error="Rate limit reached. Please try again later.",
                )
            response.raise_for_status()
            data = response.json()

        results = []
        for item in data.get("organic", []):
            results.append(SearchResult(
                title=item.get("title", ""),
                url=item.get("link", ""),
                snippet=item.get("snippet", ""),
                score=item.get("position"),
            ).to_dict())

        # Include answer box if available
        answer = None
        if "answerBox" in data:
            answer = data["answerBox"].get("answer") or data["answerBox"].get("snippet")

        logger.info(f"Serper search returned {len(results)} results")

        return ToolResult(
            status=ToolStatus.SUCCESS if results else ToolStatus.NO_RESULTS,
            data=results,
            metadata={
                "backend": "serper",
                "query": query,
                "answer": answer,
                "result_count": len(results),
            },
        )

    async def _search_bing(
        self,
        query: str,
        max_results: int,
        **kwargs,
    ) -> ToolResult:
        """Search using Bing Web Search API."""
        headers = {
            "Ocp-Apim-Subscription-Key": self._api_key,
        }
        params = {
            "q": query,
            "count": max_results,
            "responseFilter": "Webpages",
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                self.BING_API_URL,
                headers=headers,
                params=params,
            )

            if response.status_code == 429:
                return ToolResult(
                    status=ToolStatus.RATE_LIMITED,
                    error="Rate limit reached. Please try again later.",
                )
            response.raise_for_status()
            data = response.json()

        results = []
        for item in data.get("webPages", {}).get("value", []):
            results.append(SearchResult(
                title=item.get("name", ""),
                url=item.get("url", ""),
                snippet=item.get("snippet", ""),
            ).to_dict())

        logger.info(f"Bing search returned {len(results)} results")

        return ToolResult(
            status=ToolStatus.SUCCESS if results else ToolStatus.NO_RESULTS,
            data=results,
            metadata={
                "backend": "bing",
                "query": query,
                "result_count": len(results),
            },
        )

    async def _search_duckduckgo(
        self,
        query: str,
        max_results: int,
        **kwargs,
    ) -> ToolResult:
        """
        Search using DuckDuckGo Instant Answer API.

        Note: This is a free fallback that doesn't require an API key.
        Results are limited to instant answers only.
        """
        params = {
            "q": query,
            "format": "json",
            "no_html": 1,
            "skip_disambig": 1,
        }

        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.get(self.DDG_API_URL, params=params)
            response.raise_for_status()
            data = response.json()

        results = []

        # Abstract (main answer)
        if data.get("Abstract"):
            results.append(SearchResult(
                title=data.get("Heading", "Summary"),
                url=data.get("AbstractURL", ""),
                snippet=data.get("Abstract", ""),
            ).to_dict())

        # Related topics
        for topic in data.get("RelatedTopics", [])[:max_results - len(results)]:
            if isinstance(topic, dict) and "Text" in topic:
                results.append(SearchResult(
                    title=topic.get("Text", "")[:100],
                    url=topic.get("FirstURL", ""),
                    snippet=topic.get("Text", ""),
                ).to_dict())

        if not results:
            logger.info("DuckDuckGo returned no instant answers")
            return ToolResult(
                status=ToolStatus.NO_RESULTS,
                data=[],
                metadata={
                    "backend": "duckduckgo",
                    "query": query,
                    "note": "No instant answers available. Consider using a different backend with an API key.",
                },
            )

        logger.info(f"DuckDuckGo search returned {len(results)} results")

        return ToolResult(
            status=ToolStatus.SUCCESS,
            data=results,
            metadata={
                "backend": "duckduckgo",
                "query": query,
                "result_count": len(results),
            },
        )

    def get_schema(self) -> Dict[str, Any]:
        """Get OpenAI function calling schema."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results (default: 5)",
                        "default": 5,
                    },
                    "search_depth": {
                        "type": "string",
                        "enum": ["basic", "advanced"],
                        "description": "Search depth (Tavily only)",
                        "default": "basic",
                    },
                },
                "required": ["query"],
            },
        }


# Convenience function for quick searches
async def web_search(
    query: str,
    max_results: int = 5,
    backend: Optional[str] = None,
    **kwargs,
) -> List[Dict[str, Any]]:
    """
    Quick web search function.

    Args:
        query: Search query
        max_results: Maximum results to return
        backend: Optional backend override ('tavily', 'brave', 'serper', 'bing', 'duckduckgo')

    Returns:
        List of result dictionaries

    Example:
        >>> results = await web_search("latest AI research 2024")
        >>> for r in results:
        ...     print(r['title'])

        >>> # Use specific backend
        >>> results = await web_search("AI news", backend="brave")
    """
    tool = WebSearchTool(backend=backend)
    result = await tool.execute(query=query, max_results=max_results, **kwargs)
    return result.data if result.success else []

"""
Tools package for the Multi-Agent Research Framework.

This package provides various tools that agents can use during research tasks.

Available Tools:
----------------
- WebSearchTool: Search the web using Tavily or DuckDuckGo
- (More tools can be added here)

Usage:
------
    >>> from tools import WebSearchTool, web_search
    >>>
    >>> # Using the tool class
    >>> tool = WebSearchTool()
    >>> result = await tool.execute(query="AI in healthcare")
    >>>
    >>> # Using the convenience function
    >>> results = await web_search("AI in healthcare")
"""

from tools.base_tool import (
    BaseTool,
    ToolResult,
    ToolStatus,
)

from tools.web_search import (
    WebSearchTool,
    SearchResult,
    SearchBackend,
    web_search,
)

__all__ = [
    # Base
    "BaseTool",
    "ToolResult",
    "ToolStatus",
    # Web Search
    "WebSearchTool",
    "SearchResult",
    "SearchBackend",
    "web_search",
]

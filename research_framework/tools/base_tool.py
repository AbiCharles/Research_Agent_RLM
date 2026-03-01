"""
Base tool interface for the Multi-Agent Research Framework.

This module provides the abstract base class and common utilities for
implementing tools that agents can use during research.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from enum import Enum

from utils.logger import get_logger


logger = get_logger(__name__)


class ToolStatus(Enum):
    """Status of a tool execution."""
    SUCCESS = "success"
    ERROR = "error"
    RATE_LIMITED = "rate_limited"
    NO_RESULTS = "no_results"


@dataclass
class ToolResult:
    """
    Result from a tool execution.

    Attributes:
        status: Execution status (success, error, etc.)
        data: The result data (type depends on tool)
        error: Error message if status is ERROR
        metadata: Additional information about the execution
    """
    status: ToolStatus
    data: Any = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        """Check if execution was successful."""
        return self.status == ToolStatus.SUCCESS

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "status": self.status.value,
            "data": self.data,
            "error": self.error,
            "metadata": self.metadata,
        }


class BaseTool(ABC):
    """
    Abstract base class for all tools.

    Tools provide specific capabilities to agents, such as web search,
    document retrieval, calculations, etc.

    Example Implementation:
        >>> class CalculatorTool(BaseTool):
        ...     name = "calculator"
        ...     description = "Performs mathematical calculations"
        ...
        ...     async def execute(self, expression: str, **kwargs) -> ToolResult:
        ...         try:
        ...             result = eval(expression)  # Simplified - use safe eval in production
        ...             return ToolResult(status=ToolStatus.SUCCESS, data=result)
        ...         except Exception as e:
        ...             return ToolResult(status=ToolStatus.ERROR, error=str(e))
    """

    # Tool identification - override in subclasses
    name: str = "base_tool"
    description: str = "Base tool interface"

    def __init__(self):
        """Initialize the tool."""
        self._call_count = 0
        self._error_count = 0
        logger.info(f"Tool '{self.name}' initialized")

    @abstractmethod
    async def execute(self, **kwargs) -> ToolResult:
        """
        Execute the tool with given parameters.

        Args:
            **kwargs: Tool-specific parameters

        Returns:
            ToolResult with execution outcome
        """
        pass

    async def __call__(self, **kwargs) -> ToolResult:
        """
        Call the tool (wrapper around execute with tracking).

        Args:
            **kwargs: Tool-specific parameters

        Returns:
            ToolResult with execution outcome
        """
        self._call_count += 1
        try:
            result = await self.execute(**kwargs)
            if not result.success:
                self._error_count += 1
            return result
        except Exception as e:
            self._error_count += 1
            logger.error(f"Tool '{self.name}' execution error: {e}")
            return ToolResult(
                status=ToolStatus.ERROR,
                error=str(e),
            )

    def get_stats(self) -> Dict[str, Any]:
        """Get tool usage statistics."""
        return {
            "name": self.name,
            "call_count": self._call_count,
            "error_count": self._error_count,
            "success_rate": (
                (self._call_count - self._error_count) / self._call_count
                if self._call_count > 0 else 0
            ),
        }

    def get_schema(self) -> Dict[str, Any]:
        """
        Get the tool schema for LLM function calling.

        Override this method to provide custom parameter schemas.

        Returns:
            OpenAI-compatible function schema
        """
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        }

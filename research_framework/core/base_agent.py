"""
Base agent class for the Multi-Agent Research Framework.

This module provides the abstract base class that all agents inherit from.
It implements the Template Method pattern, defining the research workflow
skeleton while allowing subclasses to implement specific steps.
"""

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Dict, Any, Optional
from uuid import uuid4

from core.openai_client import OpenAIClient, get_client, CompletionResponse
from config.settings import settings
from utils.logger import get_logger


logger = get_logger(__name__)


class AgentStatus(Enum):
    """Agent execution status."""

    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AgentRole(Enum):
    """Predefined agent roles."""

    LEAD_RESEARCHER = "lead_researcher"
    RESEARCH_AGENT = "research_agent"
    CITATION_AGENT = "citation_agent"
    CUSTOM = "custom"


@dataclass
class AgentConfig:
    """Configuration for an agent instance."""

    name: str
    role: AgentRole = AgentRole.CUSTOM
    focus: str = ""
    model: str = field(default_factory=lambda: settings.DEFAULT_MODEL)
    temperature: float = field(default_factory=lambda: settings.TEMPERATURE)
    max_tokens: int = field(default_factory=lambda: settings.MAX_TOKENS)
    tools: List[str] = field(default_factory=lambda: ["web_search"])
    system_prompt: Optional[str] = None

    def __post_init__(self):
        if not self.name:
            raise ValueError("Agent name is required")


@dataclass
class AgentResult:
    """Result from an agent's research execution."""

    agent_id: str
    agent_name: str
    status: AgentStatus
    content: str = ""
    findings: List[Dict[str, Any]] = field(default_factory=list)
    sources: List[Dict[str, str]] = field(default_factory=list)
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    token_usage: Dict[str, int] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration_seconds(self) -> Optional[float]:
        """Calculate execution duration."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "status": self.status.value,
            "content": self.content,
            "findings": self.findings,
            "sources": self.sources,
            "error": self.error,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_seconds": self.duration_seconds,
            "token_usage": self.token_usage,
            "metadata": self.metadata,
        }


class BaseAgent(ABC):
    """
    Abstract base class for all research agents.

    This class implements the Template Method pattern, defining the overall
    research workflow while allowing subclasses to implement specific steps.

    The research workflow consists of:
    1. Initialize - Set up agent state
    2. Plan - Create research strategy (optional)
    3. Execute - Perform the main research task
    4. Synthesize - Combine and format findings
    5. Cleanup - Release resources

    Example:
        >>> class MyAgent(BaseAgent):
        ...     async def _execute(self, context):
        ...         # Implementation here
        ...         return {"findings": [...]}
        ...
        >>> agent = MyAgent(config)
        >>> result = await agent.run(hypothesis="...")
    """

    def __init__(
        self,
        config: AgentConfig,
        client: Optional[OpenAIClient] = None,
    ):
        """
        Initialize the base agent.

        Args:
            config: Agent configuration
            client: OpenAI client (uses singleton if not provided)
        """
        self.config = config
        self.client = client or get_client()

        # Agent identity
        self.agent_id = str(uuid4())[:8]
        self.name = config.name
        self.role = config.role

        # State tracking
        self.status = AgentStatus.IDLE
        self.conversation_history: List[Dict[str, str]] = []
        self.findings: List[Dict[str, Any]] = []
        self.sources: List[Dict[str, str]] = []

        # Execution tracking
        self._started_at: Optional[datetime] = None
        self._completed_at: Optional[datetime] = None
        self._total_tokens = 0

        # Build system prompt
        self._system_prompt = config.system_prompt or self._build_system_prompt()

        logger.info(
            f"Agent initialized: id={self.agent_id}, "
            f"name={self.name}, role={self.role.value}"
        )

    def _build_system_prompt(self) -> str:
        """
        Build the default system prompt for this agent.

        Returns:
            System prompt string
        """
        base_prompt = f"""You are {self.name}, a specialized research agent.

Your role: {self.role.value}
Your focus area: {self.config.focus or 'General research'}

Guidelines:
1. Use chain-of-thought reasoning to analyze information
2. Cite sources for all factual claims
3. Be thorough but concise in your findings
4. Acknowledge uncertainty when appropriate
5. Structure your response clearly with key findings

When researching, follow this process:
- First, understand the hypothesis or question
- Break down the research into logical steps
- Gather relevant information from available tools
- Analyze and synthesize findings
- Present conclusions with supporting evidence
"""
        return base_prompt

    async def run(
        self,
        hypothesis: str,
        research_questions: Optional[List[str]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> AgentResult:
        """
        Execute the full research workflow.

        This is the template method that orchestrates the research process.

        Args:
            hypothesis: The main research hypothesis or question
            research_questions: Optional specific questions to investigate
            context: Optional additional context for research

        Returns:
            AgentResult with findings and metadata
        """
        self.status = AgentStatus.RUNNING
        self._started_at = datetime.now()
        context = context or {}

        logger.info(f"Agent {self.name} starting research: {hypothesis[:100]}...")

        try:
            # Step 1: Initialize
            await self._initialize(hypothesis, research_questions, context)

            # Step 2: Plan (optional - subclasses can override)
            plan = await self._plan(hypothesis, research_questions, context)
            if plan:
                context["plan"] = plan

            # Step 3: Execute main research
            execution_result = await self._execute(hypothesis, research_questions, context)

            # Step 4: Synthesize findings
            synthesis = await self._synthesize(execution_result, context)

            # Step 5: Cleanup
            await self._cleanup()

            self.status = AgentStatus.COMPLETED
            self._completed_at = datetime.now()

            result = AgentResult(
                agent_id=self.agent_id,
                agent_name=self.name,
                status=self.status,
                content=synthesis.get("content", ""),
                findings=self.findings,
                sources=self.sources,
                started_at=self._started_at,
                completed_at=self._completed_at,
                token_usage={"total_tokens": self._total_tokens},
                metadata={
                    "model": self.config.model,
                    "focus": self.config.focus,
                    "plan": plan,
                },
            )

            logger.info(
                f"Agent {self.name} completed: "
                f"duration={result.duration_seconds:.1f}s, "
                f"findings={len(self.findings)}"
            )

            return result

        except asyncio.CancelledError:
            self.status = AgentStatus.CANCELLED
            self._completed_at = datetime.now()
            logger.warning(f"Agent {self.name} was cancelled")
            return AgentResult(
                agent_id=self.agent_id,
                agent_name=self.name,
                status=self.status,
                error="Agent execution was cancelled",
                started_at=self._started_at,
                completed_at=self._completed_at,
            )

        except Exception as e:
            self.status = AgentStatus.FAILED
            self._completed_at = datetime.now()
            logger.error(f"Agent {self.name} failed: {e}")
            return AgentResult(
                agent_id=self.agent_id,
                agent_name=self.name,
                status=self.status,
                error=str(e),
                started_at=self._started_at,
                completed_at=self._completed_at,
            )

    async def _initialize(
        self,
        hypothesis: str,
        research_questions: Optional[List[str]],
        context: Dict[str, Any],
    ) -> None:
        """
        Initialize agent state before research.

        Args:
            hypothesis: Research hypothesis
            research_questions: Specific questions
            context: Additional context
        """
        # Reset state
        self.conversation_history = []
        self.findings = []
        self.sources = []
        self._total_tokens = 0

        # Add system message
        self.conversation_history.append({
            "role": "system",
            "content": self._system_prompt,
        })

        logger.debug(f"Agent {self.name} initialized for research")

    async def _plan(
        self,
        hypothesis: str,
        research_questions: Optional[List[str]],
        context: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """
        Create a research plan (optional step).

        Override in subclasses that need planning capability.

        Args:
            hypothesis: Research hypothesis
            research_questions: Specific questions
            context: Additional context

        Returns:
            Research plan dict or None
        """
        return None

    @abstractmethod
    async def _execute(
        self,
        hypothesis: str,
        research_questions: Optional[List[str]],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Execute the main research task.

        This is the core method that subclasses must implement.

        Args:
            hypothesis: Research hypothesis
            research_questions: Specific questions
            context: Additional context including plan

        Returns:
            Execution results dict
        """
        pass

    async def _synthesize(
        self,
        execution_result: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Synthesize and format findings.

        Args:
            execution_result: Results from _execute
            context: Additional context

        Returns:
            Synthesized results dict with 'content' key
        """
        # Default implementation - subclasses can override
        content = execution_result.get("content", "")
        if not content and self.findings:
            # Generate content from findings
            content = "\n\n".join(
                f"**Finding {i+1}:** {f.get('summary', str(f))}"
                for i, f in enumerate(self.findings)
            )
        return {"content": content}

    async def _cleanup(self) -> None:
        """
        Cleanup resources after research.

        Override in subclasses that need cleanup.
        """
        pass

    async def chat(
        self,
        message: str,
        include_history: bool = True,
    ) -> CompletionResponse:
        """
        Send a chat message and get a response.

        Args:
            message: User message
            include_history: Whether to include conversation history

        Returns:
            CompletionResponse from the model
        """
        # Build messages
        if include_history:
            messages = self.conversation_history.copy()
        else:
            messages = [{"role": "system", "content": self._system_prompt}]

        messages.append({"role": "user", "content": message})

        # Get response
        response = await self.client.chat_completion(
            messages=messages,
            model=self.config.model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )

        # Update history
        self.conversation_history.append({"role": "user", "content": message})
        self.conversation_history.append({"role": "assistant", "content": response.content})

        # Track tokens
        self._total_tokens += response.usage.total_tokens

        return response

    def add_finding(
        self,
        summary: str,
        details: Optional[str] = None,
        confidence: float = 0.8,
        source: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Add a research finding.

        Args:
            summary: Brief summary of the finding
            details: Detailed explanation
            confidence: Confidence score 0-1
            source: Source information
        """
        finding = {
            "summary": summary,
            "details": details,
            "confidence": confidence,
            "timestamp": datetime.now().isoformat(),
        }
        self.findings.append(finding)

        if source:
            self.add_source(**source)

        logger.debug(f"Agent {self.name} added finding: {summary[:50]}...")

    def add_source(
        self,
        title: str,
        url: Optional[str] = None,
        author: Optional[str] = None,
        date: Optional[str] = None,
        **kwargs,
    ) -> None:
        """
        Add a source citation.

        Args:
            title: Source title
            url: Source URL
            author: Author name
            date: Publication date
            **kwargs: Additional source metadata
        """
        source = {
            "title": title,
            "url": url,
            "author": author,
            "date": date,
            **kwargs,
        }
        # Remove None values
        source = {k: v for k, v in source.items() if v is not None}
        self.sources.append(source)

    def get_state(self) -> Dict[str, Any]:
        """
        Get current agent state.

        Returns:
            Dictionary with agent state
        """
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "role": self.role.value,
            "status": self.status.value,
            "findings_count": len(self.findings),
            "sources_count": len(self.sources),
            "total_tokens": self._total_tokens,
            "conversation_length": len(self.conversation_history),
        }

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(id={self.agent_id}, name={self.name})>"

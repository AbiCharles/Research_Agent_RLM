"""
Multi-Agent Research Framework - Main Orchestrator

This is the main entry point for the research framework, providing
a high-level API for conducting multi-agent research investigations.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional
from uuid import uuid4

from agents.research_agent import ResearchAgent, ResearchAgentConfig, create_research_agent
from core.openai_client import OpenAIClient, get_client
from core.base_agent import AgentResult, AgentStatus
from core.memory_manager import MemoryManager
from config.settings import settings
from utils.logger import get_logger, setup_logging


logger = get_logger(__name__)


@dataclass
class ResearchRequest:
    """Request configuration for a research investigation."""

    hypothesis: str
    research_questions: List[str] = field(default_factory=list)
    agents: List[Dict[str, str]] = field(default_factory=list)
    model_tier: str = "default"
    max_agents: int = 5
    parallel_execution: bool = True


@dataclass
class ResearchReport:
    """Complete research report with all findings."""

    request_id: str
    hypothesis: str
    status: str
    summary: str
    agent_results: List[Dict[str, Any]]
    all_findings: List[Dict[str, Any]]
    all_sources: List[Dict[str, str]]
    started_at: datetime
    completed_at: Optional[datetime]
    total_tokens: int
    estimated_cost_usd: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration_seconds(self) -> Optional[float]:
        if self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "request_id": self.request_id,
            "hypothesis": self.hypothesis,
            "status": self.status,
            "summary": self.summary,
            "findings_count": len(self.all_findings),
            "sources_count": len(self.all_sources),
            "agent_count": len(self.agent_results),
            "duration_seconds": self.duration_seconds,
            "total_tokens": self.total_tokens,
            "estimated_cost_usd": self.estimated_cost_usd,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "metadata": self.metadata,
        }


class ResearchFramework:
    """
    Main orchestrator for multi-agent research investigations.

    This class coordinates multiple research agents to investigate
    complex hypotheses across different domains simultaneously.

    Example:
        >>> framework = ResearchFramework()
        >>> report = await framework.research(
        ...     hypothesis="AI will transform healthcare",
        ...     agents=[
        ...         {"name": "Tech Expert", "focus": "technology trends"},
        ...         {"name": "Market Analyst", "focus": "market dynamics"},
        ...     ]
        ... )
        >>> print(report.summary)
    """

    def __init__(
        self,
        client: Optional[OpenAIClient] = None,
        model_tier: str = "default",
    ):
        """
        Initialize the research framework.

        Args:
            client: OpenAI client (uses singleton if not provided)
            model_tier: Default model tier ('default', 'budget', 'premium', 'fast')
        """
        self.client = client or get_client()
        self.model_tier = model_tier
        self.memory_manager = MemoryManager()

        # Get model config for tier
        self.model_config = settings.get_model_config(model_tier)

        # Active research tracking
        self._active_agents: List[ResearchAgent] = []
        self._request_count = 0

        logger.info(f"ResearchFramework initialized with tier={model_tier}")

    async def research(
        self,
        hypothesis: str,
        research_questions: Optional[List[str]] = None,
        agents: Optional[List[Dict[str, str]]] = None,
        max_agents: int = 5,
        parallel: bool = True,
    ) -> ResearchReport:
        """
        Conduct a multi-agent research investigation.

        Args:
            hypothesis: The main hypothesis to investigate
            research_questions: Optional specific questions to address
            agents: List of agent configs [{"name": "...", "focus": "..."}]
            max_agents: Maximum number of agents to use
            parallel: Whether to run agents in parallel

        Returns:
            ResearchReport with all findings
        """
        request_id = str(uuid4())[:8]
        started_at = datetime.now()
        self._request_count += 1

        logger.info(f"Starting research request {request_id}: {hypothesis[:50]}...")

        # Create agents
        research_agents = self._create_agents(agents, max_agents)

        if not research_agents:
            # Create a default agent if none specified
            research_agents = [
                create_research_agent(
                    name="Research Analyst",
                    focus="general research and analysis",
                    model=self.model_config["agent_model"],
                )
            ]

        self._active_agents = research_agents
        logger.info(f"Created {len(research_agents)} agents for research")

        # Execute research
        try:
            if parallel and len(research_agents) > 1:
                results = await self._run_parallel(
                    research_agents, hypothesis, research_questions
                )
            else:
                results = await self._run_sequential(
                    research_agents, hypothesis, research_questions
                )

            # Aggregate results
            report = self._create_report(
                request_id=request_id,
                hypothesis=hypothesis,
                results=results,
                started_at=started_at,
            )

            logger.info(
                f"Research {request_id} completed: "
                f"{len(report.all_findings)} findings, "
                f"${report.estimated_cost_usd:.4f}"
            )

            return report

        except Exception as e:
            logger.error(f"Research {request_id} failed: {e}")
            return ResearchReport(
                request_id=request_id,
                hypothesis=hypothesis,
                status="failed",
                summary=f"Research failed: {str(e)}",
                agent_results=[],
                all_findings=[],
                all_sources=[],
                started_at=started_at,
                completed_at=datetime.now(),
                total_tokens=0,
                estimated_cost_usd=0,
                metadata={"error": str(e)},
            )

        finally:
            self._active_agents = []

    def _create_agents(
        self,
        agent_configs: Optional[List[Dict[str, str]]],
        max_agents: int,
    ) -> List[ResearchAgent]:
        """Create research agents from configurations."""
        if not agent_configs:
            return []

        agents = []
        for config in agent_configs[:max_agents]:
            agent = create_research_agent(
                name=config.get("name", f"Agent {len(agents) + 1}"),
                focus=config.get("focus", "general research"),
                tools=config.get("tools", ["web_search"]),
                model=self.model_config["agent_model"],
            )
            agents.append(agent)

        return agents

    async def _run_parallel(
        self,
        agents: List[ResearchAgent],
        hypothesis: str,
        questions: Optional[List[str]],
    ) -> List[AgentResult]:
        """Run multiple agents in parallel."""
        logger.info(f"Running {len(agents)} agents in parallel")

        tasks = [
            agent.run(hypothesis=hypothesis, research_questions=questions)
            for agent in agents
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle any exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Agent {agents[i].name} failed: {result}")
                processed_results.append(
                    AgentResult(
                        agent_id=agents[i].agent_id,
                        agent_name=agents[i].name,
                        status=AgentStatus.FAILED,
                        error=str(result),
                    )
                )
            else:
                processed_results.append(result)

        return processed_results

    async def _run_sequential(
        self,
        agents: List[ResearchAgent],
        hypothesis: str,
        questions: Optional[List[str]],
    ) -> List[AgentResult]:
        """Run agents sequentially."""
        logger.info(f"Running {len(agents)} agents sequentially")

        results = []
        for agent in agents:
            try:
                result = await agent.run(
                    hypothesis=hypothesis, research_questions=questions
                )
                results.append(result)
            except Exception as e:
                logger.error(f"Agent {agent.name} failed: {e}")
                results.append(
                    AgentResult(
                        agent_id=agent.agent_id,
                        agent_name=agent.name,
                        status=AgentStatus.FAILED,
                        error=str(e),
                    )
                )

        return results

    def _create_report(
        self,
        request_id: str,
        hypothesis: str,
        results: List[AgentResult],
        started_at: datetime,
    ) -> ResearchReport:
        """Create a comprehensive research report from agent results."""
        completed_at = datetime.now()

        # Aggregate all findings and sources
        all_findings = []
        all_sources = []
        agent_results = []
        total_tokens = 0

        for result in results:
            agent_results.append(result.to_dict())
            all_findings.extend(result.findings)
            all_sources.extend(result.sources)
            total_tokens += result.token_usage.get("total_tokens", 0)

        # Get cost from client
        usage = self.client.get_usage_summary()

        # Generate summary
        successful_agents = [r for r in results if r.status == AgentStatus.COMPLETED]
        summary_parts = []

        if successful_agents:
            summary_parts.append(
                f"Research completed with {len(successful_agents)}/{len(results)} agents successful."
            )
            summary_parts.append(f"Total findings: {len(all_findings)}")

            # Add top findings to summary
            if all_findings:
                summary_parts.append("\nKey findings:")
                for i, finding in enumerate(all_findings[:3], 1):
                    summary_parts.append(f"{i}. {finding.get('summary', 'N/A')[:100]}")
        else:
            summary_parts.append("Research completed but no agents succeeded.")

        status = "completed" if successful_agents else "failed"

        return ResearchReport(
            request_id=request_id,
            hypothesis=hypothesis,
            status=status,
            summary="\n".join(summary_parts),
            agent_results=agent_results,
            all_findings=all_findings,
            all_sources=all_sources,
            started_at=started_at,
            completed_at=completed_at,
            total_tokens=total_tokens,
            estimated_cost_usd=usage["total_cost_usd"],
            metadata={
                "model_tier": self.model_tier,
                "agents_count": len(results),
                "successful_agents": len(successful_agents),
            },
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get framework usage statistics."""
        usage = self.client.get_usage_summary()
        return {
            "request_count": self._request_count,
            "active_agents": len(self._active_agents),
            "model_tier": self.model_tier,
            **usage,
        }


# Convenience function for quick research
async def quick_research(
    hypothesis: str,
    agents: Optional[List[Dict[str, str]]] = None,
    questions: Optional[List[str]] = None,
) -> ResearchReport:
    """
    Quick research function for simple use cases.

    Args:
        hypothesis: Research hypothesis
        agents: Optional agent configurations
        questions: Optional research questions

    Returns:
        ResearchReport
    """
    framework = ResearchFramework()
    return await framework.research(
        hypothesis=hypothesis,
        agents=agents,
        research_questions=questions,
    )

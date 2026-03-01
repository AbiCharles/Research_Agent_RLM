"""
Lead Researcher agent for the Multi-Agent Research Framework.

This module provides the orchestrator agent that coordinates multiple
specialized research agents to conduct comprehensive investigations.
The Lead Researcher creates research plans, assigns tasks to agents,
manages parallel execution, and synthesizes findings into coherent reports.
"""

import asyncio
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime

from core.base_agent import (
    BaseAgent,
    AgentConfig,
    AgentRole,
    AgentResult,
    AgentStatus,
)
from core.openai_client import OpenAIClient
from core.memory_manager import MemoryManager
from agents.research_agent import ResearchAgent, ResearchAgentConfig, create_research_agent
from config.settings import settings
from utils.logger import get_logger


logger = get_logger(__name__)


class QueryComplexity:
    """Research query complexity levels."""
    SIMPLE = "simple"      # Single domain, straightforward
    MODERATE = "moderate"  # 2-3 domains, some nuance
    COMPLEX = "complex"    # Multiple domains, requires deep investigation


@dataclass
class AgentAssignment:
    """Assignment of a research task to an agent."""
    agent: ResearchAgent
    objective: str
    questions: List[str]
    priority: int = 1


@dataclass
class LeadResearcherConfig(AgentConfig):
    """Configuration for the Lead Researcher."""

    # Orchestration settings
    max_agents: int = field(default_factory=lambda: settings.MAX_AGENTS)
    max_parallel_agents: int = field(default_factory=lambda: settings.MAX_PARALLEL_AGENTS)
    use_chain_of_thought: bool = True

    # Model settings (Lead uses premium model by default)
    model: str = field(default_factory=lambda: settings.PREMIUM_MODEL)

    # Research settings
    max_research_time_minutes: int = field(
        default_factory=lambda: settings.MAX_RESEARCH_TIME_MINUTES
    )
    synthesize_intermediate: bool = True  # Synthesize after each batch

    def __post_init__(self):
        super().__post_init__()
        self.role = AgentRole.LEAD_RESEARCHER


@dataclass
class ResearchPlan:
    """A structured research plan created by the Lead Researcher."""

    hypothesis: str
    complexity: str
    domains: List[str]
    agent_assignments: List[Dict[str, Any]]
    research_questions: List[str]
    estimated_iterations: int
    strategy_notes: str
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hypothesis": self.hypothesis,
            "complexity": self.complexity,
            "domains": self.domains,
            "agent_assignments": self.agent_assignments,
            "research_questions": self.research_questions,
            "estimated_iterations": self.estimated_iterations,
            "strategy_notes": self.strategy_notes,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class OrchestrationResult:
    """Result from the full orchestration process."""

    hypothesis: str
    plan: ResearchPlan
    agent_results: List[AgentResult]
    synthesis: str
    final_report: str
    total_duration_seconds: float
    total_tokens: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hypothesis": self.hypothesis,
            "plan": self.plan.to_dict(),
            "agent_results": [r.to_dict() for r in self.agent_results],
            "synthesis": self.synthesis,
            "final_report": self.final_report,
            "total_duration_seconds": self.total_duration_seconds,
            "total_tokens": self.total_tokens,
            "metadata": self.metadata,
        }


class LeadResearcher(BaseAgent):
    """
    Lead Researcher agent that orchestrates multi-agent research.

    The Lead Researcher is the "brain" of the research framework. It:
    1. Analyzes the hypothesis to determine complexity and domains
    2. Creates a comprehensive research plan
    3. Assigns specific objectives to specialized agents
    4. Manages parallel execution of research agents
    5. Synthesizes all findings into a coherent final report

    Example:
        >>> lead = LeadResearcher(LeadResearcherConfig(name="Research Director"))
        >>>
        >>> # Define custom agents
        >>> agents = [
        ...     {"name": "Market Analyst", "focus": "market trends"},
        ...     {"name": "Tech Expert", "focus": "technical feasibility"},
        ... ]
        >>>
        >>> result = await lead.orchestrate(
        ...     hypothesis="AI will transform healthcare by 2030",
        ...     custom_agents=agents,
        ... )
    """

    def __init__(
        self,
        config: LeadResearcherConfig,
        client: Optional[OpenAIClient] = None,
        memory_manager: Optional[MemoryManager] = None,
    ):
        """
        Initialize the Lead Researcher.

        Args:
            config: Lead researcher configuration
            client: OpenAI client
            memory_manager: Memory manager for context handling
        """
        self.lead_config = config
        self.memory_manager = memory_manager or MemoryManager()

        super().__init__(config, client)

        # Orchestration state
        self._research_agents: List[ResearchAgent] = []
        self._current_plan: Optional[ResearchPlan] = None
        self._agent_results: List[AgentResult] = []

    def _build_system_prompt(self) -> str:
        """Build the Lead Researcher's system prompt."""
        cot_instructions = ""
        if self.lead_config.use_chain_of_thought:
            cot_instructions = """

## Chain-of-Thought Planning Process
When creating research plans, follow this structured approach:

1. **Hypothesis Analysis**: What exactly needs to be investigated?
2. **Domain Identification**: What knowledge domains are relevant?
3. **Complexity Assessment**: Is this simple, moderate, or complex?
4. **Agent Strategy**: How should specialized agents divide the work?
5. **Question Generation**: What specific questions need answers?
6. **Risk Assessment**: What challenges or gaps might arise?

Show your reasoning explicitly when making planning decisions."""

        return f"""You are {self.name}, the Lead Researcher orchestrating a multi-agent research investigation.

## Your Role
You are the central coordinator responsible for:
- Analyzing research hypotheses and determining investigation strategy
- Creating comprehensive research plans
- Assigning specific objectives to specialized research agents
- Synthesizing findings from multiple agents into coherent reports
- Ensuring thorough coverage of all relevant domains

## Guidelines
1. Break complex hypotheses into manageable research questions
2. Assign questions to agents based on their focus areas
3. Identify gaps or contradictions across agent findings
4. Provide balanced, evidence-based conclusions
5. Acknowledge uncertainty and limitations
{cot_instructions}

## Planning Output Format
When creating a research plan, structure it as:
- **Complexity**: simple/moderate/complex
- **Domains**: List of relevant knowledge domains
- **Agent Assignments**: Which agent investigates what
- **Key Questions**: Specific investigable questions
- **Strategy Notes**: Overall approach and considerations
"""

    async def orchestrate(
        self,
        hypothesis: str,
        research_questions: Optional[List[str]] = None,
        custom_agents: Optional[List[Dict[str, str]]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> OrchestrationResult:
        """
        Orchestrate a full multi-agent research investigation.

        This is the main entry point for conducting research. It:
        1. Creates a research plan based on the hypothesis
        2. Initializes or uses provided custom agents
        3. Executes research in parallel batches
        4. Synthesizes all findings into a final report

        Args:
            hypothesis: The main research hypothesis to investigate
            research_questions: Optional specific questions to address
            custom_agents: Optional list of custom agent definitions
                           [{"name": "...", "focus": "..."}, ...]
            context: Optional additional context for research

        Returns:
            OrchestrationResult with complete research findings
        """
        start_time = datetime.now()
        context = context or {}
        total_tokens = 0

        logger.info(f"Lead Researcher starting orchestration: {hypothesis[:100]}...")

        try:
            # Step 1: Create research plan
            plan = await self._create_research_plan(
                hypothesis=hypothesis,
                research_questions=research_questions,
                custom_agents=custom_agents,
            )
            self._current_plan = plan
            total_tokens += self._total_tokens

            # Step 2: Initialize research agents
            await self._initialize_agents(plan, custom_agents)

            # Step 3: Execute research in parallel batches
            agent_results = await self._execute_parallel_research(
                hypothesis=hypothesis,
                plan=plan,
            )
            self._agent_results = agent_results

            # Calculate total tokens from agents
            for result in agent_results:
                total_tokens += result.token_usage.get("total_tokens", 0)

            # Step 4: Synthesize findings
            synthesis = await self._synthesize_all_findings(
                hypothesis=hypothesis,
                plan=plan,
                agent_results=agent_results,
            )
            total_tokens += self._total_tokens

            # Step 5: Generate final report
            final_report = await self._generate_final_report(
                hypothesis=hypothesis,
                plan=plan,
                synthesis=synthesis,
                agent_results=agent_results,
            )
            total_tokens += self._total_tokens

            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            logger.info(
                f"Orchestration complete: {len(agent_results)} agents, "
                f"{duration:.1f}s, {total_tokens} tokens"
            )

            return OrchestrationResult(
                hypothesis=hypothesis,
                plan=plan,
                agent_results=agent_results,
                synthesis=synthesis,
                final_report=final_report,
                total_duration_seconds=duration,
                total_tokens=total_tokens,
                metadata={
                    "agents_used": len(agent_results),
                    "complexity": plan.complexity,
                    "domains": plan.domains,
                },
            )

        except Exception as e:
            logger.error(f"Orchestration failed: {e}")
            raise

    async def _create_research_plan(
        self,
        hypothesis: str,
        research_questions: Optional[List[str]],
        custom_agents: Optional[List[Dict[str, str]]],
    ) -> ResearchPlan:
        """
        Create a comprehensive research plan.

        Args:
            hypothesis: Research hypothesis
            research_questions: Optional specific questions
            custom_agents: Optional custom agent definitions

        Returns:
            ResearchPlan with strategy and assignments
        """
        # Build agent info for planning
        agent_info = ""
        if custom_agents:
            agent_list = "\n".join(
                f"- {a['name']}: {a.get('focus', 'general research')}"
                for a in custom_agents
            )
            agent_info = f"\n\nAvailable research agents:\n{agent_list}"

        questions_info = ""
        if research_questions:
            questions_info = "\n\nSpecific questions to address:\n" + "\n".join(
                f"- {q}" for q in research_questions
            )

        planning_prompt = f"""Create a research plan for the following hypothesis:

"{hypothesis}"
{questions_info}
{agent_info}

Analyze this hypothesis and create a comprehensive research plan:

1. **Complexity Assessment**: Is this simple, moderate, or complex?
   - Simple: Single domain, straightforward investigation
   - Moderate: 2-3 domains, some nuance required
   - Complex: Multiple domains, requires deep investigation

2. **Domain Identification**: What knowledge domains are relevant?
   (e.g., technology, economics, healthcare, legal, etc.)

3. **Research Questions**: What specific questions need investigation?
   Generate 3-5 key questions if not provided.

4. **Agent Assignments**: How should the available agents divide the work?
   Assign specific objectives and questions to each agent.

5. **Strategy Notes**: Any special considerations or approaches?

Provide a structured plan with clear assignments for each agent."""

        response = await self.chat(planning_prompt)

        # Parse the plan from the response
        plan = self._parse_research_plan(
            content=response.content,
            hypothesis=hypothesis,
            research_questions=research_questions,
            custom_agents=custom_agents,
        )

        logger.info(
            f"Research plan created: complexity={plan.complexity}, "
            f"domains={plan.domains}, agents={len(plan.agent_assignments)}"
        )

        return plan

    def _parse_research_plan(
        self,
        content: str,
        hypothesis: str,
        research_questions: Optional[List[str]],
        custom_agents: Optional[List[Dict[str, str]]],
    ) -> ResearchPlan:
        """
        Parse LLM response into a structured ResearchPlan.

        Args:
            content: Raw LLM response
            hypothesis: Original hypothesis
            research_questions: Original questions
            custom_agents: Custom agent definitions

        Returns:
            Structured ResearchPlan
        """
        content_lower = content.lower()

        # Determine complexity
        complexity = QueryComplexity.MODERATE
        if "complex" in content_lower and "not complex" not in content_lower:
            complexity = QueryComplexity.COMPLEX
        elif "simple" in content_lower:
            complexity = QueryComplexity.SIMPLE

        # Extract domains
        domains = []
        domain_keywords = [
            "technology", "economics", "healthcare", "legal", "financial",
            "market", "technical", "regulatory", "social", "environmental",
            "scientific", "business", "policy", "industry",
        ]
        for keyword in domain_keywords:
            if keyword in content_lower:
                domains.append(keyword)
        if not domains:
            domains = ["general"]

        # Build agent assignments
        agent_assignments = []
        if custom_agents:
            for agent in custom_agents:
                agent_assignments.append({
                    "name": agent["name"],
                    "focus": agent.get("focus", "general research"),
                    "objective": f"Investigate aspects related to {agent.get('focus', 'the hypothesis')}",
                })
        else:
            # Default agents based on complexity
            if complexity == QueryComplexity.SIMPLE:
                agent_assignments.append({
                    "name": "General Analyst",
                    "focus": "comprehensive analysis",
                    "objective": "Conduct thorough investigation of the hypothesis",
                })
            else:
                agent_assignments.append({
                    "name": "Domain Expert",
                    "focus": domains[0] if domains else "primary analysis",
                    "objective": f"Investigate {domains[0] if domains else 'primary'} aspects",
                })
                if len(domains) > 1:
                    agent_assignments.append({
                        "name": "Secondary Analyst",
                        "focus": domains[1],
                        "objective": f"Investigate {domains[1]} aspects",
                    })

        # Use provided questions or extract from content
        questions = research_questions or []
        if not questions:
            # Try to extract questions from the plan
            lines = content.split("\n")
            for line in lines:
                line = line.strip()
                if line.endswith("?") and len(line) > 20:
                    questions.append(line.lstrip("- *0123456789.)"))
            questions = questions[:5]  # Limit to 5

        if not questions:
            questions = [f"What evidence supports or refutes: {hypothesis}"]

        return ResearchPlan(
            hypothesis=hypothesis,
            complexity=complexity,
            domains=domains[:5],
            agent_assignments=agent_assignments,
            research_questions=questions,
            estimated_iterations=len(questions),
            strategy_notes=content[:500],
        )

    async def _initialize_agents(
        self,
        plan: ResearchPlan,
        custom_agents: Optional[List[Dict[str, str]]],
    ) -> None:
        """
        Initialize research agents based on the plan.

        Args:
            plan: Research plan with agent assignments
            custom_agents: Optional custom agent definitions
        """
        self._research_agents = []

        for assignment in plan.agent_assignments[:self.lead_config.max_agents]:
            agent = create_research_agent(
                name=assignment["name"],
                focus=assignment["focus"],
                model=settings.DEFAULT_MODEL,  # Agents use default model
            )
            self._research_agents.append(agent)

        logger.info(f"Initialized {len(self._research_agents)} research agents")

    async def _execute_parallel_research(
        self,
        hypothesis: str,
        plan: ResearchPlan,
    ) -> List[AgentResult]:
        """
        Execute research agents in parallel batches.

        Args:
            hypothesis: Research hypothesis
            plan: Research plan with assignments

        Returns:
            List of AgentResult from all agents
        """
        all_results = []
        batch_size = self.lead_config.max_parallel_agents

        # Distribute questions among agents
        questions_per_agent = self._distribute_questions(
            questions=plan.research_questions,
            num_agents=len(self._research_agents),
        )

        # Execute in batches
        for batch_start in range(0, len(self._research_agents), batch_size):
            batch_end = min(batch_start + batch_size, len(self._research_agents))
            batch_agents = self._research_agents[batch_start:batch_end]

            logger.info(
                f"Executing agent batch {batch_start//batch_size + 1}: "
                f"{len(batch_agents)} agents"
            )

            # Create tasks for parallel execution
            tasks = []
            for i, agent in enumerate(batch_agents):
                agent_idx = batch_start + i
                agent_questions = questions_per_agent.get(agent_idx, [])

                task = agent.run(
                    hypothesis=hypothesis,
                    research_questions=agent_questions,
                    context={
                        "plan": plan.to_dict(),
                        "assignment": plan.agent_assignments[agent_idx]
                        if agent_idx < len(plan.agent_assignments)
                        else {},
                    },
                )
                tasks.append(task)

            # Execute batch in parallel
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results
            for result in batch_results:
                if isinstance(result, Exception):
                    logger.error(f"Agent execution failed: {result}")
                    all_results.append(
                        AgentResult(
                            agent_id="error",
                            agent_name="Unknown",
                            status=AgentStatus.FAILED,
                            error=str(result),
                        )
                    )
                else:
                    all_results.append(result)

            # Optional: Synthesize intermediate results
            if self.lead_config.synthesize_intermediate and batch_end < len(self._research_agents):
                await self._intermediate_synthesis(all_results)

        return all_results

    def _distribute_questions(
        self,
        questions: List[str],
        num_agents: int,
    ) -> Dict[int, List[str]]:
        """
        Distribute research questions among agents.

        Args:
            questions: List of research questions
            num_agents: Number of agents

        Returns:
            Dict mapping agent index to questions
        """
        distribution = {i: [] for i in range(num_agents)}

        for i, question in enumerate(questions):
            agent_idx = i % num_agents
            distribution[agent_idx].append(question)

        return distribution

    async def _intermediate_synthesis(
        self,
        current_results: List[AgentResult],
    ) -> None:
        """
        Perform intermediate synthesis of results.

        Args:
            current_results: Results gathered so far
        """
        successful_results = [r for r in current_results if r.status == AgentStatus.COMPLETED]
        if not successful_results:
            return

        # Quick synthesis for progress tracking
        findings_summary = "\n".join(
            f"- {r.agent_name}: {len(r.findings)} findings"
            for r in successful_results
        )
        logger.info(f"Intermediate progress:\n{findings_summary}")

    async def _synthesize_all_findings(
        self,
        hypothesis: str,
        plan: ResearchPlan,
        agent_results: List[AgentResult],
    ) -> str:
        """
        Synthesize findings from all agents.

        Args:
            hypothesis: Research hypothesis
            plan: Research plan
            agent_results: Results from all agents

        Returns:
            Synthesized findings text
        """
        # Gather all findings
        all_findings = []
        for result in agent_results:
            if result.status == AgentStatus.COMPLETED:
                for finding in result.findings:
                    all_findings.append({
                        "agent": result.agent_name,
                        "finding": finding,
                    })

        if not all_findings:
            return "No findings were generated by the research agents."

        # Build synthesis prompt
        findings_text = "\n\n".join(
            f"**From {f['agent']}**:\n{f['finding'].get('summary', str(f['finding']))}"
            for f in all_findings[:20]  # Limit for context
        )

        synthesis_prompt = f"""Synthesize the following research findings related to the hypothesis:

Hypothesis: "{hypothesis}"

Agent Findings:
{findings_text}

Please provide:
1. **Key Themes**: What common themes emerge across agents?
2. **Convergent Findings**: Where do agents agree?
3. **Divergent Findings**: Where do agents disagree or find contradictions?
4. **Gaps**: What areas weren't adequately covered?
5. **Overall Assessment**: What can we conclude about the hypothesis?"""

        response = await self.chat(synthesis_prompt)
        return response.content

    async def _generate_final_report(
        self,
        hypothesis: str,
        plan: ResearchPlan,
        synthesis: str,
        agent_results: List[AgentResult],
    ) -> str:
        """
        Generate the comprehensive final report.

        Args:
            hypothesis: Research hypothesis
            plan: Research plan
            synthesis: Synthesized findings
            agent_results: All agent results

        Returns:
            Formatted final report
        """
        # Collect all sources
        all_sources = []
        for result in agent_results:
            all_sources.extend(result.sources)

        sources_text = ""
        if all_sources:
            unique_sources = {s.get("title", str(s)): s for s in all_sources}
            sources_text = "\n".join(
                f"- {s.get('title', 'Unknown')} ({s.get('url', 'N/A')})"
                for s in unique_sources.values()
            )

        report_prompt = f"""Generate a comprehensive research report based on the following:

## Hypothesis
{hypothesis}

## Research Summary
{synthesis}

## Sources Referenced
{sources_text if sources_text else "No external sources cited."}

Create a well-structured final report that:
1. Introduces the research question and approach
2. Presents key findings organized by theme
3. Provides evidence-based conclusions
4. Acknowledges limitations and uncertainties
5. Suggests areas for further investigation

Format the report professionally with clear sections and headers."""

        response = await self.chat(report_prompt)
        return response.content

    async def _execute(
        self,
        hypothesis: str,
        research_questions: Optional[List[str]],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Execute method required by BaseAgent.

        For LeadResearcher, the main entry point is orchestrate(),
        but this implements the abstract method for compatibility.
        """
        # Run orchestration
        result = await self.orchestrate(
            hypothesis=hypothesis,
            research_questions=research_questions,
            context=context,
        )

        # Store findings
        for agent_result in result.agent_results:
            for finding in agent_result.findings:
                self.findings.append(finding)
            for source in agent_result.sources:
                self.sources.append(source)

        return {
            "content": result.final_report,
            "synthesis": result.synthesis,
            "agent_count": len(result.agent_results),
        }

    def get_orchestration_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the orchestration.

        Returns:
            Dictionary with orchestration statistics
        """
        stats = self.get_state()
        stats.update({
            "agents_initialized": len(self._research_agents),
            "agent_results": len(self._agent_results),
            "current_plan": self._current_plan.to_dict() if self._current_plan else None,
        })
        return stats


def create_lead_researcher(
    name: str = "Lead Researcher",
    model: Optional[str] = None,
    max_agents: int = 5,
    **kwargs,
) -> LeadResearcher:
    """
    Factory function to create a Lead Researcher.

    Args:
        name: Researcher name
        model: Model to use (defaults to premium model)
        max_agents: Maximum number of research agents
        **kwargs: Additional config options

    Returns:
        Configured LeadResearcher instance
    """
    config = LeadResearcherConfig(
        name=name,
        model=model or settings.PREMIUM_MODEL,
        max_agents=max_agents,
        **kwargs,
    )
    return LeadResearcher(config)

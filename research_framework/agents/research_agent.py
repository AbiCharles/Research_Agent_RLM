"""
Research agent implementation for the Multi-Agent Research Framework.

This module provides a concrete implementation of the BaseAgent class
specialized for conducting research investigations with chain-of-thought
reasoning and tool integration.
"""

import asyncio
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

from core.base_agent import (
    BaseAgent,
    AgentConfig,
    AgentRole,
    AgentResult,
)
from core.openai_client import OpenAIClient, CompletionResponse
from core.memory_manager import MemoryManager
from config.settings import settings
from utils.logger import get_logger


logger = get_logger(__name__)


@dataclass
class ResearchAgentConfig(AgentConfig):
    """Extended configuration for research agents."""

    # Research-specific settings
    max_iterations: int = 5  # Max research iterations
    min_confidence: float = 0.6  # Minimum confidence to accept finding
    use_chain_of_thought: bool = True  # Enable CoT reasoning
    parallel_queries: int = 3  # Queries to run in parallel

    def __post_init__(self):
        super().__post_init__()
        if self.role == AgentRole.CUSTOM:
            self.role = AgentRole.RESEARCH_AGENT


class ResearchAgent(BaseAgent):
    """
    Specialized research agent with chain-of-thought reasoning.

    This agent conducts research by:
    1. Breaking down the hypothesis into investigable questions
    2. Using available tools to gather information
    3. Applying chain-of-thought reasoning to analyze findings
    4. Synthesizing results into coherent conclusions

    Example:
        >>> config = ResearchAgentConfig(
        ...     name="Market Analyst",
        ...     focus="market trends and competitive analysis"
        ... )
        >>> agent = ResearchAgent(config)
        >>> result = await agent.run(
        ...     hypothesis="AI will transform healthcare by 2030",
        ...     research_questions=["What AI applications exist in healthcare?"]
        ... )
    """

    def __init__(
        self,
        config: ResearchAgentConfig,
        client: Optional[OpenAIClient] = None,
        memory_manager: Optional[MemoryManager] = None,
    ):
        """
        Initialize research agent.

        Args:
            config: Research agent configuration
            client: OpenAI client
            memory_manager: Memory manager for context handling
        """
        # Set research_config BEFORE super().__init__ because _build_system_prompt needs it
        self.research_config = config
        self.memory_manager = memory_manager or MemoryManager()

        super().__init__(config, client)

        # Research state
        self._current_iteration = 0
        self._queries_executed = 0

    def _build_system_prompt(self) -> str:
        """Build research-specific system prompt with CoT instructions."""
        cot_instructions = ""
        if self.research_config.use_chain_of_thought:
            cot_instructions = """

## Chain-of-Thought Reasoning Process
When analyzing information, follow this structured thinking process:

1. **Understand**: Restate the question/hypothesis in your own words
2. **Break Down**: Identify key components that need investigation
3. **Analyze**: For each piece of evidence:
   - What does this tell us?
   - How reliable is this source?
   - What are the limitations?
4. **Connect**: How do different findings relate to each other?
5. **Conclude**: What can we confidently conclude? What remains uncertain?

Always show your reasoning process explicitly."""

        return f"""You are {self.name}, a specialized research agent in a multi-agent research system.

## Your Role
{self.config.role.value}

## Your Focus Area
{self.config.focus or 'General research and analysis'}

## Research Guidelines
1. Be thorough but focused on the specific research questions
2. Cite sources for all factual claims using [Source: title/url] format
3. Distinguish between facts, analysis, and speculation
4. Acknowledge uncertainty and gaps in available information
5. Provide actionable insights when possible
{cot_instructions}

## Response Format
Structure your findings as:
- **Key Finding**: Main insight
- **Evidence**: Supporting information with sources
- **Confidence**: How certain are you (high/medium/low)
- **Implications**: What this means for the hypothesis
"""

    async def _plan(
        self,
        hypothesis: str,
        research_questions: Optional[List[str]],
        context: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """
        Create a research plan by breaking down the hypothesis.

        Args:
            hypothesis: Main research hypothesis
            research_questions: Optional specific questions
            context: Additional context

        Returns:
            Research plan with questions and approach
        """
        # Build planning prompt
        questions_text = ""
        if research_questions:
            questions_text = "\n".join(f"- {q}" for q in research_questions)
            questions_text = f"\n\nSpecific questions to address:\n{questions_text}"

        planning_prompt = f"""I need to research the following hypothesis:

"{hypothesis}"
{questions_text}

Create a focused research plan by:
1. Identifying 3-5 key aspects that need investigation
2. For each aspect, specify what information to look for
3. Note any potential challenges or limitations

Format your response as a structured plan with clear investigation points."""

        response = await self.chat(planning_prompt)

        # Parse the plan from response
        plan = {
            "hypothesis": hypothesis,
            "research_questions": research_questions or [],
            "plan_text": response.content,
            "iterations_planned": min(
                self.research_config.max_iterations,
                len(research_questions) if research_questions else 3,
            ),
        }

        logger.info(f"Research plan created for agent {self.name}")
        return plan

    async def _execute(
        self,
        hypothesis: str,
        research_questions: Optional[List[str]],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Execute the main research process.

        This method:
        1. Iterates through research questions
        2. Gathers information using available tools
        3. Applies chain-of-thought analysis
        4. Collects findings and sources

        Args:
            hypothesis: Research hypothesis
            research_questions: Specific questions
            context: Context including plan

        Returns:
            Execution results with all findings
        """
        plan = context.get("plan", {})
        questions = research_questions or []

        # If no specific questions, generate from hypothesis
        if not questions:
            questions = await self._generate_research_questions(hypothesis)

        all_findings = []
        all_content = []

        # Process each research question
        for i, question in enumerate(questions[: self.research_config.max_iterations]):
            self._current_iteration = i + 1
            logger.info(
                f"Agent {self.name} processing question {i+1}/{len(questions)}: {question[:50]}..."
            )

            # Check memory limits before continuing
            util = self.memory_manager.get_utilization(self.conversation_history)
            if util['needs_processing']:
                logger.warning(
                    f"Agent {self.name} approaching context limit "
                    f"({util['utilization_pct']}), truncating history"
                )
                self.conversation_history = self.memory_manager.truncate_simple(
                    self.conversation_history
                )

            # Research this question
            finding = await self._research_question(hypothesis, question)

            if finding:
                all_findings.append(finding)
                all_content.append(finding.get("analysis", ""))

                # Add to agent findings
                self.add_finding(
                    summary=finding.get("summary", ""),
                    details=finding.get("analysis", ""),
                    confidence=finding.get("confidence", 0.7),
                )

                # Extract and add sources
                for source in finding.get("sources", []):
                    self.add_source(**source)

        return {
            "findings": all_findings,
            "content": "\n\n".join(all_content),
            "questions_processed": len(all_findings),
            "iterations": self._current_iteration,
        }

    async def _generate_research_questions(
        self,
        hypothesis: str,
        num_questions: int = 3,
    ) -> List[str]:
        """
        Generate research questions from hypothesis.

        Args:
            hypothesis: The hypothesis to investigate
            num_questions: Number of questions to generate

        Returns:
            List of research questions
        """
        prompt = f"""Given this research hypothesis:
"{hypothesis}"

Generate {num_questions} specific, investigable research questions that would help validate or refute this hypothesis.

Format: Return only the questions, one per line, without numbering."""

        response = await self.chat(prompt)

        # Parse questions from response
        questions = [
            q.strip().lstrip("0123456789.-) ")
            for q in response.content.strip().split("\n")
            if q.strip() and not q.strip().startswith("#")
        ]

        return questions[:num_questions]

    async def _research_question(
        self,
        hypothesis: str,
        question: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Research a specific question.

        Args:
            hypothesis: Overall hypothesis
            question: Specific question to research

        Returns:
            Finding dictionary with analysis and sources
        """
        # Build research prompt with CoT
        if self.research_config.use_chain_of_thought:
            research_prompt = f"""Research Question: {question}

Context: This is part of investigating the hypothesis: "{hypothesis}"

Please investigate this question using the following chain-of-thought process:

**Step 1 - Understanding**: What exactly is being asked?

**Step 2 - Key Factors**: What are the key factors to consider?

**Step 3 - Analysis**: Based on your knowledge, what can you determine about this?
(Note: In a production system, this would include tool calls to search for current information)

**Step 4 - Evidence Assessment**: How reliable is this information? What are the limitations?

**Step 5 - Conclusion**: What is your finding regarding this question?

Provide your response with:
- A brief summary (1-2 sentences)
- Your detailed analysis following the steps above
- Confidence level (high/medium/low) with justification
- Any sources or references used"""
        else:
            research_prompt = f"""Research the following question in the context of the hypothesis "{hypothesis}":

Question: {question}

Provide:
1. A brief summary of your finding
2. Supporting analysis
3. Confidence level (high/medium/low)
4. Sources used"""

        response = await self.chat(research_prompt)

        # Parse the response into structured finding
        finding = self._parse_finding(response.content, question)

        return finding

    def _parse_finding(
        self,
        content: str,
        question: str,
    ) -> Dict[str, Any]:
        """
        Parse research response into structured finding.

        Args:
            content: Raw response content
            question: The research question

        Returns:
            Structured finding dictionary
        """
        # Extract confidence level
        confidence = 0.7  # Default
        content_lower = content.lower()
        if "high confidence" in content_lower or "confidence: high" in content_lower:
            confidence = 0.9
        elif "low confidence" in content_lower or "confidence: low" in content_lower:
            confidence = 0.5
        elif "medium confidence" in content_lower or "confidence: medium" in content_lower:
            confidence = 0.7

        # Extract summary (first substantial paragraph or sentence)
        lines = content.strip().split("\n")
        summary = ""
        for line in lines:
            line = line.strip()
            if line and not line.startswith("#") and not line.startswith("*"):
                # Skip step labels
                if not any(
                    line.lower().startswith(s)
                    for s in ["step", "**step", "understanding:", "key factors:"]
                ):
                    summary = line[:200]
                    break

        if not summary:
            summary = content[:200]

        # Extract sources mentioned
        sources = []
        source_indicators = ["[source:", "[ref:", "source:", "reference:"]
        for line in lines:
            line_lower = line.lower()
            for indicator in source_indicators:
                if indicator in line_lower:
                    # Extract source info
                    source_text = line.split(indicator)[-1].strip().rstrip("]")
                    sources.append({"title": source_text[:100]})

        return {
            "question": question,
            "summary": summary,
            "analysis": content,
            "confidence": confidence,
            "sources": sources,
        }

    async def _synthesize(
        self,
        execution_result: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Synthesize all findings into a coherent summary.

        Args:
            execution_result: Results from execution
            context: Additional context

        Returns:
            Synthesized results with final content
        """
        if not self.findings:
            return {"content": "No findings were generated during research."}

        # Build synthesis prompt
        findings_text = "\n\n".join(
            f"**Finding {i+1}** (Confidence: {f.get('confidence', 'unknown')}):\n{f.get('summary', '')}"
            for i, f in enumerate(self.findings)
        )

        synthesis_prompt = f"""Based on my research, I found the following:

{findings_text}

Please synthesize these findings into a coherent summary that:
1. Highlights the key insights
2. Notes any contradictions or gaps
3. Provides an overall assessment related to the original research focus
4. Lists remaining questions or areas needing further investigation

Focus area: {self.config.focus}"""

        response = await self.chat(synthesis_prompt)

        return {
            "content": response.content,
            "findings_count": len(self.findings),
            "synthesis_tokens": response.usage.total_tokens,
        }

    def get_research_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the research process.

        Returns:
            Dictionary with research statistics
        """
        base_stats = self.get_state()
        base_stats.update({
            "iterations_completed": self._current_iteration,
            "max_iterations": self.research_config.max_iterations,
            "use_chain_of_thought": self.research_config.use_chain_of_thought,
            "average_confidence": (
                sum(f.get("confidence", 0) for f in self.findings) / len(self.findings)
                if self.findings
                else 0
            ),
        })
        return base_stats


def create_research_agent(
    name: str,
    focus: str,
    tools: Optional[List[str]] = None,
    model: Optional[str] = None,
    **kwargs,
) -> ResearchAgent:
    """
    Factory function to create a research agent.

    Args:
        name: Agent name
        focus: Research focus area
        tools: List of tool names to enable
        model: Model to use
        **kwargs: Additional config options

    Returns:
        Configured ResearchAgent instance
    """
    config = ResearchAgentConfig(
        name=name,
        focus=focus,
        tools=tools or ["web_search"],
        model=model or settings.DEFAULT_MODEL,
        **kwargs,
    )
    return ResearchAgent(config)

"""
Skills Framework for Stage 4 of the Knowledge Pipeline (Application Layer).

This module implements the Application Layer of the RLM paradigm, providing
specialized analysis capabilities by combining tools, domain expertise, and
procedural workflows. Skills receive filtered and compressed context from
Stages 1-3 and apply domain-specific logic to generate insights.

Key Concepts (per §7.5 Python Developer Guide):
-----------------------------------------------
- **Skill**: Encapsulates domain expertise with tools and procedures
- **Tool**: A callable that performs a specific analysis task
- **Procedure**: A workflow that combines tool outputs
- **SkillRegistry**: Central management of available skills
- **SkillExecutor**: Async execution with error handling

Built-in Skills:
----------------
1. ResearchAnalysisSkill: Analyze research findings for patterns
2. FinancialAnalysisSkill: Financial ratio and trend analysis
3. SentimentAnalysisSkill: Analyze sentiment in content
4. EntityExtractionSkill: Extract and link named entities
5. SummarySkill: Generate structured summaries

Usage Examples:
---------------
    # Define a custom skill
    >>> from filters import Skill, SkillRegistry
    >>>
    >>> skill = Skill(
    ...     name="market_analysis",
    ...     description="Analyze market trends and opportunities"
    ... )
    >>> skill.add_tool(calculate_market_share)
    >>> skill.add_procedure(trend_analysis)

    # Register and execute
    >>> registry = SkillRegistry()
    >>> registry.register(skill)
    >>> result = await registry.execute("market_analysis", context)

    # Use built-in skills
    >>> from filters import ResearchAnalysisSkill
    >>> skill = ResearchAnalysisSkill()
    >>> result = await skill.execute_async(context)

Integration with Pipeline:
--------------------------
    >>> from filters import KnowledgePipeline
    >>>
    >>> pipeline = KnowledgePipeline(
    ...     knowledge_base=kb,
    ...     skills=["research_analysis", "entity_extraction"]
    ... )
    >>> result = await pipeline.process("What are AI trends in healthcare?")
"""

import asyncio
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)

logger = logging.getLogger(__name__)

# Type aliases
ToolFunc = Callable[[Dict[str, Any]], Any]
ProcedureFunc = Callable[[Dict[str, Any], Dict[str, Any]], Any]
AsyncToolFunc = Callable[[Dict[str, Any]], Any]


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class SkillConfig:
    """
    Configuration for skill execution.

    Attributes:
        timeout: Maximum execution time per skill (seconds).
        max_retries: Number of retry attempts on failure.
        parallel_tools: Whether to execute tools in parallel.
        capture_metrics: Whether to capture execution metrics.
        fallback_on_error: Return partial results on error.

    Example:
        >>> config = SkillConfig(timeout=30, parallel_tools=True)
        >>> skill = Skill("analysis", config=config)
    """

    timeout: float = 60.0
    max_retries: int = 2
    parallel_tools: bool = True
    capture_metrics: bool = True
    fallback_on_error: bool = True


@dataclass
class SkillResult:
    """
    Result of skill execution.

    Attributes:
        skill_name: Name of the executed skill.
        success: Whether execution completed successfully.
        tool_results: Results from individual tools.
        procedure_results: Results from procedures.
        errors: Any errors encountered.
        metrics: Execution metrics (timing, counts).
        timestamp: When execution completed.
    """

    skill_name: str
    success: bool = True
    tool_results: Dict[str, Any] = field(default_factory=dict)
    procedure_results: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def all_results(self) -> Dict[str, Any]:
        """Combined tool and procedure results."""
        return {**self.tool_results, **self.procedure_results}

    def get(self, key: str, default: Any = None) -> Any:
        """Get a result by key."""
        return self.all_results.get(key, default)


# =============================================================================
# Base Skill Class
# =============================================================================

class Skill:
    """
    Base class for skills in the Application Layer.

    A Skill encapsulates domain expertise by combining:
    - **Tools**: Individual analysis functions
    - **Procedures**: Workflows that combine tool outputs

    Execution Flow:
    ---------------
    1. All tools execute (optionally in parallel)
    2. Procedures execute sequentially with tool results
    3. Results are aggregated and returned

    Example:
        >>> skill = Skill(
        ...     name="financial_analysis",
        ...     description="Analyze financial data"
        ... )
        >>>
        >>> # Add tools
        >>> skill.add_tool(calculate_ratios)
        >>> skill.add_tool(analyze_cashflow)
        >>>
        >>> # Add procedure that uses tool results
        >>> skill.add_procedure(generate_insights)
        >>>
        >>> # Execute
        >>> result = skill.execute({"financial_data": data})
    """

    def __init__(
        self,
        name: str,
        description: str = "",
        config: Optional[SkillConfig] = None,
    ):
        """
        Initialize a skill.

        Args:
            name: Unique skill identifier.
            description: Human-readable description.
            config: Execution configuration.
        """
        self.name = name
        self.description = description
        self.config = config or SkillConfig()
        self.tools: List[Tuple[str, ToolFunc]] = []
        self.procedures: List[Tuple[str, ProcedureFunc]] = []
        self._metadata: Dict[str, Any] = {}

    def add_tool(
        self,
        tool: ToolFunc,
        name: Optional[str] = None,
    ) -> "Skill":
        """
        Add a tool to this skill.

        Args:
            tool: Callable that takes context and returns results.
            name: Optional name override (defaults to function name).

        Returns:
            Self for chaining.

        Example:
            >>> skill.add_tool(calculate_ratios)
            >>> skill.add_tool(custom_func, name="ratio_calculator")
        """
        tool_name = name or tool.__name__
        self.tools.append((tool_name, tool))
        logger.debug(f"Added tool '{tool_name}' to skill '{self.name}'")
        return self

    def add_procedure(
        self,
        procedure: ProcedureFunc,
        name: Optional[str] = None,
    ) -> "Skill":
        """
        Add a procedure to this skill.

        Procedures receive both context and tool results, enabling
        them to combine and analyze tool outputs.

        Args:
            procedure: Callable(context, tool_results) -> result.
            name: Optional name override.

        Returns:
            Self for chaining.

        Example:
            >>> def analyze_trends(context, tool_results):
            ...     ratios = tool_results["calculate_ratios"]
            ...     return {"trend": "improving"}
            >>> skill.add_procedure(analyze_trends)
        """
        proc_name = name or procedure.__name__
        self.procedures.append((proc_name, procedure))
        logger.debug(f"Added procedure '{proc_name}' to skill '{self.name}'")
        return self

    def execute(self, context: Dict[str, Any]) -> SkillResult:
        """
        Execute skill synchronously.

        Args:
            context: Execution context with input data.

        Returns:
            SkillResult with tool and procedure results.
        """
        start_time = datetime.now()
        result = SkillResult(skill_name=self.name)

        try:
            # Execute tools
            for tool_name, tool in self.tools:
                try:
                    tool_result = tool(context)
                    result.tool_results[tool_name] = tool_result
                except Exception as e:
                    error_msg = f"Tool '{tool_name}' failed: {e}"
                    logger.warning(error_msg)
                    result.errors.append(error_msg)
                    if not self.config.fallback_on_error:
                        raise

            # Execute procedures with tool results
            for proc_name, procedure in self.procedures:
                try:
                    proc_result = procedure(context, result.tool_results)
                    result.procedure_results[proc_name] = proc_result
                except Exception as e:
                    error_msg = f"Procedure '{proc_name}' failed: {e}"
                    logger.warning(error_msg)
                    result.errors.append(error_msg)
                    if not self.config.fallback_on_error:
                        raise

        except Exception as e:
            result.success = False
            result.errors.append(f"Skill execution failed: {e}")

        # Capture metrics
        if self.config.capture_metrics:
            result.metrics = {
                "duration_ms": (datetime.now() - start_time).total_seconds() * 1000,
                "tools_executed": len(result.tool_results),
                "procedures_executed": len(result.procedure_results),
                "error_count": len(result.errors),
            }

        return result

    async def execute_async(self, context: Dict[str, Any]) -> SkillResult:
        """
        Execute skill asynchronously.

        Supports parallel tool execution when config.parallel_tools=True.

        Args:
            context: Execution context with input data.

        Returns:
            SkillResult with tool and procedure results.
        """
        start_time = datetime.now()
        result = SkillResult(skill_name=self.name)

        try:
            # Execute tools (parallel or sequential)
            if self.config.parallel_tools and len(self.tools) > 1:
                result.tool_results = await self._execute_tools_parallel(
                    context, result
                )
            else:
                result.tool_results = await self._execute_tools_sequential(
                    context, result
                )

            # Execute procedures sequentially (they depend on tool results)
            for proc_name, procedure in self.procedures:
                try:
                    if asyncio.iscoroutinefunction(procedure):
                        proc_result = await procedure(context, result.tool_results)
                    else:
                        proc_result = procedure(context, result.tool_results)
                    result.procedure_results[proc_name] = proc_result
                except Exception as e:
                    error_msg = f"Procedure '{proc_name}' failed: {e}"
                    logger.warning(error_msg)
                    result.errors.append(error_msg)
                    if not self.config.fallback_on_error:
                        raise

        except Exception as e:
            result.success = False
            result.errors.append(f"Skill execution failed: {e}")

        # Capture metrics
        if self.config.capture_metrics:
            result.metrics = {
                "duration_ms": (datetime.now() - start_time).total_seconds() * 1000,
                "tools_executed": len(result.tool_results),
                "procedures_executed": len(result.procedure_results),
                "error_count": len(result.errors),
                "parallel_execution": self.config.parallel_tools,
            }

        return result

    async def _execute_tools_parallel(
        self,
        context: Dict[str, Any],
        result: SkillResult,
    ) -> Dict[str, Any]:
        """Execute tools in parallel."""
        tool_results = {}

        async def run_tool(name: str, tool: ToolFunc):
            try:
                if asyncio.iscoroutinefunction(tool):
                    return name, await tool(context)
                else:
                    return name, tool(context)
            except Exception as e:
                error_msg = f"Tool '{name}' failed: {e}"
                logger.warning(error_msg)
                result.errors.append(error_msg)
                return name, None

        tasks = [run_tool(name, tool) for name, tool in self.tools]
        results = await asyncio.gather(*tasks)

        for name, res in results:
            if res is not None:
                tool_results[name] = res

        return tool_results

    async def _execute_tools_sequential(
        self,
        context: Dict[str, Any],
        result: SkillResult,
    ) -> Dict[str, Any]:
        """Execute tools sequentially."""
        tool_results = {}

        for tool_name, tool in self.tools:
            try:
                if asyncio.iscoroutinefunction(tool):
                    tool_result = await tool(context)
                else:
                    tool_result = tool(context)
                tool_results[tool_name] = tool_result
            except Exception as e:
                error_msg = f"Tool '{tool_name}' failed: {e}"
                logger.warning(error_msg)
                result.errors.append(error_msg)
                if not self.config.fallback_on_error:
                    raise

        return tool_results


# =============================================================================
# Skill Registry
# =============================================================================

class SkillRegistry:
    """
    Central registry for managing skills.

    The registry provides:
    - Skill registration and lookup
    - Skill execution by name
    - Skill discovery and listing
    - Default skill management

    Example:
        >>> registry = SkillRegistry()
        >>>
        >>> # Register custom skill
        >>> registry.register(my_skill)
        >>>
        >>> # Execute by name
        >>> result = await registry.execute("my_skill", context)
        >>>
        >>> # List available skills
        >>> for name in registry.list_skills():
        ...     print(name)
    """

    def __init__(self, load_defaults: bool = True):
        """
        Initialize the skill registry.

        Args:
            load_defaults: Whether to load built-in skills.
        """
        self._skills: Dict[str, Skill] = {}
        self._metadata: Dict[str, Dict[str, Any]] = {}

        if load_defaults:
            self._load_default_skills()

    def _load_default_skills(self):
        """Load built-in skills."""
        # Research analysis
        self.register(ResearchAnalysisSkill())
        # Summary generation
        self.register(SummarySkill())
        # Entity extraction
        self.register(EntityExtractionSkill())
        # Sentiment analysis
        self.register(SentimentAnalysisSkill())

        logger.info(f"Loaded {len(self._skills)} default skills")

    def register(
        self,
        skill: Skill,
        overwrite: bool = False,
    ) -> "SkillRegistry":
        """
        Register a skill.

        Args:
            skill: Skill instance to register.
            overwrite: Whether to overwrite existing skill.

        Returns:
            Self for chaining.

        Raises:
            ValueError: If skill exists and overwrite=False.
        """
        if skill.name in self._skills and not overwrite:
            raise ValueError(
                f"Skill '{skill.name}' already registered. "
                "Use overwrite=True to replace."
            )

        self._skills[skill.name] = skill
        self._metadata[skill.name] = {
            "description": skill.description,
            "tool_count": len(skill.tools),
            "procedure_count": len(skill.procedures),
            "registered_at": datetime.now(),
        }

        logger.info(f"Registered skill: {skill.name}")
        return self

    def get(self, name: str) -> Optional[Skill]:
        """
        Get a skill by name.

        Args:
            name: Skill name.

        Returns:
            Skill instance or None if not found.
        """
        return self._skills.get(name)

    def list_skills(self) -> List[str]:
        """List all registered skill names."""
        return list(self._skills.keys())

    def get_skill_info(self, name: str) -> Optional[Dict[str, Any]]:
        """Get metadata about a skill."""
        skill = self._skills.get(name)
        if not skill:
            return None

        return {
            "name": skill.name,
            "description": skill.description,
            "tools": [t[0] for t in skill.tools],
            "procedures": [p[0] for p in skill.procedures],
            **self._metadata.get(name, {}),
        }

    def execute(self, name: str, context: Dict[str, Any]) -> SkillResult:
        """
        Execute a skill synchronously.

        Args:
            name: Skill name.
            context: Execution context.

        Returns:
            SkillResult.

        Raises:
            ValueError: If skill not found.
        """
        skill = self._skills.get(name)
        if not skill:
            raise ValueError(f"Skill '{name}' not found")

        return skill.execute(context)

    async def execute_async(
        self,
        name: str,
        context: Dict[str, Any],
    ) -> SkillResult:
        """
        Execute a skill asynchronously.

        Args:
            name: Skill name.
            context: Execution context.

        Returns:
            SkillResult.

        Raises:
            ValueError: If skill not found.
        """
        skill = self._skills.get(name)
        if not skill:
            raise ValueError(f"Skill '{name}' not found")

        return await skill.execute_async(context)

    async def execute_multiple(
        self,
        names: List[str],
        context: Dict[str, Any],
        parallel: bool = True,
    ) -> Dict[str, SkillResult]:
        """
        Execute multiple skills.

        Args:
            names: List of skill names.
            context: Shared execution context.
            parallel: Whether to execute in parallel.

        Returns:
            Dict mapping skill names to results.
        """
        results = {}

        if parallel:
            async def run_skill(name: str):
                return name, await self.execute_async(name, context)

            tasks = [run_skill(n) for n in names if n in self._skills]
            completed = await asyncio.gather(*tasks, return_exceptions=True)

            for item in completed:
                if isinstance(item, Exception):
                    logger.error(f"Skill execution error: {item}")
                else:
                    name, result = item
                    results[name] = result
        else:
            for name in names:
                if name in self._skills:
                    results[name] = await self.execute_async(name, context)

        return results


# =============================================================================
# Built-in Skills
# =============================================================================

class ResearchAnalysisSkill(Skill):
    """
    Skill for analyzing research findings.

    Tools:
    - extract_key_findings: Extract main findings from content
    - identify_themes: Identify recurring themes
    - assess_confidence: Assess confidence levels

    Procedures:
    - synthesize_insights: Combine findings into insights
    """

    def __init__(self, config: Optional[SkillConfig] = None):
        """Initialize research analysis skill."""
        super().__init__(
            name="research_analysis",
            description="Analyze research findings for patterns and insights",
            config=config,
        )

        # Add tools
        self.add_tool(self._extract_key_findings, "extract_key_findings")
        self.add_tool(self._identify_themes, "identify_themes")
        self.add_tool(self._assess_confidence, "assess_confidence")

        # Add procedures
        self.add_procedure(self._synthesize_insights, "synthesize_insights")

    def _extract_key_findings(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key findings from research content."""
        content = context.get("compressed_context", context.get("content", ""))
        query = context.get("query", "")

        # Simple extraction based on patterns
        findings = []

        # Look for finding patterns
        patterns = [
            r"(?:found|discovered|revealed|showed|demonstrated)\s+that\s+([^.]+)",
            r"(?:results|findings|data)\s+(?:indicate|suggest|show)\s+([^.]+)",
            r"(?:key|main|important)\s+finding[s]?\s*:?\s*([^.]+)",
        ]

        for pattern in patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            findings.extend(matches[:3])  # Limit per pattern

        # Deduplicate and clean
        unique_findings = list(set(f.strip() for f in findings if len(f) > 20))

        return {
            "findings": unique_findings[:10],
            "count": len(unique_findings),
            "query_relevance": query.lower() in content.lower() if query else False,
        }

    def _identify_themes(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Identify recurring themes in content."""
        content = context.get("compressed_context", context.get("content", ""))

        # Common research theme keywords
        theme_keywords = {
            "technology": ["AI", "machine learning", "algorithm", "digital", "software"],
            "healthcare": ["medical", "patient", "health", "clinical", "treatment"],
            "business": ["market", "revenue", "growth", "strategy", "competitive"],
            "science": ["research", "study", "experiment", "data", "evidence"],
            "policy": ["regulation", "policy", "government", "compliance", "law"],
        }

        content_lower = content.lower()
        themes = {}

        for theme, keywords in theme_keywords.items():
            matches = sum(1 for kw in keywords if kw.lower() in content_lower)
            if matches > 0:
                themes[theme] = matches

        # Sort by frequency
        sorted_themes = sorted(themes.items(), key=lambda x: x[1], reverse=True)

        return {
            "themes": [t[0] for t in sorted_themes[:5]],
            "theme_scores": dict(sorted_themes[:5]),
            "dominant_theme": sorted_themes[0][0] if sorted_themes else None,
        }

    def _assess_confidence(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Assess confidence level of findings."""
        content = context.get("compressed_context", context.get("content", ""))

        # Confidence indicators
        high_confidence = ["proven", "demonstrated", "confirmed", "established", "significant"]
        medium_confidence = ["suggests", "indicates", "appears", "likely", "probable"]
        low_confidence = ["may", "might", "could", "possibly", "uncertain"]

        content_lower = content.lower()

        high_count = sum(1 for w in high_confidence if w in content_lower)
        medium_count = sum(1 for w in medium_confidence if w in content_lower)
        low_count = sum(1 for w in low_confidence if w in content_lower)

        total = high_count + medium_count + low_count
        if total == 0:
            confidence_score = 0.5
        else:
            confidence_score = (high_count * 1.0 + medium_count * 0.5 + low_count * 0.2) / total

        return {
            "confidence_score": round(confidence_score, 2),
            "confidence_level": (
                "high" if confidence_score > 0.7
                else "medium" if confidence_score > 0.4
                else "low"
            ),
            "indicators": {
                "high": high_count,
                "medium": medium_count,
                "low": low_count,
            },
        }

    def _synthesize_insights(
        self,
        context: Dict[str, Any],
        tool_results: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Synthesize findings into actionable insights."""
        findings = tool_results.get("extract_key_findings", {})
        themes = tool_results.get("identify_themes", {})
        confidence = tool_results.get("assess_confidence", {})

        insights = []

        # Generate insights based on findings and themes
        dominant_theme = themes.get("dominant_theme")
        if dominant_theme and findings.get("findings"):
            insights.append(
                f"Research primarily focuses on {dominant_theme} "
                f"with {len(findings['findings'])} key findings identified."
            )

        if confidence.get("confidence_level") == "high":
            insights.append("Findings show high confidence level with strong evidence.")
        elif confidence.get("confidence_level") == "low":
            insights.append("Findings show lower confidence - further research recommended.")

        return {
            "insights": insights,
            "summary": {
                "finding_count": findings.get("count", 0),
                "dominant_theme": dominant_theme,
                "confidence": confidence.get("confidence_level", "unknown"),
            },
        }


class SummarySkill(Skill):
    """
    Skill for generating structured summaries.

    Tools:
    - extract_sections: Identify document sections
    - calculate_statistics: Calculate content statistics

    Procedures:
    - generate_summary: Create structured summary
    """

    def __init__(self, config: Optional[SkillConfig] = None):
        """Initialize summary skill."""
        super().__init__(
            name="summary",
            description="Generate structured summaries of content",
            config=config,
        )

        self.add_tool(self._extract_sections, "extract_sections")
        self.add_tool(self._calculate_statistics, "calculate_statistics")
        self.add_procedure(self._generate_summary, "generate_summary")

    def _extract_sections(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract logical sections from content."""
        content = context.get("compressed_context", context.get("content", ""))

        # Split by paragraph
        paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]

        sections = []
        for i, para in enumerate(paragraphs[:10]):  # Limit to 10
            # Get first sentence as section title
            sentences = para.split(".")
            title = sentences[0][:100] if sentences else f"Section {i+1}"

            sections.append({
                "index": i,
                "title": title,
                "length": len(para),
                "preview": para[:200] + "..." if len(para) > 200 else para,
            })

        return {
            "sections": sections,
            "section_count": len(sections),
        }

    def _calculate_statistics(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate content statistics."""
        content = context.get("compressed_context", context.get("content", ""))

        words = content.split()
        sentences = re.split(r'[.!?]+', content)
        paragraphs = [p for p in content.split("\n\n") if p.strip()]

        return {
            "word_count": len(words),
            "sentence_count": len([s for s in sentences if s.strip()]),
            "paragraph_count": len(paragraphs),
            "avg_sentence_length": len(words) / max(len(sentences), 1),
            "character_count": len(content),
        }

    def _generate_summary(
        self,
        context: Dict[str, Any],
        tool_results: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Generate structured summary."""
        sections = tool_results.get("extract_sections", {})
        stats = tool_results.get("calculate_statistics", {})
        query = context.get("query", "Content Analysis")

        # Build summary
        summary_parts = [
            f"## Summary: {query}",
            "",
            f"**Content Statistics:**",
            f"- Words: {stats.get('word_count', 0):,}",
            f"- Sentences: {stats.get('sentence_count', 0):,}",
            f"- Paragraphs: {stats.get('paragraph_count', 0):,}",
            "",
            "**Key Sections:**",
        ]

        for section in sections.get("sections", [])[:5]:
            summary_parts.append(f"- {section['title']}")

        return {
            "summary_text": "\n".join(summary_parts),
            "statistics": stats,
            "section_titles": [s["title"] for s in sections.get("sections", [])],
        }


class EntityExtractionSkill(Skill):
    """
    Skill for extracting named entities.

    Tools:
    - extract_entities: Find named entities in text
    - categorize_entities: Categorize by type

    Procedures:
    - build_entity_graph: Create entity relationships
    """

    def __init__(self, config: Optional[SkillConfig] = None):
        """Initialize entity extraction skill."""
        super().__init__(
            name="entity_extraction",
            description="Extract and categorize named entities",
            config=config,
        )

        self.add_tool(self._extract_entities, "extract_entities")
        self.add_tool(self._categorize_entities, "categorize_entities")
        self.add_procedure(self._build_entity_graph, "build_entity_graph")

    def _extract_entities(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract named entities using pattern matching."""
        content = context.get("compressed_context", context.get("content", ""))

        entities = {
            "organizations": [],
            "people": [],
            "locations": [],
            "dates": [],
            "money": [],
            "percentages": [],
        }

        # Organization patterns (capitalized multi-word)
        org_pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b'
        entities["organizations"] = list(set(re.findall(org_pattern, content)))[:20]

        # Date patterns
        date_pattern = r'\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}|\w+\s+\d{1,2},?\s+\d{4})\b'
        entities["dates"] = list(set(re.findall(date_pattern, content)))[:10]

        # Money patterns
        money_pattern = r'\$[\d,]+(?:\.\d{2})?(?:\s*(?:million|billion|trillion))?'
        entities["money"] = list(set(re.findall(money_pattern, content, re.IGNORECASE)))[:10]

        # Percentage patterns
        pct_pattern = r'\d+(?:\.\d+)?%'
        entities["percentages"] = list(set(re.findall(pct_pattern, content)))[:10]

        return {
            "entities": entities,
            "total_count": sum(len(v) for v in entities.values()),
        }

    def _categorize_entities(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Categorize entities by domain."""
        content = context.get("compressed_context", context.get("content", ""))

        # Domain-specific entity patterns
        domains = {
            "technology": ["AI", "ML", "API", "software", "algorithm", "data"],
            "medical": ["patient", "treatment", "diagnosis", "clinical", "drug"],
            "financial": ["revenue", "profit", "market", "stock", "investment"],
        }

        content_lower = content.lower()
        domain_entities = {}

        for domain, keywords in domains.items():
            matches = [kw for kw in keywords if kw.lower() in content_lower]
            if matches:
                domain_entities[domain] = matches

        return {
            "domain_entities": domain_entities,
            "primary_domain": max(
                domain_entities.keys(),
                key=lambda k: len(domain_entities[k]),
                default=None
            ) if domain_entities else None,
        }

    def _build_entity_graph(
        self,
        context: Dict[str, Any],
        tool_results: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Build entity relationship summary."""
        entities = tool_results.get("extract_entities", {})
        categories = tool_results.get("categorize_entities", {})

        return {
            "entity_summary": {
                "total_entities": entities.get("total_count", 0),
                "by_type": {
                    k: len(v)
                    for k, v in entities.get("entities", {}).items()
                    if v
                },
                "primary_domain": categories.get("primary_domain"),
            },
            "key_entities": {
                k: v[:5]
                for k, v in entities.get("entities", {}).items()
                if v
            },
        }


class SentimentAnalysisSkill(Skill):
    """
    Skill for analyzing sentiment in content.

    Tools:
    - analyze_sentiment: Determine overall sentiment
    - extract_opinions: Find opinion statements

    Procedures:
    - generate_sentiment_report: Create sentiment summary
    """

    def __init__(self, config: Optional[SkillConfig] = None):
        """Initialize sentiment analysis skill."""
        super().__init__(
            name="sentiment_analysis",
            description="Analyze sentiment and opinions in content",
            config=config,
        )

        self.add_tool(self._analyze_sentiment, "analyze_sentiment")
        self.add_tool(self._extract_opinions, "extract_opinions")
        self.add_procedure(self._generate_sentiment_report, "generate_sentiment_report")

    def _analyze_sentiment(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze overall sentiment."""
        content = context.get("compressed_context", context.get("content", ""))
        content_lower = content.lower()

        # Sentiment word lists
        positive = [
            "good", "great", "excellent", "positive", "success", "improve",
            "benefit", "advantage", "effective", "innovative", "promising",
            "growth", "increase", "better", "best", "significant",
        ]
        negative = [
            "bad", "poor", "negative", "failure", "decline", "problem",
            "risk", "challenge", "difficult", "concern", "decrease",
            "worse", "worst", "limited", "issue", "threat",
        ]
        neutral = [
            "however", "although", "while", "but", "despite", "nonetheless",
        ]

        pos_count = sum(1 for w in positive if w in content_lower)
        neg_count = sum(1 for w in negative if w in content_lower)
        neu_count = sum(1 for w in neutral if w in content_lower)

        total = pos_count + neg_count + neu_count
        if total == 0:
            sentiment_score = 0.0
        else:
            sentiment_score = (pos_count - neg_count) / total

        return {
            "sentiment_score": round(sentiment_score, 2),
            "sentiment_label": (
                "positive" if sentiment_score > 0.2
                else "negative" if sentiment_score < -0.2
                else "neutral"
            ),
            "indicator_counts": {
                "positive": pos_count,
                "negative": neg_count,
                "neutral": neu_count,
            },
        }

    def _extract_opinions(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract opinion statements."""
        content = context.get("compressed_context", context.get("content", ""))

        # Opinion patterns
        patterns = [
            r"(?:I|we)\s+(?:believe|think|feel|consider)\s+([^.]+)",
            r"(?:it\s+is|this\s+is)\s+(?:clear|evident|obvious)\s+that\s+([^.]+)",
            r"(?:should|must|need\s+to)\s+([^.]+)",
        ]

        opinions = []
        for pattern in patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            opinions.extend(matches[:3])

        return {
            "opinions": opinions[:10],
            "opinion_count": len(opinions),
        }

    def _generate_sentiment_report(
        self,
        context: Dict[str, Any],
        tool_results: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Generate sentiment analysis report."""
        sentiment = tool_results.get("analyze_sentiment", {})
        opinions = tool_results.get("extract_opinions", {})

        report_lines = [
            f"## Sentiment Analysis Report",
            "",
            f"**Overall Sentiment:** {sentiment.get('sentiment_label', 'unknown').title()}",
            f"**Sentiment Score:** {sentiment.get('sentiment_score', 0):.2f} (-1 to +1)",
            "",
            f"**Indicator Distribution:**",
            f"- Positive indicators: {sentiment.get('indicator_counts', {}).get('positive', 0)}",
            f"- Negative indicators: {sentiment.get('indicator_counts', {}).get('negative', 0)}",
            f"- Neutral markers: {sentiment.get('indicator_counts', {}).get('neutral', 0)}",
            "",
            f"**Opinion Statements Found:** {opinions.get('opinion_count', 0)}",
        ]

        return {
            "report": "\n".join(report_lines),
            "summary": {
                "sentiment": sentiment.get("sentiment_label"),
                "score": sentiment.get("sentiment_score"),
                "opinions_found": opinions.get("opinion_count", 0),
            },
        }


# =============================================================================
# Convenience Functions
# =============================================================================

def create_skill(
    name: str,
    description: str = "",
    tools: Optional[List[ToolFunc]] = None,
    procedures: Optional[List[ProcedureFunc]] = None,
    config: Optional[SkillConfig] = None,
) -> Skill:
    """
    Factory function to create a skill.

    Args:
        name: Skill name.
        description: Skill description.
        tools: List of tool functions.
        procedures: List of procedure functions.
        config: Skill configuration.

    Returns:
        Configured Skill instance.

    Example:
        >>> skill = create_skill(
        ...     name="analysis",
        ...     tools=[my_tool1, my_tool2],
        ...     procedures=[my_procedure],
        ... )
    """
    skill = Skill(name=name, description=description, config=config)

    for tool in (tools or []):
        skill.add_tool(tool)

    for procedure in (procedures or []):
        skill.add_procedure(procedure)

    return skill


def get_default_registry() -> SkillRegistry:
    """
    Get a skill registry with default skills.

    Returns:
        SkillRegistry with built-in skills loaded.

    Example:
        >>> registry = get_default_registry()
        >>> result = await registry.execute("research_analysis", context)
    """
    return SkillRegistry(load_defaults=True)


async def execute_skill(
    skill_name: str,
    context: Dict[str, Any],
    registry: Optional[SkillRegistry] = None,
) -> SkillResult:
    """
    Execute a skill by name.

    Args:
        skill_name: Name of skill to execute.
        context: Execution context.
        registry: Optional registry (uses default if not provided).

    Returns:
        SkillResult.

    Example:
        >>> result = await execute_skill("research_analysis", {"content": text})
    """
    if registry is None:
        registry = get_default_registry()

    return await registry.execute_async(skill_name, context)

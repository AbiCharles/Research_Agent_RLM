"""
Citation Agent for the Multi-Agent Research Framework.

This module provides an agent specialized in adding proper source attribution
to research reports. It processes documents, adds inline citations [1][2],
and creates formatted source lists.
"""

import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

from core.base_agent import (
    BaseAgent,
    AgentConfig,
    AgentRole,
    AgentResult,
)
from core.openai_client import OpenAIClient
from config.settings import settings
from utils.logger import get_logger


logger = get_logger(__name__)


@dataclass
class Citation:
    """A single citation reference."""

    index: int
    title: str
    url: Optional[str] = None
    author: Optional[str] = None
    date: Optional[str] = None
    accessed: Optional[str] = None
    snippet: Optional[str] = None

    def to_inline(self) -> str:
        """Generate inline citation marker."""
        return f"[{self.index}]"

    def to_reference(self, style: str = "apa") -> str:
        """
        Generate formatted reference entry.

        Args:
            style: Citation style ('apa', 'mla', 'chicago', 'simple')

        Returns:
            Formatted reference string
        """
        if style == "simple":
            parts = [f"[{self.index}]", self.title]
            if self.url:
                parts.append(f"({self.url})")
            return " ".join(parts)

        elif style == "apa":
            # APA-like format
            parts = []
            if self.author:
                parts.append(f"{self.author}.")
            if self.date:
                parts.append(f"({self.date}).")
            parts.append(f"{self.title}.")
            if self.url:
                parts.append(f"Retrieved from {self.url}")
            return f"[{self.index}] " + " ".join(parts)

        elif style == "mla":
            # MLA-like format
            parts = []
            if self.author:
                parts.append(f"{self.author}.")
            parts.append(f'"{self.title}."')
            if self.url:
                parts.append(f"Web. {self.accessed or 'n.d.'}")
            return f"[{self.index}] " + " ".join(parts)

        else:  # chicago or default
            parts = [f"[{self.index}]", self.title]
            if self.author:
                parts.insert(1, f"{self.author},")
            if self.url:
                parts.append(f"<{self.url}>")
            return " ".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "title": self.title,
            "url": self.url,
            "author": self.author,
            "date": self.date,
            "accessed": self.accessed,
            "snippet": self.snippet,
        }


@dataclass
class CitationAgentConfig(AgentConfig):
    """Configuration for the Citation Agent."""

    # Citation settings
    citation_style: str = "simple"  # simple, apa, mla, chicago
    max_citations: int = 50
    include_snippets: bool = False
    auto_number: bool = True

    def __post_init__(self):
        super().__post_init__()
        self.role = AgentRole.CITATION_AGENT


@dataclass
class CitationResult:
    """Result from citation processing."""

    original_content: str
    cited_content: str
    citations: List[Citation]
    reference_list: str
    stats: Dict[str, int]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_length": len(self.original_content),
            "cited_length": len(self.cited_content),
            "citation_count": len(self.citations),
            "reference_list": self.reference_list,
            "stats": self.stats,
        }


class CitationAgent(BaseAgent):
    """
    Agent specialized in adding citations to research content.

    The Citation Agent:
    1. Identifies factual claims in research content
    2. Matches claims to available sources
    3. Adds inline citations [1], [2], etc.
    4. Generates a formatted reference list

    Example:
        >>> agent = CitationAgent(CitationAgentConfig(name="Citation Agent"))
        >>> result = await agent.process_content(
        ...     content="AI is transforming healthcare...",
        ...     sources=[{"title": "AI in Medicine", "url": "..."}]
        ... )
        >>> print(result.cited_content)  # Content with [1], [2] markers
        >>> print(result.reference_list)  # Formatted source list
    """

    def __init__(
        self,
        config: CitationAgentConfig,
        client: Optional[OpenAIClient] = None,
    ):
        """
        Initialize the Citation Agent.

        Args:
            config: Citation agent configuration
            client: OpenAI client
        """
        self.citation_config = config
        super().__init__(config, client)

        # Citation state
        self._citations: List[Citation] = []
        self._source_map: Dict[str, int] = {}  # title -> citation index

    def _build_system_prompt(self) -> str:
        """Build the Citation Agent's system prompt."""
        return f"""You are {self.name}, a Citation Agent specialized in academic source attribution.

## Your Role
You add proper citations to research content by:
1. Identifying factual claims that need source attribution
2. Matching claims to provided sources
3. Adding inline citation markers [1], [2], etc.
4. Ensuring all cited sources are properly referenced

## Guidelines
- Only cite sources that directly support the claim
- Use the exact source index provided
- Don't cite opinions or general knowledge
- Maintain the original meaning while adding citations
- Place citations at the end of the relevant sentence or clause

## Citation Style: {self.citation_config.citation_style}

When adding citations:
- Place citation markers after punctuation: "...healthcare. [1]"
- Multiple sources for one claim: "...diagnosis [1][3]"
- Don't cite every sentence - only factual claims
"""

    async def process_content(
        self,
        content: str,
        sources: List[Dict[str, Any]],
        context: Optional[str] = None,
    ) -> CitationResult:
        """
        Process content and add citations.

        Args:
            content: Research content to cite
            sources: List of source dictionaries
            context: Optional context about the research

        Returns:
            CitationResult with cited content and references
        """
        logger.info(f"Processing content for citations: {len(content)} chars, {len(sources)} sources")

        # Reset state
        self._citations = []
        self._source_map = {}

        # Create citation objects from sources
        for i, source in enumerate(sources[: self.citation_config.max_citations]):
            citation = Citation(
                index=i + 1,
                title=source.get("title", f"Source {i+1}"),
                url=source.get("url"),
                author=source.get("author"),
                date=source.get("date"),
                accessed=datetime.now().strftime("%Y-%m-%d"),
                snippet=source.get("snippet"),
            )
            self._citations.append(citation)
            self._source_map[citation.title.lower()] = citation.index

        # Use LLM to intelligently add citations
        cited_content = await self._add_citations_with_llm(content, sources)

        # Generate reference list
        reference_list = self._generate_reference_list()

        # Calculate stats
        citation_count = len(re.findall(r'\[\d+\]', cited_content))
        unique_citations = len(set(re.findall(r'\[(\d+)\]', cited_content)))

        stats = {
            "total_sources": len(sources),
            "citations_added": citation_count,
            "unique_sources_cited": unique_citations,
            "content_length_original": len(content),
            "content_length_cited": len(cited_content),
        }

        return CitationResult(
            original_content=content,
            cited_content=cited_content,
            citations=self._citations,
            reference_list=reference_list,
            stats=stats,
        )

    async def _add_citations_with_llm(
        self,
        content: str,
        sources: List[Dict[str, Any]],
    ) -> str:
        """
        Use LLM to intelligently add citations to content.

        Args:
            content: Original content
            sources: Available sources

        Returns:
            Content with inline citations added
        """
        # Build source reference for the LLM
        source_list = "\n".join(
            f"[{i+1}] {s.get('title', 'Unknown')} - {s.get('snippet', s.get('url', 'No description'))[:100]}"
            for i, s in enumerate(sources[:20])
        )

        citation_prompt = f"""Add inline citations to the following research content.

## Available Sources:
{source_list}

## Content to Cite:
{content}

## Instructions:
1. Identify factual claims that should be attributed to a source
2. Add citation markers [1], [2], etc. after relevant claims
3. Only cite sources that directly support the claim
4. Keep the original text intact, just add citation markers
5. Return ONLY the cited content, no explanations

## Output:
Return the content with citations added inline."""

        response = await self.chat(citation_prompt)

        # Clean up the response
        cited_content = response.content.strip()

        # Validate citations reference existing sources
        cited_content = self._validate_citations(cited_content, len(sources))

        return cited_content

    def _validate_citations(self, content: str, max_source: int) -> str:
        """
        Validate that all citations reference existing sources.

        Args:
            content: Content with citations
            max_source: Maximum valid source number

        Returns:
            Content with invalid citations removed
        """
        def replace_invalid(match):
            num = int(match.group(1))
            if 1 <= num <= max_source:
                return match.group(0)
            return ""

        return re.sub(r'\[(\d+)\]', replace_invalid, content)

    def _generate_reference_list(self) -> str:
        """
        Generate formatted reference list.

        Returns:
            Formatted reference list string
        """
        if not self._citations:
            return ""

        lines = ["## References", ""]
        for citation in self._citations:
            lines.append(citation.to_reference(self.citation_config.citation_style))

        return "\n".join(lines)

    async def add_citations_to_report(
        self,
        report: str,
        agent_results: List[Any],
    ) -> Tuple[str, str]:
        """
        Add citations to a complete research report.

        Args:
            report: The research report content
            agent_results: Results from research agents (with sources)

        Returns:
            Tuple of (cited_report, reference_list)
        """
        # Collect all sources from agent results
        all_sources = []
        for result in agent_results:
            if hasattr(result, 'sources'):
                all_sources.extend(result.sources)

        # Deduplicate sources
        seen_titles = set()
        unique_sources = []
        for source in all_sources:
            title = source.get("title", "").lower()
            if title and title not in seen_titles:
                seen_titles.add(title)
                unique_sources.append(source)

        if not unique_sources:
            return report, ""

        # Process the report
        result = await self.process_content(report, unique_sources)

        return result.cited_content, result.reference_list

    async def _execute(
        self,
        hypothesis: str,
        research_questions: Optional[List[str]],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Execute method required by BaseAgent.

        For CitationAgent, this processes content from context.
        """
        content = context.get("content", "")
        sources = context.get("sources", [])

        if not content:
            return {"content": "", "error": "No content provided for citation"}

        result = await self.process_content(content, sources)

        return {
            "content": result.cited_content,
            "reference_list": result.reference_list,
            "stats": result.stats,
        }

    def quick_cite(
        self,
        text: str,
        source_index: int,
    ) -> str:
        """
        Quickly add a citation to text.

        Args:
            text: Text to cite
            source_index: Source index to cite

        Returns:
            Text with citation added
        """
        # Add citation at end of sentence or text
        if text.endswith(('.', '!', '?')):
            return text[:-1] + f" [{source_index}]" + text[-1]
        return text + f" [{source_index}]"

    def format_sources_as_footnotes(
        self,
        sources: List[Dict[str, Any]],
    ) -> str:
        """
        Format sources as footnotes for a document.

        Args:
            sources: List of source dictionaries

        Returns:
            Formatted footnotes string
        """
        lines = []
        for i, source in enumerate(sources, 1):
            title = source.get("title", "Unknown")
            url = source.get("url", "")
            lines.append(f"^{i}^ {title}" + (f" - {url}" if url else ""))

        return "\n".join(lines)


def create_citation_agent(
    name: str = "Citation Agent",
    citation_style: str = "simple",
    **kwargs,
) -> CitationAgent:
    """
    Factory function to create a Citation Agent.

    Args:
        name: Agent name
        citation_style: Citation style (simple, apa, mla, chicago)
        **kwargs: Additional config options

    Returns:
        Configured CitationAgent instance
    """
    config = CitationAgentConfig(
        name=name,
        citation_style=citation_style,
        model=settings.DEFAULT_MODEL,
        **kwargs,
    )
    return CitationAgent(config)

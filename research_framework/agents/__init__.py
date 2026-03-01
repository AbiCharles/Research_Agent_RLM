"""
Agents module for the Multi-Agent Research Framework.

This module exports all agent types used in research orchestration.
"""

from agents.research_agent import (
    ResearchAgent,
    ResearchAgentConfig,
    create_research_agent,
)
from agents.lead_researcher import (
    LeadResearcher,
    LeadResearcherConfig,
    ResearchPlan,
    OrchestrationResult,
    QueryComplexity,
    create_lead_researcher,
)
from agents.citation_agent import (
    CitationAgent,
    CitationAgentConfig,
    Citation,
    CitationResult,
    create_citation_agent,
)

__all__ = [
    # Research Agent
    "ResearchAgent",
    "ResearchAgentConfig",
    "create_research_agent",
    # Lead Researcher
    "LeadResearcher",
    "LeadResearcherConfig",
    "ResearchPlan",
    "OrchestrationResult",
    "QueryComplexity",
    "create_lead_researcher",
    # Citation Agent
    "CitationAgent",
    "CitationAgentConfig",
    "Citation",
    "CitationResult",
    "create_citation_agent",
]

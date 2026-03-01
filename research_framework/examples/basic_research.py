"""
Basic research example using the Multi-Agent Research Framework.

This example demonstrates how to:
1. Create a research agent
2. Run a research investigation
3. Access findings and results

Usage:
    python examples/basic_research.py

Prerequisites:
    - Set OPENAI_API_KEY in .env file or environment
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.research_agent import ResearchAgent, ResearchAgentConfig, create_research_agent
from core import get_client
from config import settings


async def basic_example():
    """Run a basic research example with a single agent."""
    print("=" * 60)
    print("Multi-Agent Research Framework - Basic Example")
    print("=" * 60)

    # Check for API key
    if not settings.OPENAI_API_KEY:
        print("\nError: OPENAI_API_KEY not set!")
        print("Please set it in your .env file or environment.")
        return

    # Create a research agent
    print("\n1. Creating research agent...")
    agent = create_research_agent(
        name="Technology Analyst",
        focus="emerging technology trends and their business impact",
        model=settings.DEFAULT_MODEL,
    )
    print(f"   Agent created: {agent.name} (ID: {agent.agent_id})")

    # Define research hypothesis
    hypothesis = "Artificial intelligence will significantly transform the healthcare industry within the next decade"

    research_questions = [
        "What are the current AI applications in healthcare?",
        "What are the main barriers to AI adoption in healthcare?",
        "What potential benefits could AI bring to patient outcomes?",
    ]

    print(f"\n2. Research Hypothesis:")
    print(f"   '{hypothesis}'")
    print(f"\n   Questions to investigate:")
    for i, q in enumerate(research_questions, 1):
        print(f"   {i}. {q}")

    # Run research
    print("\n3. Running research...")
    print("   (This may take a minute...)\n")

    result = await agent.run(
        hypothesis=hypothesis,
        research_questions=research_questions,
    )

    # Display results
    print("=" * 60)
    print("RESEARCH RESULTS")
    print("=" * 60)

    print(f"\nStatus: {result.status.value}")
    print(f"Duration: {result.duration_seconds:.1f} seconds")
    print(f"Findings: {len(result.findings)}")
    print(f"Sources: {len(result.sources)}")

    if result.error:
        print(f"\nError: {result.error}")
    else:
        print("\n--- Summary ---")
        print(result.content[:1000] + "..." if len(result.content) > 1000 else result.content)

        if result.findings:
            print("\n--- Key Findings ---")
            for i, finding in enumerate(result.findings[:5], 1):
                print(f"\n{i}. {finding.get('summary', 'No summary')[:150]}...")
                print(f"   Confidence: {finding.get('confidence', 'N/A')}")

    # Show usage stats
    print("\n--- API Usage ---")
    client = get_client()
    usage = client.get_usage_summary()
    print(f"Total tokens: {usage['total_tokens']}")
    print(f"Estimated cost: ${usage['total_cost_usd']:.4f}")


async def multi_agent_example():
    """Example with multiple agents researching in parallel."""
    print("\n" + "=" * 60)
    print("Multi-Agent Example (Parallel Research)")
    print("=" * 60)

    if not settings.OPENAI_API_KEY:
        print("\nError: OPENAI_API_KEY not set!")
        return

    # Create multiple specialized agents
    agents = [
        create_research_agent(
            name="Market Analyst",
            focus="market trends, competitive landscape, and business models",
        ),
        create_research_agent(
            name="Technical Expert",
            focus="technical capabilities, limitations, and implementation challenges",
        ),
        create_research_agent(
            name="Policy Researcher",
            focus="regulatory environment, ethical considerations, and policy implications",
        ),
    ]

    hypothesis = "Electric vehicles will dominate the automotive market by 2035"

    print(f"\nHypothesis: '{hypothesis}'")
    print(f"\nAgents: {[a.name for a in agents]}")
    print("\nRunning parallel research...")

    # Run all agents in parallel
    tasks = [
        agent.run(hypothesis=hypothesis)
        for agent in agents
    ]

    results = await asyncio.gather(*tasks)

    # Display results from each agent
    for agent, result in zip(agents, results):
        print(f"\n--- {agent.name} ---")
        print(f"Status: {result.status.value}")
        print(f"Findings: {len(result.findings)}")
        if result.content:
            print(f"Summary: {result.content[:300]}...")


if __name__ == "__main__":
    print("\nRunning basic example...\n")
    asyncio.run(basic_example())

    # Uncomment to run multi-agent example:
    # asyncio.run(multi_agent_example())

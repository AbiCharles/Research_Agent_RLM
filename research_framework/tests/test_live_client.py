"""
Live tests for OpenAI client and RLM Memory Manager.

These tests make REAL API calls and cost money.
Run only when you want to verify actual functionality.

Usage:
    # Set your API key first
    export OPENAI_API_KEY="sk-your-key-here"

    # Or create .env file in research_framework directory
    echo "OPENAI_API_KEY=sk-your-key-here" > .env

    # Run tests
    cd research_framework
    python -m pytest tests/test_live_client.py -v

    # Or run directly
    python tests/test_live_client.py
"""

import asyncio
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import settings
from core.openai_client import OpenAIClient, get_client
from core.memory_manager import (
    MemoryManager,
    MemoryConfig,
    REPLEnvironment,
    AsyncLLMQueryPool,
)
from tools import WebSearchTool, web_search
from agents.research_agent import ResearchAgent, ResearchAgentConfig, create_research_agent


def check_api_key():
    """Verify API key is configured."""
    if not settings.OPENAI_API_KEY:
        print("ERROR: OPENAI_API_KEY not set!")
        print("\nSet it via:")
        print("  export OPENAI_API_KEY='sk-your-key-here'")
        print("  OR")
        print("  echo 'OPENAI_API_KEY=sk-your-key-here' > .env")
        return False
    print(f"✓ API key configured (ends with ...{settings.OPENAI_API_KEY[-4:]})")
    return True


async def test_basic_client():
    """Test 1: Basic OpenAI client chat completion."""
    print("\n" + "="*60)
    print("TEST 1: Basic Client Chat Completion")
    print("="*60)

    client = get_client()

    response = await client.chat_completion(
        messages=[{"role": "user", "content": "Say 'Hello, RLM!' and nothing else."}],
        model=settings.DEFAULT_MODEL,
        max_tokens=20,
    )

    print(f"Model: {settings.DEFAULT_MODEL}")
    print(f"Response: {response.content}")
    print(f"Tokens used: {response.usage.total_tokens}")

    assert "Hello" in response.content or "RLM" in response.content
    print("✓ PASSED")


async def test_async_query_pool():
    """Test 2: AsyncLLMQueryPool batch processing."""
    print("\n" + "="*60)
    print("TEST 2: Async Query Pool (Parallel Requests)")
    print("="*60)

    client = get_client()
    pool = AsyncLLMQueryPool(
        client=client,
        max_concurrent=3,
        requests_per_minute=60,
    )

    # Batch of 3 queries
    queries = [
        {"query": "What is 2+2? Reply with just the number.", "context": "Math question"},
        {"query": "What is the capital of France? Reply with just the city name.", "context": "Geography question"},
        {"query": "What color is the sky? Reply with just the color.", "context": "Simple question"},
    ]

    print(f"Sending {len(queries)} queries in parallel...")
    results = await pool.batch_query(queries, max_tokens=10)

    for i, result in enumerate(results):
        status = "✓" if result.success else "✗"
        print(f"  {status} Query {i+1}: {result.content.strip()[:30]}")

    stats = pool.get_stats()
    print(f"\nPool stats:")
    print(f"  Total queries: {stats['total_queries']}")
    print(f"  Successful: {stats['successful_queries']}")
    print(f"  Total tokens: {stats['total_tokens']}")

    assert all(r.success for r in results)
    print("✓ PASSED")


async def test_memory_manager_pipeline():
    """Test 3: Full RLM Pipeline processing."""
    print("\n" + "="*60)
    print("TEST 3: RLM Memory Manager Pipeline")
    print("="*60)

    client = get_client()

    config = MemoryConfig(
        max_concurrent_queries=3,
        selection_threshold=0.0,       # Allow all content through for testing
        target_selection_ratio=1.0,    # No token budget limit in selection
        compression_ratio=0.5,         # 50% compression for visible effect
        use_abstractive_compression=True,
        enable_selection=False,        # Skip selection, just test compression
    )

    manager = MemoryManager(config)
    manager.set_client(client)

    # Sample research content
    content = """
    Artificial Intelligence in Healthcare: A Comprehensive Overview

    AI is revolutionizing healthcare through improved diagnostics. Machine learning
    algorithms can now detect diseases from medical images with accuracy matching
    or exceeding human experts. For example, AI systems have shown remarkable
    success in identifying cancerous lesions in mammograms and CT scans.

    Drug discovery is another area where AI shows promise. Traditional drug
    development takes 10-15 years and costs billions. AI can analyze molecular
    structures and predict drug interactions, potentially reducing this timeline
    significantly.

    However, there are challenges. Data privacy concerns are paramount when
    dealing with sensitive medical records. Additionally, the "black box" nature
    of some AI models makes it difficult for clinicians to understand and trust
    their recommendations.

    The weather today is sunny with a chance of rain. This paragraph is completely
    irrelevant to healthcare AI and should be filtered out by the selection stage.

    Looking forward, personalized medicine powered by AI could tailor treatments
    to individual genetic profiles. This represents a paradigm shift from the
    current one-size-fits-all approach to healthcare.
    """

    query = "What is the impact of AI on healthcare diagnostics and drug discovery?"

    print(f"Query: {query[:60]}...")
    print(f"Content length: {len(content)} chars")

    # Process through pipeline
    result = await manager.process_through_pipeline(
        content=content,
        query=query,
        chunk_size=300,
    )

    print(f"\n--- Pipeline Results ---")
    print(f"Input stage: {result['input']['chunks']} chunks, {result['input']['tokens']} tokens")

    sel_meta = result['selection'].metadata
    if 'skipped' in sel_meta:
        print(f"Selection stage: SKIPPED")
    else:
        print(f"Selection stage: {sel_meta.get('chunks_before', '?')} → {sel_meta.get('chunks_after', '?')} chunks")
        print(f"  Reduction: {(1 - result['selection'].reduction_ratio) * 100:.1f}%")

    opt_meta = result['optimization'].metadata
    if 'skipped' in opt_meta:
        print(f"Optimization stage: SKIPPED")
    else:
        print(f"Optimization stage: {result['optimization'].original_tokens} → {result['optimization'].output_tokens} tokens")
        print(f"  Reduction: {(1 - result['optimization'].reduction_ratio) * 100:.1f}%")

    print(f"\nFinal: {result['final']['original_tokens']} → {result['final']['final_tokens']} tokens")
    print(f"Total reduction: {result['final']['total_reduction']}")

    print(f"\n--- Compressed Content Preview ---")
    print(result['final']['content'][:500] + "..." if len(result['final']['content']) > 500 else result['final']['content'])

    # Verify reduction happened
    assert result['final']['final_tokens'] < result['final']['original_tokens']
    print("\n✓ PASSED")


async def test_repl_environment():
    """Test 4: REPL Environment (no API calls)."""
    print("\n" + "="*60)
    print("TEST 4: REPL Environment (Local Only)")
    print("="*60)

    repl = REPLEnvironment()

    # Set context
    repl.set_context("findings", [
        {"topic": "diagnostics", "confidence": 0.9},
        {"topic": "drug discovery", "confidence": 0.75},
        {"topic": "privacy", "confidence": 0.6},
    ])

    # Execute code
    result, error = repl.execute("len(findings)")
    print(f"len(findings) = {result}")
    assert result == 3

    result, error = repl.execute("[f['topic'] for f in findings if f['confidence'] > 0.7]")
    print(f"High confidence topics: {result}")
    assert result == ["diagnostics", "drug discovery"]

    # Test error handling
    result, error = repl.execute("open('/etc/passwd')")  # Should fail - open not allowed
    print(f"Blocked dangerous operation: {error is not None}")
    assert error is not None

    print("✓ PASSED")


async def test_token_counting():
    """Test 5: Token counting accuracy."""
    print("\n" + "="*60)
    print("TEST 5: Token Counting")
    print("="*60)

    manager = MemoryManager()

    test_cases = [
        ("Hello, world!", 4),  # Approximate
        ("The quick brown fox jumps over the lazy dog.", 10),  # Approximate
        ("AI", 1),
    ]

    for text, expected_approx in test_cases:
        tokens = manager.count_tokens(text)
        print(f"  '{text[:30]}...' → {tokens} tokens (expected ~{expected_approx})")
        # Allow some variance in token counts
        assert abs(tokens - expected_approx) <= 3

    print("✓ PASSED")


async def test_web_search():
    """Test 6: Web Search Tool."""
    print("\n" + "="*60)
    print("TEST 6: Web Search Tool")
    print("="*60)

    tool = WebSearchTool()
    print(f"Backend: {tool._backend}")

    # Execute search
    result = await tool.execute(
        query="artificial intelligence healthcare applications",
        max_results=3,
    )

    print(f"Status: {result.status.value}")
    print(f"Results: {len(result.data) if result.data else 0}")

    if result.success and result.data:
        for i, item in enumerate(result.data[:3], 1):
            print(f"  {i}. {item.get('title', 'N/A')[:50]}...")
            print(f"     URL: {item.get('url', 'N/A')[:60]}")
        print("✓ PASSED")
    elif result.status.value == "no_results":
        print("  No instant answers (DuckDuckGo limitation)")
        print("✓ PASSED (no results is acceptable for DDG)")
    else:
        print(f"  Error: {result.error}")
        print("⚠ SKIPPED (search backend unavailable)")


async def test_research_agent():
    """Test 7: Research Agent End-to-End."""
    print("\n" + "="*60)
    print("TEST 7: Research Agent (End-to-End)")
    print("="*60)

    # Create agent
    agent = create_research_agent(
        name="Test Analyst",
        focus="technology trends",
        model=settings.DEFAULT_MODEL,
    )
    print(f"Agent created: {agent.name}")

    # Simple research task
    hypothesis = "Artificial intelligence is improving medical diagnosis accuracy"
    questions = [
        "What are the main AI techniques used in medical diagnosis?",
    ]

    print(f"Hypothesis: {hypothesis[:50]}...")
    print(f"Questions: {len(questions)}")
    print("\nRunning research (this may take a moment)...")

    result = await agent.run(
        hypothesis=hypothesis,
        research_questions=questions,
    )

    print(f"\n--- Agent Results ---")
    print(f"Status: {result.status.value}")
    print(f"Findings: {len(result.findings)}")
    print(f"Sources: {len(result.sources)}")

    if result.content:
        print(f"\nContent preview:")
        print(result.content[:500] + "..." if len(result.content) > 500 else result.content)

    # Get stats
    stats = agent.get_research_stats()
    print(f"\nAgent stats:")
    print(f"  Iterations: {stats['iterations_completed']}")
    print(f"  Messages: {stats.get('message_count', 0)}")
    print(f"  Avg confidence: {stats.get('average_confidence', 0):.2f}")

    assert result.status.value in ["completed", "partial"]
    print("\n✓ PASSED")


async def run_all_tests():
    """Run all tests sequentially."""
    print("="*60)
    print("LIVE CLIENT TESTS")
    print("="*60)
    print(f"Model: {settings.DEFAULT_MODEL}")

    # Test that doesn't need API
    await test_repl_environment()
    await test_token_counting()

    # Test web search (may work without API key via DuckDuckGo)
    await test_web_search()

    # Tests that need API
    if check_api_key():
        await test_basic_client()
        await test_async_query_pool()
        await test_memory_manager_pipeline()
        await test_research_agent()

        # Print usage summary
        client = get_client()
        usage = client.get_usage_summary()
        print("\n" + "="*60)
        print("USAGE SUMMARY")
        print("="*60)
        print(f"Total tokens: {usage['total_tokens']}")
        print(f"  Prompt: {usage['total_prompt_tokens']}")
        print(f"  Completion: {usage['total_completion_tokens']}")
        print(f"Estimated cost: ${usage['total_cost_usd']:.4f}")
    else:
        print("\nSkipping API tests (no key configured)")

    print("\n" + "="*60)
    print("ALL TESTS COMPLETED")
    print("="*60)


if __name__ == "__main__":
    asyncio.run(run_all_tests())

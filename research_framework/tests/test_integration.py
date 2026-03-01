"""
Integration Tests - RLM Paradigm End-to-End Workflows.

These tests verify the RLM (Recursive Language Model) paradigm where LLMs
receive metadata and write code to access content, rather than receiving
raw documents directly.

Test Scenarios:
---------------
1. KB Foundation: Document ingestion and semantic search
2. RLM Context Access: KnowledgeEnvironment bridge with REPL execution
3. RLM Selection + Compression: MemoryManager pipeline processing
4. Skills Execution: Domain-specific analysis on RLM output
5. End-to-End RLM: Full flow from documents to insights via code execution

Usage:
    python -m pytest tests/test_integration.py -v
"""

import asyncio
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock

sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# Sample Research Data
# =============================================================================

SAMPLE_RESEARCH_DOCUMENTS = [
    {
        "title": "AI in Medical Diagnostics",
        "content": """
        Artificial intelligence is revolutionizing medical diagnostics. Recent studies
        demonstrate that machine learning algorithms can detect cancer in medical images
        with 94% accuracy, matching or exceeding human radiologists. Deep learning models
        have shown particular promise in analyzing X-rays, MRIs, and CT scans.

        Key findings from clinical trials indicate that AI-assisted diagnosis reduces
        diagnostic errors by 30% and speeds up the diagnostic process by 50%. However,
        challenges remain including data privacy concerns, regulatory approval processes,
        and the need for extensive validation across diverse patient populations.

        The global AI in healthcare market is projected to reach $45 billion by 2030,
        with diagnostic applications representing 35% of this market. Major healthcare
        systems including Mayo Clinic and Cleveland Clinic have begun integrating AI
        diagnostic tools into their workflows.
        """,
        "metadata": {"type": "research", "topic": "AI healthcare", "year": 2024}
    },
    {
        "title": "Machine Learning Drug Discovery",
        "content": """
        Machine learning is transforming pharmaceutical drug discovery. Traditional drug
        development takes 10-15 years and costs over $2 billion per approved drug. AI
        approaches are demonstrating the potential to reduce this timeline by 40% and
        costs by 60%.

        Recent breakthroughs include AlphaFold's protein structure predictions and
        generative AI models that can design novel molecular compounds. Pfizer reported
        using AI to accelerate COVID-19 vaccine development, reducing certain analysis
        phases from months to days.

        Key applications include target identification, lead optimization, toxicity
        prediction, and clinical trial design. However, experts caution that AI models
        require extensive validation and human oversight to ensure safety and efficacy.
        """,
        "metadata": {"type": "research", "topic": "drug discovery", "year": 2024}
    },
    {
        "title": "Natural Language Processing in Healthcare",
        "content": """
        Natural language processing (NLP) is enabling new capabilities in healthcare
        information management. Clinical NLP systems can extract structured data from
        unstructured medical notes, improving documentation efficiency by 40%.

        Applications include automated medical coding, clinical decision support, and
        patient communication chatbots. Studies show that NLP-powered systems can
        identify potential adverse drug interactions with 89% accuracy.

        Privacy-preserving NLP techniques are being developed to enable analysis of
        sensitive medical data while maintaining patient confidentiality. These
        approaches use federated learning and differential privacy.
        """,
        "metadata": {"type": "research", "topic": "NLP healthcare", "year": 2024}
    },
    {
        "title": "Weather Patterns 2024",
        "content": """
        Global weather patterns in 2024 showed significant variations from historical
        norms. The El Nino phenomenon contributed to above-average temperatures across
        much of the Pacific region. Precipitation levels varied considerably.

        Hurricane season was particularly active with 18 named storms. Drought conditions
        affected agricultural regions in multiple continents. Climate scientists continue
        to monitor these trends for long-term implications.
        """,
        "metadata": {"type": "news", "topic": "weather", "year": 2024}
    },
    {
        "title": "Sports Championship Results",
        "content": """
        The 2024 sports season saw numerous exciting championships across various leagues.
        Baseball, basketball, football, and soccer all crowned new champions. Fan
        attendance reached record levels following post-pandemic recovery.

        Notable achievements included several record-breaking performances and historic
        comebacks. Youth sports programs also saw increased participation rates.
        """,
        "metadata": {"type": "news", "topic": "sports", "year": 2024}
    },
]

RESEARCH_QUERY = "How is artificial intelligence improving medical diagnosis and drug discovery?"


# =============================================================================
# Test 1: KB Foundation
# =============================================================================

def test_kb_foundation():
    """
    Test KB document ingestion, semantic search, and persistence.
    This is the foundation that KnowledgeEnvironment builds on.
    """
    from filters import KnowledgeBase, KnowledgeBaseConfig

    with tempfile.TemporaryDirectory() as tmpdir:
        config = KnowledgeBaseConfig(
            name="healthcare_research",
            chunk_size=500,
            chunk_overlap=50,
            auto_save=True,
        )
        kb = KnowledgeBase(config=config, persist_directory=tmpdir)

        for doc in SAMPLE_RESEARCH_DOCUMENTS:
            kb.add_text(
                text=doc["content"],
                metadata={**doc["metadata"], "title": doc["title"]}
            )

        stats = kb.get_stats()
        assert stats.document_count >= 5, "Should have indexed documents"

        results = kb.query(query=RESEARCH_QUERY, top_k=10)
        assert len(results) > 0, "Should return search results"

        relevant_topics = {"AI healthcare", "drug discovery", "NLP healthcare"}
        top_5_topics = [r.get("metadata", {}).get("topic", "") for r in results[:5]]
        relevant_in_top_5 = sum(1 for t in top_5_topics if t in relevant_topics)
        assert relevant_in_top_5 >= 3, "Relevant docs should rank in top 5"

        # Test persistence (save + explicit load)
        kb.save(tmpdir)
        kb2 = KnowledgeBase(config=config, persist_directory=tmpdir)
        kb2.vector_store.load(tmpdir)
        stats2 = kb2.get_stats()
        assert stats2.document_count > 0, "Should persist and reload"


# =============================================================================
# Test 2: RLM Context Access via KnowledgeEnvironment
# =============================================================================

def test_rlm_context_access():
    """
    Test the RLM paradigm: KnowledgeEnvironment exposes KB as REPL functions,
    LLM receives metadata (not raw content), and writes code to access data.
    """
    from filters import KnowledgeBase, KnowledgeBaseConfig
    from core.memory_manager import REPLEnvironment
    from core.knowledge_environment import (
        KnowledgeEnvironment,
        KnowledgeEnvironmentConfig,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        # Build KB
        kb = KnowledgeBase(
            config=KnowledgeBaseConfig(name="rlm_test", chunk_size=500),
            persist_directory=tmpdir,
        )
        for doc in SAMPLE_RESEARCH_DOCUMENTS:
            kb.add_text(
                text=doc["content"],
                metadata={**doc["metadata"], "title": doc["title"]}
            )

        # Create RLM bridge
        repl = REPLEnvironment()
        config = KnowledgeEnvironmentConfig(
            include_topics=True,
            include_sources=True,
        )
        env = KnowledgeEnvironment(kb, repl, config=config)

        # Verify metadata (what the LLM would receive instead of raw content)
        metadata = env.get_metadata()
        assert metadata.document_count >= 5
        assert len(metadata.functions_available) >= 5
        assert "kb_search" in metadata.functions_available

        # Verify context prompt describes available resources
        prompt = env.get_context_prompt()
        assert "KNOWLEDGE BASE CONTEXT" in prompt
        assert "kb_search" in prompt
        assert "filter_by_score" in prompt

        # Simulate LLM writing code to search KB
        _, err = env.execute_code('results = kb_search("AI diagnosis", top_k=5)')
        assert err is None

        count_result, err = env.execute_code('len(results)')
        assert err is None
        assert count_result > 0

        # Simulate LLM filtering results by score
        _, err = env.execute_code('filtered = filter_by_score(results, 0.3)')
        assert err is None

        filtered_count, err = env.execute_code('len(filtered)')
        assert err is None
        assert filtered_count > 0

        # Simulate LLM accessing content from results
        content_result, err = env.execute_code('filtered[0]["content"][:100]')
        assert err is None
        assert len(content_result) > 0

        # Verify metadata is cached
        metadata2 = env.get_metadata()
        assert metadata2.document_count == metadata.document_count


# =============================================================================
# Test 3: MemoryManager Selection + Compression
# =============================================================================

def test_memory_manager_pipeline():
    """
    Test MemoryManager's internal selection and compression pipeline,
    which replaces the traditional Stage 2 and Stage 3.
    """
    from core.memory_manager import MemoryManager, MemoryConfig

    config = MemoryConfig(
        max_context_tokens=2000,
        enable_selection=False,
        compression_ratio=0.3,
    )
    manager = MemoryManager(config=config)

    # Add content (simulating what kb_search would return)
    healthcare_content = "\n\n".join([
        doc["content"] for doc in SAMPLE_RESEARCH_DOCUMENTS[:3]
    ])

    async def run_pipeline():
        result = await manager.process_through_pipeline(
            content=healthcare_content,
            query=RESEARCH_QUERY,
        )
        return result

    result = asyncio.run(run_pipeline())

    assert result is not None
    # Result is a dict with 'final' key containing processed content
    assert "final" in result
    final_content = result["final"]["content"]
    assert len(final_content) > 0
    # Pipeline should reduce content size
    assert len(final_content) <= len(healthcare_content)


# =============================================================================
# Test 4: Skills Execution on RLM Output
# =============================================================================

def test_skills_on_rlm_output():
    """
    Test skills framework executing on content retrieved via RLM paradigm.
    Skills receive context dict and produce structured analysis.
    """
    from filters import SkillRegistry

    # Simulated RLM output (what code execution would produce)
    context = {
        "compressed_context": """
        Artificial intelligence is revolutionizing medical diagnostics. Machine learning
        algorithms can detect cancer with 94% accuracy. AI-assisted diagnosis reduces
        errors by 30% and speeds up the process by 50%. The AI healthcare market is
        projected to reach $45 billion by 2030. Major healthcare systems including
        Mayo Clinic have begun integrating AI tools. Machine learning is transforming
        drug discovery, potentially reducing development costs by 60%.
        """,
        "query": RESEARCH_QUERY,
    }
    context["content"] = context["compressed_context"]

    registry = SkillRegistry(load_defaults=True)

    async def run_skills():
        # Execute individual skills
        research_result = await registry.execute_async("research_analysis", context)
        assert research_result.success

        summary_result = await registry.execute_async("summary", context)
        assert summary_result.success

        entity_result = await registry.execute_async("entity_extraction", context)
        assert entity_result.success

        # Execute multiple in parallel
        multi_results = await registry.execute_multiple(
            names=["research_analysis", "summary", "sentiment_analysis"],
            context=context,
            parallel=True,
        )
        assert len(multi_results) == 3

        # Verify structured output
        entities = entity_result.tool_results.get("extract_entities", {})
        assert entities.get("total_count", 0) > 0

        return multi_results

    results = asyncio.run(run_skills())
    assert all(r.success for r in results.values())


# =============================================================================
# Test 5: End-to-End RLM Flow
# =============================================================================

def test_end_to_end_rlm_flow():
    """
    End-to-end test: KB → KnowledgeEnvironment → REPL code execution → Skills.
    Simulates the complete RLM paradigm from document ingestion to insights.
    """
    from filters import KnowledgeBase, KnowledgeBaseConfig, SkillRegistry
    from core.memory_manager import REPLEnvironment
    from core.knowledge_environment import (
        KnowledgeEnvironment,
        KnowledgeEnvironmentConfig,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        # Stage: Build Knowledge Base
        kb = KnowledgeBase(
            config=KnowledgeBaseConfig(name="e2e_rlm", chunk_size=500),
            persist_directory=tmpdir,
        )
        for doc in SAMPLE_RESEARCH_DOCUMENTS:
            kb.add_text(
                text=doc["content"],
                metadata={**doc["metadata"], "title": doc["title"]}
            )

        # Stage: Create RLM Environment
        repl = REPLEnvironment()
        env = KnowledgeEnvironment(
            kb, repl,
            config=KnowledgeEnvironmentConfig(include_topics=True),
        )

        # Stage: LLM receives metadata prompt (not raw content)
        prompt = env.get_context_prompt(query_context=RESEARCH_QUERY)
        assert "KNOWLEDGE BASE CONTEXT" in prompt
        assert "CURRENT TASK" in prompt

        # Stage: LLM writes code to search and filter
        _, err = env.execute_code('results = kb_search("AI healthcare diagnosis", top_k=10)')
        assert err is None

        _, err = env.execute_code('high_quality = filter_by_score(results, 0.3)')
        assert err is None

        count, err = env.execute_code('len(high_quality)')
        assert err is None
        assert count > 0

        # Stage: LLM extracts content for analysis
        _, err = env.execute_code(
            'combined_content = " ".join([r["content"] for r in high_quality[:3]])'
        )
        assert err is None

        content, err = env.execute_code('combined_content')
        assert err is None
        assert len(content) > 0

        # Stage: Apply skills to the retrieved content
        context = {
            "compressed_context": content,
            "content": content,
            "query": RESEARCH_QUERY,
        }

        registry = SkillRegistry(load_defaults=True)

        async def run_skills():
            return await registry.execute_multiple(
                names=["research_analysis", "entity_extraction", "summary"],
                context=context,
                parallel=True,
            )

        skill_results = asyncio.run(run_skills())

        assert len(skill_results) == 3
        assert all(r.success for r in skill_results.values())

        # Verify entity extraction found healthcare entities
        entity_result = skill_results["entity_extraction"]
        entities = entity_result.tool_results.get("extract_entities", {})
        assert entities.get("total_count", 0) > 0

        # Verify research analysis identified themes
        research_result = skill_results["research_analysis"]
        themes = research_result.tool_results.get("identify_themes", {})
        assert len(themes.get("themes", [])) > 0


# =============================================================================
# Test 6: Developer Usage Examples
# =============================================================================

def test_developer_usage_examples():
    """
    Test common developer usage patterns with the RLM paradigm.
    """
    # Example 1: Quick KB search
    from filters import KnowledgeBase

    with tempfile.TemporaryDirectory() as tmpdir:
        kb = KnowledgeBase(name="quick_search", persist_directory=tmpdir)
        kb.add_text("AI improves medical diagnosis accuracy to 94%.")
        kb.add_text("Machine learning reduces drug development costs by 60%.")
        kb.add_text("The weather today is sunny with mild temperatures.")

        results = kb.query("AI in healthcare", top_k=2)
        assert len(results) > 0

    # Example 2: Custom skill creation
    from filters import create_skill

    def word_counter(context):
        content = context.get("content", "")
        return {"word_count": len(content.split())}

    def keyword_finder(context):
        content = context.get("content", "").lower()
        keywords = context.get("keywords", ["ai", "ml"])
        found = [kw for kw in keywords if kw in content]
        return {"keywords_found": found}

    custom_skill = create_skill(
        name="text_analyzer",
        description="Analyze text content",
        tools=[word_counter, keyword_finder],
    )

    result = custom_skill.execute({
        "content": "AI and ML are transforming healthcare diagnostics.",
        "keywords": ["ai", "ml", "healthcare"]
    })

    assert result.success
    assert result.tool_results["word_counter"]["word_count"] > 0
    assert len(result.tool_results["keyword_finder"]["keywords_found"]) == 3

    # Example 3: KnowledgeEnvironment with REPL
    from core.memory_manager import REPLEnvironment
    from core.knowledge_environment import KnowledgeEnvironment

    with tempfile.TemporaryDirectory() as tmpdir:
        kb = KnowledgeBase(name="dev_example", persist_directory=tmpdir)
        kb.add_text("AI systems detect diseases with 94% accuracy.")
        kb.add_text("Drug discovery costs reduced by 60% with ML.")

        repl = REPLEnvironment()
        env = KnowledgeEnvironment(kb, repl)

        # Developer gets metadata prompt for LLM
        prompt = env.get_context_prompt()
        assert "kb_search" in prompt

        # Developer executes LLM-generated code
        _, err = env.execute_code('results = kb_search("AI", top_k=5)')
        assert err is None

        count, err = env.execute_code('len(results)')
        assert err is None
        assert count > 0


# =============================================================================
# Run All Integration Tests
# =============================================================================

def run_all_integration_tests():
    """Run all integration tests and generate summary report."""
    print("=" * 70)
    print("RLM PARADIGM - INTEGRATION TESTS")
    print("=" * 70)

    results = {}
    tests = [
        ("KB Foundation", test_kb_foundation),
        ("RLM Context Access", test_rlm_context_access),
        ("MemoryManager Pipeline", test_memory_manager_pipeline),
        ("Skills on RLM Output", test_skills_on_rlm_output),
        ("End-to-End RLM Flow", test_end_to_end_rlm_flow),
        ("Developer Usage Examples", test_developer_usage_examples),
    ]

    for name, test_fn in tests:
        try:
            test_fn()
            results[name] = True
            print(f"  {name}: PASSED")
        except Exception as e:
            results[name] = False
            print(f"  {name}: FAILED - {e}")

    passed = sum(1 for v in results.values() if v)
    print(f"\nTotal: {passed}/{len(results)} tests passed")
    return passed == len(results)


if __name__ == "__main__":
    success = run_all_integration_tests()
    sys.exit(0 if success else 1)

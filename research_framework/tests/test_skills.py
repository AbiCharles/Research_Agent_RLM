"""
Tests for Skills Framework (Stage 4) and Pipeline Integration.

This module tests the filters package including:
- Skill base class and execution
- Built-in skills (ResearchAnalysis, Summary, Entity, Sentiment)
- SkillRegistry management
- KnowledgePipeline integration

Usage:
    # Run all tests
    python -m pytest tests/test_skills.py -v

    # Run specific test
    python -m pytest tests/test_skills.py::test_skill_creation -v

    # Run directly
    python tests/test_skills.py
"""

import asyncio
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_skill_creation():
    """Test 1: Basic skill creation and configuration."""
    print("\n" + "=" * 60)
    print("TEST 1: Skill Creation")
    print("=" * 60)

    from filters.skills import Skill, SkillConfig

    # Create skill with config
    config = SkillConfig(
        timeout=30.0,
        parallel_tools=True,
        capture_metrics=True,
    )

    skill = Skill(
        name="test_skill",
        description="A test skill for validation",
        config=config,
    )

    print(f"Skill name: {skill.name}")
    print(f"Description: {skill.description}")
    print(f"Timeout: {skill.config.timeout}s")
    print(f"Parallel tools: {skill.config.parallel_tools}")

    assert skill.name == "test_skill"
    assert skill.description == "A test skill for validation"
    assert skill.config.timeout == 30.0

    print("PASSED")
    return True


def test_skill_with_tools():
    """Test 2: Skill with tools and procedures."""
    print("\n" + "=" * 60)
    print("TEST 2: Skill with Tools and Procedures")
    print("=" * 60)

    from filters.skills import Skill

    skill = Skill(name="analysis_skill", description="Test analysis")

    # Define tools
    def extract_numbers(context):
        """Extract numbers from text."""
        import re
        text = context.get("content", "")
        numbers = re.findall(r'\d+', text)
        return {"numbers": numbers, "count": len(numbers)}

    def calculate_stats(context):
        """Calculate basic statistics."""
        import re
        text = context.get("content", "")
        words = text.split()
        return {"word_count": len(words), "char_count": len(text)}

    # Define procedure
    def generate_report(context, tool_results):
        """Generate analysis report."""
        numbers = tool_results.get("extract_numbers", {})
        stats = tool_results.get("calculate_stats", {})
        return {
            "summary": f"Found {numbers.get('count', 0)} numbers in {stats.get('word_count', 0)} words"
        }

    # Add to skill
    skill.add_tool(extract_numbers)
    skill.add_tool(calculate_stats)
    skill.add_procedure(generate_report)

    print(f"Tools added: {len(skill.tools)}")
    print(f"Procedures added: {len(skill.procedures)}")

    # Execute
    context = {"content": "The year 2024 saw 50% growth with 3 major releases."}
    result = skill.execute(context)

    print(f"Tool results: {list(result.tool_results.keys())}")
    print(f"Procedure results: {list(result.procedure_results.keys())}")
    print(f"Success: {result.success}")
    print(f"Numbers found: {result.tool_results['extract_numbers']['numbers']}")
    print(f"Report: {result.procedure_results['generate_report']['summary']}")

    assert result.success
    assert "extract_numbers" in result.tool_results
    assert "calculate_stats" in result.tool_results
    assert "generate_report" in result.procedure_results
    assert len(result.tool_results["extract_numbers"]["numbers"]) > 0

    print("PASSED")
    return True


def test_skill_async_execution():
    """Test 3: Async skill execution."""
    print("\n" + "=" * 60)
    print("TEST 3: Async Skill Execution")
    print("=" * 60)

    from filters.skills import Skill, SkillConfig

    config = SkillConfig(parallel_tools=True)
    skill = Skill(name="async_skill", config=config)

    # Add sync and async tools
    def sync_tool(context):
        return {"type": "sync", "value": 42}

    async def async_tool(context):
        await asyncio.sleep(0.01)  # Simulate async work
        return {"type": "async", "value": 84}

    skill.add_tool(sync_tool)
    skill.add_tool(async_tool)

    async def run_test():
        context = {"query": "test"}
        result = await skill.execute_async(context)

        print(f"Success: {result.success}")
        print(f"Tools executed: {len(result.tool_results)}")
        print(f"Duration: {result.metrics.get('duration_ms', 0):.2f}ms")
        print(f"Parallel: {result.metrics.get('parallel_execution', False)}")

        assert result.success
        assert "sync_tool" in result.tool_results
        assert "async_tool" in result.tool_results
        assert result.tool_results["sync_tool"]["value"] == 42
        assert result.tool_results["async_tool"]["value"] == 84

        return True

    result = asyncio.run(run_test())
    print("PASSED")
    return result


def test_research_analysis_skill():
    """Test 4: Built-in ResearchAnalysisSkill."""
    print("\n" + "=" * 60)
    print("TEST 4: Research Analysis Skill")
    print("=" * 60)

    from filters.skills import ResearchAnalysisSkill

    skill = ResearchAnalysisSkill()

    print(f"Skill: {skill.name}")
    print(f"Tools: {[t[0] for t in skill.tools]}")
    print(f"Procedures: {[p[0] for p in skill.procedures]}")

    # Test content
    content = """
    This research demonstrates that AI significantly improves medical diagnosis accuracy.
    Machine learning algorithms showed 95% accuracy in detecting cancer from medical images.
    The findings indicate that deep learning models can reduce diagnostic errors by 30%.
    Key evidence suggests that AI-assisted diagnosis leads to better patient outcomes.
    However, concerns about data privacy remain a challenge for healthcare AI adoption.
    """

    context = {
        "compressed_context": content,
        "query": "AI in healthcare diagnostics",
    }

    result = skill.execute(context)

    print(f"\nResults:")
    print(f"  Findings: {result.tool_results.get('extract_key_findings', {}).get('count', 0)}")
    print(f"  Themes: {result.tool_results.get('identify_themes', {}).get('themes', [])}")
    print(f"  Confidence: {result.tool_results.get('assess_confidence', {}).get('confidence_level', 'unknown')}")
    print(f"  Insights: {len(result.procedure_results.get('synthesize_insights', {}).get('insights', []))}")

    assert result.success
    assert "extract_key_findings" in result.tool_results
    assert "identify_themes" in result.tool_results
    assert "synthesize_insights" in result.procedure_results

    print("PASSED")
    return True


def test_summary_skill():
    """Test 5: Built-in SummarySkill."""
    print("\n" + "=" * 60)
    print("TEST 5: Summary Skill")
    print("=" * 60)

    from filters.skills import SummarySkill

    skill = SummarySkill()

    content = """
    Artificial intelligence is transforming healthcare.
    Machine learning enables precise diagnostics.
    Deep learning analyzes medical images effectively.
    Natural language processing extracts clinical insights.
    These technologies improve patient outcomes significantly.
    """

    context = {"content": content, "query": "AI Healthcare Summary"}
    result = skill.execute(context)

    stats = result.tool_results.get("calculate_statistics", {})
    summary = result.procedure_results.get("generate_summary", {})

    print(f"Statistics:")
    print(f"  Words: {stats.get('word_count', 0)}")
    print(f"  Sentences: {stats.get('sentence_count', 0)}")
    print(f"  Paragraphs: {stats.get('paragraph_count', 0)}")
    print(f"\nSummary generated: {len(summary.get('summary_text', '')) > 0}")

    assert result.success
    assert stats.get("word_count", 0) > 0
    assert summary.get("summary_text")

    print("PASSED")
    return True


def test_entity_extraction_skill():
    """Test 6: Built-in EntityExtractionSkill."""
    print("\n" + "=" * 60)
    print("TEST 6: Entity Extraction Skill")
    print("=" * 60)

    from filters.skills import EntityExtractionSkill

    skill = EntityExtractionSkill()

    content = """
    Google and Microsoft invested $5 billion in AI research during 2024.
    The European Union approved new regulations on artificial intelligence.
    Dr. Jane Smith from Stanford University published findings showing 85% improvement.
    """

    context = {"content": content}
    result = skill.execute(context)

    entities = result.tool_results.get("extract_entities", {}).get("entities", {})
    graph = result.procedure_results.get("build_entity_graph", {})

    print(f"Entities found:")
    print(f"  Organizations: {entities.get('organizations', [])[:3]}")
    print(f"  Dates: {entities.get('dates', [])[:3]}")
    print(f"  Money: {entities.get('money', [])[:3]}")
    print(f"  Percentages: {entities.get('percentages', [])[:3]}")
    print(f"Total entities: {graph.get('entity_summary', {}).get('total_entities', 0)}")

    assert result.success
    assert result.tool_results.get("extract_entities", {}).get("total_count", 0) > 0

    print("PASSED")
    return True


def test_sentiment_skill():
    """Test 7: Built-in SentimentAnalysisSkill."""
    print("\n" + "=" * 60)
    print("TEST 7: Sentiment Analysis Skill")
    print("=" * 60)

    from filters.skills import SentimentAnalysisSkill

    skill = SentimentAnalysisSkill()

    # Positive content
    positive_content = """
    The results are excellent and show significant improvement.
    This innovative approach demonstrates great success.
    The positive outcomes benefit everyone involved.
    """

    context = {"content": positive_content}
    result = skill.execute(context)

    sentiment = result.tool_results.get("analyze_sentiment", {})
    report = result.procedure_results.get("generate_sentiment_report", {})

    print(f"Sentiment Analysis:")
    print(f"  Score: {sentiment.get('sentiment_score', 0):.2f}")
    print(f"  Label: {sentiment.get('sentiment_label', 'unknown')}")
    print(f"  Indicators: {sentiment.get('indicator_counts', {})}")

    assert result.success
    assert sentiment.get("sentiment_label") == "positive"

    print("PASSED")
    return True


def test_skill_registry():
    """Test 8: SkillRegistry management."""
    print("\n" + "=" * 60)
    print("TEST 8: Skill Registry")
    print("=" * 60)

    from filters.skills import SkillRegistry, Skill

    # Create registry with defaults
    registry = SkillRegistry(load_defaults=True)

    print(f"Default skills loaded: {len(registry.list_skills())}")
    print(f"Skills: {registry.list_skills()}")

    # Get skill info
    info = registry.get_skill_info("research_analysis")
    print(f"\nResearch Analysis skill:")
    print(f"  Description: {info.get('description', '')[:50]}...")
    print(f"  Tools: {info.get('tools', [])}")
    print(f"  Procedures: {info.get('procedures', [])}")

    # Register custom skill
    custom = Skill(name="custom_skill", description="Custom test skill")
    custom.add_tool(lambda ctx: {"result": "custom"})
    registry.register(custom)

    print(f"\nAfter custom registration: {len(registry.list_skills())} skills")

    assert "research_analysis" in registry.list_skills()
    assert "summary" in registry.list_skills()
    assert "custom_skill" in registry.list_skills()

    print("PASSED")
    return True


def test_registry_execution():
    """Test 9: Registry skill execution."""
    print("\n" + "=" * 60)
    print("TEST 9: Registry Execution")
    print("=" * 60)

    from filters.skills import SkillRegistry

    registry = SkillRegistry()

    context = {
        "content": "AI demonstrates significant improvements in healthcare diagnostics.",
        "query": "AI healthcare",
    }

    async def run_test():
        # Execute single skill
        result = await registry.execute_async("research_analysis", context)
        print(f"Single execution:")
        print(f"  Skill: {result.skill_name}")
        print(f"  Success: {result.success}")
        print(f"  Duration: {result.metrics.get('duration_ms', 0):.2f}ms")

        # Execute multiple skills
        results = await registry.execute_multiple(
            names=["research_analysis", "summary"],
            context=context,
            parallel=True,
        )
        print(f"\nMultiple execution:")
        print(f"  Skills executed: {list(results.keys())}")

        assert result.success
        assert len(results) == 2

        return True

    result = asyncio.run(run_test())
    print("PASSED")
    return result


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("SKILLS FRAMEWORK TESTS")
    print("=" * 60)

    results = []

    # Run tests
    results.append(("Skill Creation", test_skill_creation()))
    results.append(("Skill with Tools", test_skill_with_tools()))
    results.append(("Async Execution", test_skill_async_execution()))
    results.append(("Research Analysis Skill", test_research_analysis_skill()))
    results.append(("Summary Skill", test_summary_skill()))
    results.append(("Entity Extraction Skill", test_entity_extraction_skill()))
    results.append(("Sentiment Skill", test_sentiment_skill()))
    results.append(("Skill Registry", test_skill_registry()))
    results.append(("Registry Execution", test_registry_execution()))

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, r in results if r)
    total = len(results)

    for name, result in results:
        status = "PASSED" if result else "FAILED"
        print(f"  {name}: {status}")

    print(f"\nTotal: {passed}/{total} tests passed")

    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

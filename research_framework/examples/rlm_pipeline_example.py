"""
RLM Pipeline Example - End-to-end usage of the research framework.

Run from the research_framework directory:
    python examples/rlm_pipeline_example.py
"""

import asyncio
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from filters import KnowledgeBase, KnowledgeBaseConfig, SkillRegistry
from core.memory_manager import REPLEnvironment
from core.knowledge_environment import KnowledgeEnvironment, KnowledgeEnvironmentConfig


def main():
    with tempfile.TemporaryDirectory() as tmpdir:

        # =====================================================================
        # Step 1: Build a Knowledge Base
        # =====================================================================
        print("=" * 60)
        print("STEP 1: Build Knowledge Base")
        print("=" * 60)

        kb = KnowledgeBase(
            config=KnowledgeBaseConfig(name="demo", chunk_size=500),
            persist_directory=tmpdir,
        )

        kb.add_text(
            "Artificial intelligence is revolutionizing medical diagnostics. "
            "Machine learning algorithms can detect cancer in medical images "
            "with 94% accuracy, matching human radiologists. AI-assisted "
            "diagnosis reduces errors by 30% and speeds up the process by 50%.",
            metadata={"topic": "AI diagnostics", "year": 2024},
        )

        kb.add_text(
            "Machine learning is transforming drug discovery. Traditional drug "
            "development takes 10-15 years and costs $2 billion per drug. AI "
            "approaches reduce timelines by 40% and costs by 60%. AlphaFold's "
            "protein structure predictions are a major breakthrough.",
            metadata={"topic": "drug discovery", "year": 2024},
        )

        kb.add_text(
            "The 2024 football season saw record attendance across all leagues. "
            "Several teams broke long-standing records. Youth participation in "
            "sports programs increased by 15% compared to previous years.",
            metadata={"topic": "sports", "year": 2024},
        )

        stats = kb.get_stats()
        print(f"  Indexed: {stats.document_count} chunks from {stats.source_count} sources")
        print(f"  Total characters: {stats.total_characters:,}")

        # =====================================================================
        # Step 2: Create the RLM Environment
        # =====================================================================
        print(f"\n{'=' * 60}")
        print("STEP 2: Create RLM Environment")
        print("=" * 60)

        repl = REPLEnvironment()
        env = KnowledgeEnvironment(
            kb, repl,
            config=KnowledgeEnvironmentConfig(
                include_topics=True,
                include_sources=True,
            ),
        )

        metadata = env.get_metadata()
        print(f"  Documents available: {metadata.document_count}")
        print(f"  Topics: {metadata.topics[:5]}")
        print(f"  Functions: {metadata.functions_available}")

        # =====================================================================
        # Step 3: Get the context prompt (what you send to your LLM)
        # =====================================================================
        print(f"\n{'=' * 60}")
        print("STEP 3: Context Prompt (sent to LLM instead of raw docs)")
        print("=" * 60)

        prompt = env.get_context_prompt(
            query_context="How is AI improving healthcare?",
        )
        # Show first 30 lines
        for line in prompt.split("\n")[:30]:
            print(f"  {line}")
        print("  ...")

        # =====================================================================
        # Step 4: Execute code against the KB (simulating LLM-generated code)
        # =====================================================================
        print(f"\n{'=' * 60}")
        print("STEP 4: Execute LLM-generated code in REPL")
        print("=" * 60)

        # Search for relevant documents
        _, err = env.execute_code('results = kb_search("AI healthcare diagnosis", top_k=5)')
        print(f"  kb_search executed: error={err}")

        count, _ = env.execute_code('len(results)')
        print(f"  Results found: {count}")

        # Filter by quality
        _, _ = env.execute_code('best = filter_by_score(results, 0.3)')
        best_count, _ = env.execute_code('len(best)')
        print(f"  After filtering (score >= 0.3): {best_count}")

        # Show what was found
        _, _ = env.execute_code('topics = [r["metadata"].get("topic", "unknown") for r in best]')
        topics, _ = env.execute_code('topics')
        print(f"  Topics retrieved: {topics}")

        # Get the top result's content
        top_score, _ = env.execute_code('round(best[0]["score"], 3)')
        preview, _ = env.execute_code('best[0]["content"][:80]')
        print(f"  Top result (score={top_score}): {preview}...")

        # Combine content for analysis
        _, _ = env.execute_code(
            'combined = " ".join([r["content"] for r in best])'
        )
        combined_content, _ = env.execute_code('combined')

        # =====================================================================
        # Step 5: Run Skills on the retrieved content
        # =====================================================================
        print(f"\n{'=' * 60}")
        print("STEP 5: Run Skills on retrieved content")
        print("=" * 60)

        context = {
            "content": combined_content,
            "compressed_context": combined_content,
            "query": "How is AI improving healthcare?",
        }

        registry = SkillRegistry(load_defaults=True)
        print(f"  Available skills: {registry.list_skills()}")

        async def run_skills():
            results = await registry.execute_multiple(
                ["research_analysis", "entity_extraction", "summary"],
                context,
                parallel=True,
            )
            return results

        results = asyncio.run(run_skills())

        # Research Analysis
        research = results["research_analysis"]
        themes = research.tool_results.get("identify_themes", {})
        confidence = research.tool_results.get("assess_confidence", {})
        print(f"\n  Research Analysis:")
        print(f"    Themes: {themes.get('themes', [])}")
        print(f"    Dominant: {themes.get('dominant_theme')}")
        print(f"    Confidence: {confidence.get('confidence_level')} ({confidence.get('confidence_score')})")

        # Entity Extraction
        entities = results["entity_extraction"]
        extracted = entities.tool_results.get("extract_entities", {}).get("entities", {})
        print(f"\n  Entity Extraction:")
        print(f"    Organizations: {extracted.get('organizations', [])[:5]}")
        print(f"    Percentages: {extracted.get('percentages', [])}")
        print(f"    Money: {extracted.get('money', [])}")

        # Summary
        summary = results["summary"]
        stats = summary.tool_results.get("calculate_statistics", {})
        print(f"\n  Summary Statistics:")
        print(f"    Words: {stats.get('word_count')}")
        print(f"    Sentences: {stats.get('sentence_count')}")

        # =====================================================================
        # Result
        # =====================================================================
        print(f"\n{'=' * 60}")
        print("COMPLETE")
        print("=" * 60)
        all_passed = all(r.success for r in results.values())
        print(f"  All skills succeeded: {all_passed}")
        print(f"  Skills executed: {list(results.keys())}")


if __name__ == "__main__":
    main()

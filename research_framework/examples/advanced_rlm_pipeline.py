"""
Advanced RLM Pipeline Example.

Demonstrates features beyond the basic rlm_pipeline_example.py:
  - Multi-domain KB with rich metadata for filtered queries
  - Custom Skill built from scratch (CompetitiveAnalysisSkill)
  - Multi-stage REPL analysis: search -> filter -> group by domain -> cross-domain compare
  - All skills (built-in + custom) executed in parallel
  - KB persistence: save -> reload -> verify chunk count survives
  - Final structured report aggregated from all skill results

Run from the research_framework directory:
    python examples/advanced_rlm_pipeline.py
"""

import asyncio
import sys
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent))

# Suppress library-level INFO/WARNING noise so printed output is readable
import logging
logging.disable(logging.WARNING)

from filters import (
    KnowledgeBase,
    KnowledgeBaseConfig,
    SkillRegistry,
    Skill,
    SkillConfig,
)
from core.memory_manager import REPLEnvironment
from core.knowledge_environment import KnowledgeEnvironment, KnowledgeEnvironmentConfig


# =============================================================================
# 1.  Rich multi-domain corpus
# =============================================================================

CORPUS = [
    # --- Climate-tech ---
    {
        "text": (
            "Solar panel efficiency has improved dramatically over the last decade. "
            "Modern photovoltaic cells now achieve up to 29% efficiency in laboratory "
            "conditions, up from 15% in 2010. Perovskite-silicon tandem cells are "
            "pushing toward the 33% Shockley-Queisser theoretical limit. "
            "Global solar capacity reached 1.2 TW in 2023, with costs falling 90% "
            "since 2010 to around $0.03/kWh for utility-scale projects."
        ),
        "metadata": {"domain": "climate_tech", "subtopic": "solar", "year": 2023, "source_type": "technical_report"},
    },
    {
        "text": (
            "Offshore wind energy is expanding at an unprecedented rate. "
            "The global offshore wind market is projected to grow from 57 GW in 2023 "
            "to over 380 GW by 2030. Floating offshore wind turbines unlock deep-water "
            "sites previously inaccessible. Levelised cost of energy (LCOE) for offshore "
            "wind has fallen 60% over the past decade to $85/MWh on average. "
            "Key challenges include grid integration, supply chain constraints, and "
            "environmental impact assessments."
        ),
        "metadata": {"domain": "climate_tech", "subtopic": "wind", "year": 2023, "source_type": "market_research"},
    },
    {
        "text": (
            "Carbon capture and storage (CCS) technology is critical for hard-to-abate "
            "industries. Current global CCS capacity stands at 45 Mt CO2/year, a fraction "
            "of the 5-7 Gt/year required by 2050 under net-zero scenarios. "
            "Direct air capture (DAC) costs remain high at $400-1000/tonne CO2, but "
            "companies like Climeworks and Carbon Engineering are targeting sub-$200 "
            "costs by 2030. Government subsidies through the US Inflation Reduction Act "
            "are providing $85/tonne tax credits for CCS projects."
        ),
        "metadata": {"domain": "climate_tech", "subtopic": "carbon_capture", "year": 2023, "source_type": "policy_brief"},
    },
    # --- Biotech ---
    {
        "text": (
            "CRISPR-Cas9 gene editing has moved from laboratory to clinic. "
            "The FDA approved the first CRISPR-based therapy, Casgevy, in December 2023 "
            "for sickle cell disease and transfusion-dependent beta-thalassemia. "
            "The therapy shows 97% efficacy in clinical trials. "
            "Next-generation base editors and prime editors offer even greater precision "
            "with fewer off-target effects, enabling corrections of point mutations "
            "that account for approximately 57% of all disease-associated variants."
        ),
        "metadata": {"domain": "biotech", "subtopic": "gene_editing", "year": 2023, "source_type": "clinical_study"},
    },
    {
        "text": (
            "mRNA therapeutics have expanded far beyond COVID-19 vaccines. "
            "Moderna and BioNTech are running Phase III trials for personalised cancer "
            "vaccines that train the immune system to recognise tumour-specific neoantigens. "
            "Early data shows a 44% reduction in recurrence for melanoma patients. "
            "mRNA-based treatments for rare diseases, cardiovascular conditions, and "
            "infectious diseases are advancing rapidly, with over 150 candidates in trials. "
            "Manufacturing costs have dropped 70% since 2020 due to improved lipid "
            "nanoparticle delivery systems."
        ),
        "metadata": {"domain": "biotech", "subtopic": "mrna", "year": 2023, "source_type": "market_research"},
    },
    {
        "text": (
            "AI-driven drug discovery is dramatically accelerating the pharmaceutical pipeline. "
            "AlphaFold2 has predicted structures for over 200 million proteins, enabling "
            "structure-based drug design at scale. Companies like Insilico Medicine achieved "
            "FDA IND approval for an AI-designed drug candidate in 18 months versus the "
            "typical 4-6 years. Generative AI models now propose novel molecular scaffolds "
            "with predicted ADMET properties, reducing attrition in early-stage discovery. "
            "The AI drug discovery market is forecast to reach $4 billion by 2027."
        ),
        "metadata": {"domain": "biotech", "subtopic": "ai_drug_discovery", "year": 2023, "source_type": "technical_report"},
    },
    # --- Fintech ---
    {
        "text": (
            "Embedded finance is transforming how non-financial companies deliver financial "
            "services. The embedded finance market is projected to grow from $54 billion in "
            "2022 to $385 billion by 2029. Buy-now-pay-later (BNPL) platforms processed "
            "$309 billion in transactions in 2023. Banking-as-a-Service (BaaS) providers "
            "enable brands to offer branded credit cards, insurance, and lending. "
            "Regulatory scrutiny is increasing, with the CFPB proposing new oversight "
            "rules for BNPL providers in 2024."
        ),
        "metadata": {"domain": "fintech", "subtopic": "embedded_finance", "year": 2023, "source_type": "market_research"},
    },
    {
        "text": (
            "Decentralised finance (DeFi) protocols are maturing despite market volatility. "
            "Total value locked (TVL) in DeFi reached $47 billion in early 2024. "
            "Real-world asset (RWA) tokenisation is bridging TradFi and DeFi, with "
            "BlackRock's tokenised treasury fund crossing $1 billion in assets. "
            "Layer-2 scaling solutions like Arbitrum and Optimism have reduced Ethereum "
            "transaction costs by 95%, making micropayments economically viable. "
            "Institutional adoption is growing, with JPMorgan's Onyx platform processing "
            "over $700 billion in repo transactions on blockchain."
        ),
        "metadata": {"domain": "fintech", "subtopic": "defi", "year": 2023, "source_type": "technical_report"},
    },
    {
        "text": (
            "AI-powered fraud detection systems are reducing financial crime losses. "
            "Machine learning models analyse thousands of transaction features in real time, "
            "reducing false positives by 60% compared to rule-based systems. "
            "Generative AI is being used to simulate adversarial attack patterns for "
            "red-teaming fraud systems. Global payment fraud losses reached $32 billion "
            "in 2023; AI-driven systems prevented an estimated $10 billion of that. "
            "Federated learning enables banks to train shared fraud models without "
            "exposing customer data across institutions."
        ),
        "metadata": {"domain": "fintech", "subtopic": "fraud_detection", "year": 2023, "source_type": "policy_brief"},
    },
]


# =============================================================================
# 2.  Custom Skill: CompetitiveAnalysisSkill
# =============================================================================

class CompetitiveAnalysisSkill(Skill):
    """
    Custom skill that performs cross-domain competitive analysis.

    Tools:
    - detect_market_metrics: Extract market size figures, growth rates, and costs
    - identify_key_players: Find company and technology names
    - score_maturity: Rate technology readiness level (1-5)

    Procedures:
    - rank_opportunities: Rank domains by combined metric signals
    """

    def __init__(self, config=None):
        super().__init__(
            name="competitive_analysis",
            description="Cross-domain competitive landscape and market maturity analysis",
            config=config,
        )
        self.add_tool(self._detect_market_metrics, "detect_market_metrics")
        self.add_tool(self._identify_key_players, "identify_key_players")
        self.add_tool(self._score_maturity, "score_maturity")
        self.add_procedure(self._rank_opportunities, "rank_opportunities")

    # ------------------------------------------------------------------
    # Tools
    # ------------------------------------------------------------------

    def _detect_market_metrics(self, context: Dict[str, Any]) -> Dict[str, Any]:
        content = context.get("compressed_context", context.get("content", ""))

        billions = re.findall(r'\$[\d,.]+\s*billion', content, re.IGNORECASE)
        trillions = re.findall(r'[\d.]+\s*TW|[\d.]+\s*Gt|[\d,.]+\s*trillion', content, re.IGNORECASE)
        percentages = re.findall(r'\d+(?:\.\d+)?%', content)
        years = re.findall(r'by\s+(20\d\d)', content)
        growth_signals = re.findall(
            r'(?:grow|reach|increase|expand|project)[a-z]*\s+(?:to|from)?\s*[\$\d]',
            content, re.IGNORECASE
        )

        return {
            "market_sizes_billions": list(set(billions))[:8],
            "large_scale_units": list(set(trillions))[:5],
            "percentage_changes": list(set(percentages))[:10],
            "target_years": sorted(set(years)),
            "growth_signal_count": len(growth_signals),
        }

    def _identify_key_players(self, context: Dict[str, Any]) -> Dict[str, Any]:
        content = context.get("compressed_context", context.get("content", ""))

        # Named entities: Title-cased multi-word phrases or known brands
        brand_pattern = r'\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)\b'
        candidates = re.findall(brand_pattern, content)

        # Known company / organisation indicators
        stopwords = {
            "AI", "FDA", "DeFi", "TradFi", "BNPL", "BaaS", "CFPB", "TVL",
            "CCS", "DAC", "CRISPR", "RWA", "ADMET", "IND", "LCOE", "MWh",
            "GW", "TW", "CO2", "Mt", "Gt", "DNA", "RNA", "USD",
        }
        companies = [
            c for c in candidates
            if len(c) > 4 and c not in stopwords and not c.isupper()
        ]
        unique_companies = list(dict.fromkeys(companies))  # preserve order, deduplicate

        # Technology terms (lowercase multi-word)
        tech_pattern = r'\b([a-z]+-[a-z]+(?:-[a-z]+)?)\b'
        tech_terms = list(set(re.findall(tech_pattern, content)))[:10]

        return {
            "organisations": unique_companies[:12],
            "technology_terms": tech_terms,
        }

    def _score_maturity(self, context: Dict[str, Any]) -> Dict[str, Any]:
        content = context.get("compressed_context", context.get("content", "")).lower()

        # TRL-like heuristics
        maturity_signals = {
            "commercial": ["approved", "deployed", "commercial", "market", "revenue", "billion"],
            "late_stage": ["phase iii", "ipo", "partnership", "scaling", "growth"],
            "mid_stage": ["phase ii", "pilot", "prototype", "trial", "demonstration"],
            "early_stage": ["research", "laboratory", "experimental", "proof-of-concept", "startup"],
        }

        scores = {stage: sum(1 for kw in kws if kw in content)
                  for stage, kws in maturity_signals.items()}

        top_stage = max(scores, key=lambda k: scores[k])

        trl_map = {"commercial": 5, "late_stage": 4, "mid_stage": 3, "early_stage": 2}
        trl = trl_map.get(top_stage, 1)

        return {
            "maturity_stage": top_stage,
            "trl_score": trl,
            "signal_counts": scores,
        }

    # ------------------------------------------------------------------
    # Procedure
    # ------------------------------------------------------------------

    def _rank_opportunities(
        self,
        context: Dict[str, Any],
        tool_results: Dict[str, Any],
    ) -> Dict[str, Any]:
        metrics = tool_results.get("detect_market_metrics", {})
        players = tool_results.get("identify_key_players", {})
        maturity = tool_results.get("score_maturity", {})

        # Simple composite score
        market_signal = len(metrics.get("market_sizes_billions", [])) * 2
        growth_signal = min(metrics.get("growth_signal_count", 0), 5)
        player_signal = min(len(players.get("organisations", [])), 8)
        trl = maturity.get("trl_score", 1)

        composite = market_signal + growth_signal + player_signal + trl

        return {
            "composite_score": composite,
            "breakdown": {
                "market_presence": market_signal,
                "growth_signals": growth_signal,
                "ecosystem_richness": player_signal,
                "maturity_trl": trl,
            },
            "recommendation": (
                "High priority  -  mature market with strong growth signals"
                if composite >= 15
                else "Medium priority  -  developing market"
                if composite >= 8
                else "Monitor  -  early stage"
            ),
        }


# =============================================================================
# 3.  Main pipeline
# =============================================================================

def _hr(title: str = "", width: int = 62) -> str:
    if title:
        pad = width - len(title) - 2
        return f"{'=' * (pad // 2)} {title} {'=' * (pad - pad // 2)}"
    return "=" * width


def main():
    with tempfile.TemporaryDirectory() as tmpdir:
        persist_path = str(Path(tmpdir) / "kb_advanced")

        # =================================================================
        # STEP 1: Build multi-domain KB with rich metadata
        # =================================================================
        print(_hr("STEP 1  -  Build Multi-Domain Knowledge Base"))

        kb = KnowledgeBase(
            config=KnowledgeBaseConfig(
                name="advanced_demo",
                chunk_size=600,
                chunk_overlap=100,
                persist_directory=persist_path,
            )
        )

        for i, entry in enumerate(CORPUS):
            domain = entry["metadata"]["domain"]
            subtopic = entry["metadata"]["subtopic"]
            kb.add_text(
                entry["text"],
                source_name=f"{domain}/{subtopic}_{i}",
                metadata=entry["metadata"],
            )

        stats = kb.get_stats()
        print(f"  Chunks indexed : {stats.document_count}")
        print(f"  Unique sources : {stats.source_count}")
        print(f"  Total chars    : {stats.total_characters:,}")

        # =================================================================
        # STEP 2: KB persistence  -  save, reload, verify
        # =================================================================
        print(_hr("STEP 2 - Persistence: Save > Reload > Verify"))

        saved_path = kb.save()
        print(f"  Saved to: {saved_path}")

        kb2 = KnowledgeBase(
            config=KnowledgeBaseConfig(
                name="reloaded",
                persist_directory=persist_path,
            )
        )
        kb2.load(persist_path)
        print(f"  Reloaded chunks: {kb2.document_count}")
        assert kb2.document_count == kb.document_count, "Chunk count mismatch after reload!"
        print("  Persistence verification passed.")

        # Continue with the original KB
        # =================================================================
        # STEP 3: RLM Environment + multi-stage REPL analysis
        # =================================================================
        print(_hr("STEP 3  -  RLM Environment + Multi-Stage REPL Analysis"))

        repl = REPLEnvironment()
        env = KnowledgeEnvironment(
            kb, repl,
            config=KnowledgeEnvironmentConfig(
                include_topics=True,
                include_sources=True,
                max_topics=20,
            ),
        )

        metadata = env.get_metadata()
        print(f"  KB topics detected : {metadata.topics[:8]}")
        print(f"  Functions exposed  : {metadata.functions_available}")

        # --- Stage 3a: Broad search across all domains ---
        env.execute_code('all_results = kb_search("technology growth market innovation", top_k=20)')
        count, _ = env.execute_code('len(all_results)')
        print(f"\n  [3a] Broad search -> {count} results")

        # --- Stage 3b: Domain-filtered queries ---
        domains = ["climate_tech", "biotech", "fintech"]
        domain_counts = {}
        for domain in domains:
            env.execute_code(
                f'domain_{domain} = kb_search('
                f'"innovation growth market", top_k=10, '
                f'filter_metadata={{"domain": "{domain}"}})'
            )
            n, _ = env.execute_code(f'len(domain_{domain})')
            domain_counts[domain] = n
            print(f"  [3b] domain={domain:15s} -> {n} filtered chunks")

        # --- Stage 3c: Score-filtered + per-domain best result ---
        env.execute_code('high_quality = filter_by_score(all_results, min_score=0.25)')
        hq_count, _ = env.execute_code('len(high_quality)')
        print(f"\n  [3c] High-quality chunks (score >= 0.25) : {hq_count}")

        # Group by domain inside REPL
        env.execute_code(
            'grouped = {}\n'
            'for r in high_quality:\n'
            '    d = r["metadata"].get("domain", "unknown")\n'
            '    grouped.setdefault(d, []).append(r)'
        )
        grouped, _ = env.execute_code('{d: len(v) for d, v in grouped.items()}')
        print(f"  [3c] Grouped distribution : {grouped}")

        # --- Stage 3d: Cross-domain comparison  - 
        # collect top content from each domain for downstream skill analysis
        env.execute_code(
            'combined_texts = {}\n'
            'for domain in ["climate_tech", "biotech", "fintech"]:\n'
            '    chunks = [r for r in all_results if r["metadata"].get("domain") == domain]\n'
            '    combined_texts[domain] = " ".join(c["content"] for c in chunks[:3])'
        )
        combined_texts, _ = env.execute_code('combined_texts')

        # Aggregate into one string for skill input
        all_content = " ".join(combined_texts.values()) if combined_texts else ""
        top_score, _ = env.execute_code('round(all_results[0]["score"], 3) if all_results else 0')
        print(f"\n  [3d] Content assembled for skills ({len(all_content):,} chars)")
        print(f"  [3d] Top similarity score            : {top_score}")

        # =================================================================
        # STEP 4: Register custom skill + run all skills in parallel
        # =================================================================
        print(_hr("STEP 4  -  Parallel Skill Execution (Built-in + Custom)"))

        registry = SkillRegistry(load_defaults=True)
        registry.register(CompetitiveAnalysisSkill())

        print(f"  Registered skills: {registry.list_skills()}")

        context = {
            "content": all_content,
            "compressed_context": all_content,
            "query": "technology market growth and competitive landscape",
        }

        async def run_all_skills():
            return await registry.execute_multiple(
                registry.list_skills(),
                context,
                parallel=True,
            )

        skill_results = asyncio.run(run_all_skills())

        for name, result in skill_results.items():
            duration = result.metrics.get("duration_ms", 0)
            status = "OK" if result.success else f"FAILED ({result.errors})"
            print(f"  {name:25s} -> {status}  ({duration:.1f} ms)")

        # =================================================================
        # STEP 5: Aggregate into final structured report
        # =================================================================
        print(_hr("STEP 5  -  Final Aggregated Report"))

        # --- Research analysis ---
        ra = skill_results["research_analysis"]
        themes = ra.tool_results.get("identify_themes", {})
        confidence = ra.tool_results.get("assess_confidence", {})
        insights = ra.procedure_results.get("synthesize_insights", {})

        print("\n  [Research Analysis]")
        print(f"    Dominant theme  : {themes.get('dominant_theme')}")
        print(f"    Theme scores    : {themes.get('theme_scores')}")
        print(f"    Confidence      : {confidence.get('confidence_level')} "
              f"(score={confidence.get('confidence_score')})")
        print(f"    Insights        : {insights.get('insights', [])}")

        # --- Entity extraction ---
        ee = skill_results["entity_extraction"]
        entities = ee.tool_results.get("extract_entities", {}).get("entities", {})
        entity_graph = ee.procedure_results.get("build_entity_graph", {})

        print("\n  [Entity Extraction]")
        print(f"    Percentages found : {entities.get('percentages', [])[:6]}")
        print(f"    Money figures     : {entities.get('money', [])[:5]}")
        print(f"    Primary domain    : {entity_graph.get('entity_summary', {}).get('primary_domain')}")
        print(f"    Total entities    : {entity_graph.get('entity_summary', {}).get('total_entities')}")

        # --- Sentiment ---
        sa = skill_results["sentiment_analysis"]
        sentiment = sa.tool_results.get("analyze_sentiment", {})
        sent_report = sa.procedure_results.get("generate_sentiment_report", {})

        print("\n  [Sentiment Analysis]")
        print(f"    Sentiment       : {sentiment.get('sentiment_label')} "
              f"(score={sentiment.get('sentiment_score')})")
        print(f"    Positive words  : {sentiment.get('indicator_counts', {}).get('positive')}")
        print(f"    Negative words  : {sentiment.get('indicator_counts', {}).get('negative')}")

        # --- Summary ---
        su = skill_results["summary"]
        stats_out = su.tool_results.get("calculate_statistics", {})
        gen_sum = su.procedure_results.get("generate_summary", {})

        print("\n  [Summary]")
        print(f"    Words           : {stats_out.get('word_count')}")
        print(f"    Sentences       : {stats_out.get('sentence_count')}")
        print(f"    Avg sent length : {stats_out.get('avg_sentence_length', 0):.1f} words")

        # --- Custom: Competitive Analysis ---
        ca = skill_results["competitive_analysis"]
        market_metrics = ca.tool_results.get("detect_market_metrics", {})
        key_players = ca.tool_results.get("identify_key_players", {})
        maturity = ca.tool_results.get("score_maturity", {})
        ranking = ca.procedure_results.get("rank_opportunities", {})

        print("\n  [Competitive Analysis  <- Custom Skill]")
        print(f"    Market size mentions : {market_metrics.get('market_sizes_billions', [])[:4]}")
        print(f"    % changes found      : {market_metrics.get('percentage_changes', [])[:5]}")
        print(f"    Target years         : {market_metrics.get('target_years', [])}")
        print(f"    Key organisations    : {key_players.get('organisations', [])[:6]}")
        print(f"    Maturity stage       : {maturity.get('maturity_stage')} "
              f"(TRL {maturity.get('trl_score')}/5)")
        print(f"    Composite score      : {ranking.get('composite_score')}  "
              f"-> {ranking.get('recommendation')}")
        print(f"    Score breakdown      : {ranking.get('breakdown')}")

        # =================================================================
        # RESULT SUMMARY
        # =================================================================
        print(_hr("RESULT SUMMARY"))

        all_passed = all(r.success for r in skill_results.values())
        total_ms = sum(r.metrics.get("duration_ms", 0) for r in skill_results.values())

        print(f"  Skills executed        : {len(skill_results)}")
        print(f"  All skills succeeded   : {all_passed}")
        print(f"  Total skill runtime    : {total_ms:.1f} ms")
        print(f"  KB chunks indexed      : {kb.document_count}")
        print(f"  Persistence verified   : True")
        print(f"  Custom skill included  : {'competitive_analysis' in skill_results}")
        print(f"  Domains analysed       : {domains}")
        print()


if __name__ == "__main__":
    main()

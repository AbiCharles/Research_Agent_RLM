# RLM Pipeline Examples — Basic vs Advanced Comparison

## Overview

Two runnable end-to-end examples demonstrating the RLM (Recursive Language Model) pipeline at different levels of complexity.

| | `rlm_pipeline_example.py` | `advanced_rlm_pipeline.py` |
|---|---|---|
| **KB entries** | 3 text blobs | 9 entries across 3 domains |
| **Metadata** | `topic`, `year` | `domain`, `subtopic`, `year`, `source_type` |
| **Unique sources** | 1 (`manual_input`) | 9 (`domain/subtopic_N` per entry) |
| **Filtered search** | None | `filter_metadata={"domain": "..."}` |
| **REPL stages** | 1 (search + score-filter) | 4 chained stages |
| **Custom skill** | None | `CompetitiveAnalysisSkill` (3 tools + 1 procedure) |
| **Skills run** | 3 built-in | 5 in parallel (4 built-in + 1 custom) |
| **Persistence** | None | Save -> reload -> chunk-count assertion |
| **Final output** | Per-skill field prints | Aggregated structured report |

---

## Basic Pipeline (`rlm_pipeline_example.py`)

### What it demonstrates

- Building a KB from raw text with `add_text()`
- Creating a `KnowledgeEnvironment` bridge to expose REPL functions
- Generating the context prompt (what an LLM would receive instead of raw documents)
- Single-stage REPL code execution: `kb_search` -> `filter_by_score`
- Running 3 built-in skills in parallel: `research_analysis`, `entity_extraction`, `summary`

### Corpus

```
3 text snippets:
  - AI diagnostics (topic: "AI diagnostics", year: 2024)
  - Drug discovery (topic: "drug discovery",  year: 2024)
  - Football season (topic: "sports",          year: 2024)
```

### Run results

```
KB:
  Chunks indexed : 3
  Sources        : 1  (all share source name "manual_input")
  Total chars    : ~780

REPL:
  kb_search("AI healthcare diagnosis", top_k=5) -> 3 results
  filter_by_score(results, 0.3)                 -> 2 results
  Topics retrieved: ['AI diagnostics', 'drug discovery']
  Top result score: 0.516

Skills (parallel):
  research_analysis   -> OK
  entity_extraction   -> OK
  summary             -> OK
  All succeeded       : True

Research Analysis:
  Themes:        ['healthcare', 'technology', 'science', 'business']
  Dominant:      healthcare
  Confidence:    medium (0.5)

Entity Extraction:
  Organisations: []
  Percentages:   ['94%', '40%', '30%', '50%', '60%']
  Money:         ['$2 billion']

Summary Statistics:
  Words:     ~110
  Sentences: ~8
```

---

## Advanced Pipeline (`advanced_rlm_pipeline.py`)

### What it demonstrates

- **Multi-domain KB** with 9 entries tagged by `domain`, `subtopic`, `year`, and `source_type`
- **Named sources** (`climate_tech/solar_0`, `biotech/gene_editing_3`, etc.) so `source_count` reflects reality
- **Metadata-filtered search** — each domain queried in isolation via `filter_metadata`
- **4-stage REPL analysis** chained inside a shared REPL namespace
- **KB persistence** — explicit save, reload into a fresh KB instance, assertion that chunk count survives
- **Custom skill** (`CompetitiveAnalysisSkill`) built with `add_tool()` / `add_procedure()` and registered alongside built-ins
- **5 skills run in parallel**, including the custom one
- **Aggregated report** drawing from all five `SkillResult` objects

### Corpus

```
9 text entries across 3 domains (3 per domain):

climate_tech:
  - solar           (technical_report)  — PV efficiency, 1.2 TW capacity, $0.03/kWh LCOE
  - wind            (market_research)   — 57 GW -> 380 GW by 2030, $85/MWh LCOE
  - carbon_capture  (policy_brief)      — 45 Mt/yr CCS, DAC costs $400-1000/tonne

biotech:
  - gene_editing    (clinical_study)    — CRISPR Casgevy FDA approval, 97% efficacy
  - mrna            (market_research)   — cancer vaccines, 44% recurrence reduction
  - ai_drug_discovery (technical_report)— AlphaFold2, 18-month IND, $4B market by 2027

fintech:
  - embedded_finance (market_research)  — $54B -> $385B by 2029, BNPL $309B
  - defi            (technical_report)  — $47B TVL, BlackRock RWA fund $1B+
  - fraud_detection (policy_brief)      — $32B losses, AI prevented $10B
```

### Run results

```
STEP 1 — Build Multi-Domain Knowledge Base
  Chunks indexed : 9
  Unique sources : 9
  Total chars    : 4,470

STEP 2 — Persistence: Save -> Reload -> Verify
  Saved to        : <tmpdir>/kb_advanced
  Reloaded chunks : 9
  Assertion       : PASSED (chunk count identical after reload)

STEP 3 — RLM Environment + Multi-Stage REPL Analysis

  Topics detected (top 8):
    offshore wind, companies like, drug discovery, embedded finance,
    reached billion, costs, systems, global

  Functions exposed:
    kb_search, kb_metadata, load_chunk, filter_by_score, list_sources

  [3a] Broad search ("technology growth market innovation", top_k=20)
       -> 9 results

  [3b] Domain-filtered queries ("innovation growth market", top_k=10):
       domain=climate_tech  -> 3 chunks
       domain=biotech       -> 3 chunks
       domain=fintech       -> 3 chunks

  [3c] High-quality filter (score >= 0.25):
       -> 2 chunks
       Grouped: {'fintech': 2}

  [3d] Content assembled for skills: 4,478 chars
       Top similarity score: 0.376

STEP 4 — Parallel Skill Execution (Built-in + Custom)
  Registered skills:
    research_analysis, summary, entity_extraction,
    sentiment_analysis, competitive_analysis

  research_analysis    -> OK  (18 ms)
  summary              -> OK  (18 ms)
  entity_extraction    -> OK  (18 ms)
  sentiment_analysis   -> OK  (18 ms)
  competitive_analysis -> OK  (17 ms)   ← custom skill

STEP 5 — Final Aggregated Report

  [Research Analysis]
    Dominant theme  : healthcare
    Theme scores    : {healthcare:3, technology:2, business:1, science:1, policy:1}
    Confidence      : medium (0.5)
    Insights        : []

  [Entity Extraction]
    Percentages     : 95%, 29%, 44%, 97%, 57%, 70%
    Money figures   : $10B, $47B, $1B, $385B, $0.03
    Primary domain  : medical
    Total entities  : 33

  [Sentiment Analysis]
    Sentiment       : positive (score=0.33)
    Positive words  : 3
    Negative words  : 1

  [Summary]
    Words           : 627
    Sentences       : 44
    Avg sent length : 13.9 words

  [Competitive Analysis  ← Custom Skill]
    Market mentions : $10B, $47B, $1B, $385B
    % changes found : 95%, 29%, 44%, 97%, 57%
    Target years    : 2027, 2029, 2030, 2050
    Key orgs        : Offshore, Floating, Levelised, Solar, Modern, Perovskite
    Maturity stage  : commercial (TRL 5/5)
    Composite score : 34  -> High priority — mature market with strong growth signals
    Breakdown       : {market_presence:16, growth_signals:5, ecosystem_richness:8, maturity_trl:5}

RESULT SUMMARY
  Skills executed       : 5
  All skills succeeded  : True
  Total skill runtime   : 89 ms
  KB chunks indexed     : 9
  Persistence verified  : True
  Custom skill included : True
  Domains analysed      : climate_tech, biotech, fintech
```

---

## Key Concepts Illustrated

### Metadata-filtered search

```python
# Basic — no filtering, all results returned
results = kb_search("AI healthcare", top_k=5)

# Advanced — restrict to a single domain
results = kb_search(
    "innovation growth market",
    top_k=10,
    filter_metadata={"domain": "biotech"},
)
```

### Custom skill anatomy

```python
class CompetitiveAnalysisSkill(Skill):
    def __init__(self):
        super().__init__(name="competitive_analysis", ...)

        # Tools execute first (in parallel when config.parallel_tools=True)
        self.add_tool(self._detect_market_metrics, "detect_market_metrics")
        self.add_tool(self._identify_key_players,  "identify_key_players")
        self.add_tool(self._score_maturity,         "score_maturity")

        # Procedure runs after all tools, receives their combined results
        self.add_procedure(self._rank_opportunities, "rank_opportunities")

    def _rank_opportunities(self, context, tool_results):
        # Combine outputs from all three tools into a composite score
        ...
```

### Multi-stage REPL chaining

```python
# Variables persist across execute_code() calls in the same REPL namespace
env.execute_code('all_results = kb_search("technology growth", top_k=20)')
env.execute_code('high_quality = filter_by_score(all_results, min_score=0.25)')
env.execute_code(
    'grouped = {}\n'
    'for r in high_quality:\n'
    '    d = r["metadata"].get("domain", "unknown")\n'
    '    grouped.setdefault(d, []).append(r)'
)
grouped_counts, _ = env.execute_code('{d: len(v) for d, v in grouped.items()}')
```

### KB persistence

```python
# Save
kb.save()                        # writes index.faiss + metadata.pkl + kb_metadata.json

# Reload into fresh instance
kb2 = KnowledgeBase(config=KnowledgeBaseConfig(persist_directory=path))
kb2.load(path)
assert kb2.document_count == kb.document_count   # verify nothing lost
```

# Research Agent RLM

A Multi-Agent Research Framework built on the **Recursive Language Model (RLM)** paradigm. Instead of feeding raw documents directly to LLMs, the framework provides metadata about available knowledge and lets LLMs write code to programmatically retrieve exactly what they need.

## Architecture

```
┌─────────────────────────────────────────────────┐
│                 Knowledge Base                   │
│         FAISS vector store + document loaders    │
└──────────────────────┬──────────────────────────┘
                       │
          ┌────────────▼────────────┐
          │  KnowledgeEnvironment   │
          │  (RLM Bridge Layer)     │
          └─────┬──────────┬────────┘
                │          │
     ┌──────────▼──┐  ┌───▼──────────┐
     │    REPL     │  │   Memory     │
     │ Environment │  │   Manager    │
     │ (sandboxed  │  │ (selection + │
     │  Python +   │  │ compression) │
     │  llm_query) │  │              │
     └─────────────┘  └──────────────┘
                │
        ┌───────▼───────┐
        │    Skills      │
        │   Framework    │
        └────────────────┘
```

### Core Components

| Component | Description |
|-----------|-------------|
| **KnowledgeBase** | FAISS-based vector storage with semantic search and multi-format document ingestion |
| **KnowledgeEnvironment** | Bridge connecting the Knowledge Base with the REPL, exposing `kb_search`, `kb_metadata`, `load_chunk`, `filter_by_score`, and `list_sources` as callable functions |
| **REPLEnvironment** | Sandboxed Python REPL with native `llm_query()` and `llm_query_batched()` for recursive sub-LM calls |
| **MemoryManager** | Internal pipeline handling selection (keyword/semantic filters) and compression (extractive/abstractive) |
| **Skills Framework** | Domain-specific analysis skills: research analysis, summarization, entity extraction, sentiment analysis |
| **Agents** | Lead researcher, research agents, and citation agent for multi-agent orchestration |

## Installation

```bash
pip install -r research_framework/requirements.txt
```

Create a `.env` file with your API key:

```
OPENAI_API_KEY=your-key-here
```

## Usage

### 1. Build a Knowledge Base

```python
from filters import KnowledgeBase, KnowledgeBaseConfig

kb = KnowledgeBase(
    config=KnowledgeBaseConfig(name="my_research", chunk_size=500),
    persist_directory="./kb_data",
)

kb.add_text("Your research content here...", metadata={"topic": "AI"})
kb.add_document("path/to/paper.pdf")
```

### 2. Create the RLM Environment

```python
from core.memory_manager import REPLEnvironment
from core.knowledge_environment import KnowledgeEnvironment, KnowledgeEnvironmentConfig

repl = REPLEnvironment()
env = KnowledgeEnvironment(
    kb, repl,
    config=KnowledgeEnvironmentConfig(
        include_topics=True,
        include_sources=True,
    ),
)
```

### 3. Get the Context Prompt

Instead of sending raw documents to the LLM, send a metadata prompt describing what's available:

```python
prompt = env.get_context_prompt(query_context="Your research question")
```

### 4. Execute Code in the REPL

The LLM generates code to retrieve exactly the content it needs:

```python
env.execute_code('results = kb_search("AI healthcare", top_k=5)')
env.execute_code('best = filter_by_score(results, 0.3)')
env.execute_code('content = " ".join([r["content"] for r in best])')
```

### 5. Run Skills on Retrieved Content

```python
from filters import SkillRegistry

registry = SkillRegistry(load_defaults=True)
results = await registry.execute_multiple(
    ["research_analysis", "entity_extraction", "summary"],
    {"content": content, "query": "Your question"},
    parallel=True,
)
```

See [examples/rlm_pipeline_example.py](research_framework/examples/rlm_pipeline_example.py) for a complete runnable example.

## Running Tests

```bash
cd research_framework
python -m pytest tests/ -v
```

97 tests passing, 7 skipped (require live API key).

## Project Structure

```
research_framework/
├── core/
│   ├── openai_client.py          # OpenAI API client wrapper
│   ├── base_agent.py             # Base agent class
│   ├── memory_manager.py         # RLM memory: REPL, selection, compression
│   └── knowledge_environment.py  # RLM bridge: KB <-> REPL
├── filters/
│   ├── vector_store.py           # FAISS vector storage
│   ├── document_loaders.py       # PDF, Word, Excel, CSV, text loaders
│   ├── knowledge_base.py         # Unified knowledge management
│   └── skills.py                 # Domain-specific analysis skills
├── agents/
│   ├── lead_researcher.py        # Lead researcher agent
│   ├── research_agent.py         # Research agent
│   └── citation_agent.py         # Citation agent
├── api/                          # FastAPI endpoints
├── tools/                        # Web search and base tools
├── config/                       # Settings and configuration
├── examples/                     # Runnable examples
└── tests/                        # Test suite
```

## License

All rights reserved.

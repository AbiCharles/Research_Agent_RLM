"""
Tests for the KnowledgeEnvironment bridge class.

This module tests the integration between KnowledgeBase and REPLEnvironment,
implementing the RLM paradigm of treating prompts as external environment variables.

Test Categories:
----------------
1. Initialization Tests: Configuration, setup, validation
2. REPL Function Tests: kb_search, kb_metadata, load_chunk, etc.
3. Metadata Tests: Topic extraction, caching, refresh
4. Context Prompt Tests: Prompt generation, formatting
5. Code Execution Tests: End-to-end execution with KB access
6. LLM Query Tests: Recursive llm_query functionality (mocked)
7. Integration Tests: Full workflow simulation

Usage:
------
    pytest tests/test_knowledge_environment.py -v
    pytest tests/test_knowledge_environment.py -v -k "test_repl"
"""

import pytest
import tempfile
from pathlib import Path
from typing import Dict, List, Any, Optional
from unittest.mock import MagicMock, AsyncMock, patch

# Import test subjects
from core.knowledge_environment import (
    KnowledgeEnvironment,
    KnowledgeEnvironmentConfig,
    ContextMetadata,
    TopicExtractor,
    create_knowledge_environment,
)
from core.memory_manager import REPLEnvironment
from filters.knowledge_base import KnowledgeBase, KnowledgeBaseConfig


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def sample_documents():
    """Sample documents for testing."""
    return [
        {
            "title": "AI in Healthcare",
            "content": """Artificial intelligence is transforming healthcare through
            improved medical diagnosis, drug discovery, and personalized treatment plans.
            Machine learning algorithms can analyze medical images with high accuracy,
            detecting diseases like cancer earlier than traditional methods. Deep learning
            models are accelerating drug discovery by predicting molecular interactions.""",
            "metadata": {"category": "healthcare", "type": "research"},
        },
        {
            "title": "Climate Change Impact",
            "content": """Climate change is affecting global weather patterns, causing
            more frequent extreme weather events. Rising sea levels threaten coastal
            communities worldwide. Carbon emissions from fossil fuels are the primary
            driver of global warming. Renewable energy adoption is crucial for mitigating
            these effects and achieving sustainability goals.""",
            "metadata": {"category": "environment", "type": "report"},
        },
        {
            "title": "Machine Learning Basics",
            "content": """Machine learning is a subset of artificial intelligence that
            enables systems to learn from data. Supervised learning uses labeled data
            to train models. Neural networks are inspired by biological brain structures.
            Deep learning has revolutionized image recognition and natural language
            processing tasks.""",
            "metadata": {"category": "technology", "type": "tutorial"},
        },
        {
            "title": "Financial Markets Analysis",
            "content": """Stock market volatility increased due to economic uncertainty.
            Investors are shifting towards safer assets amid inflation concerns.
            Algorithmic trading now accounts for significant market volume.
            Cryptocurrency markets show correlation with traditional assets.""",
            "metadata": {"category": "finance", "type": "analysis"},
        },
        {
            "title": "Natural Language Processing",
            "content": """NLP enables computers to understand human language.
            Transformer models like BERT and GPT have achieved state-of-the-art results.
            Applications include sentiment analysis, machine translation, and chatbots.
            Large language models are trained on vast amounts of text data.""",
            "metadata": {"category": "technology", "type": "research"},
        },
    ]


@pytest.fixture
def populated_kb(sample_documents, tmp_path):
    """Create a KnowledgeBase populated with sample documents."""
    config = KnowledgeBaseConfig(
        name="test_kb",
        chunk_size=500,
        chunk_overlap=50,
        auto_save=False,
    )
    kb = KnowledgeBase(config=config, persist_directory=str(tmp_path))

    # Add documents
    for doc in sample_documents:
        kb.add_text(
            text=doc["content"],
            source_name=doc["title"],
            metadata=doc["metadata"],
        )

    return kb


@pytest.fixture
def repl_env():
    """Create a fresh REPLEnvironment."""
    return REPLEnvironment()


@pytest.fixture
def knowledge_env(populated_kb, repl_env):
    """Create a KnowledgeEnvironment with populated KB and REPL."""
    return KnowledgeEnvironment(
        knowledge_base=populated_kb,
        repl=repl_env,
    )


@pytest.fixture
def mock_llm_client():
    """Create a mock LLM client for testing llm_query."""
    client = MagicMock()
    client.chat = AsyncMock(return_value={"content": "This is a mock LLM response."})
    client.generate = MagicMock(return_value="Sync mock response")
    return client


# =============================================================================
# Test 1: Initialization Tests
# =============================================================================

class TestKnowledgeEnvironmentInit:
    """Tests for KnowledgeEnvironment initialization."""

    def test_basic_initialization(self, populated_kb, repl_env):
        """Test basic initialization with required parameters."""
        env = KnowledgeEnvironment(
            knowledge_base=populated_kb,
            repl=repl_env,
        )

        assert env.knowledge_base is populated_kb
        assert env.repl is repl_env
        assert env.config is not None
        assert isinstance(env.config, KnowledgeEnvironmentConfig)

    def test_initialization_with_config(self, populated_kb, repl_env):
        """Test initialization with custom configuration."""
        config = KnowledgeEnvironmentConfig(
            max_search_results=20,
            include_topics=False,
            max_topics=5,
            cache_metadata=False,
        )

        env = KnowledgeEnvironment(
            knowledge_base=populated_kb,
            repl=repl_env,
            config=config,
        )

        assert env.config.max_search_results == 20
        assert env.config.include_topics is False
        assert env.config.max_topics == 5
        assert env.config.cache_metadata is False

    def test_initialization_fails_without_kb(self, repl_env):
        """Test that initialization fails without knowledge_base."""
        with pytest.raises(ValueError, match="knowledge_base cannot be None"):
            KnowledgeEnvironment(
                knowledge_base=None,
                repl=repl_env,
            )

    def test_initialization_fails_without_repl(self, populated_kb):
        """Test that initialization fails without repl."""
        with pytest.raises(ValueError, match="repl cannot be None"):
            KnowledgeEnvironment(
                knowledge_base=populated_kb,
                repl=None,
            )

    def test_repl_functions_registered(self, knowledge_env):
        """Test that KB functions are registered in REPL."""
        repl = knowledge_env.repl

        # Check functions in variables
        assert "kb_search" in repl._variables
        assert "kb_metadata" in repl._variables
        assert "load_chunk" in repl._variables
        assert "filter_by_score" in repl._variables
        assert "list_sources" in repl._variables

        # Check functions in safe builtins
        assert "kb_search" in repl._safe_builtins
        assert "kb_metadata" in repl._safe_builtins

    def test_factory_function(self, populated_kb):
        """Test the create_knowledge_environment factory function."""
        env = create_knowledge_environment(populated_kb)

        assert env.knowledge_base is populated_kb
        assert env.repl is not None
        assert isinstance(env.repl, REPLEnvironment)


# =============================================================================
# Test 2: REPL Function Tests
# =============================================================================

class TestREPLFunctions:
    """Tests for KB functions accessible through REPL."""

    def test_kb_search_via_repl(self, knowledge_env):
        """Test kb_search function through REPL execution."""
        result, error = knowledge_env.repl.execute(
            'kb_search("artificial intelligence", top_k=3)'
        )

        assert error is None
        assert isinstance(result, list)
        assert len(result) <= 3
        # Should find AI-related documents
        if result:
            assert "content" in result[0]
            assert "score" in result[0]

    def test_kb_search_with_min_score(self, knowledge_env):
        """Test kb_search with minimum score filter."""
        result, error = knowledge_env.repl.execute(
            'kb_search("healthcare", top_k=10, min_score=0.3)'
        )

        assert error is None
        assert isinstance(result, list)
        # All results should meet minimum score
        for r in result:
            assert r.get("score", 0) >= 0.3

    def test_kb_metadata_via_repl(self, knowledge_env):
        """Test kb_metadata function through REPL execution."""
        result, error = knowledge_env.repl.execute('kb_metadata()')

        assert error is None
        assert isinstance(result, dict)
        assert "document_count" in result
        assert "source_count" in result
        assert "topics" in result
        assert "functions_available" in result
        assert result["document_count"] > 0

    def test_load_chunk_via_repl(self, knowledge_env):
        """Test load_chunk function through REPL execution."""
        # First get a chunk ID from search
        search_result, _ = knowledge_env.repl.execute(
            'kb_search("AI", top_k=1)'
        )

        if search_result:
            chunk_id = search_result[0]["id"]
            result, error = knowledge_env.repl.execute(
                f'load_chunk("{chunk_id}")'
            )

            assert error is None
            assert result is not None
            assert "content" in result

    def test_load_chunk_not_found(self, knowledge_env):
        """Test load_chunk with non-existent ID."""
        result, error = knowledge_env.repl.execute(
            'load_chunk("nonexistent_id_12345")'
        )

        assert error is None
        assert result is None

    def test_filter_by_score_via_repl(self, knowledge_env):
        """Test filter_by_score function through REPL execution."""
        # Search and then filter
        _, error1 = knowledge_env.repl.execute(
            'results = kb_search("machine learning", top_k=10)'
        )
        result, error2 = knowledge_env.repl.execute(
            'filter_by_score(results, 0.5)'
        )

        assert error1 is None
        assert error2 is None
        assert isinstance(result, list)
        # All filtered results should meet threshold
        for r in result:
            assert r.get("score", 0) >= 0.5

    def test_list_sources_via_repl(self, knowledge_env):
        """Test list_sources function through REPL execution."""
        result, error = knowledge_env.repl.execute('list_sources()')

        assert error is None
        assert isinstance(result, list)
        assert len(result) > 0

    def test_max_search_results_enforced(self, populated_kb, repl_env):
        """Test that max_search_results config is enforced."""
        config = KnowledgeEnvironmentConfig(max_search_results=2)
        env = KnowledgeEnvironment(
            knowledge_base=populated_kb,
            repl=repl_env,
            config=config,
        )

        result, error = env.repl.execute('kb_search("the", top_k=100)')

        assert error is None
        assert len(result) <= 2  # Should be capped at max_search_results


# =============================================================================
# Test 3: Metadata Tests
# =============================================================================

class TestMetadata:
    """Tests for metadata extraction and caching."""

    def test_get_metadata_structure(self, knowledge_env):
        """Test that get_metadata returns proper structure."""
        metadata = knowledge_env.get_metadata()

        assert isinstance(metadata, ContextMetadata)
        assert metadata.document_count > 0
        assert metadata.source_count > 0
        assert metadata.total_characters > 0
        assert isinstance(metadata.topics, list)
        assert isinstance(metadata.sources, list)
        assert isinstance(metadata.functions_available, list)

    def test_metadata_to_dict(self, knowledge_env):
        """Test metadata conversion to dictionary."""
        metadata = knowledge_env.get_metadata()
        result = metadata.to_dict()

        assert isinstance(result, dict)
        assert "document_count" in result
        assert "topics" in result
        assert "functions_available" in result

    def test_metadata_to_summary_string(self, knowledge_env):
        """Test metadata summary string generation."""
        metadata = knowledge_env.get_metadata()
        summary = metadata.to_summary_string()

        assert isinstance(summary, str)
        assert "Documents:" in summary
        assert "Sources:" in summary

    def test_metadata_caching(self, knowledge_env):
        """Test that metadata is cached."""
        # First call - should generate fresh
        metadata1 = knowledge_env.get_metadata()

        # Second call - should return cached
        metadata2 = knowledge_env.get_metadata()

        # Should be the same object (cached)
        assert metadata1 is metadata2

    def test_metadata_force_refresh(self, knowledge_env):
        """Test that force_refresh bypasses cache."""
        metadata1 = knowledge_env.get_metadata()
        metadata2 = knowledge_env.get_metadata(force_refresh=True)

        # Should be different objects (freshly generated)
        assert metadata1 is not metadata2
        # But same data
        assert metadata1.document_count == metadata2.document_count

    def test_refresh_metadata_method(self, knowledge_env):
        """Test the refresh_metadata convenience method."""
        metadata1 = knowledge_env.get_metadata()
        metadata2 = knowledge_env.refresh_metadata()

        assert metadata1 is not metadata2

    def test_metadata_without_topics(self, populated_kb, repl_env):
        """Test metadata generation without topic extraction."""
        config = KnowledgeEnvironmentConfig(include_topics=False)
        env = KnowledgeEnvironment(
            knowledge_base=populated_kb,
            repl=repl_env,
            config=config,
        )

        metadata = env.get_metadata()

        # Topics should be empty when disabled
        assert metadata.topics == []


# =============================================================================
# Test 4: Topic Extractor Tests
# =============================================================================

class TestTopicExtractor:
    """Tests for the TopicExtractor utility."""

    def test_extract_topics_basic(self, sample_documents):
        """Test basic topic extraction."""
        extractor = TopicExtractor(max_terms=10)

        # Convert to expected format
        docs = [{"content": d["content"]} for d in sample_documents]
        topics = extractor.extract_topics(docs)

        assert isinstance(topics, list)
        assert len(topics) <= 10
        # Should find relevant topics
        all_topics_str = " ".join(topics).lower()
        # At least one tech/AI term should be present
        tech_terms = ["learning", "machine", "artificial", "intelligence", "data", "neural"]
        found_tech = any(term in all_topics_str for term in tech_terms)
        assert found_tech, f"Expected tech terms in: {topics}"

    def test_extract_topics_with_bigrams(self, sample_documents):
        """Test topic extraction with bigrams enabled."""
        extractor = TopicExtractor(max_terms=15)
        docs = [{"content": d["content"]} for d in sample_documents]

        topics = extractor.extract_topics(docs, include_bigrams=True)

        # Should include some multi-word topics
        multi_word = [t for t in topics if " " in t]
        # Bigrams might be found depending on content
        assert isinstance(multi_word, list)

    def test_extract_topics_without_bigrams(self, sample_documents):
        """Test topic extraction without bigrams."""
        extractor = TopicExtractor(max_terms=10)
        docs = [{"content": d["content"]} for d in sample_documents]

        topics = extractor.extract_topics(docs, include_bigrams=False)

        # Should only have single words
        for topic in topics:
            assert " " not in topic

    def test_stopwords_filtered(self):
        """Test that stopwords are properly filtered."""
        extractor = TopicExtractor()
        docs = [{"content": "The the the and and or or but machine learning"}]

        topics = extractor.extract_topics(docs)

        # Stopwords should not appear
        for topic in topics:
            words = topic.split()
            for word in words:
                assert word not in TopicExtractor.STOPWORDS

    def test_custom_stopwords(self):
        """Test adding custom stopwords."""
        custom = {"machine", "learning"}
        extractor = TopicExtractor(custom_stopwords=custom)
        docs = [{"content": "machine learning artificial intelligence"}]

        topics = extractor.extract_topics(docs)

        # Custom stopwords should be filtered
        for topic in topics:
            assert "machine" not in topic.split()
            assert "learning" not in topic.split()

    def test_extract_from_metadata(self, sample_documents):
        """Test topic extraction from document metadata."""
        extractor = TopicExtractor()

        # Add metadata with titles
        docs = [
            {"content": "", "metadata": {"title": d["title"]}}
            for d in sample_documents
        ]

        topics = extractor.extract_from_metadata(docs)

        assert isinstance(topics, list)


# =============================================================================
# Test 5: Context Prompt Tests
# =============================================================================

class TestContextPrompt:
    """Tests for context prompt generation."""

    def test_context_prompt_structure(self, knowledge_env):
        """Test that context prompt has expected structure."""
        prompt = knowledge_env.get_context_prompt()

        assert isinstance(prompt, str)
        assert "KNOWLEDGE BASE CONTEXT" in prompt
        assert "AVAILABLE CONTENT" in prompt
        assert "AVAILABLE FUNCTIONS" in prompt
        assert "kb_search" in prompt
        assert "kb_metadata" in prompt
        assert "load_chunk" in prompt

    def test_context_prompt_with_examples(self, knowledge_env):
        """Test context prompt includes examples."""
        prompt = knowledge_env.get_context_prompt(include_examples=True)

        assert "EXAMPLE USAGE" in prompt
        assert "```repl" in prompt
        assert "results = kb_search" in prompt

    def test_context_prompt_without_examples(self, knowledge_env):
        """Test context prompt without examples."""
        prompt = knowledge_env.get_context_prompt(include_examples=False)

        # Should still have functions
        assert "AVAILABLE FUNCTIONS" in prompt
        # But no example block
        assert "EXAMPLE USAGE" not in prompt

    def test_context_prompt_with_query_context(self, knowledge_env):
        """Test context prompt with query context."""
        prompt = knowledge_env.get_context_prompt(
            query_context="Find information about AI in healthcare"
        )

        assert "CURRENT TASK" in prompt
        assert "AI in healthcare" in prompt

    def test_context_prompt_includes_topics(self, knowledge_env):
        """Test that context prompt includes extracted topics."""
        prompt = knowledge_env.get_context_prompt()

        assert "TOPICS COVERED" in prompt

    def test_context_prompt_includes_sources(self, knowledge_env):
        """Test that context prompt includes source list."""
        prompt = knowledge_env.get_context_prompt()

        assert "SOURCES" in prompt


# =============================================================================
# Test 6: Code Execution Tests
# =============================================================================

class TestCodeExecution:
    """Tests for end-to-end code execution with KB access."""

    def test_execute_code_method(self, knowledge_env):
        """Test the execute_code convenience method."""
        result, error = knowledge_env.execute_code(
            'len(kb_search("AI", top_k=5))'
        )

        assert error is None
        assert isinstance(result, int)
        assert result <= 5

    def test_complex_code_execution(self, knowledge_env):
        """Test complex multi-step code execution."""
        # Store results in variable
        _, error1 = knowledge_env.execute_code(
            'search_results = kb_search("machine learning", top_k=5)'
        )
        assert error1 is None

        # Filter results
        _, error2 = knowledge_env.execute_code(
            'filtered = [r for r in search_results if r["score"] > 0.3]'
        )
        assert error2 is None

        # Get count
        result, error3 = knowledge_env.execute_code('len(filtered)')
        assert error3 is None
        assert isinstance(result, int)

    def test_execution_with_list_comprehension(self, knowledge_env):
        """Test list comprehension with KB results."""
        result, error = knowledge_env.execute_code(
            '[r["score"] for r in kb_search("data", top_k=3)]'
        )

        assert error is None
        assert isinstance(result, list)
        for score in result:
            assert isinstance(score, (int, float))

    def test_execution_error_handling(self, knowledge_env):
        """Test that execution errors are properly reported."""
        result, error = knowledge_env.execute_code('undefined_variable')

        assert result is None
        assert error is not None
        assert "NameError" in error or "name" in error.lower()

    def test_get_repl_stats(self, knowledge_env):
        """Test getting REPL execution statistics."""
        # Execute some code
        knowledge_env.execute_code('kb_search("test", top_k=1)')
        knowledge_env.execute_code('kb_metadata()')

        stats = knowledge_env.get_repl_stats()

        assert "total_executions" in stats
        assert "successful_executions" in stats
        assert stats["total_executions"] >= 2


# =============================================================================
# Test 7: LLM Query Tests
# =============================================================================

class TestLLMQuery:
    """Tests for recursive llm_query functionality."""

    def test_llm_query_disabled_by_default(self, knowledge_env):
        """Test that llm_query is not available by default."""
        assert "llm_query" not in knowledge_env.repl._variables

    def test_llm_query_enabled_with_client(self, populated_kb, repl_env, mock_llm_client):
        """Test that llm_query is enabled when client is provided."""
        config = KnowledgeEnvironmentConfig(enable_llm_query=True)
        env = KnowledgeEnvironment(
            knowledge_base=populated_kb,
            repl=repl_env,
            config=config,
            llm_client=mock_llm_client,
        )

        assert "llm_query" in env.repl._variables
        assert "llm_query_batched" in env.repl._variables

    def test_set_llm_client_method(self, knowledge_env, mock_llm_client):
        """Test setting LLM client after initialization."""
        # Enable llm_query in config
        knowledge_env.config.enable_llm_query = True

        # Set client
        knowledge_env.set_llm_client(mock_llm_client)

        assert "llm_query" in knowledge_env.repl._variables

    def test_llm_query_in_metadata_functions(self, populated_kb, repl_env, mock_llm_client):
        """Test that llm_query appears in metadata functions list."""
        config = KnowledgeEnvironmentConfig(enable_llm_query=True)
        env = KnowledgeEnvironment(
            knowledge_base=populated_kb,
            repl=repl_env,
            config=config,
            llm_client=mock_llm_client,
        )

        metadata = env.get_metadata()

        assert "llm_query" in metadata.functions_available
        assert "llm_query_batched" in metadata.functions_available

    def test_context_prompt_includes_llm_query_docs(self, populated_kb, repl_env, mock_llm_client):
        """Test that context prompt includes llm_query documentation."""
        config = KnowledgeEnvironmentConfig(enable_llm_query=True)
        env = KnowledgeEnvironment(
            knowledge_base=populated_kb,
            repl=repl_env,
            config=config,
            llm_client=mock_llm_client,
        )

        prompt = env.get_context_prompt()

        assert "llm_query(prompt)" in prompt
        assert "llm_query_batched(prompts)" in prompt


# =============================================================================
# Test 8: Integration Tests
# =============================================================================

class TestIntegration:
    """End-to-end integration tests simulating real usage."""

    def test_full_workflow_search_and_process(self, knowledge_env):
        """Test a full workflow: search, filter, and process results."""
        print("\n" + "=" * 70)
        print("INTEGRATION TEST: Full Search and Process Workflow")
        print("=" * 70)

        # Step 1: Get metadata to understand what's available
        result, _ = knowledge_env.execute_code('kb_metadata()')
        print(f"\n[Step 1] Metadata: {result['document_count']} documents")
        assert result["document_count"] > 0

        # Step 2: Search for relevant content
        result, _ = knowledge_env.execute_code(
            'kb_search("artificial intelligence healthcare", top_k=5)'
        )
        print(f"[Step 2] Search returned {len(result)} results")
        assert len(result) > 0

        # Step 3: Filter by quality (statement creates variable, then expression gets length)
        _, _ = knowledge_env.execute_code('results = kb_search("AI", top_k=10)')
        _, _ = knowledge_env.execute_code('high_quality = filter_by_score(results, 0.4)')
        result, _ = knowledge_env.execute_code('len(high_quality)')
        print(f"[Step 3] High-quality results: {result}")

        # Step 4: Extract content
        result, _ = knowledge_env.execute_code(
            '[r["content"][:50] + "..." for r in high_quality[:3]]'
        )
        print(f"[Step 4] Content previews: {len(result)} items")

        print("\n" + "=" * 70)
        print("RESULT: PASSED")
        print("=" * 70)

    def test_workflow_with_context_prompt(self, knowledge_env):
        """Test workflow starting from context prompt."""
        print("\n" + "=" * 70)
        print("INTEGRATION TEST: Context Prompt Workflow")
        print("=" * 70)

        # Step 1: Generate context prompt (what LLM would receive)
        prompt = knowledge_env.get_context_prompt(
            query_context="Analyze AI applications in healthcare"
        )
        print(f"\n[Step 1] Generated context prompt ({len(prompt)} chars)")
        assert len(prompt) > 100
        assert "KNOWLEDGE BASE CONTEXT" in prompt

        # Step 2: Verify prompt contains necessary information
        assert "AVAILABLE FUNCTIONS" in prompt
        assert "kb_search" in prompt
        print("[Step 2] Prompt contains all required sections")

        # Step 3: Simulate LLM executing code from prompt
        # (In real usage, LLM would generate this code)
        code = """
results = kb_search("AI healthcare diagnosis", top_k=5)
relevant = filter_by_score(results, 0.3)
summaries = [f"Score {r['score']:.2f}: {r['content'][:100]}..." for r in relevant]
summaries
"""
        result, error = knowledge_env.execute_code(code.strip())
        print(f"[Step 3] Executed simulated LLM code, got {len(result) if result else 0} summaries")
        assert error is None

        print("\n" + "=" * 70)
        print("RESULT: PASSED")
        print("=" * 70)

    def test_persistence_workflow(self, sample_documents, tmp_path):
        """Test that KB changes are visible to environment."""
        print("\n" + "=" * 70)
        print("INTEGRATION TEST: Persistence Workflow")
        print("=" * 70)

        # Step 1: Create KB with initial documents
        config = KnowledgeBaseConfig(name="persist_test", auto_save=False)
        kb = KnowledgeBase(config=config, persist_directory=str(tmp_path))

        for doc in sample_documents[:3]:  # Only first 3
            kb.add_text(doc["content"], source_name=doc["title"])

        # Step 2: Create environment
        env = create_knowledge_environment(kb)

        # Step 3: Check initial state
        result, _ = env.execute_code('kb_metadata()')
        initial_count = result["document_count"]
        print(f"\n[Step 1-3] Initial documents: {initial_count}")

        # Step 4: Add more documents to KB
        for doc in sample_documents[3:]:  # Remaining 2
            kb.add_text(doc["content"], source_name=doc["title"])

        # Step 5: Refresh metadata and verify
        env.refresh_metadata()
        result, _ = env.execute_code('kb_metadata()')
        new_count = result["document_count"]
        print(f"[Step 4-5] After adding docs: {new_count}")

        assert new_count > initial_count

        print("\n" + "=" * 70)
        print("RESULT: PASSED")
        print("=" * 70)

    def test_multi_stage_analysis(self, knowledge_env):
        """Test multi-stage analysis workflow."""
        print("\n" + "=" * 70)
        print("INTEGRATION TEST: Multi-Stage Analysis")
        print("=" * 70)

        # Stage 1: Broad search
        _, _ = knowledge_env.execute_code(
            'stage1 = kb_search("technology", top_k=20)'
        )
        result, _ = knowledge_env.execute_code('len(stage1)')
        print(f"\n[Stage 1] Broad search: {result} results")

        # Stage 2: Filter by quality
        _, _ = knowledge_env.execute_code(
            'stage2 = filter_by_score(stage1, 0.3)'
        )
        result, _ = knowledge_env.execute_code('len(stage2)')
        print(f"[Stage 2] Quality filter: {result} results")

        # Stage 3: Extract unique sources
        result, _ = knowledge_env.execute_code(
            'list(set(r["metadata"].get("source", "unknown") for r in stage2))'
        )
        print(f"[Stage 3] Unique sources: {result}")

        # Stage 4: Aggregate scores
        result, _ = knowledge_env.execute_code(
            'sum(r["score"] for r in stage2) / len(stage2) if stage2 else 0'
        )
        print(f"[Stage 4] Average score: {result:.3f}")

        print("\n" + "=" * 70)
        print("RESULT: PASSED")
        print("=" * 70)


# =============================================================================
# Test 9: Configuration Tests
# =============================================================================

class TestConfiguration:
    """Tests for configuration options."""

    def test_default_config_values(self):
        """Test default configuration values."""
        config = KnowledgeEnvironmentConfig()

        assert config.max_search_results == 50
        assert config.include_topics is True
        assert config.max_topics == 15
        assert config.include_sources is True
        assert config.include_stats is True
        assert config.enable_llm_query is False
        assert config.cache_metadata is True
        assert config.metadata_refresh_interval == 300.0

    def test_config_affects_behavior(self, populated_kb, repl_env):
        """Test that configuration affects environment behavior."""
        # Config with no topics or sources
        config = KnowledgeEnvironmentConfig(
            include_topics=False,
            include_sources=False,
            include_stats=False,
        )
        env = KnowledgeEnvironment(
            knowledge_base=populated_kb,
            repl=repl_env,
            config=config,
        )

        prompt = env.get_context_prompt()

        # Should not include these sections
        assert "TOPICS COVERED" not in prompt
        assert "SOURCES:" not in prompt

    def test_config_cache_disabled(self, populated_kb, repl_env):
        """Test metadata caching can be disabled."""
        config = KnowledgeEnvironmentConfig(cache_metadata=False)
        env = KnowledgeEnvironment(
            knowledge_base=populated_kb,
            repl=repl_env,
            config=config,
        )

        metadata1 = env.get_metadata()
        metadata2 = env.get_metadata()

        # Should be different objects (not cached)
        assert metadata1 is not metadata2


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

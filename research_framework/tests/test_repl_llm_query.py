"""
Tests for native llm_query() functionality in REPLEnvironment.

This module tests the recursive LLM calling capability that is native to
the REPLEnvironment class, enabling the core RLM paradigm feature.

Test Categories:
----------------
1. LLM Client Setup: set_llm_client, has_llm_client
2. llm_query Function: Basic calls, error handling
3. llm_query_batched: Batch processing
4. Integration: End-to-end REPL execution with llm_query
5. Statistics: llm_query tracking

Usage:
------
    pytest tests/test_repl_llm_query.py -v
"""

import pytest
from typing import Dict, List, Any, Optional
from unittest.mock import MagicMock, AsyncMock, patch
import asyncio

from core.memory_manager import REPLEnvironment


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def repl():
    """Create a fresh REPLEnvironment."""
    return REPLEnvironment()


@pytest.fixture
def mock_async_llm_client():
    """Create a mock async LLM client."""
    client = MagicMock()

    async def mock_chat(messages):
        # Extract prompt from messages
        prompt = messages[-1]["content"] if messages else ""
        return {
            "content": f"Mock response to: {prompt[:50]}...",
            "usage": {"total_tokens": 100}
        }

    client.chat = AsyncMock(side_effect=mock_chat)
    return client


@pytest.fixture
def mock_sync_llm_client():
    """Create a mock sync LLM client with generate method."""
    client = MagicMock()
    client.generate = MagicMock(return_value="Sync mock response")
    return client


@pytest.fixture
def repl_with_llm(repl, mock_async_llm_client):
    """Create REPLEnvironment with LLM client configured."""
    repl.set_llm_client(mock_async_llm_client)
    return repl


# =============================================================================
# Test 1: LLM Client Setup
# =============================================================================

class TestLLMClientSetup:
    """Tests for LLM client configuration."""

    def test_no_llm_client_by_default(self, repl):
        """Test that llm_query is not available by default."""
        assert not repl.has_llm_client()
        assert "llm_query" not in repl._variables
        assert "llm_query" not in repl._safe_builtins

    def test_set_llm_client(self, repl, mock_async_llm_client):
        """Test setting LLM client enables llm_query."""
        repl.set_llm_client(mock_async_llm_client)

        assert repl.has_llm_client()
        assert "llm_query" in repl._variables
        assert "llm_query_batched" in repl._variables
        assert "llm_query" in repl._safe_builtins
        assert "llm_query_batched" in repl._safe_builtins

    def test_init_with_llm_client(self, mock_async_llm_client):
        """Test initializing REPL with LLM client."""
        repl = REPLEnvironment(llm_client=mock_async_llm_client)

        assert repl.has_llm_client()
        assert "llm_query" in repl._variables

    def test_has_llm_client_false_without_client(self, repl):
        """Test has_llm_client returns False when no client set."""
        assert repl.has_llm_client() is False

    def test_has_llm_client_true_with_client(self, repl_with_llm):
        """Test has_llm_client returns True when client is set."""
        assert repl_with_llm.has_llm_client() is True


# =============================================================================
# Test 2: llm_query Function
# =============================================================================

class TestLLMQuery:
    """Tests for the llm_query function."""

    def test_llm_query_via_repl_execute(self, repl_with_llm):
        """Test calling llm_query through REPL execution."""
        result, error = repl_with_llm.execute('llm_query("Hello, world!")')

        assert error is None
        assert result is not None
        assert isinstance(result, str)
        assert "Mock response" in result

    def test_llm_query_with_context_variable(self, repl_with_llm):
        """Test llm_query with context from variable."""
        repl_with_llm.set_context("text", "Some important research findings")
        result, error = repl_with_llm.execute('llm_query(f"Summarize: {text}")')

        assert error is None
        assert isinstance(result, str)

    def test_llm_query_with_system_prompt(self, repl_with_llm):
        """Test llm_query with system prompt parameter."""
        result, error = repl_with_llm.execute(
            'llm_query("Explain AI", system_prompt="Be concise")'
        )

        assert error is None
        assert isinstance(result, str)

    def test_llm_query_updates_stats(self, repl_with_llm):
        """Test that llm_query updates statistics."""
        initial_stats = repl_with_llm.get_llm_query_stats()
        assert initial_stats['llm_queries'] == 0

        repl_with_llm.execute('llm_query("Test query")')

        stats = repl_with_llm.get_llm_query_stats()
        assert stats['llm_queries'] == 1

    def test_llm_query_not_available_without_client(self, repl):
        """Test that llm_query fails without client."""
        result, error = repl.execute('llm_query("Hello")')

        assert error is not None
        assert "NameError" in error or "name" in error.lower()

    def test_llm_query_with_sync_client(self, repl, mock_sync_llm_client):
        """Test llm_query with synchronous client."""
        repl.set_llm_client(mock_sync_llm_client)
        result, error = repl.execute('llm_query("Test")')

        assert error is None
        assert result == "Sync mock response"


# =============================================================================
# Test 3: llm_query_batched Function
# =============================================================================

class TestLLMQueryBatched:
    """Tests for the llm_query_batched function."""

    def test_llm_query_batched_basic(self, repl_with_llm):
        """Test basic batch processing."""
        result, error = repl_with_llm.execute('''
llm_query_batched(["Query 1", "Query 2", "Query 3"])
''')

        assert error is None
        assert isinstance(result, list)
        assert len(result) == 3

    def test_llm_query_batched_with_system_prompt(self, repl_with_llm):
        """Test batch processing with shared system prompt."""
        result, error = repl_with_llm.execute('''
llm_query_batched(
    ["Query 1", "Query 2"],
    system_prompt="Be brief"
)
''')

        assert error is None
        assert isinstance(result, list)
        assert len(result) == 2

    def test_llm_query_batched_updates_stats(self, repl_with_llm):
        """Test that batched queries update statistics correctly."""
        repl_with_llm.execute('llm_query_batched(["Q1", "Q2", "Q3"])')

        stats = repl_with_llm.get_llm_query_stats()
        assert stats['llm_queries'] == 3  # One for each query

    def test_llm_query_batched_empty_list(self, repl_with_llm):
        """Test batch processing with empty list."""
        result, error = repl_with_llm.execute('llm_query_batched([])')

        assert error is None
        assert result == []

    def test_llm_query_batched_with_context(self, repl_with_llm):
        """Test batch processing using context variables."""
        repl_with_llm.set_context("chunks", ["Chunk 1 content", "Chunk 2 content"])
        result, error = repl_with_llm.execute('''
llm_query_batched([f"Summarize: {c}" for c in chunks])
''')

        assert error is None
        assert isinstance(result, list)
        assert len(result) == 2


# =============================================================================
# Test 4: Integration Tests
# =============================================================================

class TestIntegration:
    """End-to-end integration tests."""

    def test_workflow_search_and_summarize(self, repl_with_llm):
        """Test a workflow that searches and summarizes."""
        # Set up context
        repl_with_llm.set_context("documents", [
            {"content": "AI is transforming healthcare through diagnosis"},
            {"content": "Machine learning improves drug discovery"},
            {"content": "Neural networks enable medical imaging analysis"},
        ])

        # Search and summarize
        _, error1 = repl_with_llm.execute('''
relevant = [d["content"] for d in documents if "AI" in d["content"] or "learning" in d["content"]]
''')
        assert error1 is None

        result, error2 = repl_with_llm.execute('''
llm_query_batched([f"Key point: {doc}" for doc in relevant])
''')
        assert error2 is None
        assert isinstance(result, list)

    def test_recursive_summarization(self, repl_with_llm):
        """Test recursive summarization pattern."""
        repl_with_llm.set_context("sections", [
            "Section 1: Introduction to AI",
            "Section 2: Methods and approaches",
            "Section 3: Results and findings",
        ])

        # First level: summarize each section
        _, error1 = repl_with_llm.execute('''
summaries = llm_query_batched([f"Summarize: {s}" for s in sections])
''')
        assert error1 is None

        # Second level: combine summaries
        result, error2 = repl_with_llm.execute('''
llm_query("Combine these summaries into one: " + " | ".join(summaries))
''')
        assert error2 is None
        assert isinstance(result, str)

    def test_clear_preserves_llm_client(self, repl_with_llm):
        """Test that clear() preserves LLM client configuration."""
        assert repl_with_llm.has_llm_client()

        repl_with_llm.clear()

        # LLM client should still be available
        assert repl_with_llm.has_llm_client()
        assert "llm_query" in repl_with_llm._safe_builtins

    def test_stats_include_llm_query_info(self, repl_with_llm):
        """Test that get_stats includes llm_query information."""
        repl_with_llm.execute('llm_query("Test")')

        stats = repl_with_llm.get_stats()

        assert 'llm_queries' in stats
        assert 'llm_query_tokens' in stats
        assert 'llm_query_enabled' in stats
        assert stats['llm_query_enabled'] is True


# =============================================================================
# Test 5: Error Handling
# =============================================================================

class TestErrorHandling:
    """Tests for error handling in llm_query."""

    def test_llm_query_handles_client_error(self, repl):
        """Test that llm_query handles client errors gracefully."""
        # Create a client that raises an error
        error_client = MagicMock()
        error_client.chat = AsyncMock(side_effect=Exception("API Error"))

        repl.set_llm_client(error_client)
        result, error = repl.execute('llm_query("Test")')

        # Should not raise, but return error message
        assert error is None
        assert "[ERROR:" in result

    def test_llm_query_with_invalid_client(self, repl):
        """Test setting an invalid client (no chat/generate method)."""
        invalid_client = MagicMock(spec=[])  # No methods
        repl.set_llm_client(invalid_client)

        result, error = repl.execute('llm_query("Test")')

        # Should handle gracefully
        assert error is None
        assert "[ERROR:" in result


# =============================================================================
# Test 6: Statistics
# =============================================================================

class TestStatistics:
    """Tests for llm_query statistics tracking."""

    def test_initial_stats_zero(self, repl_with_llm):
        """Test initial statistics are zero."""
        stats = repl_with_llm.get_llm_query_stats()

        assert stats['llm_queries'] == 0
        assert stats['llm_query_tokens'] == 0

    def test_stats_increment_on_query(self, repl_with_llm):
        """Test statistics increment with each query."""
        repl_with_llm.execute('llm_query("Query 1")')
        repl_with_llm.execute('llm_query("Query 2")')

        stats = repl_with_llm.get_llm_query_stats()
        assert stats['llm_queries'] == 2

    def test_stats_track_tokens(self, repl_with_llm):
        """Test token tracking in statistics."""
        repl_with_llm.execute('llm_query("Test")')

        stats = repl_with_llm.get_llm_query_stats()
        # Mock returns 100 tokens per query
        assert stats['llm_query_tokens'] >= 0

    def test_stats_persist_after_clear(self, repl_with_llm):
        """Test that stats are reset after clear."""
        repl_with_llm.execute('llm_query("Test")')
        assert repl_with_llm.get_llm_query_stats()['llm_queries'] == 1

        repl_with_llm.clear()

        # Stats should be reset
        assert repl_with_llm.get_llm_query_stats()['llm_queries'] == 0


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

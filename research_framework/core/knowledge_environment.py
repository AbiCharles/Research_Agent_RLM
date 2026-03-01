"""
Knowledge Environment Bridge for RLM-style Context Access.

Bridges Knowledge Base (vector storage) with REPL Environment (code execution),
enabling the RLM paradigm where LLMs receive metadata and write code to access
content rather than receiving raw documents.

Exposes: kb_search, kb_metadata, load_chunk, filter_by_score, list_sources
llm_query is delegated to REPL's native implementation via set_llm_client().
"""

import asyncio
import re
from dataclasses import dataclass, field
from typing import (
    List,
    Dict,
    Any,
    Optional,
    Callable,
    Awaitable,
    Set,
    Tuple,
    Union,
)
from collections import Counter
from datetime import datetime

from utils.logger import get_logger


logger = get_logger(__name__)


@dataclass
class KnowledgeEnvironmentConfig:
    """Configuration for KnowledgeEnvironment bridge."""
    max_search_results: int = 50
    include_topics: bool = True
    max_topics: int = 15
    include_sources: bool = True
    include_stats: bool = True
    enable_llm_query: bool = False
    cache_metadata: bool = True
    metadata_refresh_interval: float = 300.0


@dataclass
class ContextMetadata:
    """Structured metadata about available KB content without loading actual data."""

    document_count: int
    source_count: int
    total_characters: int
    topics: List[str]
    sources: List[str]
    embedding_dimension: Optional[int]
    last_updated: Optional[str]
    functions_available: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary format."""
        return {
            "document_count": self.document_count,
            "source_count": self.source_count,
            "total_characters": self.total_characters,
            "topics": self.topics,
            "sources": self.sources,
            "embedding_dimension": self.embedding_dimension,
            "last_updated": self.last_updated,
            "functions_available": self.functions_available,
        }

    def to_summary_string(self) -> str:
        """Generate a human-readable summary of the metadata."""
        lines = [
            f"Documents: {self.document_count} chunks",
            f"Sources: {self.source_count} files",
            f"Total size: ~{self.total_characters:,} characters",
        ]
        if self.topics:
            lines.append(f"Topics: {', '.join(self.topics[:10])}")
        if self.last_updated:
            lines.append(f"Last updated: {self.last_updated}")
        return "\n".join(lines)


# =============================================================================
# Topic Extractor
# =============================================================================

class TopicExtractor:
    """Extracts topics using TF-IDF-inspired term frequency analysis."""

    # Common English stopwords to filter out
    STOPWORDS: Set[str] = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
        'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
        'could', 'should', 'may', 'might', 'must', 'shall', 'can', 'need',
        'this', 'that', 'these', 'those', 'it', 'its', 'they', 'them', 'their',
        'he', 'she', 'him', 'her', 'his', 'we', 'us', 'our', 'you', 'your',
        'i', 'me', 'my', 'mine', 'who', 'what', 'which', 'when', 'where', 'why',
        'how', 'all', 'each', 'every', 'both', 'few', 'more', 'most', 'other',
        'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than',
        'too', 'very', 'just', 'also', 'now', 'then', 'here', 'there', 'where',
        'about', 'after', 'before', 'during', 'while', 'above', 'below',
        'between', 'through', 'into', 'over', 'under', 'again', 'further',
        'once', 'any', 'many', 'much', 'new', 'old', 'used', 'using', 'use',
        'based', 'well', 'however', 'therefore', 'thus', 'hence', 'although',
    }

    def __init__(
        self,
        min_term_length: int = 3,
        max_terms: int = 20,
        custom_stopwords: Optional[Set[str]] = None,
    ):
        self.min_term_length = min_term_length
        self.max_terms = max_terms
        self.stopwords = self.STOPWORDS.copy()
        if custom_stopwords:
            self.stopwords.update(custom_stopwords)

    def extract_topics(
        self,
        documents: List[Dict[str, Any]],
        include_bigrams: bool = True,
    ) -> List[str]:
        """Extract topics from documents. Returns list sorted by frequency."""
        # Combine all document content
        all_text = " ".join(
            doc.get("content", "") for doc in documents
        ).lower()

        # Extract single words (unigrams)
        words = re.findall(r'\b[a-z]+\b', all_text)
        filtered_words = [
            w for w in words
            if w not in self.stopwords and len(w) >= self.min_term_length
        ]

        # Count word frequencies
        word_counts = Counter(filtered_words)

        # Extract bigrams if requested
        bigram_counts = Counter()
        if include_bigrams:
            bigrams = self._extract_bigrams(words)
            bigram_counts = Counter(bigrams)

        # Combine and rank topics
        topics = []

        # Add top bigrams (often more meaningful than single words)
        for bigram, count in bigram_counts.most_common(self.max_terms // 2):
            if count >= 2:  # Minimum occurrence threshold
                topics.append(bigram)

        # Add top unigrams not already covered by bigrams
        bigram_words = set(" ".join(topics).split())
        for word, count in word_counts.most_common(self.max_terms):
            if word not in bigram_words and count >= 3:
                topics.append(word)
            if len(topics) >= self.max_terms:
                break

        return topics[:self.max_terms]

    def _extract_bigrams(self, words: List[str]) -> List[str]:
        """Extract meaningful two-word phrases from a word list."""
        bigrams = []
        for i in range(len(words) - 1):
            w1, w2 = words[i], words[i + 1]
            # Filter out bigrams with stopwords or short words
            if (w1 not in self.stopwords and
                w2 not in self.stopwords and
                len(w1) >= self.min_term_length and
                len(w2) >= self.min_term_length):
                bigrams.append(f"{w1} {w2}")
        return bigrams

    def extract_from_metadata(
        self,
        documents: List[Dict[str, Any]],
    ) -> List[str]:
        """Extract topics from document metadata (titles, tags, etc.)."""
        topics = []

        for doc in documents:
            metadata = doc.get("metadata", {})

            # Extract from title
            title = metadata.get("title", "")
            if title:
                title_words = re.findall(r'\b[a-z]+\b', title.lower())
                topics.extend([
                    w for w in title_words
                    if w not in self.stopwords and len(w) >= self.min_term_length
                ])

            # Extract from tags/categories if present
            tags = metadata.get("tags", [])
            if isinstance(tags, list):
                topics.extend(tags)

            # Extract from topic field if present
            topic = metadata.get("topic", "")
            if topic:
                topics.append(topic.lower())

        # Count and return unique topics
        topic_counts = Counter(topics)
        return [
            topic for topic, _ in topic_counts.most_common(self.max_terms)
        ]


# =============================================================================
# Knowledge Environment Bridge
# =============================================================================

class KnowledgeEnvironment:
    """
    Bridge between Knowledge Base and REPL Environment for RLM-style context access.

    Exposes KB functionality as REPL functions (kb_search, kb_metadata, load_chunk,
    filter_by_score, list_sources). LLM queries are delegated to REPL's native
    llm_query implementation via set_llm_client().

    Attributes:
        knowledge_base: The underlying Knowledge Base instance.
        repl: The REPL Environment for code execution.
        config: Configuration settings.

    Example:
        >>> env = KnowledgeEnvironment(kb, repl, llm_client=client,
        ...     config=KnowledgeEnvironmentConfig(enable_llm_query=True))
        >>> prompt = env.get_context_prompt()  # Metadata, not raw content
        >>> # LLM writes: results = kb_search("AI", top_k=5)
        >>> # LLM writes: summary = llm_query(f"Summarize: {results[0]['content']}")
    """

    def __init__(
        self,
        knowledge_base: Any,
        repl: Any,
        config: Optional[KnowledgeEnvironmentConfig] = None,
        llm_client: Optional[Any] = None,
    ):
        """
        Initialize the Knowledge Environment bridge.

        Args:
            knowledge_base: KnowledgeBase instance.
            repl: REPLEnvironment instance.
            config: Optional configuration.
            llm_client: Optional LLM client (delegates to REPL's native llm_query).
        """
        if knowledge_base is None:
            raise ValueError("knowledge_base cannot be None")
        if repl is None:
            raise ValueError("repl cannot be None")

        self.knowledge_base = knowledge_base
        self.repl = repl
        self.config = config or KnowledgeEnvironmentConfig()
        self._llm_client = llm_client

        # Initialize topic extractor
        self._topic_extractor = TopicExtractor(
            max_terms=self.config.max_topics
        )

        # Metadata cache
        self._cached_metadata: Optional[ContextMetadata] = None
        self._metadata_timestamp: Optional[float] = None

        # Setup the REPL environment with KB access functions
        self._setup_environment()

        # Set LLM client if provided (delegates to REPL's native llm_query)
        if llm_client and self.config.enable_llm_query:
            self.set_llm_client(llm_client)

        logger.info(
            f"KnowledgeEnvironment initialized: "
            f"kb={knowledge_base.config.name}, "
            f"llm_query={'enabled' if self.repl.has_llm_client() else 'disabled'}"
        )

    def _setup_environment(self) -> None:
        """
        Set up the REPL environment with Knowledge Base access functions.

        Registers kb_search, kb_metadata, load_chunk, filter_by_score, and
        list_sources in the REPL namespace. llm_query is handled separately
        via set_llm_client() which delegates to REPL's native implementation.
        """
        def kb_search(
            query: str,
            top_k: int = 10,
            min_score: float = 0.0,
            filter_metadata: Optional[Dict[str, Any]] = None,
        ) -> List[Dict[str, Any]]:
            """Search KB for relevant documents. Returns list with id, content, score, metadata."""
            actual_top_k = min(top_k, self.config.max_search_results)
            results = self.knowledge_base.query(
                query=query,
                top_k=actual_top_k,
                min_score=min_score,
                filter_metadata=filter_metadata,
            )
            logger.debug(f"kb_search: {len(results)} results for '{query[:30]}...'")
            return results

        def kb_metadata() -> Dict[str, Any]:
            """Get KB metadata (document_count, topics, sources) without loading content."""
            return self.get_metadata().to_dict()

        def load_chunk(chunk_id: str) -> Optional[Dict[str, Any]]:
            """Load a specific document chunk by ID. Returns dict or None."""
            return self.knowledge_base.get_document(chunk_id)

        def filter_by_score(
            results: List[Dict[str, Any]],
            min_score: float,
        ) -> List[Dict[str, Any]]:
            """Filter search results by minimum similarity score."""
            return [r for r in results if r.get("score", 0) >= min_score]

        def list_sources() -> List[str]:
            """Get list of all source files in the knowledge base."""
            return self.knowledge_base.list_sources()

        # Register functions in REPL namespace
        self.repl._variables["kb_search"] = kb_search
        self.repl._variables["kb_metadata"] = kb_metadata
        self.repl._variables["load_chunk"] = load_chunk
        self.repl._variables["filter_by_score"] = filter_by_score
        self.repl._variables["list_sources"] = list_sources

        self.repl._safe_builtins["kb_search"] = kb_search
        self.repl._safe_builtins["kb_metadata"] = kb_metadata
        self.repl._safe_builtins["load_chunk"] = load_chunk
        self.repl._safe_builtins["filter_by_score"] = filter_by_score
        self.repl._safe_builtins["list_sources"] = list_sources

        logger.debug("REPL configured with KB functions")

    def set_llm_client(self, llm_client: Any) -> None:
        """
        Set the LLM client for recursive queries.

        Delegates to REPL's native llm_query implementation.

        Args:
            llm_client: LLM client instance with chat() or generate() method.
        """
        self._llm_client = llm_client

        if self.config.enable_llm_query:
            self.repl.set_llm_client(llm_client)
            logger.info("LLM client set via REPL's native llm_query")
        else:
            logger.warning(
                "LLM client set but enable_llm_query=False in config"
            )

    def get_metadata(self, force_refresh: bool = False) -> ContextMetadata:
        """Get structured metadata about available KB content (cached)."""
        import time

        # Check cache validity
        if (not force_refresh and
            self.config.cache_metadata and
            self._cached_metadata is not None and
            self._metadata_timestamp is not None):
            cache_age = time.time() - self._metadata_timestamp
            if cache_age < self.config.metadata_refresh_interval:
                return self._cached_metadata

        stats = self.knowledge_base.get_stats()

        topics = []
        if self.config.include_topics:
            documents = list(self.knowledge_base.vector_store.documents.values())
            topics = self._topic_extractor.extract_topics(documents)

        sources = []
        if self.config.include_sources:
            sources = self.knowledge_base.list_sources()

        # Determine available functions (check REPL's native llm_query)
        functions = ["kb_search", "kb_metadata", "load_chunk", "filter_by_score", "list_sources"]
        if self.repl.has_llm_client():
            functions.extend(["llm_query", "llm_query_batched"])

        # Build metadata
        metadata = ContextMetadata(
            document_count=stats.document_count,
            source_count=stats.source_count,
            total_characters=stats.total_characters,
            topics=topics,
            sources=sources,
            embedding_dimension=stats.embedding_dimension,
            last_updated=stats.last_updated,
            functions_available=functions,
        )

        # Cache the result
        if self.config.cache_metadata:
            self._cached_metadata = metadata
            self._metadata_timestamp = time.time()

        logger.debug(f"Generated metadata: {metadata.document_count} docs, {len(topics)} topics")

        return metadata

    def refresh_metadata(self) -> ContextMetadata:
        """Force refresh of cached metadata."""
        return self.get_metadata(force_refresh=True)

    def get_context_prompt(
        self,
        include_examples: bool = True,
        query_context: Optional[str] = None,
    ) -> str:
        """
        Generate a context prompt describing available KB resources.

        Returns metadata and available functions (not raw content) so LLM
        can write code to selectively access what it needs.
        """
        metadata = self.get_metadata()

        # Build the prompt
        lines = [
            "=" * 60,
            "KNOWLEDGE BASE CONTEXT (RLM Environment)",
            "=" * 60,
            "",
            "You have access to a knowledge base via code execution.",
            "DO NOT expect raw content in this prompt - use functions to access data.",
            "",
        ]

        # Metadata section
        if self.config.include_stats:
            lines.extend([
                "AVAILABLE CONTENT:",
                "-" * 40,
                f"  Documents: {metadata.document_count} chunks",
                f"  Sources: {metadata.source_count} files",
                f"  Total size: ~{metadata.total_characters:,} characters",
            ])
            if metadata.last_updated:
                lines.append(f"  Last updated: {metadata.last_updated}")
            lines.append("")

        # Topics section
        if self.config.include_topics and metadata.topics:
            lines.extend([
                "TOPICS COVERED:",
                "-" * 40,
                f"  {', '.join(metadata.topics[:self.config.max_topics])}",
                "",
            ])

        # Sources section
        if self.config.include_sources and metadata.sources:
            lines.extend([
                "SOURCES:",
                "-" * 40,
            ])
            for source in metadata.sources[:10]:  # Limit to first 10
                lines.append(f"  - {source}")
            if len(metadata.sources) > 10:
                lines.append(f"  ... and {len(metadata.sources) - 10} more")
            lines.append("")

        # Functions section
        lines.extend([
            "AVAILABLE FUNCTIONS:",
            "-" * 40,
            "",
            "kb_search(query, top_k=10, min_score=0.0)",
            "    Search for relevant documents.",
            "    Returns: List of {id, content, score, metadata}",
            "",
            "kb_metadata()",
            "    Get metadata about available content.",
            "    Returns: Dict with counts, topics, sources",
            "",
            "load_chunk(chunk_id)",
            "    Load a specific document by ID.",
            "    Returns: {id, content, metadata} or None",
            "",
            "filter_by_score(results, min_score)",
            "    Filter search results by score threshold.",
            "    Returns: Filtered list of results",
            "",
            "list_sources()",
            "    Get list of all source files.",
            "    Returns: List of source names",
            "",
        ])

        # Add llm_query docs if available
        if "llm_query" in metadata.functions_available:
            lines.extend([
                "llm_query(prompt)",
                "    Make a recursive LLM query.",
                "    Returns: LLM response string",
                "",
                "llm_query_batched(prompts)",
                "    Make multiple LLM queries concurrently.",
                "    Returns: List of response strings",
                "",
            ])

        # Query context section
        if query_context:
            lines.extend([
                "CURRENT TASK:",
                "-" * 40,
                f"  {query_context}",
                "",
            ])

        # Examples section
        if include_examples:
            lines.extend([
                "EXAMPLE USAGE:",
                "-" * 40,
                "```repl",
                "# Search for relevant content",
                "results = kb_search(\"AI in healthcare\", top_k=5)",
                "",
                "# Check what we found",
                "print(f\"Found {len(results)} results\")",
                "",
                "# Filter high-quality results",
                "good_results = filter_by_score(results, 0.6)",
                "",
                "# Process each result",
                "for r in good_results:",
                "    print(f\"Score: {r['score']:.2f}\")",
                "    print(f\"Content: {r['content'][:200]}...\")",
            ])

            # Add llm_query example if available
            if "llm_query" in metadata.functions_available:
                lines.extend([
                    "",
                    "# Summarize a specific result",
                    "summary = llm_query(f\"Summarize: {good_results[0]['content']}\")",
                ])

            lines.extend([
                "```",
                "",
            ])

        # Instructions
        lines.extend([
            "INSTRUCTIONS:",
            "-" * 40,
            "1. Use kb_search() to find relevant documents",
            "2. Filter results by score for quality",
            "3. Process content programmatically",
            "4. Use llm_query() for sub-tasks if available",
            "5. Return your final answer using FINAL() or FINAL_VAR()",
            "",
            "=" * 60,
        ])

        return "\n".join(lines)

    def execute_code(self, code: str) -> Tuple[Any, Optional[str]]:
        """Execute code in REPL with KB access. Returns (result, error)."""
        return self.repl.execute(code)

    def get_repl_stats(self) -> Dict[str, Any]:
        """Get REPL execution statistics."""
        return self.repl.get_stats()


def create_knowledge_environment(
    knowledge_base: Any,
    repl: Optional[Any] = None,
    config: Optional[KnowledgeEnvironmentConfig] = None,
    llm_client: Optional[Any] = None,
) -> KnowledgeEnvironment:
    """Factory function. Creates default REPL if not provided."""
    if repl is None:
        from core.memory_manager import REPLEnvironment
        repl = REPLEnvironment()

    return KnowledgeEnvironment(
        knowledge_base=knowledge_base,
        repl=repl,
        config=config,
        llm_client=llm_client,
    )


__all__ = [
    "KnowledgeEnvironmentConfig",
    "ContextMetadata",
    "TopicExtractor",
    "KnowledgeEnvironment",
    "create_knowledge_environment",
]

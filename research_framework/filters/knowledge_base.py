"""
Knowledge Base module for the Semantic Layer.

This module provides a unified interface for managing organizational knowledge,
combining document loading, embedding, and vector storage into a cohesive
knowledge management system.

The Knowledge Base is the core component of Stage 1 (Semantic Layer) in the
Complete Knowledge Pipeline. It enables research agents to access and query
proprietary organizational documents and data.

Architecture:
    - KnowledgeBaseConfig: Configuration for knowledge base settings
    - KnowledgeBase: Main class for knowledge management
    - KnowledgeBaseStats: Statistics about the knowledge base

Key Features:
    - Multi-format document ingestion (PDF, Word, Excel, CSV, text)
    - Automatic chunking and embedding
    - Semantic similarity search via FAISS
    - Metadata filtering for targeted retrieval
    - Persistence for production use
    - Directory watching for automatic updates (optional)

Usage:
    >>> from filters.knowledge_base import KnowledgeBase, KnowledgeBaseConfig
    >>>
    >>> # Create knowledge base
    >>> config = KnowledgeBaseConfig(
    ...     name="company_docs",
    ...     persist_directory="./kb_data"
    ... )
    >>> kb = KnowledgeBase(config)
    >>>
    >>> # Add documents
    >>> kb.add_directory("./documents/")
    >>> kb.add_document("important_report.pdf")
    >>>
    >>> # Query
    >>> results = kb.query("AI healthcare applications", top_k=5)

Example:
    >>> # Full workflow
    >>> kb = KnowledgeBase(name="research_kb")
    >>>
    >>> # Add various sources
    >>> kb.add_document("report.pdf", metadata={"type": "report"})
    >>> kb.add_directory("./research_papers/")
    >>>
    >>> # Query with filters
    >>> results = kb.query(
    ...     "machine learning diagnosis",
    ...     top_k=10,
    ...     filter_metadata={"type": "report"}
    ... )
    >>> for r in results:
    ...     print(f"Score: {r['score']:.3f} - {r['content'][:100]}")
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Set
from datetime import datetime
from uuid import uuid4

from filters.vector_store import FAISSVectorStore, VectorStoreConfig
from filters.document_loaders import (
    DocumentLoaderConfig,
    LoadedDocument,
    DirectoryLoader,
    get_loader_for_file,
    load_document,
)
from utils.logger import get_logger


logger = get_logger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class KnowledgeBaseConfig:
    """
    Configuration for Knowledge Base initialization.

    This dataclass holds all settings for the knowledge base, including
    vector store configuration, document loading options, and persistence.

    Attributes:
        name: Unique name for the knowledge base (used for persistence).
        persist_directory: Directory for saving/loading the knowledge base.
            If None, the knowledge base is in-memory only.
        embedding_model: Name of the sentence-transformers model.
            - "all-MiniLM-L6-v2" (default): Fast, good quality
            - "all-mpnet-base-v2": Higher quality, slower
        chunk_size: Maximum characters per document chunk.
        chunk_overlap: Characters to overlap between chunks.
        index_type: FAISS index type ("flat", "ivf", "hnsw").
        auto_save: Whether to automatically save after modifications.
        supported_extensions: Set of file extensions to process.

    Example:
        >>> config = KnowledgeBaseConfig(
        ...     name="company_kb",
        ...     persist_directory="./data/kb",
        ...     embedding_model="all-MiniLM-L6-v2",
        ...     chunk_size=1000
        ... )
    """

    name: str = "default"
    persist_directory: Optional[str] = None
    embedding_model: str = "all-MiniLM-L6-v2"
    chunk_size: int = 1500
    chunk_overlap: int = 200
    index_type: str = "flat"
    auto_save: bool = True
    supported_extensions: Set[str] = field(default_factory=lambda: {
        ".pdf", ".docx", ".xlsx", ".xls", ".csv", ".txt", ".md", ".json"
    })


@dataclass
class KnowledgeBaseStats:
    """
    Statistics about the knowledge base.

    Attributes:
        document_count: Total number of document chunks.
        source_count: Number of unique source files.
        total_characters: Total characters across all documents.
        embedding_dimension: Dimension of embedding vectors.
        index_type: Type of vector index.
        last_updated: Timestamp of last modification.
    """

    document_count: int
    source_count: int
    total_characters: int
    embedding_dimension: Optional[int]
    index_type: str
    last_updated: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "document_count": self.document_count,
            "source_count": self.source_count,
            "total_characters": self.total_characters,
            "embedding_dimension": self.embedding_dimension,
            "index_type": self.index_type,
            "last_updated": self.last_updated,
        }


# =============================================================================
# Knowledge Base
# =============================================================================

class KnowledgeBase:
    """
    Unified knowledge management system for the Semantic Layer.

    The KnowledgeBase provides a high-level interface for managing organizational
    knowledge. It combines document loading, text chunking, embedding generation,
    and vector storage into a single cohesive system.

    This class is the primary entry point for Stage 1 (Semantic Layer) of the
    Complete Knowledge Pipeline. Research agents use it to access proprietary
    documents and data during investigations.

    Features:
        - Multi-format document ingestion (PDF, Word, Excel, CSV, text)
        - Automatic text chunking for optimal retrieval
        - Semantic similarity search via FAISS
        - Metadata filtering for targeted queries
        - Persistence for production deployments
        - Source tracking for citations

    Attributes:
        config: Knowledge base configuration.
        vector_store: Underlying FAISS vector store.
        sources: Set of source file paths that have been added.
        created_at: Timestamp of creation.

    Example:
        >>> # Create and populate
        >>> kb = KnowledgeBase(name="research")
        >>> kb.add_directory("./papers/")
        >>> kb.add_document("findings.pdf")
        >>>
        >>> # Query
        >>> results = kb.query("neural network architecture", top_k=5)
        >>> for r in results:
        ...     print(f"{r['score']:.2f}: {r['content'][:100]}...")
        >>>
        >>> # Save for later
        >>> kb.save()

    Note:
        For large knowledge bases (>100K documents), consider using
        index_type="ivf" for faster search at the cost of some accuracy.
    """

    def __init__(
        self,
        config: Optional[KnowledgeBaseConfig] = None,
        name: Optional[str] = None,
        persist_directory: Optional[str] = None,
    ):
        """
        Initialize the Knowledge Base.

        Args:
            config: Full configuration object. If provided, name and
                persist_directory parameters are ignored.
            name: Name for the knowledge base (shortcut for config.name).
            persist_directory: Directory for persistence (shortcut).

        Example:
            >>> # Using config
            >>> kb = KnowledgeBase(config=KnowledgeBaseConfig(name="docs"))
            >>>
            >>> # Using shortcuts
            >>> kb = KnowledgeBase(name="docs", persist_directory="./kb")
        """
        # Build config from parameters if not provided
        if config is None:
            config = KnowledgeBaseConfig(
                name=name or "default",
                persist_directory=persist_directory,
            )
        self.config = config

        # Initialize vector store with matching config
        vector_config = VectorStoreConfig(
            embedding_model=config.embedding_model,
            index_type=config.index_type,
            persist_directory=config.persist_directory,
        )
        self.vector_store = FAISSVectorStore(vector_config)

        # Initialize document loader config
        self._loader_config = DocumentLoaderConfig(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
        )

        # Track sources
        self.sources: Set[str] = set()
        self.created_at = datetime.now().isoformat()
        self._last_updated: Optional[str] = None

        # Try to load existing knowledge base
        if config.persist_directory:
            self._try_load()

        logger.info(
            f"KnowledgeBase '{config.name}' initialized: "
            f"model={config.embedding_model}, "
            f"documents={self.document_count}"
        )

    def _try_load(self) -> bool:
        """
        Try to load existing knowledge base from persist_directory.

        Returns:
            bool: True if loaded successfully.
        """
        if self.config.persist_directory and Path(self.config.persist_directory).exists():
            if self.vector_store.load(self.config.persist_directory):
                # Restore sources from document metadata
                for doc_id, doc in self.vector_store.documents.items():
                    source = doc.get("metadata", {}).get("source")
                    if source:
                        self.sources.add(source)
                logger.info(f"Loaded existing knowledge base: {self.document_count} documents")
                return True
        return False

    # -------------------------------------------------------------------------
    # Document Management
    # -------------------------------------------------------------------------

    def add_document(
        self,
        file_path: Union[str, Path],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """
        Add a document to the knowledge base.

        This method loads a document file, chunks it, generates embeddings,
        and adds it to the vector store.

        Args:
            file_path: Path to the document file.
            metadata: Optional additional metadata to attach to all chunks.

        Returns:
            int: Number of chunks added.

        Raises:
            FileNotFoundError: If the file doesn't exist.
            ValueError: If the file format is not supported.

        Example:
            >>> kb.add_document("report.pdf", metadata={"department": "sales"})
            >>> kb.add_document("data.csv", metadata={"year": 2024})
        """
        file_path = Path(file_path)

        # Check if already added
        if str(file_path) in self.sources:
            logger.warning(f"Document already in knowledge base: {file_path}")
            return 0

        # Load document
        try:
            documents = load_document(file_path, self._loader_config)
        except ValueError as e:
            logger.error(f"Unsupported format: {file_path}")
            raise

        if not documents:
            logger.warning(f"No content extracted from: {file_path}")
            return 0

        # Add custom metadata
        if metadata:
            for doc in documents:
                doc.metadata.update(metadata)

        # Convert to dict format for vector store
        doc_dicts = [doc.to_dict() for doc in documents]

        # Add to vector store
        added_ids = self.vector_store.add_documents(doc_dicts)

        # Track source
        self.sources.add(str(file_path))
        self._last_updated = datetime.now().isoformat()

        # Auto-save if configured
        if self.config.auto_save and self.config.persist_directory:
            self.save()

        logger.info(f"Added {len(added_ids)} chunks from: {file_path}")
        return len(added_ids)

    def add_directory(
        self,
        directory: Union[str, Path],
        recursive: bool = True,
        file_patterns: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """
        Add all supported documents from a directory.

        This method scans a directory for supported files and adds them
        all to the knowledge base.

        Args:
            directory: Path to the directory.
            recursive: Whether to scan subdirectories.
            file_patterns: Optional glob patterns (e.g., ["*.pdf"]).
            metadata: Optional metadata to attach to all documents.

        Returns:
            int: Total number of chunks added.

        Example:
            >>> # Add all documents
            >>> kb.add_directory("./docs/")
            >>>
            >>> # Add only PDFs
            >>> kb.add_directory("./papers/", file_patterns=["*.pdf"])
        """
        directory = Path(directory)
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")

        logger.info(f"Adding documents from directory: {directory}")

        # Use directory loader
        loader = DirectoryLoader(
            directory=directory,
            recursive=recursive,
            file_patterns=file_patterns,
            config=self._loader_config,
        )

        # Load all documents
        documents = loader.load()

        if not documents:
            logger.warning(f"No documents found in: {directory}")
            return 0

        # Add custom metadata
        if metadata:
            for doc in documents:
                doc.metadata.update(metadata)

        # Convert to dict format
        doc_dicts = [doc.to_dict() for doc in documents]

        # Add to vector store
        added_ids = self.vector_store.add_documents(doc_dicts)

        # Track sources
        for doc in documents:
            source = doc.metadata.get("source")
            if source:
                self.sources.add(source)

        self._last_updated = datetime.now().isoformat()

        # Auto-save if configured
        if self.config.auto_save and self.config.persist_directory:
            self.save()

        logger.info(f"Added {len(added_ids)} chunks from directory: {directory}")
        return len(added_ids)

    def add_text(
        self,
        text: str,
        source_name: str = "manual_input",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """
        Add raw text content to the knowledge base.

        This method is useful for adding content that doesn't come from
        files, such as API responses or user input.

        Args:
            text: The text content to add.
            source_name: Name to use as the source identifier.
            metadata: Optional metadata to attach.

        Returns:
            int: Number of chunks added.

        Example:
            >>> kb.add_text(
            ...     "AI is transforming healthcare...",
            ...     source_name="research_notes",
            ...     metadata={"author": "John", "date": "2024-01-15"}
            ... )
        """
        if not text.strip():
            logger.warning("Empty text provided, nothing to add")
            return 0

        # Create a unique ID
        doc_id = f"{source_name}_{uuid4().hex[:8]}"

        # Build metadata
        doc_metadata = {
            "source": source_name,
            "type": "text",
            "added_at": datetime.now().isoformat(),
        }
        if metadata:
            doc_metadata.update(metadata)

        # Chunk the text if needed
        from filters.document_loaders import TextChunker
        chunker = TextChunker(
            max_size=self.config.chunk_size,
            overlap=self.config.chunk_overlap,
        )

        chunks = chunker.chunk(text)
        documents = []

        for i, chunk in enumerate(chunks):
            chunk_metadata = {
                **doc_metadata,
                "chunk_index": i,
                "total_chunks": len(chunks),
            }
            documents.append({
                "id": f"{doc_id}_chunk{i}",
                "content": chunk,
                "metadata": chunk_metadata,
            })

        # Add to vector store
        added_ids = self.vector_store.add_documents(documents)

        self.sources.add(source_name)
        self._last_updated = datetime.now().isoformat()

        if self.config.auto_save and self.config.persist_directory:
            self.save()

        logger.info(f"Added {len(added_ids)} chunks from text: {source_name}")
        return len(added_ids)

    # -------------------------------------------------------------------------
    # Querying
    # -------------------------------------------------------------------------

    def query(
        self,
        query: str,
        top_k: int = 10,
        filter_metadata: Optional[Dict[str, Any]] = None,
        min_score: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """
        Query the knowledge base for relevant documents.

        This method performs semantic similarity search to find documents
        that are most relevant to the query. Results are ranked by
        similarity score.

        Args:
            query: The query text to search for.
            top_k: Maximum number of results to return.
            filter_metadata: Optional metadata filters. Only documents
                matching ALL specified key-value pairs are returned.
            min_score: Minimum similarity score threshold (0.0-1.0).

        Returns:
            List[Dict]: Search results, each containing:
                - 'id': Document chunk ID
                - 'content': Document text
                - 'metadata': Document metadata
                - 'score': Similarity score (higher = more relevant)

        Example:
            >>> # Simple query
            >>> results = kb.query("machine learning in healthcare")
            >>>
            >>> # With metadata filter
            >>> results = kb.query(
            ...     "revenue trends",
            ...     filter_metadata={"department": "sales"}
            ... )
            >>>
            >>> # With minimum score
            >>> results = kb.query("AI diagnosis", min_score=0.5)
        """
        if self.document_count == 0:
            logger.warning("Knowledge base is empty, returning no results")
            return []

        # Search vector store
        results = self.vector_store.search(
            query=query,
            top_k=top_k,
            filter_metadata=filter_metadata,
        )

        # Apply minimum score filter
        if min_score > 0:
            results = [r for r in results if r["score"] >= min_score]

        logger.debug(f"Query '{query[:50]}...' returned {len(results)} results")
        return results

    def query_with_context(
        self,
        query: str,
        top_k: int = 10,
        context_size: int = 3,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Query and return results with combined context.

        This method is useful for RAG applications where you want a
        single context string combining the top results.

        Args:
            query: The query text.
            top_k: Number of results to retrieve.
            context_size: Number of results to include in context.
            filter_metadata: Optional metadata filters.

        Returns:
            Dict containing:
                - 'context': Combined text from top results
                - 'results': Full search results
                - 'sources': List of source files used

        Example:
            >>> response = kb.query_with_context(
            ...     "What are the benefits of AI in healthcare?",
            ...     context_size=5
            ... )
            >>> print(response['context'])
        """
        results = self.query(
            query=query,
            top_k=top_k,
            filter_metadata=filter_metadata,
        )

        # Build context from top results
        context_results = results[:context_size]
        context_parts = []
        sources = set()

        for i, r in enumerate(context_results, 1):
            context_parts.append(f"[{i}] {r['content']}")
            source = r.get("metadata", {}).get("source")
            if source:
                sources.add(source)

        context = "\n\n".join(context_parts)

        return {
            "context": context,
            "results": results,
            "sources": list(sources),
            "query": query,
        }

    # -------------------------------------------------------------------------
    # Persistence
    # -------------------------------------------------------------------------

    def save(self, path: Optional[str] = None) -> str:
        """
        Save the knowledge base to disk.

        Args:
            path: Directory to save to. Uses config.persist_directory if None.

        Returns:
            str: Path where the knowledge base was saved.

        Raises:
            ValueError: If no path is specified and no persist_directory in config.

        Example:
            >>> kb.save("./my_kb")
            >>> # Later...
            >>> kb.load("./my_kb")
        """
        save_path = path or self.config.persist_directory
        if not save_path:
            raise ValueError("No save path specified")

        # Save vector store (includes documents and index)
        result_path = self.vector_store.save(save_path)

        # Save additional metadata
        import json
        metadata_path = Path(save_path) / "kb_metadata.json"
        kb_metadata = {
            "name": self.config.name,
            "sources": list(self.sources),
            "created_at": self.created_at,
            "last_updated": self._last_updated,
            "config": {
                "embedding_model": self.config.embedding_model,
                "chunk_size": self.config.chunk_size,
                "chunk_overlap": self.config.chunk_overlap,
                "index_type": self.config.index_type,
            },
        }
        with open(metadata_path, "w") as f:
            json.dump(kb_metadata, f, indent=2)

        logger.info(f"Knowledge base saved to: {result_path}")
        return result_path

    def load(self, path: Optional[str] = None) -> bool:
        """
        Load the knowledge base from disk.

        Args:
            path: Directory to load from. Uses config.persist_directory if None.

        Returns:
            bool: True if loaded successfully.

        Example:
            >>> kb = KnowledgeBase(name="my_kb")
            >>> if kb.load("./saved_kb"):
            ...     print(f"Loaded {kb.document_count} documents")
        """
        load_path = path or self.config.persist_directory
        if not load_path:
            logger.error("No load path specified")
            return False

        # Load vector store
        if not self.vector_store.load(load_path):
            return False

        # Load additional metadata
        import json
        metadata_path = Path(load_path) / "kb_metadata.json"
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                kb_metadata = json.load(f)
            self.sources = set(kb_metadata.get("sources", []))
            self.created_at = kb_metadata.get("created_at", self.created_at)
            self._last_updated = kb_metadata.get("last_updated")

        logger.info(f"Knowledge base loaded: {self.document_count} documents")
        return True

    # -------------------------------------------------------------------------
    # Management
    # -------------------------------------------------------------------------

    def delete_source(self, source: str) -> int:
        """
        Delete all documents from a specific source.

        Args:
            source: Source file path to delete.

        Returns:
            int: Number of documents deleted.

        Example:
            >>> kb.delete_source("./docs/old_report.pdf")
        """
        if source not in self.sources:
            logger.warning(f"Source not found in knowledge base: {source}")
            return 0

        # Find all document IDs from this source
        ids_to_delete = []
        for doc_id, doc in self.vector_store.documents.items():
            if doc.get("metadata", {}).get("source") == source:
                ids_to_delete.append(doc_id)

        # Delete from vector store
        deleted = self.vector_store.delete(ids_to_delete)

        # Remove from sources
        self.sources.discard(source)
        self._last_updated = datetime.now().isoformat()

        if self.config.auto_save and self.config.persist_directory:
            self.save()

        logger.info(f"Deleted {deleted} documents from source: {source}")
        return deleted

    def clear(self) -> None:
        """
        Clear all documents from the knowledge base.

        Example:
            >>> kb.clear()
            >>> assert kb.document_count == 0
        """
        self.vector_store.clear()
        self.sources.clear()
        self._last_updated = datetime.now().isoformat()

        if self.config.auto_save and self.config.persist_directory:
            self.save()

        logger.info("Knowledge base cleared")

    # -------------------------------------------------------------------------
    # Properties and Stats
    # -------------------------------------------------------------------------

    @property
    def document_count(self) -> int:
        """Get the number of document chunks in the knowledge base."""
        return self.vector_store.count

    @property
    def source_count(self) -> int:
        """Get the number of unique source files."""
        return len(self.sources)

    def get_stats(self) -> KnowledgeBaseStats:
        """
        Get statistics about the knowledge base.

        Returns:
            KnowledgeBaseStats: Statistics object.

        Example:
            >>> stats = kb.get_stats()
            >>> print(f"Documents: {stats.document_count}")
            >>> print(f"Sources: {stats.source_count}")
        """
        # Calculate total characters
        total_chars = sum(
            len(doc.get("content", ""))
            for doc in self.vector_store.documents.values()
        )

        vs_stats = self.vector_store.get_stats()

        return KnowledgeBaseStats(
            document_count=self.document_count,
            source_count=self.source_count,
            total_characters=total_chars,
            embedding_dimension=vs_stats.get("dimension"),
            index_type=vs_stats.get("index_type", "unknown"),
            last_updated=self._last_updated,
        )

    def list_sources(self) -> List[str]:
        """
        Get list of all source files in the knowledge base.

        Returns:
            List[str]: Sorted list of source file paths.

        Example:
            >>> for source in kb.list_sources():
            ...     print(source)
        """
        return sorted(self.sources)

    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific document by ID.

        Args:
            doc_id: Document chunk ID.

        Returns:
            Dict or None: Document data if found.

        Example:
            >>> doc = kb.get_document("report_p1_chunk0")
            >>> if doc:
            ...     print(doc["content"])
        """
        return self.vector_store.get_document(doc_id)

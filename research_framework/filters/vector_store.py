"""
Vector Store module for the Semantic Layer.

This module provides vector database functionality for storing and retrieving
document embeddings using similarity search. It supports multiple backends
with FAISS as the primary local implementation.

The vector store is a critical component of Stage 1 (Semantic Layer) in the
Complete Knowledge Pipeline, enabling efficient retrieval of relevant documents
based on semantic similarity to queries.

Architecture:
    - VectorStoreConfig: Configuration dataclass for vector store settings
    - BaseVectorStore: Abstract base class defining the vector store interface
    - FAISSVectorStore: FAISS-based implementation for local vector storage
    - EmbeddingModel: Wrapper for generating text embeddings

Supported Backends:
    - FAISS (default): Facebook AI Similarity Search - fast, local, no server needed
    - (Future): Pinecone, Weaviate, Chroma, Qdrant

Usage:
    >>> from filters.vector_store import FAISSVectorStore, VectorStoreConfig
    >>>
    >>> # Create vector store
    >>> config = VectorStoreConfig(embedding_model="all-MiniLM-L6-v2")
    >>> store = FAISSVectorStore(config)
    >>>
    >>> # Add documents
    >>> docs = [{"id": "1", "content": "AI in healthcare...", "metadata": {...}}]
    >>> store.add_documents(docs)
    >>>
    >>> # Search
    >>> results = store.search("medical diagnosis AI", top_k=5)

Example:
    >>> # Full workflow
    >>> store = FAISSVectorStore()
    >>> store.add_documents([
    ...     {"id": "doc1", "content": "Machine learning improves diagnosis"},
    ...     {"id": "doc2", "content": "Deep learning in radiology"},
    ... ])
    >>> results = store.search("AI medical imaging", top_k=2)
    >>> print(results[0]["content"])
"""

import os
import pickle
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
from datetime import datetime

import numpy as np

from utils.logger import get_logger


logger = get_logger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class VectorStoreConfig:
    """
    Configuration for vector store initialization.

    This dataclass holds all configuration parameters for setting up a vector store,
    including embedding model selection, index parameters, and persistence options.

    Attributes:
        embedding_model: Name of the sentence-transformers model for embeddings.
            Common options:
            - "all-MiniLM-L6-v2" (default): Fast, good quality, 384 dimensions
            - "all-mpnet-base-v2": Higher quality, 768 dimensions
            - "paraphrase-multilingual-MiniLM-L12-v2": Multilingual support
        embedding_dimension: Dimension of embedding vectors (auto-detected if None).
        index_type: Type of FAISS index to use:
            - "flat": Exact search, best for <100K documents
            - "ivf": Approximate search, faster for large collections
            - "hnsw": Graph-based, good balance of speed and accuracy
        nlist: Number of clusters for IVF index (default: 100).
        nprobe: Number of clusters to search in IVF (default: 10).
        persist_directory: Directory for saving/loading index (None = in-memory only).
        normalize_embeddings: Whether to L2-normalize embeddings for cosine similarity.
        batch_size: Number of documents to embed in one batch.

    Example:
        >>> config = VectorStoreConfig(
        ...     embedding_model="all-mpnet-base-v2",
        ...     index_type="ivf",
        ...     persist_directory="./vector_db"
        ... )
    """

    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dimension: Optional[int] = None
    index_type: str = "flat"  # "flat", "ivf", "hnsw"
    nlist: int = 100  # Number of clusters for IVF
    nprobe: int = 10  # Number of clusters to search
    persist_directory: Optional[str] = None
    normalize_embeddings: bool = True
    batch_size: int = 32


# =============================================================================
# Embedding Model Wrapper
# =============================================================================

class EmbeddingModel:
    """
    Wrapper for generating text embeddings using sentence-transformers.

    This class provides a unified interface for converting text into dense
    vector representations (embeddings) that capture semantic meaning.
    These embeddings enable similarity-based search in the vector store.

    The class handles lazy loading of the model to avoid loading it until
    actually needed, which improves startup time.

    Attributes:
        model_name: Name of the sentence-transformers model.
        dimension: Dimension of the output embeddings.
        normalize: Whether to L2-normalize embeddings.

    Example:
        >>> model = EmbeddingModel("all-MiniLM-L6-v2")
        >>> embedding = model.embed("Hello, world!")
        >>> print(embedding.shape)  # (384,)
        >>>
        >>> # Batch embedding
        >>> embeddings = model.embed_batch(["Text 1", "Text 2", "Text 3"])
        >>> print(embeddings.shape)  # (3, 384)
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        normalize: bool = True,
    ):
        """
        Initialize the embedding model.

        Args:
            model_name: Name of the sentence-transformers model to use.
            normalize: Whether to L2-normalize output embeddings.
        """
        self.model_name = model_name
        self.normalize = normalize
        self._model = None
        self._dimension: Optional[int] = None

        logger.info(f"EmbeddingModel initialized: model={model_name}, normalize={normalize}")

    @property
    def model(self):
        """
        Lazy-load the sentence-transformers model.

        Returns:
            SentenceTransformer: The loaded model instance.

        Raises:
            ImportError: If sentence-transformers is not installed.
        """
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                logger.info(f"Loading embedding model: {self.model_name}")
                self._model = SentenceTransformer(self.model_name)
                self._dimension = self._model.get_sentence_embedding_dimension()
                logger.info(f"Model loaded: dimension={self._dimension}")
            except ImportError:
                raise ImportError(
                    "sentence-transformers is required for embeddings. "
                    "Install with: pip install sentence-transformers"
                )
        return self._model

    @property
    def dimension(self) -> int:
        """
        Get the embedding dimension.

        Returns:
            int: The dimension of embedding vectors produced by this model.
        """
        if self._dimension is None:
            # Trigger model load to get dimension
            _ = self.model
        return self._dimension

    def embed(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.

        Args:
            text: The text to embed.

        Returns:
            np.ndarray: Embedding vector of shape (dimension,).

        Example:
            >>> model = EmbeddingModel()
            >>> vec = model.embed("AI improves healthcare")
            >>> print(vec.shape)  # (384,)
        """
        embedding = self.model.encode(
            text,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
        )
        return embedding.astype(np.float32)

    def embed_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> np.ndarray:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed.
            batch_size: Number of texts to process in each batch.
            show_progress: Whether to show a progress bar.

        Returns:
            np.ndarray: Embedding matrix of shape (len(texts), dimension).

        Example:
            >>> model = EmbeddingModel()
            >>> vecs = model.embed_batch(["Text 1", "Text 2"])
            >>> print(vecs.shape)  # (2, 384)
        """
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
            show_progress_bar=show_progress,
        )
        return embeddings.astype(np.float32)


# =============================================================================
# Base Vector Store (Abstract)
# =============================================================================

class BaseVectorStore(ABC):
    """
    Abstract base class for vector stores.

    This class defines the interface that all vector store implementations
    must follow. It provides a consistent API for adding documents, searching,
    and managing the vector index regardless of the underlying backend.

    Subclasses must implement:
        - add_documents(): Add documents with embeddings to the store
        - search(): Find similar documents given a query
        - delete(): Remove documents by ID
        - save(): Persist the index to disk
        - load(): Load the index from disk

    Attributes:
        config: Configuration settings for the vector store.
        embedding_model: Model for generating text embeddings.
    """

    def __init__(self, config: Optional[VectorStoreConfig] = None):
        """
        Initialize the base vector store.

        Args:
            config: Configuration for the vector store. Uses defaults if None.
        """
        self.config = config or VectorStoreConfig()
        self.embedding_model = EmbeddingModel(
            model_name=self.config.embedding_model,
            normalize=self.config.normalize_embeddings,
        )

    @abstractmethod
    def add_documents(
        self,
        documents: List[Dict[str, Any]],
        embeddings: Optional[np.ndarray] = None,
    ) -> List[str]:
        """
        Add documents to the vector store.

        Args:
            documents: List of document dicts with 'id', 'content', and optional 'metadata'.
            embeddings: Pre-computed embeddings (optional, will compute if not provided).

        Returns:
            List[str]: IDs of added documents.
        """
        pass

    @abstractmethod
    def search(
        self,
        query: str,
        top_k: int = 10,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents.

        Args:
            query: Query text to search for.
            top_k: Number of results to return.
            filter_metadata: Optional metadata filters.

        Returns:
            List[Dict]: Search results with 'id', 'content', 'metadata', 'score'.
        """
        pass

    @abstractmethod
    def delete(self, ids: List[str]) -> int:
        """
        Delete documents by ID.

        Args:
            ids: List of document IDs to delete.

        Returns:
            int: Number of documents deleted.
        """
        pass

    @abstractmethod
    def save(self, path: Optional[str] = None) -> str:
        """
        Save the vector store to disk.

        Args:
            path: Directory path for saving. Uses config if None.

        Returns:
            str: Path where the store was saved.
        """
        pass

    @abstractmethod
    def load(self, path: Optional[str] = None) -> bool:
        """
        Load the vector store from disk.

        Args:
            path: Directory path to load from. Uses config if None.

        Returns:
            bool: True if loaded successfully.
        """
        pass

    @property
    @abstractmethod
    def count(self) -> int:
        """Get the number of documents in the store."""
        pass


# =============================================================================
# FAISS Vector Store Implementation
# =============================================================================

class FAISSVectorStore(BaseVectorStore):
    """
    FAISS-based vector store implementation.

    This class implements vector storage and similarity search using Facebook AI
    Similarity Search (FAISS). FAISS is highly optimized for similarity search
    and works well for collections up to millions of vectors.

    Features:
        - Multiple index types (flat, IVF, HNSW) for different use cases
        - In-memory and persistent storage options
        - Metadata filtering support
        - Batch operations for efficiency

    Index Types:
        - flat: Exact brute-force search, best for <100K vectors
        - ivf: Inverted file index, approximate but faster for large collections
        - hnsw: Hierarchical Navigable Small World graphs, good accuracy/speed trade-off

    Attributes:
        config: Configuration settings.
        embedding_model: Model for generating embeddings.
        index: The FAISS index instance.
        documents: Dictionary mapping IDs to document data.
        id_to_index: Mapping from document ID to FAISS index position.

    Example:
        >>> # Create and populate store
        >>> store = FAISSVectorStore()
        >>> store.add_documents([
        ...     {"id": "1", "content": "AI in healthcare", "metadata": {"type": "article"}},
        ...     {"id": "2", "content": "Machine learning basics", "metadata": {"type": "tutorial"}},
        ... ])
        >>>
        >>> # Search
        >>> results = store.search("medical AI applications", top_k=5)
        >>> for r in results:
        ...     print(f"{r['id']}: {r['score']:.3f} - {r['content'][:50]}")
        >>>
        >>> # Save and load
        >>> store.save("./my_index")
        >>> new_store = FAISSVectorStore()
        >>> new_store.load("./my_index")
    """

    def __init__(self, config: Optional[VectorStoreConfig] = None):
        """
        Initialize the FAISS vector store.

        Args:
            config: Configuration for the vector store.

        Raises:
            ImportError: If FAISS is not installed.
        """
        super().__init__(config)

        # Verify FAISS is available
        try:
            import faiss
            self._faiss = faiss
        except ImportError:
            raise ImportError(
                "faiss-cpu is required for FAISSVectorStore. "
                "Install with: pip install faiss-cpu"
            )

        # Initialize storage
        self.index = None
        self.documents: Dict[str, Dict[str, Any]] = {}  # id -> {content, metadata}
        self.id_to_index: Dict[str, int] = {}  # id -> faiss index position
        self.index_to_id: Dict[int, str] = {}  # faiss index position -> id
        self._next_index = 0

        # Track index state
        self._is_trained = False
        self._dimension: Optional[int] = None

        logger.info(
            f"FAISSVectorStore initialized: index_type={self.config.index_type}, "
            f"model={self.config.embedding_model}"
        )

    def _create_index(self, dimension: int) -> None:
        """
        Create the FAISS index based on configuration.

        Args:
            dimension: Embedding vector dimension.

        This method creates the appropriate FAISS index type based on the
        configuration. For IVF indices, training is required before adding vectors.
        """
        self._dimension = dimension

        if self.config.index_type == "flat":
            # Exact search - IndexFlatIP for inner product (cosine with normalized vectors)
            self.index = self._faiss.IndexFlatIP(dimension)
            self._is_trained = True
            logger.info(f"Created Flat index: dimension={dimension}")

        elif self.config.index_type == "ivf":
            # Approximate search with inverted file index
            quantizer = self._faiss.IndexFlatIP(dimension)
            self.index = self._faiss.IndexIVFFlat(
                quantizer,
                dimension,
                self.config.nlist,
                self._faiss.METRIC_INNER_PRODUCT,
            )
            self._is_trained = False
            logger.info(f"Created IVF index: dimension={dimension}, nlist={self.config.nlist}")

        elif self.config.index_type == "hnsw":
            # Graph-based approximate search
            self.index = self._faiss.IndexHNSWFlat(dimension, 32)  # 32 = M parameter
            self._is_trained = True
            logger.info(f"Created HNSW index: dimension={dimension}")

        else:
            raise ValueError(f"Unknown index type: {self.config.index_type}")

    def add_documents(
        self,
        documents: List[Dict[str, Any]],
        embeddings: Optional[np.ndarray] = None,
    ) -> List[str]:
        """
        Add documents to the FAISS index.

        This method adds documents to the vector store by:
        1. Generating embeddings if not provided
        2. Creating the FAISS index if this is the first addition
        3. Training the index if required (IVF indices)
        4. Adding vectors to the index
        5. Storing document metadata

        Args:
            documents: List of document dicts. Each must have:
                - 'id' (str): Unique document identifier
                - 'content' (str): Text content to embed
                - 'metadata' (dict, optional): Additional metadata
            embeddings: Pre-computed embeddings array of shape (n_docs, dimension).
                If None, embeddings will be computed from content.

        Returns:
            List[str]: IDs of successfully added documents.

        Raises:
            ValueError: If documents list is empty or documents lack required fields.

        Example:
            >>> store = FAISSVectorStore()
            >>> ids = store.add_documents([
            ...     {"id": "doc1", "content": "First document text"},
            ...     {"id": "doc2", "content": "Second document text", "metadata": {"author": "John"}},
            ... ])
            >>> print(f"Added {len(ids)} documents")
        """
        if not documents:
            logger.warning("No documents provided to add_documents")
            return []

        # Validate documents
        for doc in documents:
            if "id" not in doc or "content" not in doc:
                raise ValueError("Documents must have 'id' and 'content' fields")

        # Generate embeddings if not provided
        if embeddings is None:
            texts = [doc["content"] for doc in documents]
            logger.info(f"Generating embeddings for {len(texts)} documents...")
            embeddings = self.embedding_model.embed_batch(
                texts,
                batch_size=self.config.batch_size,
                show_progress=len(texts) > 100,
            )

        # Create index if this is first addition
        if self.index is None:
            self._create_index(embeddings.shape[1])

        # Train IVF index if needed
        if not self._is_trained and self.config.index_type == "ivf":
            logger.info(f"Training IVF index with {len(embeddings)} vectors...")
            self.index.train(embeddings)
            self._is_trained = True
            self.index.nprobe = self.config.nprobe

        # Add vectors to index
        added_ids = []
        for i, doc in enumerate(documents):
            doc_id = doc["id"]

            # Skip if document already exists
            if doc_id in self.documents:
                logger.warning(f"Document {doc_id} already exists, skipping")
                continue

            # Store document
            self.documents[doc_id] = {
                "content": doc["content"],
                "metadata": doc.get("metadata", {}),
                "added_at": datetime.now().isoformat(),
            }

            # Map ID to index position
            idx = self._next_index
            self.id_to_index[doc_id] = idx
            self.index_to_id[idx] = doc_id
            self._next_index += 1

            added_ids.append(doc_id)

        # Add embeddings to FAISS index
        if added_ids:
            # Get only the embeddings for newly added documents
            new_indices = [documents.index(d) for d in documents if d["id"] in added_ids]
            new_embeddings = embeddings[new_indices]
            self.index.add(new_embeddings)

        logger.info(f"Added {len(added_ids)} documents to vector store (total: {self.count})")
        return added_ids

    def search(
        self,
        query: str,
        top_k: int = 10,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for documents similar to the query.

        This method performs semantic similarity search by:
        1. Embedding the query text
        2. Finding nearest neighbors in the FAISS index
        3. Applying optional metadata filters
        4. Returning results with similarity scores

        Args:
            query: Query text to search for.
            top_k: Maximum number of results to return.
            filter_metadata: Optional dict of metadata key-value pairs to filter by.
                Only documents matching ALL specified metadata will be returned.

        Returns:
            List[Dict]: Search results, each containing:
                - 'id': Document ID
                - 'content': Document text
                - 'metadata': Document metadata
                - 'score': Similarity score (higher = more similar)

        Example:
            >>> results = store.search("AI in healthcare", top_k=5)
            >>> for r in results:
            ...     print(f"Score: {r['score']:.3f} - {r['content'][:100]}")
            >>>
            >>> # With metadata filter
            >>> results = store.search("AI", filter_metadata={"type": "article"})
        """
        if self.index is None or self.count == 0:
            logger.warning("Vector store is empty, returning no results")
            return []

        # Embed query
        query_embedding = self.embedding_model.embed(query)
        query_embedding = query_embedding.reshape(1, -1)

        # Search index (get more results if filtering to ensure enough after filter)
        search_k = top_k * 3 if filter_metadata else top_k
        search_k = min(search_k, self.count)

        scores, indices = self.index.search(query_embedding, search_k)

        # Build results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # FAISS returns -1 for empty slots
                continue

            doc_id = self.index_to_id.get(idx)
            if doc_id is None:
                continue

            doc = self.documents.get(doc_id)
            if doc is None:
                continue

            # Apply metadata filter
            if filter_metadata:
                match = all(
                    doc["metadata"].get(k) == v
                    for k, v in filter_metadata.items()
                )
                if not match:
                    continue

            results.append({
                "id": doc_id,
                "content": doc["content"],
                "metadata": doc["metadata"],
                "score": float(score),
            })

            if len(results) >= top_k:
                break

        logger.debug(f"Search returned {len(results)} results for query: {query[:50]}...")
        return results

    def search_by_vector(
        self,
        embedding: np.ndarray,
        top_k: int = 10,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search using a pre-computed embedding vector.

        This is useful when you already have an embedding and want to avoid
        recomputing it, or when using embeddings from external sources.

        Args:
            embedding: Query embedding vector of shape (dimension,) or (1, dimension).
            top_k: Maximum number of results to return.
            filter_metadata: Optional metadata filters.

        Returns:
            List[Dict]: Search results with 'id', 'content', 'metadata', 'score'.

        Example:
            >>> embedding = model.embed("search query")
            >>> results = store.search_by_vector(embedding, top_k=5)
        """
        if self.index is None or self.count == 0:
            return []

        # Ensure correct shape
        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)

        embedding = embedding.astype(np.float32)

        # Search
        search_k = top_k * 3 if filter_metadata else top_k
        search_k = min(search_k, self.count)

        scores, indices = self.index.search(embedding, search_k)

        # Build results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue

            doc_id = self.index_to_id.get(idx)
            if doc_id is None:
                continue

            doc = self.documents.get(doc_id)
            if doc is None:
                continue

            if filter_metadata:
                match = all(
                    doc["metadata"].get(k) == v
                    for k, v in filter_metadata.items()
                )
                if not match:
                    continue

            results.append({
                "id": doc_id,
                "content": doc["content"],
                "metadata": doc["metadata"],
                "score": float(score),
            })

            if len(results) >= top_k:
                break

        return results

    def delete(self, ids: List[str]) -> int:
        """
        Delete documents by ID.

        Note: FAISS doesn't support efficient deletion. This method marks
        documents as deleted but doesn't remove them from the index.
        For large-scale deletions, consider rebuilding the index.

        Args:
            ids: List of document IDs to delete.

        Returns:
            int: Number of documents deleted.

        Example:
            >>> deleted = store.delete(["doc1", "doc2"])
            >>> print(f"Deleted {deleted} documents")
        """
        deleted = 0
        for doc_id in ids:
            if doc_id in self.documents:
                del self.documents[doc_id]
                # Note: We can't efficiently remove from FAISS index
                # The ID mapping is kept but document is gone
                deleted += 1

        logger.info(f"Deleted {deleted} documents (note: FAISS index not compacted)")
        return deleted

    def save(self, path: Optional[str] = None) -> str:
        """
        Save the vector store to disk.

        This saves both the FAISS index and the document metadata to the
        specified directory. The saved store can be loaded later with load().

        Args:
            path: Directory path for saving. Uses config.persist_directory if None.

        Returns:
            str: Path where the store was saved.

        Raises:
            ValueError: If no path specified and no persist_directory in config.

        Example:
            >>> store.save("./my_vector_store")
            >>> # Later...
            >>> store.load("./my_vector_store")
        """
        save_path = path or self.config.persist_directory
        if save_path is None:
            raise ValueError("No save path specified and no persist_directory in config")

        save_dir = Path(save_path)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        if self.index is not None:
            index_path = save_dir / "index.faiss"
            self._faiss.write_index(self.index, str(index_path))
            logger.info(f"Saved FAISS index to {index_path}")

        # Save metadata
        metadata = {
            "documents": self.documents,
            "id_to_index": self.id_to_index,
            "index_to_id": self.index_to_id,
            "next_index": self._next_index,
            "is_trained": self._is_trained,
            "dimension": self._dimension,
            "config": {
                "embedding_model": self.config.embedding_model,
                "index_type": self.config.index_type,
                "nlist": self.config.nlist,
                "nprobe": self.config.nprobe,
            },
            "saved_at": datetime.now().isoformat(),
        }

        metadata_path = save_dir / "metadata.pkl"
        with open(metadata_path, "wb") as f:
            pickle.dump(metadata, f)
        logger.info(f"Saved metadata to {metadata_path}")

        return str(save_dir)

    def load(self, path: Optional[str] = None) -> bool:
        """
        Load the vector store from disk.

        This loads both the FAISS index and document metadata from a
        previously saved store.

        Args:
            path: Directory path to load from. Uses config.persist_directory if None.

        Returns:
            bool: True if loaded successfully, False otherwise.

        Example:
            >>> store = FAISSVectorStore()
            >>> if store.load("./my_vector_store"):
            ...     print(f"Loaded {store.count} documents")
        """
        load_path = path or self.config.persist_directory
        if load_path is None:
            logger.error("No load path specified")
            return False

        load_dir = Path(load_path)
        if not load_dir.exists():
            logger.error(f"Load directory does not exist: {load_dir}")
            return False

        try:
            # Load metadata
            metadata_path = load_dir / "metadata.pkl"
            with open(metadata_path, "rb") as f:
                metadata = pickle.load(f)

            self.documents = metadata["documents"]
            self.id_to_index = metadata["id_to_index"]
            self.index_to_id = {int(k): v for k, v in metadata["index_to_id"].items()}
            self._next_index = metadata["next_index"]
            self._is_trained = metadata["is_trained"]
            self._dimension = metadata["dimension"]

            # Load FAISS index
            index_path = load_dir / "index.faiss"
            if index_path.exists():
                self.index = self._faiss.read_index(str(index_path))

                # Restore IVF nprobe if applicable
                if self.config.index_type == "ivf" and hasattr(self.index, "nprobe"):
                    self.index.nprobe = self.config.nprobe

            logger.info(f"Loaded vector store from {load_dir}: {self.count} documents")
            return True

        except Exception as e:
            logger.error(f"Failed to load vector store: {e}")
            return False

    @property
    def count(self) -> int:
        """
        Get the number of documents in the store.

        Returns:
            int: Number of documents (excluding deleted ones).
        """
        return len(self.documents)

    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a document by ID.

        Args:
            doc_id: Document ID to retrieve.

        Returns:
            Dict or None: Document data if found, None otherwise.

        Example:
            >>> doc = store.get_document("doc1")
            >>> if doc:
            ...     print(doc["content"])
        """
        doc = self.documents.get(doc_id)
        if doc:
            return {
                "id": doc_id,
                "content": doc["content"],
                "metadata": doc["metadata"],
            }
        return None

    def clear(self) -> None:
        """
        Clear all documents from the store.

        This removes all documents and resets the FAISS index.

        Example:
            >>> store.clear()
            >>> assert store.count == 0
        """
        self.index = None
        self.documents.clear()
        self.id_to_index.clear()
        self.index_to_id.clear()
        self._next_index = 0
        self._is_trained = False
        logger.info("Vector store cleared")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector store.

        Returns:
            Dict: Statistics including document count, index type, and dimension.

        Example:
            >>> stats = store.get_stats()
            >>> print(f"Documents: {stats['document_count']}")
        """
        return {
            "document_count": self.count,
            "index_type": self.config.index_type,
            "embedding_model": self.config.embedding_model,
            "dimension": self._dimension,
            "is_trained": self._is_trained,
            "persist_directory": self.config.persist_directory,
        }

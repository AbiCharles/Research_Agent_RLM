"""
Filters package for the Multi-Agent Research Framework.

Provides foundational components for the RLM paradigm:
  - vector_store.py: FAISS-based vector storage and similarity search
  - document_loaders.py: Multi-format document loading and chunking
  - knowledge_base.py: Unified knowledge management interface
  - skills.py: Skills framework for domain-specific analysis

Selection and compression are handled by core.memory_manager (MemoryManager).
Context access is handled by core.knowledge_environment (KnowledgeEnvironment).
"""

# -----------------------------------------------------------------------------
# Vector Store
# -----------------------------------------------------------------------------
from filters.vector_store import (
    VectorStoreConfig,
    BaseVectorStore,
    FAISSVectorStore,
    EmbeddingModel,
)

# -----------------------------------------------------------------------------
# Document Loaders
# -----------------------------------------------------------------------------
from filters.document_loaders import (
    DocumentLoaderConfig,
    LoadedDocument,
    TextChunker,
    BaseDocumentLoader,
    PDFLoader,
    WordLoader,
    ExcelLoader,
    CSVLoader,
    TextLoader,
    DirectoryLoader,
    get_loader_for_file,
    load_document,
)

# -----------------------------------------------------------------------------
# Knowledge Base
# -----------------------------------------------------------------------------
from filters.knowledge_base import (
    KnowledgeBaseConfig,
    KnowledgeBaseStats,
    KnowledgeBase,
)

# -----------------------------------------------------------------------------
# Skills Framework
# -----------------------------------------------------------------------------
from filters.skills import (
    SkillConfig,
    SkillResult,
    Skill,
    SkillRegistry,
    ResearchAnalysisSkill,
    SummarySkill,
    EntityExtractionSkill,
    SentimentAnalysisSkill,
    create_skill,
    get_default_registry,
    execute_skill,
)

# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------
__all__ = [
    # Vector Store
    "VectorStoreConfig",
    "BaseVectorStore",
    "FAISSVectorStore",
    "EmbeddingModel",
    # Document Loaders
    "DocumentLoaderConfig",
    "LoadedDocument",
    "TextChunker",
    "BaseDocumentLoader",
    "PDFLoader",
    "WordLoader",
    "ExcelLoader",
    "CSVLoader",
    "TextLoader",
    "DirectoryLoader",
    "get_loader_for_file",
    "load_document",
    # Knowledge Base
    "KnowledgeBaseConfig",
    "KnowledgeBaseStats",
    "KnowledgeBase",
    # Skills Framework
    "SkillConfig",
    "SkillResult",
    "Skill",
    "SkillRegistry",
    "ResearchAnalysisSkill",
    "SummarySkill",
    "EntityExtractionSkill",
    "SentimentAnalysisSkill",
    "create_skill",
    "get_default_registry",
    "execute_skill",
]

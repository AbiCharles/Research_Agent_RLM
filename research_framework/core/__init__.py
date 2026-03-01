from .openai_client import OpenAIClient, get_client, TokenUsage, ModelTier
from .base_agent import BaseAgent, AgentConfig, AgentResult, AgentStatus, AgentRole
from .memory_manager import (
    # RLM Core Components
    MemoryManager,
    MemoryConfig,
    REPLEnvironment,
    LLMQueryFunction,
    LLMQueryResult,
    # Async Query Pool (NEW - addresses RLM Paper performance concerns)
    AsyncLLMQueryPool,
    # Pipeline
    PipelineStage,
    PipelineResult,
    # Filters
    SelectionFilter,
    KeywordRelevanceFilter,
    SemanticRelevanceFilter,
    # Compression
    CompressionStrategy,
    ExtractiveCompression,
    AbstractiveCompression,
)
from .knowledge_environment import (
    # RLM Environment Bridge (connects KB with REPL for environment variable paradigm)
    KnowledgeEnvironment,
    KnowledgeEnvironmentConfig,
    ContextMetadata,
    TopicExtractor,
    create_knowledge_environment,
)

__all__ = [
    # OpenAI client
    "OpenAIClient",
    "get_client",
    "TokenUsage",
    "ModelTier",
    # Base agent
    "BaseAgent",
    "AgentConfig",
    "AgentResult",
    "AgentStatus",
    "AgentRole",
    # RLM Memory Manager
    "MemoryManager",
    "MemoryConfig",
    "REPLEnvironment",
    "LLMQueryFunction",
    "LLMQueryResult",
    # Async Query Pool (addresses RLM Paper performance concerns)
    "AsyncLLMQueryPool",
    # Pipeline
    "PipelineStage",
    "PipelineResult",
    # Filters
    "SelectionFilter",
    "KeywordRelevanceFilter",
    "SemanticRelevanceFilter",
    # Compression
    "CompressionStrategy",
    "ExtractiveCompression",
    "AbstractiveCompression",
    # Knowledge Environment Bridge (RLM environment variable paradigm)
    "KnowledgeEnvironment",
    "KnowledgeEnvironmentConfig",
    "ContextMetadata",
    "TopicExtractor",
    "create_knowledge_environment",
]

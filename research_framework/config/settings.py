"""
Configuration management using environment variables and YAML configs.

This module provides centralized configuration for the Multi-Agent Research Framework.
Configuration values are loaded from environment variables with sensible defaults.
"""

import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv


# Load environment variables from .env file
load_dotenv()


class Settings:
    """
    Application settings loaded from environment variables.

    All configuration values can be overridden via environment variables.
    The RF_ prefix is used for framework-specific variables to avoid conflicts.
    """

    # OpenAI Configuration
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_ORG_ID: Optional[str] = os.getenv("OPENAI_ORG_ID")

    # Model Configuration
    DEFAULT_MODEL: str = os.getenv("DEFAULT_MODEL", "gpt-4o-mini")
    PREMIUM_MODEL: str = os.getenv("PREMIUM_MODEL", "gpt-4o")
    MAX_TOKENS: int = int(os.getenv("MAX_TOKENS", "4000"))
    TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.7"))

    # Framework Configuration
    MAX_AGENTS: int = int(os.getenv("MAX_AGENTS", "5"))
    MAX_RESEARCH_TIME_MINUTES: int = int(os.getenv("MAX_RESEARCH_TIME_MINUTES", "20"))
    DEFAULT_MODEL_CONFIG: str = os.getenv("DEFAULT_MODEL_CONFIG", "default")

    # Search Tool Configuration
    # Supported backends: tavily, brave, serper, bing, duckduckgo
    SEARCH_BACKEND: str = os.getenv("SEARCH_BACKEND", "auto")  # 'auto' selects based on available keys
    TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "")
    BRAVE_API_KEY: str = os.getenv("BRAVE_API_KEY", "")
    SERPER_API_KEY: str = os.getenv("SERPER_API_KEY", "")
    BING_API_KEY: str = os.getenv("BING_API_KEY", "")

    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE: str = os.getenv("LOG_FILE", "research_framework.log")

    # API Configuration
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))
    API_WORKERS: int = int(os.getenv("API_WORKERS", "4"))

    # Performance
    ENABLE_CACHING: bool = os.getenv("ENABLE_CACHING", "true").lower() == "true"
    CACHE_TTL_SECONDS: int = int(os.getenv("CACHE_TTL_SECONDS", "3600"))
    MAX_PARALLEL_AGENTS: int = int(os.getenv("MAX_PARALLEL_AGENTS", "5"))

    # Paths
    BASE_DIR: Path = Path(__file__).parent.parent
    CONFIG_DIR: Path = BASE_DIR / "config"

    # Model tier configurations
    MODEL_TIERS = {
        "default": {
            "lead_model": "gpt-4o-mini",
            "agent_model": "gpt-4o-mini",
            "citation_model": "gpt-4o-mini",
            "max_tokens": 4000,
            "temperature": 0.7,
        },
        "budget": {
            "lead_model": "gpt-4o-mini",
            "agent_model": "gpt-4o-mini",
            "citation_model": "gpt-4o-mini",
            "max_tokens": 2000,
            "temperature": 0.5,
        },
        "premium": {
            "lead_model": "gpt-4o",
            "agent_model": "gpt-4o-mini",
            "citation_model": "gpt-4o-mini",
            "max_tokens": 8000,
            "temperature": 0.7,
        },
        "fast": {
            "lead_model": "gpt-4o-mini",
            "agent_model": "gpt-4o-mini",
            "citation_model": "gpt-4o-mini",
            "max_tokens": 2000,
            "temperature": 0.3,
        },
    }

    @classmethod
    def get_model_config(cls, tier: str = "default") -> dict:
        """
        Get model configuration for a specific tier.

        Args:
            tier: One of 'default', 'budget', 'premium', 'fast'

        Returns:
            Dictionary with model configuration
        """
        return cls.MODEL_TIERS.get(tier, cls.MODEL_TIERS["default"])

    @classmethod
    def validate(cls) -> list[str]:
        """
        Validate required configuration values.

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []

        if not cls.OPENAI_API_KEY:
            errors.append("OPENAI_API_KEY is required")

        if cls.MAX_AGENTS < 1 or cls.MAX_AGENTS > 10:
            errors.append("MAX_AGENTS must be between 1 and 10")

        if cls.TEMPERATURE < 0 or cls.TEMPERATURE > 2:
            errors.append("TEMPERATURE must be between 0 and 2")

        return errors


# Singleton instance for easy import
settings = Settings()

"""
API module for the Multi-Agent Research Framework.

This module provides a REST API interface using FastAPI for accessing the
multi-agent research capabilities. It exposes endpoints for research orchestration,
web search, and citation processing.

Module Structure:
    - main.py: FastAPI application factory and configuration
    - routes.py: API endpoint definitions organized by router
    - models.py: Pydantic request/response models

Exports:
    Application:
        - app: The configured FastAPI application instance
        - create_app: Factory function for creating new app instances

    Routers:
        - router: Base router with health and info endpoints
        - research_router: Multi-agent research endpoints
        - search_router: Web search endpoints
        - citation_router: Citation processing endpoints

    Models:
        - ResearchRequest: Request model for research tasks
        - ResearchResponse: Response model for research results
        - ResearchStatus: Enum for task status tracking
        - SearchResponse: Response model for search results
        - CitationResponse: Response model for citation processing
        - HealthResponse: Response model for health checks

Usage:
    # Run the API server
    >>> from api import app
    >>> import uvicorn
    >>> uvicorn.run(app, host="0.0.0.0", port=8000)

    # Or use the models for type hints
    >>> from api import ResearchRequest, ResearchResponse
    >>> def process_research(request: ResearchRequest) -> ResearchResponse:
    ...     pass

    # Access individual routers for custom app composition
    >>> from api import research_router
    >>> custom_app.include_router(research_router)
"""

from api.main import app, create_app
from api.routes import router, research_router, search_router, citation_router
from api.models import (
    ResearchRequest,
    ResearchResponse,
    ResearchStatus,
    SearchResponse,
    CitationResponse,
    HealthResponse,
)

__all__ = [
    # Application
    "app",
    "create_app",
    # Routers
    "router",
    "research_router",
    "search_router",
    "citation_router",
    # Models
    "ResearchRequest",
    "ResearchResponse",
    "ResearchStatus",
    "SearchResponse",
    "CitationResponse",
    "HealthResponse",
]

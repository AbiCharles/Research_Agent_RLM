"""
FastAPI application for the Multi-Agent Research Framework.

This module provides the main FastAPI application instance and configuration
for the Multi-Agent Research Framework REST API. It serves as the entry point
for the API service and handles application lifecycle, middleware configuration,
exception handling, and route registration.

Module Components:
    - Application Factory: create_app() function for creating configured FastAPI instances
    - Lifespan Management: Startup and shutdown event handling
    - Exception Handlers: Global error handling for consistent error responses
    - Custom OpenAPI Schema: Enhanced API documentation configuration
    - CORS Middleware: Cross-origin request handling

Usage:
    # Run with uvicorn (recommended for development)
    uvicorn api.main:app --reload --port 8000

    # Run with multiple workers (production)
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4

    # Run directly using the built-in server
    python -m api.main

    # Import in other modules
    from api.main import app, create_app

Environment Variables:
    OPENAI_API_KEY: Required - OpenAI API key for LLM operations
    TAVILY_API_KEY: Optional - Tavily API key for enhanced web search
    API_HOST: Optional - Host to bind to (default: 0.0.0.0)
    API_PORT: Optional - Port to listen on (default: 8000)

Example:
    >>> from api.main import app
    >>> # Use with TestClient for testing
    >>> from fastapi.testclient import TestClient
    >>> client = TestClient(app)
    >>> response = client.get("/health")
    >>> assert response.status_code == 200
"""

import sys
from pathlib import Path
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.openapi.utils import get_openapi

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from api.routes import router, research_router, search_router, citation_router
from api.models import ErrorResponse
from config.settings import settings
from utils.logger import get_logger


logger = get_logger(__name__)


# ============================================================================
# Lifespan Management
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage application lifespan events using async context manager.

    This function handles startup and shutdown events for the FastAPI application.
    It uses the modern lifespan context manager pattern (introduced in FastAPI 0.93+)
    instead of the deprecated on_event decorators.

    Startup Actions:
        - Log API initialization status
        - Verify API key configuration
        - Log configuration settings

    Shutdown Actions:
        - Log shutdown message
        - (Future: Close database connections, cleanup resources)

    Args:
        app: The FastAPI application instance.

    Yields:
        None - Control returns to the application during its lifetime.

    Example:
        This function is passed to FastAPI's lifespan parameter:
        >>> app = FastAPI(lifespan=lifespan)

    Note:
        Any resources initialized during startup should be cleaned up
        after the yield statement in the shutdown phase.
    """
    # -------------------------------------------------------------------------
    # Startup Phase
    # -------------------------------------------------------------------------
    logger.info("Starting Multi-Agent Research Framework API...")
    logger.info(f"OpenAI API configured: {bool(settings.OPENAI_API_KEY)}")
    logger.info(f"Tavily API configured: {bool(settings.TAVILY_API_KEY)}")
    logger.info(f"Max agents: {settings.MAX_AGENTS}")

    # Yield control to the application - it runs while we're suspended here
    yield

    # -------------------------------------------------------------------------
    # Shutdown Phase
    # -------------------------------------------------------------------------
    logger.info("Shutting down API...")
    # Future: Add cleanup for database connections, background tasks, etc.


# ============================================================================
# Application Factory
# ============================================================================

def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application instance.

    This factory function creates a fully configured FastAPI application with:
    - API metadata (title, description, version)
    - Documentation endpoints (Swagger UI, ReDoc)
    - CORS middleware for cross-origin requests
    - All API routers registered
    - Global exception handlers

    The factory pattern allows for:
    - Creating multiple app instances (useful for testing)
    - Consistent configuration across environments
    - Easy dependency injection and mocking

    Returns:
        FastAPI: A fully configured FastAPI application instance ready to serve requests.

    Example:
        >>> app = create_app()
        >>> # Run with uvicorn
        >>> import uvicorn
        >>> uvicorn.run(app, host="0.0.0.0", port=8000)

    Note:
        This function is called at module load time to create the global `app` instance.
        For testing, you can call this function to create isolated app instances.
    """
    # -------------------------------------------------------------------------
    # Create FastAPI Instance with Metadata
    # -------------------------------------------------------------------------
    app = FastAPI(
        title="Multi-Agent Research Framework",
        description="""
## Overview

The Multi-Agent Research Framework API provides AI-powered research capabilities
using coordinated multi-agent systems built on OpenAI GPT models.

## Features

- **Multi-Agent Orchestration**: Coordinate multiple specialized research agents
- **Parallel Execution**: Run research agents in parallel for faster results
- **RLM Context Management**: Intelligent context compression and management
- **Citation Generation**: Automatic inline citations and reference lists
- **Multiple Search Backends**: Tavily, Brave, Serper, Bing, DuckDuckGo

## Quick Start

1. **Simple Search**: `GET /search?q=your+query`
2. **Quick Research**: `POST /research/quick?hypothesis=your+hypothesis`
3. **Full Research**: `POST /research` with custom agents

## Authentication

Set your API keys in environment variables:
- `OPENAI_API_KEY`: Required for all operations
- `TAVILY_API_KEY`: Optional, for enhanced search

        """,
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # -------------------------------------------------------------------------
    # Configure CORS Middleware
    # -------------------------------------------------------------------------
    # CORS (Cross-Origin Resource Sharing) middleware allows the API to be
    # accessed from web browsers running on different domains.
    #
    # SECURITY NOTE: The current configuration allows all origins ("*").
    # For production deployments, restrict this to specific trusted domains:
    #   allow_origins=["https://yourfrontend.com", "https://admin.yourapp.com"]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # TODO: Restrict for production
        allow_credentials=True,  # Allow cookies and authorization headers
        allow_methods=["*"],  # Allow all HTTP methods (GET, POST, PUT, DELETE, etc.)
        allow_headers=["*"],  # Allow all headers
    )

    # -------------------------------------------------------------------------
    # Register API Routers
    # -------------------------------------------------------------------------
    # Each router handles a specific group of related endpoints:
    # - router: Base routes (health, info)
    # - research_router: Research orchestration endpoints (/research/*)
    # - search_router: Web search endpoints (/search/*)
    # - citation_router: Citation processing endpoints (/citations/*)
    app.include_router(router)
    app.include_router(research_router)
    app.include_router(search_router)
    app.include_router(citation_router)

    # -------------------------------------------------------------------------
    # Register Exception Handlers
    # -------------------------------------------------------------------------
    # Global exception handlers ensure consistent error responses across all endpoints
    register_exception_handlers(app)

    return app


# ============================================================================
# Exception Handlers
# ============================================================================

def register_exception_handlers(app: FastAPI):
    """
    Register global exception handlers for consistent error responses.

    This function attaches exception handlers to the FastAPI application that
    catch and format errors in a consistent way across all endpoints. This ensures
    that clients always receive well-structured error responses regardless of
    where the error originated.

    Exception Handling Strategy:
        - ValueError: Treated as client validation errors (400 Bad Request)
        - Exception: Catch-all for unexpected server errors (500 Internal Server Error)

    All error responses follow the ErrorResponse schema:
        {
            "error": "Human-readable error type",
            "detail": "Specific error message",
            "code": "MACHINE_READABLE_CODE",
            "timestamp": "2024-01-15T10:30:00Z"
        }

    Args:
        app: The FastAPI application instance to attach handlers to.

    Note:
        Additional exception handlers can be added here for specific exception types
        like HTTPException, RequestValidationError, etc.
    """

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        """
        Handle all unhandled exceptions as 500 Internal Server Error.

        This is the catch-all handler for any exceptions not caught by more
        specific handlers. It logs the full exception with traceback for
        debugging while returning a safe error message to the client.

        Args:
            request: The incoming request that caused the exception.
            exc: The exception that was raised.

        Returns:
            JSONResponse: A 500 error response with error details.
        """
        # Log the full exception with traceback for debugging
        logger.error(f"Unhandled exception: {exc}", exc_info=True)

        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error="Internal Server Error",
                detail=str(exc),
                code="INTERNAL_ERROR",
                timestamp=datetime.now(),
            ).model_dump(mode="json"),
        )

    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError):
        """
        Handle ValueError as 400 Bad Request (validation error).

        ValueErrors typically indicate invalid input data or failed validation.
        These are returned as 400 Bad Request to indicate client error.

        Args:
            request: The incoming request that caused the exception.
            exc: The ValueError that was raised.

        Returns:
            JSONResponse: A 400 error response with validation error details.
        """
        return JSONResponse(
            status_code=400,
            content=ErrorResponse(
                error="Validation Error",
                detail=str(exc),
                code="VALIDATION_ERROR",
                timestamp=datetime.now(),
            ).model_dump(mode="json"),
        )


# ============================================================================
# Custom OpenAPI Schema
# ============================================================================

def custom_openapi():
    """
    Generate a customized OpenAPI schema for API documentation.

    This function creates and caches a custom OpenAPI 3.0 schema that enhances
    the auto-generated documentation with additional metadata like a logo URL.
    The schema is used by Swagger UI (/docs) and ReDoc (/redoc).

    Customizations Applied:
        - Custom logo URL for branding in documentation UIs
        - (Future: Security schemes for API key authentication)
        - (Future: Custom tags and groupings)

    Returns:
        dict: The OpenAPI schema dictionary.

    Note:
        The schema is cached after first generation. To regenerate,
        set app.openapi_schema = None before calling.
    """
    # Return cached schema if available (performance optimization)
    if app.openapi_schema:
        return app.openapi_schema

    # Generate base OpenAPI schema from routes
    openapi_schema = get_openapi(
        title="Multi-Agent Research Framework",
        version="1.0.0",
        description="AI-powered multi-agent research system",
        routes=app.routes,
    )

    # -------------------------------------------------------------------------
    # Apply Customizations
    # -------------------------------------------------------------------------

    # Add logo for documentation UI branding
    openapi_schema["info"]["x-logo"] = {
        "url": "https://example.com/logo.png"  # TODO: Replace with actual logo URL
    }

    # Future: Add security schemes
    # openapi_schema["components"]["securitySchemes"] = {
    #     "api_key": {
    #         "type": "apiKey",
    #         "in": "header",
    #         "name": "X-API-Key"
    #     }
    # }

    # Cache the schema for subsequent requests
    app.openapi_schema = openapi_schema
    return app.openapi_schema


# ============================================================================
# Create Application Instance
# ============================================================================

# Create the global application instance
# This is the instance that uvicorn and other ASGI servers will use
app = create_app()

# Override the default OpenAPI schema generator with our custom version
app.openapi = custom_openapi


# ============================================================================
# Root Endpoint
# ============================================================================

@app.get("/", tags=["Root"])
async def root():
    """
    Root endpoint providing API information and navigation.

    This endpoint serves as the entry point for the API, providing
    basic information about the service and links to key endpoints.
    It's useful for API discovery and health verification.

    Returns:
        dict: API metadata including:
            - name: The API service name
            - version: Current API version
            - status: Service status ("running")
            - docs: Link to Swagger UI documentation
            - health: Link to health check endpoint
            - endpoints: Dictionary of main endpoint groups

    Example Response:
        {
            "name": "Multi-Agent Research Framework",
            "version": "1.0.0",
            "status": "running",
            "docs": "/docs",
            "health": "/health",
            "endpoints": {
                "research": "/research",
                "search": "/search",
                "citations": "/citations"
            }
        }
    """
    return {
        "name": "Multi-Agent Research Framework",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "research": "/research",
            "search": "/search",
            "citations": "/citations",
        },
    }


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    # Run the application using uvicorn ASGI server
    # This block is executed when running: python -m api.main
    #
    # Configuration:
    #   - host/port: From settings (default 0.0.0.0:8000)
    #   - reload: Auto-reload on code changes (development feature)
    #   - workers: Number of worker processes (1 for dev, more for production)
    #
    # For production, use: uvicorn api.main:app --workers 4
    uvicorn.run(
        "api.main:app",  # Import string for the app (required for reload)
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=True,  # Enable hot-reload for development
        workers=1,  # Single worker for development (reload requires workers=1)
    )

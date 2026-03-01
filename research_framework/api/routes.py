"""
API routes for the FastAPI service.

This module defines all API endpoints for the Research Framework, organized
into logical groups:

Route Groups:
    - Research Routes (/research): Multi-agent research orchestration
    - Search Routes (/search): Web search functionality
    - Citation Routes (/citations): Citation processing
    - Health Routes (/health, /info): API health and information

Endpoint Summary:
    POST /research          - Start full multi-agent research
    POST /research/async    - Start research in background
    GET  /research/{id}     - Get research results
    GET  /research/{id}/status - Check research status
    POST /research/quick    - Quick single-agent research

    GET/POST /search        - Perform web search
    GET  /search/backends   - List available search backends

    POST /citations         - Add citations to content

    GET  /health           - API health check
    GET  /info             - Framework information

Example:
    >>> # Start the API server
    >>> uvicorn api.main:app --reload --port 8000
    >>>
    >>> # Make a search request
    >>> import requests
    >>> response = requests.get("http://localhost:8000/search?q=AI+healthcare")
    >>> print(response.json())
"""

import asyncio
from datetime import datetime
from typing import Dict, Optional, Any
from uuid import uuid4

from fastapi import APIRouter, HTTPException, BackgroundTasks, Query
from fastapi.responses import JSONResponse

# Import Pydantic models for request/response validation
from api.models import (
    ResearchRequest,
    ResearchResponse,
    ResearchStatusResponse,
    ResearchStatus,
    QuickSearchRequest,
    SearchResponse,
    SearchResult,
    CitationRequest,
    CitationResponse,
    HealthResponse,
    FrameworkInfoResponse,
    BackendStatus,
    ErrorResponse,
    AgentResultResponse,
    ResearchPlanResponse,
)

# Import framework components
from agents import (
    create_lead_researcher,
    create_citation_agent,
    create_research_agent,
)
from tools import WebSearchTool
from config.settings import settings
from utils.logger import get_logger


# Initialize logger for this module
logger = get_logger(__name__)


# ============================================================================
# Router Configuration
# ============================================================================
# Create separate routers for each endpoint group.
# These are combined in main.py using app.include_router().

# Base router for health/info endpoints (no prefix)
router = APIRouter()

# Research endpoints - handles multi-agent orchestration
research_router = APIRouter(prefix="/research", tags=["Research"])

# Search endpoints - handles web search queries
search_router = APIRouter(prefix="/search", tags=["Search"])

# Citation endpoints - handles citation processing
citation_router = APIRouter(prefix="/citations", tags=["Citations"])


# ============================================================================
# In-Memory Task Storage
# ============================================================================
# Stores async research task state. In production, use Redis or a database
# for persistence across restarts and horizontal scaling.
#
# Structure:
#   {
#       "task_id": {
#           "status": ResearchStatus,
#           "progress": float (0-1),
#           "current_stage": str,
#           "agents_completed": int,
#           "agents_total": int,
#           "started_at": datetime,
#           "result": OrchestrationResult or None,
#           "error": str or None
#       }
#   }

_research_tasks: Dict[str, Dict] = {}


# ============================================================================
# Research Endpoints
# ============================================================================
# These endpoints handle multi-agent research orchestration.

@research_router.post(
    "",
    response_model=ResearchResponse,
    summary="Start a research investigation",
    description="""
    Initiates a comprehensive multi-agent research investigation.

    This endpoint orchestrates multiple specialized AI agents to investigate
    a hypothesis from different angles. The Lead Researcher coordinates the
    agents, synthesizes their findings, and produces a final report.

    **Process:**
    1. Lead Researcher analyzes hypothesis and creates research plan
    2. Custom agents (or auto-generated ones) execute in parallel
    3. Findings are synthesized across all agents
    4. Citations are optionally added to the final report

    **Note:** This is a synchronous endpoint - the request blocks until
    research is complete. For long-running research, use /research/async.
    """,
    responses={
        200: {"description": "Research completed successfully"},
        500: {"description": "Research failed", "model": ErrorResponse},
    },
)
async def start_research(request: ResearchRequest) -> ResearchResponse:
    """
    Start a new synchronous research investigation.

    This endpoint blocks until the research is complete. For long-running
    tasks, consider using the /async endpoint instead.

    Args:
        request: ResearchRequest containing hypothesis and configuration

    Returns:
        ResearchResponse with complete results, findings, and final report

    Raises:
        HTTPException: 500 if research fails
    """
    # Generate unique task ID for tracking
    task_id = str(uuid4())[:8]
    started_at = datetime.now()

    logger.info(f"Starting research task {task_id}: {request.hypothesis[:50]}...")

    try:
        # Step 1: Create Lead Researcher to orchestrate the investigation
        lead = create_lead_researcher(
            name="Lead Researcher",
            max_agents=request.max_agents,
        )

        # Step 2: Convert custom agents from Pydantic models to dict format
        custom_agents = None
        if request.custom_agents:
            custom_agents = [
                {"name": a.name, "focus": a.focus}
                for a in request.custom_agents
            ]

        # Step 3: Execute the orchestration pipeline
        result = await lead.orchestrate(
            hypothesis=request.hypothesis,
            research_questions=request.research_questions,
            custom_agents=custom_agents,
        )

        # Step 4: Optionally add citations to the final report
        final_report = result.final_report
        if request.include_citations and result.agent_results:
            citer = create_citation_agent(citation_style=request.citation_style.value)
            cited_report, references = await citer.add_citations_to_report(
                report=result.final_report,
                agent_results=result.agent_results,
            )
            if references:
                final_report = cited_report + "\n\n" + references

        # Calculate timing and cost
        completed_at = datetime.now()
        duration = (completed_at - started_at).total_seconds()
        # Cost estimate: ~$0.002 per 1K tokens (simplified average)
        estimated_cost = (result.total_tokens / 1000) * 0.002

        # Step 5: Build response from results
        # Convert agent results to response models
        agent_results = [
            AgentResultResponse(
                agent_id=ar.agent_id,
                agent_name=ar.agent_name,
                status=ar.status.value,
                content=ar.content,
                findings=ar.findings,
                sources=ar.sources,
                duration_seconds=ar.duration_seconds,
                token_usage=ar.token_usage,
            )
            for ar in result.agent_results
        ]

        # Convert research plan to response model
        plan_response = ResearchPlanResponse(
            hypothesis=result.plan.hypothesis,
            complexity=result.plan.complexity,
            domains=result.plan.domains,
            research_questions=result.plan.research_questions,
            agent_assignments=result.plan.agent_assignments,
            strategy_notes=result.plan.strategy_notes[:500],  # Truncate for response
        )

        # Assemble final response
        response = ResearchResponse(
            task_id=task_id,
            status=ResearchStatus.COMPLETED,
            hypothesis=request.hypothesis,
            plan=plan_response,
            agent_results=agent_results,
            synthesis=result.synthesis,
            final_report=final_report,
            total_duration_seconds=duration,
            total_tokens=result.total_tokens,
            estimated_cost=estimated_cost,
            created_at=started_at,
            completed_at=completed_at,
        )

        logger.info(
            f"Research task {task_id} completed: "
            f"{duration:.1f}s, {result.total_tokens} tokens"
        )
        return response

    except Exception as e:
        logger.error(f"Research task {task_id} failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@research_router.post(
    "/async",
    response_model=ResearchStatusResponse,
    summary="Start research asynchronously",
    description="""
    Starts research in the background and returns immediately with a task ID.

    Use this endpoint for long-running research tasks. Poll the status
    endpoint to check progress, and retrieve results when complete.

    **Workflow:**
    1. POST to /research/async - returns task_id
    2. GET /research/{task_id}/status - poll for completion
    3. GET /research/{task_id} - retrieve results
    """,
)
async def start_research_async(
    request: ResearchRequest,
    background_tasks: BackgroundTasks,
) -> ResearchStatusResponse:
    """
    Start research as a background task.

    Returns immediately with a task ID that can be used to poll
    for status and retrieve results when complete.

    Args:
        request: ResearchRequest containing hypothesis and configuration
        background_tasks: FastAPI background task manager

    Returns:
        ResearchStatusResponse with task_id for polling
    """
    # Generate unique task ID
    task_id = str(uuid4())[:8]

    # Initialize task tracking state
    _research_tasks[task_id] = {
        "status": ResearchStatus.PENDING,
        "progress": 0.0,
        "current_stage": "initializing",
        "agents_completed": 0,
        "agents_total": request.max_agents,
        "started_at": datetime.now(),
        "result": None,
        "error": None,
    }

    # Queue the research task to run in the background
    background_tasks.add_task(
        _run_research_background,
        task_id,
        request,
    )

    # Return immediately with status
    return ResearchStatusResponse(
        task_id=task_id,
        status=ResearchStatus.PENDING,
        progress=0.0,
        current_stage="queued",
        agents_completed=0,
        agents_total=request.max_agents,
        elapsed_seconds=0.0,
        message="Research task queued",
    )


async def _run_research_background(task_id: str, request: ResearchRequest):
    """
    Background task runner for async research.

    This function is executed in the background by FastAPI's BackgroundTasks.
    It updates the task state in _research_tasks as it progresses.

    Args:
        task_id: Unique identifier for tracking
        request: Original research request
    """
    task = _research_tasks[task_id]
    task["status"] = ResearchStatus.RUNNING
    task["current_stage"] = "planning"

    try:
        # Create Lead Researcher
        lead = create_lead_researcher(
            name="Lead Researcher",
            max_agents=request.max_agents,
        )

        # Convert custom agents
        custom_agents = None
        if request.custom_agents:
            custom_agents = [
                {"name": a.name, "focus": a.focus}
                for a in request.custom_agents
            ]

        # Execute research
        task["current_stage"] = "researching"
        result = await lead.orchestrate(
            hypothesis=request.hypothesis,
            research_questions=request.research_questions,
            custom_agents=custom_agents,
        )

        # Update task with results
        task["status"] = ResearchStatus.COMPLETED
        task["progress"] = 1.0
        task["current_stage"] = "completed"
        task["agents_completed"] = len(result.agent_results)
        task["result"] = result

    except Exception as e:
        # Record error state
        task["status"] = ResearchStatus.FAILED
        task["error"] = str(e)
        task["current_stage"] = "failed"
        logger.error(f"Background research task {task_id} failed: {e}")


@research_router.get(
    "/{task_id}/status",
    response_model=ResearchStatusResponse,
    summary="Get research task status",
    description="Check the current status and progress of an async research task.",
    responses={
        200: {"description": "Task status retrieved"},
        404: {"description": "Task not found"},
    },
)
async def get_research_status(task_id: str) -> ResearchStatusResponse:
    """
    Get the status of an async research task.

    Args:
        task_id: Task identifier from /research/async response

    Returns:
        ResearchStatusResponse with current progress

    Raises:
        HTTPException: 404 if task not found
    """
    if task_id not in _research_tasks:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

    task = _research_tasks[task_id]
    elapsed = (datetime.now() - task["started_at"]).total_seconds()

    return ResearchStatusResponse(
        task_id=task_id,
        status=task["status"],
        progress=task["progress"],
        current_stage=task["current_stage"],
        agents_completed=task["agents_completed"],
        agents_total=task["agents_total"],
        elapsed_seconds=elapsed,
        message=task.get("error"),
    )


@research_router.get(
    "/{task_id}",
    response_model=ResearchResponse,
    summary="Get research results",
    description="Retrieve the full results of a completed research task.",
    responses={
        200: {"description": "Research results retrieved"},
        400: {"description": "Task not completed yet"},
        404: {"description": "Task not found"},
    },
)
async def get_research_result(task_id: str) -> ResearchResponse:
    """
    Get the full results of a completed research task.

    Args:
        task_id: Task identifier from /research/async response

    Returns:
        ResearchResponse with complete results

    Raises:
        HTTPException: 404 if task not found, 400 if not completed
    """
    if task_id not in _research_tasks:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

    task = _research_tasks[task_id]

    # Ensure task is completed
    if task["status"] != ResearchStatus.COMPLETED:
        raise HTTPException(
            status_code=400,
            detail=f"Task not completed. Status: {task['status'].value}"
        )

    result = task["result"]
    elapsed = (datetime.now() - task["started_at"]).total_seconds()

    # Convert agent results to response format
    agent_results = [
        AgentResultResponse(
            agent_id=ar.agent_id,
            agent_name=ar.agent_name,
            status=ar.status.value,
            content=ar.content,
            findings=ar.findings,
            sources=ar.sources,
            duration_seconds=ar.duration_seconds,
            token_usage=ar.token_usage,
        )
        for ar in result.agent_results
    ]

    return ResearchResponse(
        task_id=task_id,
        status=ResearchStatus.COMPLETED,
        hypothesis=result.hypothesis,
        plan=None,  # Simplified for async results
        agent_results=agent_results,
        synthesis=result.synthesis,
        final_report=result.final_report,
        total_duration_seconds=elapsed,
        total_tokens=result.total_tokens,
        estimated_cost=(result.total_tokens / 1000) * 0.002,
        created_at=task["started_at"],
        completed_at=datetime.now(),
    )


@research_router.post(
    "/quick",
    summary="Quick single-agent research",
    description="""
    Run a quick research task with a single agent.

    This is a lightweight alternative to full multi-agent research,
    useful for simple queries that don't require multiple perspectives.
    """,
    responses={
        200: {"description": "Research completed"},
        500: {"description": "Research failed"},
    },
)
async def quick_research(
    hypothesis: str = Query(..., description="Research hypothesis or question"),
    focus: str = Query("general analysis", description="Research focus area"),
) -> Dict[str, Any]:
    """
    Quick research with a single agent.

    Useful for simple research tasks that don't need multi-agent orchestration.

    Args:
        hypothesis: The research hypothesis or question
        focus: Focus area for the research agent

    Returns:
        Dict containing findings, sources, and execution metadata

    Raises:
        HTTPException: 500 if research fails
    """
    try:
        # Create a single research agent
        agent = create_research_agent(
            name="Quick Researcher",
            focus=focus,
        )

        # Execute research
        result = await agent.run(
            hypothesis=hypothesis,
            research_questions=None,
        )

        return {
            "status": result.status.value,
            "content": result.content,
            "findings": result.findings,
            "sources": result.sources,
            "duration_seconds": result.duration_seconds,
            "token_usage": result.token_usage,
        }

    except Exception as e:
        logger.error(f"Quick research failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Search Endpoints
# ============================================================================
# These endpoints provide web search functionality.

@search_router.post(
    "",
    response_model=SearchResponse,
    summary="Perform a web search",
    description="""
    Perform a web search using the configured backend.

    **Supported backends:**
    - **tavily**: AI-optimized search (recommended)
    - **brave**: Privacy-focused search
    - **serper**: Google search results
    - **bing**: Microsoft Bing search
    - **duckduckgo**: Free fallback (limited results)

    The backend is auto-selected based on available API keys,
    or you can specify one explicitly.
    """,
)
async def search(request: QuickSearchRequest) -> SearchResponse:
    """
    Perform a web search using the configured backend.

    Args:
        request: QuickSearchRequest with query and options

    Returns:
        SearchResponse with results and metadata

    Raises:
        HTTPException: 500 if search fails
    """
    try:
        # Create search tool with optional backend override
        tool = WebSearchTool(backend=request.backend)

        # Execute search
        result = await tool.execute(
            query=request.query,
            max_results=request.max_results,
        )

        # Check for errors
        if not result.success:
            raise HTTPException(
                status_code=500,
                detail=result.error or "Search failed"
            )

        # Convert results to response format
        search_results = [
            SearchResult(
                title=r.get("title", ""),
                url=r.get("url", ""),
                snippet=r.get("snippet", ""),
                score=r.get("score"),
            )
            for r in result.data
        ]

        return SearchResponse(
            query=request.query,
            backend=tool.backend,
            results=search_results,
            result_count=len(search_results),
            answer=result.metadata.get("answer"),  # AI answer if available
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@search_router.get(
    "",
    response_model=SearchResponse,
    summary="Quick web search via GET",
    description="Simple search endpoint using query parameters.",
)
async def search_get(
    q: str = Query(..., description="Search query"),
    max_results: int = Query(5, ge=1, le=20, description="Maximum results"),
    backend: Optional[str] = Query(None, description="Search backend to use"),
) -> SearchResponse:
    """
    Quick search endpoint using GET method.

    This is a convenience endpoint for simple searches via URL.

    Args:
        q: Search query string
        max_results: Maximum number of results (1-20)
        backend: Optional backend override

    Returns:
        SearchResponse with results
    """
    # Delegate to POST handler
    request = QuickSearchRequest(query=q, max_results=max_results, backend=backend)
    return await search(request)


@search_router.get(
    "/backends",
    summary="Get available search backends",
    description="Returns which search backends are configured and available.",
)
async def get_search_backends() -> Dict[str, bool]:
    """
    Get the availability status of all search backends.

    Returns:
        Dict mapping backend names to availability (True/False)
    """
    return WebSearchTool.get_available_backends()


# ============================================================================
# Citation Endpoints
# ============================================================================
# These endpoints handle citation processing.

@citation_router.post(
    "",
    response_model=CitationResponse,
    summary="Add citations to content",
    description="""
    Process content and add inline citations based on provided sources.

    The citation agent identifies factual claims in the content and
    matches them to the provided sources, adding citation markers
    [1], [2], etc. and generating a formatted reference list.

    **Supported styles:**
    - simple: [1] Title (URL)
    - apa: APA format
    - mla: MLA format
    - chicago: Chicago style
    """,
)
async def add_citations(request: CitationRequest) -> CitationResponse:
    """
    Add inline citations and references to content.

    Args:
        request: CitationRequest with content and sources

    Returns:
        CitationResponse with cited content and reference list

    Raises:
        HTTPException: 500 if citation processing fails
    """
    try:
        # Create citation agent with specified style
        citer = create_citation_agent(citation_style=request.style.value)

        # Process content
        result = await citer.process_content(
            content=request.content,
            sources=request.sources,
        )

        return CitationResponse(
            original_length=len(request.content),
            cited_content=result.cited_content,
            reference_list=result.reference_list,
            citations_added=result.stats["citations_added"],
            unique_sources_cited=result.stats["unique_sources_cited"],
        )

    except Exception as e:
        logger.error(f"Citation processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Health & Info Endpoints
# ============================================================================
# These endpoints provide API health checks and framework information.

@router.get(
    "/health",
    response_model=HealthResponse,
    tags=["Health"],
    summary="Health check",
    description="Check the health status of the API and its components.",
)
async def health_check() -> HealthResponse:
    """
    Check the health of the API and its components.

    Returns status of:
    - API server
    - OpenAI configuration
    - Search backend configuration

    Returns:
        HealthResponse with component statuses
    """
    components = {
        "api": "healthy",
        "openai": "configured" if settings.OPENAI_API_KEY else "not configured",
        "search": "configured" if settings.TAVILY_API_KEY else "fallback only",
    }

    return HealthResponse(
        status="healthy",
        version="1.0.0",
        timestamp=datetime.now(),
        components=components,
    )


@router.get(
    "/info",
    response_model=FrameworkInfoResponse,
    tags=["Info"],
    summary="Framework information",
    description="Get detailed information about the framework and its capabilities.",
)
async def get_info() -> FrameworkInfoResponse:
    """
    Get information about the framework and its capabilities.

    Provides details about:
    - Framework version and description
    - Available search backends
    - Supported model tiers
    - Feature list

    Returns:
        FrameworkInfoResponse with capability details
    """
    # Get backend availability status
    backends = WebSearchTool.get_available_backends()
    backend_status = [
        BackendStatus(name=name, available=available, configured=available)
        for name, available in backends.items()
    ]

    return FrameworkInfoResponse(
        name="Multi-Agent Research Framework",
        version="1.0.0",
        description="AI-powered multi-agent research system using OpenAI GPT models",
        available_backends=backend_status,
        model_tiers=["default", "budget", "premium", "fast"],
        max_agents=settings.MAX_AGENTS,
        features=[
            "Multi-agent orchestration",
            "Parallel research execution",
            "RLM-based context management",
            "Automatic citation generation",
            "Multiple search backends",
            "Chain-of-thought reasoning",
        ],
    )

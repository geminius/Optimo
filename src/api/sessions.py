"""
Sessions API endpoints for the robotics model optimization platform.

This module provides endpoints for listing and filtering optimization sessions
with pagination support and model information enrichment.
"""

import logging
from typing import Optional, List
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, Request, Query
from fastapi import status as http_status
from pydantic import BaseModel, Field

from .models import SessionListResponse, OptimizationSessionSummary, ErrorResponse
from .dependencies import get_current_user
from .auth import User
from ..services.memory_manager import MemoryManager
from ..models.store import ModelStore


logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/optimization", tags=["Sessions"])


class SessionFilter(BaseModel):
    """Model for session filtering parameters."""
    status: Optional[str] = Field(None, description="Filter by session status")
    model_id: Optional[str] = Field(None, description="Filter by model ID")
    start_date: Optional[datetime] = Field(None, description="Filter sessions created after this date")
    end_date: Optional[datetime] = Field(None, description="Filter sessions created before this date")
    skip: int = Field(0, ge=0, description="Number of sessions to skip for pagination")
    limit: int = Field(50, ge=1, le=100, description="Maximum number of sessions to return")
    
    def validate(self) -> None:
        """Validate filter parameters."""
        valid_statuses = ["pending", "running", "completed", "failed", "cancelled"]
        
        if self.limit > 100:
            raise ValueError("Limit cannot exceed 100")
        if self.skip < 0:
            raise ValueError("Skip must be non-negative")
        if self.status and self.status.lower() not in valid_statuses:
            raise ValueError(f"Invalid status: {self.status}. Must be one of {valid_statuses}")
        if self.start_date and self.end_date and self.start_date > self.end_date:
            raise ValueError("start_date must be before end_date")


async def get_model_store(request: Request) -> ModelStore:
    """Get ModelStore from app state."""
    if not hasattr(request.app.state, 'platform_integrator'):
        raise HTTPException(
            status_code=http_status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Platform integrator not available"
        )
    
    platform_integrator = request.app.state.platform_integrator
    model_store = platform_integrator.get_model_store()
    
    if model_store is None:
        raise HTTPException(
            status_code=http_status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model store not available"
        )
    
    return model_store


async def get_memory_manager(request: Request) -> MemoryManager:
    """Get MemoryManager from app state."""
    if not hasattr(request.app.state, 'platform_integrator'):
        raise HTTPException(
            status_code=http_status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Platform integrator not available"
        )
    
    platform_integrator = request.app.state.platform_integrator
    memory_manager = platform_integrator.get_memory_manager()
    
    if memory_manager is None:
        raise HTTPException(
            status_code=http_status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Memory manager not available"
        )
    
    return memory_manager


@router.get(
    "/sessions",
    response_model=SessionListResponse,
    summary="List Optimization Sessions",
    description="""
    Retrieve a paginated list of optimization sessions with optional filtering.
    
    This endpoint provides comprehensive session information including:
    - Session status and progress
    - Associated model information
    - Optimization techniques applied
    - Performance metrics (for completed sessions)
    - Timestamps and duration
    
    **Filtering Options:**
    - **status**: Filter by session status (pending, running, completed, failed, cancelled)
    - **model_id**: Show only sessions for a specific model
    - **start_date**: Show sessions created after this date (ISO 8601 format)
    - **end_date**: Show sessions created before this date (ISO 8601 format)
    
    **Pagination:**
    - **skip**: Number of sessions to skip (default: 0)
    - **limit**: Maximum sessions to return (default: 50, max: 100)
    
    **Authorization:**
    - Administrators can view all sessions
    - Regular users can only view their own sessions
    
    **Use Cases:**
    - Display optimization history on dashboard
    - Monitor active optimizations
    - Filter sessions by status or model
    - Track optimization trends over time
    
    **Performance:**
    - Response time: < 1s for typical queries
    - Supports up to 100 sessions per request
    - Model information enriched from ModelStore
    
    **Example Queries:**
    ```bash
    # Get all running sessions
    GET /optimization/sessions?status=running
    
    # Get completed sessions for a specific model
    GET /optimization/sessions?status=completed&model_id=550e8400-e29b-41d4-a716-446655440000
    
    # Get sessions from last week with pagination
    GET /optimization/sessions?start_date=2024-01-01T00:00:00Z&skip=0&limit=20
    ```
    """,
    responses={
        200: {
            "description": "Sessions retrieved successfully",
            "content": {
                "application/json": {
                    "example": {
                        "sessions": [
                            {
                                "session_id": "session_abc123",
                                "model_id": "550e8400-e29b-41d4-a716-446655440000",
                                "model_name": "robotics_vla_model.pt",
                                "status": "running",
                                "progress_percentage": 45.0,
                                "techniques": ["quantization", "pruning"],
                                "size_reduction_percent": None,
                                "speed_improvement_percent": None,
                                "created_at": "2024-01-01T12:00:00Z",
                                "updated_at": "2024-01-01T12:15:00Z",
                                "completed_at": None
                            }
                        ],
                        "total": 47,
                        "skip": 0,
                        "limit": 50
                    }
                }
            }
        },
        400: {
            "description": "Bad request - Invalid query parameters",
            "model": ErrorResponse,
            "content": {
                "application/json": {
                    "example": {
                        "error": "ValidationError",
                        "message": "Invalid status: invalid_status. Must be one of ['pending', 'running', 'completed', 'failed', 'cancelled']",
                        "details": {
                            "field": "status",
                            "provided_value": "invalid_status"
                        },
                        "timestamp": "2024-01-01T12:00:00Z",
                        "request_id": "req_abc123"
                    }
                }
            }
        },
        401: {
            "description": "Unauthorized - Invalid or missing authentication token",
            "model": ErrorResponse
        },
        500: {
            "description": "Internal server error - Failed to retrieve sessions",
            "model": ErrorResponse
        },
        503: {
            "description": "Service unavailable - Required services not initialized",
            "model": ErrorResponse
        }
    }
)
async def list_optimization_sessions(
    status: Optional[str] = Query(None, description="Filter by session status (pending, running, completed, failed, cancelled)"),
    model_id: Optional[str] = Query(None, description="Filter by model ID"),
    start_date: Optional[datetime] = Query(None, description="Filter sessions created after this date (ISO format)"),
    end_date: Optional[datetime] = Query(None, description="Filter sessions created before this date (ISO format)"),
    skip: int = Query(0, ge=0, description="Number of sessions to skip for pagination"),
    limit: int = Query(50, ge=1, le=100, description="Maximum number of sessions to return"),
    current_user: User = Depends(get_current_user),
    memory_manager: MemoryManager = Depends(get_memory_manager),
    model_store: ModelStore = Depends(get_model_store)
) -> SessionListResponse:
    """
    List all optimization sessions with optional filtering and pagination.
    
    This endpoint retrieves optimization sessions from the MemoryManager,
    applies filters, enriches with model information, and returns paginated results.
    
    Query Parameters:
        - status: Filter by session status
        - model_id: Filter by specific model
        - start_date: Filter sessions created after this date
        - end_date: Filter sessions created before this date
        - skip: Pagination offset (default: 0)
        - limit: Pagination limit (default: 50, max: 100)
    
    Returns:
        SessionListResponse: Paginated list of optimization sessions with model information
        
    Raises:
        HTTPException: If query parameters are invalid or retrieval fails
    """
    try:
        logger.info(f"Listing optimization sessions for user: {current_user.username}")
        
        # Create and validate filter
        session_filter = SessionFilter(
            status=status,
            model_id=model_id,
            start_date=start_date,
            end_date=end_date,
            skip=skip,
            limit=limit
        )
        
        try:
            session_filter.validate()
        except ValueError as e:
            logger.warning(f"Invalid filter parameters: {e}")
            raise HTTPException(
                status_code=http_status.HTTP_400_BAD_REQUEST,
                detail=str(e)
            )
        
        # Build status filter list for MemoryManager
        status_filter_list = None
        if session_filter.status:
            status_filter_list = [session_filter.status.lower()]
        
        # Query MemoryManager with filters
        logger.debug(
            f"Querying sessions with filters: status={status_filter_list}, model_id={session_filter.model_id}, skip={session_filter.skip}, limit={session_filter.limit}",
            extra={
                "component": "SessionsAPI",
                "user_id": current_user.id,
                "username": current_user.username
            }
        )
        
        try:
            # Get sessions from MemoryManager
            session_list = memory_manager.list_sessions(
                status_filter=status_filter_list,
                model_id_filter=session_filter.model_id,
                limit=session_filter.limit,
                offset=session_filter.skip
            )
            
            logger.debug(
                f"Retrieved {len(session_list)} sessions from MemoryManager",
                extra={
                    "component": "SessionsAPI",
                    "user_id": current_user.id,
                    "session_count": len(session_list)
                }
            )
        except Exception as e:
            logger.error(
                f"Failed to query MemoryManager: {e}",
                extra={
                    "component": "SessionsAPI",
                    "user_id": current_user.id
                }
            )
            raise HTTPException(
                status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to retrieve sessions: {str(e)}"
            )
        
        # Enrich session data with model information and format response
        enriched_sessions: List[OptimizationSessionSummary] = []
        
        for session_info in session_list:
            try:
                # Retrieve full session to get detailed information
                session = memory_manager.retrieve_session(session_info["id"])
                
                if session is None:
                    logger.warning(
                        f"Could not retrieve full session data for {session_info['id']}",
                        extra={
                            "component": "SessionsAPI",
                            "session_id": session_info["id"]
                        }
                    )
                    continue
                
                # Authorization check: non-admin users can only see their own sessions
                # Check if session has owner information
                session_owner = getattr(session, 'owner_id', None) or getattr(session, 'user_id', None)
                
                if current_user.role != "administrator":
                    # If session has owner info, check if it matches current user
                    if session_owner and session_owner != current_user.id:
                        logger.debug(
                            f"User {current_user.username} denied access to session {session.id} owned by {session_owner}",
                            extra={
                                "component": "SessionsAPI",
                                "user_id": current_user.id,
                                "session_id": session.id,
                                "session_owner": session_owner
                            }
                        )
                        continue  # Skip sessions not owned by current user
                
                # Get model name from ModelStore
                model_name = "Unknown Model"
                try:
                    model_metadata = model_store.get_metadata(session.model_id)
                    if model_metadata:
                        model_name = model_metadata.name
                    else:
                        # Fallback: try to extract from model_id or use model_id
                        model_name = session.model_id
                        logger.debug(f"Model metadata not found for {session.model_id}, using model_id as name")
                except Exception as e:
                    logger.debug(f"Could not fetch model name for {session.model_id}: {e}")
                    model_name = session.model_id
                
                # Extract techniques from session
                techniques = []
                if hasattr(session, 'criteria') and session.criteria:
                    if hasattr(session.criteria, 'constraints') and session.criteria.constraints:
                        if hasattr(session.criteria.constraints, 'allowed_techniques'):
                            techniques = [tech.value for tech in session.criteria.constraints.allowed_techniques]
                
                # Extract performance metrics if available
                size_reduction_percent = None
                speed_improvement_percent = None
                
                if hasattr(session, 'results') and session.results:
                    if hasattr(session.results, 'size_reduction_percent'):
                        size_reduction_percent = session.results.size_reduction_percent
                    if hasattr(session.results, 'speed_improvement_percent'):
                        speed_improvement_percent = session.results.speed_improvement_percent
                
                # Calculate progress percentage
                progress_percentage = 0.0
                if session.status.value == "completed":
                    progress_percentage = 100.0
                elif session.status.value == "running":
                    # Calculate based on completed steps
                    if hasattr(session, 'steps') and session.steps:
                        total_steps = len(session.steps)
                        completed_steps = sum(1 for step in session.steps if step.status == "completed")
                        if total_steps > 0:
                            progress_percentage = (completed_steps / total_steps) * 100.0
                
                # Parse timestamps
                created_at = session.created_at if isinstance(session.created_at, datetime) else datetime.fromisoformat(session_info["created_at"])
                updated_at = datetime.now()  # Default to now if not available
                completed_at = None
                
                if session.completed_at:
                    completed_at = session.completed_at if isinstance(session.completed_at, datetime) else datetime.fromisoformat(session_info.get("completed_at"))
                
                # Create session summary
                session_summary = OptimizationSessionSummary(
                    session_id=session.id,
                    model_id=session.model_id,
                    model_name=model_name,
                    status=session.status.value,
                    progress_percentage=progress_percentage,
                    techniques=techniques,
                    size_reduction_percent=size_reduction_percent,
                    speed_improvement_percent=speed_improvement_percent,
                    created_at=created_at,
                    updated_at=updated_at,
                    completed_at=completed_at
                )
                
                enriched_sessions.append(session_summary)
                
            except Exception as e:
                logger.warning(f"Failed to enrich session {session_info['id']}: {e}")
                # Continue processing other sessions even if one fails
                continue
        
        # Apply date filtering if specified (post-processing since MemoryManager doesn't support date filters)
        if session_filter.start_date or session_filter.end_date:
            filtered_sessions = []
            for session_summary in enriched_sessions:
                if session_filter.start_date and session_summary.created_at < session_filter.start_date:
                    continue
                if session_filter.end_date and session_summary.created_at > session_filter.end_date:
                    continue
                filtered_sessions.append(session_summary)
            enriched_sessions = filtered_sessions
        
        # Get total count (for pagination metadata)
        # Note: This is an approximation since we're doing post-filtering
        total_count = len(enriched_sessions) + session_filter.skip
        
        logger.info(f"Returning {len(enriched_sessions)} sessions (total: {total_count})")
        
        return SessionListResponse(
            sessions=enriched_sessions,
            total=total_count,
            skip=session_filter.skip,
            limit=session_filter.limit
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Unexpected error listing sessions: {e}", exc_info=True)
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list optimization sessions: {str(e)}"
        )

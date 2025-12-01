"""
Dashboard API endpoints for the robotics model optimization platform.

This module provides endpoints for retrieving dashboard statistics
and aggregate metrics about the platform's optimization activities.
"""

import logging
from typing import Dict, Any
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, status, Request

from .models import DashboardStats, ErrorResponse
from .dependencies import get_current_user
from .auth import User
from ..services.optimization_manager import OptimizationManager
from ..models.store import ModelStore
from ..services.memory_manager import MemoryManager
from ..services.cache_service import CacheService


logger = logging.getLogger(__name__)

# Cache TTL for dashboard statistics (30 seconds)
DASHBOARD_STATS_CACHE_TTL = 30.0

# Create router
router = APIRouter(prefix="/dashboard", tags=["Dashboard"])


async def get_model_store(request: Request) -> ModelStore:
    """Get ModelStore from app state."""
    if not hasattr(request.app.state, 'platform_integrator'):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Platform integrator not available"
        )
    
    platform_integrator = request.app.state.platform_integrator
    model_store = platform_integrator.get_model_store()
    
    if model_store is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model store not available"
        )
    
    return model_store


async def get_memory_manager(request: Request) -> MemoryManager:
    """Get MemoryManager from app state."""
    if not hasattr(request.app.state, 'platform_integrator'):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Platform integrator not available"
        )
    
    platform_integrator = request.app.state.platform_integrator
    memory_manager = platform_integrator.get_memory_manager()
    
    if memory_manager is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Memory manager not available"
        )
    
    return memory_manager


async def get_optimization_manager_from_request(request: Request) -> OptimizationManager:
    """Get OptimizationManager from app state."""
    if not hasattr(request.app.state, 'optimization_manager'):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Optimization manager not available"
        )
    
    return request.app.state.optimization_manager


@router.get(
    "/stats",
    response_model=DashboardStats,
    summary="Get Dashboard Statistics",
    description="""
    Retrieve aggregate statistics and metrics for the dashboard view.
    
    This endpoint provides a comprehensive overview of the platform's current state
    and historical performance by aggregating data from multiple services:
    
    - **Model Store**: Total number of uploaded models
    - **Optimization Manager**: Currently active optimization sessions
    - **Memory Manager**: Historical session data and completion statistics
    
    **Metrics Included:**
    - Total models in the system
    - Active optimization sessions
    - Completed and failed optimization counts
    - Average size reduction across completed optimizations
    - Average speed improvement across completed optimizations
    - Total session count
    
    **Use Cases:**
    - Display platform overview on dashboard
    - Monitor system activity and performance
    - Track optimization success rates
    - Analyze historical trends
    
    **Performance:**
    - Response time: < 500ms typical
    - Cached for 30 seconds to reduce database load
    - Aggregates data from up to 1000 recent sessions
    - Cache automatically invalidated on session updates
    
    **Authentication:**
    Requires valid Bearer token in Authorization header.
    """,
    responses={
        200: {
            "description": "Dashboard statistics retrieved successfully",
            "content": {
                "application/json": {
                    "example": {
                        "total_models": 15,
                        "active_optimizations": 3,
                        "completed_optimizations": 42,
                        "failed_optimizations": 2,
                        "average_size_reduction": 32.5,
                        "average_speed_improvement": 18.7,
                        "total_sessions": 47,
                        "last_updated": "2024-01-01T12:00:00Z"
                    }
                }
            }
        },
        401: {
            "description": "Unauthorized - Invalid or missing authentication token",
            "model": ErrorResponse
        },
        500: {
            "description": "Internal server error - Failed to calculate statistics",
            "model": ErrorResponse
        },
        503: {
            "description": "Service unavailable - Required services not initialized",
            "model": ErrorResponse
        }
    }
)
async def get_dashboard_stats(
    current_user: User = Depends(get_current_user),
    optimization_manager: OptimizationManager = Depends(get_optimization_manager_from_request),
    model_store: ModelStore = Depends(get_model_store),
    memory_manager: MemoryManager = Depends(get_memory_manager)
) -> DashboardStats:
    """
    Get dashboard statistics including model counts, optimization metrics, and performance data.
    
    This endpoint aggregates data from multiple services to provide a comprehensive
    overview of the platform's current state and historical performance.
    
    Uses caching to reduce database load and improve response times. Cache is
    automatically invalidated after 30 seconds or when sessions are updated.
    
    Returns:
        DashboardStats: Aggregate statistics for the dashboard
        
    Raises:
        HTTPException: If statistics cannot be calculated
    """
    try:
        logger.info(
            f"Fetching dashboard statistics for user: {current_user.username}",
            extra={
                "component": "DashboardAPI",
                "user_id": current_user.id,
                "username": current_user.username,
                "role": current_user.role
            }
        )
        
        # Get cache service
        cache_service = CacheService()
        cache_key = "dashboard:stats"
        
        # Try to get from cache first
        def compute_stats() -> DashboardStats:
            """Compute dashboard statistics (called if cache miss)."""
            return _compute_dashboard_stats(
                current_user,
                optimization_manager,
                model_store,
                memory_manager
            )
        
        # Get from cache or compute
        stats = cache_service.get_or_set(
            cache_key,
            compute_stats,
            ttl_seconds=DASHBOARD_STATS_CACHE_TTL
        )
        
        logger.info(
            f"Dashboard statistics retrieved: {stats.total_models} models, {stats.active_optimizations} active",
            extra={
                "component": "DashboardAPI",
                "user_id": current_user.id,
                "total_models": stats.total_models,
                "active_optimizations": stats.active_optimizations,
                "completed_optimizations": stats.completed_optimizations
            }
        )
        return stats
        
    except Exception as e:
        logger.error(f"Failed to get dashboard statistics: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve dashboard statistics: {str(e)}"
        )


def _compute_dashboard_stats(
    current_user: User,
    optimization_manager: OptimizationManager,
    model_store: ModelStore,
    memory_manager: MemoryManager
) -> DashboardStats:
    """
    Compute dashboard statistics from platform services.
    
    This function is separated to allow caching of the computation.
    
    Args:
        current_user: Authenticated user
        optimization_manager: OptimizationManager instance
        model_store: ModelStore instance
        memory_manager: MemoryManager instance
        
    Returns:
        DashboardStats with computed statistics
    """
    try:
        # Query ModelStore for total model count
        total_models = 0
        try:
            models = model_store.list_models()
            total_models = len(models)
            logger.debug(f"Total models: {total_models}")
        except Exception as e:
            logger.warning(f"Failed to get model count: {e}")
        
        # Query OptimizationManager for active sessions
        active_optimizations = 0
        try:
            active_session_ids = optimization_manager.get_active_sessions()
            active_optimizations = len(active_session_ids)
            logger.debug(f"Active optimizations: {active_optimizations}")
        except Exception as e:
            logger.warning(f"Failed to get active sessions: {e}")
        
        # Query MemoryManager for session statistics
        completed_optimizations = 0
        failed_optimizations = 0
        total_sessions = 0
        average_size_reduction = 0.0
        average_speed_improvement = 0.0
        
        try:
            session_stats = memory_manager.get_session_statistics()
            
            # Get status distribution
            status_distribution = session_stats.get("status_distribution", {})
            completed_optimizations = status_distribution.get("completed", 0)
            failed_optimizations = status_distribution.get("failed", 0)
            total_sessions = session_stats.get("total_sessions", 0)
            
            logger.debug(f"Session statistics: completed={completed_optimizations}, failed={failed_optimizations}, total={total_sessions}")
        except Exception as e:
            logger.warning(f"Failed to get session statistics: {e}")
        
        # Calculate averages from completed sessions
        try:
            # Get completed sessions to calculate averages
            completed_sessions = memory_manager.list_sessions(
                status_filter=["completed"],
                limit=1000  # Get recent completed sessions for average calculation
            )
            
            if completed_sessions:
                size_reductions = []
                speed_improvements = []
                
                # Extract metrics from each completed session
                for session_info in completed_sessions:
                    try:
                        # Retrieve full session to get results
                        session = memory_manager.retrieve_session(session_info["id"])
                        if session and hasattr(session, 'results') and session.results:
                            # Extract size reduction if available
                            if hasattr(session.results, 'size_reduction_percent'):
                                size_reductions.append(session.results.size_reduction_percent)
                            
                            # Extract speed improvement if available
                            if hasattr(session.results, 'speed_improvement_percent'):
                                speed_improvements.append(session.results.speed_improvement_percent)
                    except Exception as e:
                        logger.debug(f"Could not extract metrics from session {session_info['id']}: {e}")
                        continue
                
                # Calculate averages
                if size_reductions:
                    average_size_reduction = sum(size_reductions) / len(size_reductions)
                    logger.debug(f"Average size reduction: {average_size_reduction:.2f}%")
                
                if speed_improvements:
                    average_speed_improvement = sum(speed_improvements) / len(speed_improvements)
                    logger.debug(f"Average speed improvement: {average_speed_improvement:.2f}%")
        except Exception as e:
            logger.warning(f"Failed to calculate averages: {e}")
        
        # Create and return dashboard statistics
        stats = DashboardStats(
            total_models=total_models,
            active_optimizations=active_optimizations,
            completed_optimizations=completed_optimizations,
            failed_optimizations=failed_optimizations,
            average_size_reduction=average_size_reduction,
            average_speed_improvement=average_speed_improvement,
            total_sessions=total_sessions,
            last_updated=datetime.now()
        )
        
        logger.debug(
            f"Dashboard statistics computed: {stats.total_models} models, {stats.active_optimizations} active",
            extra={
                "component": "DashboardAPI",
                "user_id": current_user.id,
                "total_models": stats.total_models,
                "active_optimizations": stats.active_optimizations
            }
        )
        return stats
        
    except Exception as e:
        logger.error(f"Failed to compute dashboard statistics: {e}", exc_info=True)
        raise

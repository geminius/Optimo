"""
Monitoring and system status endpoints.
"""

import psutil
import logging
from typing import Dict, Any
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException

from .models import SystemStatusResponse, SystemMetrics
from .dependencies import get_current_user, get_optimization_manager, get_admin_user
from .auth import User
from ..services.optimization_manager import OptimizationManager


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/monitoring", tags=["Monitoring"])


@router.get("/system", response_model=SystemStatusResponse)
async def get_system_status(
    current_user: User = Depends(get_current_user),
    optimization_manager: OptimizationManager = Depends(get_optimization_manager)
):
    """Get comprehensive system status and metrics."""
    try:
        # Get system metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Try to get GPU usage if available
        gpu_percent = None
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu_percent = gpus[0].load * 100
        except ImportError:
            pass
        
        # Get optimization manager metrics
        active_sessions = len(optimization_manager.get_active_sessions())
        
        # Count total models (placeholder - would be from database in real implementation)
        total_models = 0
        try:
            from pathlib import Path
            upload_dir = Path("uploads")
            if upload_dir.exists():
                total_models = len(list(upload_dir.glob("*")))
        except Exception:
            pass
        
        metrics = SystemMetrics(
            cpu_usage_percent=cpu_percent,
            memory_usage_percent=memory.percent,
            disk_usage_percent=disk.percent,
            gpu_usage_percent=gpu_percent,
            active_sessions=active_sessions,
            total_models=total_models,
            timestamp=datetime.now()
        )
        
        # Service status
        services = {
            "optimization_manager": {
                "status": "healthy",
                "active_sessions": active_sessions,
                "max_sessions": optimization_manager.max_concurrent_sessions
            },
            "analysis_agent": {
                "status": "healthy" if optimization_manager.analysis_agent else "unavailable"
            },
            "planning_agent": {
                "status": "healthy" if optimization_manager.planning_agent else "unavailable"
            },
            "evaluation_agent": {
                "status": "healthy" if optimization_manager.evaluation_agent else "unavailable"
            },
            "optimization_agents": {
                "status": "healthy",
                "available_techniques": list(optimization_manager.optimization_agents.keys())
            }
        }
        
        return SystemStatusResponse(
            status="healthy",
            metrics=metrics,
            services=services,
            version="1.0.0",
            uptime_seconds=0.0  # Would track actual uptime in production
        )
        
    except Exception as e:
        logger.error(f"Failed to get system status: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get system status: {str(e)}"
        )


@router.get("/sessions/metrics")
async def get_session_metrics(
    current_user: User = Depends(get_current_user),
    optimization_manager: OptimizationManager = Depends(get_optimization_manager)
):
    """Get detailed metrics for all active sessions."""
    try:
        active_session_ids = optimization_manager.get_active_sessions()
        
        session_metrics = []
        for session_id in active_session_ids:
            try:
                status_info = optimization_manager.get_session_status(session_id)
                session_metrics.append({
                    "session_id": session_id,
                    "status": status_info["status"],
                    "progress_percentage": status_info["progress_percentage"],
                    "current_step": status_info["current_step"],
                    "model_id": status_info["session_data"]["model_id"],
                    "steps_completed": status_info["session_data"]["steps_completed"],
                    "start_time": status_info["start_time"],
                    "last_update": status_info["last_update"]
                })
            except Exception as e:
                logger.warning(f"Failed to get metrics for session {session_id}: {e}")
        
        return {
            "total_active_sessions": len(active_session_ids),
            "session_details": session_metrics,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get session metrics: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get session metrics: {str(e)}"
        )


@router.get("/logs")
async def get_system_logs(
    lines: int = 100,
    level: str = "INFO",
    current_user: User = Depends(get_admin_user)
):
    """Get system logs (admin only)."""
    try:
        # This is a placeholder implementation
        # In production, you would read from actual log files
        logs = [
            {
                "timestamp": datetime.now().isoformat(),
                "level": "INFO",
                "message": "System running normally",
                "component": "api"
            },
            {
                "timestamp": datetime.now().isoformat(),
                "level": "INFO", 
                "message": "Optimization manager initialized",
                "component": "optimization_manager"
            }
        ]
        
        return {
            "logs": logs[-lines:],
            "total_lines": len(logs),
            "level_filter": level,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get system logs: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get system logs: {str(e)}"
        )


@router.get("/health/detailed")
async def detailed_health_check(
    current_user: User = Depends(get_current_user),
    optimization_manager: OptimizationManager = Depends(get_optimization_manager)
):
    """Detailed health check with component status."""
    try:
        health_status = {
            "overall_status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "components": {
                "api": {
                    "status": "healthy",
                    "response_time_ms": 0.5
                },
                "optimization_manager": {
                    "status": "healthy" if optimization_manager else "unavailable",
                    "active_sessions": len(optimization_manager.get_active_sessions()) if optimization_manager else 0
                },
                "agents": {
                    "analysis_agent": "healthy" if optimization_manager and optimization_manager.analysis_agent else "unavailable",
                    "planning_agent": "healthy" if optimization_manager and optimization_manager.planning_agent else "unavailable",
                    "evaluation_agent": "healthy" if optimization_manager and optimization_manager.evaluation_agent else "unavailable",
                    "optimization_agents": len(optimization_manager.optimization_agents) if optimization_manager else 0
                },
                "storage": {
                    "upload_directory": "healthy",  # Would check actual directory access
                    "database": "healthy"  # Would check database connection
                }
            },
            "system_resources": {
                "cpu_usage_percent": psutil.cpu_percent(),
                "memory_usage_percent": psutil.virtual_memory().percent,
                "disk_usage_percent": psutil.disk_usage('/').percent
            }
        }
        
        # Determine overall status
        component_statuses = []
        for component, details in health_status["components"].items():
            if isinstance(details, dict) and "status" in details:
                component_statuses.append(details["status"])
            elif isinstance(details, dict):
                # Check nested components
                for subcomponent, status in details.items():
                    if isinstance(status, str) and status in ["healthy", "unavailable", "error"]:
                        component_statuses.append(status)
        
        if "error" in component_statuses:
            health_status["overall_status"] = "error"
        elif "unavailable" in component_statuses:
            health_status["overall_status"] = "degraded"
        
        return health_status
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "overall_status": "error",
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }
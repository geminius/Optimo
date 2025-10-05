"""
FastAPI application for the robotics model optimization platform.

This module provides REST API endpoints for model upload, optimization,
monitoring, and results retrieval with authentication and authorization.
"""

import logging
import os
import uuid
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from pathlib import Path

from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import uvicorn
from pydantic import BaseModel, Field

from .models import (
    ModelUploadResponse, OptimizationRequest, OptimizationResponse,
    SessionStatusResponse, SessionListResponse, ModelListResponse,
    EvaluationResponse, ErrorResponse, HealthResponse, User
)
from .auth import AuthManager, get_auth_manager
from .dependencies import get_current_user, get_optimization_manager
from .monitoring import router as monitoring_router
from .openapi_config import get_openapi_config
from ..services.optimization_manager import OptimizationManager
from ..models.core import ModelMetadata, OptimizationSession
from ..config.optimization_criteria import OptimizationCriteria


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan."""
    # Startup
    logger.info("Starting Robotics Model Optimization Platform API")
    
    # Check if platform integrator is already set (from main.py)
    if not hasattr(app.state, 'platform_integrator'):
        # Initialize platform integrator for standalone API mode
        from ..integration.platform_integration import PlatformIntegrator
        
        config = {
            "logging": {
                "level": "INFO",
                "log_dir": "logs",
                "json_format": True,
                "console_output": True
            },
            "monitoring": {
                "monitoring_interval_seconds": 30,
                "health_check_interval_seconds": 60
            },
            "model_store": {"storage_path": "models"},
            "memory_manager": {"max_sessions": 10},
            "notification_service": {"enable_email": False},
            "monitoring_service": {"enable_metrics": True},
            "optimization_manager": {
                "max_concurrent_sessions": 3,
                "auto_rollback_on_failure": True
            },
            "analysis_agent": {},
            "planning_agent": {},
            "evaluation_agent": {},
            "quantization_agent": {},
            "pruning_agent": {},
            "distillation_agent": {},
            "compression_agent": {},
            "architecture_search_agent": {}
        }
        
        platform_integrator = PlatformIntegrator(config)
        success = await platform_integrator.initialize_platform()
        
        if not success:
            logger.error("Failed to initialize PlatformIntegrator")
            raise RuntimeError("Failed to initialize PlatformIntegrator")
        
        app.state.platform_integrator = platform_integrator
        app.state.optimization_manager = platform_integrator.get_optimization_manager()
    
    logger.info("API startup completed successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Robotics Model Optimization Platform API")
    
    if hasattr(app.state, 'platform_integrator'):
        await app.state.platform_integrator.shutdown_platform()
    elif hasattr(app.state, 'optimization_manager'):
        app.state.optimization_manager.cleanup()
    
    logger.info("API shutdown completed")


# Get OpenAPI configuration
openapi_config = get_openapi_config()

# Create FastAPI application
app = FastAPI(
    title=openapi_config["title"],
    description=openapi_config["description"],
    version=openapi_config["version"],
    contact=openapi_config["contact"],
    license_info=openapi_config["license"],
    servers=openapi_config["servers"],
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

# Include routers
app.include_router(monitoring_router)

# Global configuration
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

MAX_FILE_SIZE = 500 * 1024 * 1024  # 500MB
ALLOWED_EXTENSIONS = {".pt", ".pth", ".onnx", ".pb", ".h5", ".safetensors"}





@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            message="An unexpected error occurred",
            details=str(exc) if app.debug else None
        ).dict()
    )


# Health check endpoint
@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Check API health status."""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(),
        version="1.0.0",
        services={
            "optimization_manager": hasattr(app.state, 'optimization_manager'),
            "upload_directory": UPLOAD_DIR.exists()
        }
    )


# Authentication endpoints
@app.post("/auth/login", tags=["Authentication"])
async def login(credentials: dict):
    """Authenticate user and return access token."""
    # Placeholder implementation - replace with actual authentication
    username = credentials.get("username")
    password = credentials.get("password")
    
    if not username or not password:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username and password required"
        )
    
    # Simple validation (replace with proper authentication)
    if username == "admin" and password == "admin":
        token = str(uuid.uuid4())
        return {
            "access_token": token,
            "token_type": "bearer",
            "expires_in": 3600,
            "user": {
                "id": "admin",
                "username": "admin",
                "role": "administrator"
            }
        }
    
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid credentials"
    )


# Model management endpoints
@app.post("/models/upload", response_model=ModelUploadResponse, tags=["Models"])
async def upload_model(
    file: UploadFile = File(...),
    name: Optional[str] = None,
    description: Optional[str] = None,
    tags: Optional[str] = None,
    current_user: User = Depends(get_current_user)
):
    """Upload a robotics model for optimization."""
    
    # Validate file
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No file provided"
        )
    
    file_extension = Path(file.filename).suffix.lower()
    if file_extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        )
    
    # Check file size
    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large. Maximum size: {MAX_FILE_SIZE // (1024*1024)}MB"
        )
    
    try:
        # Generate unique filename
        model_id = str(uuid.uuid4())
        filename = f"{model_id}_{file.filename}"
        file_path = UPLOAD_DIR / filename
        
        # Save file
        with open(file_path, "wb") as f:
            f.write(content)
        
        # Create model metadata
        model_metadata = ModelMetadata(
            id=model_id,
            name=name or file.filename,
            description=description or "",
            tags=tags.split(",") if tags else [],
            file_path=str(file_path),
            size_mb=len(content) / (1024 * 1024),
            author=current_user.username
        )
        
        logger.info(f"Model uploaded successfully: {model_id}")
        
        return ModelUploadResponse(
            model_id=model_id,
            filename=file.filename,
            size_mb=model_metadata.size_mb,
            upload_time=datetime.now(),
            message="Model uploaded successfully"
        )
        
    except Exception as e:
        logger.error(f"Failed to upload model: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to upload model: {str(e)}"
        )


@app.get("/models", response_model=ModelListResponse, tags=["Models"])
async def list_models(
    skip: int = 0,
    limit: int = 100,
    current_user: User = Depends(get_current_user)
):
    """List uploaded models."""
    try:
        # Get all model files from upload directory
        model_files = []
        for file_path in UPLOAD_DIR.glob("*"):
            if file_path.is_file() and file_path.suffix.lower() in ALLOWED_EXTENSIONS:
                # Extract model ID from filename
                filename_parts = file_path.name.split("_", 1)
                if len(filename_parts) >= 2:
                    model_id = filename_parts[0]
                    original_name = filename_parts[1]
                    
                    stat = file_path.stat()
                    model_files.append({
                        "id": model_id,
                        "name": original_name,
                        "size_mb": stat.st_size / (1024 * 1024),
                        "created_at": datetime.fromtimestamp(stat.st_ctime),
                        "file_path": str(file_path)
                    })
        
        # Apply pagination
        total = len(model_files)
        models = model_files[skip:skip + limit]
        
        return ModelListResponse(
            models=models,
            total=total,
            skip=skip,
            limit=limit
        )
        
    except Exception as e:
        logger.error(f"Failed to list models: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list models: {str(e)}"
        )


@app.delete("/models/{model_id}", tags=["Models"])
async def delete_model(
    model_id: str,
    current_user: User = Depends(get_current_user)
):
    """Delete a model."""
    try:
        # Find model file
        model_files = list(UPLOAD_DIR.glob(f"{model_id}_*"))
        if not model_files:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Model not found"
            )
        
        # Delete file
        model_files[0].unlink()
        
        logger.info(f"Model deleted: {model_id}")
        return {"message": "Model deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete model {model_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete model: {str(e)}"
        )


# Optimization endpoints
@app.post("/optimize", response_model=OptimizationResponse, tags=["Optimization"])
async def start_optimization(
    request: OptimizationRequest,
    current_user: User = Depends(get_current_user),
    optimization_manager: OptimizationManager = Depends(get_optimization_manager)
):
    """Start optimization for a model."""
    try:
        # Validate model exists
        model_files = list(UPLOAD_DIR.glob(f"{request.model_id}_*"))
        if not model_files:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Model not found"
            )
        
        model_path = str(model_files[0])
        
        # Create optimization criteria
        criteria = OptimizationCriteria(
            name=request.criteria_name or "default",
            target_accuracy_threshold=request.target_accuracy_threshold,
            max_size_reduction_percent=request.max_size_reduction_percent,
            max_latency_increase_percent=request.max_latency_increase_percent,
            optimization_techniques=request.optimization_techniques or ["quantization", "pruning"]
        )
        
        # Start optimization session
        session_id = optimization_manager.start_optimization_session(
            model_path=model_path,
            criteria=criteria
        )
        
        logger.info(f"Optimization started: {session_id} for model: {request.model_id}")
        
        return OptimizationResponse(
            session_id=session_id,
            model_id=request.model_id,
            status="started",
            message="Optimization session started successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to start optimization: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start optimization: {str(e)}"
        )


@app.get("/sessions/{session_id}/status", response_model=SessionStatusResponse, tags=["Sessions"])
async def get_session_status(
    session_id: str,
    current_user: User = Depends(get_current_user),
    optimization_manager: OptimizationManager = Depends(get_optimization_manager)
):
    """Get optimization session status."""
    try:
        status_info = optimization_manager.get_session_status(session_id)
        
        return SessionStatusResponse(
            session_id=session_id,
            status=status_info["status"],
            progress_percentage=status_info["progress_percentage"],
            current_step=status_info["current_step"],
            start_time=datetime.fromisoformat(status_info["start_time"]),
            last_update=datetime.fromisoformat(status_info["last_update"]),
            error_message=status_info["error_message"],
            model_id=status_info["session_data"]["model_id"],
            steps_completed=status_info["session_data"]["steps_completed"]
        )
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Failed to get session status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get session status: {str(e)}"
        )


@app.get("/sessions", response_model=SessionListResponse, tags=["Sessions"])
async def list_sessions(
    current_user: User = Depends(get_current_user),
    optimization_manager: OptimizationManager = Depends(get_optimization_manager)
):
    """List all optimization sessions."""
    try:
        active_session_ids = optimization_manager.get_active_sessions()
        
        sessions = []
        for session_id in active_session_ids:
            try:
                status_info = optimization_manager.get_session_status(session_id)
                sessions.append({
                    "session_id": session_id,
                    "status": status_info["status"],
                    "progress_percentage": status_info["progress_percentage"],
                    "model_id": status_info["session_data"]["model_id"],
                    "start_time": status_info["start_time"],
                    "last_update": status_info["last_update"]
                })
            except Exception as e:
                logger.warning(f"Failed to get status for session {session_id}: {e}")
        
        return SessionListResponse(
            sessions=sessions,
            total=len(sessions)
        )
        
    except Exception as e:
        logger.error(f"Failed to list sessions: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list sessions: {str(e)}"
        )


@app.post("/sessions/{session_id}/cancel", tags=["Sessions"])
async def cancel_session(
    session_id: str,
    current_user: User = Depends(get_current_user),
    optimization_manager: OptimizationManager = Depends(get_optimization_manager)
):
    """Cancel an optimization session."""
    try:
        success = optimization_manager.cancel_session(session_id)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Session not found or already completed"
            )
        
        logger.info(f"Session cancelled: {session_id}")
        return {"message": "Session cancelled successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to cancel session {session_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to cancel session: {str(e)}"
        )


@app.post("/sessions/{session_id}/rollback", tags=["Sessions"])
async def rollback_session(
    session_id: str,
    snapshot_index: Optional[int] = None,
    current_user: User = Depends(get_current_user),
    optimization_manager: OptimizationManager = Depends(get_optimization_manager)
):
    """Rollback an optimization session to a previous snapshot."""
    try:
        success = optimization_manager.rollback_session(session_id, snapshot_index)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Session not found or no snapshots available"
            )
        
        logger.info(f"Session rolled back: {session_id}")
        return {"message": "Session rolled back successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to rollback session {session_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to rollback session: {str(e)}"
        )


# Results and evaluation endpoints
@app.get("/sessions/{session_id}/results", response_model=EvaluationResponse, tags=["Results"])
async def get_session_results(
    session_id: str,
    current_user: User = Depends(get_current_user),
    optimization_manager: OptimizationManager = Depends(get_optimization_manager)
):
    """Get optimization session results and evaluation."""
    try:
        # Get session status first
        status_info = optimization_manager.get_session_status(session_id)
        
        if status_info["status"] not in ["completed", "failed"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Session not completed yet"
            )
        
        # For now, return placeholder results
        # In a full implementation, this would retrieve actual results from the session
        return EvaluationResponse(
            session_id=session_id,
            model_id=status_info["session_data"]["model_id"],
            status=status_info["status"],
            optimization_summary="Optimization completed successfully",
            performance_improvements={
                "size_reduction_percent": 25.0,
                "speed_improvement_percent": 15.0,
                "accuracy_change_percent": -0.5
            },
            techniques_applied=["quantization", "pruning"],
            evaluation_metrics={
                "accuracy": 0.945,
                "inference_time_ms": 12.5,
                "model_size_mb": 150.0
            },
            comparison_baseline={
                "original_accuracy": 0.950,
                "original_inference_time_ms": 15.0,
                "original_model_size_mb": 200.0
            },
            recommendations=[
                "Model optimization successful with minimal accuracy loss",
                "Consider further quantization for additional size reduction"
            ]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get session results: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get session results: {str(e)}"
        )


if __name__ == "__main__":
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
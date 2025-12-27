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
import socketio

from .models import (
    ModelUploadResponse, OptimizationRequest, OptimizationResponse,
    SessionStatusResponse, SessionListResponse, ModelListResponse,
    EvaluationResponse, ErrorResponse, HealthResponse, User, DashboardStats
)
from .auth import AuthManager, get_auth_manager
from .dependencies import get_current_user, get_optimization_manager
from .monitoring import router as monitoring_router
from .openapi_config import get_openapi_config
from .error_handlers import register_error_handlers
from ..services.optimization_manager import OptimizationManager
from ..models.core import ModelMetadata, OptimizationSession
from ..config.optimization_criteria import OptimizationCriteria


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan with comprehensive startup validation."""
    # Startup
    logger.info("=" * 80)
    logger.info("Starting Robotics Model Optimization Platform API")
    logger.info("=" * 80)
    
    startup_errors = []
    startup_warnings = []
    
    # Check if platform integrator is already set (from main.py)
    if not hasattr(app.state, 'platform_integrator'):
        # Initialize platform integrator for standalone API mode
        logger.info("Initializing PlatformIntegrator...")
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
        
        try:
            platform_integrator = PlatformIntegrator(config)
            success = await platform_integrator.initialize_platform()
            
            if not success:
                error_msg = "Failed to initialize PlatformIntegrator"
                logger.error(error_msg)
                startup_errors.append(error_msg)
                raise RuntimeError(error_msg)
            
            app.state.platform_integrator = platform_integrator
            app.state.optimization_manager = platform_integrator.get_optimization_manager()
            logger.info("✓ PlatformIntegrator initialized successfully")
        except Exception as e:
            error_msg = f"Failed to initialize PlatformIntegrator: {e}"
            logger.error(error_msg, exc_info=True)
            startup_errors.append(error_msg)
            raise
    else:
        logger.info("✓ PlatformIntegrator already initialized")
    
    # Validate required services are available
    logger.info("Validating required services...")
    
    # Validate OptimizationManager
    if hasattr(app.state, 'optimization_manager') and app.state.optimization_manager:
        logger.info("✓ OptimizationManager available")
    else:
        warning_msg = "OptimizationManager not available"
        logger.warning(warning_msg)
        startup_warnings.append(warning_msg)
    
    # Validate ModelStore
    try:
        if hasattr(app.state, 'platform_integrator'):
            model_store = app.state.platform_integrator.get_model_store()
            if model_store:
                logger.info("✓ ModelStore available")
            else:
                warning_msg = "ModelStore not available"
                logger.warning(warning_msg)
                startup_warnings.append(warning_msg)
    except Exception as e:
        warning_msg = f"Failed to validate ModelStore: {e}"
        logger.warning(warning_msg)
        startup_warnings.append(warning_msg)
    
    # Validate MemoryManager
    try:
        if hasattr(app.state, 'platform_integrator'):
            memory_manager = app.state.platform_integrator.get_memory_manager()
            if memory_manager:
                logger.info("✓ MemoryManager available")
            else:
                warning_msg = "MemoryManager not available"
                logger.warning(warning_msg)
                startup_warnings.append(warning_msg)
    except Exception as e:
        warning_msg = f"Failed to validate MemoryManager: {e}"
        logger.warning(warning_msg)
        startup_warnings.append(warning_msg)
    
    # Validate NotificationService
    try:
        if hasattr(app.state, 'platform_integrator'):
            notification_service = app.state.platform_integrator.get_notification_service()
            if notification_service:
                logger.info("✓ NotificationService available")
            else:
                warning_msg = "NotificationService not available"
                logger.warning(warning_msg)
                startup_warnings.append(warning_msg)
    except Exception as e:
        warning_msg = f"Failed to validate NotificationService: {e}"
        logger.warning(warning_msg)
        startup_warnings.append(warning_msg)
    
    # Initialize ConfigurationManager
    logger.info("Initializing ConfigurationManager...")
    try:
        from ..services.config_manager import ConfigurationManager
        config_manager = ConfigurationManager()
        app.state.config_manager = config_manager
        
        # Load current configuration
        current_config = config_manager.get_current_configuration()
        logger.info(f"✓ ConfigurationManager initialized with config: {current_config.name}")
    except Exception as e:
        warning_msg = f"Failed to initialize ConfigurationManager: {e}"
        logger.warning(warning_msg)
        startup_warnings.append(warning_msg)
    
    # Initialize WebSocket manager
    logger.info("Initializing WebSocketManager...")
    try:
        from ..services.websocket_manager import WebSocketManager
        websocket_manager = WebSocketManager()
        app.state.websocket_manager = websocket_manager
        logger.info("✓ WebSocketManager initialized")
        
        # Connect WebSocket manager to notification service
        if hasattr(app.state, 'platform_integrator'):
            notification_service = app.state.platform_integrator.get_notification_service()
            if notification_service:
                websocket_manager.setup_notification_handlers(notification_service)
                logger.info("✓ WebSocketManager connected to NotificationService")
            else:
                warning_msg = "NotificationService not available for WebSocket integration"
                logger.warning(warning_msg)
                startup_warnings.append(warning_msg)
    except Exception as e:
        error_msg = f"Failed to initialize WebSocketManager: {e}"
        logger.error(error_msg, exc_info=True)
        startup_errors.append(error_msg)
    
    # Validate upload directory
    logger.info("Validating upload directory...")
    if UPLOAD_DIR.exists():
        logger.info(f"✓ Upload directory exists: {UPLOAD_DIR}")
    else:
        logger.info(f"Creating upload directory: {UPLOAD_DIR}")
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        logger.info("✓ Upload directory created")
    
    # Log startup summary
    logger.info("=" * 80)
    logger.info("Startup Summary:")
    logger.info(f"  Errors: {len(startup_errors)}")
    logger.info(f"  Warnings: {len(startup_warnings)}")
    
    if startup_errors:
        logger.error("Startup errors:")
        for error in startup_errors:
            logger.error(f"  - {error}")
    
    if startup_warnings:
        logger.warning("Startup warnings:")
        for warning in startup_warnings:
            logger.warning(f"  - {warning}")
    
    if not startup_errors:
        logger.info("✓ API startup completed successfully")
    else:
        logger.error("✗ API startup completed with errors")
    
    logger.info("=" * 80)
    
    yield
    
    # Shutdown
    logger.info("=" * 80)
    logger.info("Shutting down Robotics Model Optimization Platform API")
    logger.info("=" * 80)
    
    # Cleanup WebSocket connections
    if hasattr(app.state, 'websocket_manager'):
        try:
            ws_stats = app.state.websocket_manager.get_stats()
            logger.info(f"WebSocket connections at shutdown: {ws_stats['total_connections']}")
        except Exception as e:
            logger.warning(f"Failed to get WebSocket stats: {e}")
    
    # Shutdown platform integrator
    if hasattr(app.state, 'platform_integrator'):
        try:
            await app.state.platform_integrator.shutdown_platform()
            logger.info("✓ PlatformIntegrator shutdown complete")
        except Exception as e:
            logger.error(f"Error during platform shutdown: {e}", exc_info=True)
    elif hasattr(app.state, 'optimization_manager'):
        try:
            app.state.optimization_manager.cleanup()
            logger.info("✓ OptimizationManager cleanup complete")
        except Exception as e:
            logger.error(f"Error during optimization manager cleanup: {e}", exc_info=True)
    
    logger.info("✓ API shutdown completed")
    logger.info("=" * 80)


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

# ============================================================================
# Configure CORS (Cross-Origin Resource Sharing)
# ============================================================================
# CORS configuration allows the frontend to make requests from different origins
# and enables WebSocket upgrade requests for Socket.IO connections.
#
# Configuration:
# - Set CORS_ORIGINS environment variable with comma-separated origins
# - Example: CORS_ORIGINS="http://localhost:3000,https://app.example.com"
# - Default: "*" (all origins - not recommended for production)
#
# WebSocket Support:
# - WebSocket upgrade requests are automatically handled by the middleware
# - Socket.IO connections use the same CORS policy as HTTP requests
# - Credentials are allowed for authenticated WebSocket connections

# Get allowed origins from environment or use defaults
allowed_origins = os.getenv("CORS_ORIGINS", "*").split(",")
if allowed_origins == ["*"]:
    logger.warning("CORS configured to allow all origins. Configure CORS_ORIGINS environment variable for production.")

# Add monitoring and logging middleware
from .middleware import RequestLoggingMiddleware, PerformanceMonitoringMiddleware

# Add performance monitoring middleware (first, so it wraps everything)
app.add_middleware(
    PerformanceMonitoringMiddleware,
    slow_request_threshold_ms=1000.0  # Log requests slower than 1 second
)

# Add request logging middleware
app.add_middleware(RequestLoggingMiddleware)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,  # Required for authenticated requests and WebSocket
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
    allow_headers=["*"],  # Allow all headers including WebSocket upgrade headers
    expose_headers=["X-Request-ID", "X-Response-Time"],  # Expose timing headers
    max_age=3600,  # Cache preflight requests for 1 hour
)

# Create Socket.IO ASGI app
# Note: WebSocketManager is initialized in lifespan, we'll mount it after app creation
sio_asgi_app = None

# Security
security = HTTPBearer()

# Register error handlers
register_error_handlers(app)

# ============================================================================
# Register API Routers
# ============================================================================
# All routers are registered here to make them available through the FastAPI app

# Monitoring endpoints (health checks, metrics)
app.include_router(monitoring_router)

# Dashboard endpoints (statistics and aggregate metrics)
from .dashboard import router as dashboard_router
app.include_router(dashboard_router)

# Session endpoints (list and filter optimization sessions)
from .sessions import router as sessions_router
app.include_router(sessions_router)

# Configuration endpoints (optimization criteria management)
from .config import router as config_router
app.include_router(config_router)

# LLM service endpoints (validation and recommendations)
from .llm_endpoints import router as llm_router
app.include_router(llm_router)

logger.info("All API routers registered successfully")

# Global configuration
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

MAX_FILE_SIZE = 500 * 1024 * 1024  # 500MB
ALLOWED_EXTENSIONS = {".pt", ".pth", ".onnx", ".pb", ".h5", ".safetensors"}


# Health check endpoint
@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Check API health status."""
    from ..services.llm_service import llm_service
    
    # Check LLM service health
    llm_health = None
    try:
        llm_health = await llm_service.health_check()
    except Exception as e:
        logger.warning(f"LLM health check failed: {e}")
        llm_health = {"status": "error", "error": str(e)}
    
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(),
        version="1.0.0",
        services={
            "optimization_manager": hasattr(app.state, 'optimization_manager'),
            "upload_directory": UPLOAD_DIR.exists(),
            "llm_service": llm_health.get("status") == "healthy" if llm_health else False
        }
    )


# Performance metrics endpoint
@app.get("/metrics", tags=["Monitoring"])
async def get_metrics():
    """
    Get API performance metrics.
    
    Returns metrics about:
    - Request performance (average duration, slow requests)
    - WebSocket connections
    - Cache statistics
    
    Requires authentication.
    """
    from .middleware import PerformanceMonitoringMiddleware, WebSocketMetricsMiddleware
    from ..services.cache_service import CacheService
    
    metrics = {
        "timestamp": datetime.now().isoformat(),
        "api_performance": {},
        "websocket": {},
        "cache": {}
    }
    
    # Get API performance metrics from middleware
    for middleware in app.user_middleware:
        if isinstance(middleware.cls, type) and issubclass(middleware.cls, PerformanceMonitoringMiddleware):
            # Access the middleware instance
            # Note: This is a simplified approach; in production, you'd want to store
            # the middleware instance in app.state during initialization
            pass
    
    # Get cache statistics
    try:
        cache_service = CacheService()
        metrics["cache"] = cache_service.get_statistics()
    except Exception as e:
        logger.warning(f"Failed to get cache statistics: {e}")
        metrics["cache"] = {"error": str(e)}
    
    # Get WebSocket metrics if available
    if hasattr(app.state, 'websocket_metrics'):
        try:
            metrics["websocket"] = app.state.websocket_metrics.get_metrics()
        except Exception as e:
            logger.warning(f"Failed to get WebSocket metrics: {e}")
            metrics["websocket"] = {"error": str(e)}
    
    return metrics


# Authentication endpoints
@app.post("/auth/login", tags=["Authentication"])
async def login(
    credentials: dict,
    auth_manager: AuthManager = Depends(get_auth_manager)
):
    """Authenticate user and return access token."""
    username = credentials.get("username")
    password = credentials.get("password")
    
    if not username or not password:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username and password required"
        )
    
    # Authenticate user with AuthManager
    user = auth_manager.authenticate_user(username, password)
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials"
        )
    
    # Create JWT token
    token = auth_manager.create_access_token(user)
    
    return {
        "access_token": token,
        "token_type": "bearer",
        "expires_in": 3600,
        "user": {
            "id": user.id,
            "username": user.username,
            "role": user.role
        }
    }


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
        
        # Create optimization criteria with new API
        from src.config.optimization_criteria import OptimizationConstraints, OptimizationTechnique
        
        # Convert technique strings to enum
        techniques = []
        for tech in (request.optimization_techniques or ["quantization", "pruning"]):
            try:
                techniques.append(OptimizationTechnique(tech.lower()))
            except ValueError:
                pass  # Skip invalid techniques
        
        constraints = OptimizationConstraints(
            preserve_accuracy_threshold=request.target_accuracy_threshold or 0.95,
            allowed_techniques=techniques if techniques else [OptimizationTechnique.QUANTIZATION, OptimizationTechnique.PRUNING]
        )
        
        criteria = OptimizationCriteria(
            name=request.criteria_name or "default",
            description=f"API optimization request for model {request.model_id}",
            constraints=constraints,
            target_deployment="general"
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
                    "model_name": status_info["session_data"].get("model_name", f"Model {session_id}"),
                    "techniques": status_info["session_data"].get("techniques", []),
                    "size_reduction_percent": status_info["session_data"].get("size_reduction_percent"),
                    "speed_improvement_percent": status_info["session_data"].get("speed_improvement_percent"),
                    "start_time": status_info["start_time"],
                    "last_update": status_info["last_update"],
                    "created_at": status_info["session_data"].get("created_at", status_info["start_time"]),
                    "updated_at": status_info["session_data"].get("updated_at", status_info["last_update"]),
                    "completed_at": status_info["session_data"].get("completed_at")
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


def create_app_with_socketio():
    """
    Create combined ASGI application with FastAPI and Socket.IO.
    
    This function wraps the FastAPI application with Socket.IO support,
    enabling real-time WebSocket communication for optimization progress
    updates and notifications.
    
    The Socket.IO server is mounted at the '/socket.io' path and handles:
    - Client connections with authentication
    - Session-based subscriptions
    - Real-time progress updates
    - System notifications and alerts
    
    Returns:
        Combined ASGI application with FastAPI and Socket.IO
    """
    from ..services.websocket_manager import WebSocketManager
    
    logger.info("Creating combined FastAPI + Socket.IO application")
    
    # Get WebSocket manager instance (singleton)
    ws_manager = WebSocketManager()
    
    # Create Socket.IO ASGI app that wraps FastAPI
    # This allows both HTTP and WebSocket traffic on the same port
    sio_asgi_app = socketio.ASGIApp(
        ws_manager.sio,
        app,
        socketio_path='/socket.io'
    )
    
    logger.info("✓ Socket.IO mounted at /socket.io")
    logger.info("✓ Combined ASGI application created")
    
    return sio_asgi_app


if __name__ == "__main__":
    # Create combined app with Socket.IO
    combined_app = create_app_with_socketio()
    
    uvicorn.run(
        combined_app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
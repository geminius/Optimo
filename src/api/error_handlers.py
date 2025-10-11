"""
Error handler middleware for consistent error responses across the API.
"""

import logging
import uuid
from typing import Union
from datetime import datetime

from fastapi import Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from .models import ErrorResponse
from ..utils.exceptions import (
    PlatformError,
    ModelLoadingError,
    OptimizationError,
    EvaluationError,
    ValidationError,
    ConfigurationError,
    SystemError,
    NetworkError,
    StorageError,
    ErrorSeverity
)


logger = logging.getLogger(__name__)


def create_error_response(
    error: str,
    message: str,
    details: Union[dict, str, None] = None,
    request_id: str = None
) -> ErrorResponse:
    """
    Create a standardized error response.
    
    Args:
        error: Error type/code
        message: Human-readable error message
        details: Additional error details (dict or string)
        request_id: Request ID for tracking
    
    Returns:
        ErrorResponse model
    """
    if request_id is None:
        request_id = str(uuid.uuid4())
    
    # Convert string details to dict
    if isinstance(details, str):
        details = {"detail": details}
    
    return ErrorResponse(
        error=error,
        message=message,
        details=details,
        timestamp=datetime.now(),
        request_id=request_id
    )


async def platform_error_handler(request: Request, exc: PlatformError) -> JSONResponse:
    """
    Handle custom platform errors.
    
    Args:
        request: FastAPI request object
        exc: Platform error exception
    
    Returns:
        JSON response with error details
    """
    request_id = str(uuid.uuid4())
    
    # Log the error with context
    logger.error(
        f"Platform error: {exc.message}",
        extra={
            "error_code": exc.error_code,
            "category": exc.category.value,
            "severity": exc.severity.value,
            "request_id": request_id,
            "context": exc.context,
            "path": request.url.path
        }
    )
    
    # Determine HTTP status code based on error type and severity
    status_code = _get_status_code_for_platform_error(exc)
    
    # Create error response
    error_response = create_error_response(
        error=exc.error_code,
        message=exc.message,
        details=exc.context if exc.context else None,
        request_id=request_id
    )
    
    return JSONResponse(
        status_code=status_code,
        content=error_response.model_dump(mode='json')
    )


async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    """
    Handle HTTP exceptions.
    
    Args:
        request: FastAPI request object
        exc: HTTP exception
    
    Returns:
        JSON response with error details
    """
    request_id = str(uuid.uuid4())
    
    logger.warning(
        f"HTTP exception: {exc.detail}",
        extra={
            "status_code": exc.status_code,
            "request_id": request_id,
            "path": request.url.path
        }
    )
    
    error_response = create_error_response(
        error=f"http_{exc.status_code}",
        message=str(exc.detail),
        request_id=request_id
    )
    
    return JSONResponse(
        status_code=exc.status_code,
        content=error_response.model_dump(mode='json')
    )


async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """
    Handle request validation errors.
    
    Args:
        request: FastAPI request object
        exc: Validation error exception
    
    Returns:
        JSON response with validation error details
    """
    request_id = str(uuid.uuid4())
    
    # Extract validation errors
    validation_errors = []
    for error in exc.errors():
        validation_errors.append({
            "field": ".".join(str(loc) for loc in error["loc"]),
            "message": error["msg"],
            "type": error["type"]
        })
    
    logger.warning(
        f"Validation error: {len(validation_errors)} field(s) failed validation",
        extra={
            "request_id": request_id,
            "path": request.url.path,
            "errors": validation_errors
        }
    )
    
    error_response = create_error_response(
        error="validation_error",
        message="Request validation failed",
        details={"validation_errors": validation_errors},
        request_id=request_id
    )
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=error_response.model_dump(mode='json')
    )


async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Handle unexpected exceptions.
    
    Args:
        request: FastAPI request object
        exc: Exception
    
    Returns:
        JSON response with error details
    """
    request_id = str(uuid.uuid4())
    
    logger.error(
        f"Unhandled exception: {str(exc)}",
        extra={
            "request_id": request_id,
            "path": request.url.path,
            "exception_type": type(exc).__name__
        },
        exc_info=True
    )
    
    # Don't expose internal error details in production
    error_response = create_error_response(
        error="internal_server_error",
        message="An unexpected error occurred",
        details={"exception_type": type(exc).__name__},
        request_id=request_id
    )
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=error_response.model_dump(mode='json')
    )


def _get_status_code_for_platform_error(exc: PlatformError) -> int:
    """
    Determine appropriate HTTP status code for platform error.
    
    Args:
        exc: Platform error exception
    
    Returns:
        HTTP status code
    """
    # Map error types to status codes
    if isinstance(exc, ValidationError):
        return status.HTTP_400_BAD_REQUEST
    elif isinstance(exc, ConfigurationError):
        return status.HTTP_400_BAD_REQUEST
    elif isinstance(exc, ModelLoadingError):
        return status.HTTP_400_BAD_REQUEST
    elif isinstance(exc, OptimizationError):
        if exc.severity == ErrorSeverity.CRITICAL:
            return status.HTTP_500_INTERNAL_SERVER_ERROR
        return status.HTTP_422_UNPROCESSABLE_ENTITY
    elif isinstance(exc, EvaluationError):
        return status.HTTP_422_UNPROCESSABLE_ENTITY
    elif isinstance(exc, SystemError):
        return status.HTTP_503_SERVICE_UNAVAILABLE
    elif isinstance(exc, NetworkError):
        return status.HTTP_502_BAD_GATEWAY
    elif isinstance(exc, StorageError):
        return status.HTTP_500_INTERNAL_SERVER_ERROR
    else:
        # Default to 500 for unknown platform errors
        return status.HTTP_500_INTERNAL_SERVER_ERROR


def register_error_handlers(app):
    """
    Register all error handlers with the FastAPI application.
    
    Args:
        app: FastAPI application instance
    """
    # Register custom platform error handlers
    app.add_exception_handler(PlatformError, platform_error_handler)
    app.add_exception_handler(ModelLoadingError, platform_error_handler)
    app.add_exception_handler(OptimizationError, platform_error_handler)
    app.add_exception_handler(EvaluationError, platform_error_handler)
    app.add_exception_handler(ValidationError, platform_error_handler)
    app.add_exception_handler(ConfigurationError, platform_error_handler)
    app.add_exception_handler(SystemError, platform_error_handler)
    app.add_exception_handler(NetworkError, platform_error_handler)
    app.add_exception_handler(StorageError, platform_error_handler)
    
    # Register standard HTTP exception handlers
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    
    # Register general exception handler as fallback
    app.add_exception_handler(Exception, general_exception_handler)
    
    logger.info("Error handlers registered successfully")

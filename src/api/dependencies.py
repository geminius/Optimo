"""
FastAPI dependencies for authentication and service injection.
"""

import uuid
import logging
from typing import Optional
from fastapi import Depends, HTTPException, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from .auth import AuthManager, User, get_auth_manager
from ..services.optimization_manager import OptimizationManager


logger = logging.getLogger(__name__)
security = HTTPBearer()


async def get_request_id(request: Request) -> str:
    """
    Generate or retrieve request ID for tracking.
    
    Args:
        request: FastAPI request object
        
    Returns:
        Request ID string
    """
    # Check if request ID already exists in headers
    request_id = request.headers.get("X-Request-ID")
    
    if not request_id:
        # Generate new request ID
        request_id = str(uuid.uuid4())
    
    # Store in request state for access in handlers
    request.state.request_id = request_id
    
    return request_id


async def get_current_user(
    request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(security),
    auth_manager: AuthManager = Depends(get_auth_manager),
    request_id: str = Depends(get_request_id)
) -> User:
    """
    Get current authenticated user.
    
    Args:
        request: FastAPI request object
        credentials: HTTP authorization credentials
        auth_manager: Authentication manager instance
        request_id: Request ID for tracking
        
    Returns:
        Authenticated user
        
    Raises:
        HTTPException: If authentication fails
    """
    token = credentials.credentials
    user = auth_manager.verify_token(token)
    
    if user is None:
        logger.warning(
            "Authentication failed",
            extra={
                "component": "Authentication",
                "request_id": request_id,
                "path": request.url.path
            }
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer", "X-Request-ID": request_id},
        )
    
    logger.info(
        f"User authenticated: {user.username}",
        extra={
            "component": "Authentication",
            "request_id": request_id,
            "user_id": user.id,
            "username": user.username,
            "role": user.role
        }
    )
    
    return user


async def get_admin_user(
    request: Request,
    current_user: User = Depends(get_current_user)
) -> User:
    """
    Get current user and verify admin role.
    
    Args:
        request: FastAPI request object
        current_user: Current authenticated user
        
    Returns:
        User with admin role
        
    Raises:
        HTTPException: If user is not an administrator
    """
    if current_user.role != "administrator":
        request_id = getattr(request.state, "request_id", "unknown")
        logger.warning(
            f"Admin access denied for user: {current_user.username}",
            extra={
                "component": "Authorization",
                "request_id": request_id,
                "user_id": current_user.id,
                "username": current_user.username,
                "role": current_user.role,
                "path": request.url.path
            }
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Administrator access required",
            headers={"X-Request-ID": request_id}
        )
    
    return current_user


async def get_optimization_manager(request: Request) -> OptimizationManager:
    """Get optimization manager from app state."""
    if not hasattr(request.app.state, 'optimization_manager'):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Optimization manager not available"
        )
    
    return request.app.state.optimization_manager


def check_permission(resource: str, action: str):
    """
    Dependency factory for permission checking.
    
    Args:
        resource: Resource being accessed
        action: Action being performed (read/write)
        
    Returns:
        Dependency function that checks permissions
    """
    def permission_checker(
        request: Request,
        current_user: User = Depends(get_current_user),
        auth_manager: AuthManager = Depends(get_auth_manager)
    ) -> User:
        """Check if user has permission for resource and action."""
        request_id = getattr(request.state, "request_id", "unknown")
        
        if not auth_manager.has_permission(current_user, resource, action):
            logger.warning(
                f"Permission denied: {action} on {resource}",
                extra={
                    "component": "Authorization",
                    "request_id": request_id,
                    "user_id": current_user.id,
                    "username": current_user.username,
                    "role": current_user.role,
                    "resource": resource,
                    "action": action,
                    "path": request.url.path
                }
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient permissions for {action} on {resource}",
                headers={"X-Request-ID": request_id}
            )
        
        logger.info(
            f"Permission granted: {action} on {resource}",
            extra={
                "component": "Authorization",
                "request_id": request_id,
                "user_id": current_user.id,
                "username": current_user.username,
                "resource": resource,
                "action": action
            }
        )
        return current_user
    
    return permission_checker


# Common permission dependencies
read_models = check_permission("models", "read")
write_models = check_permission("models", "write")
read_sessions = check_permission("sessions", "read")
write_sessions = check_permission("sessions", "write")
read_results = check_permission("results", "read")
optimize_permission = check_permission("optimize", "write")
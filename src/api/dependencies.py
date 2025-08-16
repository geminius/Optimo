"""
FastAPI dependencies for authentication and service injection.
"""

from typing import Optional
from fastapi import Depends, HTTPException, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from .auth import AuthManager, User, get_auth_manager
from ..services.optimization_manager import OptimizationManager


security = HTTPBearer()


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    auth_manager: AuthManager = Depends(get_auth_manager)
) -> User:
    """Get current authenticated user."""
    token = credentials.credentials
    user = auth_manager.verify_token(token)
    
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return user


async def get_admin_user(
    current_user: User = Depends(get_current_user)
) -> User:
    """Get current user and verify admin role."""
    if current_user.role != "administrator":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Administrator access required"
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
    """Dependency factory for permission checking."""
    def permission_checker(
        current_user: User = Depends(get_current_user),
        auth_manager: AuthManager = Depends(get_auth_manager)
    ) -> User:
        if not auth_manager.has_permission(current_user, resource, action):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient permissions for {action} on {resource}"
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
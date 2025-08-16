"""
Authentication and authorization module for the API.
"""

import logging
import jwt
import hashlib
from typing import Dict, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
from fastapi import HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials


logger = logging.getLogger(__name__)

# Configuration
SECRET_KEY = "your-secret-key-change-in-production"  # Change in production
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60


@dataclass
class User:
    """User data class."""
    id: str
    username: str
    role: str
    email: Optional[str] = None
    is_active: bool = True


class AuthManager:
    """Authentication and authorization manager."""
    
    def __init__(self):
        self.users_db: Dict[str, Dict] = {
            "admin": {
                "id": "admin",
                "username": "admin",
                "password_hash": self._hash_password("admin"),  # Change in production
                "role": "administrator",
                "email": "admin@example.com",
                "is_active": True
            },
            "user": {
                "id": "user",
                "username": "user",
                "password_hash": self._hash_password("user"),
                "role": "user",
                "email": "user@example.com",
                "is_active": True
            }
        }
        self.active_tokens: Dict[str, str] = {}  # token -> username mapping
    
    def _hash_password(self, password: str) -> str:
        """Hash password using SHA-256."""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def authenticate_user(self, username: str, password: str) -> Optional[User]:
        """Authenticate user with username and password."""
        user_data = self.users_db.get(username)
        if not user_data:
            return None
        
        password_hash = self._hash_password(password)
        if password_hash != user_data["password_hash"]:
            return None
        
        if not user_data["is_active"]:
            return None
        
        return User(
            id=user_data["id"],
            username=user_data["username"],
            role=user_data["role"],
            email=user_data.get("email"),
            is_active=user_data["is_active"]
        )
    
    def create_access_token(self, user: User) -> str:
        """Create JWT access token for user."""
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        to_encode = {
            "sub": user.username,
            "user_id": user.id,
            "role": user.role,
            "exp": expire
        }
        
        token = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
        self.active_tokens[token] = user.username
        
        return token
    
    def verify_token(self, token: str) -> Optional[User]:
        """Verify JWT token and return user."""
        try:
            # Check if token is in active tokens
            if token not in self.active_tokens:
                return None
            
            # Decode token
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            username = payload.get("sub")
            
            if username is None:
                return None
            
            # Get user data
            user_data = self.users_db.get(username)
            if not user_data or not user_data["is_active"]:
                return None
            
            return User(
                id=user_data["id"],
                username=user_data["username"],
                role=user_data["role"],
                email=user_data.get("email"),
                is_active=user_data["is_active"]
            )
            
        except jwt.PyJWTError:
            return None
    
    def revoke_token(self, token: str) -> bool:
        """Revoke access token."""
        if token in self.active_tokens:
            del self.active_tokens[token]
            return True
        return False
    
    def has_permission(self, user: User, resource: str, action: str) -> bool:
        """Check if user has permission for resource and action."""
        # Simple role-based permissions
        if user.role == "administrator":
            return True
        
        if user.role == "user":
            # Users can read their own data and perform basic operations
            read_permissions = ["models", "sessions", "results"]
            write_permissions = ["models", "optimize"]
            
            if action == "read" and resource in read_permissions:
                return True
            if action == "write" and resource in write_permissions:
                return True
        
        return False


# Global auth manager instance
auth_manager = AuthManager()


def get_auth_manager() -> AuthManager:
    """Get authentication manager instance."""
    return auth_manager
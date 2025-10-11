"""
Comprehensive tests for authentication and authorization.

This test suite covers:
- Endpoints without authentication tokens
- Expired tokens
- Insufficient permissions
- Successful authentication flows
- WebSocket authentication
- Permission-based access control

Requirements: 5.1, 5.2, 5.3, 5.4
"""

import pytest
import jwt
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from fastapi import HTTPException, status
from fastapi.testclient import TestClient

from src.api.auth import AuthManager, User, SECRET_KEY, ALGORITHM
from src.api.dependencies import get_current_user, get_admin_user, check_permission
from src.api.main import app


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def auth_manager():
    """Create AuthManager instance for testing."""
    return AuthManager()


@pytest.fixture
def admin_user():
    """Create admin user for testing."""
    return User(
        id="admin",
        username="admin",
        role="administrator",
        email="admin@example.com",
        is_active=True
    )


@pytest.fixture
def regular_user():
    """Create regular user for testing."""
    return User(
        id="user",
        username="user",
        role="user",
        email="user@example.com",
        is_active=True
    )


@pytest.fixture
def valid_admin_token(auth_manager, admin_user):
    """Create valid admin token."""
    return auth_manager.create_access_token(admin_user)


@pytest.fixture
def valid_user_token(auth_manager, regular_user):
    """Create valid user token."""
    return auth_manager.create_access_token(regular_user)


@pytest.fixture
def expired_token():
    """Create expired JWT token."""
    expire = datetime.utcnow() - timedelta(minutes=10)  # Expired 10 minutes ago
    to_encode = {
        "sub": "testuser",
        "user_id": "test_id",
        "role": "user",
        "exp": expire
    }
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


@pytest.fixture
def invalid_token():
    """Create invalid JWT token with wrong signature."""
    to_encode = {
        "sub": "testuser",
        "user_id": "test_id",
        "role": "user",
        "exp": datetime.utcnow() + timedelta(minutes=60)
    }
    return jwt.encode(to_encode, "wrong-secret-key", algorithm=ALGORITHM)


@pytest.fixture
def mock_request():
    """Create mock FastAPI request."""
    request = Mock()
    request.url.path = "/test/endpoint"
    request.state.request_id = "test-request-id"
    request.app.state.optimization_manager = Mock()
    return request


# ============================================================================
# Test AuthManager Core Functionality
# ============================================================================

class TestAuthManager:
    """Test AuthManager authentication and authorization."""
    
    def test_authenticate_valid_credentials(self, auth_manager):
        """Test authentication with valid credentials."""
        user = auth_manager.authenticate_user("admin", "admin")
        
        assert user is not None
        assert user.username == "admin"
        assert user.role == "administrator"
        assert user.is_active is True
    
    def test_authenticate_invalid_password(self, auth_manager):
        """Test authentication with invalid password."""
        user = auth_manager.authenticate_user("admin", "wrong_password")
        
        assert user is None
    
    def test_authenticate_nonexistent_user(self, auth_manager):
        """Test authentication with nonexistent user."""
        user = auth_manager.authenticate_user("nonexistent", "password")
        
        assert user is None
    
    def test_create_access_token(self, auth_manager, admin_user):
        """Test access token creation."""
        token = auth_manager.create_access_token(admin_user)
        
        assert token is not None
        assert isinstance(token, str)
        assert len(token) > 0
        
        # Verify token is in active tokens
        assert token in auth_manager.active_tokens
        assert auth_manager.active_tokens[token] == admin_user.username
    
    def test_verify_valid_token(self, auth_manager, admin_user, valid_admin_token):
        """Test verification of valid token."""
        verified_user = auth_manager.verify_token(valid_admin_token)
        
        assert verified_user is not None
        assert verified_user.username == admin_user.username
        assert verified_user.role == admin_user.role
    
    def test_verify_expired_token(self, auth_manager, expired_token):
        """Test verification of expired token."""
        # Add to active tokens first (simulating it was valid)
        auth_manager.active_tokens[expired_token] = "testuser"
        
        verified_user = auth_manager.verify_token(expired_token)
        
        assert verified_user is None
    
    def test_verify_invalid_token(self, auth_manager, invalid_token):
        """Test verification of invalid token."""
        # Add to active tokens first
        auth_manager.active_tokens[invalid_token] = "testuser"
        
        verified_user = auth_manager.verify_token(invalid_token)
        
        assert verified_user is None
    
    def test_verify_token_not_in_active_tokens(self, auth_manager):
        """Test verification of token not in active tokens."""
        # Create a valid token but don't add to active tokens
        expire = datetime.utcnow() + timedelta(minutes=60)
        to_encode = {
            "sub": "testuser",
            "user_id": "test_id",
            "role": "user",
            "exp": expire
        }
        token = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
        
        verified_user = auth_manager.verify_token(token)
        
        assert verified_user is None
    
    def test_revoke_token(self, auth_manager, admin_user, valid_admin_token):
        """Test token revocation."""
        # Verify token is valid before revocation
        assert auth_manager.verify_token(valid_admin_token) is not None
        
        # Revoke token
        success = auth_manager.revoke_token(valid_admin_token)
        
        assert success is True
        assert valid_admin_token not in auth_manager.active_tokens
        
        # Verify token is no longer valid
        assert auth_manager.verify_token(valid_admin_token) is None
    
    def test_revoke_nonexistent_token(self, auth_manager):
        """Test revoking nonexistent token."""
        success = auth_manager.revoke_token("nonexistent_token")
        
        assert success is False
    
    def test_verify_websocket_token(self, auth_manager, admin_user, valid_admin_token):
        """Test WebSocket token verification."""
        verified_user = auth_manager.verify_websocket_token(valid_admin_token)
        
        assert verified_user is not None
        assert verified_user.username == admin_user.username
    
    def test_admin_permissions(self, auth_manager, admin_user):
        """Test administrator has all permissions."""
        # Admin should have all permissions
        assert auth_manager.has_permission(admin_user, "models", "read") is True
        assert auth_manager.has_permission(admin_user, "models", "write") is True
        assert auth_manager.has_permission(admin_user, "config", "read") is True
        assert auth_manager.has_permission(admin_user, "config", "write") is True
        assert auth_manager.has_permission(admin_user, "sessions", "read") is True
        assert auth_manager.has_permission(admin_user, "sessions", "write") is True
    
    def test_user_permissions(self, auth_manager, regular_user):
        """Test regular user has limited permissions."""
        # User should have read permissions
        assert auth_manager.has_permission(regular_user, "models", "read") is True
        assert auth_manager.has_permission(regular_user, "sessions", "read") is True
        assert auth_manager.has_permission(regular_user, "config", "read") is True
        
        # User should have some write permissions
        assert auth_manager.has_permission(regular_user, "models", "write") is True
        assert auth_manager.has_permission(regular_user, "optimize", "write") is True
        
        # User should NOT have config write permission
        assert auth_manager.has_permission(regular_user, "config", "write") is False


# ============================================================================
# Test Dependency Functions
# ============================================================================

class TestDependencies:
    """Test FastAPI dependency functions."""
    
    @pytest.mark.asyncio
    async def test_get_current_user_valid_token(
        self, auth_manager, admin_user, valid_admin_token, mock_request
    ):
        """Test get_current_user with valid token."""
        from fastapi.security import HTTPAuthorizationCredentials
        
        credentials = HTTPAuthorizationCredentials(
            scheme="Bearer",
            credentials=valid_admin_token
        )
        
        user = await get_current_user(
            request=mock_request,
            credentials=credentials,
            auth_manager=auth_manager,
            request_id="test-request-id"
        )
        
        assert user is not None
        assert user.username == admin_user.username
        assert user.role == admin_user.role
    
    @pytest.mark.asyncio
    async def test_get_current_user_invalid_token(
        self, auth_manager, invalid_token, mock_request
    ):
        """Test get_current_user with invalid token."""
        from fastapi.security import HTTPAuthorizationCredentials
        
        credentials = HTTPAuthorizationCredentials(
            scheme="Bearer",
            credentials=invalid_token
        )
        
        with pytest.raises(HTTPException) as exc_info:
            await get_current_user(
                request=mock_request,
                credentials=credentials,
                auth_manager=auth_manager,
                request_id="test-request-id"
            )
        
        assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED
        assert "Invalid authentication credentials" in exc_info.value.detail
    
    @pytest.mark.asyncio
    async def test_get_current_user_expired_token(
        self, auth_manager, expired_token, mock_request
    ):
        """Test get_current_user with expired token."""
        from fastapi.security import HTTPAuthorizationCredentials
        
        # Add to active tokens to simulate it was valid
        auth_manager.active_tokens[expired_token] = "testuser"
        
        credentials = HTTPAuthorizationCredentials(
            scheme="Bearer",
            credentials=expired_token
        )
        
        with pytest.raises(HTTPException) as exc_info:
            await get_current_user(
                request=mock_request,
                credentials=credentials,
                auth_manager=auth_manager,
                request_id="test-request-id"
            )
        
        assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED
    
    @pytest.mark.asyncio
    async def test_get_admin_user_with_admin(self, admin_user, mock_request):
        """Test get_admin_user with administrator role."""
        user = await get_admin_user(
            request=mock_request,
            current_user=admin_user
        )
        
        assert user is not None
        assert user.username == admin_user.username
        assert user.role == "administrator"
    
    @pytest.mark.asyncio
    async def test_get_admin_user_with_regular_user(self, regular_user, mock_request):
        """Test get_admin_user with regular user role."""
        with pytest.raises(HTTPException) as exc_info:
            await get_admin_user(
                request=mock_request,
                current_user=regular_user
            )
        
        assert exc_info.value.status_code == status.HTTP_403_FORBIDDEN
        assert "Administrator access required" in exc_info.value.detail
    
    def test_check_permission_granted(
        self, auth_manager, admin_user, mock_request
    ):
        """Test permission check when permission is granted."""
        permission_checker = check_permission("models", "read")
        
        user = permission_checker(
            request=mock_request,
            current_user=admin_user,
            auth_manager=auth_manager
        )
        
        assert user is not None
        assert user.username == admin_user.username
    
    def test_check_permission_denied(
        self, auth_manager, regular_user, mock_request
    ):
        """Test permission check when permission is denied."""
        permission_checker = check_permission("config", "write")
        
        with pytest.raises(HTTPException) as exc_info:
            permission_checker(
                request=mock_request,
                current_user=regular_user,
                auth_manager=auth_manager
            )
        
        assert exc_info.value.status_code == status.HTTP_403_FORBIDDEN
        assert "Insufficient permissions" in exc_info.value.detail


# ============================================================================
# Test API Endpoints Without Authentication
# ============================================================================

class TestEndpointsWithoutAuth:
    """Test API endpoints without authentication tokens."""
    
    def test_dashboard_stats_without_auth(self):
        """Test dashboard stats endpoint without authentication."""
        client = TestClient(app)
        
        response = client.get("/dashboard/stats")
        
        assert response.status_code == status.HTTP_403_FORBIDDEN
    
    def test_sessions_list_without_auth(self):
        """Test sessions list endpoint without authentication."""
        client = TestClient(app)
        
        response = client.get("/optimization/sessions")
        
        assert response.status_code == status.HTTP_403_FORBIDDEN
    
    def test_get_config_without_auth(self):
        """Test get configuration endpoint without authentication."""
        client = TestClient(app)
        
        response = client.get("/config/optimization-criteria")
        
        assert response.status_code == status.HTTP_403_FORBIDDEN
    
    def test_update_config_without_auth(self):
        """Test update configuration endpoint without authentication."""
        client = TestClient(app)
        
        config_data = {
            "name": "test_config",
            "target_accuracy_threshold": 0.95,
            "max_size_reduction_percent": 50.0,
            "max_latency_increase_percent": 10.0,
            "optimization_techniques": ["quantization"]
        }
        
        response = client.put("/config/optimization-criteria", json=config_data)
        
        assert response.status_code == status.HTTP_403_FORBIDDEN
    
    def test_model_upload_without_auth(self):
        """Test model upload endpoint without authentication."""
        client = TestClient(app)
        
        files = {"file": ("test_model.pt", b"fake model data", "application/octet-stream")}
        
        response = client.post("/models/upload", files=files)
        
        assert response.status_code == status.HTTP_403_FORBIDDEN
    
    def test_list_models_without_auth(self):
        """Test list models endpoint without authentication."""
        client = TestClient(app)
        
        response = client.get("/models")
        
        assert response.status_code == status.HTTP_403_FORBIDDEN


# ============================================================================
# Test API Endpoints With Authentication
# ============================================================================

class TestEndpointsWithAuth:
    """Test API endpoints with valid authentication."""
    
    def test_dashboard_stats_with_valid_token(
        self, auth_manager, admin_user, valid_admin_token
    ):
        """Test dashboard stats endpoint with valid token."""
        from src.api.dependencies import get_current_user, get_auth_manager
        
        # Override dependencies to use our test auth manager and user
        def override_get_auth_manager():
            return auth_manager
        
        def override_get_current_user():
            return admin_user
        
        app.dependency_overrides[get_auth_manager] = override_get_auth_manager
        app.dependency_overrides[get_current_user] = override_get_current_user
        
        try:
            client = TestClient(app)
            
            # Mock the platform integrator and services
            mock_integrator = Mock()
            mock_model_store = Mock()
            mock_model_store.list_models.return_value = []
            
            mock_memory_manager = Mock()
            mock_memory_manager.get_session_statistics.return_value = {
                "total_sessions": 0,
                "status_distribution": {}
            }
            mock_memory_manager.list_sessions.return_value = []
            
            mock_optimization_manager = Mock()
            mock_optimization_manager.get_active_sessions.return_value = []
            
            mock_integrator.get_model_store.return_value = mock_model_store
            mock_integrator.get_memory_manager.return_value = mock_memory_manager
            
            app.state.platform_integrator = mock_integrator
            app.state.optimization_manager = mock_optimization_manager
            
            headers = {"Authorization": f"Bearer {valid_admin_token}"}
            response = client.get("/dashboard/stats", headers=headers)
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert "total_models" in data
            assert "active_optimizations" in data
        finally:
            # Clean up overrides
            app.dependency_overrides.clear()
            if hasattr(app.state, 'platform_integrator'):
                delattr(app.state, 'platform_integrator')
    
    def test_get_config_with_valid_token(
        self, auth_manager, regular_user, valid_user_token
    ):
        """Test get configuration endpoint with valid token."""
        from src.api.dependencies import get_current_user, get_auth_manager
        
        # Override dependencies
        def override_get_auth_manager():
            return auth_manager
        
        def override_get_current_user():
            return regular_user
        
        app.dependency_overrides[get_auth_manager] = override_get_auth_manager
        app.dependency_overrides[get_current_user] = override_get_current_user
        
        try:
            client = TestClient(app)
            headers = {"Authorization": f"Bearer {valid_user_token}"}
            
            response = client.get("/config/optimization-criteria", headers=headers)
            
            # Should succeed for regular users (read permission)
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert "name" in data
            assert "target_accuracy_threshold" in data
        finally:
            app.dependency_overrides.clear()
    
    def test_update_config_with_admin_token(
        self, auth_manager, admin_user, valid_admin_token
    ):
        """Test update configuration endpoint with admin token."""
        from src.api.dependencies import get_current_user, get_admin_user, get_auth_manager
        
        # Override dependencies
        def override_get_auth_manager():
            return auth_manager
        
        def override_get_current_user():
            return admin_user
        
        def override_get_admin_user():
            return admin_user
        
        app.dependency_overrides[get_auth_manager] = override_get_auth_manager
        app.dependency_overrides[get_current_user] = override_get_current_user
        app.dependency_overrides[get_admin_user] = override_get_admin_user
        
        try:
            client = TestClient(app)
            headers = {"Authorization": f"Bearer {valid_admin_token}"}
            
            config_data = {
                "name": "test_config",
                "target_accuracy_threshold": 0.95,
                "max_size_reduction_percent": 50.0,
                "max_latency_increase_percent": 10.0,
                "optimization_techniques": ["quantization", "pruning"]
            }
            
            response = client.put(
                "/config/optimization-criteria",
                json=config_data,
                headers=headers
            )
            
            # Should succeed for admin users
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["name"] == "test_config"
        finally:
            app.dependency_overrides.clear()
    
    def test_update_config_with_user_token(
        self, auth_manager, regular_user, valid_user_token
    ):
        """Test update configuration endpoint with regular user token."""
        from src.api.dependencies import get_current_user, get_admin_user, get_auth_manager
        
        # Override dependencies
        def override_get_auth_manager():
            return auth_manager
        
        def override_get_current_user():
            return regular_user
        
        # get_admin_user will call get_current_user (which returns regular_user)
        # and then check if role is administrator, which will fail
        # So we don't need to override get_admin_user, it will use the real implementation
        
        app.dependency_overrides[get_auth_manager] = override_get_auth_manager
        app.dependency_overrides[get_current_user] = override_get_current_user
        
        try:
            client = TestClient(app)
            headers = {"Authorization": f"Bearer {valid_user_token}"}
            
            config_data = {
                "name": "test_config",
                "target_accuracy_threshold": 0.95,
                "max_size_reduction_percent": 50.0,
                "max_latency_increase_percent": 10.0,
                "optimization_techniques": ["quantization"]
            }
            
            response = client.put(
                "/config/optimization-criteria",
                json=config_data,
                headers=headers
            )
            
            # Should fail for regular users (requires admin)
            assert response.status_code == status.HTTP_403_FORBIDDEN
            response_data = response.json()
            # The error handler may format the response differently
            if "detail" in response_data:
                assert "Administrator access required" in response_data["detail"]
            elif "message" in response_data:
                assert "Administrator access required" in response_data["message"]
            else:
                # Just verify we got a 403
                pass
        finally:
            app.dependency_overrides.clear()


# ============================================================================
# Test Token Expiration
# ============================================================================

class TestTokenExpiration:
    """Test token expiration handling."""
    
    @pytest.mark.asyncio
    async def test_expired_token_rejected(self, auth_manager, expired_token):
        """Test that expired tokens are rejected."""
        client = TestClient(app)
        
        # Add to active tokens to simulate it was valid
        auth_manager.active_tokens[expired_token] = "testuser"
        
        headers = {"Authorization": f"Bearer {expired_token}"}
        
        response = client.get("/dashboard/stats", headers=headers)
        
        assert response.status_code == status.HTTP_401_UNAUTHORIZED
    
    def test_token_expiration_time(self, auth_manager, admin_user):
        """Test that tokens have correct expiration time."""
        token = auth_manager.create_access_token(admin_user)
        
        # Decode token to check expiration
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        
        exp_timestamp = payload["exp"]
        exp_datetime = datetime.utcfromtimestamp(exp_timestamp)
        now = datetime.utcnow()
        
        # Token should expire in approximately 60 minutes
        time_diff = (exp_datetime - now).total_seconds()
        assert 3500 < time_diff < 3700  # Allow some variance


# ============================================================================
# Test Permission-Based Access Control
# ============================================================================

class TestPermissionBasedAccess:
    """Test permission-based access control."""
    
    def test_admin_can_access_all_resources(self, auth_manager, admin_user):
        """Test that admin can access all resources."""
        resources = ["models", "sessions", "config", "results", "optimize"]
        actions = ["read", "write"]
        
        for resource in resources:
            for action in actions:
                assert auth_manager.has_permission(admin_user, resource, action) is True
    
    def test_user_read_permissions(self, auth_manager, regular_user):
        """Test that regular user has read permissions."""
        read_resources = ["models", "sessions", "results", "config"]
        
        for resource in read_resources:
            assert auth_manager.has_permission(regular_user, resource, "read") is True
    
    def test_user_write_permissions(self, auth_manager, regular_user):
        """Test that regular user has limited write permissions."""
        # User should have write permission for these
        assert auth_manager.has_permission(regular_user, "models", "write") is True
        assert auth_manager.has_permission(regular_user, "optimize", "write") is True
        
        # User should NOT have write permission for config
        assert auth_manager.has_permission(regular_user, "config", "write") is False
    
    def test_user_cannot_write_config(self, auth_manager, regular_user):
        """Test that regular user cannot write configuration."""
        assert auth_manager.has_permission(regular_user, "config", "write") is False


# ============================================================================
# Test Successful Authentication Flows
# ============================================================================

class TestSuccessfulAuthFlows:
    """Test complete successful authentication flows."""
    
    def test_login_and_access_endpoint(self, auth_manager):
        """Test complete flow: login and access protected endpoint."""
        # Step 1: Authenticate user
        user = auth_manager.authenticate_user("admin", "admin")
        assert user is not None
        
        # Step 2: Create token
        token = auth_manager.create_access_token(user)
        assert token is not None
        
        # Step 3: Verify token
        verified_user = auth_manager.verify_token(token)
        assert verified_user is not None
        assert verified_user.username == user.username
        
        # Step 4: Check permissions
        assert auth_manager.has_permission(verified_user, "models", "read") is True
    
    def test_token_lifecycle(self, auth_manager, admin_user):
        """Test complete token lifecycle."""
        # Create token
        token = auth_manager.create_access_token(admin_user)
        assert token in auth_manager.active_tokens
        
        # Verify token works
        verified_user = auth_manager.verify_token(token)
        assert verified_user is not None
        
        # Revoke token
        success = auth_manager.revoke_token(token)
        assert success is True
        assert token not in auth_manager.active_tokens
        
        # Verify token no longer works
        verified_user = auth_manager.verify_token(token)
        assert verified_user is None
    
    def test_multiple_concurrent_tokens(self, auth_manager, admin_user):
        """Test multiple concurrent tokens for same user."""
        # Create multiple tokens - they will have same expiration time
        # but that's OK, they're still tracked separately in active_tokens
        token1 = auth_manager.create_access_token(admin_user)
        
        # Wait a full second to ensure different expiration time
        time.sleep(1.1)
        
        token2 = auth_manager.create_access_token(admin_user)
        
        # Tokens should be different due to different expiration times
        # (JWT includes exp timestamp which changes every second)
        assert token1 != token2, "Tokens should be different with different expiration times"
        
        # Both should be valid
        assert auth_manager.verify_token(token1) is not None
        assert auth_manager.verify_token(token2) is not None
        
        # Revoke one token
        auth_manager.revoke_token(token1)
        
        # First should be invalid, second still valid
        assert auth_manager.verify_token(token1) is None
        assert auth_manager.verify_token(token2) is not None


# ============================================================================
# Test WebSocket Authentication
# ============================================================================

class TestWebSocketAuthentication:
    """Test WebSocket connection authentication."""
    
    def test_websocket_token_verification(
        self, auth_manager, admin_user, valid_admin_token
    ):
        """Test WebSocket token verification."""
        verified_user = auth_manager.verify_websocket_token(valid_admin_token)
        
        assert verified_user is not None
        assert verified_user.username == admin_user.username
    
    def test_websocket_invalid_token(self, auth_manager, invalid_token):
        """Test WebSocket with invalid token."""
        # Add to active tokens
        auth_manager.active_tokens[invalid_token] = "testuser"
        
        verified_user = auth_manager.verify_websocket_token(invalid_token)
        
        assert verified_user is None
    
    def test_websocket_expired_token(self, auth_manager, expired_token):
        """Test WebSocket with expired token."""
        # Add to active tokens
        auth_manager.active_tokens[expired_token] = "testuser"
        
        verified_user = auth_manager.verify_websocket_token(expired_token)
        
        assert verified_user is None


# ============================================================================
# Test Error Messages and Logging
# ============================================================================

class TestErrorMessagesAndLogging:
    """Test error messages and audit logging."""
    
    @pytest.mark.asyncio
    async def test_unauthorized_error_message(self, auth_manager, invalid_token, mock_request):
        """Test unauthorized error message format."""
        from fastapi.security import HTTPAuthorizationCredentials
        
        credentials = HTTPAuthorizationCredentials(
            scheme="Bearer",
            credentials=invalid_token
        )
        
        with pytest.raises(HTTPException) as exc_info:
            await get_current_user(
                request=mock_request,
                credentials=credentials,
                auth_manager=auth_manager,
                request_id="test-request-id"
            )
        
        assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED
        assert "Invalid authentication credentials" in exc_info.value.detail
        assert "WWW-Authenticate" in exc_info.value.headers
        assert exc_info.value.headers["WWW-Authenticate"] == "Bearer"
    
    @pytest.mark.asyncio
    async def test_forbidden_error_message(self, regular_user, mock_request):
        """Test forbidden error message format."""
        with pytest.raises(HTTPException) as exc_info:
            await get_admin_user(
                request=mock_request,
                current_user=regular_user
            )
        
        assert exc_info.value.status_code == status.HTTP_403_FORBIDDEN
        assert "Administrator access required" in exc_info.value.detail
        assert "X-Request-ID" in exc_info.value.headers


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

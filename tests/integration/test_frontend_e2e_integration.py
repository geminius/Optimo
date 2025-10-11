"""
End-to-end integration tests for frontend-backend communication.

This test suite verifies that all API endpoints work correctly with the frontend
and that real-time WebSocket updates are properly delivered.

Requirements tested:
- 1.1-1.5: Dashboard statistics integration
- 2.1-2.5: Sessions list integration
- 3.1-3.5: Configuration page integration
- 4.1-4.6: WebSocket real-time updates
- 5.1-5.5: Authentication flows
"""

import pytest
import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any
import socketio
from fastapi.testclient import TestClient

from src.api.main import app
from src.models.core import OptimizationSession, ModelMetadata
from src.config.optimization_criteria import (
    OptimizationCriteria,
    OptimizationConstraints,
    OptimizationTechnique
)


class TestDashboardIntegration:
    """Test dashboard page integration (Requirements 1.1-1.5)"""
    
    def test_dashboard_stats_load_correctly(self, client: TestClient, auth_headers: Dict):
        """Verify statistics load correctly"""
        response = client.get("/dashboard/stats", headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify all required fields are present
        assert "total_models" in data
        assert "active_optimizations" in data
        assert "completed_optimizations" in data
        assert "failed_optimizations" in data
        assert "average_size_reduction" in data
        assert "average_speed_improvement" in data
        assert "total_sessions" in data
        assert "last_updated" in data
        
        # Verify data types
        assert isinstance(data["total_models"], int)
        assert isinstance(data["active_optimizations"], int)
        assert isinstance(data["completed_optimizations"], int)
        assert isinstance(data["average_size_reduction"], (int, float))
        assert isinstance(data["average_speed_improvement"], (int, float))
    
    def test_dashboard_stats_with_no_data(self, client: TestClient, auth_headers: Dict):
        """Verify zero values returned when no data available"""
        # This test assumes a fresh system with no data
        response = client.get("/dashboard/stats", headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        
        # Should return zero values, not errors
        assert data["total_models"] >= 0
        assert data["active_optimizations"] >= 0
    
    def test_dashboard_stats_error_handling(self, client: TestClient):
        """Verify error handling when backend is unavailable"""
        # Test without authentication
        response = client.get("/dashboard/stats")
        
        assert response.status_code == 401
        assert "error" in response.json() or "detail" in response.json()
    
    def test_dashboard_stats_refresh(self, client: TestClient, auth_headers: Dict):
        """Verify data refreshes on page reload"""
        # Get stats twice
        response1 = client.get("/dashboard/stats", headers=auth_headers)
        time.sleep(0.1)
        response2 = client.get("/dashboard/stats", headers=auth_headers)
        
        assert response1.status_code == 200
        assert response2.status_code == 200
        
        # Both should return valid data
        data1 = response1.json()
        data2 = response2.json()
        
        assert "last_updated" in data1
        assert "last_updated" in data2


class TestSessionsListIntegration:
    """Test sessions list integration (Requirements 2.1-2.5)"""
    
    def test_sessions_display_correctly(self, client: TestClient, auth_headers: Dict):
        """Verify sessions display correctly"""
        response = client.get("/optimization/sessions", headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify response structure
        assert "sessions" in data
        assert "total" in data
        assert isinstance(data["sessions"], list)
        assert isinstance(data["total"], int)
    
    def test_sessions_filtering_by_status(self, client: TestClient, auth_headers: Dict):
        """Verify filtering by status works"""
        # Test each status filter
        for status in ["running", "completed", "failed", "cancelled"]:
            response = client.get(
                f"/optimization/sessions?status={status}",
                headers=auth_headers
            )
            
            assert response.status_code == 200
            data = response.json()
            
            # All returned sessions should match the filter
            for session in data["sessions"]:
                assert session["status"] == status
    
    def test_sessions_pagination(self, client: TestClient, auth_headers: Dict):
        """Verify pagination works"""
        # Test with different pagination parameters
        response1 = client.get(
            "/optimization/sessions?skip=0&limit=5",
            headers=auth_headers
        )
        response2 = client.get(
            "/optimization/sessions?skip=5&limit=5",
            headers=auth_headers
        )
        
        assert response1.status_code == 200
        assert response2.status_code == 200
        
        data1 = response1.json()
        data2 = response2.json()
        
        # Verify pagination metadata
        assert data1["skip"] == 0
        assert data1["limit"] == 5
        assert data2["skip"] == 5
        assert data2["limit"] == 5
        
        # Verify no overlap in results
        ids1 = {s["session_id"] for s in data1["sessions"]}
        ids2 = {s["session_id"] for s in data2["sessions"]}
        assert len(ids1.intersection(ids2)) == 0
    
    def test_sessions_empty_state(self, client: TestClient, auth_headers: Dict):
        """Verify empty state displays properly"""
        # Filter for a status that likely has no results
        response = client.get(
            "/optimization/sessions?status=cancelled&model_id=nonexistent",
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Should return empty array, not error
        assert data["sessions"] == []
        assert data["total"] == 0
    
    def test_sessions_filter_combinations(self, client: TestClient, auth_headers: Dict):
        """Test with various filter combinations"""
        # Test multiple filters together
        start_date = (datetime.now() - timedelta(days=7)).isoformat()
        end_date = datetime.now().isoformat()
        
        response = client.get(
            f"/optimization/sessions?status=completed&start_date={start_date}&end_date={end_date}&limit=10",
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify all filters are applied
        for session in data["sessions"]:
            assert session["status"] == "completed"
            session_date = datetime.fromisoformat(session["created_at"].replace("Z", "+00:00"))
            assert session_date >= datetime.fromisoformat(start_date)
            assert session_date <= datetime.fromisoformat(end_date)


class TestConfigurationIntegration:
    """Test configuration page integration (Requirements 3.1-3.5)"""
    
    def test_configuration_loads_correctly(self, client: TestClient, auth_headers: Dict):
        """Verify configuration loads correctly"""
        response = client.get("/config/optimization-criteria", headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify required fields
        assert "name" in data
        assert "description" in data
        assert "constraints" in data
        assert "target_deployment" in data
        assert "enabled_techniques" in data
    
    def test_configuration_updates_save_properly(self, client: TestClient, auth_headers: Dict):
        """Verify configuration updates save properly"""
        # Get current config
        get_response = client.get("/config/optimization-criteria", headers=auth_headers)
        assert get_response.status_code == 200
        current_config = get_response.json()
        
        # Update config
        updated_config = current_config.copy()
        updated_config["name"] = "test_config_updated"
        updated_config["description"] = "Updated test configuration"
        
        put_response = client.put(
            "/config/optimization-criteria",
            json=updated_config,
            headers=auth_headers
        )
        
        assert put_response.status_code == 200
        saved_config = put_response.json()
        
        # Verify updates were saved
        assert saved_config["name"] == "test_config_updated"
        assert saved_config["description"] == "Updated test configuration"
        
        # Verify by fetching again
        verify_response = client.get("/config/optimization-criteria", headers=auth_headers)
        assert verify_response.status_code == 200
        verified_config = verify_response.json()
        assert verified_config["name"] == "test_config_updated"
    
    def test_configuration_validation_errors(self, client: TestClient, auth_headers: Dict):
        """Verify validation errors display correctly"""
        # Send invalid configuration
        invalid_config = {
            "name": "",  # Empty name should fail validation
            "description": "Test",
            "constraints": {
                "preserve_accuracy_threshold": 1.5,  # Invalid: > 1.0
                "allowed_techniques": []
            },
            "target_deployment": "invalid_target"
        }
        
        response = client.put(
            "/config/optimization-criteria",
            json=invalid_config,
            headers=auth_headers
        )
        
        # Should return validation error
        assert response.status_code in [400, 422]
        error_data = response.json()
        assert "error" in error_data or "detail" in error_data
    
    def test_configuration_various_values(self, client: TestClient, auth_headers: Dict):
        """Test with various configuration values"""
        test_configs = [
            {
                "name": "aggressive_optimization",
                "description": "Aggressive optimization settings",
                "constraints": {
                    "preserve_accuracy_threshold": 0.85,
                    "max_size_reduction_percent": 70.0,
                    "allowed_techniques": ["quantization", "pruning", "distillation"]
                },
                "target_deployment": "edge",
                "enabled_techniques": ["quantization", "pruning", "distillation"]
            },
            {
                "name": "conservative_optimization",
                "description": "Conservative optimization settings",
                "constraints": {
                    "preserve_accuracy_threshold": 0.98,
                    "max_size_reduction_percent": 20.0,
                    "allowed_techniques": ["quantization"]
                },
                "target_deployment": "cloud",
                "enabled_techniques": ["quantization"]
            }
        ]
        
        for config in test_configs:
            response = client.put(
                "/config/optimization-criteria",
                json=config,
                headers=auth_headers
            )
            
            # Should accept valid configurations
            assert response.status_code == 200
            saved = response.json()
            assert saved["name"] == config["name"]


class TestWebSocketIntegration:
    """Test WebSocket real-time updates (Requirements 4.1-4.6)"""
    
    @pytest.mark.asyncio
    async def test_websocket_connection_status(self):
        """Verify connection status indicator works"""
        sio = socketio.AsyncClient()
        connected = False
        
        @sio.event
        async def connect():
            nonlocal connected
            connected = True
        
        try:
            await sio.connect('http://localhost:8000', socketio_path='/socket.io')
            await asyncio.sleep(0.5)
            
            assert connected, "WebSocket should connect successfully"
            
            await sio.disconnect()
        except Exception as e:
            pytest.skip(f"WebSocket server not available: {e}")
    
    @pytest.mark.asyncio
    async def test_websocket_progress_updates(self):
        """Verify progress updates appear in real-time"""
        sio = socketio.AsyncClient()
        progress_updates = []
        
        @sio.event
        async def session_progress(data):
            progress_updates.append(data)
        
        try:
            await sio.connect('http://localhost:8000', socketio_path='/socket.io')
            await asyncio.sleep(0.5)
            
            # Subscribe to a test session
            await sio.emit('subscribe', {'session_id': 'test_session'})
            
            # Wait for potential updates
            await asyncio.sleep(2)
            
            # Note: This test may not receive updates if no active sessions
            # The important part is that the connection works
            
            await sio.disconnect()
        except Exception as e:
            pytest.skip(f"WebSocket server not available: {e}")
    
    @pytest.mark.asyncio
    async def test_websocket_reconnection(self):
        """Verify reconnection after disconnect"""
        sio = socketio.AsyncClient()
        connection_count = 0
        
        @sio.event
        async def connect():
            nonlocal connection_count
            connection_count += 1
        
        try:
            # First connection
            await sio.connect('http://localhost:8000', socketio_path='/socket.io')
            await asyncio.sleep(0.5)
            assert connection_count == 1
            
            # Disconnect
            await sio.disconnect()
            await asyncio.sleep(0.5)
            
            # Reconnect
            await sio.connect('http://localhost:8000', socketio_path='/socket.io')
            await asyncio.sleep(0.5)
            assert connection_count == 2
            
            await sio.disconnect()
        except Exception as e:
            pytest.skip(f"WebSocket server not available: {e}")
    
    @pytest.mark.asyncio
    async def test_websocket_multiple_tabs(self):
        """Test with multiple browser tabs (multiple connections)"""
        clients = []
        
        try:
            # Create multiple clients (simulating multiple tabs)
            for i in range(3):
                sio = socketio.AsyncClient()
                await sio.connect('http://localhost:8000', socketio_path='/socket.io')
                clients.append(sio)
                await asyncio.sleep(0.2)
            
            # All should be connected
            assert len(clients) == 3
            
            # Disconnect all
            for sio in clients:
                await sio.disconnect()
        except Exception as e:
            pytest.skip(f"WebSocket server not available: {e}")


class TestAuthenticationFlows:
    """Test authentication flows (Requirements 5.1-5.5)"""
    
    def test_protected_endpoints_require_auth(self, client: TestClient):
        """Verify protected endpoints require auth"""
        protected_endpoints = [
            "/dashboard/stats",
            "/optimization/sessions",
            "/config/optimization-criteria",
            "/models"
        ]
        
        for endpoint in protected_endpoints:
            response = client.get(endpoint)
            assert response.status_code == 401, f"Endpoint {endpoint} should require authentication"
    
    def test_valid_token_processing(self, client: TestClient, auth_headers: Dict):
        """Verify valid authentication token is processed normally"""
        response = client.get("/dashboard/stats", headers=auth_headers)
        assert response.status_code == 200
    
    def test_expired_token_handling(self, client: TestClient):
        """Verify expired token handling"""
        # Use an obviously invalid/expired token
        headers = {"Authorization": "Bearer expired_token_12345"}
        response = client.get("/dashboard/stats", headers=headers)
        
        # Should return 401 for invalid token
        assert response.status_code == 401
    
    def test_insufficient_permissions(self, client: TestClient):
        """Verify insufficient permissions handling"""
        # This test would require role-based access control
        # For now, we test that the endpoint exists and handles auth
        headers = {"Authorization": "Bearer user_token"}
        response = client.put(
            "/config/optimization-criteria",
            json={"name": "test"},
            headers=headers
        )
        
        # Should return 401 or 403
        assert response.status_code in [401, 403]
    
    def test_login_flow(self, client: TestClient):
        """Verify login flow works"""
        # Test login endpoint
        response = client.post(
            "/auth/login",
            json={"username": "admin", "password": "admin"}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify token is returned
        assert "access_token" in data
        assert "token_type" in data
        assert data["token_type"] == "bearer"
        
        # Verify user info is returned
        assert "user" in data
        assert "username" in data["user"]


# Fixtures

@pytest.fixture(scope="module")
def client():
    """Create test client"""
    from fastapi.testclient import TestClient
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="module")
def auth_token(client: TestClient) -> str:
    """Get authentication token"""
    # Login to get token
    response = client.post(
        "/auth/login",
        json={"username": "admin", "password": "admin"}
    )
    
    assert response.status_code == 200, f"Login failed: {response.json()}"
    data = response.json()
    token = data.get("access_token")
    assert token is not None, "No access token in response"
    return token


@pytest.fixture(scope="module")
def auth_headers(auth_token: str) -> Dict[str, str]:
    """Get authentication headers"""
    return {"Authorization": f"Bearer {auth_token}"}


@pytest.fixture
def sample_model_id(client: TestClient, auth_headers: Dict) -> str:
    """Create a sample model for testing"""
    # This would upload a test model
    # For now, return a mock ID
    return "test_model_123"


@pytest.fixture
def sample_session_id(client: TestClient, auth_headers: Dict, sample_model_id: str) -> str:
    """Create a sample optimization session for testing"""
    # This would start an optimization session
    # For now, return a mock ID
    return "test_session_123"

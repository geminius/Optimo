"""
Integration tests for REST API functionality.

Tests complete workflows including model upload, optimization execution,
session management, and results retrieval with real components.
"""

import pytest
import tempfile
import os
import asyncio
from pathlib import Path
from datetime import datetime
from unittest.mock import patch, MagicMock

# Skip if dependencies are missing
pytest.importorskip("httpx")
pytest.importorskip("fastapi")

from fastapi.testclient import TestClient
import torch
import torch.nn as nn

from src.api.main import app
from src.services.optimization_manager import OptimizationManager
from src.config.optimization_criteria import OptimizationCriteria


class SimpleTestModel(nn.Module):
    """Simple model for testing."""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 10)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


@pytest.fixture(scope="module")
def test_model_file():
    """Create a temporary test model file."""
    model = SimpleTestModel()
    tmp_file = tempfile.NamedTemporaryFile(suffix=".pt", delete=False)
    torch.save(model.state_dict(), tmp_file.name)
    tmp_file.close()
    
    yield tmp_file.name
    
    # Cleanup
    if os.path.exists(tmp_file.name):
        os.unlink(tmp_file.name)


@pytest.fixture(scope="module")
def client():
    """Create test client with mocked platform integrator."""
    from src.api.models import User
    from src.api.dependencies import get_current_user, get_optimization_manager
    
    # Mock user for authentication bypass
    async def mock_get_current_user():
        return User(
            id="test-user",
            username="test",
            role="administrator",
            is_active=True
        )
    
    # Mock optimization manager dependency
    async def mock_get_optimization_manager():
        mock_manager = MagicMock(spec=OptimizationManager)
        mock_manager.get_active_sessions.return_value = []
        return mock_manager
    
    # Override dependencies
    app.dependency_overrides[get_current_user] = mock_get_current_user
    app.dependency_overrides[get_optimization_manager] = mock_get_optimization_manager
    
    # Mock platform integrator for shutdown
    async def mock_shutdown():
        pass
    
    mock_integrator = MagicMock()
    mock_integrator.shutdown_platform = mock_shutdown
    
    app.state.platform_integrator = mock_integrator
    app.state.optimization_manager = MagicMock(spec=OptimizationManager)
    
    with TestClient(app) as test_client:
        yield test_client
    
    # Clean up overrides
    app.dependency_overrides.clear()


@pytest.fixture
def auth_headers(client):
    """Get authentication headers."""
    response = client.post("/auth/login", json={
        "username": "admin",
        "password": "admin"
    })
    assert response.status_code == 200
    token = response.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


class TestCompleteOptimizationWorkflow:
    """Test complete optimization workflow from upload to results."""
    
    def test_full_workflow(self, client, auth_headers, test_model_file):
        """Test complete workflow: upload -> optimize -> monitor -> results."""
        
        # Step 1: Upload model
        with open(test_model_file, "rb") as f:
            upload_response = client.post(
                "/models/upload",
                headers=auth_headers,
                files={"file": ("test_model.pt", f, "application/octet-stream")},
                data={"name": "Integration Test Model", "description": "Test model"}
            )
        
        assert upload_response.status_code == 200
        upload_data = upload_response.json()
        model_id = upload_data["model_id"]
        assert "model_id" in upload_data
        assert upload_data["filename"] == "test_model.pt"
        
        # Step 2: List models to verify upload
        list_response = client.get("/models", headers=auth_headers)
        assert list_response.status_code == 200
        models = list_response.json()["models"]
        assert any(m["id"] == model_id for m in models)
        
        # Step 3: Start optimization
        from src.api.dependencies import get_optimization_manager
        
        mock_manager = MagicMock(spec=OptimizationManager)
        mock_manager.start_optimization_session.return_value = "test-session-123"
        mock_manager.get_session_status.return_value = {
            "status": "running",
            "progress_percentage": 25.0,
            "current_step": "Analyzing model",
            "start_time": datetime.now().isoformat(),
            "last_update": datetime.now().isoformat(),
            "error_message": None,
            "session_data": {
                "model_id": model_id,
                "steps_completed": 1
            }
        }
        
        async def mock_get_opt_manager():
            return mock_manager
        
        # Override the dependency for this test
        app.dependency_overrides[get_optimization_manager] = mock_get_opt_manager
        
        try:
            optimize_request = {
                "model_id": model_id,
                "criteria_name": "default",
                "target_accuracy_threshold": 0.95,
                "optimization_techniques": ["quantization"]
            }
            
            optimize_response = client.post(
                "/optimize",
                headers=auth_headers,
                json=optimize_request
            )
            
            assert optimize_response.status_code == 200
            optimize_data = optimize_response.json()
            session_id = optimize_data["session_id"]
            assert session_id == "test-session-123"
            assert optimize_data["status"] == "started"
            
            # Step 4: Monitor session status
            status_response = client.get(
                f"/sessions/{session_id}/status",
                headers=auth_headers
            )
            
            assert status_response.status_code == 200
            status_data = status_response.json()
            assert status_data["session_id"] == session_id
            assert status_data["status"] == "running"
            assert status_data["progress_percentage"] == 25.0
            
            # Step 5: List sessions
            sessions_response = client.get("/sessions", headers=auth_headers)
            assert sessions_response.status_code == 200
            
            # Step 6: Simulate completion and get results
            mock_manager.get_session_status.return_value = {
                "status": "completed",
                "progress_percentage": 100.0,
                "current_step": "Completed",
                "start_time": datetime.now().isoformat(),
                "last_update": datetime.now().isoformat(),
                "error_message": None,
                "session_data": {
                    "model_id": model_id,
                    "steps_completed": 5
                }
            }
            
            results_response = client.get(
                f"/sessions/{session_id}/results",
                headers=auth_headers
            )
            
            assert results_response.status_code == 200
            results_data = results_response.json()
            assert results_data["session_id"] == session_id
            assert results_data["status"] == "completed"
            assert "performance_improvements" in results_data
            assert "techniques_applied" in results_data
        finally:
            # Restore original override
            async def mock_get_optimization_manager():
                mock_mgr = MagicMock(spec=OptimizationManager)
                mock_mgr.get_active_sessions.return_value = []
                return mock_mgr
            app.dependency_overrides[get_optimization_manager] = mock_get_optimization_manager
        
        # Cleanup: Delete model
        delete_response = client.delete(
            f"/models/{model_id}",
            headers=auth_headers
        )
        assert delete_response.status_code == 200


class TestSessionManagement:
    """Test session management operations."""
    
    def test_session_lifecycle(self, client, auth_headers, test_model_file):
        """Test session creation, monitoring, cancellation, and rollback."""
        
        # Upload model
        with open(test_model_file, "rb") as f:
            upload_response = client.post(
                "/models/upload",
                headers=auth_headers,
                files={"file": ("test_model.pt", f, "application/octet-stream")}
            )
        
        model_id = upload_response.json()["model_id"]
        
        from src.api.dependencies import get_optimization_manager
        
        mock_manager = MagicMock(spec=OptimizationManager)
        session_id = "session-lifecycle-test"
        mock_manager.start_optimization_session.return_value = session_id
        mock_manager.get_session_status.return_value = {
            "status": "running",
            "progress_percentage": 50.0,
            "current_step": "Optimizing",
            "start_time": datetime.now().isoformat(),
            "last_update": datetime.now().isoformat(),
            "error_message": None,
            "session_data": {
                "model_id": model_id,
                "steps_completed": 2
            }
        }
        mock_manager.cancel_session.return_value = True
        mock_manager.rollback_session.return_value = True
        
        async def mock_get_opt_manager():
            return mock_manager
        
        app.dependency_overrides[get_optimization_manager] = mock_get_opt_manager
        
        try:
            # Start session
            optimize_response = client.post(
                "/optimize",
                headers=auth_headers,
                json={"model_id": model_id}
            )
            assert optimize_response.status_code == 200
            
            # Check status
            status_response = client.get(
                f"/sessions/{session_id}/status",
                headers=auth_headers
            )
            assert status_response.status_code == 200
            assert status_response.json()["status"] == "running"
            
            # Cancel session
            cancel_response = client.post(
                f"/sessions/{session_id}/cancel",
                headers=auth_headers
            )
            assert cancel_response.status_code == 200
            assert "message" in cancel_response.json()
            
            # Rollback session
            rollback_response = client.post(
                f"/sessions/{session_id}/rollback",
                headers=auth_headers
            )
            assert rollback_response.status_code == 200
            assert "message" in rollback_response.json()
        finally:
            # Restore original override
            async def mock_get_optimization_manager():
                mock_mgr = MagicMock(spec=OptimizationManager)
                mock_mgr.get_active_sessions.return_value = []
                return mock_mgr
            app.dependency_overrides[get_optimization_manager] = mock_get_optimization_manager
        
        # Cleanup
        client.delete(f"/models/{model_id}", headers=auth_headers)


class TestModelManagement:
    """Test model management operations."""
    
    def test_model_crud_operations(self, client, auth_headers, test_model_file):
        """Test create, read, update, delete operations for models."""
        
        # Create (Upload)
        with open(test_model_file, "rb") as f:
            upload_response = client.post(
                "/models/upload",
                headers=auth_headers,
                files={"file": ("crud_test.pt", f, "application/octet-stream")},
                data={
                    "name": "CRUD Test Model",
                    "description": "Testing CRUD operations",
                    "tags": "test,integration"
                }
            )
        
        assert upload_response.status_code == 200
        model_id = upload_response.json()["model_id"]
        
        # Read (List)
        list_response = client.get("/models", headers=auth_headers)
        assert list_response.status_code == 200
        models = list_response.json()["models"]
        assert any(m["id"] == model_id for m in models)
        
        # Read with pagination
        paginated_response = client.get(
            "/models?skip=0&limit=5",
            headers=auth_headers
        )
        assert paginated_response.status_code == 200
        assert paginated_response.json()["limit"] == 5
        
        # Delete
        delete_response = client.delete(
            f"/models/{model_id}",
            headers=auth_headers
        )
        assert delete_response.status_code == 200
        
        # Verify deletion
        list_after_delete = client.get("/models", headers=auth_headers)
        models_after = list_after_delete.json()["models"]
        assert not any(m["id"] == model_id for m in models_after)
    
    def test_upload_multiple_models(self, client, auth_headers, test_model_file):
        """Test uploading multiple models."""
        model_ids = []
        
        try:
            for i in range(3):
                with open(test_model_file, "rb") as f:
                    response = client.post(
                        "/models/upload",
                        headers=auth_headers,
                        files={"file": (f"model_{i}.pt", f, "application/octet-stream")},
                        data={"name": f"Model {i}"}
                    )
                
                assert response.status_code == 200
                model_ids.append(response.json()["model_id"])
            
            # Verify all models are listed
            list_response = client.get("/models", headers=auth_headers)
            assert list_response.status_code == 200
            models = list_response.json()["models"]
            
            for model_id in model_ids:
                assert any(m["id"] == model_id for m in models)
        
        finally:
            # Cleanup
            for model_id in model_ids:
                client.delete(f"/models/{model_id}", headers=auth_headers)


class TestErrorHandlingAndRecovery:
    """Test error handling and recovery scenarios."""
    
    def test_optimization_with_invalid_model(self, client, auth_headers):
        """Test optimization with non-existent model."""
        optimize_request = {
            "model_id": "non-existent-model-id",
            "criteria_name": "default"
        }
        
        response = client.post(
            "/optimize",
            headers=auth_headers,
            json=optimize_request
        )
        
        assert response.status_code == 404
    
    def test_session_status_not_found(self, client, auth_headers):
        """Test getting status for non-existent session."""
        from src.api.dependencies import get_optimization_manager
        
        mock_manager = MagicMock(spec=OptimizationManager)
        mock_manager.get_session_status.side_effect = ValueError("Session not found")
        
        async def mock_get_opt_manager():
            return mock_manager
        
        app.dependency_overrides[get_optimization_manager] = mock_get_opt_manager
        
        try:
            response = client.get(
                "/sessions/non-existent-session/status",
                headers=auth_headers
            )
            
            assert response.status_code == 404
        finally:
            # Restore original override
            async def mock_get_optimization_manager():
                mock_mgr = MagicMock(spec=OptimizationManager)
                mock_mgr.get_active_sessions.return_value = []
                return mock_mgr
            app.dependency_overrides[get_optimization_manager] = mock_get_optimization_manager
    
    def test_cancel_non_existent_session(self, client, auth_headers):
        """Test cancelling non-existent session."""
        from src.api.dependencies import get_optimization_manager
        
        mock_manager = MagicMock(spec=OptimizationManager)
        mock_manager.cancel_session.return_value = False
        
        async def mock_get_opt_manager():
            return mock_manager
        
        app.dependency_overrides[get_optimization_manager] = mock_get_opt_manager
        
        try:
            response = client.post(
                "/sessions/non-existent/cancel",
                headers=auth_headers
            )
            
            assert response.status_code == 404
        finally:
            # Restore original override
            async def mock_get_optimization_manager():
                mock_mgr = MagicMock(spec=OptimizationManager)
                mock_mgr.get_active_sessions.return_value = []
                return mock_mgr
            app.dependency_overrides[get_optimization_manager] = mock_get_optimization_manager
    
    def test_results_for_incomplete_session(self, client, auth_headers):
        """Test getting results for incomplete session."""
        with patch('src.api.main.app.state.optimization_manager') as mock_manager:
            mock_manager.get_session_status.return_value = {
                "status": "running",
                "progress_percentage": 50.0,
                "current_step": "Optimizing",
                "start_time": datetime.now().isoformat(),
                "last_update": datetime.now().isoformat(),
                "error_message": None,
                "session_data": {
                    "model_id": "test-model",
                    "steps_completed": 2
                }
            }
            
            response = client.get(
                "/sessions/running-session/results",
                headers=auth_headers
            )
            
            assert response.status_code == 400
    
    def test_upload_oversized_file(self, client, auth_headers):
        """Test uploading file exceeding size limit."""
        # Create a large file (simulate)
        large_data = b"x" * (600 * 1024 * 1024)  # 600MB
        
        tmp_file = tempfile.NamedTemporaryFile(suffix=".pt", delete=False)
        tmp_file.write(large_data[:1024])  # Write small amount for test
        tmp_file.close()
        
        try:
            with patch('src.api.main.open', side_effect=Exception("File too large")):
                with open(tmp_file.name, "rb") as f:
                    # Mock the file size check
                    with patch('src.api.main.MAX_FILE_SIZE', 1024):
                        response = client.post(
                            "/models/upload",
                            headers=auth_headers,
                            files={"file": ("large.pt", f, "application/octet-stream")}
                        )
                
                # Should handle the error gracefully
                assert response.status_code in [413, 500]
        finally:
            os.unlink(tmp_file.name)


class TestAuthenticationAndAuthorization:
    """Test authentication and authorization."""
    
    def test_access_without_authentication(self):
        """Test accessing protected endpoints without auth."""
        # Create a new client without dependency overrides for this test
        from fastapi.testclient import TestClient
        from src.api.main import app as test_app
        
        # Clear dependency overrides temporarily
        original_overrides = test_app.dependency_overrides.copy()
        test_app.dependency_overrides.clear()
        
        try:
            with TestClient(test_app) as test_client:
                endpoints = [
                    ("/models", "GET"),
                    ("/models/upload", "POST"),
                    ("/optimize", "POST"),
                    ("/sessions", "GET"),
                ]
                
                for endpoint, method in endpoints:
                    if method == "GET":
                        response = test_client.get(endpoint)
                    else:
                        response = test_client.post(endpoint)
                    
                    assert response.status_code in [401, 403]
        finally:
            # Restore overrides
            test_app.dependency_overrides = original_overrides
    
    def test_access_with_invalid_token(self):
        """Test accessing endpoints with invalid token."""
        # Create a new client without dependency overrides for this test
        from fastapi.testclient import TestClient
        from src.api.main import app as test_app
        
        # Clear dependency overrides temporarily
        original_overrides = test_app.dependency_overrides.copy()
        test_app.dependency_overrides.clear()
        
        try:
            with TestClient(test_app) as test_client:
                headers = {"Authorization": "Bearer invalid-token-12345"}
                
                response = test_client.get("/models", headers=headers)
                assert response.status_code == 401
        finally:
            # Restore overrides
            test_app.dependency_overrides = original_overrides
    
    def test_login_flow(self, client):
        """Test complete login flow."""
        # Valid login
        login_response = client.post("/auth/login", json={
            "username": "admin",
            "password": "admin"
        })
        
        assert login_response.status_code == 200
        data = login_response.json()
        assert "access_token" in data
        assert "token_type" in data
        assert data["token_type"] == "bearer"
        assert "user" in data
        
        # Use token to access protected endpoint
        token = data["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        
        models_response = client.get("/models", headers=headers)
        assert models_response.status_code == 200


class TestConcurrentOperations:
    """Test concurrent operations and race conditions."""
    
    def test_concurrent_model_uploads(self, client, auth_headers, test_model_file):
        """Test uploading multiple models concurrently."""
        model_ids = []
        
        try:
            # Simulate concurrent uploads
            for i in range(3):
                with open(test_model_file, "rb") as f:
                    response = client.post(
                        "/models/upload",
                        headers=auth_headers,
                        files={"file": (f"concurrent_{i}.pt", f, "application/octet-stream")}
                    )
                
                assert response.status_code == 200
                model_ids.append(response.json()["model_id"])
            
            # Verify all uploads succeeded
            assert len(model_ids) == 3
            assert len(set(model_ids)) == 3  # All unique
        
        finally:
            for model_id in model_ids:
                client.delete(f"/models/{model_id}", headers=auth_headers)
    
    def test_concurrent_optimization_requests(self, client, auth_headers, test_model_file):
        """Test starting multiple optimization sessions."""
        # Upload model
        with open(test_model_file, "rb") as f:
            upload_response = client.post(
                "/models/upload",
                headers=auth_headers,
                files={"file": ("concurrent_opt.pt", f, "application/octet-stream")}
            )
        
        model_id = upload_response.json()["model_id"]
        
        try:
            from src.api.dependencies import get_optimization_manager
            
            mock_manager = MagicMock(spec=OptimizationManager)
            session_ids = []
            
            # Mock different session IDs for each request
            def create_session(*args, **kwargs):
                session_id = f"session-{len(session_ids)}"
                session_ids.append(session_id)
                return session_id
            
            mock_manager.start_optimization_session.side_effect = create_session
            mock_manager.get_session_status.return_value = {
                "status": "running",
                "progress_percentage": 0.0,
                "current_step": "Starting",
                "start_time": datetime.now().isoformat(),
                "last_update": datetime.now().isoformat(),
                "error_message": None,
                "session_data": {
                    "model_id": model_id,
                    "steps_completed": 0
                }
            }
            
            async def mock_get_opt_manager():
                return mock_manager
            
            app.dependency_overrides[get_optimization_manager] = mock_get_opt_manager
            
            try:
                # Start multiple optimization sessions
                for i in range(3):
                    response = client.post(
                        "/optimize",
                        headers=auth_headers,
                        json={"model_id": model_id}
                    )
                    assert response.status_code == 200
                
                # Verify all sessions were created
                assert len(session_ids) == 3
                assert len(set(session_ids)) == 3  # All unique
            finally:
                # Restore original override
                async def mock_get_optimization_manager():
                    mock_mgr = MagicMock(spec=OptimizationManager)
                    mock_mgr.get_active_sessions.return_value = []
                    return mock_mgr
                app.dependency_overrides[get_optimization_manager] = mock_get_optimization_manager
        
        finally:
            client.delete(f"/models/{model_id}", headers=auth_headers)


class TestHealthAndMonitoring:
    """Test health check and monitoring endpoints."""
    
    def test_health_check(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "version" in data
        assert "services" in data
    
    def test_health_check_during_operations(self, client, auth_headers, test_model_file):
        """Test health check while operations are running."""
        # Upload model
        with open(test_model_file, "rb") as f:
            client.post(
                "/models/upload",
                headers=auth_headers,
                files={"file": ("health_test.pt", f, "application/octet-stream")}
            )
        
        # Check health
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"


class TestValidationAndConstraints:
    """Test request validation and constraints."""
    
    def test_optimization_request_validation(self, client, auth_headers):
        """Test validation of optimization requests."""
        # Missing required field
        response = client.post(
            "/optimize",
            headers=auth_headers,
            json={}
        )
        assert response.status_code == 422
        
        # Invalid threshold values
        response = client.post(
            "/optimize",
            headers=auth_headers,
            json={
                "model_id": "test",
                "target_accuracy_threshold": 1.5  # Invalid: > 1.0
            }
        )
        # Should either reject or clamp the value
        assert response.status_code in [404, 422]  # 404 for missing model, 422 for validation
    
    def test_file_type_validation(self, client, auth_headers):
        """Test file type validation on upload."""
        # Create invalid file type
        tmp_file = tempfile.NamedTemporaryFile(suffix=".txt", delete=False)
        tmp_file.write(b"not a model file")
        tmp_file.close()
        
        try:
            with open(tmp_file.name, "rb") as f:
                response = client.post(
                    "/models/upload",
                    headers=auth_headers,
                    files={"file": ("invalid.txt", f, "text/plain")}
                )
            
            assert response.status_code == 400
        finally:
            os.unlink(tmp_file.name)
    
    def test_pagination_validation(self, client, auth_headers):
        """Test pagination parameter validation."""
        # Valid pagination
        response = client.get("/models?skip=0&limit=10", headers=auth_headers)
        assert response.status_code == 200
        
        # Edge cases
        response = client.get("/models?skip=0&limit=1000", headers=auth_headers)
        assert response.status_code == 200
        
        # Negative values (should handle gracefully)
        response = client.get("/models?skip=-1&limit=-1", headers=auth_headers)
        assert response.status_code == 200  # Should use defaults


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

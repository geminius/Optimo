"""Integration tests for the REST API."""

import pytest

# Skip the entire module if required dependencies are missing
pytest.importorskip("httpx")
pytest.importorskip("jwt")

import tempfile
import json
import os
from pathlib import Path
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch

# Import the FastAPI app
from src.api.main import app
from src.services.optimization_manager import OptimizationManager


@pytest.fixture
def mock_user():
    """Create a mock user for testing."""
    from src.api.models import User
    return User(
        id="test_user",
        username="test",
        role="user",
        email="test@example.com",
        is_active=True
    )


@pytest.fixture
def client(mock_optimization_manager, mock_user):
    """Create test client for API testing with authentication enabled."""
    # Ensure upload directory exists
    from pathlib import Path
    upload_dir = Path("uploads")
    upload_dir.mkdir(exist_ok=True)
    
    # Override authentication to return mock user
    from src.api.dependencies import get_current_user, get_optimization_manager
    app.dependency_overrides[get_current_user] = lambda: mock_user
    app.dependency_overrides[get_optimization_manager] = lambda: mock_optimization_manager
    
    # Also set app state for any code that accesses it directly
    app.state.optimization_manager = mock_optimization_manager
    app.state.platform_integrator = Mock()
    
    # Create client
    client = TestClient(app)
    yield client
    
    # Clean up
    app.dependency_overrides.clear()
    if hasattr(app.state, 'optimization_manager'):
        delattr(app.state, 'optimization_manager')
    if hasattr(app.state, 'platform_integrator'):
        delattr(app.state, 'platform_integrator')


@pytest.fixture
def unauthenticated_client(mock_optimization_manager):
    """Create test client without authentication for testing auth failures."""
    # Ensure upload directory exists
    from pathlib import Path
    upload_dir = Path("uploads")
    upload_dir.mkdir(exist_ok=True)
    
    # Override get_optimization_manager but NOT get_current_user
    from src.api.dependencies import get_optimization_manager
    app.dependency_overrides[get_optimization_manager] = lambda: mock_optimization_manager
    
    # Also set app state
    app.state.optimization_manager = mock_optimization_manager
    app.state.platform_integrator = Mock()
    
    # Create client WITHOUT auth override
    client = TestClient(app)
    yield client
    
    # Clean up
    app.dependency_overrides.clear()
    if hasattr(app.state, 'optimization_manager'):
        delattr(app.state, 'optimization_manager')
    if hasattr(app.state, 'platform_integrator'):
        delattr(app.state, 'platform_integrator')


@pytest.fixture
def auth_token():
    """Get authentication token for tests."""
    return "test_token_12345"


@pytest.fixture
def auth_headers(auth_token):
    """Get authorization headers for authenticated requests."""
    return {"Authorization": f"Bearer {auth_token}"}


@pytest.fixture
def mock_optimization_manager():
    """Mock optimization manager for testing."""
    mock_manager = Mock(spec=OptimizationManager)
    mock_manager.get_active_sessions.return_value = []
    mock_manager.get_session_status.return_value = {
        "status": "running",
        "progress_percentage": 50.0,
        "current_step": "Analyzing model",
        "start_time": "2024-01-01T00:00:00",
        "last_update": "2024-01-01T00:30:00",
        "error_message": None,
        "session_data": {
            "model_id": "test-model",
            "steps_completed": 2
        }
    }
    mock_manager.start_optimization_session.return_value = "test-session-id"
    mock_manager.cancel_session.return_value = True
    mock_manager.rollback_session.return_value = True
    return mock_manager


class TestHealthEndpoints:
    """Test health check endpoints."""
    
    def test_health_check(self, client):
        """Test basic health check."""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "version" in data
        assert "services" in data


class TestAuthenticationEndpoints:
    """Test authentication endpoints."""
    
    def test_login_success(self, client):
        """Test successful login."""
        response = client.post("/auth/login", json={
            "username": "admin",
            "password": "admin"
        })
        
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"
        assert "user" in data
        assert data["user"]["username"] == "admin"
    
    def test_login_invalid_credentials(self, client):
        """Test login with invalid credentials."""
        response = client.post("/auth/login", json={
            "username": "invalid",
            "password": "invalid"
        })
        
        assert response.status_code == 401
    
    def test_login_missing_credentials(self, client):
        """Test login with missing credentials."""
        response = client.post("/auth/login", json={})
        assert response.status_code == 400


class TestModelEndpoints:
    """Test model management endpoints."""
    
    def test_upload_model_success(self, client, auth_headers):
        """Test successful model upload."""
        # Create a temporary file to upload
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp_file:
            tmp_file.write(b"fake model data")
            tmp_file_path = tmp_file.name
        
        try:
            with open(tmp_file_path, "rb") as f:
                response = client.post(
                    "/models/upload",
                    headers=auth_headers,
                    files={"file": ("test_model.pt", f, "application/octet-stream")},
                    data={"name": "Test Model", "description": "Test description"}
                )
            
            assert response.status_code == 200
            data = response.json()
            assert "model_id" in data
            assert data["filename"] == "test_model.pt"
            assert "upload_time" in data
            
        finally:
            os.unlink(tmp_file_path)
    
    def test_upload_model_unauthorized(self, unauthenticated_client):
        """Test model upload without authentication."""
        with tempfile.NamedTemporaryFile(suffix=".pt") as tmp_file:
            tmp_file.write(b"fake model data")
            tmp_file.seek(0)
            
            response = unauthenticated_client.post(
                "/models/upload",
                files={"file": ("test_model.pt", tmp_file, "application/octet-stream")}
            )
            
            assert response.status_code == 403  # No auth header
    
    def test_upload_invalid_file_type(self, client, auth_headers):
        """Test upload with invalid file type."""
        with tempfile.NamedTemporaryFile(suffix=".txt") as tmp_file:
            tmp_file.write(b"not a model file")
            tmp_file.seek(0)
            
            response = client.post(
                "/models/upload",
                headers=auth_headers,
                files={"file": ("test.txt", tmp_file, "text/plain")}
            )
            
            assert response.status_code == 400
    
    def test_list_models(self, client, auth_headers):
        """Test listing models."""
        response = client.get("/models", headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert "models" in data
        assert "total" in data
        assert "skip" in data
        assert "limit" in data
    
    def test_list_models_with_pagination(self, client, auth_headers):
        """Test listing models with pagination."""
        response = client.get("/models?skip=0&limit=10", headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert data["skip"] == 0
        assert data["limit"] == 10


class TestOptimizationEndpoints:
    """Test optimization endpoints."""
    
    @patch('src.api.main.app.state')
    def test_start_optimization(self, mock_state, client, auth_headers, mock_optimization_manager):
        """Test starting optimization."""
        mock_state.optimization_manager = mock_optimization_manager
        
        # First upload a model (mock the file existence)
        with patch('pathlib.Path.glob') as mock_glob:
            mock_glob.return_value = [Path("uploads/test-model_test.pt")]
            
            request_data = {
                "model_id": "test-model",
                "criteria_name": "default",
                "target_accuracy_threshold": 0.95,
                "optimization_techniques": ["quantization", "pruning"]
            }
            
            response = client.post(
                "/optimize",
                headers=auth_headers,
                json=request_data
            )
            
            assert response.status_code == 200
            data = response.json()
            assert "session_id" in data
            assert data["model_id"] == "test-model"
            assert data["status"] == "started"
    
    @patch('src.api.main.app.state')
    def test_start_optimization_model_not_found(self, mock_state, client, auth_headers, mock_optimization_manager):
        """Test starting optimization with non-existent model."""
        mock_state.optimization_manager = mock_optimization_manager
        
        with patch('pathlib.Path.glob') as mock_glob:
            mock_glob.return_value = []  # No model files found
            
            request_data = {
                "model_id": "non-existent-model",
                "criteria_name": "default"
            }
            
            response = client.post(
                "/optimize",
                headers=auth_headers,
                json=request_data
            )
            
            assert response.status_code == 404


class TestSessionEndpoints:
    """Test session management endpoints."""
    
    @patch('src.api.main.app.state')
    def test_get_session_status(self, mock_state, client, auth_headers, mock_optimization_manager):
        """Test getting session status."""
        mock_state.optimization_manager = mock_optimization_manager
        
        response = client.get(
            "/sessions/test-session-id/status",
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == "test-session-id"
        assert "status" in data
        assert "progress_percentage" in data
    
    @patch('src.api.main.app.state')
    def test_get_session_status_not_found(self, mock_state, client, auth_headers, mock_optimization_manager):
        """Test getting status for non-existent session."""
        mock_state.optimization_manager = mock_optimization_manager
        mock_optimization_manager.get_session_status.side_effect = ValueError("Session not found")
        
        response = client.get(
            "/sessions/non-existent/status",
            headers=auth_headers
        )
        
        assert response.status_code == 404
    
    @patch('src.api.main.app.state')
    def test_list_sessions(self, mock_state, client, auth_headers, mock_optimization_manager):
        """Test listing sessions."""
        mock_state.optimization_manager = mock_optimization_manager
        mock_optimization_manager.get_active_sessions.return_value = ["session1", "session2"]
        
        # Mock get_session_status to return different data for each session
        def mock_get_session_status(session_id):
            return {
                "status": "running" if session_id == "session1" else "completed",
                "progress_percentage": 50.0 if session_id == "session1" else 100.0,
                "start_time": "2024-01-01T00:00:00",
                "last_update": "2024-01-01T00:30:00",
                "session_data": {
                    "model_id": f"model-{session_id}",
                    "model_name": f"test-model-{session_id}",
                    "created_at": "2024-01-01T00:00:00",
                    "updated_at": "2024-01-01T00:30:00"
                }
            }
        
        mock_optimization_manager.get_session_status.side_effect = mock_get_session_status
        
        response = client.get("/sessions", headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert "sessions" in data
        assert "total" in data
        assert data["total"] == 2
        assert len(data["sessions"]) == 2
    
    @patch('src.api.main.app.state')
    def test_cancel_session(self, mock_state, client, auth_headers, mock_optimization_manager):
        """Test cancelling session."""
        mock_state.optimization_manager = mock_optimization_manager
        
        response = client.post(
            "/sessions/test-session-id/cancel",
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
    
    @patch('src.api.main.app.state')
    def test_rollback_session(self, mock_state, client, auth_headers, mock_optimization_manager):
        """Test rolling back session."""
        mock_state.optimization_manager = mock_optimization_manager
        
        response = client.post(
            "/sessions/test-session-id/rollback",
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "message" in data


class TestResultsEndpoints:
    """Test results and evaluation endpoints."""
    
    @patch('src.api.main.app.state')
    def test_get_session_results_completed(self, mock_state, client, auth_headers, mock_optimization_manager):
        """Test getting results for completed session."""
        mock_state.optimization_manager = mock_optimization_manager
        
        # Mock completed session status
        mock_optimization_manager.get_session_status.return_value = {
            "status": "completed",
            "progress_percentage": 100.0,
            "current_step": "Completed",
            "start_time": "2024-01-01T00:00:00",
            "last_update": "2024-01-01T01:00:00",
            "error_message": None,
            "session_data": {
                "model_id": "test-model",
                "steps_completed": 5
            }
        }
        
        response = client.get(
            "/sessions/test-session-id/results",
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == "test-session-id"
        assert "optimization_summary" in data
        assert "performance_improvements" in data
        assert "techniques_applied" in data
    
    @patch('src.api.main.app.state')
    def test_get_session_results_not_completed(self, mock_state, client, auth_headers, mock_optimization_manager):
        """Test getting results for incomplete session."""
        mock_state.optimization_manager = mock_optimization_manager
        
        # Mock running session status
        mock_optimization_manager.get_session_status.return_value = {
            "status": "running",
            "progress_percentage": 50.0,
            "current_step": "Optimizing",
            "start_time": "2024-01-01T00:00:00",
            "last_update": "2024-01-01T00:30:00",
            "error_message": None,
            "session_data": {
                "model_id": "test-model",
                "steps_completed": 2
            }
        }
        
        response = client.get(
            "/sessions/test-session-id/results",
            headers=auth_headers
        )
        
        assert response.status_code == 400


class TestErrorHandling:
    """Test error handling scenarios."""
    
    def test_unauthorized_access(self, unauthenticated_client):
        """Test accessing protected endpoints without authentication."""
        response = unauthenticated_client.get("/models")
        assert response.status_code == 403
    
    def test_invalid_token(self, unauthenticated_client):
        """Test accessing endpoints with invalid token."""
        headers = {"Authorization": "Bearer invalid-token"}
        response = unauthenticated_client.get("/models", headers=headers)
        assert response.status_code == 401
    
    def test_service_unavailable(self, client, auth_headers):
        """Test handling when optimization manager is unavailable."""
        from src.api.dependencies import get_optimization_manager
        from fastapi import HTTPException, status
        
        # Override to raise 503 error
        def unavailable_manager():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Optimization manager not available"
            )
        
        app.dependency_overrides[get_optimization_manager] = unavailable_manager
        
        try:
            response = client.get("/sessions", headers=auth_headers)
            assert response.status_code == 503
        finally:
            # Restore original override
            from tests.test_api import mock_optimization_manager
            app.dependency_overrides[get_optimization_manager] = lambda: client.app.state.optimization_manager


class TestValidation:
    """Test request validation."""
    
    def test_optimization_request_validation(self, client, auth_headers):
        """Test optimization request validation."""
        # Missing required model_id
        response = client.post(
            "/optimize",
            headers=auth_headers,
            json={}
        )
        assert response.status_code == 422
    
    def test_invalid_pagination_parameters(self, client, auth_headers):
        """Test invalid pagination parameters."""
        response = client.get("/models?skip=-1&limit=0", headers=auth_headers)
        # Should still work but with corrected parameters
        assert response.status_code == 200


class TestDistilBertIntegration:
    """End-to-end API validation using a DistilBERT model."""

    @patch('src.api.main.app.state')
    def test_distilbert_e2e(self, mock_state, client, auth_headers):
        """Upload and optimize a DistilBERT model through the API."""
        from unittest.mock import MagicMock

        mock_optimization_manager = MagicMock()
        mock_optimization_manager.start_optimization_session.return_value = "test-session-id"
        mock_optimization_manager.get_session_status.return_value = {
            "status": "running",
            "progress_percentage": 50.0,
            "current_step": "Analyzing model",
            "start_time": "2024-01-01T00:00:00",
            "last_update": "2024-01-01T00:30:00",
            "error_message": None,
            "session_data": {"model_id": "test-model", "steps_completed": 2},
        }

        mock_state.optimization_manager = mock_optimization_manager

        from transformers import AutoModel
        import torch
        import tempfile
        from pathlib import Path
        import os

        model = AutoModel.from_pretrained(
            "hf-internal-testing/tiny-random-distilbert"
        )

        tmp_file = tempfile.NamedTemporaryFile(suffix=".pt", delete=False)
        try:
            torch.save(model, tmp_file)
            tmp_file.close()

            with open(tmp_file.name, "rb") as f:
                upload_resp = client.post(
                    "/models/upload",
                    headers=auth_headers,
                    files={"file": ("distilbert.pt", f, "application/octet-stream")},
                    data={"name": "distilbert", "description": "test model"},
                )

            assert upload_resp.status_code == 200
            model_id = upload_resp.json()["model_id"]

            request_data = {"model_id": model_id, "criteria_name": "default"}
            optimize_resp = client.post(
                "/optimize", headers=auth_headers, json=request_data
            )

            assert optimize_resp.status_code == 200
            session_id = optimize_resp.json()["session_id"]
            assert session_id == "test-session-id"

            status_resp = client.get(
                f"/sessions/{session_id}/status", headers=auth_headers
            )

            assert status_resp.status_code == 200
            assert status_resp.json()["session_id"] == session_id
        finally:
            tmp_path = Path(tmp_file.name)
            if tmp_path.exists():
                tmp_path.unlink()
            if 'model_id' in locals():
                for fp in Path('uploads').glob(f"{model_id}_*"):
                    fp.unlink()

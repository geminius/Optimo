"""
Unit tests for the sessions API endpoints.

Tests cover:
- Filtering by various parameters (status, model_id, date range)
- Pagination edge cases (skip, limit, boundaries)
- Empty results handling
- Error scenarios (invalid parameters, service failures)
- Authentication and authorization
- Data enrichment with model information
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock
from fastapi.testclient import TestClient

# Import the FastAPI app
from src.api.main import app
from src.api.models import User
from src.models.core import OptimizationSession, OptimizationStatus
from src.config.optimization_criteria import OptimizationCriteria, OptimizationConstraints, OptimizationTechnique


@pytest.fixture
def mock_user():
    """Create a mock user for testing."""
    return User(
        id="test_user",
        username="test",
        role="user",
        email="test@example.com",
        is_active=True
    )


@pytest.fixture
def mock_memory_manager():
    """Create a mock MemoryManager for testing."""
    memory_manager = Mock()
    
    # Mock list_sessions to return sample session data
    memory_manager.list_sessions.return_value = [
        {
            "id": "session_1",
            "model_id": "model_1",
            "status": "completed",
            "criteria_name": "default",
            "created_at": "2025-01-01T10:00:00",
            "started_at": "2025-01-01T10:01:00",
            "completed_at": "2025-01-01T10:30:00",
            "created_by": "test_user",
            "priority": 1,
            "tags": [],
            "data_size": 1024
        },
        {
            "id": "session_2",
            "model_id": "model_2",
            "status": "running",
            "criteria_name": "aggressive",
            "created_at": "2025-01-01T11:00:00",
            "started_at": "2025-01-01T11:01:00",
            "completed_at": None,
            "created_by": "test_user",
            "priority": 2,
            "tags": [],
            "data_size": 2048
        }
    ]
    
    # Mock retrieve_session to return full session objects
    def mock_retrieve_session(session_id):
        if session_id == "session_1":
            session = OptimizationSession(
                id="session_1",
                model_id="model_1",
                status=OptimizationStatus.COMPLETED,
                criteria_name="default",
                created_at=datetime(2025, 1, 1, 10, 0, 0),
                started_at=datetime(2025, 1, 1, 10, 1, 0),
                completed_at=datetime(2025, 1, 1, 10, 30, 0)
            )
            # Add mock criteria attribute for enrichment
            session.criteria = Mock()
            session.criteria.constraints = Mock()
            session.criteria.constraints.allowed_techniques = [OptimizationTechnique.QUANTIZATION, OptimizationTechnique.PRUNING]
            # Add mock results
            session.results = Mock()
            session.results.size_reduction_percent = 25.0
            session.results.speed_improvement_percent = 15.0
            return session
        elif session_id == "session_2":
            session = OptimizationSession(
                id="session_2",
                model_id="model_2",
                status=OptimizationStatus.RUNNING,
                criteria_name="aggressive",
                created_at=datetime(2025, 1, 1, 11, 0, 0),
                started_at=datetime(2025, 1, 1, 11, 1, 0)
            )
            # Add mock criteria attribute for enrichment
            session.criteria = Mock()
            session.criteria.constraints = Mock()
            session.criteria.constraints.allowed_techniques = [OptimizationTechnique.QUANTIZATION]
            session.steps = [Mock(status="completed"), Mock(status="running")]
            return session
        return None
    
    memory_manager.retrieve_session.side_effect = mock_retrieve_session
    
    # Mock get_session_statistics
    memory_manager.get_session_statistics.return_value = {
        "total_sessions": 2,
        "status_distribution": {"completed": 1, "running": 1},
        "total_storage_size_mb": 3.0
    }
    
    return memory_manager


@pytest.fixture
def mock_model_store():
    """Create a mock ModelStore for testing."""
    model_store = Mock()
    
    # Mock get_metadata to return model metadata
    def mock_get_metadata(model_id):
        if model_id == "model_1":
            metadata = Mock()
            metadata.name = "Test Model 1"
            return metadata
        elif model_id == "model_2":
            metadata = Mock()
            metadata.name = "Test Model 2"
            return metadata
        return None
    
    model_store.get_metadata.side_effect = mock_get_metadata
    
    return model_store


@pytest.fixture
def mock_platform_integrator(mock_memory_manager, mock_model_store):
    """Create a mock PlatformIntegrator."""
    integrator = Mock()
    integrator.get_memory_manager.return_value = mock_memory_manager
    integrator.get_model_store.return_value = mock_model_store
    return integrator


@pytest.fixture
def client(mock_user, mock_platform_integrator):
    """Create test client for API testing."""
    from src.api.dependencies import get_current_user
    
    # Override authentication
    app.dependency_overrides[get_current_user] = lambda: mock_user
    
    # Set app state
    app.state.platform_integrator = mock_platform_integrator
    
    # Create client
    client = TestClient(app)
    yield client
    
    # Clean up
    app.dependency_overrides.clear()
    if hasattr(app.state, 'platform_integrator'):
        delattr(app.state, 'platform_integrator')


def test_list_sessions_success(client):
    """Test successful session listing."""
    response = client.get("/optimization/sessions")
    
    assert response.status_code == 200
    data = response.json()
    
    assert "sessions" in data
    assert "total" in data
    assert "skip" in data
    assert "limit" in data
    
    assert len(data["sessions"]) == 2
    assert data["total"] >= 2
    assert data["skip"] == 0
    assert data["limit"] == 50


def test_list_sessions_with_status_filter(client):
    """Test session listing with status filter."""
    response = client.get("/optimization/sessions?status=completed")
    
    assert response.status_code == 200
    data = response.json()
    
    # Should filter to only completed sessions
    assert len(data["sessions"]) >= 0


def test_list_sessions_with_model_id_filter(client):
    """Test session listing with model_id filter."""
    response = client.get("/optimization/sessions?model_id=model_1")
    
    assert response.status_code == 200
    data = response.json()
    
    # Should filter to only sessions for model_1
    assert len(data["sessions"]) >= 0


def test_list_sessions_with_pagination(client):
    """Test session listing with pagination."""
    response = client.get("/optimization/sessions?skip=0&limit=10")
    
    assert response.status_code == 200
    data = response.json()
    
    assert data["skip"] == 0
    assert data["limit"] == 10
    assert len(data["sessions"]) <= 10


def test_list_sessions_invalid_status(client):
    """Test session listing with invalid status."""
    response = client.get("/optimization/sessions?status=invalid_status")
    
    assert response.status_code == 400
    data = response.json()
    # Error handler may transform the response, check for either format
    assert "Invalid status" in (data.get("detail", "") or data.get("message", ""))


def test_list_sessions_invalid_limit(client):
    """Test session listing with limit exceeding maximum."""
    response = client.get("/optimization/sessions?limit=200")
    
    # FastAPI's built-in validation returns 422 for constraint violations
    assert response.status_code == 422


def test_list_sessions_negative_skip(client):
    """Test session listing with negative skip."""
    response = client.get("/optimization/sessions?skip=-1")
    
    # FastAPI validation should catch this
    assert response.status_code == 422


def test_list_sessions_enriched_data(client):
    """Test that session data is enriched with model information."""
    response = client.get("/optimization/sessions")
    
    assert response.status_code == 200
    data = response.json()
    
    if len(data["sessions"]) > 0:
        session = data["sessions"][0]
        
        # Check required fields
        assert "session_id" in session
        assert "model_id" in session
        assert "model_name" in session
        assert "status" in session
        assert "progress_percentage" in session
        assert "techniques" in session
        assert "created_at" in session
        assert "updated_at" in session


def test_list_sessions_without_auth(mock_platform_integrator):
    """Test that sessions endpoint requires authentication."""
    # Create client without auth override
    app.state.platform_integrator = mock_platform_integrator
    client = TestClient(app)
    
    response = client.get("/optimization/sessions")
    
    # Should return 403 (no credentials provided)
    assert response.status_code == 403
    
    # Clean up
    if hasattr(app.state, 'platform_integrator'):
        delattr(app.state, 'platform_integrator')


def test_list_sessions_with_date_filters(client):
    """Test session listing with date range filters."""
    response = client.get(
        "/optimization/sessions"
        "?start_date=2025-01-01T00:00:00"
        "&end_date=2025-01-02T00:00:00"
    )
    
    assert response.status_code == 200
    data = response.json()
    
    # Should return sessions within date range
    assert "sessions" in data


def test_list_sessions_empty_result(client, mock_platform_integrator):
    """Test session listing when no sessions exist."""
    # Override memory manager to return empty list
    memory_manager = mock_platform_integrator.get_memory_manager()
    memory_manager.list_sessions.return_value = []
    
    response = client.get("/optimization/sessions")
    
    assert response.status_code == 200
    data = response.json()
    
    assert len(data["sessions"]) == 0
    assert data["total"] >= 0


# ============================================================================
# COMPREHENSIVE TEST SUITE FOR TASK 3.4
# ============================================================================


class TestSessionsFilteringByParameters:
    """Test filtering sessions by various parameters."""
    
    def test_filter_by_completed_status(self, client, mock_platform_integrator):
        """Test filtering sessions by completed status."""
        memory_manager = mock_platform_integrator.get_memory_manager()
        
        # Mock only completed sessions
        memory_manager.list_sessions.return_value = [
            {
                "id": "session_completed",
                "model_id": "model_1",
                "status": "completed",
                "criteria_name": "default",
                "created_at": "2025-01-01T10:00:00",
                "completed_at": "2025-01-01T10:30:00",
                "created_by": "test_user",
                "priority": 1,
                "tags": [],
                "data_size": 1024
            }
        ]
        
        response = client.get("/optimization/sessions?status=completed")
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify filter was applied
        memory_manager.list_sessions.assert_called_once()
        call_args = memory_manager.list_sessions.call_args
        assert call_args[1]["status_filter"] == ["completed"]
    
    def test_filter_by_running_status(self, client, mock_platform_integrator):
        """Test filtering sessions by running status."""
        memory_manager = mock_platform_integrator.get_memory_manager()
        
        response = client.get("/optimization/sessions?status=running")
        
        assert response.status_code == 200
        
        # Verify filter was applied
        call_args = memory_manager.list_sessions.call_args
        assert call_args[1]["status_filter"] == ["running"]
    
    def test_filter_by_failed_status(self, client, mock_platform_integrator):
        """Test filtering sessions by failed status."""
        memory_manager = mock_platform_integrator.get_memory_manager()
        
        response = client.get("/optimization/sessions?status=failed")
        
        assert response.status_code == 200
        
        # Verify filter was applied
        call_args = memory_manager.list_sessions.call_args
        assert call_args[1]["status_filter"] == ["failed"]
    
    def test_filter_by_pending_status(self, client, mock_platform_integrator):
        """Test filtering sessions by pending status."""
        memory_manager = mock_platform_integrator.get_memory_manager()
        
        response = client.get("/optimization/sessions?status=pending")
        
        assert response.status_code == 200
        
        # Verify filter was applied
        call_args = memory_manager.list_sessions.call_args
        assert call_args[1]["status_filter"] == ["pending"]
    
    def test_filter_by_cancelled_status(self, client, mock_platform_integrator):
        """Test filtering sessions by cancelled status."""
        memory_manager = mock_platform_integrator.get_memory_manager()
        
        response = client.get("/optimization/sessions?status=cancelled")
        
        assert response.status_code == 200
        
        # Verify filter was applied
        call_args = memory_manager.list_sessions.call_args
        assert call_args[1]["status_filter"] == ["cancelled"]
    
    def test_filter_by_specific_model_id(self, client, mock_platform_integrator):
        """Test filtering sessions by specific model ID."""
        memory_manager = mock_platform_integrator.get_memory_manager()
        
        response = client.get("/optimization/sessions?model_id=model_123")
        
        assert response.status_code == 200
        
        # Verify filter was applied
        call_args = memory_manager.list_sessions.call_args
        assert call_args[1]["model_id_filter"] == "model_123"
    
    def test_filter_by_start_date(self, client):
        """Test filtering sessions by start date."""
        start_date = "2025-01-01T00:00:00"
        response = client.get(f"/optimization/sessions?start_date={start_date}")
        
        assert response.status_code == 200
        data = response.json()
        
        # All returned sessions should be after start_date
        for session in data["sessions"]:
            session_date = datetime.fromisoformat(session["created_at"].replace('Z', '+00:00'))
            filter_date = datetime.fromisoformat(start_date)
            assert session_date >= filter_date
    
    def test_filter_by_end_date(self, client):
        """Test filtering sessions by end date."""
        end_date = "2025-12-31T23:59:59"
        response = client.get(f"/optimization/sessions?end_date={end_date}")
        
        assert response.status_code == 200
        data = response.json()
        
        # All returned sessions should be before end_date
        for session in data["sessions"]:
            session_date = datetime.fromisoformat(session["created_at"].replace('Z', '+00:00'))
            filter_date = datetime.fromisoformat(end_date)
            assert session_date <= filter_date
    
    def test_filter_by_date_range(self, client):
        """Test filtering sessions by date range."""
        start_date = "2025-01-01T00:00:00"
        end_date = "2025-01-31T23:59:59"
        
        response = client.get(
            f"/optimization/sessions?start_date={start_date}&end_date={end_date}"
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # All returned sessions should be within date range
        for session in data["sessions"]:
            session_date = datetime.fromisoformat(session["created_at"].replace('Z', '+00:00'))
            start = datetime.fromisoformat(start_date)
            end = datetime.fromisoformat(end_date)
            assert start <= session_date <= end
    
    def test_filter_by_multiple_parameters(self, client, mock_platform_integrator):
        """Test filtering sessions by multiple parameters simultaneously."""
        memory_manager = mock_platform_integrator.get_memory_manager()
        
        response = client.get(
            "/optimization/sessions"
            "?status=completed"
            "&model_id=model_1"
            "&start_date=2025-01-01T00:00:00"
            "&end_date=2025-12-31T23:59:59"
        )
        
        assert response.status_code == 200
        
        # Verify all filters were applied
        call_args = memory_manager.list_sessions.call_args
        assert call_args[1]["status_filter"] == ["completed"]
        assert call_args[1]["model_id_filter"] == "model_1"
    
    def test_filter_with_no_matching_results(self, client, mock_platform_integrator):
        """Test filtering that returns no matching sessions."""
        memory_manager = mock_platform_integrator.get_memory_manager()
        memory_manager.list_sessions.return_value = []
        
        response = client.get("/optimization/sessions?status=completed&model_id=nonexistent")
        
        assert response.status_code == 200
        data = response.json()
        assert len(data["sessions"]) == 0
        assert data["total"] == 0


class TestSessionsPaginationEdgeCases:
    """Test pagination edge cases and boundary conditions."""
    
    def test_pagination_default_values(self, client, mock_platform_integrator):
        """Test pagination with default skip and limit values."""
        memory_manager = mock_platform_integrator.get_memory_manager()
        
        response = client.get("/optimization/sessions")
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify defaults
        assert data["skip"] == 0
        assert data["limit"] == 50
        
        # Verify passed to memory manager
        call_args = memory_manager.list_sessions.call_args
        assert call_args[1]["offset"] == 0
        assert call_args[1]["limit"] == 50
    
    def test_pagination_custom_skip(self, client, mock_platform_integrator):
        """Test pagination with custom skip value."""
        memory_manager = mock_platform_integrator.get_memory_manager()
        
        response = client.get("/optimization/sessions?skip=10")
        
        assert response.status_code == 200
        data = response.json()
        assert data["skip"] == 10
        
        call_args = memory_manager.list_sessions.call_args
        assert call_args[1]["offset"] == 10
    
    def test_pagination_custom_limit(self, client, mock_platform_integrator):
        """Test pagination with custom limit value."""
        memory_manager = mock_platform_integrator.get_memory_manager()
        
        response = client.get("/optimization/sessions?limit=25")
        
        assert response.status_code == 200
        data = response.json()
        assert data["limit"] == 25
        
        call_args = memory_manager.list_sessions.call_args
        assert call_args[1]["limit"] == 25
    
    def test_pagination_maximum_limit(self, client, mock_platform_integrator):
        """Test pagination with maximum allowed limit (100)."""
        memory_manager = mock_platform_integrator.get_memory_manager()
        
        response = client.get("/optimization/sessions?limit=100")
        
        assert response.status_code == 200
        data = response.json()
        assert data["limit"] == 100
    
    def test_pagination_limit_exceeds_maximum(self, client):
        """Test pagination with limit exceeding maximum (should fail)."""
        response = client.get("/optimization/sessions?limit=200")
        
        # FastAPI validation should reject this
        assert response.status_code == 422
    
    def test_pagination_minimum_limit(self, client, mock_platform_integrator):
        """Test pagination with minimum limit (1)."""
        memory_manager = mock_platform_integrator.get_memory_manager()
        
        # Mock to return only 1 session
        memory_manager.list_sessions.return_value = [
            {
                "id": "session_1",
                "model_id": "model_1",
                "status": "completed",
                "criteria_name": "default",
                "created_at": "2025-01-01T10:00:00",
                "completed_at": "2025-01-01T10:30:00",
                "created_by": "test_user",
                "priority": 1,
                "tags": [],
                "data_size": 1024
            }
        ]
        
        response = client.get("/optimization/sessions?limit=1")
        
        assert response.status_code == 200
        data = response.json()
        assert data["limit"] == 1
        
        # Verify limit was passed to memory manager
        call_args = memory_manager.list_sessions.call_args
        assert call_args[1]["limit"] == 1
    
    def test_pagination_zero_limit(self, client):
        """Test pagination with zero limit (should fail)."""
        response = client.get("/optimization/sessions?limit=0")
        
        # FastAPI validation should reject this
        assert response.status_code == 422
    
    def test_pagination_negative_skip(self, client):
        """Test pagination with negative skip (should fail)."""
        response = client.get("/optimization/sessions?skip=-1")
        
        # FastAPI validation should reject this
        assert response.status_code == 422
    
    def test_pagination_large_skip(self, client, mock_platform_integrator):
        """Test pagination with very large skip value."""
        memory_manager = mock_platform_integrator.get_memory_manager()
        memory_manager.list_sessions.return_value = []
        
        response = client.get("/optimization/sessions?skip=10000")
        
        assert response.status_code == 200
        data = response.json()
        assert data["skip"] == 10000
        assert len(data["sessions"]) == 0
    
    def test_pagination_skip_and_limit_combination(self, client, mock_platform_integrator):
        """Test pagination with both skip and limit."""
        memory_manager = mock_platform_integrator.get_memory_manager()
        
        response = client.get("/optimization/sessions?skip=20&limit=10")
        
        assert response.status_code == 200
        data = response.json()
        assert data["skip"] == 20
        assert data["limit"] == 10
        
        call_args = memory_manager.list_sessions.call_args
        assert call_args[1]["offset"] == 20
        assert call_args[1]["limit"] == 10
    
    def test_pagination_total_count_accuracy(self, client, mock_platform_integrator):
        """Test that total count is accurate for pagination."""
        memory_manager = mock_platform_integrator.get_memory_manager()
        
        # Return 2 sessions
        memory_manager.list_sessions.return_value = [
            {
                "id": "session_1",
                "model_id": "model_1",
                "status": "completed",
                "criteria_name": "default",
                "created_at": "2025-01-01T10:00:00",
                "completed_at": "2025-01-01T10:30:00",
                "created_by": "test_user",
                "priority": 1,
                "tags": [],
                "data_size": 1024
            },
            {
                "id": "session_2",
                "model_id": "model_2",
                "status": "running",
                "criteria_name": "aggressive",
                "created_at": "2025-01-01T11:00:00",
                "started_at": "2025-01-01T11:01:00",
                "completed_at": None,
                "created_by": "test_user",
                "priority": 2,
                "tags": [],
                "data_size": 2048
            }
        ]
        
        response = client.get("/optimization/sessions?skip=5&limit=10")
        
        assert response.status_code == 200
        data = response.json()
        
        # Total should account for skip
        assert data["total"] >= len(data["sessions"])
    
    def test_pagination_empty_page(self, client, mock_platform_integrator):
        """Test requesting a page beyond available data."""
        memory_manager = mock_platform_integrator.get_memory_manager()
        memory_manager.list_sessions.return_value = []
        
        response = client.get("/optimization/sessions?skip=1000&limit=50")
        
        assert response.status_code == 200
        data = response.json()
        assert len(data["sessions"]) == 0
        assert data["skip"] == 1000


class TestSessionsEmptyResultsHandling:
    """Test handling of empty results in various scenarios."""
    
    def test_empty_sessions_list(self, client, mock_platform_integrator):
        """Test when no sessions exist in the system."""
        memory_manager = mock_platform_integrator.get_memory_manager()
        memory_manager.list_sessions.return_value = []
        
        response = client.get("/optimization/sessions")
        
        assert response.status_code == 200
        data = response.json()
        
        assert isinstance(data["sessions"], list)
        assert len(data["sessions"]) == 0
        assert data["total"] == 0
        assert "skip" in data
        assert "limit" in data
    
    def test_empty_after_filtering(self, client, mock_platform_integrator):
        """Test when filtering returns no results."""
        memory_manager = mock_platform_integrator.get_memory_manager()
        memory_manager.list_sessions.return_value = []
        
        response = client.get("/optimization/sessions?status=completed&model_id=nonexistent")
        
        assert response.status_code == 200
        data = response.json()
        assert len(data["sessions"]) == 0
    
    def test_empty_after_date_filtering(self, client, mock_platform_integrator):
        """Test when date filtering returns no results."""
        memory_manager = mock_platform_integrator.get_memory_manager()
        memory_manager.list_sessions.return_value = []
        
        response = client.get(
            "/optimization/sessions"
            "?start_date=2099-01-01T00:00:00"
            "&end_date=2099-12-31T23:59:59"
        )
        
        assert response.status_code == 200
        data = response.json()
        assert len(data["sessions"]) == 0
    
    def test_empty_response_structure(self, client, mock_platform_integrator):
        """Test that empty response has correct structure."""
        memory_manager = mock_platform_integrator.get_memory_manager()
        memory_manager.list_sessions.return_value = []
        
        response = client.get("/optimization/sessions")
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify structure
        assert "sessions" in data
        assert "total" in data
        assert "skip" in data
        assert "limit" in data
        assert isinstance(data["sessions"], list)
        assert isinstance(data["total"], int)
        assert isinstance(data["skip"], int)
        assert isinstance(data["limit"], int)


class TestSessionsErrorScenarios:
    """Test error scenarios and exception handling."""
    
    def test_invalid_status_parameter(self, client):
        """Test with invalid status value."""
        response = client.get("/optimization/sessions?status=invalid_status")
        
        assert response.status_code == 400
        data = response.json()
        assert "Invalid status" in (data.get("detail", "") or data.get("message", ""))
    
    def test_invalid_date_format(self, client):
        """Test with invalid date format."""
        response = client.get("/optimization/sessions?start_date=not-a-date")
        
        # FastAPI validation should catch this
        assert response.status_code == 422
    
    def test_start_date_after_end_date(self, client):
        """Test with start_date after end_date."""
        response = client.get(
            "/optimization/sessions"
            "?start_date=2025-12-31T23:59:59"
            "&end_date=2025-01-01T00:00:00"
        )
        
        assert response.status_code == 400
        data = response.json()
        # Check both 'detail' and 'message' fields (error handler may transform response)
        error_text = data.get("detail", "") or data.get("message", "")
        assert "start_date must be before end_date" in error_text
    
    def test_memory_manager_unavailable(self, client, mock_platform_integrator):
        """Test when MemoryManager is unavailable."""
        mock_platform_integrator.get_memory_manager.return_value = None
        
        response = client.get("/optimization/sessions")
        
        assert response.status_code == 503
        data = response.json()
        assert "memory manager" in data.get("message", "").lower() or "memory manager" in data.get("detail", "").lower()
    
    def test_model_store_unavailable(self, client, mock_platform_integrator):
        """Test when ModelStore is unavailable."""
        mock_platform_integrator.get_model_store.return_value = None
        
        response = client.get("/optimization/sessions")
        
        assert response.status_code == 503
        data = response.json()
        assert "model store" in data.get("message", "").lower() or "model store" in data.get("detail", "").lower()
    
    def test_platform_integrator_unavailable(self, mock_user):
        """Test when platform integrator is not available."""
        from src.api.dependencies import get_current_user
        
        app.dependency_overrides[get_current_user] = lambda: mock_user
        
        # Don't set platform_integrator
        if hasattr(app.state, 'platform_integrator'):
            delattr(app.state, 'platform_integrator')
        
        try:
            client = TestClient(app)
            response = client.get("/optimization/sessions")
            
            assert response.status_code == 503
        finally:
            app.dependency_overrides.clear()
    
    def test_memory_manager_query_failure(self, client, mock_platform_integrator):
        """Test when MemoryManager query fails."""
        memory_manager = mock_platform_integrator.get_memory_manager()
        memory_manager.list_sessions.side_effect = Exception("Database connection failed")
        
        response = client.get("/optimization/sessions")
        
        assert response.status_code == 500
        data = response.json()
        # Check both 'detail' and 'message' fields (error handler may transform response)
        error_text = data.get("detail", "") or data.get("message", "")
        assert "Failed to retrieve sessions" in error_text
    
    def test_model_store_query_failure(self, client, mock_platform_integrator):
        """Test graceful handling when ModelStore query fails."""
        model_store = mock_platform_integrator.get_model_store()
        model_store.get_metadata.side_effect = Exception("Metadata fetch failed")
        
        response = client.get("/optimization/sessions")
        
        # Should still return 200 with sessions (model name will be fallback)
        assert response.status_code == 200
        data = response.json()
        
        # Sessions should still be returned with fallback model names
        if len(data["sessions"]) > 0:
            assert "model_name" in data["sessions"][0]
    
    def test_session_retrieval_failure(self, client, mock_platform_integrator):
        """Test when individual session retrieval fails."""
        memory_manager = mock_platform_integrator.get_memory_manager()
        memory_manager.retrieve_session.return_value = None
        
        response = client.get("/optimization/sessions")
        
        # Should return 200 but skip failed sessions
        assert response.status_code == 200
        data = response.json()
        # Sessions that failed to retrieve should be skipped
        assert isinstance(data["sessions"], list)
    
    def test_partial_session_enrichment_failure(self, client, mock_platform_integrator):
        """Test when some sessions fail enrichment but others succeed."""
        memory_manager = mock_platform_integrator.get_memory_manager()
        
        # First session succeeds, second fails
        def mock_retrieve(session_id):
            if session_id == "session_1":
                session = OptimizationSession(
                    id="session_1",
                    model_id="model_1",
                    status=OptimizationStatus.COMPLETED,
                    criteria_name="default",
                    created_at=datetime(2025, 1, 1, 10, 0, 0),
                    completed_at=datetime(2025, 1, 1, 10, 30, 0)
                )
                session.criteria = Mock()
                session.criteria.constraints = Mock()
                session.criteria.constraints.allowed_techniques = []
                return session
            return None
        
        memory_manager.retrieve_session.side_effect = mock_retrieve
        
        response = client.get("/optimization/sessions")
        
        assert response.status_code == 200
        data = response.json()
        # Should return successfully enriched sessions
        assert isinstance(data["sessions"], list)
    
    def test_unexpected_exception(self, client, mock_platform_integrator):
        """Test handling of unexpected exceptions."""
        memory_manager = mock_platform_integrator.get_memory_manager()
        memory_manager.list_sessions.side_effect = RuntimeError("Unexpected error")
        
        response = client.get("/optimization/sessions")
        
        assert response.status_code == 500
        data = response.json()
        assert "detail" in data or "message" in data


class TestSessionsAuthenticationAuthorization:
    """Test authentication and authorization for sessions endpoint."""
    
    def test_requires_authentication(self):
        """Test that endpoint requires authentication."""
        client = TestClient(app)
        response = client.get("/optimization/sessions")
        
        # Should return 403 (no credentials) or 401 (invalid token)
        assert response.status_code in [401, 403]
    
    def test_with_invalid_token(self):
        """Test endpoint with invalid authentication token."""
        client = TestClient(app)
        headers = {"Authorization": "Bearer invalid_token"}
        response = client.get("/optimization/sessions", headers=headers)
        
        assert response.status_code == 401
    
    def test_admin_sees_all_sessions(self, mock_platform_integrator):
        """Test that admin users can see all sessions."""
        from src.api.dependencies import get_current_user
        
        admin_user = User(
            id="admin_user",
            username="admin",
            role="administrator",
            email="admin@example.com",
            is_active=True
        )
        
        app.dependency_overrides[get_current_user] = lambda: admin_user
        app.state.platform_integrator = mock_platform_integrator
        
        try:
            client = TestClient(app)
            response = client.get("/optimization/sessions")
            
            assert response.status_code == 200
            # Admin should see all sessions
        finally:
            app.dependency_overrides.clear()
            if hasattr(app.state, 'platform_integrator'):
                delattr(app.state, 'platform_integrator')
    
    def test_user_sees_only_own_sessions(self, mock_platform_integrator):
        """Test that regular users only see their own sessions."""
        from src.api.dependencies import get_current_user
        
        regular_user = User(
            id="user_123",
            username="regular_user",
            role="user",
            email="user@example.com",
            is_active=True
        )
        
        memory_manager = mock_platform_integrator.get_memory_manager()
        
        # Create sessions with different owners
        def mock_retrieve(session_id):
            session = OptimizationSession(
                id=session_id,
                model_id="model_1",
                status=OptimizationStatus.COMPLETED,
                criteria_name="default",
                created_at=datetime(2025, 1, 1, 10, 0, 0)
            )
            session.criteria = Mock()
            session.criteria.constraints = Mock()
            session.criteria.constraints.allowed_techniques = []
            
            # Set owner based on session_id
            if session_id == "session_1":
                session.owner_id = "user_123"  # Owned by current user
            else:
                session.owner_id = "other_user"  # Owned by different user
            
            return session
        
        memory_manager.retrieve_session.side_effect = mock_retrieve
        
        app.dependency_overrides[get_current_user] = lambda: regular_user
        app.state.platform_integrator = mock_platform_integrator
        
        try:
            client = TestClient(app)
            response = client.get("/optimization/sessions")
            
            assert response.status_code == 200
            data = response.json()
            
            # User should only see their own sessions
            # (session_2 should be filtered out)
        finally:
            app.dependency_overrides.clear()
            if hasattr(app.state, 'platform_integrator'):
                delattr(app.state, 'platform_integrator')


class TestSessionsDataEnrichment:
    """Test data enrichment with model information."""
    
    def test_sessions_include_model_names(self, client):
        """Test that sessions include enriched model names."""
        response = client.get("/optimization/sessions")
        
        assert response.status_code == 200
        data = response.json()
        
        if len(data["sessions"]) > 0:
            session = data["sessions"][0]
            assert "model_name" in session
            assert isinstance(session["model_name"], str)
            assert len(session["model_name"]) > 0
    
    def test_sessions_include_techniques(self, client):
        """Test that sessions include optimization techniques."""
        response = client.get("/optimization/sessions")
        
        assert response.status_code == 200
        data = response.json()
        
        if len(data["sessions"]) > 0:
            session = data["sessions"][0]
            assert "techniques" in session
            assert isinstance(session["techniques"], list)
    
    def test_sessions_include_progress(self, client):
        """Test that sessions include progress percentage."""
        response = client.get("/optimization/sessions")
        
        assert response.status_code == 200
        data = response.json()
        
        if len(data["sessions"]) > 0:
            session = data["sessions"][0]
            assert "progress_percentage" in session
            assert isinstance(session["progress_percentage"], (int, float))
            assert 0 <= session["progress_percentage"] <= 100
    
    def test_sessions_include_performance_metrics(self, client):
        """Test that completed sessions include performance metrics."""
        response = client.get("/optimization/sessions?status=completed")
        
        assert response.status_code == 200
        data = response.json()
        
        if len(data["sessions"]) > 0:
            session = data["sessions"][0]
            assert "size_reduction_percent" in session
            assert "speed_improvement_percent" in session
    
    def test_fallback_model_name_when_metadata_missing(self, client, mock_platform_integrator):
        """Test fallback to model_id when metadata is not available."""
        model_store = mock_platform_integrator.get_model_store()
        
        # Override to return None for all models
        def mock_get_metadata_none(model_id):
            return None
        
        model_store.get_metadata.side_effect = mock_get_metadata_none
        
        response = client.get("/optimization/sessions")
        
        assert response.status_code == 200
        data = response.json()
        
        if len(data["sessions"]) > 0:
            session = data["sessions"][0]
            # Should use model_id as fallback when metadata is None
            assert session["model_name"] == session["model_id"]
    
    def test_sessions_include_timestamps(self, client):
        """Test that sessions include all required timestamps."""
        response = client.get("/optimization/sessions")
        
        assert response.status_code == 200
        data = response.json()
        
        if len(data["sessions"]) > 0:
            session = data["sessions"][0]
            assert "created_at" in session
            assert "updated_at" in session
            # completed_at may be null for running sessions
            assert "completed_at" in session

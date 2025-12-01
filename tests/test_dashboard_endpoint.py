"""
Unit tests for the dashboard API endpoint.

Tests cover:
- Dashboard statistics retrieval with various data scenarios
- Error handling when services are unavailable
- Response format and data type validation
- Authentication and authorization
- Caching behavior
"""

import pytest
from datetime import datetime
from unittest.mock import Mock, MagicMock, patch
from fastapi import HTTPException, status
from fastapi.testclient import TestClient

from src.api.main import app
from src.api.models import DashboardStats, User
from src.api.dashboard import _compute_dashboard_stats
from src.services.optimization_manager import OptimizationManager
from src.models.store import ModelStore
from src.services.memory_manager import MemoryManager


@pytest.fixture(autouse=True)
def clear_cache():
    """Clear cache before and after each test."""
    from src.services.cache_service import CacheService
    cache = CacheService()
    cache.clear()
    yield
    cache.clear()


@pytest.fixture
def mock_user():
    """Create a mock authenticated user."""
    return User(
        id="test_user_123",
        username="testuser",
        role="user",
        email="test@example.com",
        is_active=True
    )


@pytest.fixture
def mock_model_store():
    """Create a mock ModelStore with test data."""
    mock_store = Mock(spec=ModelStore)
    mock_store.list_models.return_value = [
        {"id": "model1", "name": "Model 1"},
        {"id": "model2", "name": "Model 2"},
        {"id": "model3", "name": "Model 3"}
    ]
    return mock_store


@pytest.fixture
def mock_optimization_manager():
    """Create a mock OptimizationManager with test data."""
    mock_manager = Mock(spec=OptimizationManager)
    mock_manager.get_active_sessions.return_value = ["session1", "session2"]
    return mock_manager


@pytest.fixture
def mock_memory_manager():
    """Create a mock MemoryManager with test data."""
    mock_manager = Mock(spec=MemoryManager)
    
    # Mock session statistics
    mock_manager.get_session_statistics.return_value = {
        "total_sessions": 50,
        "status_distribution": {
            "completed": 42,
            "failed": 3,
            "running": 2,
            "cancelled": 3
        }
    }
    
    # Mock completed sessions with results
    mock_session_1 = Mock()
    mock_session_1.results = Mock()
    mock_session_1.results.size_reduction_percent = 35.5
    mock_session_1.results.speed_improvement_percent = 20.3
    
    mock_session_2 = Mock()
    mock_session_2.results = Mock()
    mock_session_2.results.size_reduction_percent = 28.7
    mock_session_2.results.speed_improvement_percent = 15.8
    
    mock_session_3 = Mock()
    mock_session_3.results = Mock()
    mock_session_3.results.size_reduction_percent = 42.1
    mock_session_3.results.speed_improvement_percent = 25.4
    
    mock_manager.list_sessions.return_value = [
        {"id": "session1"},
        {"id": "session2"},
        {"id": "session3"}
    ]
    
    mock_manager.retrieve_session.side_effect = [
        mock_session_1,
        mock_session_2,
        mock_session_3
    ]
    
    return mock_manager


@pytest.fixture
def mock_platform_integrator(mock_model_store, mock_memory_manager):
    """Create a mock platform integrator."""
    mock_integrator = Mock()
    mock_integrator.get_model_store.return_value = mock_model_store
    mock_integrator.get_memory_manager.return_value = mock_memory_manager
    return mock_integrator


@pytest.fixture
def authenticated_client(mock_user, mock_optimization_manager, mock_platform_integrator):
    """Create an authenticated test client."""
    from src.api.dependencies import get_current_user
    
    # Override authentication
    app.dependency_overrides[get_current_user] = lambda: mock_user
    
    # Set app state
    app.state.optimization_manager = mock_optimization_manager
    app.state.platform_integrator = mock_platform_integrator
    
    client = TestClient(app)
    yield client
    
    # Cleanup
    app.dependency_overrides.clear()
    if hasattr(app.state, 'optimization_manager'):
        delattr(app.state, 'optimization_manager')
    if hasattr(app.state, 'platform_integrator'):
        delattr(app.state, 'platform_integrator')


class TestDashboardStatsEndpoint:
    """Test suite for GET /dashboard/stats endpoint."""
    
    def test_get_dashboard_stats_success(self, authenticated_client):
        """Test successful retrieval of dashboard statistics."""
        response = authenticated_client.get("/dashboard/stats")
        
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
        assert isinstance(data["failed_optimizations"], int)
        assert isinstance(data["average_size_reduction"], (int, float))
        assert isinstance(data["average_speed_improvement"], (int, float))
        assert isinstance(data["total_sessions"], int)
        assert isinstance(data["last_updated"], str)
        
        # Verify expected values based on mocks
        assert data["total_models"] == 3
        assert data["active_optimizations"] == 2
        assert data["completed_optimizations"] == 42
        assert data["failed_optimizations"] == 3
        assert data["total_sessions"] == 50
    
    def test_get_dashboard_stats_calculates_averages(self, authenticated_client):
        """Test that averages are calculated correctly from completed sessions."""
        response = authenticated_client.get("/dashboard/stats")
        
        assert response.status_code == 200
        data = response.json()
        
        # Expected averages: (35.5 + 28.7 + 42.1) / 3 = 35.43
        # Expected speed: (20.3 + 15.8 + 25.4) / 3 = 20.5
        assert data["average_size_reduction"] > 0
        assert data["average_speed_improvement"] > 0
        assert 30 < data["average_size_reduction"] < 40
        assert 15 < data["average_speed_improvement"] < 25
    
    def test_get_dashboard_stats_with_no_models(self, mock_user):
        """Test dashboard stats when no models exist."""
        from src.api.dependencies import get_current_user
        
        # Create fresh mocks for this test
        mock_model_store = Mock(spec=ModelStore)
        mock_model_store.list_models.return_value = []
        
        mock_opt_manager = Mock(spec=OptimizationManager)
        mock_opt_manager.get_active_sessions.return_value = []
        
        mock_mem_manager = Mock(spec=MemoryManager)
        mock_mem_manager.get_session_statistics.return_value = {
            "total_sessions": 0,
            "status_distribution": {}
        }
        mock_mem_manager.list_sessions.return_value = []
        
        mock_integrator = Mock()
        mock_integrator.get_model_store.return_value = mock_model_store
        mock_integrator.get_memory_manager.return_value = mock_mem_manager
        
        # Override dependencies
        app.dependency_overrides[get_current_user] = lambda: mock_user
        app.state.optimization_manager = mock_opt_manager
        app.state.platform_integrator = mock_integrator
        
        try:
            client = TestClient(app)
            response = client.get("/dashboard/stats")
            
            assert response.status_code == 200
            data = response.json()
            assert data["total_models"] == 0
        finally:
            app.dependency_overrides.clear()
            if hasattr(app.state, 'optimization_manager'):
                delattr(app.state, 'optimization_manager')
            if hasattr(app.state, 'platform_integrator'):
                delattr(app.state, 'platform_integrator')
    
    def test_get_dashboard_stats_with_no_active_sessions(self, mock_user):
        """Test dashboard stats when no sessions are active."""
        from src.api.dependencies import get_current_user
        
        # Create fresh mocks
        mock_model_store = Mock(spec=ModelStore)
        mock_model_store.list_models.return_value = []
        
        mock_opt_manager = Mock(spec=OptimizationManager)
        mock_opt_manager.get_active_sessions.return_value = []  # No active sessions
        
        mock_mem_manager = Mock(spec=MemoryManager)
        mock_mem_manager.get_session_statistics.return_value = {
            "total_sessions": 0,
            "status_distribution": {}
        }
        mock_mem_manager.list_sessions.return_value = []
        
        mock_integrator = Mock()
        mock_integrator.get_model_store.return_value = mock_model_store
        mock_integrator.get_memory_manager.return_value = mock_mem_manager
        
        app.dependency_overrides[get_current_user] = lambda: mock_user
        app.state.optimization_manager = mock_opt_manager
        app.state.platform_integrator = mock_integrator
        
        try:
            client = TestClient(app)
            response = client.get("/dashboard/stats")
            
            assert response.status_code == 200
            data = response.json()
            assert data["active_optimizations"] == 0
        finally:
            app.dependency_overrides.clear()
            if hasattr(app.state, 'optimization_manager'):
                delattr(app.state, 'optimization_manager')
            if hasattr(app.state, 'platform_integrator'):
                delattr(app.state, 'platform_integrator')
    
    def test_get_dashboard_stats_with_no_completed_sessions(self, mock_user):
        """Test dashboard stats when no sessions are completed."""
        from src.api.dependencies import get_current_user
        
        mock_model_store = Mock(spec=ModelStore)
        mock_model_store.list_models.return_value = []
        
        mock_opt_manager = Mock(spec=OptimizationManager)
        mock_opt_manager.get_active_sessions.return_value = []
        
        mock_memory_manager = Mock(spec=MemoryManager)
        mock_memory_manager.get_session_statistics.return_value = {
            "total_sessions": 5,
            "status_distribution": {
                "completed": 0,
                "failed": 2,
                "running": 3
            }
        }
        mock_memory_manager.list_sessions.return_value = []
        
        mock_integrator = Mock()
        mock_integrator.get_model_store.return_value = mock_model_store
        mock_integrator.get_memory_manager.return_value = mock_memory_manager
        
        app.dependency_overrides[get_current_user] = lambda: mock_user
        app.state.optimization_manager = mock_opt_manager
        app.state.platform_integrator = mock_integrator
        
        try:
            client = TestClient(app)
            response = client.get("/dashboard/stats")
            
            assert response.status_code == 200
            data = response.json()
            assert data["completed_optimizations"] == 0
            assert data["average_size_reduction"] == 0.0
            assert data["average_speed_improvement"] == 0.0
        finally:
            app.dependency_overrides.clear()
            if hasattr(app.state, 'optimization_manager'):
                delattr(app.state, 'optimization_manager')
            if hasattr(app.state, 'platform_integrator'):
                delattr(app.state, 'platform_integrator')
    
    def test_get_dashboard_stats_response_format(self, authenticated_client):
        """Test that response conforms to DashboardStats schema."""
        response = authenticated_client.get("/dashboard/stats")
        
        assert response.status_code == 200
        data = response.json()
        
        # Validate against Pydantic model
        stats = DashboardStats(**data)
        assert stats.total_models >= 0
        assert stats.active_optimizations >= 0
        assert stats.completed_optimizations >= 0
        assert stats.failed_optimizations >= 0
        assert stats.total_sessions >= 0


class TestDashboardStatsErrorHandling:
    """Test error handling scenarios for dashboard endpoint."""
    
    def test_dashboard_stats_model_store_unavailable(
        self, authenticated_client, mock_platform_integrator
    ):
        """Test handling when ModelStore is unavailable."""
        mock_platform_integrator.get_model_store.return_value = None
        
        response = authenticated_client.get("/dashboard/stats")
        
        assert response.status_code == 503
        data = response.json()
        assert "message" in data
        assert "model store" in data["message"].lower()
    
    def test_dashboard_stats_memory_manager_unavailable(
        self, authenticated_client, mock_platform_integrator
    ):
        """Test handling when MemoryManager is unavailable."""
        mock_platform_integrator.get_memory_manager.return_value = None
        
        response = authenticated_client.get("/dashboard/stats")
        
        assert response.status_code == 503
        data = response.json()
        assert "message" in data
        assert "memory manager" in data["message"].lower()
    
    def test_dashboard_stats_platform_integrator_unavailable(self, authenticated_client):
        """Test handling when platform integrator is not available."""
        # Remove platform integrator from app state
        delattr(app.state, 'platform_integrator')
        
        response = authenticated_client.get("/dashboard/stats")
        
        assert response.status_code == 503
        data = response.json()
        assert "message" in data
    
    def test_dashboard_stats_model_store_error(
        self, authenticated_client, mock_platform_integrator
    ):
        """Test handling when ModelStore raises an exception."""
        mock_model_store = Mock(spec=ModelStore)
        mock_model_store.list_models.side_effect = Exception("Database connection failed")
        mock_platform_integrator.get_model_store.return_value = mock_model_store
        
        # The endpoint gracefully handles errors and returns partial data
        response = authenticated_client.get("/dashboard/stats")
        
        # Should return 200 with default value for total_models (0)
        assert response.status_code == 200
        data = response.json()
        assert data["total_models"] == 0  # Defaults to 0 on error
    
    def test_dashboard_stats_memory_manager_error(
        self, authenticated_client, mock_platform_integrator
    ):
        """Test handling when MemoryManager raises an exception."""
        mock_memory_manager = Mock(spec=MemoryManager)
        mock_memory_manager.get_session_statistics.side_effect = Exception("Query failed")
        mock_platform_integrator.get_memory_manager.return_value = mock_memory_manager
        
        response = authenticated_client.get("/dashboard/stats")
        
        # Should return 200 with default values (graceful degradation)
        assert response.status_code == 200
        data = response.json()
        assert data["completed_optimizations"] == 0
        assert data["failed_optimizations"] == 0
    
    def test_dashboard_stats_optimization_manager_error(
        self, authenticated_client, mock_optimization_manager
    ):
        """Test handling when OptimizationManager raises an exception."""
        mock_optimization_manager.get_active_sessions.side_effect = Exception("Service error")
        
        response = authenticated_client.get("/dashboard/stats")
        
        # Should return 200 with default value (graceful degradation)
        assert response.status_code == 200
        data = response.json()
        assert data["active_optimizations"] == 0  # Defaults to 0 on error


class TestDashboardStatsAuthentication:
    """Test authentication requirements for dashboard endpoint."""
    
    def test_dashboard_stats_requires_authentication(self):
        """Test that endpoint requires authentication."""
        client = TestClient(app)
        response = client.get("/dashboard/stats")
        
        # Should return 403 (no auth header) or 401 (invalid token)
        assert response.status_code in [401, 403]
    
    def test_dashboard_stats_with_invalid_token(self):
        """Test endpoint with invalid authentication token."""
        client = TestClient(app)
        headers = {"Authorization": "Bearer invalid_token"}
        response = client.get("/dashboard/stats", headers=headers)
        
        assert response.status_code == 401


class TestDashboardStatsCaching:
    """Test caching behavior of dashboard endpoint."""
    
    @patch('src.api.dashboard.CacheService')
    def test_dashboard_stats_uses_cache(
        self, mock_cache_service_class, authenticated_client
    ):
        """Test that dashboard stats uses caching."""
        mock_cache_instance = Mock()
        mock_cache_service_class.return_value = mock_cache_instance
        
        # Mock cache hit
        cached_stats = DashboardStats(
            total_models=10,
            active_optimizations=5,
            completed_optimizations=100,
            failed_optimizations=5,
            average_size_reduction=30.0,
            average_speed_improvement=20.0,
            total_sessions=110,
            last_updated=datetime.now()
        )
        mock_cache_instance.get_or_set.return_value = cached_stats
        
        response = authenticated_client.get("/dashboard/stats")
        
        assert response.status_code == 200
        # Verify cache was called
        mock_cache_instance.get_or_set.assert_called_once()


class TestComputeDashboardStats:
    """Test the _compute_dashboard_stats helper function."""
    
    def test_compute_stats_with_all_data(
        self, mock_user, mock_optimization_manager, mock_model_store, mock_memory_manager
    ):
        """Test computing stats with complete data."""
        stats = _compute_dashboard_stats(
            mock_user,
            mock_optimization_manager,
            mock_model_store,
            mock_memory_manager
        )
        
        assert isinstance(stats, DashboardStats)
        assert stats.total_models == 3
        assert stats.active_optimizations == 2
        assert stats.completed_optimizations == 42
        assert stats.failed_optimizations == 3
        assert stats.total_sessions == 50
        assert stats.average_size_reduction > 0
        assert stats.average_speed_improvement > 0
    
    def test_compute_stats_with_empty_data(self, mock_user):
        """Test computing stats with no data."""
        empty_model_store = Mock(spec=ModelStore)
        empty_model_store.list_models.return_value = []
        
        empty_opt_manager = Mock(spec=OptimizationManager)
        empty_opt_manager.get_active_sessions.return_value = []
        
        empty_mem_manager = Mock(spec=MemoryManager)
        empty_mem_manager.get_session_statistics.return_value = {
            "total_sessions": 0,
            "status_distribution": {}
        }
        empty_mem_manager.list_sessions.return_value = []
        
        stats = _compute_dashboard_stats(
            mock_user,
            empty_opt_manager,
            empty_model_store,
            empty_mem_manager
        )
        
        assert isinstance(stats, DashboardStats)
        assert stats.total_models == 0
        assert stats.active_optimizations == 0
        assert stats.completed_optimizations == 0
        assert stats.failed_optimizations == 0
        assert stats.total_sessions == 0
        assert stats.average_size_reduction == 0.0
        assert stats.average_speed_improvement == 0.0
    
    def test_compute_stats_handles_missing_results(
        self, mock_user, mock_optimization_manager, mock_model_store, mock_memory_manager
    ):
        """Test computing stats when some sessions lack results."""
        # Mock sessions with missing or incomplete results
        mock_session_with_results = Mock()
        mock_session_with_results.results = Mock()
        mock_session_with_results.results.size_reduction_percent = 30.0
        mock_session_with_results.results.speed_improvement_percent = 15.0
        
        mock_session_no_results = Mock()
        mock_session_no_results.results = None
        
        mock_memory_manager.retrieve_session.side_effect = [
            mock_session_with_results,
            mock_session_no_results,
            mock_session_with_results
        ]
        
        stats = _compute_dashboard_stats(
            mock_user,
            mock_optimization_manager,
            mock_model_store,
            mock_memory_manager
        )
        
        # Should still compute averages from available data
        assert isinstance(stats, DashboardStats)
        assert stats.average_size_reduction > 0
        assert stats.average_speed_improvement > 0
    
    def test_compute_stats_handles_partial_service_failures(
        self, mock_user, mock_optimization_manager, mock_model_store, mock_memory_manager
    ):
        """Test that partial service failures don't crash the computation."""
        # Make model store fail
        mock_model_store.list_models.side_effect = Exception("Connection error")
        
        # Should still compute other stats
        stats = _compute_dashboard_stats(
            mock_user,
            mock_optimization_manager,
            mock_model_store,
            mock_memory_manager
        )
        
        assert isinstance(stats, DashboardStats)
        assert stats.total_models == 0  # Failed, so defaults to 0
        assert stats.active_optimizations == 2  # This should still work
        assert stats.completed_optimizations == 42  # This should still work


class TestDashboardStatsDataTypes:
    """Test data type validation for dashboard stats."""
    
    def test_all_counts_are_non_negative(self, authenticated_client):
        """Test that all count fields are non-negative integers."""
        response = authenticated_client.get("/dashboard/stats")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["total_models"] >= 0
        assert data["active_optimizations"] >= 0
        assert data["completed_optimizations"] >= 0
        assert data["failed_optimizations"] >= 0
        assert data["total_sessions"] >= 0
    
    def test_averages_are_numeric(self, authenticated_client):
        """Test that average fields are numeric."""
        response = authenticated_client.get("/dashboard/stats")
        
        assert response.status_code == 200
        data = response.json()
        
        assert isinstance(data["average_size_reduction"], (int, float))
        assert isinstance(data["average_speed_improvement"], (int, float))
        assert data["average_size_reduction"] >= 0
        assert data["average_speed_improvement"] >= 0
    
    def test_timestamp_is_valid_iso_format(self, authenticated_client):
        """Test that last_updated is a valid ISO timestamp."""
        response = authenticated_client.get("/dashboard/stats")
        
        assert response.status_code == 200
        data = response.json()
        
        # Should be able to parse as datetime
        timestamp_str = data["last_updated"]
        parsed_timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        assert isinstance(parsed_timestamp, datetime)


class TestDashboardStatsEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_dashboard_stats_with_large_numbers(self, mock_user):
        """Test dashboard stats with large numbers of models and sessions."""
        from src.api.dependencies import get_current_user
        
        # Create mock with large numbers
        mock_model_store = Mock(spec=ModelStore)
        mock_model_store.list_models.return_value = [{"id": f"model{i}"} for i in range(10000)]
        
        mock_opt_manager = Mock(spec=OptimizationManager)
        mock_opt_manager.get_active_sessions.return_value = []
        
        mock_memory_manager = Mock(spec=MemoryManager)
        mock_memory_manager.get_session_statistics.return_value = {
            "total_sessions": 50000,
            "status_distribution": {
                "completed": 45000,
                "failed": 2000,
                "running": 1000,
                "cancelled": 2000
            }
        }
        mock_memory_manager.list_sessions.return_value = []
        
        mock_integrator = Mock()
        mock_integrator.get_model_store.return_value = mock_model_store
        mock_integrator.get_memory_manager.return_value = mock_memory_manager
        
        app.dependency_overrides[get_current_user] = lambda: mock_user
        app.state.optimization_manager = mock_opt_manager
        app.state.platform_integrator = mock_integrator
        
        try:
            client = TestClient(app)
            response = client.get("/dashboard/stats")
            
            assert response.status_code == 200
            data = response.json()
            assert data["total_models"] == 10000
            assert data["total_sessions"] == 50000
            assert data["completed_optimizations"] == 45000
        finally:
            app.dependency_overrides.clear()
            if hasattr(app.state, 'optimization_manager'):
                delattr(app.state, 'optimization_manager')
            if hasattr(app.state, 'platform_integrator'):
                delattr(app.state, 'platform_integrator')
    
    def test_dashboard_stats_with_extreme_averages(
        self, authenticated_client, mock_platform_integrator
    ):
        """Test dashboard stats with extreme average values."""
        mock_memory_manager = Mock(spec=MemoryManager)
        mock_memory_manager.get_session_statistics.return_value = {
            "total_sessions": 2,
            "status_distribution": {"completed": 2}
        }
        
        # Create sessions with extreme values
        mock_session_1 = Mock()
        mock_session_1.results = Mock()
        mock_session_1.results.size_reduction_percent = 99.9
        mock_session_1.results.speed_improvement_percent = 500.0
        
        mock_session_2 = Mock()
        mock_session_2.results = Mock()
        mock_session_2.results.size_reduction_percent = 0.1
        mock_session_2.results.speed_improvement_percent = 1.0
        
        mock_memory_manager.list_sessions.return_value = [
            {"id": "session1"},
            {"id": "session2"}
        ]
        mock_memory_manager.retrieve_session.side_effect = [mock_session_1, mock_session_2]
        
        mock_platform_integrator.get_memory_manager.return_value = mock_memory_manager
        
        response = authenticated_client.get("/dashboard/stats")
        
        assert response.status_code == 200
        data = response.json()
        # Averages should be computed correctly even with extreme values
        assert 0 <= data["average_size_reduction"] <= 100
        assert data["average_speed_improvement"] > 0

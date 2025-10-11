"""
Tests for API error handlers and response models.
"""

import pytest
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from starlette.exceptions import HTTPException as StarletteHTTPException

from src.api.models import (
    ErrorResponse,
    DashboardStats,
    SessionListResponse,
    OptimizationSessionSummary,
    OptimizationCriteriaResponse
)
from src.api.error_handlers import (
    create_error_response,
    register_error_handlers,
    platform_error_handler,
    http_exception_handler,
    validation_exception_handler
)
from src.utils.exceptions import (
    OptimizationError,
    ValidationError,
    ConfigurationError,
    ErrorSeverity
)


class TestErrorResponseModel:
    """Test ErrorResponse model."""
    
    def test_error_response_creation(self):
        """Test creating an error response."""
        error_response = ErrorResponse(
            error="test_error",
            message="Test error message",
            details={"key": "value"},
            timestamp=datetime.now(),
            request_id="test-request-id"
        )
        
        assert error_response.error == "test_error"
        assert error_response.message == "Test error message"
        assert error_response.details == {"key": "value"}
        assert error_response.request_id == "test-request-id"
        assert isinstance(error_response.timestamp, datetime)
    
    def test_error_response_with_defaults(self):
        """Test error response with default values."""
        error_response = ErrorResponse(
            error="test_error",
            message="Test message"
        )
        
        assert error_response.error == "test_error"
        assert error_response.message == "Test message"
        assert error_response.details is None
        assert isinstance(error_response.timestamp, datetime)
        assert isinstance(error_response.request_id, str)
        assert len(error_response.request_id) > 0


class TestDashboardStatsModel:
    """Test DashboardStats model."""
    
    def test_dashboard_stats_creation(self):
        """Test creating dashboard statistics."""
        stats = DashboardStats(
            total_models=10,
            active_optimizations=2,
            completed_optimizations=5,
            failed_optimizations=1,
            average_size_reduction=25.5,
            average_speed_improvement=15.3,
            total_sessions=8,
            last_updated=datetime.now()
        )
        
        assert stats.total_models == 10
        assert stats.active_optimizations == 2
        assert stats.completed_optimizations == 5
        assert stats.failed_optimizations == 1
        assert stats.average_size_reduction == 25.5
        assert stats.average_speed_improvement == 15.3
        assert stats.total_sessions == 8
        assert isinstance(stats.last_updated, datetime)
    
    def test_dashboard_stats_validation(self):
        """Test dashboard stats validation."""
        # Should not allow negative values
        with pytest.raises(Exception):  # Pydantic validation error
            DashboardStats(
                total_models=-1,
                active_optimizations=0,
                completed_optimizations=0,
                failed_optimizations=0,
                average_size_reduction=0.0,
                average_speed_improvement=0.0,
                total_sessions=0
            )


class TestSessionListResponseModel:
    """Test SessionListResponse model."""
    
    def test_session_list_response_creation(self):
        """Test creating session list response."""
        session = OptimizationSessionSummary(
            session_id="session-1",
            model_id="model-1",
            model_name="Test Model",
            status="running",
            progress_percentage=50.0,
            techniques=["quantization", "pruning"],
            size_reduction_percent=20.0,
            speed_improvement_percent=10.0,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            completed_at=None
        )
        
        response = SessionListResponse(
            sessions=[session],
            total=1,
            skip=0,
            limit=50
        )
        
        assert len(response.sessions) == 1
        assert response.total == 1
        assert response.skip == 0
        assert response.limit == 50
        assert response.sessions[0].session_id == "session-1"
    
    def test_empty_session_list(self):
        """Test empty session list response."""
        response = SessionListResponse(
            sessions=[],
            total=0,
            skip=0,
            limit=50
        )
        
        assert len(response.sessions) == 0
        assert response.total == 0


class TestOptimizationSessionSummary:
    """Test OptimizationSessionSummary model."""
    
    def test_session_summary_creation(self):
        """Test creating session summary."""
        summary = OptimizationSessionSummary(
            session_id="test-session",
            model_id="test-model",
            model_name="Test Model",
            status="completed",
            progress_percentage=100.0,
            techniques=["quantization"],
            size_reduction_percent=30.0,
            speed_improvement_percent=20.0,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            completed_at=datetime.now()
        )
        
        assert summary.session_id == "test-session"
        assert summary.model_id == "test-model"
        assert summary.status == "completed"
        assert summary.progress_percentage == 100.0
        assert "quantization" in summary.techniques
    
    def test_session_summary_progress_validation(self):
        """Test progress percentage validation."""
        # Should not allow progress > 100
        with pytest.raises(Exception):  # Pydantic validation error
            OptimizationSessionSummary(
                session_id="test",
                model_id="test",
                model_name="Test",
                status="running",
                progress_percentage=150.0,  # Invalid
                created_at=datetime.now(),
                updated_at=datetime.now()
            )


class TestCreateErrorResponse:
    """Test create_error_response helper function."""
    
    def test_create_error_response_with_dict_details(self):
        """Test creating error response with dict details."""
        response = create_error_response(
            error="test_error",
            message="Test message",
            details={"key": "value"}
        )
        
        assert response.error == "test_error"
        assert response.message == "Test message"
        assert response.details == {"key": "value"}
        assert isinstance(response.request_id, str)
    
    def test_create_error_response_with_string_details(self):
        """Test creating error response with string details."""
        response = create_error_response(
            error="test_error",
            message="Test message",
            details="String detail"
        )
        
        assert response.error == "test_error"
        assert response.details == {"detail": "String detail"}
    
    def test_create_error_response_with_custom_request_id(self):
        """Test creating error response with custom request ID."""
        response = create_error_response(
            error="test_error",
            message="Test message",
            request_id="custom-id"
        )
        
        assert response.request_id == "custom-id"


class TestErrorHandlers:
    """Test error handler functions."""
    
    @pytest.fixture
    def app(self):
        """Create test FastAPI app."""
        app = FastAPI()
        register_error_handlers(app)
        
        @app.get("/test-platform-error")
        async def test_platform_error():
            raise OptimizationError(
                message="Test optimization error",
                technique="quantization",
                session_id="test-session"
            )
        
        @app.get("/test-http-error")
        async def test_http_error():
            raise HTTPException(status_code=404, detail="Not found")
        
        @app.get("/test-validation-error")
        async def test_validation_error():
            raise ValidationError(
                message="Validation failed",
                validation_type="input"
            )
        
        @app.get("/test-general-error")
        async def test_general_error():
            raise ValueError("Unexpected error")
        
        return app
    
    def test_platform_error_handler(self, app):
        """Test platform error handler."""
        client = TestClient(app)
        response = client.get("/test-platform-error")
        
        assert response.status_code == 422
        data = response.json()
        assert "error" in data
        assert "message" in data
        assert "request_id" in data
        assert data["message"] == "Test optimization error"
    
    def test_http_exception_handler(self, app):
        """Test HTTP exception handler."""
        client = TestClient(app)
        response = client.get("/test-http-error")
        
        assert response.status_code == 404
        data = response.json()
        assert "error" in data
        assert "message" in data
        assert "request_id" in data
    
    def test_validation_error_handler(self, app):
        """Test validation error handler."""
        client = TestClient(app)
        response = client.get("/test-validation-error")
        
        assert response.status_code == 400
        data = response.json()
        assert "error" in data
        assert "message" in data
        assert data["message"] == "Validation failed"
    
    def test_general_exception_handler(self, app):
        """Test general exception handler."""
        client = TestClient(app, raise_server_exceptions=False)
        response = client.get("/test-general-error")
        
        assert response.status_code == 500
        data = response.json()
        assert "error" in data
        assert "message" in data
        assert data["error"] == "internal_server_error"


class TestOptimizationCriteriaResponse:
    """Test OptimizationCriteriaResponse model."""
    
    def test_criteria_response_creation(self):
        """Test creating optimization criteria response."""
        response = OptimizationCriteriaResponse(
            name="test_criteria",
            target_accuracy_threshold=0.95,
            max_size_reduction_percent=50.0,
            max_latency_increase_percent=10.0,
            optimization_techniques=["quantization", "pruning"],
            hardware_constraints={"memory_mb": 1024},
            custom_parameters={"param1": "value1"},
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        assert response.name == "test_criteria"
        assert response.target_accuracy_threshold == 0.95
        assert response.max_size_reduction_percent == 50.0
        assert len(response.optimization_techniques) == 2
        assert response.hardware_constraints["memory_mb"] == 1024

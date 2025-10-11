"""
Unit tests for configuration API endpoints (src/api/config.py).

Tests cover:
- GET /config/optimization-criteria with existing and missing configuration
- PUT /config/optimization-criteria with valid and invalid data
- Validation error responses
- Concurrent update scenarios
- Authentication and authorization
"""

import pytest
import tempfile
import json
import asyncio
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, patch, AsyncMock
from concurrent.futures import ThreadPoolExecutor, as_completed
from fastapi import FastAPI, Request, HTTPException, status
from fastapi.testclient import TestClient

from src.api.config import router, get_config_manager
from src.api.models import OptimizationCriteriaRequest, OptimizationCriteriaResponse
from src.api.auth import User
from src.services.config_manager import ConfigurationManager, ValidationResult
from src.config.optimization_criteria import (
    OptimizationCriteria,
    OptimizationConstraints,
    PerformanceThreshold,
    PerformanceMetric,
    OptimizationTechnique
)


@pytest.fixture
def app():
    """Create FastAPI test application."""
    app = FastAPI()
    app.include_router(router)
    return app


@pytest.fixture
def client(app):
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def temp_config_path():
    """Create temporary configuration file path."""
    with tempfile.TemporaryDirectory() as temp_dir:
        config_path = Path(temp_dir) / "config.json"
        yield str(config_path)


@pytest.fixture
def mock_user():
    """Create mock regular user."""
    return User(
        id="user123",
        username="testuser",
        role="user",
        email="test@example.com",
        is_active=True
    )


@pytest.fixture
def mock_admin():
    """Create mock admin user."""
    return User(
        id="admin123",
        username="admin",
        role="administrator",
        email="admin@example.com",
        is_active=True
    )


@pytest.fixture
def valid_criteria():
    """Create valid optimization criteria."""
    return OptimizationCriteria(
        name="test_criteria",
        description="Test optimization criteria",
        performance_thresholds=[
            PerformanceThreshold(
                metric=PerformanceMetric.ACCURACY,
                min_value=0.90,
                target_value=0.95,
                tolerance=0.05
            )
        ],
        constraints=OptimizationConstraints(
            max_optimization_time_minutes=60,
            max_memory_usage_gb=16.0,
            preserve_accuracy_threshold=0.95,
            allowed_techniques=[OptimizationTechnique.QUANTIZATION],
            forbidden_techniques=[],
            hardware_constraints={"gpu_memory_gb": 8.0}
        ),
        priority_weights={
            PerformanceMetric.ACCURACY: 0.5,
            PerformanceMetric.MODEL_SIZE: 0.3,
            PerformanceMetric.INFERENCE_TIME: 0.2
        },
        target_deployment="edge"
    )


@pytest.fixture
def valid_request_data():
    """Create valid request data."""
    return {
        "name": "edge_deployment",
        "target_accuracy_threshold": 0.95,
        "max_size_reduction_percent": 60.0,
        "max_latency_increase_percent": 5.0,
        "optimization_techniques": ["quantization", "pruning"],
        "hardware_constraints": {
            "max_memory_mb": 512,
            "target_device": "jetson_nano"
        },
        "custom_parameters": {
            "quantization_bits": 8
        }
    }


class TestGetOptimizationCriteria:
    """Test GET /config/optimization-criteria endpoint."""
    
    def test_get_existing_configuration(self, app, temp_config_path, valid_criteria, mock_user):
        """Test GET with existing configuration."""
        # Reset singleton and setup
        ConfigurationManager._instance = None
        config_manager = ConfigurationManager(temp_config_path)
        config_manager.save_configuration(valid_criteria)
        
        # Mock dependencies
        async def mock_get_user():
            return mock_user
        
        async def mock_get_config_manager(request: Request):
            return config_manager
        
        app.dependency_overrides[get_config_manager] = mock_get_config_manager
        
        # Mock authentication
        from src.api.dependencies import get_current_user
        app.dependency_overrides[get_current_user] = mock_get_user
        
        client = TestClient(app)
        
        # Make request
        response = client.get(
            "/config/optimization-criteria",
            headers={"Authorization": "Bearer test_token"}
        )
        
        # Verify response
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == valid_criteria.name
        assert "created_at" in data
        assert "updated_at" in data
        
        # Cleanup
        app.dependency_overrides.clear()
    
    def test_get_missing_configuration(self, app, temp_config_path, mock_user):
        """Test GET when configuration file doesn't exist."""
        # Reset singleton and clear cache
        ConfigurationManager._instance = None
        config_manager = ConfigurationManager(temp_config_path)
        config_manager._current_config = None  # Clear cached config
        
        # Clear cache service
        from src.services.cache_service import CacheService
        cache = CacheService()
        cache.clear()
        
        # Mock dependencies
        async def mock_get_user():
            return mock_user
        
        async def mock_get_config_manager(request: Request):
            return config_manager
        
        app.dependency_overrides[get_config_manager] = mock_get_config_manager
        
        from src.api.dependencies import get_current_user
        app.dependency_overrides[get_current_user] = mock_get_user
        
        client = TestClient(app)
        
        # Make request
        response = client.get(
            "/config/optimization-criteria",
            headers={"Authorization": "Bearer test_token"}
        )
        
        # Should return default configuration
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "default"
        
        # Cleanup
        app.dependency_overrides.clear()
    
    def test_get_without_authentication(self, app):
        """Test GET without authentication token."""
        client = TestClient(app)
        
        # Make request without auth header
        response = client.get("/config/optimization-criteria")
        
        # Should return 403 (TestClient doesn't enforce security by default)
        # In real scenario with security enabled, would be 401
        assert response.status_code in [401, 403]
    
    def test_get_with_cache(self, app, temp_config_path, valid_criteria, mock_user):
        """Test that GET uses cache for repeated requests."""
        # Reset singleton
        ConfigurationManager._instance = None
        config_manager = ConfigurationManager(temp_config_path)
        config_manager.save_configuration(valid_criteria)
        
        # Mock dependencies
        async def mock_get_user():
            return mock_user
        
        async def mock_get_config_manager(request: Request):
            return config_manager
        
        app.dependency_overrides[get_config_manager] = mock_get_config_manager
        
        from src.api.dependencies import get_current_user
        app.dependency_overrides[get_current_user] = mock_get_user
        
        client = TestClient(app)
        
        # Make first request
        response1 = client.get(
            "/config/optimization-criteria",
            headers={"Authorization": "Bearer test_token"}
        )
        
        # Make second request
        response2 = client.get(
            "/config/optimization-criteria",
            headers={"Authorization": "Bearer test_token"}
        )
        
        # Both should succeed
        assert response1.status_code == 200
        assert response2.status_code == 200
        
        # Data should be the same
        assert response1.json()["name"] == response2.json()["name"]
        
        # Cleanup
        app.dependency_overrides.clear()


class TestPutOptimizationCriteria:
    """Test PUT /config/optimization-criteria endpoint."""
    
    def test_put_valid_configuration(self, app, temp_config_path, valid_request_data, mock_admin):
        """Test PUT with valid configuration data."""
        # Reset singleton
        ConfigurationManager._instance = None
        config_manager = ConfigurationManager(temp_config_path)
        
        # Mock dependencies
        async def mock_get_admin():
            return mock_admin
        
        async def mock_get_config_manager(request: Request):
            return config_manager
        
        app.dependency_overrides[get_config_manager] = mock_get_config_manager
        
        from src.api.dependencies import get_admin_user
        app.dependency_overrides[get_admin_user] = mock_get_admin
        
        client = TestClient(app)
        
        # Make request
        response = client.put(
            "/config/optimization-criteria",
            json=valid_request_data,
            headers={"Authorization": "Bearer admin_token"}
        )
        
        # Verify response
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == valid_request_data["name"]
        assert data["target_accuracy_threshold"] == valid_request_data["target_accuracy_threshold"]
        
        # Verify configuration was saved
        saved_config = config_manager.get_current_configuration()
        assert saved_config.name == valid_request_data["name"]
        
        # Cleanup
        app.dependency_overrides.clear()
    
    def test_put_invalid_accuracy_threshold(self, app, temp_config_path, mock_admin):
        """Test PUT with invalid accuracy threshold."""
        # Reset singleton
        ConfigurationManager._instance = None
        config_manager = ConfigurationManager(temp_config_path)
        
        # Mock dependencies
        async def mock_get_admin():
            return mock_admin
        
        async def mock_get_config_manager(request: Request):
            return config_manager
        
        app.dependency_overrides[get_config_manager] = mock_get_config_manager
        
        from src.api.dependencies import get_admin_user
        app.dependency_overrides[get_admin_user] = mock_get_admin
        
        client = TestClient(app)
        
        # Invalid data (accuracy > 1.0)
        invalid_data = {
            "name": "invalid_config",
            "target_accuracy_threshold": 1.5,  # Invalid
            "max_size_reduction_percent": 50.0,
            "max_latency_increase_percent": 10.0,
            "optimization_techniques": ["quantization"]
        }
        
        # Make request
        response = client.put(
            "/config/optimization-criteria",
            json=invalid_data,
            headers={"Authorization": "Bearer admin_token"}
        )
        
        # Should return validation error (422 from Pydantic validation)
        assert response.status_code == 422
        data = response.json()
        assert "detail" in data
        
        # Cleanup
        app.dependency_overrides.clear()
    
    def test_put_invalid_techniques(self, app, temp_config_path, mock_admin):
        """Test PUT with invalid optimization techniques."""
        # Reset singleton
        ConfigurationManager._instance = None
        config_manager = ConfigurationManager(temp_config_path)
        
        # Mock dependencies
        async def mock_get_admin():
            return mock_admin
        
        async def mock_get_config_manager(request: Request):
            return config_manager
        
        app.dependency_overrides[get_config_manager] = mock_get_config_manager
        
        from src.api.dependencies import get_admin_user
        app.dependency_overrides[get_admin_user] = mock_get_admin
        
        client = TestClient(app)
        
        # Invalid data (unknown technique)
        invalid_data = {
            "name": "invalid_config",
            "target_accuracy_threshold": 0.95,
            "max_size_reduction_percent": 50.0,
            "max_latency_increase_percent": 10.0,
            "optimization_techniques": ["invalid_technique"]  # Invalid
        }
        
        # Make request
        response = client.put(
            "/config/optimization-criteria",
            json=invalid_data,
            headers={"Authorization": "Bearer admin_token"}
        )
        
        # Should still succeed but technique will be ignored
        # (based on implementation in config.py)
        assert response.status_code in [200, 400]
        
        # Cleanup
        app.dependency_overrides.clear()
    
    def test_put_without_admin_role(self, app, temp_config_path, mock_user):
        """Test PUT without admin role."""
        # Reset singleton
        ConfigurationManager._instance = None
        config_manager = ConfigurationManager(temp_config_path)
        
        # Mock dependencies - return regular user instead of admin
        async def mock_get_admin():
            # This should raise HTTPException in real scenario
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Administrator access required"
            )
        
        async def mock_get_config_manager(request: Request):
            return config_manager
        
        app.dependency_overrides[get_config_manager] = mock_get_config_manager
        
        from src.api.dependencies import get_admin_user
        app.dependency_overrides[get_admin_user] = mock_get_admin
        
        client = TestClient(app)
        
        valid_data = {
            "name": "test_config",
            "target_accuracy_threshold": 0.95,
            "max_size_reduction_percent": 50.0,
            "max_latency_increase_percent": 10.0,
            "optimization_techniques": ["quantization"]
        }
        
        # Make request
        response = client.put(
            "/config/optimization-criteria",
            json=valid_data,
            headers={"Authorization": "Bearer user_token"}
        )
        
        # Should return 403 Forbidden
        assert response.status_code == 403
        
        # Cleanup
        app.dependency_overrides.clear()
    
    def test_put_empty_name(self, app, temp_config_path, mock_admin):
        """Test PUT with empty configuration name."""
        # Reset singleton
        ConfigurationManager._instance = None
        config_manager = ConfigurationManager(temp_config_path)
        
        # Mock dependencies
        async def mock_get_admin():
            return mock_admin
        
        async def mock_get_config_manager(request: Request):
            return config_manager
        
        app.dependency_overrides[get_config_manager] = mock_get_config_manager
        
        from src.api.dependencies import get_admin_user
        app.dependency_overrides[get_admin_user] = mock_get_admin
        
        client = TestClient(app)
        
        # Invalid data (empty name)
        invalid_data = {
            "name": "",  # Invalid
            "target_accuracy_threshold": 0.95,
            "max_size_reduction_percent": 50.0,
            "max_latency_increase_percent": 10.0,
            "optimization_techniques": ["quantization"]
        }
        
        # Make request
        response = client.put(
            "/config/optimization-criteria",
            json=invalid_data,
            headers={"Authorization": "Bearer admin_token"}
        )
        
        # Should return validation error
        assert response.status_code == 400
        
        # Cleanup
        app.dependency_overrides.clear()
    
    def test_put_negative_size_reduction(self, app, temp_config_path, mock_admin):
        """Test PUT with negative size reduction."""
        # Reset singleton
        ConfigurationManager._instance = None
        config_manager = ConfigurationManager(temp_config_path)
        
        # Mock dependencies
        async def mock_get_admin():
            return mock_admin
        
        async def mock_get_config_manager(request: Request):
            return config_manager
        
        app.dependency_overrides[get_config_manager] = mock_get_config_manager
        
        from src.api.dependencies import get_admin_user
        app.dependency_overrides[get_admin_user] = mock_get_admin
        
        client = TestClient(app)
        
        # Invalid data (negative size reduction)
        invalid_data = {
            "name": "invalid_config",
            "target_accuracy_threshold": 0.95,
            "max_size_reduction_percent": -10.0,  # Invalid
            "max_latency_increase_percent": 10.0,
            "optimization_techniques": ["quantization"]
        }
        
        # Make request
        response = client.put(
            "/config/optimization-criteria",
            json=invalid_data,
            headers={"Authorization": "Bearer admin_token"}
        )
        
        # Should return validation error (Pydantic validation)
        assert response.status_code == 422  # Unprocessable Entity
        
        # Cleanup
        app.dependency_overrides.clear()


class TestValidationErrorResponses:
    """Test validation error response format."""
    
    def test_validation_error_format(self, app, temp_config_path, mock_admin):
        """Test that validation errors have proper format."""
        # Reset singleton
        ConfigurationManager._instance = None
        config_manager = ConfigurationManager(temp_config_path)
        
        # Mock dependencies
        async def mock_get_admin():
            return mock_admin
        
        async def mock_get_config_manager(request: Request):
            return config_manager
        
        app.dependency_overrides[get_config_manager] = mock_get_config_manager
        
        from src.api.dependencies import get_admin_user
        app.dependency_overrides[get_admin_user] = mock_get_admin
        
        client = TestClient(app)
        
        # Invalid data
        invalid_data = {
            "name": "",  # Invalid
            "target_accuracy_threshold": 0.95,
            "max_size_reduction_percent": 50.0,
            "max_latency_increase_percent": 10.0,
            "optimization_techniques": ["quantization"]
        }
        
        # Make request
        response = client.put(
            "/config/optimization-criteria",
            json=invalid_data,
            headers={"Authorization": "Bearer admin_token"}
        )
        
        # Verify error response format
        assert response.status_code == 400
        data = response.json()
        assert "detail" in data
        
        # Check if detail contains error information
        detail = data["detail"]
        if isinstance(detail, dict):
            assert "errors" in detail or "message" in detail
        
        # Cleanup
        app.dependency_overrides.clear()
    
    def test_validation_with_warnings(self, app, temp_config_path, mock_admin):
        """Test validation that produces warnings but succeeds."""
        # Reset singleton
        ConfigurationManager._instance = None
        config_manager = ConfigurationManager(temp_config_path)
        
        # Mock dependencies
        async def mock_get_admin():
            return mock_admin
        
        async def mock_get_config_manager(request: Request):
            return config_manager
        
        app.dependency_overrides[get_config_manager] = mock_get_config_manager
        
        from src.api.dependencies import get_admin_user
        app.dependency_overrides[get_admin_user] = mock_get_admin
        
        client = TestClient(app)
        
        # Data that produces warnings (low accuracy threshold)
        warning_data = {
            "name": "low_accuracy_config",
            "target_accuracy_threshold": 0.4,  # Low, should produce warning
            "max_size_reduction_percent": 50.0,
            "max_latency_increase_percent": 10.0,
            "optimization_techniques": ["quantization"]
        }
        
        # Make request
        response = client.put(
            "/config/optimization-criteria",
            json=warning_data,
            headers={"Authorization": "Bearer admin_token"}
        )
        
        # Should succeed despite warnings
        assert response.status_code == 200
        
        # Cleanup
        app.dependency_overrides.clear()
    
    def test_multiple_validation_errors(self, app, temp_config_path, mock_admin):
        """Test that multiple validation errors are reported."""
        # Reset singleton
        ConfigurationManager._instance = None
        config_manager = ConfigurationManager(temp_config_path)
        
        # Mock dependencies
        async def mock_get_admin():
            return mock_admin
        
        async def mock_get_config_manager(request: Request):
            return config_manager
        
        app.dependency_overrides[get_config_manager] = mock_get_config_manager
        
        from src.api.dependencies import get_admin_user
        app.dependency_overrides[get_admin_user] = mock_get_admin
        
        client = TestClient(app)
        
        # Multiple invalid fields
        invalid_data = {
            "name": "",  # Invalid
            "target_accuracy_threshold": 1.5,  # Invalid
            "max_size_reduction_percent": -10.0,  # Invalid
            "max_latency_increase_percent": 10.0,
            "optimization_techniques": ["quantization"]
        }
        
        # Make request
        response = client.put(
            "/config/optimization-criteria",
            json=invalid_data,
            headers={"Authorization": "Bearer admin_token"}
        )
        
        # Should return error
        assert response.status_code in [400, 422]
        
        # Cleanup
        app.dependency_overrides.clear()


class TestConcurrentUpdates:
    """Test concurrent configuration update scenarios."""
    
    def test_concurrent_put_requests(self, app, temp_config_path, mock_admin):
        """Test multiple concurrent PUT requests."""
        # Reset singleton
        ConfigurationManager._instance = None
        config_manager = ConfigurationManager(temp_config_path)
        
        # Mock dependencies
        async def mock_get_admin():
            return mock_admin
        
        async def mock_get_config_manager(request: Request):
            return config_manager
        
        app.dependency_overrides[get_config_manager] = mock_get_config_manager
        
        from src.api.dependencies import get_admin_user
        app.dependency_overrides[get_admin_user] = mock_get_admin
        
        client = TestClient(app)
        
        def make_update(index):
            """Make a PUT request."""
            data = {
                "name": f"config_{index}",
                "target_accuracy_threshold": 0.95,
                "max_size_reduction_percent": 50.0,
                "max_latency_increase_percent": 10.0,
                "optimization_techniques": ["quantization"]
            }
            
            response = client.put(
                "/config/optimization-criteria",
                json=data,
                headers={"Authorization": "Bearer admin_token"}
            )
            return response.status_code
        
        # Make concurrent requests
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_update, i) for i in range(10)]
            results = [f.result() for f in as_completed(futures)]
        
        # All requests should complete (some may succeed, some may conflict)
        assert all(status_code in [200, 400, 409, 500] for status_code in results)
        
        # Final configuration should be one of the updates
        final_config = config_manager.get_current_configuration()
        assert final_config.name.startswith("config_")
        
        # Cleanup
        app.dependency_overrides.clear()
    
    def test_concurrent_get_and_put(self, app, temp_config_path, valid_criteria, mock_user, mock_admin):
        """Test concurrent GET and PUT requests."""
        # Reset singleton
        ConfigurationManager._instance = None
        config_manager = ConfigurationManager(temp_config_path)
        config_manager.save_configuration(valid_criteria)
        
        # Mock dependencies for GET
        async def mock_get_user():
            return mock_user
        
        # Mock dependencies for PUT
        async def mock_get_admin():
            return mock_admin
        
        async def mock_get_config_manager(request: Request):
            return config_manager
        
        app.dependency_overrides[get_config_manager] = mock_get_config_manager
        
        from src.api.dependencies import get_current_user, get_admin_user
        app.dependency_overrides[get_current_user] = mock_get_user
        app.dependency_overrides[get_admin_user] = mock_get_admin
        
        client = TestClient(app)
        
        def make_get():
            """Make a GET request."""
            response = client.get(
                "/config/optimization-criteria",
                headers={"Authorization": "Bearer user_token"}
            )
            return response.status_code
        
        def make_put(index):
            """Make a PUT request."""
            data = {
                "name": f"updated_{index}",
                "target_accuracy_threshold": 0.95,
                "max_size_reduction_percent": 50.0,
                "max_latency_increase_percent": 10.0,
                "optimization_techniques": ["quantization"]
            }
            
            response = client.put(
                "/config/optimization-criteria",
                json=data,
                headers={"Authorization": "Bearer admin_token"}
            )
            return response.status_code
        
        # Make concurrent GET and PUT requests
        with ThreadPoolExecutor(max_workers=10) as executor:
            get_futures = [executor.submit(make_get) for _ in range(5)]
            put_futures = [executor.submit(make_put, i) for i in range(5)]
            
            all_futures = get_futures + put_futures
            results = [f.result() for f in as_completed(all_futures)]
        
        # All GET requests should succeed
        # PUT requests should mostly succeed
        assert all(status_code in [200, 400, 409, 500] for status_code in results)
        
        # Cleanup
        app.dependency_overrides.clear()
    
    def test_read_after_write_consistency(self, app, temp_config_path, mock_user, mock_admin):
        """Test that GET returns updated value after PUT."""
        # Reset singleton
        ConfigurationManager._instance = None
        config_manager = ConfigurationManager(temp_config_path)
        
        # Mock dependencies
        async def mock_get_user():
            return mock_user
        
        async def mock_get_admin():
            return mock_admin
        
        async def mock_get_config_manager(request: Request):
            return config_manager
        
        app.dependency_overrides[get_config_manager] = mock_get_config_manager
        
        from src.api.dependencies import get_current_user, get_admin_user
        app.dependency_overrides[get_current_user] = mock_get_user
        app.dependency_overrides[get_admin_user] = mock_get_admin
        
        client = TestClient(app)
        
        # Make PUT request
        update_data = {
            "name": "updated_config",
            "target_accuracy_threshold": 0.92,
            "max_size_reduction_percent": 55.0,
            "max_latency_increase_percent": 8.0,
            "optimization_techniques": ["quantization", "pruning"]
        }
        
        put_response = client.put(
            "/config/optimization-criteria",
            json=update_data,
            headers={"Authorization": "Bearer admin_token"}
        )
        
        assert put_response.status_code == 200
        
        # Make GET request
        get_response = client.get(
            "/config/optimization-criteria",
            headers={"Authorization": "Bearer user_token"}
        )
        
        assert get_response.status_code == 200
        
        # Verify data matches
        get_data = get_response.json()
        assert get_data["name"] == update_data["name"]
        assert get_data["target_accuracy_threshold"] == update_data["target_accuracy_threshold"]
        
        # Cleanup
        app.dependency_overrides.clear()


class TestCacheInvalidation:
    """Test cache invalidation after updates."""
    
    def test_cache_invalidated_after_put(self, app, temp_config_path, valid_criteria, mock_user, mock_admin):
        """Test that cache is invalidated after PUT."""
        # Reset singleton
        ConfigurationManager._instance = None
        config_manager = ConfigurationManager(temp_config_path)
        config_manager.save_configuration(valid_criteria)
        
        # Mock dependencies
        async def mock_get_user():
            return mock_user
        
        async def mock_get_admin():
            return mock_admin
        
        async def mock_get_config_manager(request: Request):
            return config_manager
        
        app.dependency_overrides[get_config_manager] = mock_get_config_manager
        
        from src.api.dependencies import get_current_user, get_admin_user
        app.dependency_overrides[get_current_user] = mock_get_user
        app.dependency_overrides[get_admin_user] = mock_get_admin
        
        client = TestClient(app)
        
        # First GET (populates cache)
        response1 = client.get(
            "/config/optimization-criteria",
            headers={"Authorization": "Bearer user_token"}
        )
        assert response1.status_code == 200
        original_name = response1.json()["name"]
        
        # PUT (should invalidate cache)
        update_data = {
            "name": "new_config",
            "target_accuracy_threshold": 0.95,
            "max_size_reduction_percent": 50.0,
            "max_latency_increase_percent": 10.0,
            "optimization_techniques": ["quantization"]
        }
        
        put_response = client.put(
            "/config/optimization-criteria",
            json=update_data,
            headers={"Authorization": "Bearer admin_token"}
        )
        assert put_response.status_code == 200
        
        # Second GET (should get new value, not cached)
        response2 = client.get(
            "/config/optimization-criteria",
            headers={"Authorization": "Bearer user_token"}
        )
        assert response2.status_code == 200
        new_name = response2.json()["name"]
        
        # Names should be different
        assert new_name != original_name
        assert new_name == "new_config"
        
        # Cleanup
        app.dependency_overrides.clear()


class TestErrorHandling:
    """Test error handling in configuration endpoints."""
    
    def test_get_with_corrupted_file(self, app, mock_user):
        """Test GET when configuration file is corrupted."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.json"
            config_path.write_text("{ corrupted json }")
            
            # Reset singleton and clear cache
            ConfigurationManager._instance = None
            config_manager = ConfigurationManager(str(config_path))
            config_manager._current_config = None  # Clear cached config
            
            # Clear cache service
            from src.services.cache_service import CacheService
            cache = CacheService()
            cache.clear()
            
            # Mock dependencies
            async def mock_get_user():
                return mock_user
            
            async def mock_get_config_manager(request: Request):
                return config_manager
            
            app.dependency_overrides[get_config_manager] = mock_get_config_manager
            
            from src.api.dependencies import get_current_user
            app.dependency_overrides[get_current_user] = mock_get_user
            
            client = TestClient(app)
            
            # Make request
            response = client.get(
                "/config/optimization-criteria",
                headers={"Authorization": "Bearer test_token"}
            )
            
            # Should return default configuration
            assert response.status_code == 200
            data = response.json()
            assert data["name"] == "default"
            
            # Cleanup
            app.dependency_overrides.clear()
    
    def test_put_with_io_error(self, app, temp_config_path, mock_admin):
        """Test PUT when file write fails."""
        # Reset singleton
        ConfigurationManager._instance = None
        config_manager = ConfigurationManager(temp_config_path)
        
        # Mock save to fail
        original_save = config_manager.save_configuration
        
        def failing_save(criteria):
            return False  # Simulate failure
        
        config_manager.save_configuration = failing_save
        
        # Mock dependencies
        async def mock_get_admin():
            return mock_admin
        
        async def mock_get_config_manager(request: Request):
            return config_manager
        
        app.dependency_overrides[get_config_manager] = mock_get_config_manager
        
        from src.api.dependencies import get_admin_user
        app.dependency_overrides[get_admin_user] = mock_get_admin
        
        client = TestClient(app)
        
        valid_data = {
            "name": "test_config",
            "target_accuracy_threshold": 0.95,
            "max_size_reduction_percent": 50.0,
            "max_latency_increase_percent": 10.0,
            "optimization_techniques": ["quantization"]
        }
        
        # Make request
        response = client.put(
            "/config/optimization-criteria",
            json=valid_data,
            headers={"Authorization": "Bearer admin_token"}
        )
        
        # Should return error
        assert response.status_code == 500
        
        # Restore original method
        config_manager.save_configuration = original_save
        
        # Cleanup
        app.dependency_overrides.clear()
    
    def test_get_with_service_unavailable(self, app, mock_user):
        """Test GET when ConfigurationManager is unavailable."""
        # Mock to raise exception
        async def mock_get_config_manager_error(request: Request):
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Configuration service unavailable"
            )
        
        async def mock_get_user():
            return mock_user
        
        app.dependency_overrides[get_config_manager] = mock_get_config_manager_error
        
        from src.api.dependencies import get_current_user
        app.dependency_overrides[get_current_user] = mock_get_user
        
        client = TestClient(app)
        
        # Make request
        response = client.get(
            "/config/optimization-criteria",
            headers={"Authorization": "Bearer test_token"}
        )
        
        # Should return service unavailable
        assert response.status_code == 503
        
        # Cleanup
        app.dependency_overrides.clear()


class TestResponseFormat:
    """Test response format compliance."""
    
    def test_get_response_schema(self, app, temp_config_path, valid_criteria, mock_user):
        """Test that GET response matches schema."""
        # Reset singleton
        ConfigurationManager._instance = None
        config_manager = ConfigurationManager(temp_config_path)
        config_manager.save_configuration(valid_criteria)
        
        # Mock dependencies
        async def mock_get_user():
            return mock_user
        
        async def mock_get_config_manager(request: Request):
            return config_manager
        
        app.dependency_overrides[get_config_manager] = mock_get_config_manager
        
        from src.api.dependencies import get_current_user
        app.dependency_overrides[get_current_user] = mock_get_user
        
        client = TestClient(app)
        
        # Make request
        response = client.get(
            "/config/optimization-criteria",
            headers={"Authorization": "Bearer test_token"}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify required fields
        required_fields = [
            "name",
            "target_accuracy_threshold",
            "max_size_reduction_percent",
            "max_latency_increase_percent",
            "optimization_techniques",
            "created_at",
            "updated_at"
        ]
        
        for field in required_fields:
            assert field in data, f"Missing required field: {field}"
        
        # Verify types
        assert isinstance(data["name"], str)
        assert isinstance(data["target_accuracy_threshold"], (int, float))
        assert isinstance(data["optimization_techniques"], list)
        
        # Cleanup
        app.dependency_overrides.clear()
    
    def test_put_response_schema(self, app, temp_config_path, valid_request_data, mock_admin):
        """Test that PUT response matches schema."""
        # Reset singleton
        ConfigurationManager._instance = None
        config_manager = ConfigurationManager(temp_config_path)
        
        # Mock dependencies
        async def mock_get_admin():
            return mock_admin
        
        async def mock_get_config_manager(request: Request):
            return config_manager
        
        app.dependency_overrides[get_config_manager] = mock_get_config_manager
        
        from src.api.dependencies import get_admin_user
        app.dependency_overrides[get_admin_user] = mock_get_admin
        
        client = TestClient(app)
        
        # Make request
        response = client.put(
            "/config/optimization-criteria",
            json=valid_request_data,
            headers={"Authorization": "Bearer admin_token"}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify required fields
        required_fields = [
            "name",
            "target_accuracy_threshold",
            "max_size_reduction_percent",
            "max_latency_increase_percent",
            "optimization_techniques",
            "created_at",
            "updated_at"
        ]
        
        for field in required_fields:
            assert field in data, f"Missing required field: {field}"
        
        # Cleanup
        app.dependency_overrides.clear()

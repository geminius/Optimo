"""
Deployment validation tests for the robotics model optimization platform.
Tests container health, service connectivity, and basic functionality.
Validates requirements 6.1 and 6.2 for model format support and deployment readiness.
"""

import pytest
import requests
import time
import subprocess
import json
import psycopg2
import redis
import websocket
import tempfile
import torch
import torch.nn as nn
import onnx
import tensorflow as tf
from typing import Dict, Any
import os
import io
from pathlib import Path


class TestDeploymentValidation:
    """Test suite for validating deployment health and functionality."""
    
    @pytest.fixture(scope="class")
    def deployment_config(self) -> Dict[str, Any]:
        """Load deployment configuration."""
        return {
            "api_url": os.getenv("API_URL", "http://localhost:8000"),
            "frontend_url": os.getenv("FRONTEND_URL", "http://localhost:3000"),
            "db_host": os.getenv("DB_HOST", "localhost"),
            "db_port": int(os.getenv("DB_PORT", "5432")),
            "db_name": os.getenv("POSTGRES_DB", "robotics_optimization"),
            "db_user": os.getenv("POSTGRES_USER", "postgres"),
            "db_password": os.getenv("POSTGRES_PASSWORD", "postgres"),
            "redis_url": os.getenv("REDIS_URL", "redis://localhost:6379"),
        }
    
    def test_docker_services_running(self):
        """Test that all Docker services are running."""
        result = subprocess.run(
            ["docker-compose", "ps", "--services", "--filter", "status=running"],
            capture_output=True,
            text=True
        )
        
        running_services = set(result.stdout.strip().split('\n'))
        expected_services = {"api", "worker", "frontend", "db", "redis", "nginx"}
        
        assert expected_services.issubset(running_services), \
            f"Missing services: {expected_services - running_services}"
    
    def test_service_health_checks(self, deployment_config):
        """Test that all services pass their health checks."""
        # Wait for services to be ready
        time.sleep(30)
        
        # Check API health
        response = requests.get(f"{deployment_config['api_url']}/health", timeout=10)
        assert response.status_code == 200
        health_data = response.json()
        assert health_data["status"] == "healthy"
        
        # Check database connectivity
        try:
            conn = psycopg2.connect(
                host=deployment_config["db_host"],
                port=deployment_config["db_port"],
                database=deployment_config["db_name"],
                user=deployment_config["db_user"],
                password=deployment_config["db_password"]
            )
            conn.close()
        except Exception as e:
            pytest.fail(f"Database connection failed: {e}")
        
        # Check Redis connectivity
        try:
            r = redis.from_url(deployment_config["redis_url"])
            r.ping()
        except Exception as e:
            pytest.fail(f"Redis connection failed: {e}")
    
    def test_api_endpoints_accessible(self, deployment_config):
        """Test that key API endpoints are accessible."""
        base_url = deployment_config["api_url"]
        
        # Test health endpoint
        response = requests.get(f"{base_url}/health")
        assert response.status_code == 200
        
        # Test API documentation
        response = requests.get(f"{base_url}/docs")
        assert response.status_code == 200
        
        # Test OpenAPI spec
        response = requests.get(f"{base_url}/openapi.json")
        assert response.status_code == 200
        spec = response.json()
        assert "openapi" in spec
        assert "paths" in spec
    
    def test_frontend_accessible(self, deployment_config):
        """Test that frontend is accessible."""
        response = requests.get(deployment_config["frontend_url"], timeout=10)
        assert response.status_code == 200
        assert "text/html" in response.headers.get("content-type", "")
    
    def test_database_schema(self, deployment_config):
        """Test that database schema is properly initialized."""
        conn = psycopg2.connect(
            host=deployment_config["db_host"],
            port=deployment_config["db_port"],
            database=deployment_config["db_name"],
            user=deployment_config["db_user"],
            password=deployment_config["db_password"]
        )
        
        cursor = conn.cursor()
        
        # Check required tables exist
        required_tables = [
            "models",
            "optimization_sessions", 
            "optimization_steps",
            "evaluation_reports"
        ]
        
        for table in required_tables:
            cursor.execute(
                "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = %s)",
                (table,)
            )
            exists = cursor.fetchone()[0]
            assert exists, f"Table {table} does not exist"
        
        # Check indexes exist
        cursor.execute("""
            SELECT indexname FROM pg_indexes 
            WHERE tablename IN ('models', 'optimization_sessions', 'optimization_steps')
        """)
        indexes = [row[0] for row in cursor.fetchall()]
        assert len(indexes) > 0, "No indexes found"
        
        conn.close()
    
    def test_api_authentication(self, deployment_config):
        """Test API authentication endpoints."""
        base_url = deployment_config["api_url"]
        
        # Test that protected endpoints require authentication
        response = requests.get(f"{base_url}/api/models")
        assert response.status_code in [401, 403], \
            "Protected endpoint should require authentication"
    
    def test_file_upload_endpoint(self, deployment_config):
        """Test file upload functionality."""
        base_url = deployment_config["api_url"]
        
        # Create a small test file
        test_content = b"test model content"
        files = {"file": ("test_model.pth", test_content, "application/octet-stream")}
        
        # Test upload endpoint exists (may require auth)
        response = requests.post(f"{base_url}/api/upload", files=files)
        # Should get 401/403 (auth required) or 200 (success), not 404
        assert response.status_code != 404, "Upload endpoint not found"
    
    def test_websocket_endpoint(self, deployment_config):
        """Test WebSocket endpoint availability."""
        ws_url = deployment_config["api_url"].replace("http://", "ws://") + "/ws"
        
        try:
            ws = websocket.create_connection(ws_url, timeout=5)
            ws.close()
        except Exception as e:
            # WebSocket may require authentication, but should not be completely unavailable
            assert "404" not in str(e), f"WebSocket endpoint not found: {e}"
    
    def test_nginx_proxy_configuration(self, deployment_config):
        """Test that Nginx is properly configured."""
        # Test that requests are properly proxied
        response = requests.get(f"{deployment_config['api_url']}/health")
        assert response.status_code == 200
        
        # Check security headers (if configured)
        headers = response.headers
        security_headers = [
            "X-Content-Type-Options",
            "X-Frame-Options", 
            "X-XSS-Protection"
        ]
        
        # At least some security headers should be present
        present_headers = [h for h in security_headers if h in headers]
        assert len(present_headers) > 0, "No security headers found"
    
    def test_service_resource_limits(self):
        """Test that services are running within expected resource limits."""
        result = subprocess.run(
            ["docker", "stats", "--no-stream", "--format", "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}"],
            capture_output=True,
            text=True
        )
        
        lines = result.stdout.strip().split('\n')[1:]  # Skip header
        
        for line in lines:
            if line.strip():
                parts = line.split('\t')
                if len(parts) >= 3:
                    name, cpu_perc, mem_usage = parts[0], parts[1], parts[2]
                    
                    # Basic sanity checks
                    cpu_val = float(cpu_perc.rstrip('%'))
                    assert cpu_val < 100, f"Service {name} using {cpu_perc} CPU"
                    
                    # Memory usage should be reasonable
                    if 'GiB' in mem_usage:
                        mem_val = float(mem_usage.split('GiB')[0].split('/')[-1])
                        assert mem_val < 10, f"Service {name} using {mem_usage} memory"
    
    def test_log_accessibility(self):
        """Test that service logs are accessible."""
        services = ["api", "worker", "db", "redis"]
        
        for service in services:
            result = subprocess.run(
                ["docker-compose", "logs", "--tail=10", service],
                capture_output=True,
                text=True
            )
            
            assert result.returncode == 0, f"Cannot access logs for {service}"
            assert len(result.stdout.strip()) > 0, f"No logs found for {service}"
    
    def test_environment_configuration(self, deployment_config):
        """Test that environment configuration is properly loaded."""
        response = requests.get(f"{deployment_config['api_url']}/health")
        health_data = response.json()
        
        # Check that environment-specific settings are loaded
        assert "environment" in health_data
        assert health_data["environment"] in ["development", "production", "testing"]
    
    def test_backup_script_executable(self):
        """Test that backup script is executable and functional."""
        # Check script exists and is executable
        result = subprocess.run(["test", "-x", "scripts/backup.sh"])
        assert result.returncode == 0, "Backup script is not executable"
        
        # Test script syntax (dry run)
        result = subprocess.run(["bash", "-n", "scripts/backup.sh"])
        assert result.returncode == 0, "Backup script has syntax errors"


class TestDeploymentPerformance:
    """Performance tests for deployed services."""
    
    def test_api_response_time(self, deployment_config):
        """Test API response times are acceptable."""
        start_time = time.time()
        response = requests.get(f"{deployment_config['api_url']}/health")
        response_time = time.time() - start_time
        
        assert response.status_code == 200
        assert response_time < 2.0, f"API response time too slow: {response_time}s"
    
    def test_frontend_load_time(self, deployment_config):
        """Test frontend load times are acceptable."""
        start_time = time.time()
        response = requests.get(deployment_config["frontend_url"])
        load_time = time.time() - start_time
        
        assert response.status_code == 200
        assert load_time < 5.0, f"Frontend load time too slow: {load_time}s"
    
    def test_database_query_performance(self, deployment_config):
        """Test database query performance."""
        conn = psycopg2.connect(
            host=deployment_config["db_host"],
            port=deployment_config["db_port"],
            database=deployment_config["db_name"],
            user=deployment_config["db_user"],
            password=deployment_config["db_password"]
        )
        
        cursor = conn.cursor()
        
        # Test simple query performance
        start_time = time.time()
        cursor.execute("SELECT COUNT(*) FROM models")
        query_time = time.time() - start_time
        
        assert query_time < 1.0, f"Database query too slow: {query_time}s"
        
        conn.close()


class TestDeploymentSecurity:
    """Security tests for deployed services."""
    
    def test_no_default_credentials(self, deployment_config):
        """Test that default credentials are not in use."""
        # This is a basic check - in production, use proper security scanning
        
        # Check that we're not using obvious default passwords
        db_password = deployment_config["db_password"]
        assert db_password != "password", "Using default password"
        assert db_password != "postgres", "Using default password"
        assert len(db_password) >= 8, "Password too short"
    
    def test_service_isolation(self):
        """Test that services are properly isolated."""
        # Check that services are running in containers
        result = subprocess.run(
            ["docker", "ps", "--format", "{{.Names}}"],
            capture_output=True,
            text=True
        )
        
        container_names = result.stdout.strip().split('\n')
        expected_containers = ["api", "worker", "db", "redis", "frontend"]
        
        for container in expected_containers:
            matching_containers = [name for name in container_names if container in name]
            assert len(matching_containers) > 0, f"Container {container} not found"
    
    def test_exposed_ports_minimal(self):
        """Test that only necessary ports are exposed."""
        result = subprocess.run(
            ["docker-compose", "ps"],
            capture_output=True,
            text=True
        )
        
        # Check that internal services don't expose unnecessary ports
        lines = result.stdout.split('\n')
        for line in lines:
            if 'db' in line and '5432' in line:
                # Database port should only be exposed in development
                assert '0.0.0.0:5432' not in line or os.getenv('ENVIRONMENT') == 'development'


class TestModelFormatSupport:
    """Test deployment support for various model formats (Requirement 6.1)."""
    
    @pytest.fixture
    def sample_pytorch_model(self):
        """Create a sample PyTorch model for testing."""
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 1)
            
            def forward(self, x):
                return self.linear(x)
        
        model = SimpleModel()
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
            torch.save(model.state_dict(), f.name)
            return f.name
    
    @pytest.fixture
    def sample_onnx_model(self):
        """Create a sample ONNX model for testing."""
        # Create a simple PyTorch model first
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 1)
            
            def forward(self, x):
                return self.linear(x)
        
        model = SimpleModel()
        dummy_input = torch.randn(1, 10)
        
        # Export to ONNX
        with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
            torch.onnx.export(model, dummy_input, f.name, 
                            input_names=['input'], output_names=['output'])
            return f.name
    
    @pytest.fixture
    def sample_tensorflow_model(self):
        """Create a sample TensorFlow model for testing."""
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(1, input_shape=(10,))
        ])
        
        # Save to temporary directory
        temp_dir = tempfile.mkdtemp()
        model_path = os.path.join(temp_dir, 'model.h5')
        model.save(model_path)
        return model_path
    
    def test_pytorch_model_upload_support(self, deployment_config, sample_pytorch_model):
        """Test that PyTorch models can be uploaded and recognized."""
        base_url = deployment_config["api_url"]
        
        with open(sample_pytorch_model, 'rb') as f:
            files = {"file": ("test_model.pth", f, "application/octet-stream")}
            response = requests.post(f"{base_url}/api/upload", files=files)
            
            # Should not return 404 (endpoint exists) or 415 (unsupported format)
            assert response.status_code not in [404, 415], \
                f"PyTorch model upload failed: {response.status_code}"
        
        # Cleanup
        os.unlink(sample_pytorch_model)
    
    def test_onnx_model_upload_support(self, deployment_config, sample_onnx_model):
        """Test that ONNX models can be uploaded and recognized."""
        base_url = deployment_config["api_url"]
        
        with open(sample_onnx_model, 'rb') as f:
            files = {"file": ("test_model.onnx", f, "application/octet-stream")}
            response = requests.post(f"{base_url}/api/upload", files=files)
            
            # Should not return 404 (endpoint exists) or 415 (unsupported format)
            assert response.status_code not in [404, 415], \
                f"ONNX model upload failed: {response.status_code}"
        
        # Cleanup
        os.unlink(sample_onnx_model)
    
    def test_tensorflow_model_upload_support(self, deployment_config, sample_tensorflow_model):
        """Test that TensorFlow models can be uploaded and recognized."""
        base_url = deployment_config["api_url"]
        
        with open(sample_tensorflow_model, 'rb') as f:
            files = {"file": ("test_model.h5", f, "application/octet-stream")}
            response = requests.post(f"{base_url}/api/upload", files=files)
            
            # Should not return 404 (endpoint exists) or 415 (unsupported format)
            assert response.status_code not in [404, 415], \
                f"TensorFlow model upload failed: {response.status_code}"
        
        # Cleanup
        import shutil
        shutil.rmtree(os.path.dirname(sample_tensorflow_model))
    
    def test_unsupported_format_handling(self, deployment_config):
        """Test that unsupported formats return appropriate error messages."""
        base_url = deployment_config["api_url"]
        
        # Create a fake model file with unsupported extension
        fake_content = b"fake model content"
        files = {"file": ("test_model.xyz", io.BytesIO(fake_content), "application/octet-stream")}
        
        response = requests.post(f"{base_url}/api/upload", files=files)
        
        # Should return appropriate error for unsupported format
        if response.status_code == 400:
            error_data = response.json()
            assert "unsupported" in error_data.get("detail", "").lower() or \
                   "format" in error_data.get("detail", "").lower(), \
                   "Error message should mention unsupported format"


class TestOptimizationTechniqueSupport:
    """Test deployment support for optimization techniques (Requirement 6.3)."""
    
    def test_quantization_support_available(self, deployment_config):
        """Test that quantization optimization is available."""
        base_url = deployment_config["api_url"]
        
        # Check if optimization techniques endpoint exists
        response = requests.get(f"{base_url}/api/optimization/techniques")
        
        if response.status_code == 200:
            techniques = response.json()
            technique_names = [t.get("name", "").lower() for t in techniques]
            assert any("quantization" in name for name in technique_names), \
                "Quantization technique should be available"
    
    def test_pruning_support_available(self, deployment_config):
        """Test that pruning optimization is available."""
        base_url = deployment_config["api_url"]
        
        response = requests.get(f"{base_url}/api/optimization/techniques")
        
        if response.status_code == 200:
            techniques = response.json()
            technique_names = [t.get("name", "").lower() for t in techniques]
            assert any("pruning" in name for name in technique_names), \
                "Pruning technique should be available"
    
    def test_distillation_support_available(self, deployment_config):
        """Test that knowledge distillation optimization is available."""
        base_url = deployment_config["api_url"]
        
        response = requests.get(f"{base_url}/api/optimization/techniques")
        
        if response.status_code == 200:
            techniques = response.json()
            technique_names = [t.get("name", "").lower() for t in techniques]
            assert any("distillation" in name for name in technique_names), \
                "Knowledge distillation technique should be available"
    
    def test_architecture_search_support_available(self, deployment_config):
        """Test that architecture search optimization is available."""
        base_url = deployment_config["api_url"]
        
        response = requests.get(f"{base_url}/api/optimization/techniques")
        
        if response.status_code == 200:
            techniques = response.json()
            technique_names = [t.get("name", "").lower() for t in techniques]
            assert any("architecture" in name or "nas" in name for name in technique_names), \
                "Architecture search technique should be available"


class TestDeploymentRobustness:
    """Test deployment robustness and error recovery."""
    
    def test_service_restart_recovery(self):
        """Test that services can recover from restarts."""
        # Restart API service
        result = subprocess.run(
            ["docker-compose", "restart", "api"],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0, "Failed to restart API service"
        
        # Wait for service to be ready
        time.sleep(30)
        
        # Test that service is accessible again
        response = requests.get("http://localhost:8000/health", timeout=10)
        assert response.status_code == 200, "API service not accessible after restart"
    
    def test_database_connection_recovery(self, deployment_config):
        """Test that API recovers from database disconnection."""
        # Restart database service
        result = subprocess.run(
            ["docker-compose", "restart", "db"],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0, "Failed to restart database service"
        
        # Wait for database to be ready
        time.sleep(45)
        
        # Test that API can connect to database again
        response = requests.get(f"{deployment_config['api_url']}/health", timeout=15)
        assert response.status_code == 200, "API not accessible after database restart"
        
        health_data = response.json()
        assert health_data.get("database") != "error", "Database connection not recovered"
    
    def test_redis_connection_recovery(self, deployment_config):
        """Test that services recover from Redis disconnection."""
        # Restart Redis service
        result = subprocess.run(
            ["docker-compose", "restart", "redis"],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0, "Failed to restart Redis service"
        
        # Wait for Redis to be ready
        time.sleep(20)
        
        # Test that API can connect to Redis again
        response = requests.get(f"{deployment_config['api_url']}/health", timeout=10)
        assert response.status_code == 200, "API not accessible after Redis restart"
        
        health_data = response.json()
        assert health_data.get("redis") != "error", "Redis connection not recovered"
    
    def test_worker_scaling_functionality(self):
        """Test that worker services can be scaled up and down."""
        # Scale up workers
        result = subprocess.run(
            ["docker-compose", "up", "-d", "--scale", "worker=3"],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0, "Failed to scale up workers"
        
        time.sleep(30)
        
        # Check that 3 workers are running
        result = subprocess.run(
            ["docker-compose", "ps", "worker"],
            capture_output=True,
            text=True
        )
        worker_lines = [line for line in result.stdout.split('\n') if 'worker' in line and 'Up' in line]
        assert len(worker_lines) == 3, f"Expected 3 workers, found {len(worker_lines)}"
        
        # Scale back down
        result = subprocess.run(
            ["docker-compose", "up", "-d", "--scale", "worker=1"],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0, "Failed to scale down workers"


class TestDeploymentMonitoring:
    """Test deployment monitoring and observability."""
    
    def test_health_check_endpoints_comprehensive(self, deployment_config):
        """Test comprehensive health check information."""
        response = requests.get(f"{deployment_config['api_url']}/health")
        assert response.status_code == 200
        
        health_data = response.json()
        
        # Check required health information
        required_fields = ["status", "timestamp", "version", "environment"]
        for field in required_fields:
            assert field in health_data, f"Health check missing {field}"
        
        # Check service dependencies
        if "dependencies" in health_data:
            deps = health_data["dependencies"]
            assert "database" in deps, "Database dependency not reported"
            assert "redis" in deps, "Redis dependency not reported"
    
    def test_metrics_endpoint_available(self, deployment_config):
        """Test that metrics endpoint is available for monitoring."""
        # Try common metrics endpoints
        metrics_endpoints = ["/metrics", "/api/metrics", "/health/metrics"]
        
        found_metrics = False
        for endpoint in metrics_endpoints:
            response = requests.get(f"{deployment_config['api_url']}{endpoint}")
            if response.status_code == 200:
                found_metrics = True
                break
        
        # At least one metrics endpoint should be available
        assert found_metrics, "No metrics endpoint found"
    
    def test_log_aggregation_working(self):
        """Test that logs are being properly aggregated."""
        # Check that logs are accessible
        result = subprocess.run(
            ["docker-compose", "logs", "--tail=10", "api"],
            capture_output=True,
            text=True
        )
        
        assert result.returncode == 0, "Cannot access API logs"
        assert len(result.stdout.strip()) > 0, "No API logs found"
        
        # Check log format (should contain timestamps)
        log_lines = result.stdout.strip().split('\n')
        timestamp_found = any('2024' in line or '2025' in line for line in log_lines)
        assert timestamp_found, "Logs should contain timestamps"
    
    def test_error_logging_functional(self, deployment_config):
        """Test that error logging is working properly."""
        # Make a request that should generate an error log
        response = requests.get(f"{deployment_config['api_url']}/api/nonexistent-endpoint")
        assert response.status_code == 404
        
        # Wait a moment for log to be written
        time.sleep(2)
        
        # Check that error was logged
        result = subprocess.run(
            ["docker-compose", "logs", "--tail=20", "api"],
            capture_output=True,
            text=True
        )
        
        # Should contain some indication of the 404 error
        log_content = result.stdout.lower()
        assert "404" in log_content or "not found" in log_content, \
            "404 error should be logged"


class TestDeploymentConfiguration:
    """Test deployment configuration management."""
    
    def test_environment_variables_loaded(self, deployment_config):
        """Test that environment variables are properly loaded."""
        response = requests.get(f"{deployment_config['api_url']}/health")
        health_data = response.json()
        
        # Environment should be set
        assert "environment" in health_data
        assert health_data["environment"] in ["development", "production", "testing"]
    
    def test_configuration_validation(self, deployment_config):
        """Test that configuration is validated on startup."""
        # Check that services started successfully (implies config validation passed)
        result = subprocess.run(
            ["docker-compose", "ps"],
            capture_output=True,
            text=True
        )
        
        # No services should be in "Exit" state due to config errors
        assert "Exit" not in result.stdout, "Some services exited (possible config error)"
    
    def test_secret_management(self, deployment_config):
        """Test that secrets are not exposed in logs or responses."""
        # Check health endpoint doesn't expose secrets
        response = requests.get(f"{deployment_config['api_url']}/health")
        health_data = response.json()
        
        # Convert to string to search for common secret patterns
        health_str = json.dumps(health_data).lower()
        
        # Should not contain password-like strings
        secret_patterns = ["password", "secret", "key", "token"]
        for pattern in secret_patterns:
            if pattern in health_str:
                # Make sure it's not just a field name, but an actual value
                assert not any(f'"{pattern}":"' in health_str for pattern in secret_patterns), \
                    f"Potential secret exposure in health endpoint"
    
    def test_cors_configuration(self, deployment_config):
        """Test that CORS is properly configured."""
        # Make a preflight request
        headers = {
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "Content-Type"
        }
        
        response = requests.options(f"{deployment_config['api_url']}/api/upload", headers=headers)
        
        # Should handle CORS preflight
        assert response.status_code in [200, 204], "CORS preflight should be handled"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
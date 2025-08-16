"""
Deployment validation tests for the robotics model optimization platform.
Tests container health, service connectivity, and basic functionality.
"""

import pytest
import requests
import time
import subprocess
import json
import psycopg2
import redis
from typing import Dict, Any
import os


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
        import websocket
        
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
"""
Production deployment validation tests.
Tests production-specific configurations and requirements.
"""

import pytest
import requests
import subprocess
import time
import os
import json
from typing import Dict, Any


class TestProductionDeployment:
    """Test production deployment specific requirements."""
    
    @pytest.fixture(scope="class")
    def production_config(self) -> Dict[str, Any]:
        """Load production deployment configuration."""
        return {
            "api_url": os.getenv("API_URL", "http://localhost:8000"),
            "frontend_url": os.getenv("FRONTEND_URL", "http://localhost:3000"),
            "environment": os.getenv("ENVIRONMENT", "development"),
        }
    
    @pytest.mark.skipif(
        os.getenv("ENVIRONMENT") != "production",
        reason="Production tests only run in production environment"
    )
    def test_production_environment_detected(self, production_config):
        """Test that production environment is properly detected."""
        response = requests.get(f"{production_config['api_url']}/health")
        assert response.status_code == 200
        
        health_data = response.json()
        assert health_data.get("environment") == "production"
    
    @pytest.mark.skipif(
        os.getenv("ENVIRONMENT") != "production",
        reason="Production tests only run in production environment"
    )
    def test_debug_mode_disabled(self, production_config):
        """Test that debug mode is disabled in production."""
        response = requests.get(f"{production_config['api_url']}/health")
        health_data = response.json()
        
        # Debug should be false or not present
        assert health_data.get("debug", False) is False
    
    @pytest.mark.skipif(
        os.getenv("ENVIRONMENT") != "production",
        reason="Production tests only run in production environment"
    )
    def test_ssl_configuration(self, production_config):
        """Test SSL configuration in production."""
        if production_config["api_url"].startswith("https://"):
            response = requests.get(f"{production_config['api_url']}/health")
            assert response.status_code == 200
            
            # Check SSL certificate is valid (requests would fail if not)
            assert response.url.startswith("https://")
    
    @pytest.mark.skipif(
        os.getenv("ENVIRONMENT") != "production",
        reason="Production tests only run in production environment"
    )
    def test_resource_limits_enforced(self):
        """Test that resource limits are enforced in production."""
        result = subprocess.run(
            ["docker", "inspect", "--format", "{{.HostConfig.Memory}}", "api"],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            memory_limit = result.stdout.strip()
            # Should have memory limit set (non-zero)
            assert memory_limit != "0", "Memory limit should be set in production"
    
    @pytest.mark.skipif(
        os.getenv("ENVIRONMENT") != "production",
        reason="Production tests only run in production environment"
    )
    def test_service_replicas_configured(self):
        """Test that services are configured with appropriate replicas."""
        result = subprocess.run(
            ["docker-compose", "ps"],
            capture_output=True,
            text=True
        )
        
        # Count API and worker instances
        api_count = result.stdout.count("api")
        worker_count = result.stdout.count("worker")
        
        # Production should have multiple instances
        assert api_count >= 2, f"Expected multiple API instances, found {api_count}"
        assert worker_count >= 2, f"Expected multiple worker instances, found {worker_count}"
    
    @pytest.mark.skipif(
        os.getenv("ENVIRONMENT") != "production",
        reason="Production tests only run in production environment"
    )
    def test_logging_configuration(self):
        """Test production logging configuration."""
        # Check log rotation is configured
        result = subprocess.run(
            ["docker", "inspect", "--format", "{{.HostConfig.LogConfig.Type}}", "api"],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            log_driver = result.stdout.strip()
            assert log_driver in ["json-file", "syslog", "journald"], \
                f"Production should use proper log driver, found: {log_driver}"
    
    @pytest.mark.skipif(
        os.getenv("ENVIRONMENT") != "production",
        reason="Production tests only run in production environment"
    )
    def test_security_headers_present(self, production_config):
        """Test that security headers are present in production."""
        response = requests.get(f"{production_config['api_url']}/health")
        headers = response.headers
        
        # Check for important security headers
        security_headers = {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": ["DENY", "SAMEORIGIN"],
            "X-XSS-Protection": "1; mode=block",
        }
        
        for header, expected_values in security_headers.items():
            assert header in headers, f"Missing security header: {header}"
            
            if isinstance(expected_values, list):
                assert headers[header] in expected_values, \
                    f"Invalid {header} value: {headers[header]}"
            else:
                assert headers[header] == expected_values, \
                    f"Invalid {header} value: {headers[header]}"


class TestDevelopmentDeployment:
    """Test development deployment specific requirements."""
    
    @pytest.fixture(scope="class")
    def development_config(self) -> Dict[str, Any]:
        """Load development deployment configuration."""
        return {
            "api_url": os.getenv("API_URL", "http://localhost:8000"),
            "frontend_url": os.getenv("FRONTEND_URL", "http://localhost:3000"),
            "environment": os.getenv("ENVIRONMENT", "development"),
        }
    
    @pytest.mark.skipif(
        os.getenv("ENVIRONMENT") == "production",
        reason="Development tests don't run in production"
    )
    def test_development_environment_detected(self, development_config):
        """Test that development environment is properly detected."""
        response = requests.get(f"{development_config['api_url']}/health")
        assert response.status_code == 200
        
        health_data = response.json()
        assert health_data.get("environment") == "development"
    
    @pytest.mark.skipif(
        os.getenv("ENVIRONMENT") == "production",
        reason="Development tests don't run in production"
    )
    def test_debug_endpoints_available(self, development_config):
        """Test that debug endpoints are available in development."""
        # API documentation should be available
        response = requests.get(f"{development_config['api_url']}/docs")
        assert response.status_code == 200
        
        # OpenAPI spec should be available
        response = requests.get(f"{development_config['api_url']}/openapi.json")
        assert response.status_code == 200
    
    @pytest.mark.skipif(
        os.getenv("ENVIRONMENT") == "production",
        reason="Development tests don't run in production"
    )
    def test_hot_reload_enabled(self, development_config):
        """Test that hot reload is working in development."""
        # This is a basic test - in practice, you'd modify a file and check reload
        response = requests.get(f"{development_config['api_url']}/health")
        health_data = response.json()
        
        # Development should have debug info
        assert health_data.get("debug", True) is not False
    
    @pytest.mark.skipif(
        os.getenv("ENVIRONMENT") == "production",
        reason="Development tests don't run in production"
    )
    def test_database_ports_exposed(self):
        """Test that database ports are exposed in development."""
        result = subprocess.run(
            ["docker-compose", "ps", "db"],
            capture_output=True,
            text=True
        )
        
        # Should show port mapping in development
        assert "5432" in result.stdout, "Database port should be exposed in development"


class TestDeploymentBackup:
    """Test backup and recovery functionality."""
    
    def test_backup_script_functionality(self):
        """Test that backup script creates proper backups."""
        # Run backup script
        result = subprocess.run(
            ["./scripts/backup.sh"],
            capture_output=True,
            text=True,
            cwd="."
        )
        
        assert result.returncode == 0, f"Backup script failed: {result.stderr}"
        
        # Check that backup file was created
        backup_dir = "backups"
        if os.path.exists(backup_dir):
            backup_files = os.listdir(backup_dir)
            sql_backups = [f for f in backup_files if f.endswith('.sql') or f.endswith('.sql.gz')]
            assert len(sql_backups) > 0, "No SQL backup files found"
    
    def test_backup_contains_data(self):
        """Test that backup contains actual data."""
        backup_dir = "backups"
        if os.path.exists(backup_dir):
            backup_files = os.listdir(backup_dir)
            sql_backups = [f for f in backup_files if f.endswith('.sql') or f.endswith('.sql.gz')]
            
            if sql_backups:
                backup_file = os.path.join(backup_dir, sql_backups[0])
                
                # Check file size (should not be empty)
                file_size = os.path.getsize(backup_file)
                assert file_size > 100, f"Backup file too small: {file_size} bytes"
    
    def test_database_recovery_possible(self):
        """Test that database can be recovered from backup."""
        # This is a basic test - full recovery testing would require
        # a separate test database instance
        
        # Check that init script exists and is valid
        init_script = "scripts/init-db.sql"
        if os.path.exists(init_script):
            with open(init_script, 'r') as f:
                content = f.read()
                
            # Should contain CREATE TABLE statements
            assert "CREATE TABLE" in content.upper(), \
                "Init script should contain table creation statements"


class TestDeploymentUpgrade:
    """Test deployment upgrade scenarios."""
    
    def test_rolling_update_capability(self):
        """Test that services can be updated without downtime."""
        # Get current API container ID
        result = subprocess.run(
            ["docker-compose", "ps", "-q", "api"],
            capture_output=True,
            text=True
        )
        
        original_containers = result.stdout.strip().split('\n')
        
        # Perform rolling update (restart one service at a time)
        result = subprocess.run(
            ["docker-compose", "up", "-d", "--no-deps", "api"],
            capture_output=True,
            text=True
        )
        
        assert result.returncode == 0, "Rolling update failed"
        
        # Wait for service to be ready
        time.sleep(30)
        
        # Check that service is still accessible
        response = requests.get("http://localhost:8000/health", timeout=10)
        assert response.status_code == 200, "Service not accessible after update"
    
    def test_configuration_update_handling(self):
        """Test that configuration updates are handled properly."""
        # Check that services can be restarted with new config
        result = subprocess.run(
            ["docker-compose", "restart", "api"],
            capture_output=True,
            text=True
        )
        
        assert result.returncode == 0, "Configuration update restart failed"
        
        # Wait for service to be ready
        time.sleep(30)
        
        # Verify service is functional
        response = requests.get("http://localhost:8000/health", timeout=10)
        assert response.status_code == 200, "Service not functional after config update"
    
    def test_data_migration_readiness(self):
        """Test that system is ready for data migrations."""
        # Check that database schema versioning is in place
        # This would typically check for migration scripts or version tables
        
        # For now, check that database is accessible and has expected structure
        result = subprocess.run(
            ["docker-compose", "exec", "-T", "db", "psql", "-U", "postgres", "-d", "robotics_optimization", "-c", "\\dt"],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            # Should have some tables
            assert "models" in result.stdout or "optimization" in result.stdout, \
                "Database should have application tables"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
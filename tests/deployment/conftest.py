"""
Pytest configuration for deployment validation tests.
"""

import pytest
import subprocess
import time
import requests
import os


@pytest.fixture(scope="session", autouse=True)
def ensure_deployment_ready():
    """Ensure deployment is ready before running tests."""
    # Check if we should skip deployment tests when services aren't available
    skip_if_no_services = os.getenv("SKIP_DEPLOYMENT_IF_NO_SERVICES", "true").lower() == "true"
    
    print("\n🔍 Checking deployment readiness...")
    
    # Quick check if services are available
    try:
        response = requests.get("http://localhost:8000/health", timeout=2)
        if response.status_code == 200:
            print("✅ API service is ready")
            return  # Services are ready, proceed with tests
    except requests.exceptions.RequestException:
        pass
    
    # Services not immediately available
    if skip_if_no_services:
        pytest.skip("Deployment services not available. Set SKIP_DEPLOYMENT_IF_NO_SERVICES=false to wait for services.")
    
    # Wait for services to be healthy
    max_wait = 300  # 5 minutes
    start_time = time.time()
    
    while time.time() - start_time < max_wait:
        try:
            # Check if API is responding
            response = requests.get("http://localhost:8000/health", timeout=5)
            if response.status_code == 200:
                print("✅ API service is ready")
                break
        except requests.exceptions.RequestException:
            pass
        
        print(f"⏳ Waiting for services... ({int(time.time() - start_time)}/{max_wait}s)")
        time.sleep(10)
    else:
        pytest.fail("Services not ready within timeout period")
    
    # Additional readiness checks
    try:
        # Check frontend
        response = requests.get("http://localhost:3000", timeout=5)
        if response.status_code == 200:
            print("✅ Frontend service is ready")
    except requests.exceptions.RequestException:
        print("⚠️  Frontend service not accessible")
    
    # Check database
    result = subprocess.run(
        ["docker-compose", "exec", "-T", "db", "pg_isready", "-U", "postgres"],
        capture_output=True,
        text=True
    )
    if result.returncode == 0:
        print("✅ Database service is ready")
    else:
        print("⚠️  Database service not ready")
    
    # Check Redis
    result = subprocess.run(
        ["docker-compose", "exec", "-T", "redis", "redis-cli", "ping"],
        capture_output=True,
        text=True
    )
    if result.returncode == 0:
        print("✅ Redis service is ready")
    else:
        print("⚠️  Redis service not ready")
    
    print("🎉 Deployment readiness check completed")


@pytest.fixture(scope="session")
def deployment_environment():
    """Get the current deployment environment."""
    return os.getenv("ENVIRONMENT", "development")


@pytest.fixture(scope="session")
def service_urls():
    """Get service URLs for testing."""
    return {
        "api": os.getenv("API_URL", "http://localhost:8000"),
        "frontend": os.getenv("FRONTEND_URL", "http://localhost:3000"),
        "websocket": os.getenv("WS_URL", "ws://localhost:8000/ws"),
    }


@pytest.fixture
def cleanup_test_data():
    """Cleanup test data after tests."""
    yield
    
    # Cleanup uploaded test files
    uploads_dir = "uploads"
    if os.path.exists(uploads_dir):
        test_files = [f for f in os.listdir(uploads_dir) if "test" in f.lower()]
        for test_file in test_files:
            try:
                os.remove(os.path.join(uploads_dir, test_file))
            except OSError:
                pass


def pytest_configure(config):
    """Configure pytest for deployment tests."""
    # Add custom markers
    config.addinivalue_line(
        "markers", "deployment: mark test as deployment validation test"
    )
    config.addinivalue_line(
        "markers", "production: mark test as production-only test"
    )
    config.addinivalue_line(
        "markers", "development: mark test as development-only test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection for deployment tests."""
    # Skip production tests in non-production environments
    environment = os.getenv("ENVIRONMENT", "development")
    
    for item in items:
        if "production" in item.keywords and environment != "production":
            item.add_marker(pytest.mark.skip(reason="Production test in non-production environment"))
        
        if "development" in item.keywords and environment == "production":
            item.add_marker(pytest.mark.skip(reason="Development test in production environment"))


@pytest.fixture(scope="session")
def docker_services_info():
    """Get information about running Docker services."""
    result = subprocess.run(
        ["docker-compose", "ps", "--format", "json"],
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        try:
            import json
            services = []
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    services.append(json.loads(line))
            return services
        except json.JSONDecodeError:
            return []
    
    return []


@pytest.fixture
def wait_for_service_ready():
    """Wait for a service to be ready."""
    def _wait_for_service(url, timeout=60):
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    return True
            except requests.exceptions.RequestException:
                pass
            time.sleep(2)
        return False
    
    return _wait_for_service
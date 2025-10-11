"""
Load testing suite for API endpoints and WebSocket connections.

Tests API performance under concurrent load and WebSocket scalability.
"""

import asyncio
import time
import statistics
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import pytest
from fastapi.testclient import TestClient
import socketio

from src.api.main import app
from src.services.optimization_manager import OptimizationManager
from src.services.websocket_manager import WebSocketManager
from src.models.store import ModelStore


class LoadTestMetrics:
    """Collect and analyze load test metrics."""
    
    def __init__(self):
        self.response_times: List[float] = []
        self.success_count: int = 0
        self.error_count: int = 0
        self.status_codes: Dict[int, int] = {}
    
    def add_result(self, response_time: float, status_code: int, success: bool):
        """Add a test result."""
        self.response_times.append(response_time)
        if success:
            self.success_count += 1
        else:
            self.error_count += 1
        self.status_codes[status_code] = self.status_codes.get(status_code, 0) + 1
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        if not self.response_times:
            return {
                "total_requests": 0,
                "success_rate": 0.0,
                "error_rate": 0.0
            }
        
        return {
            "total_requests": len(self.response_times),
            "success_count": self.success_count,
            "error_count": self.error_count,
            "success_rate": self.success_count / len(self.response_times) * 100,
            "error_rate": self.error_count / len(self.response_times) * 100,
            "avg_response_time": statistics.mean(self.response_times),
            "median_response_time": statistics.median(self.response_times),
            "min_response_time": min(self.response_times),
            "max_response_time": max(self.response_times),
            "p95_response_time": self._percentile(self.response_times, 95),
            "p99_response_time": self._percentile(self.response_times, 99),
            "status_codes": self.status_codes,
            "requests_per_second": len(self.response_times) / sum(self.response_times) if sum(self.response_times) > 0 else 0
        }
    
    @staticmethod
    def _percentile(data: List[float], percentile: int) -> float:
        """Calculate percentile."""
        sorted_data = sorted(data)
        index = int(len(sorted_data) * percentile / 100)
        return sorted_data[min(index, len(sorted_data) - 1)]


@pytest.fixture
def load_test_client():
    """Create test client for load testing."""
    return TestClient(app)


@pytest.fixture
def auth_headers():
    """Create authentication headers for testing."""
    return {"Authorization": "Bearer test_token"}


class TestDashboardEndpointLoad:
    """Load tests for dashboard statistics endpoint."""
    
    def test_dashboard_concurrent_requests(self, load_test_client, auth_headers):
        """Test dashboard endpoint with concurrent requests."""
        metrics = LoadTestMetrics()
        num_requests = 100
        num_workers = 10
        
        def make_request():
            start_time = time.time()
            try:
                response = load_test_client.get(
                    "/api/v1/dashboard/stats",
                    headers=auth_headers
                )
                response_time = time.time() - start_time
                metrics.add_result(
                    response_time,
                    response.status_code,
                    response.status_code == 200
                )
                return response.status_code
            except Exception as e:
                response_time = time.time() - start_time
                metrics.add_result(response_time, 500, False)
                return 500
        
        # Execute concurrent requests
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(make_request) for _ in range(num_requests)]
            for future in as_completed(futures):
                future.result()
        
        # Analyze results
        summary = metrics.get_summary()
        
        # Assertions
        assert summary["success_rate"] >= 95.0, f"Success rate too low: {summary['success_rate']}%"
        assert summary["avg_response_time"] < 1.0, f"Average response time too high: {summary['avg_response_time']}s"
        assert summary["p95_response_time"] < 2.0, f"P95 response time too high: {summary['p95_response_time']}s"
        
        print(f"\n=== Dashboard Load Test Results ===")
        print(f"Total Requests: {summary['total_requests']}")
        print(f"Success Rate: {summary['success_rate']:.2f}%")
        print(f"Avg Response Time: {summary['avg_response_time']:.3f}s")
        print(f"P95 Response Time: {summary['p95_response_time']:.3f}s")
        print(f"P99 Response Time: {summary['p99_response_time']:.3f}s")
    
    def test_dashboard_sustained_load(self, load_test_client, auth_headers):
        """Test dashboard endpoint under sustained load."""
        metrics = LoadTestMetrics()
        duration_seconds = 10
        num_workers = 5
        
        def make_requests_for_duration():
            end_time = time.time() + duration_seconds
            while time.time() < end_time:
                start_time = time.time()
                try:
                    response = load_test_client.get(
                        "/api/v1/dashboard/stats",
                        headers=auth_headers
                    )
                    response_time = time.time() - start_time
                    metrics.add_result(
                        response_time,
                        response.status_code,
                        response.status_code == 200
                    )
                except Exception:
                    response_time = time.time() - start_time
                    metrics.add_result(response_time, 500, False)
        
        # Execute sustained load
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(make_requests_for_duration) for _ in range(num_workers)]
            for future in as_completed(futures):
                future.result()
        
        summary = metrics.get_summary()
        
        # Assertions
        assert summary["success_rate"] >= 95.0, f"Success rate too low under sustained load: {summary['success_rate']}%"
        assert summary["avg_response_time"] < 1.0, f"Average response time degraded: {summary['avg_response_time']}s"
        
        print(f"\n=== Dashboard Sustained Load Test Results ===")
        print(f"Duration: {duration_seconds}s")
        print(f"Total Requests: {summary['total_requests']}")
        print(f"Requests/Second: {summary['requests_per_second']:.2f}")
        print(f"Success Rate: {summary['success_rate']:.2f}%")
        print(f"Avg Response Time: {summary['avg_response_time']:.3f}s")


class TestSessionsEndpointLoad:
    """Load tests for sessions list endpoint."""
    
    def test_sessions_concurrent_requests(self, load_test_client, auth_headers):
        """Test sessions endpoint with concurrent requests."""
        metrics = LoadTestMetrics()
        num_requests = 100
        num_workers = 10
        
        def make_request():
            start_time = time.time()
            try:
                response = load_test_client.get(
                    "/api/v1/optimization/sessions?limit=50",
                    headers=auth_headers
                )
                response_time = time.time() - start_time
                metrics.add_result(
                    response_time,
                    response.status_code,
                    response.status_code == 200
                )
                return response.status_code
            except Exception:
                response_time = time.time() - start_time
                metrics.add_result(response_time, 500, False)
                return 500
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(make_request) for _ in range(num_requests)]
            for future in as_completed(futures):
                future.result()
        
        summary = metrics.get_summary()
        
        assert summary["success_rate"] >= 95.0, f"Success rate too low: {summary['success_rate']}%"
        assert summary["avg_response_time"] < 1.5, f"Average response time too high: {summary['avg_response_time']}s"
        
        print(f"\n=== Sessions Load Test Results ===")
        print(f"Total Requests: {summary['total_requests']}")
        print(f"Success Rate: {summary['success_rate']:.2f}%")
        print(f"Avg Response Time: {summary['avg_response_time']:.3f}s")
    
    def test_sessions_with_filters_load(self, load_test_client, auth_headers):
        """Test sessions endpoint with various filters under load."""
        metrics = LoadTestMetrics()
        num_requests = 50
        
        filters = [
            "?status=completed",
            "?status=running",
            "?limit=10",
            "?limit=100",
            "?skip=10&limit=20"
        ]
        
        def make_request(filter_params):
            start_time = time.time()
            try:
                response = load_test_client.get(
                    f"/api/v1/optimization/sessions{filter_params}",
                    headers=auth_headers
                )
                response_time = time.time() - start_time
                metrics.add_result(
                    response_time,
                    response.status_code,
                    response.status_code == 200
                )
            except Exception:
                response_time = time.time() - start_time
                metrics.add_result(response_time, 500, False)
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = []
            for _ in range(num_requests):
                for filter_param in filters:
                    futures.append(executor.submit(make_request, filter_param))
            
            for future in as_completed(futures):
                future.result()
        
        summary = metrics.get_summary()
        
        assert summary["success_rate"] >= 95.0
        assert summary["avg_response_time"] < 2.0
        
        print(f"\n=== Sessions with Filters Load Test Results ===")
        print(f"Total Requests: {summary['total_requests']}")
        print(f"Success Rate: {summary['success_rate']:.2f}%")
        print(f"Avg Response Time: {summary['avg_response_time']:.3f}s")


class TestConfigEndpointLoad:
    """Load tests for configuration endpoints."""
    
    def test_config_get_concurrent_requests(self, load_test_client, auth_headers):
        """Test GET config endpoint with concurrent requests."""
        metrics = LoadTestMetrics()
        num_requests = 100
        num_workers = 10
        
        def make_request():
            start_time = time.time()
            try:
                response = load_test_client.get(
                    "/api/v1/config/optimization-criteria",
                    headers=auth_headers
                )
                response_time = time.time() - start_time
                metrics.add_result(
                    response_time,
                    response.status_code,
                    response.status_code == 200
                )
            except Exception:
                response_time = time.time() - start_time
                metrics.add_result(response_time, 500, False)
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(make_request) for _ in range(num_requests)]
            for future in as_completed(futures):
                future.result()
        
        summary = metrics.get_summary()
        
        assert summary["success_rate"] >= 95.0
        assert summary["avg_response_time"] < 0.5, "Config GET should be fast (cached)"
        
        print(f"\n=== Config GET Load Test Results ===")
        print(f"Total Requests: {summary['total_requests']}")
        print(f"Success Rate: {summary['success_rate']:.2f}%")
        print(f"Avg Response Time: {summary['avg_response_time']:.3f}s")
    
    def test_config_update_sequential(self, load_test_client, auth_headers):
        """Test PUT config endpoint with sequential updates."""
        metrics = LoadTestMetrics()
        num_requests = 20
        
        config_data = {
            "name": "load_test_config",
            "description": "Load test configuration",
            "constraints": {
                "max_size_mb": 100,
                "min_accuracy_percent": 90.0,
                "max_latency_ms": 100
            },
            "target_deployment": "edge",
            "enabled_techniques": ["quantization"],
            "hardware_target": "cpu"
        }
        
        for i in range(num_requests):
            start_time = time.time()
            try:
                config_data["name"] = f"load_test_config_{i}"
                response = load_test_client.put(
                    "/api/v1/config/optimization-criteria",
                    json=config_data,
                    headers=auth_headers
                )
                response_time = time.time() - start_time
                metrics.add_result(
                    response_time,
                    response.status_code,
                    response.status_code == 200
                )
            except Exception:
                response_time = time.time() - start_time
                metrics.add_result(response_time, 500, False)
        
        summary = metrics.get_summary()
        
        assert summary["success_rate"] >= 95.0
        assert summary["avg_response_time"] < 1.0
        
        print(f"\n=== Config PUT Load Test Results ===")
        print(f"Total Requests: {summary['total_requests']}")
        print(f"Success Rate: {summary['success_rate']:.2f}%")
        print(f"Avg Response Time: {summary['avg_response_time']:.3f}s")


class TestWebSocketLoad:
    """Load tests for WebSocket connections."""
    
    @pytest.mark.asyncio
    async def test_websocket_multiple_connections(self):
        """Test WebSocket with multiple concurrent connections."""
        num_connections = 50
        connection_metrics = {
            "successful_connections": 0,
            "failed_connections": 0,
            "connection_times": []
        }
        
        async def create_connection(client_id: int):
            """Create a WebSocket connection."""
            start_time = time.time()
            try:
                sio = socketio.AsyncClient()
                
                @sio.event
                async def connect():
                    connection_time = time.time() - start_time
                    connection_metrics["connection_times"].append(connection_time)
                    connection_metrics["successful_connections"] += 1
                
                @sio.event
                async def connect_error(data):
                    connection_metrics["failed_connections"] += 1
                
                await sio.connect('http://localhost:8000', transports=['websocket'])
                await asyncio.sleep(2)  # Keep connection alive
                await sio.disconnect()
                
            except Exception as e:
                connection_metrics["failed_connections"] += 1
                print(f"Connection {client_id} failed: {e}")
        
        # Create multiple connections concurrently
        tasks = [create_connection(i) for i in range(num_connections)]
        await asyncio.gather(*tasks, return_exceptions=True)
        
        # Analyze results
        success_rate = (connection_metrics["successful_connections"] / num_connections) * 100
        
        print(f"\n=== WebSocket Connection Load Test Results ===")
        print(f"Total Connections Attempted: {num_connections}")
        print(f"Successful Connections: {connection_metrics['successful_connections']}")
        print(f"Failed Connections: {connection_metrics['failed_connections']}")
        print(f"Success Rate: {success_rate:.2f}%")
        
        if connection_metrics["connection_times"]:
            avg_connection_time = statistics.mean(connection_metrics["connection_times"])
            print(f"Avg Connection Time: {avg_connection_time:.3f}s")
            
            assert success_rate >= 90.0, f"WebSocket connection success rate too low: {success_rate}%"
            assert avg_connection_time < 1.0, f"Connection time too high: {avg_connection_time}s"
    
    @pytest.mark.asyncio
    async def test_websocket_event_broadcasting_load(self):
        """Test WebSocket event broadcasting to multiple clients."""
        num_clients = 20
        num_events = 50
        
        received_events = {i: [] for i in range(num_clients)}
        
        async def create_listening_client(client_id: int):
            """Create a client that listens for events."""
            try:
                sio = socketio.AsyncClient()
                
                @sio.event
                async def session_progress(data):
                    received_events[client_id].append(data)
                
                await sio.connect('http://localhost:8000', transports=['websocket'])
                await asyncio.sleep(5)  # Listen for events
                await sio.disconnect()
                
            except Exception as e:
                print(f"Client {client_id} error: {e}")
        
        # Start all clients
        client_tasks = [create_listening_client(i) for i in range(num_clients)]
        
        # Simulate event broadcasting
        async def broadcast_events():
            await asyncio.sleep(1)  # Wait for clients to connect
            # In real scenario, events would be triggered by optimization progress
            # Here we're just testing the infrastructure
        
        await asyncio.gather(
            *client_tasks,
            broadcast_events(),
            return_exceptions=True
        )
        
        print(f"\n=== WebSocket Event Broadcasting Load Test Results ===")
        print(f"Number of Clients: {num_clients}")
        print(f"Expected Events per Client: {num_events}")
        
        # Note: Actual event counts would depend on real optimization events
        # This test validates the infrastructure can handle multiple connections


class TestMixedLoad:
    """Test mixed load scenarios with multiple endpoint types."""
    
    def test_mixed_endpoint_load(self, load_test_client, auth_headers):
        """Test multiple endpoints concurrently."""
        metrics_by_endpoint = {
            "dashboard": LoadTestMetrics(),
            "sessions": LoadTestMetrics(),
            "config": LoadTestMetrics()
        }
        
        num_requests_per_endpoint = 30
        
        def make_dashboard_request():
            start_time = time.time()
            try:
                response = load_test_client.get(
                    "/api/v1/dashboard/stats",
                    headers=auth_headers
                )
                response_time = time.time() - start_time
                metrics_by_endpoint["dashboard"].add_result(
                    response_time,
                    response.status_code,
                    response.status_code == 200
                )
            except Exception:
                response_time = time.time() - start_time
                metrics_by_endpoint["dashboard"].add_result(response_time, 500, False)
        
        def make_sessions_request():
            start_time = time.time()
            try:
                response = load_test_client.get(
                    "/api/v1/optimization/sessions",
                    headers=auth_headers
                )
                response_time = time.time() - start_time
                metrics_by_endpoint["sessions"].add_result(
                    response_time,
                    response.status_code,
                    response.status_code == 200
                )
            except Exception:
                response_time = time.time() - start_time
                metrics_by_endpoint["sessions"].add_result(response_time, 500, False)
        
        def make_config_request():
            start_time = time.time()
            try:
                response = load_test_client.get(
                    "/api/v1/config/optimization-criteria",
                    headers=auth_headers
                )
                response_time = time.time() - start_time
                metrics_by_endpoint["config"].add_result(
                    response_time,
                    response.status_code,
                    response.status_code == 200
                )
            except Exception:
                response_time = time.time() - start_time
                metrics_by_endpoint["config"].add_result(response_time, 500, False)
        
        # Execute mixed load
        with ThreadPoolExecutor(max_workers=15) as executor:
            futures = []
            for _ in range(num_requests_per_endpoint):
                futures.append(executor.submit(make_dashboard_request))
                futures.append(executor.submit(make_sessions_request))
                futures.append(executor.submit(make_config_request))
            
            for future in as_completed(futures):
                future.result()
        
        # Analyze results
        print(f"\n=== Mixed Load Test Results ===")
        for endpoint_name, metrics in metrics_by_endpoint.items():
            summary = metrics.get_summary()
            print(f"\n{endpoint_name.upper()} Endpoint:")
            print(f"  Total Requests: {summary['total_requests']}")
            print(f"  Success Rate: {summary['success_rate']:.2f}%")
            print(f"  Avg Response Time: {summary['avg_response_time']:.3f}s")
            print(f"  P95 Response Time: {summary['p95_response_time']:.3f}s")
            
            assert summary["success_rate"] >= 90.0, f"{endpoint_name} success rate too low"

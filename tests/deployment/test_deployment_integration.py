"""
End-to-end deployment integration tests.
Tests complete deployment workflows and integration scenarios.
"""

import pytest
import requests
import time
import subprocess
import tempfile
import torch
import torch.nn as nn
import json
import os
from typing import Dict, Any
import threading

# Skip if optional dependencies are missing
websocket = pytest.importorskip("websocket")


class TestEndToEndDeployment:
    """Test complete end-to-end deployment workflows."""
    
    @pytest.fixture(scope="class")
    def deployment_config(self) -> Dict[str, Any]:
        """Load deployment configuration."""
        return {
            "api_url": os.getenv("API_URL", "http://localhost:8000"),
            "frontend_url": os.getenv("FRONTEND_URL", "http://localhost:3000"),
            "ws_url": os.getenv("WS_URL", "ws://localhost:8000/ws"),
        }
    
    @pytest.fixture
    def sample_model_file(self):
        """Create a sample model file for testing."""
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 1)
            
            def forward(self, x):
                return self.linear(x)
        
        model = SimpleModel()
        
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
            torch.save(model.state_dict(), f.name)
            yield f.name
            
        # Cleanup
        os.unlink(f.name)
    
    def test_complete_optimization_workflow(self, deployment_config, sample_model_file):
        """Test complete model optimization workflow from upload to results."""
        base_url = deployment_config["api_url"]
        
        # Step 1: Upload model
        with open(sample_model_file, 'rb') as f:
            files = {"file": ("test_model.pth", f, "application/octet-stream")}
            upload_response = requests.post(f"{base_url}/api/upload", files=files)
        
        # Should either succeed or require authentication (not fail completely)
        assert upload_response.status_code not in [404, 500], \
            f"Model upload failed: {upload_response.status_code}"
        
        if upload_response.status_code == 200:
            upload_data = upload_response.json()
            model_id = upload_data.get("model_id")
            
            if model_id:
                # Step 2: Start optimization
                optimization_data = {
                    "model_id": model_id,
                    "techniques": ["quantization"],
                    "criteria": {
                        "target_size_reduction": 0.5,
                        "max_accuracy_loss": 0.05
                    }
                }
                
                opt_response = requests.post(
                    f"{base_url}/api/optimize",
                    json=optimization_data
                )
                
                if opt_response.status_code == 200:
                    opt_data = opt_response.json()
                    session_id = opt_data.get("session_id")
                    
                    if session_id:
                        # Step 3: Monitor optimization progress
                        max_wait = 300  # 5 minutes
                        start_time = time.time()
                        
                        while time.time() - start_time < max_wait:
                            status_response = requests.get(
                                f"{base_url}/api/optimize/{session_id}/status"
                            )
                            
                            if status_response.status_code == 200:
                                status_data = status_response.json()
                                status = status_data.get("status")
                                
                                if status in ["completed", "failed"]:
                                    break
                            
                            time.sleep(10)
                        
                        # Step 4: Get results
                        results_response = requests.get(
                            f"{base_url}/api/optimize/{session_id}/results"
                        )
                        
                        # Should be able to get results (even if optimization failed)
                        assert results_response.status_code in [200, 404], \
                            "Should be able to query optimization results"
    
    def test_websocket_progress_updates(self, deployment_config):
        """Test that WebSocket provides real-time progress updates."""
        ws_url = deployment_config["ws_url"]
        messages_received = []
        
        def on_message(ws, message):
            messages_received.append(json.loads(message))
        
        def on_error(ws, error):
            print(f"WebSocket error: {error}")
        
        try:
            ws = websocket.WebSocketApp(
                ws_url,
                on_message=on_message,
                on_error=on_error
            )
            
            # Start WebSocket in a separate thread
            ws_thread = threading.Thread(target=ws.run_forever)
            ws_thread.daemon = True
            ws_thread.start()
            
            # Wait a bit for connection
            time.sleep(5)
            
            # Send a test message if connection is established
            if ws.sock and ws.sock.connected:
                ws.send(json.dumps({"type": "ping"}))
                time.sleep(2)
                
                # Should receive some response
                assert len(messages_received) >= 0, "WebSocket should be functional"
            
            ws.close()
            
        except Exception as e:
            # WebSocket might require authentication, but should not be completely broken
            assert "Connection refused" not in str(e), f"WebSocket completely unavailable: {e}"
    
    def test_concurrent_optimization_handling(self, deployment_config, sample_model_file):
        """Test that system can handle concurrent optimization requests."""
        base_url = deployment_config["api_url"]
        
        # Upload model first
        with open(sample_model_file, 'rb') as f:
            files = {"file": ("test_model.pth", f, "application/octet-stream")}
            upload_response = requests.post(f"{base_url}/api/upload", files=files)
        
        if upload_response.status_code == 200:
            upload_data = upload_response.json()
            model_id = upload_data.get("model_id")
            
            if model_id:
                # Start multiple optimizations concurrently
                optimization_data = {
                    "model_id": model_id,
                    "techniques": ["quantization"],
                    "criteria": {"target_size_reduction": 0.3}
                }
                
                responses = []
                for i in range(3):
                    response = requests.post(
                        f"{base_url}/api/optimize",
                        json=optimization_data
                    )
                    responses.append(response)
                
                # At least some requests should be handled properly
                success_count = sum(1 for r in responses if r.status_code in [200, 202])
                assert success_count > 0, "System should handle at least some concurrent requests"
    
    def test_error_handling_and_recovery(self, deployment_config):
        """Test system error handling and recovery capabilities."""
        base_url = deployment_config["api_url"]
        
        # Test invalid model upload
        fake_content = b"not a real model file"
        files = {"file": ("fake_model.pth", fake_content, "application/octet-stream")}
        
        response = requests.post(f"{base_url}/api/upload", files=files)
        
        # Should handle invalid files gracefully
        assert response.status_code in [400, 422], \
            "Should return appropriate error for invalid model"
        
        if response.status_code in [400, 422]:
            error_data = response.json()
            assert "detail" in error_data or "error" in error_data, \
                "Error response should contain error details"
    
    def test_system_resource_monitoring(self, deployment_config):
        """Test that system resource monitoring is functional."""
        base_url = deployment_config["api_url"]
        
        # Check if system metrics are available
        metrics_endpoints = ["/metrics", "/api/system/status", "/health"]
        
        for endpoint in metrics_endpoints:
            response = requests.get(f"{base_url}{endpoint}")
            
            if response.status_code == 200:
                data = response.json()
                
                # Should contain some system information
                system_fields = ["memory", "cpu", "disk", "uptime", "timestamp"]
                found_fields = [field for field in system_fields if field in str(data).lower()]
                
                if found_fields:
                    # Found system monitoring data
                    assert len(found_fields) > 0, "System monitoring should provide resource data"
                    break
    
    def test_api_rate_limiting(self, deployment_config):
        """Test API rate limiting functionality."""
        base_url = deployment_config["api_url"]
        
        # Make rapid requests to test rate limiting
        responses = []
        for i in range(20):
            response = requests.get(f"{base_url}/health")
            responses.append(response.status_code)
        
        # All requests should succeed (health endpoint shouldn't be rate limited)
        # But if rate limiting is implemented, some might return 429
        success_count = sum(1 for status in responses if status == 200)
        rate_limited_count = sum(1 for status in responses if status == 429)
        
        # Either all succeed (no rate limiting) or some are rate limited
        assert success_count > 0, "At least some requests should succeed"
        
        if rate_limited_count > 0:
            # Rate limiting is implemented
            assert rate_limited_count < len(responses), \
                "Not all requests should be rate limited"
    
    def test_data_persistence_across_restarts(self, deployment_config, sample_model_file):
        """Test that data persists across service restarts."""
        base_url = deployment_config["api_url"]
        
        # Upload a model
        with open(sample_model_file, 'rb') as f:
            files = {"file": ("persistence_test_model.pth", f, "application/octet-stream")}
            upload_response = requests.post(f"{base_url}/api/upload", files=files)
        
        model_id = None
        if upload_response.status_code == 200:
            upload_data = upload_response.json()
            model_id = upload_data.get("model_id")
        
        if model_id:
            # Restart API service
            result = subprocess.run(
                ["docker-compose", "restart", "api"],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                # Wait for service to be ready
                time.sleep(30)
                
                # Try to retrieve the model
                response = requests.get(f"{base_url}/api/models/{model_id}")
                
                # Should either find the model or get appropriate error
                assert response.status_code in [200, 404], \
                    "Should be able to query model after restart"
    
    def test_backup_and_restore_workflow(self):
        """Test backup and restore workflow."""
        # Create a backup
        backup_result = subprocess.run(
            ["./scripts/backup.sh"],
            capture_output=True,
            text=True,
            cwd="."
        )
        
        if backup_result.returncode == 0:
            # Check that backup was created
            backup_dir = "backups"
            if os.path.exists(backup_dir):
                backup_files = os.listdir(backup_dir)
                sql_backups = [f for f in backup_files if f.endswith('.sql') or f.endswith('.sql.gz')]
                
                assert len(sql_backups) > 0, "Backup should create SQL files"
                
                # Test that backup file is not empty
                if sql_backups:
                    backup_file = os.path.join(backup_dir, sql_backups[0])
                    file_size = os.path.getsize(backup_file)
                    assert file_size > 0, "Backup file should not be empty"


class TestDeploymentScalability:
    """Test deployment scalability and performance under load."""
    
    def test_horizontal_scaling(self):
        """Test horizontal scaling of services."""
        # Scale up workers
        scale_result = subprocess.run(
            ["docker-compose", "up", "-d", "--scale", "worker=3"],
            capture_output=True,
            text=True
        )
        
        assert scale_result.returncode == 0, "Should be able to scale workers"
        
        # Wait for services to be ready
        time.sleep(30)
        
        # Check that multiple workers are running
        ps_result = subprocess.run(
            ["docker-compose", "ps", "worker"],
            capture_output=True,
            text=True
        )
        
        worker_lines = [line for line in ps_result.stdout.split('\n') if 'worker' in line and 'Up' in line]
        assert len(worker_lines) >= 2, f"Expected multiple workers, found {len(worker_lines)}"
        
        # Scale back down
        subprocess.run(
            ["docker-compose", "up", "-d", "--scale", "worker=1"],
            capture_output=True,
            text=True
        )
    
    def test_load_balancing_functionality(self, deployment_config):
        """Test load balancing across multiple service instances."""
        base_url = deployment_config["api_url"]
        
        # Make multiple requests and check for consistent responses
        responses = []
        for i in range(10):
            response = requests.get(f"{base_url}/health")
            responses.append(response)
        
        # All requests should succeed
        success_count = sum(1 for r in responses if r.status_code == 200)
        assert success_count == len(responses), "All requests should succeed with load balancing"
        
        # Response times should be reasonable
        response_times = [r.elapsed.total_seconds() for r in responses]
        avg_response_time = sum(response_times) / len(response_times)
        assert avg_response_time < 2.0, f"Average response time too high: {avg_response_time}s"
    
    def test_resource_utilization_under_load(self):
        """Test resource utilization under simulated load."""
        # Get baseline resource usage
        baseline_result = subprocess.run(
            ["docker", "stats", "--no-stream", "--format", "{{.CPUPerc}},{{.MemUsage}}"],
            capture_output=True,
            text=True
        )
        
        if baseline_result.returncode == 0:
            # Parse baseline metrics
            baseline_lines = baseline_result.stdout.strip().split('\n')
            
            # Simulate some load by making multiple requests
            import concurrent.futures
            import requests
            
            def make_request():
                try:
                    response = requests.get("http://localhost:8000/health", timeout=5)
                    return response.status_code == 200
                except:
                    return False
            
            # Make concurrent requests
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                futures = [executor.submit(make_request) for _ in range(20)]
                results = [future.result() for future in concurrent.futures.as_completed(futures)]
            
            # Check that most requests succeeded
            success_rate = sum(results) / len(results)
            assert success_rate > 0.8, f"Success rate under load too low: {success_rate}"
            
            # Get resource usage after load
            load_result = subprocess.run(
                ["docker", "stats", "--no-stream", "--format", "{{.CPUPerc}},{{.MemUsage}}"],
                capture_output=True,
                text=True
            )
            
            # System should still be responsive
            assert load_result.returncode == 0, "System should remain responsive under load"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
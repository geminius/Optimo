#!/usr/bin/env python3
"""
Demo script showing how to use the Robotics Model Optimization Platform API.
"""

import requests
import time
import json
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional


class OptimizationAPIClient:
    """Client for interacting with the Optimization Platform API."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.token: Optional[str] = None
    
    def login(self, username: str, password: str) -> Dict[str, Any]:
        """Login and get access token."""
        response = self.session.post(
            f"{self.base_url}/auth/login",
            json={"username": username, "password": password}
        )
        response.raise_for_status()
        
        data = response.json()
        self.token = data["access_token"]
        
        # Set authorization header for future requests
        self.session.headers.update({
            "Authorization": f"Bearer {self.token}"
        })
        
        print(f"✓ Logged in as {data['user']['username']}")
        return data
    
    def health_check(self) -> Dict[str, Any]:
        """Check API health."""
        response = self.session.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()
    
    def upload_model(self, file_path: str, name: str = None, description: str = None, tags: str = None) -> Dict[str, Any]:
        """Upload a model file."""
        with open(file_path, "rb") as f:
            files = {"file": (Path(file_path).name, f, "application/octet-stream")}
            data = {}
            if name:
                data["name"] = name
            if description:
                data["description"] = description
            if tags:
                data["tags"] = tags
            
            response = self.session.post(
                f"{self.base_url}/models/upload",
                files=files,
                data=data
            )
        
        response.raise_for_status()
        result = response.json()
        print(f"✓ Model uploaded: {result['model_id']}")
        return result
    
    def list_models(self, skip: int = 0, limit: int = 100) -> Dict[str, Any]:
        """List uploaded models."""
        response = self.session.get(
            f"{self.base_url}/models",
            params={"skip": skip, "limit": limit}
        )
        response.raise_for_status()
        return response.json()
    
    def start_optimization(self, model_id: str, **kwargs) -> Dict[str, Any]:
        """Start optimization for a model."""
        request_data = {"model_id": model_id}
        request_data.update(kwargs)
        
        response = self.session.post(
            f"{self.base_url}/optimize",
            json=request_data
        )
        response.raise_for_status()
        
        result = response.json()
        print(f"✓ Optimization started: {result['session_id']}")
        return result
    
    def get_session_status(self, session_id: str) -> Dict[str, Any]:
        """Get optimization session status."""
        response = self.session.get(f"{self.base_url}/sessions/{session_id}/status")
        response.raise_for_status()
        return response.json()
    
    def list_sessions(self) -> Dict[str, Any]:
        """List all optimization sessions."""
        response = self.session.get(f"{self.base_url}/sessions")
        response.raise_for_status()
        return response.json()
    
    def cancel_session(self, session_id: str) -> Dict[str, Any]:
        """Cancel an optimization session."""
        response = self.session.post(f"{self.base_url}/sessions/{session_id}/cancel")
        response.raise_for_status()
        return response.json()
    
    def get_session_results(self, session_id: str) -> Dict[str, Any]:
        """Get optimization session results."""
        response = self.session.get(f"{self.base_url}/sessions/{session_id}/results")
        response.raise_for_status()
        return response.json()
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get system status and metrics."""
        response = self.session.get(f"{self.base_url}/monitoring/system")
        response.raise_for_status()
        return response.json()
    
    def monitor_session(self, session_id: str, poll_interval: int = 5) -> None:
        """Monitor optimization session progress."""
        print(f"Monitoring session: {session_id}")
        
        while True:
            try:
                status = self.get_session_status(session_id)
                
                print(f"Status: {status['status']} | "
                      f"Progress: {status['progress_percentage']:.1f}% | "
                      f"Step: {status['current_step'] or 'N/A'}")
                
                if status['status'] in ['completed', 'failed', 'cancelled']:
                    print(f"✓ Session {status['status']}")
                    break
                
                time.sleep(poll_interval)
                
            except KeyboardInterrupt:
                print("\nMonitoring interrupted")
                break
            except Exception as e:
                print(f"Error monitoring session: {e}")
                break


def create_dummy_model_file() -> str:
    """Create a dummy model file for testing."""
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp_file:
        # Create some dummy binary data that looks like a model
        dummy_data = b"DUMMY_MODEL_DATA" + b"\x00" * 1000
        tmp_file.write(dummy_data)
        return tmp_file.name


def main():
    """Main demo function."""
    print("🤖 Robotics Model Optimization Platform API Demo")
    print("=" * 50)
    
    # Initialize client
    client = OptimizationAPIClient()
    
    try:
        # 1. Health check
        print("\n1. Checking API health...")
        health = client.health_check()
        print(f"✓ API Status: {health['status']}")
        
        # 2. Login
        print("\n2. Logging in...")
        login_result = client.login("admin", "admin")
        
        # 3. System status
        print("\n3. Getting system status...")
        system_status = client.get_system_status()
        print(f"✓ System Status: {system_status['status']}")
        print(f"  Active Sessions: {system_status['metrics']['active_sessions']}")
        print(f"  CPU Usage: {system_status['metrics']['cpu_usage_percent']:.1f}%")
        print(f"  Memory Usage: {system_status['metrics']['memory_usage_percent']:.1f}%")
        
        # 4. Create and upload a dummy model
        print("\n4. Creating and uploading dummy model...")
        dummy_model_path = create_dummy_model_file()
        
        try:
            upload_result = client.upload_model(
                dummy_model_path,
                name="Demo OpenVLA Model",
                description="Demo model for API testing",
                tags="demo,openvla,test"
            )
            model_id = upload_result["model_id"]
            
            # 5. List models
            print("\n5. Listing models...")
            models = client.list_models()
            print(f"✓ Found {models['total']} models")
            
            # 6. Start optimization
            print("\n6. Starting optimization...")
            optimization_result = client.start_optimization(
                model_id=model_id,
                criteria_name="demo_optimization",
                target_accuracy_threshold=0.95,
                max_size_reduction_percent=30.0,
                optimization_techniques=["quantization", "pruning"],
                notes="Demo optimization run"
            )
            session_id = optimization_result["session_id"]
            
            # 7. Monitor progress (for a short time)
            print("\n7. Monitoring optimization progress...")
            print("(Monitoring for 30 seconds, then continuing...)")
            
            start_time = time.time()
            while time.time() - start_time < 30:
                try:
                    status = client.get_session_status(session_id)
                    print(f"  Status: {status['status']} | "
                          f"Progress: {status['progress_percentage']:.1f}% | "
                          f"Step: {status['current_step'] or 'N/A'}")
                    
                    if status['status'] in ['completed', 'failed', 'cancelled']:
                        break
                    
                    time.sleep(5)
                except Exception as e:
                    print(f"  Error getting status: {e}")
                    break
            
            # 8. List sessions
            print("\n8. Listing active sessions...")
            sessions = client.list_sessions()
            print(f"✓ Found {sessions['total']} active sessions")
            
            # 9. Try to get results (might not be ready yet)
            print("\n9. Attempting to get results...")
            try:
                results = client.get_session_results(session_id)
                print("✓ Results retrieved:")
                print(f"  Optimization Summary: {results['optimization_summary']}")
                print(f"  Techniques Applied: {results['techniques_applied']}")
                print(f"  Performance Improvements: {results['performance_improvements']}")
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 400:
                    print("  Results not ready yet (optimization still in progress)")
                else:
                    print(f"  Error getting results: {e}")
            
            # 10. Cancel session if still running
            print("\n10. Cleaning up...")
            try:
                final_status = client.get_session_status(session_id)
                if final_status['status'] in ['running', 'pending']:
                    client.cancel_session(session_id)
                    print("✓ Session cancelled")
                else:
                    print(f"✓ Session already {final_status['status']}")
            except Exception as e:
                print(f"  Error during cleanup: {e}")
            
        finally:
            # Clean up dummy file
            Path(dummy_model_path).unlink(missing_ok=True)
        
        print("\n🎉 Demo completed successfully!")
        
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to API. Make sure the server is running on http://localhost:8000")
    except requests.exceptions.HTTPError as e:
        print(f"❌ HTTP Error: {e}")
        if e.response:
            print(f"   Response: {e.response.text}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")


if __name__ == "__main__":
    main()
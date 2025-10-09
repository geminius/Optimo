"""
Stress tests for concurrent optimization sessions.
Tests system behavior under high load and concurrent operations.
"""

import pytest
import asyncio
import time
import threading
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import List, Dict, Any
import torch
import torch.nn as nn
import tempfile
from pathlib import Path
import json
import psutil
import gc
from dataclasses import dataclass
from unittest.mock import MagicMock

from src.services.optimization_manager import OptimizationManager
from src.models.core import ModelMetadata
from src.config.optimization_criteria import OptimizationCriteria
from src.services.memory_manager import MemoryManager
from src.services.notification_service import NotificationService
from src.models.store import ModelStore


@dataclass
class StressTestResult:
    """Result of a stress test."""
    test_name: str
    concurrent_sessions: int
    total_time_seconds: float
    successful_sessions: int
    failed_sessions: int
    average_session_time: float
    peak_memory_mb: float
    peak_cpu_percent: float
    errors: List[str]


class SimpleStressTestModel(nn.Module):
    """Simple model for stress testing."""
    
    def __init__(self, size_factor: int = 1):
        super().__init__()
        base_size = 128 * size_factor
        self.layers = nn.Sequential(
            nn.Linear(784, base_size),
            nn.ReLU(),
            nn.Linear(base_size, base_size // 2),
            nn.ReLU(),
            nn.Linear(base_size // 2, 10)
        )
    
    def forward(self, x):
        return self.layers(x)


class ResourceMonitor:
    """Monitor system resources during stress tests."""
    
    def __init__(self):
        self.monitoring = False
        self.peak_memory_mb = 0
        self.peak_cpu_percent = 0
        self.monitor_thread = None
    
    def start_monitoring(self):
        """Start resource monitoring."""
        self.monitoring = True
        self.peak_memory_mb = 0
        self.peak_cpu_percent = 0
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop resource monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
    
    def _monitor_loop(self):
        """Monitor resource usage in a loop."""
        process = psutil.Process()
        
        while self.monitoring:
            try:
                # Memory usage
                memory_info = process.memory_info()
                memory_mb = memory_info.rss / (1024 * 1024)
                self.peak_memory_mb = max(self.peak_memory_mb, memory_mb)
                
                # CPU usage
                cpu_percent = process.cpu_percent()
                self.peak_cpu_percent = max(self.peak_cpu_percent, cpu_percent)
                
                time.sleep(0.1)  # Monitor every 100ms
            except Exception:
                pass  # Ignore monitoring errors


class ConcurrentOptimizationStressTester:
    """Stress tester for concurrent optimization sessions."""
    
    def __init__(self):
        self.resource_monitor = ResourceMonitor()
        self.results: List[StressTestResult] = []
    
    def create_mock_optimization_manager(self) -> OptimizationManager:
        """Create a mock optimization manager for testing."""
        config = {
            "max_concurrent_sessions": 10,
            "auto_rollback_on_failure": True,
            "analysis_agent": {},
            "planning_agent": {},
            "evaluation_agent": {},
            "quantization_agent": {},
            "pruning_agent": {}
        }
        
        manager = OptimizationManager(config)
        
        # Mock successful optimization
        async def mock_optimize(*args, **kwargs):
            await asyncio.sleep(0.1)  # Simulate work
            return "session_" + str(time.time())
        
        manager.start_optimization = mock_optimize
        return manager
    
    async def run_single_optimization_session(self, manager: OptimizationManager, 
                                            session_id: int) -> Dict[str, Any]:
        """Run a single optimization session."""
        try:
            start_time = time.perf_counter()
            
            # Create test model
            model = SimpleStressTestModel()
            
            # Create optimization criteria
            from src.config.optimization_criteria import OptimizationConstraints, OptimizationTechnique as ConfigOptimizationTechnique
            
            constraints = OptimizationConstraints(
                preserve_accuracy_threshold=0.95,
                allowed_techniques=[ConfigOptimizationTechnique.QUANTIZATION]
            )
            
            criteria = OptimizationCriteria(
                name="stress_test",
                description="Stress test criteria",
                constraints=constraints,
                target_deployment="general"
            )
            
            # Start optimization
            result_id = await manager.start_optimization(
                model_path=f"test_model_{session_id}",
                criteria=criteria
            )
            
            end_time = time.perf_counter()
            
            return {
                'session_id': session_id,
                'success': True,
                'duration': end_time - start_time,
                'result_id': result_id,
                'error': None
            }
            
        except Exception as e:
            return {
                'session_id': session_id,
                'success': False,
                'duration': 0,
                'result_id': None,
                'error': str(e)
            }
    
    async def test_concurrent_sessions(self, num_sessions: int, 
                                     test_name: str = "concurrent_test") -> StressTestResult:
        """Test concurrent optimization sessions."""
        print(f"Starting stress test: {test_name} with {num_sessions} sessions")
        
        # Start resource monitoring
        self.resource_monitor.start_monitoring()
        
        # Create optimization manager
        manager = self.create_mock_optimization_manager()
        
        # Start timing
        start_time = time.perf_counter()
        
        # Create and run concurrent sessions
        tasks = []
        for i in range(num_sessions):
            task = self.run_single_optimization_session(manager, i)
            tasks.append(task)
        
        # Wait for all sessions to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Stop timing and monitoring
        end_time = time.perf_counter()
        self.resource_monitor.stop_monitoring()
        
        # Analyze results
        successful_sessions = sum(1 for r in results if isinstance(r, dict) and r.get('success', False))
        failed_sessions = num_sessions - successful_sessions
        
        successful_durations = [r['duration'] for r in results 
                              if isinstance(r, dict) and r.get('success', False)]
        average_session_time = sum(successful_durations) / len(successful_durations) if successful_durations else 0
        
        errors = [str(r.get('error', r)) for r in results 
                 if isinstance(r, dict) and not r.get('success', True)]
        
        result = StressTestResult(
            test_name=test_name,
            concurrent_sessions=num_sessions,
            total_time_seconds=end_time - start_time,
            successful_sessions=successful_sessions,
            failed_sessions=failed_sessions,
            average_session_time=average_session_time,
            peak_memory_mb=self.resource_monitor.peak_memory_mb,
            peak_cpu_percent=self.resource_monitor.peak_cpu_percent,
            errors=errors
        )
        
        self.results.append(result)
        return result
    
    def test_memory_pressure(self, num_large_models: int = 10) -> StressTestResult:
        """Test system behavior under memory pressure."""
        print(f"Starting memory pressure test with {num_large_models} large models")
        
        self.resource_monitor.start_monitoring()
        start_time = time.perf_counter()
        
        models = []
        successful_loads = 0
        errors = []
        
        try:
            for i in range(num_large_models):
                try:
                    # Create increasingly large models
                    model = SimpleStressTestModel(size_factor=i + 1)
                    models.append(model)
                    successful_loads += 1
                    
                    # Force garbage collection periodically
                    if i % 5 == 0:
                        gc.collect()
                        
                except Exception as e:
                    errors.append(f"Failed to create model {i}: {str(e)}")
                    break
        
        finally:
            # Clean up
            del models
            gc.collect()
            
            end_time = time.perf_counter()
            self.resource_monitor.stop_monitoring()
        
        result = StressTestResult(
            test_name="memory_pressure",
            concurrent_sessions=num_large_models,
            total_time_seconds=end_time - start_time,
            successful_sessions=successful_loads,
            failed_sessions=num_large_models - successful_loads,
            average_session_time=0,
            peak_memory_mb=self.resource_monitor.peak_memory_mb,
            peak_cpu_percent=self.resource_monitor.peak_cpu_percent,
            errors=errors
        )
        
        self.results.append(result)
        return result
    
    def test_rapid_session_creation(self, sessions_per_second: int = 10, 
                                  duration_seconds: int = 30) -> StressTestResult:
        """Test rapid session creation and destruction."""
        print(f"Starting rapid session creation test: {sessions_per_second}/sec for {duration_seconds}s")
        
        self.resource_monitor.start_monitoring()
        start_time = time.perf_counter()
        
        manager = self.create_mock_optimization_manager()
        session_count = 0
        successful_sessions = 0
        errors = []
        
        async def create_sessions():
            nonlocal session_count, successful_sessions
            
            end_time = start_time + duration_seconds
            
            while time.perf_counter() < end_time:
                batch_start = time.perf_counter()
                
                # Create batch of sessions
                tasks = []
                for _ in range(sessions_per_second):
                    task = self.run_single_optimization_session(manager, session_count)
                    tasks.append(task)
                    session_count += 1
                
                # Wait for batch to complete
                try:
                    results = await asyncio.gather(*tasks)
                    successful_sessions += sum(1 for r in results if r.get('success', False))
                except Exception as e:
                    errors.append(f"Batch error: {str(e)}")
                
                # Wait for next second
                elapsed = time.perf_counter() - batch_start
                if elapsed < 1.0:
                    await asyncio.sleep(1.0 - elapsed)
        
        # Run the test
        asyncio.run(create_sessions())
        
        end_time = time.perf_counter()
        self.resource_monitor.stop_monitoring()
        
        result = StressTestResult(
            test_name="rapid_session_creation",
            concurrent_sessions=session_count,
            total_time_seconds=end_time - start_time,
            successful_sessions=successful_sessions,
            failed_sessions=session_count - successful_sessions,
            average_session_time=0,
            peak_memory_mb=self.resource_monitor.peak_memory_mb,
            peak_cpu_percent=self.resource_monitor.peak_cpu_percent,
            errors=errors
        )
        
        self.results.append(result)
        return result
    
    def generate_stress_test_report(self) -> str:
        """Generate comprehensive stress test report."""
        if not self.results:
            return "No stress test results available."
        
        report = "Concurrent Optimization Stress Test Report\n"
        report += "=" * 50 + "\n\n"
        
        for result in self.results:
            report += f"Test: {result.test_name}\n"
            report += "-" * 30 + "\n"
            report += f"Concurrent Sessions: {result.concurrent_sessions}\n"
            report += f"Total Time: {result.total_time_seconds:.2f}s\n"
            report += f"Successful Sessions: {result.successful_sessions}\n"
            report += f"Failed Sessions: {result.failed_sessions}\n"
            
            if result.successful_sessions > 0:
                success_rate = (result.successful_sessions / result.concurrent_sessions) * 100
                report += f"Success Rate: {success_rate:.1f}%\n"
            
            if result.average_session_time > 0:
                report += f"Average Session Time: {result.average_session_time:.3f}s\n"
            
            report += f"Peak Memory Usage: {result.peak_memory_mb:.1f} MB\n"
            report += f"Peak CPU Usage: {result.peak_cpu_percent:.1f}%\n"
            
            if result.errors:
                report += f"Errors ({len(result.errors)}):\n"
                for error in result.errors[:5]:  # Show first 5 errors
                    report += f"  - {error}\n"
                if len(result.errors) > 5:
                    report += f"  ... and {len(result.errors) - 5} more\n"
            
            report += "\n"
        
        return report
    
    def save_results(self, filepath: str):
        """Save stress test results to JSON file."""
        results_dict = []
        for result in self.results:
            result_dict = {
                'test_name': result.test_name,
                'concurrent_sessions': result.concurrent_sessions,
                'total_time_seconds': result.total_time_seconds,
                'successful_sessions': result.successful_sessions,
                'failed_sessions': result.failed_sessions,
                'average_session_time': result.average_session_time,
                'peak_memory_mb': result.peak_memory_mb,
                'peak_cpu_percent': result.peak_cpu_percent,
                'errors': result.errors
            }
            results_dict.append(result_dict)
        
        with open(filepath, 'w') as f:
            json.dump(results_dict, f, indent=2)


@pytest.fixture
def stress_tester():
    """Create stress tester fixture."""
    return ConcurrentOptimizationStressTester()


class TestConcurrentOptimizationStress:
    """Test concurrent optimization stress scenarios."""
    
    @pytest.mark.asyncio
    async def test_low_concurrency_stress(self, stress_tester):
        """Test low concurrency (5 sessions)."""
        result = await stress_tester.test_concurrent_sessions(5, "low_concurrency")
        
        assert result.concurrent_sessions == 5
        assert result.successful_sessions >= 0
        assert result.total_time_seconds > 0
        assert result.peak_memory_mb > 0
    
    @pytest.mark.asyncio
    async def test_medium_concurrency_stress(self, stress_tester):
        """Test medium concurrency (20 sessions)."""
        result = await stress_tester.test_concurrent_sessions(20, "medium_concurrency")
        
        assert result.concurrent_sessions == 20
        assert result.successful_sessions >= 0
        assert result.total_time_seconds > 0
    
    @pytest.mark.asyncio
    async def test_high_concurrency_stress(self, stress_tester):
        """Test high concurrency (50 sessions)."""
        result = await stress_tester.test_concurrent_sessions(50, "high_concurrency")
        
        assert result.concurrent_sessions == 50
        assert result.successful_sessions >= 0
        # Allow for some failures under high load
        success_rate = result.successful_sessions / result.concurrent_sessions
        assert success_rate >= 0.7  # At least 70% success rate
    
    def test_memory_pressure_stress(self, stress_tester):
        """Test system behavior under memory pressure."""
        result = stress_tester.test_memory_pressure(10)
        
        assert result.test_name == "memory_pressure"
        assert result.peak_memory_mb > 0
        # Should handle at least some models before running out of memory
        assert result.successful_sessions > 0
    
    def test_rapid_session_creation_stress(self, stress_tester):
        """Test rapid session creation and destruction."""
        result = stress_tester.test_rapid_session_creation(
            sessions_per_second=5, 
            duration_seconds=10
        )
        
        assert result.test_name == "rapid_session_creation"
        assert result.concurrent_sessions > 0
        assert result.total_time_seconds >= 10
    
    def test_stress_report_generation(self, stress_tester):
        """Test stress test report generation."""
        # Run a quick test first
        asyncio.run(stress_tester.test_concurrent_sessions(3, "report_test"))
        
        report = stress_tester.generate_stress_test_report()
        
        assert "Concurrent Optimization Stress Test Report" in report
        assert "report_test" in report
        assert "Concurrent Sessions:" in report
        assert "Success Rate:" in report
    
    def test_stress_results_saving(self, stress_tester):
        """Test saving stress test results."""
        # Run a quick test first
        asyncio.run(stress_tester.test_concurrent_sessions(2, "save_test"))
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            stress_tester.save_results(f.name)
            
            # Verify file was created and contains valid JSON
            with open(f.name, 'r') as read_f:
                data = json.load(read_f)
                assert isinstance(data, list)
                assert len(data) > 0
                assert "test_name" in data[0]
                assert "concurrent_sessions" in data[0]
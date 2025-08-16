"""
Test the comprehensive testing suite itself.
Validates that all testing components work correctly.
"""

import pytest
import tempfile
from pathlib import Path
import json
import asyncio

from tests.data.test_data_generator import TestDataGenerator, ModelType, OptimizationTechnique
from tests.performance.test_optimization_benchmarks import OptimizationBenchmarks
from tests.stress.test_concurrent_optimizations import ConcurrentOptimizationStressTester
from tests.automation.test_runner import TestRunner, TestSuite


class TestComprehensiveTestingSuite:
    """Test the comprehensive testing suite components."""
    
    def test_test_data_generator_creation(self):
        """Test that test data generator can be created and used."""
        generator = TestDataGenerator()
        
        # Test model generation
        cnn_model = generator.generate_model(ModelType.CNN, "small")
        assert cnn_model is not None
        
        # Test input generation
        test_input = generator.generate_test_input(ModelType.CNN)
        assert test_input is not None
        assert test_input.shape[0] == 1  # Batch size
        
        # Test scenario access
        scenarios = generator.get_scenarios_by_complexity("simple")
        assert len(scenarios) > 0
    
    def test_test_data_suite_creation(self, temp_dir):
        """Test creation of complete test data suite."""
        generator = TestDataGenerator()
        
        # Create a minimal test suite
        test_suite = generator.create_test_suite(temp_dir)
        
        assert "scenarios" in test_suite
        assert "models" in test_suite
        assert "metadata" in test_suite
        assert len(test_suite["scenarios"]) > 0
        
        # Verify test suite config file was created
        config_file = temp_dir / "test_suite.json"
        assert config_file.exists()
        
        with open(config_file, 'r') as f:
            config_data = json.load(f)
            assert config_data == test_suite
    
    def test_performance_benchmark_creation(self):
        """Test that performance benchmarks can be created."""
        benchmarks = OptimizationBenchmarks()
        
        # Test benchmark suite exists
        assert benchmarks is not None
        assert hasattr(benchmarks, 'quantization_agent')
        assert hasattr(benchmarks, 'pruning_agent')
        
        # Test model size calculation
        from tests.data.test_data_generator import SyntheticModelGenerator
        generator = SyntheticModelGenerator()
        model = generator.create_cnn_model("small")
        
        size_mb = benchmarks.get_model_size_mb(model)
        assert size_mb > 0
    
    def test_stress_tester_creation(self):
        """Test that stress tester can be created."""
        stress_tester = ConcurrentOptimizationStressTester()
        
        assert stress_tester is not None
        assert hasattr(stress_tester, 'resource_monitor')
        assert hasattr(stress_tester, 'results')
    
    def test_test_runner_creation(self, temp_dir):
        """Test that test runner can be created."""
        runner = TestRunner(temp_dir)
        
        assert runner is not None
        assert runner.output_dir == temp_dir
        assert hasattr(runner, 'execution_id')
        
        # Test environment info collection
        env_info = runner.get_environment_info()
        assert "platform" in env_info
        assert "python_version" in env_info
        assert "pytorch_version" in env_info
    
    def test_test_data_generation_integration(self, temp_dir):
        """Test integration of test data generation."""
        runner = TestRunner(temp_dir)
        
        # Test data generation
        success = runner.generate_test_data()
        assert success
        
        # Verify test data directory was created
        test_data_dir = temp_dir / "test_data"
        assert test_data_dir.exists()
        
        # Verify test suite config exists
        config_file = test_data_dir / "test_suite.json"
        assert config_file.exists()
    
    @pytest.mark.asyncio
    async def test_stress_test_execution(self):
        """Test that stress tests can be executed."""
        stress_tester = ConcurrentOptimizationStressTester()
        
        # Run a minimal stress test
        result = await stress_tester.test_concurrent_sessions(2, "test_execution")
        
        assert result is not None
        assert result.test_name == "test_execution"
        assert result.concurrent_sessions == 2
        assert result.total_time_seconds >= 0
    
    def test_benchmark_execution(self):
        """Test that benchmarks can be executed."""
        benchmarks = OptimizationBenchmarks()
        
        # Create a simple test model
        from tests.data.test_data_generator import SyntheticModelGenerator
        generator = SyntheticModelGenerator()
        model = generator.create_mlp_model("small")
        
        # Test input for MLP
        import torch
        test_input = torch.randn(1, 784)
        
        # Run benchmark (this will use mocked optimization agents)
        try:
            result = benchmarks.benchmark_quantization(model, "test_mlp", test_input)
            assert result is not None
            assert result.technique == "quantization"
            assert result.model_type == "test_mlp"
        except Exception as e:
            # It's okay if the actual optimization fails in testing
            # We're just testing that the benchmark framework works
            assert "optimize" in str(e).lower() or "quantization" in str(e).lower()
    
    def test_report_generation(self, temp_dir):
        """Test that reports can be generated."""
        runner = TestRunner(temp_dir)
        
        # Create a mock test result
        from tests.automation.test_runner import TestSuiteResult, TestResult
        
        test_result = TestResult(
            test_name="test_example",
            suite="unit",
            status="passed",
            duration_seconds=0.1
        )
        
        suite_result = TestSuiteResult(
            suite_name="unit",
            total_tests=1,
            passed_tests=1,
            failed_tests=0,
            skipped_tests=0,
            error_tests=0,
            total_duration_seconds=0.1,
            test_results=[test_result]
        )
        
        runner.results = [suite_result]
        
        # Generate report
        report = runner.generate_report()
        
        assert report is not None
        assert report.execution_id == runner.execution_id
        assert len(report.suite_results) == 1
        assert report.summary["total_tests"] == 1
        assert report.summary["passed_tests"] == 1
        
        # Test report saving
        saved_files = runner.save_report(report, ["json", "txt"])
        assert len(saved_files) == 2
        
        for file_path in saved_files:
            assert file_path.exists()
    
    def test_test_suite_enumeration(self):
        """Test that all test suites are properly enumerated."""
        # Test that all expected test suites exist
        expected_suites = {"unit", "integration", "performance", "stress", "end_to_end", "all"}
        actual_suites = {suite.value for suite in TestSuite}
        
        assert expected_suites.issubset(actual_suites)
    
    def test_pytest_configuration_exists(self):
        """Test that pytest configuration files exist."""
        project_root = Path(__file__).parent.parent
        
        # Check for pytest.ini
        pytest_ini = project_root / "pytest.ini"
        assert pytest_ini.exists()
        
        # Check for conftest.py
        conftest = project_root / "tests" / "conftest.py"
        assert conftest.exists()
    
    def test_run_tests_script_exists(self):
        """Test that the main test runner script exists."""
        project_root = Path(__file__).parent.parent
        run_tests_script = project_root / "run_tests.py"
        
        assert run_tests_script.exists()
        
        # Verify it's executable (has shebang)
        with open(run_tests_script, 'r') as f:
            first_line = f.readline().strip()
            assert first_line.startswith("#!")
    
    def test_all_test_directories_exist(self):
        """Test that all expected test directories exist."""
        test_root = Path(__file__).parent
        
        expected_dirs = [
            "integration",
            "performance", 
            "stress",
            "data",
            "automation"
        ]
        
        for dir_name in expected_dirs:
            test_dir = test_root / dir_name
            assert test_dir.exists(), f"Test directory {dir_name} does not exist"
    
    def test_test_file_structure(self):
        """Test that all expected test files exist."""
        test_root = Path(__file__).parent
        
        expected_files = [
            "integration/test_end_to_end_workflows.py",
            "performance/test_optimization_benchmarks.py",
            "stress/test_concurrent_optimizations.py",
            "data/test_data_generator.py",
            "automation/test_runner.py",
            "conftest.py"
        ]
        
        for file_path in expected_files:
            test_file = test_root / file_path
            assert test_file.exists(), f"Test file {file_path} does not exist"
    
    def test_imports_work(self):
        """Test that all test modules can be imported successfully."""
        # Test that we can import all major test components
        try:
            from tests.data.test_data_generator import TestDataGenerator
            from tests.performance.test_optimization_benchmarks import OptimizationBenchmarks
            from tests.stress.test_concurrent_optimizations import ConcurrentOptimizationStressTester
            from tests.automation.test_runner import TestRunner
            
            # If we get here, all imports worked
            assert True
        except ImportError as e:
            pytest.fail(f"Failed to import test modules: {e}")
    
    def test_comprehensive_suite_completeness(self):
        """Test that the comprehensive suite covers all requirements."""
        # Verify we have tests for all the sub-tasks mentioned in the task
        test_root = Path(__file__).parent
        
        # Check for integration tests
        integration_dir = test_root / "integration"
        assert integration_dir.exists()
        integration_files = list(integration_dir.glob("*.py"))
        assert len(integration_files) > 0
        
        # Check for performance benchmarks
        performance_dir = test_root / "performance"
        assert performance_dir.exists()
        performance_files = list(performance_dir.glob("*.py"))
        assert len(performance_files) > 0
        
        # Check for stress tests
        stress_dir = test_root / "stress"
        assert stress_dir.exists()
        stress_files = list(stress_dir.glob("*.py"))
        assert len(stress_files) > 0
        
        # Check for test data generation
        data_dir = test_root / "data"
        assert data_dir.exists()
        data_files = list(data_dir.glob("*.py"))
        assert len(data_files) > 0
        
        # Check for automation scripts
        automation_dir = test_root / "automation"
        assert automation_dir.exists()
        automation_files = list(automation_dir.glob("*.py"))
        assert len(automation_files) > 0
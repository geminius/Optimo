"""
Pytest configuration and shared fixtures for the robotics model optimization platform.
"""

import pytest
import torch
import tempfile
import shutil
from pathlib import Path
from unittest.mock import MagicMock
import asyncio
from typing import Generator, Any

# Import project modules
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.models.store import ModelStore
from src.services.memory_manager import MemoryManager
from src.services.notification_service import NotificationService
from src.services.optimization_manager import OptimizationManager
from src.agents.analysis.agent import AnalysisAgent
from src.agents.planning.agent import PlanningAgent
from src.agents.optimization.quantization import QuantizationAgent
from src.agents.optimization.pruning import PruningAgent
from src.agents.evaluation.agent import EvaluationAgent
from tests.data.test_data_generator import TestDataGenerator, SyntheticModelGenerator


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom settings."""
    # Set torch to use CPU only for consistent testing
    torch.set_default_tensor_type('torch.FloatTensor')
    
    # Disable CUDA for testing to ensure reproducibility
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = ''


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers and organize tests."""
    for item in items:
        # Add markers based on test path
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "performance" in str(item.fspath):
            item.add_marker(pytest.mark.performance)
        elif "stress" in str(item.fspath):
            item.add_marker(pytest.mark.stress)
        else:
            item.add_marker(pytest.mark.unit)
        
        # Mark slow tests
        if "stress" in str(item.fspath) or "performance" in str(item.fspath):
            item.add_marker(pytest.mark.slow)
        
        # Mark GPU tests
        if "gpu" in item.name.lower() or "cuda" in item.name.lower():
            item.add_marker(pytest.mark.gpu)


# Fixtures for temporary directories and files
@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    temp_path = Path(tempfile.mkdtemp())
    try:
        yield temp_path
    finally:
        shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def temp_model_dir(temp_dir) -> Path:
    """Create a temporary directory specifically for model files."""
    model_dir = temp_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir


@pytest.fixture
def temp_config_dir(temp_dir) -> Path:
    """Create a temporary directory for configuration files."""
    config_dir = temp_dir / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir


# Model fixtures
@pytest.fixture
def synthetic_model_generator() -> SyntheticModelGenerator:
    """Create synthetic model generator."""
    return SyntheticModelGenerator()


@pytest.fixture
def test_data_generator() -> TestDataGenerator:
    """Create test data generator."""
    return TestDataGenerator()


@pytest.fixture
def small_cnn_model(synthetic_model_generator) -> torch.nn.Module:
    """Create a small CNN model for testing."""
    return synthetic_model_generator.create_cnn_model("small")


@pytest.fixture
def medium_transformer_model(synthetic_model_generator) -> torch.nn.Module:
    """Create a medium transformer model for testing."""
    return synthetic_model_generator.create_transformer_model("medium")


@pytest.fixture
def test_model_with_path(small_cnn_model, temp_model_dir) -> tuple:
    """Create a test model and save it to a temporary path."""
    model_path = temp_model_dir / "test_model.pth"
    torch.save(small_cnn_model.state_dict(), model_path)
    return small_cnn_model, model_path


# Mock fixtures for services and agents
@pytest.fixture
def mock_model_store() -> MagicMock:
    """Create a mock model store."""
    mock = MagicMock(spec=ModelStore)
    mock.load_model.return_value = MagicMock()
    mock.save_model.return_value = "test_model_id"
    mock.get_model_metadata.return_value = MagicMock()
    return mock


@pytest.fixture
def mock_memory_manager() -> MagicMock:
    """Create a mock memory manager."""
    mock = MagicMock(spec=MemoryManager)
    mock.create_session.return_value = "test_session_id"
    mock.get_session.return_value = MagicMock()
    mock.update_session.return_value = None
    return mock


@pytest.fixture
def mock_notification_service() -> MagicMock:
    """Create a mock notification service."""
    mock = MagicMock(spec=NotificationService)
    mock.send_notification.return_value = None
    mock.subscribe.return_value = None
    return mock


@pytest.fixture
def mock_analysis_agent() -> MagicMock:
    """Create a mock analysis agent."""
    mock = MagicMock(spec=AnalysisAgent)
    mock.analyze_model.return_value = MagicMock()
    mock.identify_bottlenecks.return_value = []
    mock.estimate_optimization_impact.return_value = MagicMock()
    return mock


@pytest.fixture
def mock_planning_agent() -> MagicMock:
    """Create a mock planning agent."""
    mock = MagicMock(spec=PlanningAgent)
    mock.plan_optimization.return_value = MagicMock()
    mock.prioritize_techniques.return_value = []
    mock.validate_plan.return_value = MagicMock()
    return mock


@pytest.fixture
def mock_quantization_agent() -> MagicMock:
    """Create a mock quantization agent."""
    mock = MagicMock(spec=QuantizationAgent)
    mock.can_optimize.return_value = True
    mock.estimate_impact.return_value = MagicMock()
    mock.optimize.return_value = MagicMock()
    mock.validate_result.return_value = MagicMock()
    return mock


@pytest.fixture
def mock_pruning_agent() -> MagicMock:
    """Create a mock pruning agent."""
    mock = MagicMock(spec=PruningAgent)
    mock.can_optimize.return_value = True
    mock.estimate_impact.return_value = MagicMock()
    mock.optimize.return_value = MagicMock()
    mock.validate_result.return_value = MagicMock()
    return mock


@pytest.fixture
def mock_evaluation_agent() -> MagicMock:
    """Create a mock evaluation agent."""
    mock = MagicMock(spec=EvaluationAgent)
    mock.evaluate_model.return_value = MagicMock()
    mock.compare_models.return_value = MagicMock()
    mock.validate_performance.return_value = MagicMock()
    return mock


# Integration fixtures
@pytest.fixture
def optimization_manager_with_mocks(
    mock_model_store,
    mock_memory_manager,
    mock_notification_service,
    mock_analysis_agent,
    mock_planning_agent,
    mock_quantization_agent,
    mock_pruning_agent,
    mock_evaluation_agent
) -> OptimizationManager:
    """Create optimization manager with all mocked dependencies."""
    manager = OptimizationManager(
        model_store=mock_model_store,
        memory_manager=mock_memory_manager,
        notification_service=mock_notification_service
    )
    
    # Set up agents
    manager.analysis_agent = mock_analysis_agent
    manager.planning_agent = mock_planning_agent
    manager.optimization_agents = {
        'quantization': mock_quantization_agent,
        'pruning': mock_pruning_agent
    }
    manager.evaluation_agent = mock_evaluation_agent
    
    return manager


# Async fixtures
@pytest.fixture
def event_loop():
    """Create an event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# Test data fixtures
@pytest.fixture(scope="session")
def test_data_suite(tmp_path_factory) -> dict:
    """Create a comprehensive test data suite (session-scoped for performance)."""
    temp_dir = tmp_path_factory.mktemp("test_data")
    generator = TestDataGenerator()
    
    # Generate a subset of test data for faster testing
    test_suite = generator.create_test_suite(temp_dir)
    return test_suite


# Performance testing fixtures
@pytest.fixture
def performance_test_config() -> dict:
    """Configuration for performance tests."""
    return {
        "benchmark_iterations": 10,
        "warmup_iterations": 3,
        "timeout_seconds": 300,
        "memory_limit_mb": 1024,
        "acceptable_slowdown": 1.5
    }


# Stress testing fixtures
@pytest.fixture
def stress_test_config() -> dict:
    """Configuration for stress tests."""
    return {
        "max_concurrent_sessions": 20,
        "test_duration_seconds": 30,
        "memory_pressure_models": 10,
        "rapid_session_rate": 5,
        "acceptable_failure_rate": 0.1
    }


# Cleanup fixtures
@pytest.fixture(autouse=True)
def cleanup_torch():
    """Automatically cleanup torch resources after each test."""
    yield
    # Clear CUDA cache if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Force garbage collection
    import gc
    gc.collect()


# Parametrized fixtures for different model types and sizes
@pytest.fixture(params=["small", "medium"])
def model_size(request):
    """Parametrized fixture for different model sizes."""
    return request.param


@pytest.fixture(params=["cnn", "transformer", "mlp"])
def model_type(request):
    """Parametrized fixture for different model types."""
    return request.param


@pytest.fixture
def parametrized_model(synthetic_model_generator, model_type, model_size):
    """Create a model based on parametrized type and size."""
    generators = {
        "cnn": synthetic_model_generator.create_cnn_model,
        "transformer": synthetic_model_generator.create_transformer_model,
        "mlp": synthetic_model_generator.create_mlp_model
    }
    
    generator = generators.get(model_type)
    if generator:
        return generator(model_size)
    else:
        pytest.skip(f"Model type {model_type} not supported")


# Utility fixtures
@pytest.fixture
def assert_model_equality():
    """Utility function to assert model equality."""
    def _assert_equal(model1, model2, tolerance=1e-6):
        """Assert that two models are approximately equal."""
        params1 = list(model1.parameters())
        params2 = list(model2.parameters())
        
        assert len(params1) == len(params2), "Models have different number of parameters"
        
        for p1, p2 in zip(params1, params2):
            assert torch.allclose(p1, p2, atol=tolerance), "Model parameters differ"
    
    return _assert_equal


@pytest.fixture
def measure_execution_time():
    """Utility to measure execution time."""
    import time
    
    def _measure(func, *args, **kwargs):
        """Measure execution time of a function."""
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        return result, end_time - start_time
    
    return _measure


# Skip conditions
def pytest_runtest_setup(item):
    """Setup function to handle test skipping based on conditions."""
    # Skip GPU tests if CUDA is not available
    if item.get_closest_marker("gpu") and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    # Skip slow tests in quick mode
    if item.get_closest_marker("slow") and item.config.getoption("--quick", default=False):
        pytest.skip("Skipping slow test in quick mode")


# Custom pytest options
def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--quick",
        action="store_true",
        default=False,
        help="Run only quick tests, skip slow ones"
    )
    
    parser.addoption(
        "--gpu",
        action="store_true",
        default=False,
        help="Run GPU tests (requires CUDA)"
    )
    
    parser.addoption(
        "--stress",
        action="store_true",
        default=False,
        help="Run stress tests"
    )
"""
Unit tests for data models and validation.
"""

import pytest
import tempfile
import os
from datetime import datetime
from pathlib import Path
from unittest.mock import patch, mock_open

from src.models import (
    ModelType, ModelFramework, OptimizationStatus, ValidationStatus,
    ModelMetadata, ArchitectureSummary, PerformanceProfile,
    OptimizationOpportunity, Recommendation, AnalysisReport,
    OptimizationStep, OptimizationResults, OptimizationSession,
    BenchmarkResult, PerformanceMetrics, ComparisonResult, EvaluationReport,
    ValidationError, ModelFormatValidator, OptimizationParameterValidator,
    validate_model_metadata, validate_optimization_session,
    validate_analysis_report, validate_evaluation_report,
    calculate_file_checksum, validate_file_integrity
)


class TestModelMetadata:
    """Test ModelMetadata data model."""
    
    def test_valid_model_metadata(self):
        """Test creating valid ModelMetadata."""
        metadata = ModelMetadata(
            name="test_model",
            version="1.0.0",
            model_type=ModelType.OPENVLA,
            framework=ModelFramework.PYTORCH,
            size_mb=100.5,
            parameters=1000000,
            tags=["robotics", "vision"]
        )
        
        assert metadata.name == "test_model"
        assert metadata.version == "1.0.0"
        assert metadata.model_type == ModelType.OPENVLA
        assert metadata.framework == ModelFramework.PYTORCH
        assert metadata.size_mb == 100.5
        assert metadata.parameters == 1000000
        assert "robotics" in metadata.tags
        assert isinstance(metadata.created_at, datetime)
        assert len(metadata.id) > 0
    
    def test_empty_name_validation(self):
        """Test validation fails for empty name."""
        with pytest.raises(ValueError, match="Model name cannot be empty"):
            ModelMetadata(name="", version="1.0.0")
    
    def test_negative_size_validation(self):
        """Test validation fails for negative size."""
        with pytest.raises(ValueError, match="Model size cannot be negative"):
            ModelMetadata(name="test", version="1.0.0", size_mb=-10.0)
    
    def test_negative_parameters_validation(self):
        """Test validation fails for negative parameters."""
        with pytest.raises(ValueError, match="Parameter count cannot be negative"):
            ModelMetadata(name="test", version="1.0.0", parameters=-100)
    
    def test_empty_version_validation(self):
        """Test validation fails for empty version."""
        with pytest.raises(ValueError, match="Model version cannot be empty"):
            ModelMetadata(name="test", version="")


class TestOptimizationSession:
    """Test OptimizationSession data model."""
    
    def test_valid_optimization_session(self):
        """Test creating valid OptimizationSession."""
        session = OptimizationSession(
            model_id="model_123",
            criteria_name="default",
            priority=2,
            tags=["test", "optimization"]
        )
        
        assert session.model_id == "model_123"
        assert session.criteria_name == "default"
        assert session.priority == 2
        assert session.status == OptimizationStatus.PENDING
        assert isinstance(session.created_at, datetime)
        assert len(session.id) > 0
    
    def test_empty_model_id_validation(self):
        """Test validation fails for empty model_id."""
        with pytest.raises(ValueError, match="Model ID cannot be empty"):
            OptimizationSession(model_id="")
    
    def test_invalid_priority_validation(self):
        """Test validation fails for invalid priority."""
        with pytest.raises(ValueError, match="Priority must be between 1 and 5"):
            OptimizationSession(model_id="test", priority=0)
        
        with pytest.raises(ValueError, match="Priority must be between 1 and 5"):
            OptimizationSession(model_id="test", priority=6)
    
    def test_duration_calculation(self):
        """Test duration calculation."""
        session = OptimizationSession(model_id="test")
        
        # No duration when not started/completed
        assert session.duration_seconds is None
        
        # Set start and end times
        start_time = datetime.now()
        end_time = datetime.now()
        session.started_at = start_time
        session.completed_at = end_time
        
        duration = session.duration_seconds
        assert duration is not None
        assert duration >= 0
    
    def test_is_active_property(self):
        """Test is_active property."""
        session = OptimizationSession(model_id="test")
        
        # Pending is active
        session.status = OptimizationStatus.PENDING
        assert session.is_active
        
        # Running is active
        session.status = OptimizationStatus.RUNNING
        assert session.is_active
        
        # Completed is not active
        session.status = OptimizationStatus.COMPLETED
        assert not session.is_active
        
        # Failed is not active
        session.status = OptimizationStatus.FAILED
        assert not session.is_active


class TestAnalysisReport:
    """Test AnalysisReport data model."""
    
    def test_valid_analysis_report(self):
        """Test creating valid AnalysisReport."""
        report = AnalysisReport(
            model_id="model_123",
            analysis_duration_seconds=45.5
        )
        
        assert report.model_id == "model_123"
        assert report.analysis_duration_seconds == 45.5
        assert isinstance(report.analysis_timestamp, datetime)
        assert len(report.analysis_id) > 0
    
    def test_empty_model_id_validation(self):
        """Test validation fails for empty model_id."""
        with pytest.raises(ValueError, match="Model ID cannot be empty"):
            AnalysisReport(model_id="")
    
    def test_negative_duration_validation(self):
        """Test validation fails for negative duration."""
        with pytest.raises(ValueError, match="Analysis duration cannot be negative"):
            AnalysisReport(model_id="test", analysis_duration_seconds=-10.0)


class TestEvaluationReport:
    """Test EvaluationReport data model."""
    
    def test_valid_evaluation_report(self):
        """Test creating valid EvaluationReport."""
        report = EvaluationReport(
            model_id="model_123",
            evaluation_duration_seconds=30.0
        )
        
        assert report.model_id == "model_123"
        assert report.evaluation_duration_seconds == 30.0
        assert isinstance(report.evaluation_timestamp, datetime)
        assert len(report.evaluation_id) > 0
    
    def test_empty_model_id_validation(self):
        """Test validation fails for empty model_id."""
        with pytest.raises(ValueError, match="Model ID cannot be empty"):
            EvaluationReport(model_id="")
    
    def test_negative_duration_validation(self):
        """Test validation fails for negative duration."""
        with pytest.raises(ValueError, match="Evaluation duration cannot be negative"):
            EvaluationReport(model_id="test", evaluation_duration_seconds=-5.0)
    
    def test_overall_success_property(self):
        """Test overall_success property."""
        report = EvaluationReport(model_id="test")
        
        # Default is not successful (pending status)
        assert not report.overall_success
        
        # Passed with no errors is successful
        report.validation_status = ValidationStatus.PASSED
        assert report.overall_success
        
        # Passed with errors is not successful
        report.validation_errors = ["Some error"]
        assert not report.overall_success


class TestModelFormatValidator:
    """Test ModelFormatValidator."""
    
    def test_nonexistent_file(self):
        """Test validation of nonexistent file."""
        is_valid, issues = ModelFormatValidator.validate_model_file("nonexistent.pt")
        
        assert not is_valid
        assert any("does not exist" in issue for issue in issues)
    
    def test_empty_file(self):
        """Test validation of empty file."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(b"")
            tmp_file.flush()
            
            try:
                is_valid, issues = ModelFormatValidator.validate_model_file(tmp_file.name)
                
                assert not is_valid
                assert any("empty" in issue for issue in issues)
            finally:
                os.unlink(tmp_file.name)
    
    def test_large_file_warning(self):
        """Test warning for very large files."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            # Write some content to avoid empty file error
            tmp_file.write(b"test content")
            tmp_file.flush()
            
            try:
                # Mock file size to be very large
                with patch('os.path.getsize', return_value=60 * 1024 * 1024 * 1024):  # 60GB
                    is_valid, issues = ModelFormatValidator.validate_model_file(tmp_file.name)
                    
                    assert any("too large" in issue for issue in issues)
            finally:
                os.unlink(tmp_file.name)
    
    def test_extension_validation(self):
        """Test file extension validation."""
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp_file:
            tmp_file.write(b"test content")
            tmp_file.flush()
            
            try:
                is_valid, issues = ModelFormatValidator.validate_model_file(
                    tmp_file.name, ModelFramework.PYTORCH
                )
                
                assert any("extension" in issue and "not supported" in issue for issue in issues)
            finally:
                os.unlink(tmp_file.name)
    
    def test_pytorch_validation_success(self):
        """Test successful PyTorch model validation."""
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp_file:
            tmp_file.write(b"PK\x03\x04test content")  # Mock PyTorch magic bytes
            tmp_file.flush()
            
            try:
                with patch('torch.load') as mock_torch_load:
                    mock_torch_load.return_value = {'layer1.weight': [1, 2, 3], 'layer1.bias': [0.1]}
                    issues = ModelFormatValidator._validate_pytorch_model(tmp_file.name)
                    assert len(issues) == 0
            except ImportError:
                # If torch is not available, test should handle gracefully
                issues = ModelFormatValidator._validate_pytorch_model(tmp_file.name)
                assert any("PyTorch not available" in issue for issue in issues)
            finally:
                os.unlink(tmp_file.name)
    
    def test_pytorch_validation_empty_state_dict(self):
        """Test PyTorch validation with empty state dict."""
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp_file:
            tmp_file.write(b"test content")
            tmp_file.flush()
            
            try:
                with patch('torch.load') as mock_torch_load:
                    mock_torch_load.return_value = {}
                    issues = ModelFormatValidator._validate_pytorch_model(tmp_file.name)
                    assert any("empty" in issue for issue in issues)
            except ImportError:
                # If torch is not available, test should handle gracefully
                issues = ModelFormatValidator._validate_pytorch_model(tmp_file.name)
                assert any("PyTorch not available" in issue for issue in issues)
            finally:
                os.unlink(tmp_file.name)


class TestOptimizationParameterValidator:
    """Test OptimizationParameterValidator."""
    
    def test_valid_quantization_parameters(self):
        """Test valid quantization parameters."""
        params = {
            'quantization_bits': 8,
            'quantization_type': 'dynamic'
        }
        
        is_valid, issues = OptimizationParameterValidator.validate_optimization_parameters(
            'quantization', params
        )
        
        assert is_valid
        assert len(issues) == 0
    
    def test_missing_required_parameter(self):
        """Test missing required parameter."""
        params = {}  # Missing quantization_bits
        
        is_valid, issues = OptimizationParameterValidator.validate_optimization_parameters(
            'quantization', params
        )
        
        assert not is_valid
        assert any("Required parameter" in issue and "quantization_bits" in issue for issue in issues)
    
    def test_invalid_parameter_range(self):
        """Test parameter outside valid range."""
        params = {
            'quantization_bits': 64,  # Outside valid range
        }
        
        is_valid, issues = OptimizationParameterValidator.validate_optimization_parameters(
            'quantization', params
        )
        
        assert not is_valid
        assert any("outside valid range" in issue for issue in issues)
    
    def test_invalid_ratio_parameter(self):
        """Test invalid ratio parameter."""
        params = {
            'pruning_ratio': 1.5,  # Invalid ratio > 1
        }
        
        is_valid, issues = OptimizationParameterValidator.validate_optimization_parameters(
            'pruning', params
        )
        
        assert not is_valid
        assert any("must be between 0 and 1" in issue for issue in issues)
    
    def test_distillation_parameter_validation(self):
        """Test distillation parameter validation."""
        params = {
            'temperature': 0.5,  # Too low
            'alpha': 0.7,
            'beta': 0.4  # alpha + beta > 1
        }
        
        is_valid, issues = OptimizationParameterValidator.validate_optimization_parameters(
            'distillation', params
        )
        
        assert not is_valid
        assert any("temperature should be >= 1.0" in issue for issue in issues)
        assert any("should sum to 1.0" in issue for issue in issues)


class TestValidationFunctions:
    """Test high-level validation functions."""
    
    def test_validate_model_metadata_success(self):
        """Test successful model metadata validation."""
        metadata = ModelMetadata(
            name="test_model",
            version="1.0.0",
            model_type=ModelType.OPENVLA,
            framework=ModelFramework.PYTORCH
        )
        
        is_valid, issues = validate_model_metadata(metadata)
        
        assert is_valid
        assert len(issues) == 0
    
    def test_validate_model_metadata_with_file(self):
        """Test model metadata validation with file."""
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp_file:
            tmp_file.write(b"test content")
            tmp_file.flush()
            
            try:
                metadata = ModelMetadata(
                    name="test_model",
                    version="1.0.0",
                    framework=ModelFramework.PYTORCH,
                    file_path=tmp_file.name
                )
                
                is_valid, issues = validate_model_metadata(metadata)
                
                # May have issues due to file format, but should not crash
                assert isinstance(is_valid, bool)
                assert isinstance(issues, list)
            finally:
                os.unlink(tmp_file.name)
    
    def test_validate_optimization_session_success(self):
        """Test successful optimization session validation."""
        session = OptimizationSession(
            model_id="model_123",
            criteria_name="default"
        )
        
        is_valid, issues = validate_optimization_session(session)
        
        assert is_valid
        assert len(issues) == 0
    
    def test_validate_optimization_session_with_steps(self):
        """Test optimization session validation with steps."""
        step = OptimizationStep(
            technique="quantization",
            parameters={'quantization_bits': 8}
        )
        
        session = OptimizationSession(
            model_id="model_123",
            steps=[step]
        )
        
        is_valid, issues = validate_optimization_session(session)
        
        assert is_valid
        assert len(issues) == 0
    
    def test_validate_analysis_report_success(self):
        """Test successful analysis report validation."""
        report = AnalysisReport(
            model_id="model_123",
            analysis_duration_seconds=30.0
        )
        
        is_valid, issues = validate_analysis_report(report)
        
        assert is_valid
        assert len(issues) == 0
    
    def test_validate_analysis_report_with_opportunities(self):
        """Test analysis report validation with opportunities."""
        opportunity = OptimizationOpportunity(
            technique="quantization",
            confidence_score=0.8
        )
        
        recommendation = Recommendation(
            technique="quantization",
            priority=2
        )
        
        report = AnalysisReport(
            model_id="model_123",
            optimization_opportunities=[opportunity],
            recommendations=[recommendation]
        )
        
        is_valid, issues = validate_analysis_report(report)
        
        assert is_valid
        assert len(issues) == 0
    
    def test_validate_evaluation_report_success(self):
        """Test successful evaluation report validation."""
        benchmark = BenchmarkResult(
            benchmark_name="accuracy_test",
            score=0.95,
            execution_time_seconds=10.0
        )
        
        metrics = PerformanceMetrics(
            accuracy=0.95,
            inference_time_ms=50.0,
            memory_usage_mb=100.0
        )
        
        report = EvaluationReport(
            model_id="model_123",
            benchmarks=[benchmark],
            performance_metrics=metrics
        )
        
        is_valid, issues = validate_evaluation_report(report)
        
        assert is_valid
        assert len(issues) == 0


class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_calculate_file_checksum(self):
        """Test file checksum calculation."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            test_content = b"test content for checksum"
            tmp_file.write(test_content)
            tmp_file.flush()
            
            try:
                checksum = calculate_file_checksum(tmp_file.name)
                
                assert isinstance(checksum, str)
                assert len(checksum) == 64  # SHA256 hex length
                
                # Calculate again to ensure consistency
                checksum2 = calculate_file_checksum(tmp_file.name)
                assert checksum == checksum2
            finally:
                os.unlink(tmp_file.name)
    
    def test_validate_file_integrity_success(self):
        """Test successful file integrity validation."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            test_content = b"test content for integrity"
            tmp_file.write(test_content)
            tmp_file.flush()
            
            try:
                # Calculate correct checksum
                correct_checksum = calculate_file_checksum(tmp_file.name)
                
                # Validate with correct checksum
                is_valid = validate_file_integrity(tmp_file.name, correct_checksum)
                assert is_valid
                
                # Validate with incorrect checksum
                wrong_checksum = "0" * 64
                is_valid = validate_file_integrity(tmp_file.name, wrong_checksum)
                assert not is_valid
            finally:
                os.unlink(tmp_file.name)
    
    def test_validate_file_integrity_nonexistent_file(self):
        """Test file integrity validation with nonexistent file."""
        is_valid = validate_file_integrity("nonexistent.txt", "dummy_checksum")
        assert not is_valid


if __name__ == "__main__":
    pytest.main([__file__])
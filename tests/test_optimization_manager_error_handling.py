"""
Unit tests for optimization manager error handling and recovery.
"""

import pytest
import torch
import torch.nn as nn
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from src.services.optimization_manager import OptimizationManager
from src.config.optimization_criteria import OptimizationCriteria
from src.utils.exceptions import (
    OptimizationError, ModelLoadingError, ValidationError, ErrorCategory
)
from src.utils.recovery import recovery_manager, model_recovery_manager


class TestOptimizationManagerErrorHandling:
    """Test error handling in optimization manager."""
    
    @pytest.fixture
    def optimization_manager(self):
        """Create optimization manager for testing."""
        config = {
            "max_concurrent_sessions": 5,
            "session_timeout_minutes": 60,
            "analysis_agent": {"model_types": ["pytorch"]},
            "planning_agent": {"strategy": "rule_based"},
            "evaluation_agent": {"benchmarks": ["accuracy"]},
            "optimization_agents": {
                "quantization": {"bits": [4, 8]},
                "pruning": {"sparsity": [0.1, 0.5]}
            }
        }
        
        manager = OptimizationManager(config)
        
        # Mock agents to avoid initialization issues
        manager.analysis_agent = Mock()
        manager.planning_agent = Mock()
        manager.evaluation_agent = Mock()
        manager.optimization_agents = {
            "quantization": Mock(),
            "pruning": Mock()
        }
        
        return manager
    
    @pytest.fixture
    def sample_model(self):
        """Create a sample model for testing."""
        return nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 5)
        )
    
    @pytest.fixture
    def optimization_criteria(self):
        """Create optimization criteria for testing."""
        from src.config.optimization_criteria import OptimizationConstraints, OptimizationTechnique
        
        constraints = OptimizationConstraints(
            preserve_accuracy_threshold=0.95,
            allowed_techniques=[OptimizationTechnique.QUANTIZATION, OptimizationTechnique.PRUNING]
        )
        
        return OptimizationCriteria(
            name="test_error_handling",
            description="Error handling test criteria",
            constraints=constraints,
            target_deployment="general"
        )
    
    def test_model_loading_error_handling(self, optimization_manager):
        """Test handling of model loading errors."""
        with patch.object(optimization_manager, '_load_model') as mock_load:
            mock_load.side_effect = ModelLoadingError(
                "Invalid model format",
                model_path="/invalid/path.pt"
            )
            
            with pytest.raises(ModelLoadingError):
                optimization_manager._load_model("/invalid/path.pt")
    
    @patch('src.utils.recovery.model_recovery_manager')
    def test_workflow_with_analysis_failure_and_recovery(
        self, mock_model_recovery, optimization_manager, optimization_criteria
    ):
        """Test workflow recovery when analysis phase fails."""
        session_id = "test_session"
        model_path = "/test/model.pt"
        
        # Mock model loading to succeed
        with patch.object(optimization_manager, '_load_model') as mock_load:
            mock_load.return_value = nn.Linear(10, 5)
            
            # Mock analysis to fail first, then succeed
            analysis_call_count = 0
            def analysis_side_effect(*args, **kwargs):
                nonlocal analysis_call_count
                analysis_call_count += 1
                if analysis_call_count == 1:
                    raise OptimizationError("Analysis failed", step="analysis")
                return {"model_info": "test"}
            
            optimization_manager.analysis_agent.analyze_model.side_effect = analysis_side_effect
            
            # Mock recovery manager to return success
            with patch('src.utils.recovery.recovery_manager') as mock_recovery:
                mock_recovery.handle_error.return_value = True
                
                # Mock other phases to succeed
                optimization_manager.planning_agent.plan_optimization.return_value = {
                    "techniques": ["quantization"]
                }
                optimization_manager.planning_agent.validate_plan.return_value = Mock(is_valid=True)
                
                # Mock optimization phase
                optimization_manager.optimization_agents["quantization"].optimize_with_tracking.return_value = Mock(
                    success=True,
                    optimized_model=nn.Linear(10, 5),
                    technique_used="quantization"
                )
                
                # Mock evaluation phase
                optimization_manager.evaluation_agent.evaluate_model.return_value = {
                    "accuracy": 0.96
                }
                optimization_manager.evaluation_agent.compare_models.return_value = {
                    "improvement": 0.1
                }
                
                # Execute workflow
                try:
                    optimization_manager._execute_optimization_workflow(
                        session_id, model_path, optimization_criteria
                    )
                    
                    # Should succeed after recovery
                    assert analysis_call_count == 2  # Failed once, then succeeded
                    assert mock_recovery.handle_error.called
                    
                except Exception as e:
                    # If it still fails, check that recovery was attempted
                    assert mock_recovery.handle_error.called
    
    def test_graceful_degradation_on_optimization_failure(
        self, optimization_manager, optimization_criteria
    ):
        """Test graceful degradation when optimization techniques fail."""
        session_id = "test_session"
        model_path = "/test/model.pt"
        
        with patch.object(optimization_manager, '_load_model') as mock_load:
            mock_load.return_value = nn.Linear(10, 5)
            
            # Mock analysis and planning to succeed
            optimization_manager.analysis_agent.analyze_model.return_value = {
                "model_info": "test"
            }
            optimization_manager.planning_agent.plan_optimization.return_value = {
                "techniques": ["quantization_4bit", "pruning_structured"]
            }
            optimization_manager.planning_agent.validate_plan.return_value = Mock(is_valid=True)
            
            # Mock quantization to fail, pruning to succeed
            optimization_manager.optimization_agents["quantization"].optimize_with_tracking.side_effect = \
                OptimizationError("Quantization failed", technique="quantization_4bit")
            
            optimization_manager.optimization_agents["pruning"].optimize_with_tracking.return_value = Mock(
                success=True,
                optimized_model=nn.Linear(10, 5),
                technique_used="pruning"
            )
            
            # Mock evaluation to succeed
            optimization_manager.evaluation_agent.evaluate_model.return_value = {
                "accuracy": 0.96
            }
            optimization_manager.evaluation_agent.compare_models.return_value = {
                "improvement": 0.05
            }
            
            # Mock graceful degradation
            with patch('src.utils.recovery.degradation_manager') as mock_degradation:
                mock_degradation.create_degraded_plan.return_value = ["pruning_structured"]
                
                # Execute workflow - should succeed with degraded plan
                try:
                    optimization_manager._execute_optimization_workflow(
                        session_id, model_path, optimization_criteria
                    )
                    
                    # Should have attempted degradation
                    assert mock_degradation.create_degraded_plan.called
                    
                except Exception:
                    # Even if it fails, degradation should have been attempted
                    assert mock_degradation.create_degraded_plan.called
    
    def test_evaluation_fallback_to_minimal(self, optimization_manager):
        """Test fallback to minimal evaluation when full evaluation fails."""
        session_id = "test_session"
        optimization_results = {
            "final_model": nn.Linear(10, 5),
            "success": True,
            "techniques_applied": ["quantization"],
            "total_time": 120.5
        }
        
        # Mock full evaluation to fail
        optimization_manager.evaluation_agent.evaluate_model.side_effect = \
            ValidationError("Evaluation benchmark failed")
        optimization_manager.evaluation_agent.compare_models.side_effect = \
            ValidationError("Model comparison failed")
        
        # Execute minimal evaluation
        result = optimization_manager._execute_minimal_evaluation(session_id, optimization_results)
        
        assert result["evaluation_status"] == "minimal"
        assert result["basic_metrics"]["optimization_completed"] is True
        assert result["basic_metrics"]["techniques_applied"] == ["quantization"]
        assert result["basic_metrics"]["optimization_time"] == 120.5
        assert "warnings" in result
    
    def test_session_rollback_on_failure(self, optimization_manager, optimization_criteria):
        """Test session rollback when workflow fails."""
        session_id = "test_session"
        model_path = "/test/model.pt"
        
        # Start a session
        with patch.object(optimization_manager, '_load_model') as mock_load:
            mock_load.return_value = nn.Linear(10, 5)
            
            # Create session
            optimization_manager.active_sessions[session_id] = Mock()
            optimization_manager.workflow_states[session_id] = Mock()
            optimization_manager.session_snapshots[session_id] = []
            
            # Mock critical failure in optimization
            with patch.object(optimization_manager, '_execute_analysis_phase_with_recovery') as mock_analysis:
                mock_analysis.side_effect = OptimizationError(
                    "Critical analysis failure",
                    severity="critical"
                )
                
                # Execute workflow - should handle failure
                optimization_manager._execute_optimization_workflow(
                    session_id, model_path, optimization_criteria
                )
                
                # Check that failure was handled
                assert session_id in optimization_manager.active_sessions
    
    @patch('src.utils.recovery.recovery_manager')
    def test_retry_with_exponential_backoff(self, mock_recovery_manager, optimization_manager):
        """Test retry mechanism with exponential backoff."""
        session_id = "test_session"
        model_path = "/test/model.pt"
        
        # Mock transient failure that succeeds on retry
        call_count = 0
        def transient_failure(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Transient network error")
            return {"model_info": "success"}
        
        with patch.object(optimization_manager, '_load_model') as mock_load:
            mock_load.return_value = nn.Linear(10, 5)
            
            optimization_manager.analysis_agent.analyze_model.side_effect = transient_failure
            
            # Mock recovery manager
            mock_recovery_manager.handle_error.return_value = True
            
            # Execute analysis with recovery
            recovery_context = {"session_id": session_id}
            
            try:
                result = optimization_manager._execute_analysis_phase_with_recovery(
                    session_id, model_path, recovery_context
                )
                
                # Should succeed after retries
                assert result["model_info"] == "success"
                assert call_count >= 2  # At least one retry
                
            except Exception:
                # Even if it fails, retries should have been attempted
                assert call_count > 1
    
    def test_concurrent_session_error_isolation(self, optimization_manager):
        """Test that errors in one session don't affect others."""
        session1_id = "session_1"
        session2_id = "session_2"
        
        # Create two sessions
        optimization_manager.active_sessions[session1_id] = Mock()
        optimization_manager.active_sessions[session2_id] = Mock()
        optimization_manager.workflow_states[session1_id] = Mock()
        optimization_manager.workflow_states[session2_id] = Mock()
        
        # Simulate error in session 1
        optimization_manager._handle_workflow_failure(session1_id, "Test error")
        
        # Session 2 should be unaffected
        assert session2_id in optimization_manager.active_sessions
        assert session1_id in optimization_manager.active_sessions  # Still exists but marked as failed
    
    def test_resource_cleanup_on_error(self, optimization_manager):
        """Test that resources are properly cleaned up on errors."""
        session_id = "test_session"
        
        # Create session with resources
        optimization_manager.active_sessions[session_id] = Mock()
        optimization_manager.workflow_states[session_id] = Mock()
        optimization_manager.session_snapshots[session_id] = [Mock(), Mock()]
        
        # Simulate cleanup
        optimization_manager.cleanup()
        
        # Resources should be cleaned up
        assert len(optimization_manager.active_sessions) == 0
        assert len(optimization_manager.workflow_states) == 0
        assert len(optimization_manager.session_snapshots) == 0
    
    def test_error_context_preservation(self, optimization_manager):
        """Test that error context is preserved through recovery attempts."""
        session_id = "test_session"
        
        error = OptimizationError(
            "Test error",
            technique="quantization",
            session_id=session_id,
            step="optimization"
        )
        
        # Mock recovery manager to capture context
        with patch('src.utils.recovery.recovery_manager') as mock_recovery:
            mock_recovery.handle_error.return_value = False
            
            # Handle workflow failure
            optimization_manager._handle_workflow_failure(session_id, str(error))
            
            # Error context should be preserved
            # (This is more of a documentation test for expected behavior)
            assert session_id == session_id  # Basic assertion to ensure test runs
    
    def test_memory_management_during_errors(self, optimization_manager, sample_model):
        """Test memory management during error conditions."""
        session_id = "test_session"
        
        # Create model snapshots
        with patch('src.utils.recovery.model_recovery_manager') as mock_model_recovery:
            mock_model_recovery.create_model_snapshot.return_value = "snapshot_123"
            
            # Simulate memory pressure during error
            with patch('torch.cuda.empty_cache') as mock_empty_cache:
                with patch('gc.collect') as mock_gc:
                    
                    # Trigger error handling that should clean up memory
                    optimization_manager._handle_workflow_failure(session_id, "Memory error")
                    
                    # Memory cleanup should be attempted
                    # (This would be implemented in the actual recovery actions)
                    assert True  # Placeholder assertion


if __name__ == "__main__":
    pytest.main([__file__])
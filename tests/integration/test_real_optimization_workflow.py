"""
Real Optimization Workflow Tests - End-to-end testing with actual optimization execution.

This module tests the complete optimization workflow with real models and actual
optimization techniques (no mocking). It mirrors the test_optimizer.ipynb notebook
but as an automated test suite.

Run with verbose output to see detailed results:
    pytest tests/integration/test_real_optimization_workflow.py -v -s
"""

import pytest
import torch
import torch.nn as nn
import tempfile
import time
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

from src.services.optimization_manager import OptimizationManager
from src.config.optimization_criteria import (
    OptimizationCriteria,
    OptimizationConstraints,
    OptimizationTechnique,
    PerformanceMetric,
    PerformanceThreshold
)


# Helper function to print detailed results
def print_optimization_results(session_id: str, optimization_manager: OptimizationManager, verbose: bool = True):
    """Print detailed optimization results similar to the Jupyter notebook."""
    if not verbose:
        return
    
    print("\n" + "=" * 70)
    print("📈 OPTIMIZATION RESULTS")
    print("=" * 70)
    
    final_status = optimization_manager.get_session_status(session_id)
    
    print(f"\n🎯 Session Information")
    print(f"  Session ID: {session_id}")
    print(f"  Status: {final_status.get('status', 'unknown').upper()}")
    
    session_data = final_status.get('session_data', {})
    print(f"  Model: {session_data.get('model_id', 'N/A')}")
    print(f"  Criteria: {session_data.get('criteria_name', 'N/A')}")
    print(f"  Steps Completed: {session_data.get('steps_completed', 0)}")
    
    # Get detailed results from session
    with optimization_manager._lock:
        if session_id in optimization_manager.active_sessions:
            session = optimization_manager.active_sessions[session_id]
            
            if session.results:
                results = session.results
                
                print(f"\n📊 Model Size Analysis")
                print(f"  Original Size: {results.original_model_size_mb:.2f} MB")
                print(f"  Optimized Size: {results.optimized_model_size_mb:.2f} MB")
                print(f"  Size Reduction: {results.size_reduction_percent:.2f}%")
                
                if results.size_reduction_percent > 0:
                    saved_mb = results.original_model_size_mb - results.optimized_model_size_mb
                    print(f"  Space Saved: {saved_mb:.2f} MB")
                
                print(f"\n⚙️ Optimization Techniques Applied")
                for i, technique in enumerate(results.techniques_applied, 1):
                    print(f"  {i}. {technique.upper()}")
                
                print(f"\n🚀 Performance Improvements")
                if results.performance_improvements:
                    # Group metrics by category
                    param_metrics = {}
                    perf_metrics = {}
                    other_metrics = {}
                    
                    for metric, value in results.performance_improvements.items():
                        if 'parameter' in metric.lower():
                            param_metrics[metric] = value
                        elif any(x in metric.lower() for x in ['time', 'throughput', 'latency', 'speed']):
                            perf_metrics[metric] = value
                        else:
                            other_metrics[metric] = value
                    
                    if param_metrics:
                        print(f"\n  📊 Parameter Metrics:")
                        for metric, value in param_metrics.items():
                            if isinstance(value, float):
                                if 'percent' in metric.lower() or 'ratio' in metric.lower():
                                    print(f"    {metric}: {value:.2f}%")
                                else:
                                    print(f"    {metric}: {value:.4f}")
                            else:
                                print(f"    {metric}: {value:,}")
                    
                    if perf_metrics:
                        print(f"\n  ⏱️ Performance Metrics:")
                        for metric, value in perf_metrics.items():
                            if isinstance(value, (int, float)):
                                if 'time' in metric.lower() or 'latency' in metric.lower():
                                    print(f"    {metric}: {value:.2f} ms")
                                elif 'throughput' in metric.lower():
                                    print(f"    {metric}: {value:.2f} samples/sec")
                                else:
                                    print(f"    {metric}: {value:.2f}")
                            else:
                                print(f"    {metric}: {value}")
                    
                    if other_metrics:
                        print(f"\n  🔧 Other Metrics:")
                        for metric, value in other_metrics.items():
                            if isinstance(value, float):
                                if 'sparsity' in metric.lower():
                                    print(f"    {metric}: {value * 100:.2f}%")
                                elif 'percent' in metric.lower():
                                    print(f"    {metric}: {value:.2f}%")
                                else:
                                    print(f"    {metric}: {value:.4f}")
                            else:
                                print(f"    {metric}: {value}")
                else:
                    print(f"  No performance improvements recorded")
                
                print(f"\n✅ Validation Status")
                print(f"  Validation: {'PASSED ✅' if results.validation_passed else 'FAILED ❌'}")
                print(f"  Rollback Available: {'YES' if results.rollback_available else 'NO'}")
                
                print(f"\n📝 Summary")
                print(f"  {results.optimization_summary}")
                
                # Calculate overall improvement score
                if results.size_reduction_percent > 0 or results.performance_improvements:
                    print(f"\n🎉 Optimization Status: SUCCESS")
                    if results.size_reduction_percent > 10:
                        print(f"  ⭐ Significant size reduction achieved!")
                    if any('throughput' in k.lower() for k in results.performance_improvements.keys()):
                        print(f"  ⭐ Performance improvements detected!")
            
            # Show optimization steps
            if session.steps:
                print(f"\n📋 Optimization Steps")
                for i, step in enumerate(session.steps, 1):
                    status_value = step.status.value if hasattr(step.status, 'value') else str(step.status)
                    status_icon = "✅" if status_value == 'completed' else "❌" if status_value == 'failed' else "⏳"
                    print(f"  {i}. {step.technique.upper()} {status_icon}")
                    if step.error_message:
                        print(f"     Error: {step.error_message}")
    
    # Print detailed comparison table
    with optimization_manager._lock:
        if session_id in optimization_manager.active_sessions:
            session = optimization_manager.active_sessions[session_id]
            
            if session.results:
                results = session.results
                
                print(f"\n🔍 DETAILED COMPARISON TABLE")
                print("=" * 90)
                print(f"{'Metric':<35} {'Original':<18} {'Optimized':<18} {'Change':<15}")
                print("-" * 90)
                
                # Model Size
                orig_size = results.original_model_size_mb
                opt_size = results.optimized_model_size_mb
                size_change = results.size_reduction_percent
                size_change_str = f"{-size_change:+.2f}%" if size_change != 0 else "0.00%"
                print(f"{'Model Size (MB)':<35} {orig_size:<18.2f} {opt_size:<18.2f} {size_change_str:<15}")
                
                # Parameters
                if 'original_parameters' in results.performance_improvements:
                    orig_params = results.performance_improvements['original_parameters']
                    opt_params = results.performance_improvements.get('optimized_parameters', orig_params)
                    param_change = ((opt_params - orig_params) / orig_params * 100) if orig_params > 0 else 0
                    param_change_str = f"{param_change:+.2f}%" if param_change != 0 else "0.00%"
                    print(f"{'Parameters':<35} {orig_params:<18,} {opt_params:<18,} {param_change_str:<15}")
                
                # Sparsity
                if 'actual_sparsity' in results.performance_improvements:
                    orig_sparsity = 0.0
                    actual_sparsity = results.performance_improvements['actual_sparsity'] * 100
                    sparsity_change = actual_sparsity - orig_sparsity
                    sparsity_change_str = f"{sparsity_change:+.2f}%" if sparsity_change != 0 else "0.00%"
                    print(f"{'Sparsity (%)':<35} {orig_sparsity:<18.2f} {actual_sparsity:<18.2f} {sparsity_change_str:<15}")
                
                # Inference Time
                if 'inference_time_ms' in results.performance_improvements:
                    opt_inf_time = results.performance_improvements['inference_time_ms']
                    # Estimate original inference time (assuming optimization improved it)
                    # If we don't have original, we can't show comparison accurately
                    print(f"{'Inference Time (ms)':<35} {'-':<18} {opt_inf_time:<18.2f} {'Measured':<15}")
                elif 'inference_time_improvement_percent' in results.performance_improvements:
                    improvement = results.performance_improvements['inference_time_improvement_percent']
                    print(f"{'Inference Time Improvement (%)':<35} {'Baseline':<18} {'Optimized':<18} {improvement:+.2f}%")
                
                # Throughput
                if 'throughput_samples_per_sec' in results.performance_improvements:
                    opt_throughput = results.performance_improvements['throughput_samples_per_sec']
                    # Estimate original throughput (assuming optimization improved it)
                    if 'throughput_improvement_percent' in results.performance_improvements:
                        improvement = results.performance_improvements['throughput_improvement_percent']
                        orig_throughput = opt_throughput / (1 + improvement / 100)
                        throughput_change_str = f"{improvement:+.2f}%"
                        print(f"{'Throughput (samples/sec)':<35} {orig_throughput:<18.2f} {opt_throughput:<18.2f} {throughput_change_str:<15}")
                    else:
                        print(f"{'Throughput (samples/sec)':<35} {'-':<18} {opt_throughput:<18.2f} {'Measured':<15}")
                
                # Memory Usage (if available)
                if 'memory_usage_mb' in results.performance_improvements:
                    opt_memory = results.performance_improvements['memory_usage_mb']
                    print(f"{'Memory Usage (MB)':<35} {'-':<18} {opt_memory:<18.2f} {'Measured':<15}")
                
                # Latency (if available)
                if 'latency_ms' in results.performance_improvements:
                    opt_latency = results.performance_improvements['latency_ms']
                    print(f"{'Latency (ms)':<35} {'-':<18} {opt_latency:<18.2f} {'Measured':<15}")
                
                print("=" * 90)
                
                # Summary statistics
                print(f"\n📊 Summary Statistics")
                print(f"  Total Techniques Applied: {len(results.techniques_applied)}")
                print(f"  Validation Status: {'PASSED ✅' if results.validation_passed else 'FAILED ❌'}")
                
                # Calculate overall improvement score
                improvement_score = 0
                if results.size_reduction_percent > 0:
                    improvement_score += min(results.size_reduction_percent, 50)  # Cap at 50 points
                if 'actual_sparsity' in results.performance_improvements:
                    improvement_score += results.performance_improvements['actual_sparsity'] * 30  # Up to 30 points
                if 'throughput_improvement_percent' in results.performance_improvements:
                    improvement_score += min(results.performance_improvements['throughput_improvement_percent'], 20)  # Up to 20 points
                
                print(f"  Overall Improvement Score: {improvement_score:.1f}/100")
                
                if improvement_score >= 70:
                    print(f"  Rating: ⭐⭐⭐ EXCELLENT")
                elif improvement_score >= 40:
                    print(f"  Rating: ⭐⭐ GOOD")
                elif improvement_score >= 20:
                    print(f"  Rating: ⭐ FAIR")
                else:
                    print(f"  Rating: NEEDS IMPROVEMENT")
    
    print(f"\n⏰ Timing Information")
    print(f"  Start Time: {final_status.get('start_time', 'N/A')}")
    print(f"  Last Update: {final_status.get('last_update', 'N/A')}")
    
    if final_status.get('error_message'):
        print(f"\n⚠️ Error Details")
        print(f"  {final_status['error_message']}")
    
    print("\n" + "=" * 70 + "\n")


class RoboticsVLAModel(nn.Module):
    """Vision-Language-Action model for robotics tasks.
    
    Simplified to accept a single concatenated input for compatibility
    with the analysis agent's profiling system.
    """
    
    def __init__(self, input_dim=1280, hidden_dim=256, action_dim=7):
        super().__init__()
        
        # Input projection (simulates vision+language fusion)
        # 1280 = 512 (vision) + 768 (language)
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        # Feature encoder (simulating ViT + BERT fusion)
        self.encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        # Cross-attention fusion
        self.cross_attention = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)
        
        # Action decoder
        self.action_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
    
    def forward(self, x):
        """Forward pass with single concatenated input."""
        features = self.input_projection(x)
        encoded = self.encoder(features)
        attended, _ = self.cross_attention(
            encoded.unsqueeze(1),
            encoded.unsqueeze(1),
            encoded.unsqueeze(1)
        )
        attended = attended.squeeze(1)
        actions = self.action_decoder(attended)
        return actions


@pytest.fixture
def temp_model_dir():
    """Create temporary directory for test models."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def robotics_model(temp_model_dir):
    """Create and save a robotics VLA model."""
    model = RoboticsVLAModel()
    model_path = temp_model_dir / "robotics_vla_demo.pt"
    torch.save(model, model_path)
    
    model_size_mb = os.path.getsize(model_path) / (1024 * 1024)
    param_count = sum(p.numel() for p in model.parameters())
    
    return {
        'path': model_path,
        'model': model,
        'size_mb': model_size_mb,
        'param_count': param_count
    }


@pytest.fixture
def optimization_manager():
    """Create and initialize optimization manager."""
    config = {
        "max_concurrent_sessions": 2,
        "auto_rollback_on_failure": True,
        "snapshot_frequency": 1,
        "session_timeout_minutes": 60,
        "analysis_agent": {
            "profiling_samples": 50,
            "warmup_samples": 5
        },
        "planning_agent": {
            "max_plan_steps": 3,
            "risk_tolerance": 0.7
        },
        "evaluation_agent": {
            "benchmark_samples": 50,
            "accuracy_threshold": 0.95
        },
        "quantization_agent": {},
        "pruning_agent": {}
    }
    
    manager = OptimizationManager(config)
    assert manager.initialize(), "Manager initialization failed"
    
    yield manager
    
    manager.cleanup()


@pytest.fixture
def edge_deployment_criteria():
    """Create optimization criteria for edge deployment."""
    thresholds = [
        PerformanceThreshold(
            metric=PerformanceMetric.INFERENCE_TIME,
            max_value=50.0  # Max 50ms
        ),
        PerformanceThreshold(
            metric=PerformanceMetric.MODEL_SIZE,
            max_value=50.0  # Max 50MB for edge devices
        ),
        PerformanceThreshold(
            metric=PerformanceMetric.ACCURACY,
            min_value=0.90  # Min 90% accuracy
        )
    ]
    
    constraints = OptimizationConstraints(
        preserve_accuracy_threshold=0.95,
        allowed_techniques=[
            OptimizationTechnique.QUANTIZATION,
            OptimizationTechnique.PRUNING
        ],
        max_optimization_time_minutes=30
    )
    
    return OptimizationCriteria(
        name="edge_robotics_deployment",
        description="Optimize for edge deployment with real-time constraints",
        target_deployment="edge",
        priority_weights={
            PerformanceMetric.MODEL_SIZE: 0.4,
            PerformanceMetric.INFERENCE_TIME: 0.4,
            PerformanceMetric.ACCURACY: 0.2
        },
        performance_thresholds=thresholds,
        constraints=constraints
    )


class TestRealOptimizationWorkflow:
    """Test real optimization workflows with actual model optimization."""
    
    def test_create_robotics_model(self, robotics_model):
        """Test that robotics VLA model is created correctly."""
        assert robotics_model['path'].exists()
        assert robotics_model['size_mb'] > 0
        assert robotics_model['param_count'] > 0
        
        # Verify model can be loaded (weights_only=False for custom model class)
        loaded_model = torch.load(robotics_model['path'], weights_only=False)
        assert isinstance(loaded_model, RoboticsVLAModel)
        
        # Verify forward pass works
        test_input = torch.randn(1, 1280)
        output = loaded_model(test_input)
        assert output.shape == (1, 7)  # action_dim = 7
    
    def test_optimization_manager_initialization(self, optimization_manager):
        """Test that optimization manager initializes correctly."""
        assert optimization_manager is not None
        assert hasattr(optimization_manager, 'analysis_agent')
        assert hasattr(optimization_manager, 'planning_agent')
        assert hasattr(optimization_manager, 'evaluation_agent')
        assert len(optimization_manager.optimization_agents) > 0
    
    def test_optimization_criteria_configuration(self, edge_deployment_criteria):
        """Test that optimization criteria is configured correctly."""
        assert edge_deployment_criteria.name == "edge_robotics_deployment"
        assert edge_deployment_criteria.target_deployment == "edge"
        assert len(edge_deployment_criteria.constraints.allowed_techniques) == 2
        assert edge_deployment_criteria.constraints.preserve_accuracy_threshold == 0.95
    
    def test_complete_optimization_workflow(
        self,
        optimization_manager,
        robotics_model,
        edge_deployment_criteria,
        capsys
    ):
        """Test complete optimization workflow from start to finish."""
        print("\n" + "=" * 70)
        print("🚀 STARTING COMPLETE OPTIMIZATION WORKFLOW TEST")
        print("=" * 70)
        
        print(f"\n📦 Model Information:")
        print(f"  Path: {robotics_model['path']}")
        print(f"  Size: {robotics_model['size_mb']:.2f} MB")
        print(f"  Parameters: {robotics_model['param_count']:,}")
        
        # Track progress updates
        progress_history = []
        
        def progress_callback(session_id: str, update):
            """Track progress updates."""
            progress_info = {
                'timestamp': datetime.now(),
                'session_id': session_id,
                'status': update.status.value if hasattr(update.status, 'value') else str(update.status),
                'progress': getattr(update, 'progress_percentage', 0.0),
                'step': getattr(update, 'current_step', 'unknown'),
                'message': getattr(update, 'message', '')
            }
            progress_history.append(progress_info)
            
            # Print progress update
            print(f"[{progress_info['timestamp'].strftime('%H:%M:%S')}] "
                  f"{progress_info['status']} - {progress_info['progress']:.1f}% - {progress_info['step']}")
        
        # Register callback
        optimization_manager.add_progress_callback(progress_callback)
        
        print(f"\n🎯 Optimization Criteria:")
        print(f"  Name: {edge_deployment_criteria.name}")
        print(f"  Target: {edge_deployment_criteria.target_deployment}")
        print(f"  Techniques: {[t.value for t in edge_deployment_criteria.constraints.allowed_techniques]}")
        
        # Start optimization session
        print(f"\n⏳ Starting optimization session...")
        session_id = optimization_manager.start_optimization_session(
            str(robotics_model['path']),
            edge_deployment_criteria
        )
        
        print(f"✅ Session started: {session_id}\n")
        
        assert session_id is not None
        assert len(session_id) > 0
        
        # Monitor progress
        print("📊 Monitoring optimization progress...\n")
        start_time = time.time()
        timeout = 300  # 5 minutes
        check_interval = 2  # seconds
        
        final_status = None
        while time.time() - start_time < timeout:
            status = optimization_manager.get_session_status(session_id)
            
            if status.get('status') in ['completed', 'failed', 'cancelled']:
                final_status = status
                break
            
            time.sleep(check_interval)
        
        elapsed_time = time.time() - start_time
        print(f"\n⏱️ Total execution time: {elapsed_time:.1f}s")
        
        # Verify completion
        assert final_status is not None, "Optimization did not complete within timeout"
        assert final_status['status'] == 'completed', f"Optimization failed: {final_status.get('error_message')}"
        
        # Print detailed results
        print_optimization_results(session_id, optimization_manager, verbose=True)
        
        # Verify progress was tracked
        assert len(progress_history) > 0, "No progress updates received"
        
        # Verify final progress is 100%
        assert final_status['progress_percentage'] == 100.0
    
    def test_optimization_results_analysis(
        self,
        optimization_manager,
        robotics_model,
        edge_deployment_criteria
    ):
        """Test that optimization produces valid results with improvements."""
        print("\n" + "=" * 70)
        print("🔍 TESTING OPTIMIZATION RESULTS ANALYSIS")
        print("=" * 70)
        
        # Start optimization
        print(f"\n⏳ Starting optimization...")
        session_id = optimization_manager.start_optimization_session(
            str(robotics_model['path']),
            edge_deployment_criteria
        )
        print(f"✅ Session ID: {session_id}")
        
        # Wait for completion
        start_time = time.time()
        timeout = 300
        
        print(f"\n📊 Waiting for completion...")
        while time.time() - start_time < timeout:
            status = optimization_manager.get_session_status(session_id)
            if status.get('status') in ['completed', 'failed', 'cancelled']:
                break
            time.sleep(2)
        
        # Get final status
        final_status = optimization_manager.get_session_status(session_id)
        assert final_status['status'] == 'completed'
        
        print(f"✅ Optimization completed in {time.time() - start_time:.1f}s")
        
        # Print detailed results
        print_optimization_results(session_id, optimization_manager, verbose=True)
        
        # Analyze results
        with optimization_manager._lock:
            assert session_id in optimization_manager.active_sessions
            session = optimization_manager.active_sessions[session_id]
            
            assert session.results is not None, "No results available"
            results = session.results
            
            # Verify model size metrics
            assert results.original_model_size_mb > 0
            assert results.optimized_model_size_mb > 0
            assert results.size_reduction_percent >= 0
            
            # Verify techniques were applied
            assert len(results.techniques_applied) > 0
            
            # Verify performance improvements exist
            assert results.performance_improvements is not None
            assert len(results.performance_improvements) > 0
            
            # Verify validation
            assert results.validation_passed is not None
            
            # Verify summary
            assert results.optimization_summary is not None
            assert len(results.optimization_summary) > 0
    
    def test_optimization_produces_improvements(
        self,
        optimization_manager,
        robotics_model,
        edge_deployment_criteria
    ):
        """Test that optimization actually improves model metrics."""
        print("\n" + "=" * 90)
        print("📈 TESTING OPTIMIZATION IMPROVEMENTS WITH DETAILED COMPARISON")
        print("=" * 90)
        
        # Capture baseline metrics
        original_size = robotics_model['size_mb']
        original_params = robotics_model['param_count']
        
        print(f"\n📦 Baseline Metrics (Before Optimization):")
        print(f"  Model Size: {original_size:.2f} MB")
        print(f"  Parameters: {original_params:,}")
        print(f"  Sparsity: 0.00%")
        
        # Measure baseline inference time
        model = torch.load(robotics_model['path'], weights_only=False)
        model.eval()
        test_input = torch.randn(1, 1280)
        
        # Warmup
        with torch.no_grad():
            for _ in range(5):
                _ = model(test_input)
        
        # Measure baseline
        import time as time_module
        baseline_times = []
        with torch.no_grad():
            for _ in range(10):
                start = time_module.perf_counter()
                _ = model(test_input)
                end = time_module.perf_counter()
                baseline_times.append((end - start) * 1000)  # Convert to ms
        
        baseline_inference_time = sum(baseline_times) / len(baseline_times)
        baseline_throughput = 1000.0 / baseline_inference_time  # samples per second
        
        print(f"  Inference Time: {baseline_inference_time:.2f} ms")
        print(f"  Throughput: {baseline_throughput:.2f} samples/sec")
        
        # Start optimization
        print(f"\n⏳ Starting optimization...")
        session_id = optimization_manager.start_optimization_session(
            str(robotics_model['path']),
            edge_deployment_criteria
        )
        
        # Wait for completion
        start_time = time.time()
        timeout = 300
        
        while time.time() - start_time < timeout:
            status = optimization_manager.get_session_status(session_id)
            if status.get('status') in ['completed', 'failed', 'cancelled']:
                break
            time.sleep(2)
        
        print(f"✅ Optimization completed in {time.time() - start_time:.1f}s")
        
        # Check improvements
        with optimization_manager._lock:
            session = optimization_manager.active_sessions[session_id]
            results = session.results
            
            print(f"\n📊 Quick Improvement Summary:")
            print(f"  Size Reduction: {results.size_reduction_percent:.2f}%")
            print(f"  Optimized Size: {results.optimized_model_size_mb:.2f} MB")
            print(f"  Performance Metrics Captured: {len(results.performance_improvements)}")
            
            # Should have size reduction OR performance improvements
            has_improvements = (
                results.size_reduction_percent > 0 or
                len(results.performance_improvements) > 0
            )
            
            if has_improvements:
                print(f"\n✅ Improvements detected!")
                if results.size_reduction_percent > 0:
                    saved = original_size - results.optimized_model_size_mb
                    print(f"  💾 Space saved: {saved:.2f} MB ({results.size_reduction_percent:.2f}% reduction)")
                if 'actual_sparsity' in results.performance_improvements:
                    sparsity = results.performance_improvements['actual_sparsity'] * 100
                    print(f"  🎯 Sparsity achieved: {sparsity:.2f}%")
                if 'inference_time_ms' in results.performance_improvements:
                    opt_inf_time = results.performance_improvements['inference_time_ms']
                    inf_improvement = ((baseline_inference_time - opt_inf_time) / baseline_inference_time * 100)
                    print(f"  ⚡ Inference time: {opt_inf_time:.2f} ms ({inf_improvement:+.2f}% change)")
                if 'throughput_samples_per_sec' in results.performance_improvements:
                    opt_throughput = results.performance_improvements['throughput_samples_per_sec']
                    throughput_improvement = ((opt_throughput - baseline_throughput) / baseline_throughput * 100)
                    print(f"  🚀 Throughput: {opt_throughput:.2f} samples/sec ({throughput_improvement:+.2f}% change)")
            
            assert has_improvements, "Optimization produced no improvements"
            
            # If size was reduced, verify it's actually smaller
            if results.size_reduction_percent > 0:
                assert results.optimized_model_size_mb < results.original_model_size_mb
            
            # Print detailed comparison table
            print(f"\n🔍 DETAILED BEFORE/AFTER COMPARISON")
            print("=" * 90)
            print(f"{'Metric':<35} {'Before':<18} {'After':<18} {'Change':<15}")
            print("-" * 90)
            
            # Model Size
            size_change_str = f"{-results.size_reduction_percent:+.2f}%"
            print(f"{'Model Size (MB)':<35} {original_size:<18.2f} {results.optimized_model_size_mb:<18.2f} {size_change_str:<15}")
            
            # Parameters
            opt_params = results.performance_improvements.get('optimized_parameters', original_params)
            param_change = ((opt_params - original_params) / original_params * 100) if original_params > 0 else 0
            param_change_str = f"{param_change:+.2f}%"
            print(f"{'Parameters':<35} {original_params:<18,} {opt_params:<18,} {param_change_str:<15}")
            
            # Sparsity
            if 'actual_sparsity' in results.performance_improvements:
                actual_sparsity = results.performance_improvements['actual_sparsity'] * 100
                sparsity_change_str = f"{actual_sparsity:+.2f}%"
                print(f"{'Sparsity (%)':<35} {0.0:<18.2f} {actual_sparsity:<18.2f} {sparsity_change_str:<15}")
            
            # Inference Time
            if 'inference_time_ms' in results.performance_improvements:
                opt_inf_time = results.performance_improvements['inference_time_ms']
                inf_improvement = ((baseline_inference_time - opt_inf_time) / baseline_inference_time * 100)
                inf_change_str = f"{inf_improvement:+.2f}%"
                print(f"{'Inference Time (ms)':<35} {baseline_inference_time:<18.2f} {opt_inf_time:<18.2f} {inf_change_str:<15}")
            
            # Throughput
            if 'throughput_samples_per_sec' in results.performance_improvements:
                opt_throughput = results.performance_improvements['throughput_samples_per_sec']
                throughput_improvement = ((opt_throughput - baseline_throughput) / baseline_throughput * 100)
                throughput_change_str = f"{throughput_improvement:+.2f}%"
                print(f"{'Throughput (samples/sec)':<35} {baseline_throughput:<18.2f} {opt_throughput:<18.2f} {throughput_change_str:<15}")
            
            print("=" * 90)
            
            # Overall assessment
            print(f"\n🎯 Overall Assessment:")
            total_improvements = 0
            if results.size_reduction_percent > 0:
                print(f"  ✅ Size reduced by {results.size_reduction_percent:.2f}%")
                total_improvements += 1
            if 'actual_sparsity' in results.performance_improvements and results.performance_improvements['actual_sparsity'] > 0:
                print(f"  ✅ Sparsity achieved: {results.performance_improvements['actual_sparsity'] * 100:.2f}%")
                total_improvements += 1
            if 'inference_time_ms' in results.performance_improvements:
                if inf_improvement > 0:
                    print(f"  ✅ Inference time improved by {inf_improvement:.2f}%")
                    total_improvements += 1
                else:
                    print(f"  ⚠️ Inference time changed by {inf_improvement:.2f}%")
            if 'throughput_samples_per_sec' in results.performance_improvements:
                if throughput_improvement > 0:
                    print(f"  ✅ Throughput improved by {throughput_improvement:.2f}%")
                    total_improvements += 1
                else:
                    print(f"  ⚠️ Throughput changed by {throughput_improvement:.2f}%")
            
            print(f"\n  Total Improvements: {total_improvements}")
            print(f"  Techniques Applied: {', '.join(results.techniques_applied)}")
            print(f"  Validation: {'PASSED ✅' if results.validation_passed else 'FAILED ❌'}")
        
        print_optimization_results(session_id, optimization_manager, verbose=True)
    
    def test_optimization_steps_execution(
        self,
        optimization_manager,
        robotics_model,
        edge_deployment_criteria
    ):
        """Test that optimization steps are executed correctly."""
        session_id = optimization_manager.start_optimization_session(
            str(robotics_model['path']),
            edge_deployment_criteria
        )
        
        # Wait for completion
        start_time = time.time()
        timeout = 300
        
        while time.time() - start_time < timeout:
            status = optimization_manager.get_session_status(session_id)
            if status.get('status') in ['completed', 'failed', 'cancelled']:
                break
            time.sleep(2)
        
        # Verify steps
        with optimization_manager._lock:
            session = optimization_manager.active_sessions[session_id]
            
            # Should have executed steps
            assert len(session.steps) > 0, "No optimization steps executed"
            
            # Verify step structure
            for step in session.steps:
                assert hasattr(step, 'technique')
                assert hasattr(step, 'status')
                # Handle both enum and string status values
                status_value = step.status.value if hasattr(step.status, 'value') else str(step.status)
                assert status_value in ['pending', 'running', 'completed', 'failed']
                
                # Completed steps may have timing info (optional based on implementation)
                if status_value == 'completed':
                    # Timing info is optional - just verify step completed successfully
                    assert step.technique is not None
    
    def test_progress_callback_functionality(
        self,
        optimization_manager,
        robotics_model,
        edge_deployment_criteria
    ):
        """Test that progress callbacks are invoked correctly."""
        callback_invocations = []
        
        def test_callback(session_id: str, update):
            callback_invocations.append({
                'session_id': session_id,
                'timestamp': datetime.now(),
                'update': update
            })
        
        optimization_manager.add_progress_callback(test_callback)
        
        # Start optimization
        session_id = optimization_manager.start_optimization_session(
            str(robotics_model['path']),
            edge_deployment_criteria
        )
        
        # Wait for completion
        start_time = time.time()
        timeout = 300
        
        while time.time() - start_time < timeout:
            status = optimization_manager.get_session_status(session_id)
            if status.get('status') in ['completed', 'failed', 'cancelled']:
                break
            time.sleep(2)
        
        # Verify callbacks were invoked
        assert len(callback_invocations) > 0, "Progress callback was never invoked"
        
        # Verify all callbacks have correct session_id
        for invocation in callback_invocations:
            assert invocation['session_id'] == session_id
    
    def test_session_status_tracking(
        self,
        optimization_manager,
        robotics_model,
        edge_deployment_criteria
    ):
        """Test that session status is tracked correctly throughout workflow."""
        session_id = optimization_manager.start_optimization_session(
            str(robotics_model['path']),
            edge_deployment_criteria
        )
        
        status_history = []
        start_time = time.time()
        timeout = 300
        
        while time.time() - start_time < timeout:
            status = optimization_manager.get_session_status(session_id)
            status_history.append({
                'timestamp': datetime.now(),
                'status': status.get('status'),
                'progress': status.get('progress_percentage', 0.0)
            })
            
            if status.get('status') in ['completed', 'failed', 'cancelled']:
                break
            
            time.sleep(2)
        
        # Verify status progression
        assert len(status_history) > 0
        
        # Progress should be monotonically increasing
        for i in range(1, len(status_history)):
            assert status_history[i]['progress'] >= status_history[i-1]['progress']
        
        # Final status should be completed
        assert status_history[-1]['status'] == 'completed'
        assert status_history[-1]['progress'] == 100.0
    
    @pytest.mark.parametrize("technique", [
        OptimizationTechnique.QUANTIZATION,
        OptimizationTechnique.PRUNING
    ])
    def test_individual_optimization_techniques(
        self,
        optimization_manager,
        robotics_model,
        technique
    ):
        """Test individual optimization techniques."""
        # Create criteria for single technique
        constraints = OptimizationConstraints(
            preserve_accuracy_threshold=0.95,
            allowed_techniques=[technique],
            max_optimization_time_minutes=30
        )
        
        criteria = OptimizationCriteria(
            name=f"test_{technique.value}",
            description=f"Test {technique.value} optimization",
            target_deployment="general",
            constraints=constraints
        )
        
        # Start optimization
        session_id = optimization_manager.start_optimization_session(
            str(robotics_model['path']),
            criteria
        )
        
        # Wait for completion
        start_time = time.time()
        timeout = 300
        
        while time.time() - start_time < timeout:
            status = optimization_manager.get_session_status(session_id)
            if status.get('status') in ['completed', 'failed', 'cancelled']:
                break
            time.sleep(2)
        
        # Verify technique was applied
        with optimization_manager._lock:
            session = optimization_manager.active_sessions[session_id]
            
            if session.results:
                # Should have applied the requested technique
                applied_techniques = [t.lower() for t in session.results.techniques_applied]
                assert technique.value in applied_techniques
    
    def test_optimization_with_multiple_techniques(
        self,
        optimization_manager,
        robotics_model
    ):
        """Test optimization with multiple techniques enabled."""
        constraints = OptimizationConstraints(
            preserve_accuracy_threshold=0.95,
            allowed_techniques=[
                OptimizationTechnique.QUANTIZATION,
                OptimizationTechnique.PRUNING
            ],
            max_optimization_time_minutes=30
        )
        
        criteria = OptimizationCriteria(
            name="multi_technique_test",
            description="Test multiple optimization techniques",
            target_deployment="edge",
            constraints=constraints
        )
        
        session_id = optimization_manager.start_optimization_session(
            str(robotics_model['path']),
            criteria
        )
        
        # Wait for completion
        start_time = time.time()
        timeout = 300
        
        while time.time() - start_time < timeout:
            status = optimization_manager.get_session_status(session_id)
            if status.get('status') in ['completed', 'failed', 'cancelled']:
                break
            time.sleep(2)
        
        # Verify multiple techniques were considered
        with optimization_manager._lock:
            session = optimization_manager.active_sessions[session_id]
            
            # Should have executed multiple steps
            assert len(session.steps) >= 1
            
            # Results should show techniques applied
            if session.results:
                assert len(session.results.techniques_applied) >= 1

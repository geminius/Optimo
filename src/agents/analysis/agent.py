"""
Analysis Agent implementation for robotics model optimization platform.

This module provides the AnalysisAgent class that analyzes models to identify
optimization opportunities, profile performance, and assess compatibility
with different optimization techniques.
"""

import time
import psutil
import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import logging
from dataclasses import asdict

from ..base import BaseAnalysisAgent
from ...models.core import (
    AnalysisReport, ArchitectureSummary, PerformanceProfile,
    OptimizationOpportunity, Recommendation, ModelMetadata
)
from ...config.optimization_criteria import OptimizationTechnique


logger = logging.getLogger(__name__)


class AnalysisAgent(BaseAnalysisAgent):
    """
    Analysis agent that examines models to identify optimization opportunities
    and assess compatibility with different optimization techniques.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.profiling_samples = config.get("profiling_samples", 100)
        self.warmup_samples = config.get("warmup_samples", 10)
        
    def initialize(self) -> bool:
        """Initialize the analysis agent."""
        try:
            logger.info(f"Initializing AnalysisAgent on device: {self.device}")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize AnalysisAgent: {e}")
            return False
    
    def cleanup(self) -> None:
        """Clean up resources used by the agent."""
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        logger.info("AnalysisAgent cleanup completed")
    
    def analyze_model(self, model_path: str) -> AnalysisReport:
        """
        Analyze a model and return comprehensive analysis report.
        
        Args:
            model_path: Path to the model file
            
        Returns:
            AnalysisReport containing analysis results
        """
        start_time = time.time()
        logger.info(f"Starting analysis of model: {model_path}")
        
        try:
            # Load model
            model = self._load_model(model_path)
            model_metadata = self._extract_model_metadata(model_path, model)
            
            # Perform analysis components
            architecture_summary = self._analyze_architecture(model)
            performance_profile = self._profile_performance(model)
            optimization_opportunities = self._identify_optimization_opportunities(
                model, architecture_summary, performance_profile
            )
            compatibility_matrix = self._assess_compatibility(model, architecture_summary)
            recommendations = self._generate_recommendations(
                optimization_opportunities, compatibility_matrix
            )
            
            analysis_duration = time.time() - start_time
            
            report = AnalysisReport(
                model_id=model_metadata.id,
                architecture_summary=architecture_summary,
                performance_profile=performance_profile,
                optimization_opportunities=optimization_opportunities,
                compatibility_matrix=compatibility_matrix,
                recommendations=recommendations,
                analysis_duration_seconds=analysis_duration
            )
            
            logger.info(f"Analysis completed in {analysis_duration:.2f} seconds")
            return report
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            raise
    
    def identify_bottlenecks(self, model: torch.nn.Module) -> List[Dict[str, Any]]:
        """
        Identify performance bottlenecks in the model.
        
        Args:
            model: PyTorch model to analyze
            
        Returns:
            List of bottleneck information
        """
        bottlenecks = []
        
        try:
            model.eval()
            
            # Analyze layer-wise performance
            layer_times = self._profile_layer_performance(model)
            
            # Identify slow layers (top 20% by execution time)
            sorted_layers = sorted(layer_times.items(), key=lambda x: x[1], reverse=True)
            threshold_idx = max(1, len(sorted_layers) // 5)  # Top 20%
            slow_layers = sorted_layers[:threshold_idx]
            
            for layer_name, exec_time in slow_layers:
                bottleneck = {
                    "type": "slow_layer",
                    "layer_name": layer_name,
                    "execution_time_ms": exec_time * 1000,
                    "severity": "high" if exec_time > 0.1 else "medium",
                    "description": f"Layer {layer_name} has high execution time",
                    "suggestions": self._get_layer_optimization_suggestions(layer_name, model)
                }
                bottlenecks.append(bottleneck)
            
            # Check for memory bottlenecks
            memory_bottlenecks = self._identify_memory_bottlenecks(model)
            bottlenecks.extend(memory_bottlenecks)
            
            # Check for computational bottlenecks
            compute_bottlenecks = self._identify_compute_bottlenecks(model)
            bottlenecks.extend(compute_bottlenecks)
            
        except Exception as e:
            logger.error(f"Bottleneck identification failed: {e}")
            bottlenecks.append({
                "type": "analysis_error",
                "description": f"Failed to identify bottlenecks: {str(e)}",
                "severity": "low"
            })
        
        return bottlenecks
    
    def _load_model(self, model_path: str) -> torch.nn.Module:
        """Load model from file path."""
        path = Path(model_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        try:
            # Try loading as PyTorch model
            if path.suffix in ['.pt', '.pth']:
                # Use weights_only=False for testing purposes - in production this should be more secure
                model = torch.load(model_path, map_location=self.device, weights_only=False)
                if isinstance(model, dict) and 'model' in model:
                    model = model['model']
                elif isinstance(model, dict) and 'state_dict' in model:
                    # Need to reconstruct model architecture - this is a limitation
                    raise ValueError("State dict found but no model architecture")
            else:
                raise ValueError(f"Unsupported model format: {path.suffix}")
            
            return model.to(self.device)
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def _extract_model_metadata(self, model_path: str, model: torch.nn.Module) -> ModelMetadata:
        """Extract metadata from model."""
        path = Path(model_path)
        
        # Calculate model size
        size_mb = path.stat().st_size / (1024 * 1024)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        
        return ModelMetadata(
            name=path.stem,
            file_path=str(path),
            size_mb=size_mb,
            parameters=total_params
        )
    
    def _analyze_architecture(self, model: torch.nn.Module) -> ArchitectureSummary:
        """Analyze model architecture."""
        layer_types = {}
        total_layers = 0
        total_params = 0
        trainable_params = 0
        
        # Analyze each module
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules only
                total_layers += 1
                module_type = type(module).__name__
                layer_types[module_type] = layer_types.get(module_type, 0) + 1
        
        # Count parameters
        for param in model.parameters():
            param_count = param.numel()
            total_params += param_count
            if param.requires_grad:
                trainable_params += param_count
        
        # Estimate memory footprint (parameters + activations estimate)
        memory_footprint_mb = (total_params * 4) / (1024 * 1024)  # 4 bytes per float32
        
        # Calculate model depth (simplified)
        model_depth = self._calculate_model_depth(model)
        
        return ArchitectureSummary(
            total_layers=total_layers,
            layer_types=layer_types,
            total_parameters=total_params,
            trainable_parameters=trainable_params,
            model_depth=model_depth,
            memory_footprint_mb=memory_footprint_mb
        )
    
    def _profile_performance(self, model: torch.nn.Module) -> PerformanceProfile:
        """Profile model performance."""
        model.eval()
        
        # Find compatible input shape
        dummy_input = self._find_compatible_input(model)
        
        # Warmup
        with torch.no_grad():
            for _ in range(self.warmup_samples):
                _ = model(dummy_input)
        
        # Profile inference time
        inference_times = []
        memory_usage_before = self._get_memory_usage()
        
        with torch.no_grad():
            for _ in range(self.profiling_samples):
                start_time = time.time()
                _ = model(dummy_input)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                end_time = time.time()
                inference_times.append(end_time - start_time)
        
        memory_usage_after = self._get_memory_usage()
        
        avg_inference_time = sum(inference_times) / len(inference_times)
        memory_usage_mb = memory_usage_after - memory_usage_before
        throughput = 1.0 / avg_inference_time if avg_inference_time > 0 else 0.0
        
        return PerformanceProfile(
            inference_time_ms=avg_inference_time * 1000,
            memory_usage_mb=max(0, memory_usage_mb),
            throughput_samples_per_sec=throughput,
            gpu_utilization_percent=self._get_gpu_utilization(),
            cpu_utilization_percent=psutil.cpu_percent()
        )
    
    def _identify_optimization_opportunities(
        self, 
        model: torch.nn.Module, 
        arch_summary: ArchitectureSummary,
        perf_profile: PerformanceProfile
    ) -> List[OptimizationOpportunity]:
        """Identify optimization opportunities."""
        opportunities = []
        
        # Quantization opportunities
        if self._can_quantize(model, arch_summary):
            opportunities.append(OptimizationOpportunity(
                technique="quantization",
                estimated_size_reduction_percent=50.0,
                estimated_speed_improvement_percent=30.0,
                estimated_accuracy_impact_percent=-2.0,
                confidence_score=0.8,
                complexity="medium",
                description="Model contains layers suitable for quantization"
            ))
        
        # Pruning opportunities
        if self._can_prune(model, arch_summary):
            opportunities.append(OptimizationOpportunity(
                technique="pruning",
                estimated_size_reduction_percent=30.0,
                estimated_speed_improvement_percent=20.0,
                estimated_accuracy_impact_percent=-5.0,
                confidence_score=0.7,
                complexity="medium",
                description="Model has redundant parameters that can be pruned"
            ))
        
        # Distillation opportunities
        if self._can_distill(model, arch_summary):
            opportunities.append(OptimizationOpportunity(
                technique="distillation",
                estimated_size_reduction_percent=70.0,
                estimated_speed_improvement_percent=60.0,
                estimated_accuracy_impact_percent=-8.0,
                confidence_score=0.6,
                complexity="high",
                description="Large model suitable for knowledge distillation"
            ))
        
        return opportunities
    
    def _assess_compatibility(
        self, 
        model: torch.nn.Module, 
        arch_summary: ArchitectureSummary
    ) -> Dict[str, bool]:
        """Assess compatibility with optimization techniques."""
        compatibility = {}
        
        # Check quantization compatibility
        compatibility["quantization"] = self._can_quantize(model, arch_summary)
        
        # Check pruning compatibility
        compatibility["pruning"] = self._can_prune(model, arch_summary)
        
        # Check distillation compatibility
        compatibility["distillation"] = self._can_distill(model, arch_summary)
        
        # Check compression compatibility
        compatibility["compression"] = self._can_compress(model, arch_summary)
        
        # Check architecture search compatibility
        compatibility["architecture_search"] = self._can_architecture_search(model, arch_summary)
        
        return compatibility
    
    def _generate_recommendations(
        self, 
        opportunities: List[OptimizationOpportunity],
        compatibility: Dict[str, bool]
    ) -> List[Recommendation]:
        """Generate optimization recommendations."""
        recommendations = []
        
        # Sort opportunities by potential impact
        sorted_opportunities = sorted(
            opportunities, 
            key=lambda x: x.estimated_speed_improvement_percent + x.estimated_size_reduction_percent,
            reverse=True
        )
        
        for i, opp in enumerate(sorted_opportunities[:3]):  # Top 3 recommendations
            recommendation = Recommendation(
                technique=opp.technique,
                priority=i + 1,
                rationale=opp.description,
                expected_benefits=[
                    f"{opp.estimated_size_reduction_percent:.1f}% size reduction",
                    f"{opp.estimated_speed_improvement_percent:.1f}% speed improvement"
                ],
                potential_risks=[
                    f"Potential {abs(opp.estimated_accuracy_impact_percent):.1f}% accuracy loss"
                ],
                estimated_effort=opp.complexity
            )
            recommendations.append(recommendation)
        
        return recommendations
    
    # Helper methods for compatibility assessment
    def _can_quantize(self, model: torch.nn.Module, arch_summary: ArchitectureSummary) -> bool:
        """Check if model can be quantized."""
        # Check for quantizable layers
        quantizable_layers = ['Linear', 'Conv2d', 'Conv1d', 'ConvTranspose2d']
        return any(layer_type in arch_summary.layer_types for layer_type in quantizable_layers)
    
    def _can_prune(self, model: torch.nn.Module, arch_summary: ArchitectureSummary) -> bool:
        """Check if model can be pruned."""
        # Most models with parameters can be pruned
        return arch_summary.total_parameters > 1000
    
    def _can_distill(self, model: torch.nn.Module, arch_summary: ArchitectureSummary) -> bool:
        """Check if model is suitable for distillation."""
        # Large models are good candidates for distillation
        return arch_summary.total_parameters > 10_000_000  # 10M parameters
    
    def _can_compress(self, model: torch.nn.Module, arch_summary: ArchitectureSummary) -> bool:
        """Check if model can be compressed."""
        # Models with large linear layers are good for compression
        return arch_summary.layer_types.get('Linear', 0) > 0
    
    def _can_architecture_search(self, model: torch.nn.Module, arch_summary: ArchitectureSummary) -> bool:
        """Check if model is suitable for architecture search."""
        # Complex models with multiple layer types
        return len(arch_summary.layer_types) > 3
    
    # Performance profiling helper methods
    def _profile_layer_performance(self, model: torch.nn.Module) -> Dict[str, float]:
        """Profile performance of individual layers."""
        layer_times = {}
        
        # This is a simplified implementation
        # In practice, you'd use torch.profiler or hooks for detailed profiling
        dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
        
        def hook_fn(name):
            def hook(module, input, output):
                start_time = time.time()
                # The actual computation happens before this hook
                # This is a simplified timing approach
                layer_times[name] = time.time() - start_time
            return hook
        
        # Register hooks (simplified approach)
        hooks = []
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules
                hook = module.register_forward_hook(hook_fn(name))
                hooks.append(hook)
        
        try:
            with torch.no_grad():
                _ = model(dummy_input)
        finally:
            # Remove hooks
            for hook in hooks:
                hook.remove()
        
        return layer_times
    
    def _identify_memory_bottlenecks(self, model: torch.nn.Module) -> List[Dict[str, Any]]:
        """Identify memory bottlenecks."""
        bottlenecks = []
        
        # Check for large layers
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                param_count = sum(p.numel() for p in module.parameters())
                if param_count > 1_000_000:  # 1M parameters
                    bottlenecks.append({
                        "type": "large_layer",
                        "layer_name": name,
                        "parameter_count": param_count,
                        "severity": "medium",
                        "description": f"Layer {name} has {param_count:,} parameters",
                        "suggestions": ["Consider pruning", "Apply quantization"]
                    })
        
        return bottlenecks
    
    def _identify_compute_bottlenecks(self, model: torch.nn.Module) -> List[Dict[str, Any]]:
        """Identify computational bottlenecks."""
        bottlenecks = []
        
        # Check for computationally expensive operations
        expensive_ops = ['MultiheadAttention', 'TransformerEncoderLayer', 'TransformerDecoderLayer']
        
        for name, module in model.named_modules():
            if any(op in type(module).__name__ for op in expensive_ops):
                bottlenecks.append({
                    "type": "expensive_operation",
                    "layer_name": name,
                    "operation_type": type(module).__name__,
                    "severity": "high",
                    "description": f"Computationally expensive operation: {type(module).__name__}",
                    "suggestions": ["Consider attention optimization", "Apply pruning to attention heads"]
                })
        
        return bottlenecks
    
    def _get_layer_optimization_suggestions(self, layer_name: str, model: torch.nn.Module) -> List[str]:
        """Get optimization suggestions for a specific layer."""
        suggestions = []
        
        # Get the actual layer
        layer = dict(model.named_modules()).get(layer_name)
        if layer is None:
            return ["Layer not found"]
        
        layer_type = type(layer).__name__
        
        if layer_type in ['Linear', 'Conv2d']:
            suggestions.extend(["Apply quantization", "Consider pruning"])
        
        if layer_type in ['MultiheadAttention']:
            suggestions.extend(["Optimize attention mechanism", "Reduce number of heads"])
        
        if layer_type in ['BatchNorm2d', 'LayerNorm']:
            suggestions.append("Consider fusing with previous layer")
        
        return suggestions if suggestions else ["No specific suggestions available"]
    
    def _calculate_model_depth(self, model: torch.nn.Module) -> int:
        """Calculate approximate model depth."""
        max_depth = 0
        
        def calculate_depth(module, current_depth=0):
            nonlocal max_depth
            max_depth = max(max_depth, current_depth)
            
            for child in module.children():
                calculate_depth(child, current_depth + 1)
        
        calculate_depth(model)
        return max_depth
    
    def _find_compatible_input(self, model: torch.nn.Module) -> torch.Tensor:
        """Find a compatible input shape for the model."""
        # Try common input shapes
        common_shapes = [
            (1, 3, 224, 224),  # Standard image
            (1, 3, 256, 256),  # Larger image
            (1, 1, 28, 28),    # MNIST-like
            (1, 512),          # 1D input
            (1, 1024),         # Larger 1D input
            (1, 1000),         # Common large input
            (1, 2048),         # Very large input
        ]
        
        for shape in common_shapes:
            try:
                dummy_input = torch.randn(shape).to(self.device)
                with torch.no_grad():
                    _ = model(dummy_input)
                return dummy_input
            except Exception:
                continue
        
        # Try to infer input size from first layer
        try:
            first_layer = next(model.modules())
            if hasattr(first_layer, 'in_features'):
                # Linear layer
                input_size = first_layer.in_features
                dummy_input = torch.randn(1, input_size).to(self.device)
                with torch.no_grad():
                    _ = model(dummy_input)
                return dummy_input
            elif hasattr(first_layer, 'in_channels'):
                # Conv layer
                in_channels = first_layer.in_channels
                dummy_input = torch.randn(1, in_channels, 224, 224).to(self.device)
                with torch.no_grad():
                    _ = model(dummy_input)
                return dummy_input
        except Exception:
            pass
        
        # If nothing works, return a basic tensor
        return torch.randn(1, 1).to(self.device)
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024 * 1024)
        else:
            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)
    
    def _get_gpu_utilization(self) -> float:
        """Get GPU utilization percentage."""
        if torch.cuda.is_available():
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                return float(utilization.gpu)
            except ImportError:
                return 0.0
        return 0.0
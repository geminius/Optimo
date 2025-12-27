"""
Evaluation Agent implementation for robotics model optimization platform.

This module provides the EvaluationAgent class that evaluates optimized models
against benchmarks, compares performance with original models, and generates
comprehensive evaluation reports with optional LLM-based validation.
"""

import time
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable
import logging
from dataclasses import asdict
from datetime import datetime
import psutil
import traceback

from ..base import BaseEvaluationAgent
from ...models.core import (
    EvaluationReport, BenchmarkResult, PerformanceMetrics,
    ComparisonResult, ValidationStatus
)
from ...services.llm_service import llm_service, ValidationRequest
from ...utils.exceptions import EvaluationError, LLMValidationError


logger = logging.getLogger(__name__)


class EvaluationAgent(BaseEvaluationAgent):
    """
    Evaluation agent that assesses optimized models through comprehensive testing,
    benchmarking, and comparison with original models.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.benchmark_samples = config.get("benchmark_samples", 100)
        self.warmup_samples = config.get("warmup_samples", 10)
        self.accuracy_threshold = config.get("accuracy_threshold", 0.95)  # 95% of original accuracy
        self.performance_threshold = config.get("performance_threshold", 1.1)  # 10% improvement
        self.timeout_seconds = config.get("timeout_seconds", 300)  # 5 minutes
        
        # LLM validation settings
        self.llm_validation_enabled = config.get("llm_validation_enabled", True)
        self.llm_confidence_threshold = config.get("llm_confidence_threshold", 0.8)
        
        # Built-in benchmark functions
        self._benchmark_registry = {
            "inference_speed": self._benchmark_inference_speed,
            "memory_usage": self._benchmark_memory_usage,
            "throughput": self._benchmark_throughput,
            "accuracy": self._benchmark_accuracy,
            "model_size": self._benchmark_model_size,
            "flops": self._benchmark_flops,
            "energy_efficiency": self._benchmark_energy_efficiency
        }
        
    def initialize(self) -> bool:
        """Initialize the evaluation agent."""
        try:
            logger.info(f"Initializing EvaluationAgent on device: {self.device}")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize EvaluationAgent: {e}")
            return False
    
    def cleanup(self) -> None:
        """Clean up resources used by the agent."""
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        logger.info("EvaluationAgent cleanup completed")
    
    async def evaluate_model(self, model: torch.nn.Module, 
                      benchmarks: List[Dict[str, Any]]) -> EvaluationReport:
        """
        Evaluate model performance against specified benchmarks.
        
        Args:
            model: PyTorch model to evaluate
            benchmarks: List of benchmark configurations
            
        Returns:
            EvaluationReport containing evaluation results
        """
        start_time = time.time()
        logger.info(f"Starting model evaluation with {len(benchmarks)} benchmarks")
        
        try:
            model.eval()
            
            # Run benchmarks
            benchmark_results = []
            performance_metrics = PerformanceMetrics()
            validation_errors = []
            
            for benchmark_config in benchmarks:
                try:
                    result = self._run_benchmark(model, benchmark_config)
                    benchmark_results.append(result)
                    
                    # Update performance metrics
                    self._update_performance_metrics(performance_metrics, result)
                    
                except Exception as e:
                    logger.error(f"Benchmark {benchmark_config.get('name', 'unknown')} failed: {e}")
                    validation_errors.append(f"Benchmark failed: {str(e)}")
            
            # Determine validation status
            validation_status = self._determine_validation_status(
                benchmark_results, validation_errors
            )
            
            # Generate recommendations (including LLM-based if enabled)
            recommendations = await self._generate_evaluation_recommendations_async(
                benchmark_results, performance_metrics, validation_errors
            )
            
            # Perform LLM validation if enabled
            llm_validation_result = None
            if self.llm_validation_enabled and llm_service.is_available():
                try:
                    llm_validation_result = await self._perform_llm_validation(
                        performance_metrics, benchmark_results
                    )
                    
                    # Incorporate LLM recommendations
                    if llm_validation_result and llm_validation_result.recommendations:
                        recommendations.extend(llm_validation_result.recommendations)
                    
                    # Update validation status based on LLM assessment
                    if (llm_validation_result and 
                        not llm_validation_result.is_valid and 
                        llm_validation_result.confidence_score >= self.llm_confidence_threshold):
                        validation_status = ValidationStatus.FAILED
                        validation_errors.extend(llm_validation_result.errors)
                        
                except Exception as e:
                    logger.warning(f"LLM validation failed: {e}")
                    recommendations.append(f"LLM validation unavailable: {str(e)}")
            
            evaluation_duration = time.time() - start_time
            
            report = EvaluationReport(
                model_id="unknown",  # Will be set by caller
                benchmarks=benchmark_results,
                performance_metrics=performance_metrics,
                validation_status=validation_status,
                validation_errors=validation_errors,
                recommendations=recommendations,
                evaluation_duration_seconds=evaluation_duration
            )
            
            logger.info(f"Model evaluation completed in {evaluation_duration:.2f} seconds")
            return report
            
        except Exception as e:
            logger.error(f"Model evaluation failed: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            
            # Return failed evaluation report
            return EvaluationReport(
                model_id="unknown",
                validation_status=ValidationStatus.FAILED,
                validation_errors=[f"Evaluation failed: {str(e)}"],
                evaluation_duration_seconds=time.time() - start_time
            )
    
    async def compare_models(self, original: torch.nn.Module, 
                      optimized: torch.nn.Module) -> ComparisonResult:
        """
        Compare performance between original and optimized models.
        
        Args:
            original: Original model
            optimized: Optimized model
            
        Returns:
            ComparisonResult containing comparison analysis
        """
        logger.info("Starting model comparison")
        
        try:
            # Evaluate both models with standard benchmarks
            standard_benchmarks = self._get_standard_benchmarks()
            
            original_report = await self.evaluate_model(original, standard_benchmarks)
            optimized_report = await self.evaluate_model(optimized, standard_benchmarks)
            
            # Calculate improvements and regressions
            improvements = {}
            regressions = {}
            
            original_metrics = original_report.performance_metrics
            optimized_metrics = optimized_report.performance_metrics
            
            # Compare key metrics
            metric_comparisons = {
                "inference_time_ms": "lower_is_better",
                "memory_usage_mb": "lower_is_better", 
                "model_size_mb": "lower_is_better",
                "throughput_samples_per_sec": "higher_is_better",
                "accuracy": "higher_is_better"
            }
            
            for metric, direction in metric_comparisons.items():
                original_value = getattr(original_metrics, metric, None)
                optimized_value = getattr(optimized_metrics, metric, None)
                
                if original_value is not None and optimized_value is not None and original_value > 0:
                    if direction == "lower_is_better":
                        improvement_pct = ((original_value - optimized_value) / original_value) * 100
                    else:
                        improvement_pct = ((optimized_value - original_value) / original_value) * 100
                    
                    if improvement_pct > 0:
                        improvements[metric] = improvement_pct
                    else:
                        regressions[metric] = abs(improvement_pct)
            
            # Calculate overall score
            overall_score = self._calculate_overall_score(improvements, regressions)
            
            # Generate recommendation
            recommendation = self._generate_comparison_recommendation(
                improvements, regressions, overall_score
            )
            
            comparison_result = ComparisonResult(
                original_metrics=original_metrics,
                optimized_metrics=optimized_metrics,
                improvements=improvements,
                regressions=regressions,
                overall_score=overall_score,
                recommendation=recommendation
            )
            
            logger.info(f"Model comparison completed. Overall score: {overall_score:.2f}")
            return comparison_result
            
        except Exception as e:
            logger.error(f"Model comparison failed: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            
            # Return empty comparison result with error
            return ComparisonResult(
                original_metrics=PerformanceMetrics(),
                optimized_metrics=PerformanceMetrics(),
                recommendation=f"Comparison failed: {str(e)}"
            )
    
    async def validate_performance(self, model: torch.nn.Module, 
                           thresholds: Dict[str, float]) -> Dict[str, Any]:
        """
        Validate model performance against specified thresholds.
        
        Args:
            model: Model to validate
            thresholds: Performance thresholds to check against
            
        Returns:
            Validation result dictionary
        """
        logger.info("Starting performance validation")
        
        try:
            # Run evaluation with standard benchmarks
            benchmarks = self._get_standard_benchmarks()
            report = await self.evaluate_model(model, benchmarks)
            
            validation_results = {
                "passed": True,
                "failures": [],
                "warnings": [],
                "metrics": asdict(report.performance_metrics)
            }
            
            # Check each threshold
            for metric_name, threshold_value in thresholds.items():
                actual_value = getattr(report.performance_metrics, metric_name, None)
                
                if actual_value is None:
                    validation_results["warnings"].append(
                        f"Metric {metric_name} not available for validation"
                    )
                    continue
                
                # Determine if threshold is met (depends on metric type)
                threshold_met = self._check_threshold(metric_name, actual_value, threshold_value)
                
                if not threshold_met:
                    validation_results["passed"] = False
                    validation_results["failures"].append(
                        f"Metric {metric_name} ({actual_value:.4f}) failed threshold ({threshold_value:.4f})"
                    )
            
            return validation_results
            
        except Exception as e:
            logger.error(f"Performance validation failed: {e}")
            return {
                "passed": False,
                "failures": [f"Validation error: {str(e)}"],
                "warnings": [],
                "metrics": {}
            }
    
    def _run_benchmark(self, model: torch.nn.Module, 
                      benchmark_config: Dict[str, Any]) -> BenchmarkResult:
        """Run a single benchmark on the model."""
        benchmark_name = benchmark_config.get("name", "unknown")
        benchmark_type = benchmark_config.get("type", benchmark_name)
        
        logger.debug(f"Running benchmark: {benchmark_name}")
        
        if benchmark_type not in self._benchmark_registry:
            raise ValueError(f"Unknown benchmark type: {benchmark_type}")
        
        start_time = time.time()
        
        try:
            benchmark_func = self._benchmark_registry[benchmark_type]
            score, unit, higher_is_better, metadata = benchmark_func(model, benchmark_config)
            
            execution_time = time.time() - start_time
            
            return BenchmarkResult(
                benchmark_name=benchmark_name,
                score=score,
                unit=unit,
                higher_is_better=higher_is_better,
                execution_time_seconds=execution_time,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"Benchmark {benchmark_name} execution failed: {e}")
            raise
    
    def _benchmark_inference_speed(self, model: torch.nn.Module, 
                                  config: Dict[str, Any]) -> Tuple[float, str, bool, Dict[str, Any]]:
        """Benchmark model inference speed."""
        model.eval()
        
        # Get compatible input
        dummy_input = self._get_compatible_input(model, config)
        
        # Warmup
        with torch.no_grad():
            for _ in range(self.warmup_samples):
                _ = model(dummy_input)
        
        # Measure inference time
        inference_times = []
        with torch.no_grad():
            for _ in range(self.benchmark_samples):
                start_time = time.time()
                _ = model(dummy_input)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                end_time = time.time()
                inference_times.append(end_time - start_time)
        
        avg_time_ms = (sum(inference_times) / len(inference_times)) * 1000
        
        metadata = {
            "samples": len(inference_times),
            "std_dev_ms": np.std(inference_times) * 1000,
            "min_time_ms": min(inference_times) * 1000,
            "max_time_ms": max(inference_times) * 1000
        }
        
        return avg_time_ms, "ms", False, metadata  # Lower is better
    
    def _benchmark_memory_usage(self, model: torch.nn.Module, 
                               config: Dict[str, Any]) -> Tuple[float, str, bool, Dict[str, Any]]:
        """Benchmark model memory usage."""
        model.eval()
        
        # Clear cache and measure baseline
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        baseline_memory = self._get_memory_usage()
        
        # Load model and measure memory
        dummy_input = self._get_compatible_input(model, config)
        
        with torch.no_grad():
            _ = model(dummy_input)
        
        peak_memory = self._get_memory_usage()
        memory_usage_mb = peak_memory - baseline_memory
        
        metadata = {
            "baseline_memory_mb": baseline_memory,
            "peak_memory_mb": peak_memory,
            "device": str(self.device)
        }
        
        return max(0.0, memory_usage_mb), "MB", False, metadata  # Lower is better
    
    def _benchmark_throughput(self, model: torch.nn.Module, 
                             config: Dict[str, Any]) -> Tuple[float, str, bool, Dict[str, Any]]:
        """Benchmark model throughput."""
        model.eval()
        
        dummy_input = self._get_compatible_input(model, config)
        batch_size = dummy_input.shape[0] if len(dummy_input.shape) > 0 else 1
        
        # Warmup
        with torch.no_grad():
            for _ in range(self.warmup_samples):
                _ = model(dummy_input)
        
        # Measure throughput
        start_time = time.time()
        with torch.no_grad():
            for _ in range(self.benchmark_samples):
                _ = model(dummy_input)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        total_time = time.time() - start_time
        throughput = (self.benchmark_samples * batch_size) / total_time
        
        metadata = {
            "batch_size": batch_size,
            "total_samples": self.benchmark_samples * batch_size,
            "total_time_seconds": total_time
        }
        
        return throughput, "samples/sec", True, metadata  # Higher is better
    
    def _benchmark_accuracy(self, model: torch.nn.Module, 
                           config: Dict[str, Any]) -> Tuple[float, str, bool, Dict[str, Any]]:
        """Benchmark model accuracy (simplified implementation)."""
        model.eval()
        
        # This is a simplified accuracy benchmark
        # In practice, you would use actual test datasets
        dummy_input = self._get_compatible_input(model, config)
        
        try:
            with torch.no_grad():
                output = model(dummy_input)
                
            # Simulate accuracy based on output consistency
            # Run multiple times and check consistency
            outputs = []
            with torch.no_grad():
                for _ in range(10):
                    out = model(dummy_input)
                    outputs.append(out)
            
            # Calculate consistency as a proxy for accuracy
            if len(outputs) > 1:
                output_tensor = torch.stack(outputs)
                std_dev = torch.std(output_tensor, dim=0).mean().item()
                # Convert to accuracy-like metric (lower std_dev = higher "accuracy")
                consistency_score = max(0, 1.0 - std_dev)
            else:
                consistency_score = 1.0
            
            metadata = {
                "method": "consistency_based",
                "output_shape": list(output.shape),
                "std_deviation": std_dev if len(outputs) > 1 else 0.0
            }
            
            return consistency_score, "score", True, metadata  # Higher is better
            
        except Exception as e:
            logger.warning(f"Accuracy benchmark failed: {e}")
            return 0.0, "score", True, {"error": str(e)}
    
    def _benchmark_model_size(self, model: torch.nn.Module, 
                             config: Dict[str, Any]) -> Tuple[float, str, bool, Dict[str, Any]]:
        """Benchmark model size."""
        # Calculate model size in MB
        param_size = 0
        buffer_size = 0
        
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        total_size_mb = (param_size + buffer_size) / (1024 * 1024)
        
        metadata = {
            "parameter_size_mb": param_size / (1024 * 1024),
            "buffer_size_mb": buffer_size / (1024 * 1024),
            "total_parameters": sum(p.numel() for p in model.parameters())
        }
        
        return total_size_mb, "MB", False, metadata  # Lower is better
    
    def _benchmark_flops(self, model: torch.nn.Module, 
                        config: Dict[str, Any]) -> Tuple[float, str, bool, Dict[str, Any]]:
        """Benchmark model FLOPs (simplified estimation)."""
        # This is a simplified FLOP estimation
        # In practice, you would use tools like fvcore or ptflops
        
        total_flops = 0
        dummy_input = self._get_compatible_input(model, config)
        
        # Estimate FLOPs based on layer types
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                # Linear layer: input_features * output_features
                flops = module.in_features * module.out_features
                if module.bias is not None:
                    flops += module.out_features
                total_flops += flops
            
            elif isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                # Convolution: output_elements * kernel_size * input_channels
                if hasattr(module, 'weight') and module.weight is not None:
                    kernel_flops = np.prod(module.kernel_size) * module.in_channels
                    # Estimate output size (simplified)
                    output_elements = 1000  # Placeholder
                    total_flops += kernel_flops * output_elements
        
        # Convert to MFLOPs
        mflops = total_flops / 1_000_000
        
        metadata = {
            "estimation_method": "simplified",
            "total_flops": total_flops,
            "input_shape": list(dummy_input.shape)
        }
        
        return mflops, "MFLOPs", False, metadata  # Lower is better for efficiency
    
    def _benchmark_energy_efficiency(self, model: torch.nn.Module, 
                                   config: Dict[str, Any]) -> Tuple[float, str, bool, Dict[str, Any]]:
        """Benchmark energy efficiency (simplified implementation)."""
        # This is a simplified energy estimation based on computation
        model.eval()
        
        dummy_input = self._get_compatible_input(model, config)
        
        # Measure CPU usage during inference
        cpu_before = psutil.cpu_percent(interval=None)
        
        start_time = time.time()
        with torch.no_grad():
            for _ in range(10):  # Smaller sample for energy measurement
                _ = model(dummy_input)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        end_time = time.time()
        cpu_after = psutil.cpu_percent(interval=None)
        
        # Estimate energy based on time and CPU usage
        duration = end_time - start_time
        avg_cpu = (cpu_before + cpu_after) / 2
        
        # Simplified energy estimation (arbitrary units)
        energy_estimate = duration * avg_cpu * 0.1  # Simplified formula
        
        metadata = {
            "duration_seconds": duration,
            "cpu_usage_percent": avg_cpu,
            "estimation_method": "cpu_time_based"
        }
        
        return energy_estimate, "energy_units", False, metadata  # Lower is better
    
    def _get_compatible_input(self, model: torch.nn.Module, 
                             config: Dict[str, Any]) -> torch.Tensor:
        """Get compatible input tensor for the model."""
        # Check if input shape is specified in config
        if "input_shape" in config:
            shape = config["input_shape"]
            return torch.randn(shape).to(self.device)
        
        # Try to infer input shape from model
        return self._find_compatible_input(model)
    
    def _find_compatible_input(self, model: torch.nn.Module) -> torch.Tensor:
        """Find a compatible input shape for the model."""
        from ...utils.model_utils import find_compatible_input
        return find_compatible_input(model, self.device)
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        from ...utils.model_utils import get_memory_usage
        return get_memory_usage()
    
    def _update_performance_metrics(self, metrics: PerformanceMetrics, 
                                   benchmark_result: BenchmarkResult) -> None:
        """Update performance metrics with benchmark result."""
        benchmark_name = benchmark_result.benchmark_name
        score = benchmark_result.score
        
        if benchmark_name == "inference_speed":
            metrics.inference_time_ms = score
        elif benchmark_name == "memory_usage":
            metrics.memory_usage_mb = score
        elif benchmark_name == "throughput":
            metrics.throughput_samples_per_sec = score
        elif benchmark_name == "accuracy":
            metrics.accuracy = score
        elif benchmark_name == "model_size":
            metrics.model_size_mb = score
        elif benchmark_name == "flops":
            metrics.flops = int(score * 1_000_000)  # Convert MFLOPs to FLOPs
        elif benchmark_name == "energy_efficiency":
            metrics.energy_consumption_watts = score
        else:
            # Store in custom metrics
            metrics.custom_metrics[benchmark_name] = score
    
    def _determine_validation_status(self, benchmark_results: List[BenchmarkResult],
                                   validation_errors: List[str]) -> ValidationStatus:
        """Determine overall validation status."""
        if validation_errors:
            return ValidationStatus.FAILED
        
        if not benchmark_results:
            return ValidationStatus.FAILED
        
        # Check if any critical benchmarks failed
        critical_benchmarks = ["inference_speed", "accuracy"]
        for result in benchmark_results:
            if result.benchmark_name in critical_benchmarks and result.score <= 0:
                return ValidationStatus.WARNING
        
        return ValidationStatus.PASSED
    
    def _generate_evaluation_recommendations(self, benchmark_results: List[BenchmarkResult],
                                           performance_metrics: PerformanceMetrics) -> List[str]:
        """Generate recommendations based on evaluation results."""
        recommendations = []
        
        # Analyze benchmark results
        for result in benchmark_results:
            if result.benchmark_name == "inference_speed" and result.score > 100:  # > 100ms
                recommendations.append("Consider optimization to reduce inference time")
            
            elif result.benchmark_name == "memory_usage" and result.score > 1000:  # > 1GB
                recommendations.append("High memory usage detected, consider model compression")
            
            elif result.benchmark_name == "accuracy" and result.score < 0.8:  # < 80%
                recommendations.append("Low accuracy detected, review optimization parameters")
            
            elif result.benchmark_name == "model_size" and result.score > 500:  # > 500MB
                recommendations.append("Large model size, consider quantization or pruning")
        
        # General recommendations
        if performance_metrics.throughput_samples_per_sec < 10:
            recommendations.append("Low throughput detected, consider performance optimization")
        
        if not recommendations:
            recommendations.append("Model performance is within acceptable ranges")
        
        return recommendations
    
    def _calculate_overall_score(self, improvements: Dict[str, float], 
                               regressions: Dict[str, float]) -> float:
        """Calculate overall comparison score."""
        # Weight different metrics
        weights = {
            "inference_time_ms": 0.3,
            "memory_usage_mb": 0.2,
            "model_size_mb": 0.2,
            "throughput_samples_per_sec": 0.2,
            "accuracy": 0.1
        }
        
        total_score = 0.0
        total_weight = 0.0
        
        # Add improvements (positive contribution)
        for metric, improvement in improvements.items():
            weight = weights.get(metric, 0.1)
            total_score += improvement * weight
            total_weight += weight
        
        # Subtract regressions (negative contribution)
        for metric, regression in regressions.items():
            weight = weights.get(metric, 0.1)
            total_score -= regression * weight
            total_weight += weight
        
        # Normalize score
        if total_weight > 0:
            return total_score / total_weight
        else:
            return 0.0
    
    def _generate_comparison_recommendation(self, improvements: Dict[str, float],
                                         regressions: Dict[str, float],
                                         overall_score: float) -> str:
        """Generate recommendation based on comparison results."""
        if overall_score > 10:
            return "Optimization shows significant improvements. Recommended for deployment."
        elif overall_score > 5:
            return "Optimization shows moderate improvements. Consider for deployment."
        elif overall_score > 0:
            return "Optimization shows minor improvements. Evaluate trade-offs carefully."
        elif overall_score > -5:
            return "Optimization shows mixed results. Review optimization parameters."
        else:
            return "Optimization shows significant regressions. Not recommended for deployment."
    
    def _check_threshold(self, metric_name: str, actual_value: float, 
                        threshold_value: float) -> bool:
        """Check if metric meets threshold based on metric type."""
        # Define which metrics should be higher vs lower
        higher_is_better_metrics = {
            "accuracy", "throughput_samples_per_sec"
        }
        
        lower_is_better_metrics = {
            "inference_time_ms", "memory_usage_mb", "model_size_mb", 
            "energy_consumption_watts"
        }
        
        if metric_name in higher_is_better_metrics:
            return actual_value >= threshold_value
        elif metric_name in lower_is_better_metrics:
            return actual_value <= threshold_value
        else:
            # Default: assume higher is better
            return actual_value >= threshold_value
    
    def _get_standard_benchmarks(self) -> List[Dict[str, Any]]:
        """Get standard benchmark configurations."""
        return [
            {"name": "inference_speed", "type": "inference_speed"},
            {"name": "memory_usage", "type": "memory_usage"},
            {"name": "throughput", "type": "throughput"},
            {"name": "accuracy", "type": "accuracy"},
            {"name": "model_size", "type": "model_size"},
            {"name": "flops", "type": "flops"},
            {"name": "energy_efficiency", "type": "energy_efficiency"}
        ]
    
    async def _generate_evaluation_recommendations_async(
        self, 
        benchmark_results: List[BenchmarkResult],
        performance_metrics: PerformanceMetrics,
        validation_errors: List[str]
    ) -> List[str]:
        """Generate recommendations including LLM-based suggestions."""
        # Start with standard recommendations
        recommendations = self._generate_evaluation_recommendations(
            benchmark_results, performance_metrics
        )
        
        # Add LLM-generated recommendations if available
        if self.llm_validation_enabled and llm_service.is_available():
            try:
                metrics_dict = asdict(performance_metrics)
                benchmark_dict = {
                    result.benchmark_name: {
                        "score": result.score,
                        "unit": result.unit,
                        "execution_time": result.execution_time_seconds
                    }
                    for result in benchmark_results
                }
                
                llm_recommendations = await llm_service.generate_recommendations(
                    model_metrics=metrics_dict,
                    optimization_config=benchmark_dict,
                    context={
                        "validation_errors": validation_errors,
                        "platform": "robotics_optimization"
                    }
                )
                
                # Add LLM recommendations with prefix
                for rec in llm_recommendations:
                    recommendations.append(f"AI Insight: {rec}")
                    
            except Exception as e:
                logger.warning(f"Failed to generate LLM recommendations: {e}")
        
        return recommendations
    
    async def _perform_llm_validation(
        self,
        performance_metrics: PerformanceMetrics,
        benchmark_results: List[BenchmarkResult]
    ) -> Optional[Any]:
        """Perform LLM-based validation of optimization results."""
        try:
            # Prepare validation request
            metrics_dict = asdict(performance_metrics)
            
            # Create optimization config from benchmark results
            optimization_config = {
                "benchmarks_run": len(benchmark_results),
                "benchmark_results": {
                    result.benchmark_name: {
                        "score": result.score,
                        "unit": result.unit,
                        "higher_is_better": result.higher_is_better,
                        "execution_time_seconds": result.execution_time_seconds
                    }
                    for result in benchmark_results
                }
            }
            
            # Add robotics-specific context
            context = {
                "platform": "robotics_model_optimization",
                "deployment_target": "edge_devices",
                "critical_constraints": {
                    "max_inference_time_ms": 100,
                    "min_accuracy_threshold": 0.95,
                    "max_memory_usage_mb": 1000
                },
                "evaluation_timestamp": datetime.now().isoformat()
            }
            
            validation_request = ValidationRequest(
                validation_type="robotics_optimization_evaluation",
                model_metrics=metrics_dict,
                optimization_config=optimization_config,
                context=context
            )
            
            # Perform LLM validation
            validation_result = await llm_service.validate_optimization_result(
                validation_request
            )
            
            logger.info(
                f"LLM validation completed - Valid: {validation_result.is_valid}, "
                f"Confidence: {validation_result.confidence_score:.2f}",
                extra={
                    "component": "EvaluationAgent",
                    "llm_validation": True,
                    "confidence": validation_result.confidence_score
                }
            )
            
            return validation_result
            
        except LLMValidationError as e:
            logger.error(f"LLM validation error: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during LLM validation: {e}")
            raise LLMValidationError(
                f"LLM validation failed: {str(e)}",
                validation_type="robotics_optimization_evaluation"
            )
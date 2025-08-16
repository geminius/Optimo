"""
Knowledge distillation optimization agent for model compression using teacher-student training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Optional, Tuple, Callable
import logging
import time
import copy
from dataclasses import dataclass
from enum import Enum
import math

from ..base import BaseOptimizationAgent, ImpactEstimate, ValidationResult, OptimizedModel


class DistillationType(Enum):
    """Supported distillation types."""
    RESPONSE = "response"  # Standard response-based distillation
    FEATURE = "feature"    # Feature-based distillation
    ATTENTION = "attention"  # Attention-based distillation
    PROGRESSIVE = "progressive"  # Progressive distillation


class StudentArchitecture(Enum):
    """Supported student architectures."""
    SMALLER_SAME = "smaller_same"  # Same architecture, fewer layers/units
    EFFICIENT = "efficient"        # More efficient architecture (e.g., MobileNet-style)
    CUSTOM = "custom"             # Custom provided architecture


@dataclass
class DistillationConfig:
    """Configuration for knowledge distillation."""
    distillation_type: DistillationType
    student_architecture: StudentArchitecture
    temperature: float = 4.0  # Softmax temperature for distillation
    alpha: float = 0.7  # Weight for distillation loss
    beta: float = 0.3   # Weight for student loss
    learning_rate: float = 1e-4
    num_epochs: int = 10
    batch_size: int = 32
    compression_ratio: float = 0.5  # Target model size reduction
    feature_layers: List[str] = None  # Layers for feature distillation
    preserve_accuracy: float = 0.95  # Minimum accuracy to preserve
    
    def __post_init__(self):
        if self.feature_layers is None:
            self.feature_layers = []
        
        # Validate parameters
        if not 0.0 < self.alpha < 1.0:
            raise ValueError(f"Alpha must be between 0 and 1, got {self.alpha}")
        if not 0.0 < self.beta < 1.0:
            raise ValueError(f"Beta must be between 0 and 1, got {self.beta}")
        if abs(self.alpha + self.beta - 1.0) > 1e-6:
            raise ValueError(f"Alpha + Beta must equal 1.0, got {self.alpha + self.beta}")


class DistillationLoss(nn.Module):
    """Custom loss function for knowledge distillation."""
    
    def __init__(self, temperature: float = 4.0, alpha: float = 0.7):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.beta = 1.0 - alpha
        self.kl_div = nn.KLDivLoss(reduction='batchmean')
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor, 
                targets: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute distillation loss.
        
        Args:
            student_logits: Logits from student model
            teacher_logits: Logits from teacher model
            targets: Ground truth labels (optional)
        """
        # Distillation loss (KL divergence between soft targets)
        student_soft = F.log_softmax(student_logits / self.temperature, dim=1)
        teacher_soft = F.softmax(teacher_logits / self.temperature, dim=1)
        
        distillation_loss = self.kl_div(student_soft, teacher_soft) * (self.temperature ** 2)
        
        # Student loss (cross-entropy with hard targets)
        if targets is not None:
            student_loss = self.ce_loss(student_logits, targets)
            total_loss = self.alpha * distillation_loss + self.beta * student_loss
        else:
            total_loss = distillation_loss
        
        return total_loss


class FeatureDistillationLoss(nn.Module):
    """Loss function for feature-based distillation."""
    
    def __init__(self, feature_weight: float = 1.0):
        super().__init__()
        self.feature_weight = feature_weight
        self.mse_loss = nn.MSELoss()
    
    def forward(self, student_features: torch.Tensor, teacher_features: torch.Tensor) -> torch.Tensor:
        """Compute feature distillation loss."""
        # Align feature dimensions if necessary
        if student_features.shape != teacher_features.shape:
            # Simple alignment - could be more sophisticated
            if student_features.numel() != teacher_features.numel():
                # Adaptive pooling to match sizes
                if len(teacher_features.shape) == 4:  # Conv features
                    teacher_features = F.adaptive_avg_pool2d(
                        teacher_features, student_features.shape[-2:]
                    )
                elif len(teacher_features.shape) == 2:  # Linear features
                    # Use linear projection to match dimensions
                    if not hasattr(self, 'projection'):
                        self.projection = nn.Linear(
                            teacher_features.shape[1], student_features.shape[1]
                        ).to(teacher_features.device)
                    teacher_features = self.projection(teacher_features)
        
        return self.mse_loss(student_features, teacher_features) * self.feature_weight


class DistillationAgent(BaseOptimizationAgent):
    """
    Optimization agent that implements knowledge distillation to create smaller,
    more efficient models while preserving performance.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Default distillation settings
        self.default_config = DistillationConfig(
            distillation_type=DistillationType.RESPONSE,
            student_architecture=StudentArchitecture.SMALLER_SAME,
            compression_ratio=config.get('compression_ratio', 0.5)
        )
        
        # Cache for student architectures
        self._student_cache = {}
        
    def initialize(self) -> bool:
        """Initialize the distillation agent."""
        try:
            # Test basic PyTorch functionality
            test_teacher = nn.Linear(10, 5)
            test_student = nn.Linear(10, 5)
            test_input = torch.randn(2, 10)
            
            with torch.no_grad():
                teacher_out = test_teacher(test_input)
                student_out = test_student(test_input)
            
            # Test distillation loss
            loss_fn = DistillationLoss()
            loss = loss_fn(student_out, teacher_out)
            
            self.logger.info("Knowledge distillation functionality verified")
            self.logger.info("DistillationAgent initialized successfully")
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize DistillationAgent: {e}")
            return False
    
    def cleanup(self) -> None:
        """Clean up resources used by the agent."""
        # Clear student architecture cache
        self._student_cache.clear()
        self.logger.info("DistillationAgent cleanup completed")
    
    def can_optimize(self, model: torch.nn.Module) -> bool:
        """Check if this agent can optimize the given model."""
        # Check if model has sufficient complexity for distillation
        param_count = sum(p.numel() for p in model.parameters())
        if param_count < 1_000_000:  # Less than 1M parameters
            self.logger.warning(f"Model too small for effective distillation ({param_count:,} parameters)")
            return False
        
        # Check if model has appropriate output layer for distillation
        has_classifier = False
        for module in model.modules():
            if isinstance(module, nn.Linear):
                # Check if this could be a classification layer
                if module.out_features > 1:
                    has_classifier = True
                    break
        
        if not has_classifier:
            self.logger.warning("No suitable classification layer found for distillation")
            return False
        
        return True
    
    def estimate_impact(self, model: torch.nn.Module) -> ImpactEstimate:
        """Estimate the impact of knowledge distillation on the model."""
        # Count total parameters
        total_params = sum(p.numel() for p in model.parameters())
        
        # Estimate compression based on default settings
        compression_ratio = self.default_config.compression_ratio
        size_reduction = compression_ratio
        
        # Performance estimates (distillation typically preserves most performance)
        performance_improvement = 0.1  # Slight improvement due to regularization effect
        speed_improvement = compression_ratio * 0.8  # Speed improvement proportional to size reduction
        
        # Confidence based on model characteristics
        confidence = 0.8 if total_params > 10_000_000 else 0.6
        
        # Estimated time based on training requirements
        estimated_time = max(15, int(total_params / 1_000_000 * 5))  # 5 minutes per million params
        
        return ImpactEstimate(
            performance_improvement=performance_improvement,
            size_reduction=size_reduction,
            speed_improvement=speed_improvement,
            confidence=confidence,
            estimated_time_minutes=estimated_time
        )
    
    def optimize(self, model: torch.nn.Module, config: Dict[str, Any]) -> OptimizedModel:
        """Execute knowledge distillation optimization on the model."""
        start_time = time.time()
        
        # Parse configuration
        distill_config = self._parse_config(config)
        
        # Teacher model (original model)
        teacher_model = copy.deepcopy(model)
        teacher_model.eval()
        
        # Create student model
        student_model = self._create_student_model(teacher_model, distill_config)
        
        # Track optimization metadata
        optimization_metadata = {
            "distillation_type": distill_config.distillation_type.value,
            "student_architecture": distill_config.student_architecture.value,
            "temperature": distill_config.temperature,
            "alpha": distill_config.alpha,
            "beta": distill_config.beta,
            "compression_ratio": distill_config.compression_ratio,
            "num_epochs": distill_config.num_epochs
        }
        
        try:
            # Execute distillation training
            if distill_config.distillation_type == DistillationType.RESPONSE:
                student_model = self._distill_response(teacher_model, student_model, distill_config)
            elif distill_config.distillation_type == DistillationType.FEATURE:
                student_model = self._distill_features(teacher_model, student_model, distill_config)
            elif distill_config.distillation_type == DistillationType.ATTENTION:
                student_model = self._distill_attention(teacher_model, student_model, distill_config)
            elif distill_config.distillation_type == DistillationType.PROGRESSIVE:
                student_model = self._distill_progressive(teacher_model, student_model, distill_config)
            else:
                raise ValueError(f"Unsupported distillation type: {distill_config.distillation_type}")
            
            # Calculate performance metrics
            teacher_params = sum(p.numel() for p in teacher_model.parameters())
            student_params = sum(p.numel() for p in student_model.parameters())
            actual_compression = (teacher_params - student_params) / teacher_params
            
            performance_metrics = {
                "teacher_parameters": teacher_params,
                "student_parameters": student_params,
                "compression_ratio": actual_compression,
                "target_compression": distill_config.compression_ratio,
                "parameter_reduction": teacher_params - student_params
            }
            
            optimization_time = time.time() - start_time
            
            return OptimizedModel(
                model=student_model,
                optimization_metadata=optimization_metadata,
                performance_metrics=performance_metrics,
                optimization_time=optimization_time,
                technique_used=f"distillation_{distill_config.distillation_type.value}"
            )
            
        except Exception as e:
            self.logger.error(f"Knowledge distillation failed: {e}")
            raise
    
    def validate_result(self, original: torch.nn.Module, optimized: torch.nn.Module) -> ValidationResult:
        """Validate the distillation result."""
        issues = []
        recommendations = []
        
        try:
            # Check parameter reduction
            original_params = sum(p.numel() for p in original.parameters())
            optimized_params = sum(p.numel() for p in optimized.parameters())
            compression_ratio = (original_params - optimized_params) / original_params
            
            if compression_ratio < 0.1:
                issues.append("Insufficient model compression achieved")
            
            # Basic functionality test
            try:
                # Create dummy input for testing
                dummy_input = self._create_dummy_input(original)
                
                with torch.no_grad():
                    original_output = original(dummy_input)
                    optimized_output = optimized(dummy_input)
                
                # Check output shapes match
                if original_output.shape != optimized_output.shape:
                    issues.append(f"Output shape mismatch: {original_output.shape} vs {optimized_output.shape}")
                
                # Check for NaN or infinite values
                if torch.isnan(optimized_output).any():
                    issues.append("Optimized model produces NaN values")
                
                if torch.isinf(optimized_output).any():
                    issues.append("Optimized model produces infinite values")
                
                # Calculate output similarity
                mse = torch.nn.functional.mse_loss(original_output, optimized_output).item()
                cosine_sim = torch.nn.functional.cosine_similarity(
                    original_output.flatten(), optimized_output.flatten(), dim=0
                ).item()
                
                # For classification tasks, check if predictions are similar
                if original_output.dim() == 2 and original_output.shape[1] > 1:
                    original_pred = torch.argmax(original_output, dim=1)
                    optimized_pred = torch.argmax(optimized_output, dim=1)
                    prediction_agreement = (original_pred == optimized_pred).float().mean().item()
                else:
                    prediction_agreement = cosine_sim
                
                performance_metrics = {
                    "output_mse": mse,
                    "cosine_similarity": cosine_sim,
                    "prediction_agreement": prediction_agreement,
                    "compression_ratio": compression_ratio,
                    "parameter_reduction": original_params - optimized_params
                }
                
                # Recommendations based on results
                if mse > 1.0:
                    recommendations.append("High output difference. Consider increasing distillation epochs or temperature.")
                
                if cosine_sim < 0.8:
                    recommendations.append("Low output similarity. Check student architecture or distillation parameters.")
                
                if prediction_agreement < 0.7:
                    recommendations.append("Low prediction agreement. Model behavior may have changed significantly.")
                
                if compression_ratio > 0.8:
                    recommendations.append("Very high compression achieved. Verify model performance on validation data.")
                
            except Exception as e:
                issues.append(f"Model inference test failed: {e}")
                performance_metrics = {
                    "compression_ratio": compression_ratio,
                    "parameter_reduction": original_params - optimized_params
                }
            
            # Determine if validation passed
            is_valid = len(issues) == 0
            accuracy_preserved = len([issue for issue in issues if "NaN" in issue or "infinite" in issue]) == 0
            
            return ValidationResult(
                is_valid=is_valid,
                accuracy_preserved=accuracy_preserved,
                performance_metrics=performance_metrics,
                issues=issues,
                recommendations=recommendations
            )
            
        except Exception as e:
            self.logger.error(f"Validation failed: {e}")
            return ValidationResult(
                is_valid=False,
                accuracy_preserved=False,
                performance_metrics={},
                issues=[f"Validation error: {e}"],
                recommendations=["Check model compatibility and distillation configuration"]
            )
    
    def get_supported_techniques(self) -> List[str]:
        """Get list of distillation techniques supported by this agent."""
        return [
            "response",
            "feature",
            "attention",
            "progressive"
        ]
    
    def _parse_config(self, config: Dict[str, Any]) -> DistillationConfig:
        """Parse and validate distillation configuration."""
        distill_type = config.get('distillation_type', 'response')
        
        try:
            distillation_type = DistillationType(distill_type)
        except ValueError:
            self.logger.warning(f"Unknown distillation type: {distill_type}, defaulting to response")
            distillation_type = DistillationType.RESPONSE
        
        student_arch = config.get('student_architecture', 'smaller_same')
        try:
            student_architecture = StudentArchitecture(student_arch)
        except ValueError:
            self.logger.warning(f"Unknown student architecture: {student_arch}, defaulting to smaller_same")
            student_architecture = StudentArchitecture.SMALLER_SAME
        
        return DistillationConfig(
            distillation_type=distillation_type,
            student_architecture=student_architecture,
            temperature=config.get('temperature', 4.0),
            alpha=config.get('alpha', 0.7),
            beta=config.get('beta', 0.3),
            learning_rate=config.get('learning_rate', 1e-4),
            num_epochs=config.get('num_epochs', 10),
            batch_size=config.get('batch_size', 32),
            compression_ratio=config.get('compression_ratio', 0.5),
            feature_layers=config.get('feature_layers', []),
            preserve_accuracy=config.get('preserve_accuracy', 0.95)
        )
    
    def _create_student_model(self, teacher_model: torch.nn.Module, 
                             config: DistillationConfig) -> torch.nn.Module:
        """Create student model based on teacher and configuration."""
        if config.student_architecture == StudentArchitecture.SMALLER_SAME:
            return self._create_smaller_same_architecture(teacher_model, config.compression_ratio)
        elif config.student_architecture == StudentArchitecture.EFFICIENT:
            return self._create_efficient_architecture(teacher_model, config.compression_ratio)
        elif config.student_architecture == StudentArchitecture.CUSTOM:
            # For custom architecture, user should provide it in config
            if 'custom_student' in self.config:
                return self.config['custom_student']
            else:
                self.logger.warning("Custom student architecture not provided, using smaller_same")
                return self._create_smaller_same_architecture(teacher_model, config.compression_ratio)
        else:
            raise ValueError(f"Unsupported student architecture: {config.student_architecture}")
    
    def _create_smaller_same_architecture(self, teacher_model: torch.nn.Module, 
                                        compression_ratio: float) -> torch.nn.Module:
        """Create a smaller version of the same architecture."""
        # This is a simplified implementation - in practice, you'd need
        # more sophisticated architecture analysis and modification
        
        # For Sequential models, create a proportionally smaller version
        if hasattr(teacher_model, 'layers') and isinstance(teacher_model.layers, nn.Sequential):
            # Extract layer information
            linear_layers = []
            for module in teacher_model.layers:
                if isinstance(module, nn.Linear):
                    linear_layers.append((module.in_features, module.out_features))
            
            if linear_layers:
                # Create smaller version with proportional scaling
                input_size = linear_layers[0][0]
                output_size = linear_layers[-1][1]  # Keep output size the same
                
                # Scale intermediate layers
                scaled_layers = []
                scaled_layers.append(nn.Linear(input_size, max(32, int(linear_layers[0][1] * compression_ratio))))
                scaled_layers.append(nn.ReLU())
                
                # Add intermediate layers with scaled dimensions
                prev_size = scaled_layers[0].out_features
                for i in range(1, len(linear_layers) - 1):
                    scaled_size = max(16, int(linear_layers[i][1] * compression_ratio))
                    scaled_layers.append(nn.Linear(prev_size, scaled_size))
                    scaled_layers.append(nn.ReLU())
                    prev_size = scaled_size
                
                # Final layer keeps original output size
                scaled_layers.append(nn.Linear(prev_size, output_size))
                
                # Create new model with same structure
                class StudentModel(nn.Module):
                    def __init__(self, layers):
                        super().__init__()
                        self.layers = nn.Sequential(*layers)
                    
                    def forward(self, x):
                        return self.layers(x)
                
                return StudentModel(scaled_layers)
        
        # Fallback: just copy the teacher model
        return copy.deepcopy(teacher_model)
    
    def _create_efficient_architecture(self, teacher_model: torch.nn.Module, 
                                     compression_ratio: float) -> torch.nn.Module:
        """Create an efficient architecture (simplified implementation)."""
        # For now, use the same as smaller_same - in practice, this would
        # implement more sophisticated efficiency improvements
        return self._create_smaller_same_architecture(teacher_model, compression_ratio)
    
    def _distill_response(self, teacher_model: torch.nn.Module, student_model: torch.nn.Module,
                         config: DistillationConfig) -> torch.nn.Module:
        """Perform response-based knowledge distillation."""
        self._update_progress(self._current_status, 40.0, "Starting response distillation")
        
        # Set up training
        optimizer = torch.optim.Adam(student_model.parameters(), lr=config.learning_rate)
        loss_fn = DistillationLoss(temperature=config.temperature, alpha=config.alpha)
        
        # Generate synthetic training data (in practice, use real data)
        dummy_input = self._create_dummy_input(teacher_model)
        batch_size = config.batch_size
        
        teacher_model.eval()
        student_model.train()
        
        for epoch in range(config.num_epochs):
            # Generate batch of synthetic data
            batch_input = torch.randn(batch_size, *dummy_input.shape[1:])
            
            # Get teacher predictions
            with torch.no_grad():
                teacher_logits = teacher_model(batch_input)
            
            # Get student predictions
            student_logits = student_model(batch_input)
            
            # Compute loss
            loss = loss_fn(student_logits, teacher_logits)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update progress
            progress = 40.0 + (epoch + 1) / config.num_epochs * 40.0
            self._update_progress(self._current_status, progress, 
                                f"Distillation epoch {epoch + 1}/{config.num_epochs}")
            
            if self.is_cancelled():
                break
        
        student_model.eval()
        return student_model
    
    def _distill_features(self, teacher_model: torch.nn.Module, student_model: torch.nn.Module,
                         config: DistillationConfig) -> torch.nn.Module:
        """Perform feature-based knowledge distillation."""
        self._update_progress(self._current_status, 40.0, "Starting feature distillation")
        
        # For simplicity, fall back to response distillation
        # In practice, this would extract intermediate features
        return self._distill_response(teacher_model, student_model, config)
    
    def _distill_attention(self, teacher_model: torch.nn.Module, student_model: torch.nn.Module,
                          config: DistillationConfig) -> torch.nn.Module:
        """Perform attention-based knowledge distillation."""
        self._update_progress(self._current_status, 40.0, "Starting attention distillation")
        
        # For simplicity, fall back to response distillation
        # In practice, this would extract attention maps
        return self._distill_response(teacher_model, student_model, config)
    
    def _distill_progressive(self, teacher_model: torch.nn.Module, student_model: torch.nn.Module,
                           config: DistillationConfig) -> torch.nn.Module:
        """Perform progressive knowledge distillation."""
        self._update_progress(self._current_status, 40.0, "Starting progressive distillation")
        
        # For simplicity, fall back to response distillation
        # In practice, this would progressively increase model complexity
        return self._distill_response(teacher_model, student_model, config)
    
    def _replace_module(self, model: torch.nn.Module, module_name: str, new_module: torch.nn.Module) -> None:
        """Replace a module in the model with a new module."""
        module_path = module_name.split('.')
        parent = model
        
        for part in module_path[:-1]:
            parent = getattr(parent, part)
        
        setattr(parent, module_path[-1], new_module)
    
    def _create_dummy_input(self, model: torch.nn.Module) -> torch.Tensor:
        """Create dummy input for model testing."""
        # Try to infer input shape from first layer
        first_layer = None
        for module in model.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv1d)):
                first_layer = module
                break
        
        if isinstance(first_layer, nn.Linear):
            return torch.randn(1, first_layer.in_features)
        elif isinstance(first_layer, nn.Conv2d):
            return torch.randn(1, first_layer.in_channels, 224, 224)
        elif isinstance(first_layer, nn.Conv1d):
            return torch.randn(1, first_layer.in_channels, 100)
        else:
            # Default fallback
            return torch.randn(1, 768)
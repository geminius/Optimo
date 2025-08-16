"""
Custom exception classes for the robotics model optimization platform.
"""

from typing import Optional, Dict, Any, List
from enum import Enum


class ErrorSeverity(Enum):
    """Severity levels for errors."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Categories of errors in the platform."""
    MODEL_LOADING = "model_loading"
    OPTIMIZATION = "optimization"
    EVALUATION = "evaluation"
    SYSTEM = "system"
    VALIDATION = "validation"
    CONFIGURATION = "configuration"
    NETWORK = "network"
    STORAGE = "storage"


class PlatformError(Exception):
    """Base exception class for all platform errors."""
    
    def __init__(
        self,
        message: str,
        category: ErrorCategory,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        error_code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        recoverable: bool = True,
        retry_after: Optional[int] = None
    ):
        super().__init__(message)
        self.message = message
        self.category = category
        self.severity = severity
        self.error_code = error_code or f"{category.value}_error"
        self.context = context or {}
        self.recoverable = recoverable
        self.retry_after = retry_after  # seconds
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for logging/serialization."""
        return {
            "message": self.message,
            "category": self.category.value,
            "severity": self.severity.value,
            "error_code": self.error_code,
            "context": self.context,
            "recoverable": self.recoverable,
            "retry_after": self.retry_after
        }


class ModelLoadingError(PlatformError):
    """Errors related to model loading and format issues."""
    
    def __init__(
        self,
        message: str,
        model_path: Optional[str] = None,
        model_format: Optional[str] = None,
        **kwargs
    ):
        context = kwargs.get('context', {})
        if model_path:
            context['model_path'] = model_path
        if model_format:
            context['model_format'] = model_format
        
        super().__init__(
            message=message,
            category=ErrorCategory.MODEL_LOADING,
            context=context,
            **kwargs
        )


class OptimizationError(PlatformError):
    """Errors during optimization operations."""
    
    def __init__(
        self,
        message: str,
        technique: Optional[str] = None,
        session_id: Optional[str] = None,
        step: Optional[str] = None,
        **kwargs
    ):
        context = kwargs.get('context', {})
        if technique:
            context['technique'] = technique
        if session_id:
            context['session_id'] = session_id
        if step:
            context['step'] = step
        
        super().__init__(
            message=message,
            category=ErrorCategory.OPTIMIZATION,
            context=context,
            **kwargs
        )


class EvaluationError(PlatformError):
    """Errors during model evaluation."""
    
    def __init__(
        self,
        message: str,
        benchmark: Optional[str] = None,
        metric: Optional[str] = None,
        **kwargs
    ):
        context = kwargs.get('context', {})
        if benchmark:
            context['benchmark'] = benchmark
        if metric:
            context['metric'] = metric
        
        super().__init__(
            message=message,
            category=ErrorCategory.EVALUATION,
            context=context,
            **kwargs
        )


class ValidationError(PlatformError):
    """Errors during validation operations."""
    
    def __init__(
        self,
        message: str,
        validation_type: Optional[str] = None,
        failed_checks: Optional[List[str]] = None,
        **kwargs
    ):
        context = kwargs.get('context', {})
        if validation_type:
            context['validation_type'] = validation_type
        if failed_checks:
            context['failed_checks'] = failed_checks
        
        super().__init__(
            message=message,
            category=ErrorCategory.VALIDATION,
            context=context,
            **kwargs
        )


class ConfigurationError(PlatformError):
    """Errors related to configuration issues."""
    
    def __init__(
        self,
        message: str,
        config_key: Optional[str] = None,
        config_value: Optional[Any] = None,
        **kwargs
    ):
        context = kwargs.get('context', {})
        if config_key:
            context['config_key'] = config_key
        if config_value is not None:
            context['config_value'] = str(config_value)
        
        super().__init__(
            message=message,
            category=ErrorCategory.CONFIGURATION,
            context=context,
            recoverable=False,  # Config errors usually require manual intervention
            **kwargs
        )


class SystemError(PlatformError):
    """System-level errors (resource exhaustion, etc.)."""
    
    def __init__(
        self,
        message: str,
        resource_type: Optional[str] = None,
        **kwargs
    ):
        context = kwargs.get('context', {})
        if resource_type:
            context['resource_type'] = resource_type
        
        super().__init__(
            message=message,
            category=ErrorCategory.SYSTEM,
            severity=ErrorSeverity.HIGH,
            context=context,
            **kwargs
        )


class NetworkError(PlatformError):
    """Network-related errors."""
    
    def __init__(
        self,
        message: str,
        endpoint: Optional[str] = None,
        status_code: Optional[int] = None,
        **kwargs
    ):
        context = kwargs.get('context', {})
        if endpoint:
            context['endpoint'] = endpoint
        if status_code:
            context['status_code'] = status_code
        
        super().__init__(
            message=message,
            category=ErrorCategory.NETWORK,
            context=context,
            retry_after=30,  # Default retry after 30 seconds for network errors
            **kwargs
        )


class StorageError(PlatformError):
    """Storage-related errors."""
    
    def __init__(
        self,
        message: str,
        file_path: Optional[str] = None,
        operation: Optional[str] = None,
        **kwargs
    ):
        context = kwargs.get('context', {})
        if file_path:
            context['file_path'] = file_path
        if operation:
            context['operation'] = operation
        
        super().__init__(
            message=message,
            category=ErrorCategory.STORAGE,
            context=context,
            **kwargs
        )


# Convenience functions for creating common errors
def create_model_loading_error(message: str, model_path: str, **kwargs) -> ModelLoadingError:
    """Create a model loading error with common context."""
    return ModelLoadingError(
        message=message,
        model_path=model_path,
        **kwargs
    )


def create_optimization_error(message: str, technique: str, session_id: str, **kwargs) -> OptimizationError:
    """Create an optimization error with common context."""
    return OptimizationError(
        message=message,
        technique=technique,
        session_id=session_id,
        **kwargs
    )


def create_validation_error(message: str, validation_type: str, **kwargs) -> ValidationError:
    """Create a validation error with common context."""
    return ValidationError(
        message=message,
        validation_type=validation_type,
        **kwargs
    )
"""
Retry logic with exponential backoff for handling transient failures.
"""

import time
import random
import logging
from typing import Callable, Any, Optional, Type, Union, List, Tuple
from functools import wraps
from .exceptions import PlatformError, ErrorSeverity, ErrorCategory


logger = logging.getLogger(__name__)


class RetryConfig:
    """Configuration for retry behavior."""
    
    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
        retryable_exceptions: Optional[List[Type[Exception]]] = None,
        non_retryable_exceptions: Optional[List[Type[Exception]]] = None
    ):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
        self.retryable_exceptions = retryable_exceptions or [
            ConnectionError,
            TimeoutError,
            OSError,
            PlatformError
        ]
        self.non_retryable_exceptions = non_retryable_exceptions or []
    
    def calculate_delay(self, attempt: int) -> float:
        """Calculate delay for the given attempt number."""
        delay = self.base_delay * (self.exponential_base ** (attempt - 1))
        delay = min(delay, self.max_delay)
        
        if self.jitter:
            # Add jitter to prevent thundering herd
            jitter_range = delay * 0.1
            delay += random.uniform(-jitter_range, jitter_range)
        
        return max(0, delay)
    
    def should_retry(self, exception: Exception, attempt: int) -> bool:
        """Determine if we should retry based on the exception and attempt count."""
        if attempt >= self.max_attempts:
            return False
        
        # Check non-retryable exceptions first
        for exc_type in self.non_retryable_exceptions:
            if isinstance(exception, exc_type):
                return False
        
        # For PlatformError, check if it's recoverable
        if isinstance(exception, PlatformError):
            if not exception.recoverable:
                return False
            # Don't retry critical errors
            if exception.severity == ErrorSeverity.CRITICAL:
                return False
        
        # Check retryable exceptions
        for exc_type in self.retryable_exceptions:
            if isinstance(exception, exc_type):
                return True
        
        return False


def retry_with_backoff(
    config: Optional[RetryConfig] = None,
    on_retry: Optional[Callable[[Exception, int], None]] = None
):
    """
    Decorator for retrying functions with exponential backoff.
    
    Args:
        config: RetryConfig instance, uses default if None
        on_retry: Optional callback called on each retry attempt
    """
    if config is None:
        config = RetryConfig()
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            last_exception = None
            
            for attempt in range(1, config.max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    
                    if not config.should_retry(e, attempt):
                        logger.error(f"Function {func.__name__} failed on attempt {attempt}, not retrying: {e}")
                        raise e
                    
                    if attempt < config.max_attempts:
                        delay = config.calculate_delay(attempt)
                        logger.warning(
                            f"Function {func.__name__} failed on attempt {attempt}/{config.max_attempts}, "
                            f"retrying in {delay:.2f}s: {e}"
                        )
                        
                        if on_retry:
                            try:
                                on_retry(e, attempt)
                            except Exception as callback_error:
                                logger.error(f"Error in retry callback: {callback_error}")
                        
                        time.sleep(delay)
                    else:
                        logger.error(f"Function {func.__name__} failed after {attempt} attempts: {e}")
            
            # If we get here, all attempts failed
            raise last_exception
        
        return wrapper
    return decorator


class RetryableOperation:
    """Context manager for retryable operations with more control."""
    
    def __init__(
        self,
        operation_name: str,
        config: Optional[RetryConfig] = None,
        on_retry: Optional[Callable[[Exception, int], None]] = None,
        on_success: Optional[Callable[[int], None]] = None,
        on_failure: Optional[Callable[[Exception, int], None]] = None
    ):
        self.operation_name = operation_name
        self.config = config or RetryConfig()
        self.on_retry = on_retry
        self.on_success = on_success
        self.on_failure = on_failure
        self.attempt = 0
        self.last_exception = None
    
    def execute(self, func: Callable, *args, **kwargs) -> Any:
        """Execute a function with retry logic."""
        self.last_exception = None
        
        for attempt in range(1, self.config.max_attempts + 1):
            self.attempt = attempt
            try:
                result = func(*args, **kwargs)
                
                if self.on_success:
                    try:
                        self.on_success(attempt)
                    except Exception as callback_error:
                        logger.error(f"Error in success callback: {callback_error}")
                
                logger.info(f"Operation {self.operation_name} succeeded on attempt {attempt}")
                return result
                
            except Exception as e:
                self.last_exception = e
                
                if not self.config.should_retry(e, attempt):
                    logger.error(f"Operation {self.operation_name} failed on attempt {attempt}, not retrying: {e}")
                    if self.on_failure:
                        try:
                            self.on_failure(e, attempt)
                        except Exception as callback_error:
                            logger.error(f"Error in failure callback: {callback_error}")
                    raise e
                
                if attempt < self.config.max_attempts:
                    delay = self.config.calculate_delay(attempt)
                    logger.warning(
                        f"Operation {self.operation_name} failed on attempt {attempt}/{self.config.max_attempts}, "
                        f"retrying in {delay:.2f}s: {e}"
                    )
                    
                    if self.on_retry:
                        try:
                            self.on_retry(e, attempt)
                        except Exception as callback_error:
                            logger.error(f"Error in retry callback: {callback_error}")
                    
                    time.sleep(delay)
                else:
                    logger.error(f"Operation {self.operation_name} failed after {attempt} attempts: {e}")
                    if self.on_failure:
                        try:
                            self.on_failure(e, attempt)
                        except Exception as callback_error:
                            logger.error(f"Error in failure callback: {callback_error}")
        
        # If we get here, all attempts failed
        raise self.last_exception


# Predefined retry configurations for common scenarios
QUICK_RETRY = RetryConfig(
    max_attempts=3,
    base_delay=0.5,
    max_delay=5.0
)

STANDARD_RETRY = RetryConfig(
    max_attempts=5,
    base_delay=1.0,
    max_delay=30.0
)

PERSISTENT_RETRY = RetryConfig(
    max_attempts=10,
    base_delay=2.0,
    max_delay=120.0
)

NETWORK_RETRY = RetryConfig(
    max_attempts=5,
    base_delay=1.0,
    max_delay=60.0,
    retryable_exceptions=[
        ConnectionError,
        TimeoutError,
        OSError
    ]
)


def create_retry_config_for_error_category(category: ErrorCategory) -> RetryConfig:
    """Create appropriate retry configuration based on error category."""
    if category == ErrorCategory.NETWORK:
        return NETWORK_RETRY
    elif category == ErrorCategory.SYSTEM:
        return PERSISTENT_RETRY
    elif category == ErrorCategory.STORAGE:
        return STANDARD_RETRY
    elif category == ErrorCategory.OPTIMIZATION:
        return STANDARD_RETRY
    else:
        return QUICK_RETRY
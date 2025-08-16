"""
Logging Integration - Comprehensive logging setup across all platform components.

This module provides centralized logging configuration and monitoring
to ensure consistent and comprehensive logging throughout the platform.
"""

import logging
import logging.handlers
import sys
import json
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime
import threading


class PlatformLogFormatter(logging.Formatter):
    """Custom log formatter for the platform."""
    
    def __init__(self):
        super().__init__()
        self.default_format = (
            "%(asctime)s - %(name)s - %(levelname)s - "
            "[%(filename)s:%(lineno)d] - %(message)s"
        )
        self.json_format = True
    
    def format(self, record):
        """Format log record."""
        if self.json_format:
            return self._format_json(record)
        else:
            return self._format_text(record)
    
    def _format_json(self, record):
        """Format as JSON."""
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "filename": record.filename,
            "line_number": record.lineno,
            "thread_id": record.thread,
            "thread_name": record.threadName
        }
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
                          'filename', 'module', 'lineno', 'funcName', 'created',
                          'msecs', 'relativeCreated', 'thread', 'threadName',
                          'processName', 'process', 'getMessage', 'exc_info',
                          'exc_text', 'stack_info']:
                log_data[key] = value
        
        return json.dumps(log_data)
    
    def _format_text(self, record):
        """Format as text."""
        formatter = logging.Formatter(self.default_format)
        return formatter.format(record)


class LoggingIntegrator:
    """
    Comprehensive logging integration for the platform.
    
    Provides centralized logging configuration, monitoring, and management
    across all platform components.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Configuration
        self.log_level = config.get("level", "INFO")
        self.log_dir = Path(config.get("log_dir", "logs"))
        self.max_file_size = config.get("max_file_size_mb", 100) * 1024 * 1024
        self.backup_count = config.get("backup_count", 5)
        self.json_format = config.get("json_format", True)
        self.console_output = config.get("console_output", True)
        
        # Component loggers
        self.component_loggers: Dict[str, logging.Logger] = {}
        self.handlers: List[logging.Handler] = []
        
        # Thread safety
        self._lock = threading.Lock()
        
        self.logger.info("LoggingIntegrator initialized")
    
    async def initialize(self) -> None:
        """Initialize logging system."""
        try:
            self.logger.info("Initializing logging integration")
            
            # Create log directory
            self.log_dir.mkdir(parents=True, exist_ok=True)
            
            # Set up root logger
            await self._setup_root_logger()
            
            # Set up component-specific loggers
            await self._setup_component_loggers()
            
            # Set up log monitoring
            await self._setup_log_monitoring()
            
            self.logger.info("Logging integration initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize logging integration: {e}")
            raise
    
    async def cleanup(self) -> None:
        """Clean up logging resources."""
        try:
            self.logger.info("Cleaning up logging integration")
            
            # Close all handlers
            for handler in self.handlers:
                handler.close()
            
            # Clear handlers
            self.handlers.clear()
            self.component_loggers.clear()
            
            self.logger.info("Logging integration cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Error during logging cleanup: {e}")
    
    async def setup_component_logging(self, components: List[Any]) -> None:
        """Set up logging for platform components."""
        with self._lock:
            for component in components:
                if component is None:
                    continue
                
                component_name = component.__class__.__name__
                
                # Create component-specific logger
                component_logger = logging.getLogger(f"platform.{component_name}")
                
                # Configure logger
                component_logger.setLevel(getattr(logging, self.log_level))
                
                # Add handlers if not already present
                if not component_logger.handlers:
                    for handler in self.handlers:
                        component_logger.addHandler(handler)
                
                self.component_loggers[component_name] = component_logger
                
                # Inject logger into component if it has a logger attribute
                if hasattr(component, 'logger'):
                    component.logger = component_logger
                
                self.logger.debug(f"Set up logging for component: {component_name}")
    
    def get_component_logger(self, component_name: str) -> logging.Logger:
        """Get logger for a specific component."""
        return self.component_loggers.get(
            component_name, 
            logging.getLogger(f"platform.{component_name}")
        )
    
    def add_log_handler(self, handler: logging.Handler) -> None:
        """Add a custom log handler."""
        with self._lock:
            self.handlers.append(handler)
            
            # Add to all existing loggers
            for logger in self.component_loggers.values():
                logger.addHandler(handler)
    
    def remove_log_handler(self, handler: logging.Handler) -> None:
        """Remove a log handler."""
        with self._lock:
            if handler in self.handlers:
                self.handlers.remove(handler)
                
                # Remove from all loggers
                for logger in self.component_loggers.values():
                    logger.removeHandler(handler)
                
                handler.close()
    
    def get_log_statistics(self) -> Dict[str, Any]:
        """Get logging statistics."""
        stats = {
            "total_loggers": len(self.component_loggers),
            "total_handlers": len(self.handlers),
            "log_level": self.log_level,
            "log_directory": str(self.log_dir),
            "json_format": self.json_format,
            "console_output": self.console_output
        }
        
        # Add handler statistics
        handler_stats = {}
        for i, handler in enumerate(self.handlers):
            handler_stats[f"handler_{i}"] = {
                "type": handler.__class__.__name__,
                "level": logging.getLevelName(handler.level)
            }
            
            if hasattr(handler, 'baseFilename'):
                handler_stats[f"handler_{i}"]["filename"] = handler.baseFilename
        
        stats["handlers"] = handler_stats
        
        return stats
    
    async def _setup_root_logger(self) -> None:
        """Set up the root logger configuration."""
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, self.log_level))
        
        # Clear existing handlers
        root_logger.handlers.clear()
        
        # Create formatter
        formatter = PlatformLogFormatter()
        formatter.json_format = self.json_format
        
        # Console handler
        if self.console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(getattr(logging, self.log_level))
            console_handler.setFormatter(formatter)
            root_logger.addHandler(console_handler)
            self.handlers.append(console_handler)
        
        # File handler for general logs
        general_log_file = self.log_dir / "platform.log"
        file_handler = logging.handlers.RotatingFileHandler(
            general_log_file,
            maxBytes=self.max_file_size,
            backupCount=self.backup_count
        )
        file_handler.setLevel(getattr(logging, self.log_level))
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
        self.handlers.append(file_handler)
        
        # Error log file
        error_log_file = self.log_dir / "errors.log"
        error_handler = logging.handlers.RotatingFileHandler(
            error_log_file,
            maxBytes=self.max_file_size,
            backupCount=self.backup_count
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(formatter)
        root_logger.addHandler(error_handler)
        self.handlers.append(error_handler)
    
    async def _setup_component_loggers(self) -> None:
        """Set up loggers for specific platform components."""
        # Define component-specific log configurations
        component_configs = {
            "optimization_manager": {"level": "INFO", "file": "optimization.log"},
            "analysis_agent": {"level": "INFO", "file": "analysis.log"},
            "planning_agent": {"level": "INFO", "file": "planning.log"},
            "evaluation_agent": {"level": "INFO", "file": "evaluation.log"},
            "quantization_agent": {"level": "INFO", "file": "quantization.log"},
            "pruning_agent": {"level": "INFO", "file": "pruning.log"},
            "distillation_agent": {"level": "INFO", "file": "distillation.log"},
            "compression_agent": {"level": "INFO", "file": "compression.log"},
            "architecture_search_agent": {"level": "INFO", "file": "architecture_search.log"},
            "model_store": {"level": "INFO", "file": "model_store.log"},
            "memory_manager": {"level": "INFO", "file": "memory.log"},
            "notification_service": {"level": "INFO", "file": "notifications.log"},
            "monitoring_service": {"level": "INFO", "file": "monitoring.log"}
        }
        
        formatter = PlatformLogFormatter()
        formatter.json_format = self.json_format
        
        for component_name, config in component_configs.items():
            # Create component logger
            logger = logging.getLogger(f"platform.{component_name}")
            logger.setLevel(getattr(logging, config["level"]))
            
            # Create component-specific file handler
            log_file = self.log_dir / config["file"]
            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=self.max_file_size,
                backupCount=self.backup_count
            )
            file_handler.setLevel(getattr(logging, config["level"]))
            file_handler.setFormatter(formatter)
            
            logger.addHandler(file_handler)
            self.handlers.append(file_handler)
            self.component_loggers[component_name] = logger
    
    async def _setup_log_monitoring(self) -> None:
        """Set up log monitoring and alerting."""
        # Create a handler for critical errors that need immediate attention
        critical_log_file = self.log_dir / "critical.log"
        critical_handler = logging.handlers.RotatingFileHandler(
            critical_log_file,
            maxBytes=self.max_file_size,
            backupCount=self.backup_count
        )
        critical_handler.setLevel(logging.CRITICAL)
        
        formatter = PlatformLogFormatter()
        formatter.json_format = self.json_format
        critical_handler.setFormatter(formatter)
        
        # Add to root logger
        root_logger = logging.getLogger()
        root_logger.addHandler(critical_handler)
        self.handlers.append(critical_handler)
        
        # Set up performance monitoring logger
        perf_log_file = self.log_dir / "performance.log"
        perf_handler = logging.handlers.RotatingFileHandler(
            perf_log_file,
            maxBytes=self.max_file_size,
            backupCount=self.backup_count
        )
        perf_handler.setLevel(logging.INFO)
        perf_handler.setFormatter(formatter)
        
        # Create performance logger
        perf_logger = logging.getLogger("platform.performance")
        perf_logger.addHandler(perf_handler)
        self.handlers.append(perf_handler)
        self.component_loggers["performance"] = perf_logger
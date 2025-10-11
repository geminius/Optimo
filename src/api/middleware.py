"""
API middleware for monitoring, logging, and performance tracking.

This module provides middleware components for:
- Request/response logging with timing
- Performance monitoring
- Error tracking
- Request ID generation
"""

import logging
import time
import uuid
from typing import Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp


logger = logging.getLogger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware for logging all API requests with timing information.
    
    Logs:
    - Request method, path, and query parameters
    - Response status code
    - Request duration in milliseconds
    - Request ID for tracking
    - User information (if authenticated)
    """
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.logger = logging.getLogger(f"{__name__}.RequestLogging")
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request and log timing information."""
        # Generate unique request ID
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        # Record start time
        start_time = time.time()
        
        # Extract request information
        method = request.method
        path = request.url.path
        query_params = str(request.query_params) if request.query_params else ""
        client_host = request.client.host if request.client else "unknown"
        
        # Log request start
        self.logger.info(
            f"Request started: {method} {path}",
            extra={
                "component": "RequestLogging",
                "request_id": request_id,
                "method": method,
                "path": path,
                "query_params": query_params,
                "client_host": client_host,
                "event": "request_start"
            }
        )
        
        # Process request
        try:
            response = await call_next(request)
            
            # Calculate duration
            duration_ms = (time.time() - start_time) * 1000
            
            # Log request completion
            self.logger.info(
                f"Request completed: {method} {path} - {response.status_code} ({duration_ms:.2f}ms)",
                extra={
                    "component": "RequestLogging",
                    "request_id": request_id,
                    "method": method,
                    "path": path,
                    "status_code": response.status_code,
                    "duration_ms": duration_ms,
                    "client_host": client_host,
                    "event": "request_complete"
                }
            )
            
            # Add request ID to response headers
            response.headers["X-Request-ID"] = request_id
            
            # Add timing header
            response.headers["X-Response-Time"] = f"{duration_ms:.2f}ms"
            
            return response
            
        except Exception as e:
            # Calculate duration even for errors
            duration_ms = (time.time() - start_time) * 1000
            
            # Log request error
            self.logger.error(
                f"Request failed: {method} {path} - {str(e)} ({duration_ms:.2f}ms)",
                extra={
                    "component": "RequestLogging",
                    "request_id": request_id,
                    "method": method,
                    "path": path,
                    "error": str(e),
                    "duration_ms": duration_ms,
                    "client_host": client_host,
                    "event": "request_error"
                },
                exc_info=True
            )
            
            # Re-raise exception to be handled by error handlers
            raise


class PerformanceMonitoringMiddleware(BaseHTTPMiddleware):
    """
    Middleware for monitoring endpoint performance and tracking slow requests.
    
    Features:
    - Track request duration
    - Log slow requests (> threshold)
    - Collect performance metrics
    - Alert on performance degradation
    """
    
    def __init__(self, app: ASGIApp, slow_request_threshold_ms: float = 1000.0):
        super().__init__(app)
        self.logger = logging.getLogger(f"{__name__}.PerformanceMonitoring")
        self.slow_request_threshold_ms = slow_request_threshold_ms
        
        # Performance metrics
        self._request_count = 0
        self._total_duration_ms = 0.0
        self._slow_request_count = 0
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Monitor request performance."""
        start_time = time.time()
        
        try:
            response = await call_next(request)
            
            # Calculate duration
            duration_ms = (time.time() - start_time) * 1000
            
            # Update metrics
            self._request_count += 1
            self._total_duration_ms += duration_ms
            
            # Check for slow requests
            if duration_ms > self.slow_request_threshold_ms:
                self._slow_request_count += 1
                
                self.logger.warning(
                    f"Slow request detected: {request.method} {request.url.path} ({duration_ms:.2f}ms)",
                    extra={
                        "component": "PerformanceMonitoring",
                        "request_id": getattr(request.state, "request_id", "unknown"),
                        "method": request.method,
                        "path": request.url.path,
                        "duration_ms": duration_ms,
                        "threshold_ms": self.slow_request_threshold_ms,
                        "event": "slow_request"
                    }
                )
            
            return response
            
        except Exception as e:
            # Still track duration for failed requests
            duration_ms = (time.time() - start_time) * 1000
            self._request_count += 1
            self._total_duration_ms += duration_ms
            raise
    
    def get_metrics(self) -> dict:
        """Get performance metrics."""
        avg_duration_ms = (
            self._total_duration_ms / self._request_count 
            if self._request_count > 0 
            else 0.0
        )
        
        slow_request_rate = (
            self._slow_request_count / self._request_count * 100
            if self._request_count > 0
            else 0.0
        )
        
        return {
            "total_requests": self._request_count,
            "average_duration_ms": avg_duration_ms,
            "slow_requests": self._slow_request_count,
            "slow_request_rate_percent": slow_request_rate,
            "threshold_ms": self.slow_request_threshold_ms
        }


class WebSocketMetricsMiddleware:
    """
    Middleware for tracking WebSocket connection metrics.
    
    This is not a standard HTTP middleware but provides utilities
    for tracking WebSocket connections.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.WebSocketMetrics")
        
        # Connection metrics
        self._total_connections = 0
        self._active_connections = 0
        self._total_messages_sent = 0
        self._total_messages_received = 0
        self._connection_errors = 0
    
    def on_connect(self, client_id: str) -> None:
        """Track new WebSocket connection."""
        self._total_connections += 1
        self._active_connections += 1
        
        self.logger.info(
            f"WebSocket connected: {client_id}",
            extra={
                "component": "WebSocketMetrics",
                "client_id": client_id,
                "active_connections": self._active_connections,
                "total_connections": self._total_connections,
                "event": "ws_connect"
            }
        )
    
    def on_disconnect(self, client_id: str) -> None:
        """Track WebSocket disconnection."""
        self._active_connections = max(0, self._active_connections - 1)
        
        self.logger.info(
            f"WebSocket disconnected: {client_id}",
            extra={
                "component": "WebSocketMetrics",
                "client_id": client_id,
                "active_connections": self._active_connections,
                "event": "ws_disconnect"
            }
        )
    
    def on_message_sent(self, client_id: str, event_type: str) -> None:
        """Track message sent to client."""
        self._total_messages_sent += 1
        
        self.logger.debug(
            f"WebSocket message sent: {event_type} to {client_id}",
            extra={
                "component": "WebSocketMetrics",
                "client_id": client_id,
                "event_type": event_type,
                "total_messages_sent": self._total_messages_sent,
                "event": "ws_message_sent"
            }
        )
    
    def on_message_received(self, client_id: str, event_type: str) -> None:
        """Track message received from client."""
        self._total_messages_received += 1
        
        self.logger.debug(
            f"WebSocket message received: {event_type} from {client_id}",
            extra={
                "component": "WebSocketMetrics",
                "client_id": client_id,
                "event_type": event_type,
                "total_messages_received": self._total_messages_received,
                "event": "ws_message_received"
            }
        )
    
    def on_error(self, client_id: str, error: str) -> None:
        """Track WebSocket error."""
        self._connection_errors += 1
        
        self.logger.error(
            f"WebSocket error: {error} for {client_id}",
            extra={
                "component": "WebSocketMetrics",
                "client_id": client_id,
                "error": error,
                "connection_errors": self._connection_errors,
                "event": "ws_error"
            }
        )
    
    def get_metrics(self) -> dict:
        """Get WebSocket metrics."""
        return {
            "total_connections": self._total_connections,
            "active_connections": self._active_connections,
            "total_messages_sent": self._total_messages_sent,
            "total_messages_received": self._total_messages_received,
            "connection_errors": self._connection_errors
        }

"""
Pydantic models for WebSocket event schemas.

This module defines all event types and their payloads for real-time
WebSocket communication in the robotics model optimization platform.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
from enum import Enum


class EventType(str, Enum):
    """WebSocket event types."""
    # Connection events
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    ERROR = "error"
    
    # Session lifecycle events
    SESSION_STARTED = "session_started"
    SESSION_PROGRESS = "progress_update"
    SESSION_COMPLETED = "session_completed"
    SESSION_FAILED = "session_failed"
    SESSION_CANCELLED = "session_cancelled"
    
    # Notification events
    NOTIFICATION = "notification"
    ALERT = "alert"
    
    # System events
    SYSTEM_STATUS = "system_status"
    
    # Subscription events
    SUBSCRIBED = "subscribed"
    UNSUBSCRIBED = "unsubscribed"
    
    # Health check
    PING = "ping"
    PONG = "pong"


class NotificationTypeEnum(str, Enum):
    """Notification types."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    SUCCESS = "success"
    PROGRESS = "progress"


class AlertSeverityEnum(str, Enum):
    """Alert severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# Connection Events

class ConnectedEvent(BaseModel):
    """Event sent when client successfully connects."""
    sid: str = Field(..., description="Socket.IO session ID")
    timestamp: datetime = Field(..., description="Connection timestamp")
    message: str = Field(..., description="Connection confirmation message")


class DisconnectedEvent(BaseModel):
    """Event sent when client disconnects."""
    sid: str = Field(..., description="Socket.IO session ID")
    timestamp: datetime = Field(..., description="Disconnection timestamp")
    reason: Optional[str] = Field(None, description="Disconnection reason")


class ErrorEvent(BaseModel):
    """Event sent when an error occurs."""
    message: str = Field(..., description="Error message")
    error_type: Optional[str] = Field(None, description="Error type")
    timestamp: datetime = Field(default_factory=datetime.now, description="Error timestamp")


# Session Lifecycle Events

class SessionStartedEvent(BaseModel):
    """Event sent when optimization session starts."""
    session_id: str = Field(..., description="Unique session identifier")
    model_id: str = Field(..., description="Model being optimized")
    model_name: str = Field(..., description="Human-readable model name")
    techniques: List[str] = Field(..., description="Optimization techniques to be applied")
    timestamp: datetime = Field(default_factory=datetime.now, description="Event timestamp")


class SessionProgressEvent(BaseModel):
    """Event sent for session progress updates."""
    session_id: str = Field(..., description="Session identifier")
    current_step: int = Field(..., ge=0, description="Current step number")
    total_steps: int = Field(..., gt=0, description="Total number of steps")
    step_name: str = Field(..., description="Name of current step")
    progress_percentage: float = Field(..., ge=0.0, le=100.0, description="Progress percentage")
    estimated_completion: Optional[datetime] = Field(None, description="Estimated completion time")
    elapsed_time: Optional[str] = Field(None, description="Elapsed time (formatted)")
    remaining_time: Optional[str] = Field(None, description="Remaining time (formatted)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class OptimizationResults(BaseModel):
    """Optimization results data."""
    size_reduction_percent: Optional[float] = Field(None, description="Size reduction achieved")
    speed_improvement_percent: Optional[float] = Field(None, description="Speed improvement achieved")
    accuracy_change_percent: Optional[float] = Field(None, description="Accuracy change")
    original_size_mb: Optional[float] = Field(None, description="Original model size")
    optimized_size_mb: Optional[float] = Field(None, description="Optimized model size")
    original_inference_time_ms: Optional[float] = Field(None, description="Original inference time")
    optimized_inference_time_ms: Optional[float] = Field(None, description="Optimized inference time")
    techniques_applied: List[str] = Field(default_factory=list, description="Techniques applied")
    evaluation_metrics: Dict[str, Any] = Field(default_factory=dict, description="Evaluation metrics")


class SessionCompletedEvent(BaseModel):
    """Event sent when session completes successfully."""
    session_id: str = Field(..., description="Session identifier")
    results: OptimizationResults = Field(..., description="Optimization results")
    timestamp: datetime = Field(default_factory=datetime.now, description="Completion timestamp")


class SessionFailedEvent(BaseModel):
    """Event sent when session fails."""
    session_id: str = Field(..., description="Session identifier")
    error_message: str = Field(..., description="Error message")
    error_type: str = Field(..., description="Error type/category")
    timestamp: datetime = Field(default_factory=datetime.now, description="Failure timestamp")


class SessionCancelledEvent(BaseModel):
    """Event sent when session is cancelled."""
    session_id: str = Field(..., description="Session identifier")
    timestamp: datetime = Field(default_factory=datetime.now, description="Cancellation timestamp")


# Notification Events

class NotificationEvent(BaseModel):
    """Event for general notifications."""
    id: str = Field(..., description="Notification ID")
    type: NotificationTypeEnum = Field(..., description="Notification type")
    title: str = Field(..., description="Notification title")
    message: str = Field(..., description="Notification message")
    timestamp: datetime = Field(..., description="Notification timestamp")
    session_id: Optional[str] = Field(None, description="Related session ID")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class AlertEvent(BaseModel):
    """Event for system alerts."""
    id: str = Field(..., description="Alert ID")
    severity: AlertSeverityEnum = Field(..., description="Alert severity")
    title: str = Field(..., description="Alert title")
    description: str = Field(..., description="Alert description")
    timestamp: datetime = Field(..., description="Alert timestamp")
    session_id: Optional[str] = Field(None, description="Related session ID")
    resolved: bool = Field(False, description="Whether alert is resolved")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


# System Events

class SystemMetrics(BaseModel):
    """System metrics data."""
    cpu_usage_percent: float = Field(..., ge=0.0, le=100.0, description="CPU usage")
    memory_usage_percent: float = Field(..., ge=0.0, le=100.0, description="Memory usage")
    disk_usage_percent: float = Field(..., ge=0.0, le=100.0, description="Disk usage")
    gpu_usage_percent: Optional[float] = Field(None, ge=0.0, le=100.0, description="GPU usage")
    active_sessions: int = Field(..., ge=0, description="Number of active sessions")
    total_models: int = Field(..., ge=0, description="Total models in system")


class SystemStatusEvent(BaseModel):
    """Event for system status updates."""
    status: Dict[str, Any] = Field(..., description="System status data")
    timestamp: datetime = Field(default_factory=datetime.now, description="Status timestamp")


# Subscription Events

class SubscribeSessionRequest(BaseModel):
    """Request to subscribe to session updates."""
    session_id: str = Field(..., description="Session ID to subscribe to")


class UnsubscribeSessionRequest(BaseModel):
    """Request to unsubscribe from session updates."""
    session_id: str = Field(..., description="Session ID to unsubscribe from")


class SubscribedEvent(BaseModel):
    """Event confirming subscription."""
    session_id: str = Field(..., description="Subscribed session ID")
    timestamp: datetime = Field(default_factory=datetime.now, description="Subscription timestamp")


class UnsubscribedEvent(BaseModel):
    """Event confirming unsubscription."""
    session_id: str = Field(..., description="Unsubscribed session ID")
    timestamp: datetime = Field(default_factory=datetime.now, description="Unsubscription timestamp")


# Health Check Events

class PingEvent(BaseModel):
    """Ping event for connection health check."""
    timestamp: datetime = Field(default_factory=datetime.now, description="Ping timestamp")


class PongEvent(BaseModel):
    """Pong response for connection health check."""
    timestamp: datetime = Field(default_factory=datetime.now, description="Pong timestamp")


# Event Documentation

EVENT_DOCUMENTATION = {
    EventType.CONNECTED: {
        "description": "Sent when client successfully connects to WebSocket server",
        "direction": "server_to_client",
        "schema": ConnectedEvent,
        "example": {
            "sid": "abc123",
            "timestamp": "2024-01-01T12:00:00Z",
            "message": "Connected to optimization platform"
        }
    },
    EventType.SESSION_STARTED: {
        "description": "Sent when a new optimization session starts",
        "direction": "server_to_client",
        "schema": SessionStartedEvent,
        "example": {
            "session_id": "session_123",
            "model_id": "model_456",
            "model_name": "robotics_vla_model.pt",
            "techniques": ["quantization", "pruning"],
            "timestamp": "2024-01-01T12:00:00Z"
        }
    },
    EventType.SESSION_PROGRESS: {
        "description": "Sent periodically during optimization with progress updates",
        "direction": "server_to_client",
        "schema": SessionProgressEvent,
        "example": {
            "session_id": "session_123",
            "current_step": 5,
            "total_steps": 10,
            "step_name": "Applying quantization",
            "progress_percentage": 50.0,
            "estimated_completion": "2024-01-01T12:30:00Z",
            "elapsed_time": "0:15:00",
            "remaining_time": "0:15:00",
            "metadata": {}
        }
    },
    EventType.SESSION_COMPLETED: {
        "description": "Sent when optimization session completes successfully",
        "direction": "server_to_client",
        "schema": SessionCompletedEvent,
        "example": {
            "session_id": "session_123",
            "results": {
                "size_reduction_percent": 25.0,
                "speed_improvement_percent": 15.0,
                "accuracy_change_percent": -0.5,
                "techniques_applied": ["quantization", "pruning"]
            },
            "timestamp": "2024-01-01T12:30:00Z"
        }
    },
    EventType.SESSION_FAILED: {
        "description": "Sent when optimization session fails",
        "direction": "server_to_client",
        "schema": SessionFailedEvent,
        "example": {
            "session_id": "session_123",
            "error_message": "Model validation failed",
            "error_type": "ValidationError",
            "timestamp": "2024-01-01T12:15:00Z"
        }
    },
    EventType.SESSION_CANCELLED: {
        "description": "Sent when optimization session is cancelled by user",
        "direction": "server_to_client",
        "schema": SessionCancelledEvent,
        "example": {
            "session_id": "session_123",
            "timestamp": "2024-01-01T12:10:00Z"
        }
    },
    EventType.NOTIFICATION: {
        "description": "General notification message",
        "direction": "server_to_client",
        "schema": NotificationEvent,
        "example": {
            "id": "notif_1",
            "type": "info",
            "title": "Optimization Started",
            "message": "Your optimization session has begun",
            "timestamp": "2024-01-01T12:00:00Z",
            "session_id": "session_123",
            "metadata": {}
        }
    },
    EventType.ALERT: {
        "description": "System alert for important events or issues",
        "direction": "server_to_client",
        "schema": AlertEvent,
        "example": {
            "id": "alert_1",
            "severity": "high",
            "title": "High Memory Usage",
            "description": "System memory usage exceeds 90%",
            "timestamp": "2024-01-01T12:00:00Z",
            "session_id": None,
            "resolved": False,
            "metadata": {}
        }
    },
    EventType.SYSTEM_STATUS: {
        "description": "System status and health information",
        "direction": "server_to_client",
        "schema": SystemStatusEvent,
        "example": {
            "status": {
                "active_sessions": 3,
                "total_notifications": 10,
                "total_alerts": 2
            },
            "timestamp": "2024-01-01T12:00:00Z"
        }
    },
    "subscribe_session": {
        "description": "Subscribe to updates for a specific session",
        "direction": "client_to_server",
        "schema": SubscribeSessionRequest,
        "example": {
            "session_id": "session_123"
        }
    },
    "unsubscribe_session": {
        "description": "Unsubscribe from updates for a specific session",
        "direction": "client_to_server",
        "schema": UnsubscribeSessionRequest,
        "example": {
            "session_id": "session_123"
        }
    },
    EventType.SUBSCRIBED: {
        "description": "Confirmation of successful subscription",
        "direction": "server_to_client",
        "schema": SubscribedEvent,
        "example": {
            "session_id": "session_123",
            "timestamp": "2024-01-01T12:00:00Z"
        }
    },
    EventType.UNSUBSCRIBED: {
        "description": "Confirmation of successful unsubscription",
        "direction": "server_to_client",
        "schema": UnsubscribedEvent,
        "example": {
            "session_id": "session_123",
            "timestamp": "2024-01-01T12:00:00Z"
        }
    },
    EventType.PING: {
        "description": "Health check ping from client",
        "direction": "client_to_server",
        "schema": PingEvent,
        "example": {
            "timestamp": "2024-01-01T12:00:00Z"
        }
    },
    EventType.PONG: {
        "description": "Health check pong response from server",
        "direction": "server_to_client",
        "schema": PongEvent,
        "example": {
            "timestamp": "2024-01-01T12:00:00Z"
        }
    }
}


def get_event_schema(event_type: str) -> Optional[type[BaseModel]]:
    """
    Get Pydantic schema for an event type.
    
    Args:
        event_type: Event type name
        
    Returns:
        Pydantic model class or None if not found
    """
    doc = EVENT_DOCUMENTATION.get(event_type)
    return doc.get("schema") if doc else None


def get_event_example(event_type: str) -> Optional[Dict[str, Any]]:
    """
    Get example payload for an event type.
    
    Args:
        event_type: Event type name
        
    Returns:
        Example payload dictionary or None if not found
    """
    doc = EVENT_DOCUMENTATION.get(event_type)
    return doc.get("example") if doc else None


def validate_event(event_type: str, data: Dict[str, Any]) -> bool:
    """
    Validate event data against schema.
    
    Args:
        event_type: Event type name
        data: Event data to validate
        
    Returns:
        True if valid, False otherwise
    """
    schema = get_event_schema(event_type)
    if not schema:
        return False
    
    try:
        schema(**data)
        return True
    except Exception:
        return False

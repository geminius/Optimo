"""
Notification Service for real-time status updates and alerts.

This module provides comprehensive notification and monitoring capabilities
for the robotics model optimization platform.
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from threading import Lock
import json

logger = logging.getLogger(__name__)


class NotificationType(Enum):
    """Types of notifications that can be sent."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    SUCCESS = "success"
    PROGRESS = "progress"


class AlertSeverity(Enum):
    """Severity levels for alerts."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class Notification:
    """Represents a notification message."""
    id: str
    type: NotificationType
    title: str
    message: str
    timestamp: datetime
    session_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Alert:
    """Represents an alert for failures or performance issues."""
    id: str
    severity: AlertSeverity
    title: str
    description: str
    timestamp: datetime
    session_id: Optional[str] = None
    resolved: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProgressUpdate:
    """Represents a progress update with completion estimates."""
    session_id: str
    current_step: int
    total_steps: int
    step_name: str
    progress_percentage: float
    estimated_completion: Optional[datetime] = None
    elapsed_time: Optional[timedelta] = None
    remaining_time: Optional[timedelta] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class NotificationService:
    """
    Service for managing notifications, alerts, and progress tracking.
    
    Provides real-time status updates, alert mechanisms, and progress tracking
    with estimated completion times for optimization sessions.
    """
    
    def __init__(self):
        self._subscribers: Dict[str, List[Callable]] = {
            'notification': [],
            'alert': [],
            'progress': []
        }
        self._notifications: List[Notification] = []
        self._alerts: List[Alert] = []
        self._progress_sessions: Dict[str, ProgressUpdate] = {}
        self._session_start_times: Dict[str, datetime] = {}
        self._lock = Lock()
        self._notification_counter = 0
        self._alert_counter = 0
        
        # Configure logging
        self._setup_logging()
    
    def _setup_logging(self):
        """Set up comprehensive logging configuration."""
        # Create formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        
        # Create handlers
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(detailed_formatter)
        
        # Configure logger
        logger.setLevel(logging.DEBUG)
        logger.addHandler(console_handler)
    
    def subscribe(self, event_type: str, callback: Callable):
        """
        Subscribe to notification events.
        
        Args:
            event_type: Type of event ('notification', 'alert', 'progress')
            callback: Function to call when event occurs
        """
        if event_type not in self._subscribers:
            raise ValueError(f"Invalid event type: {event_type}")
        
        with self._lock:
            self._subscribers[event_type].append(callback)
        
        logger.debug(f"Subscribed to {event_type} events")
    
    def unsubscribe(self, event_type: str, callback: Callable):
        """
        Unsubscribe from notification events.
        
        Args:
            event_type: Type of event to unsubscribe from
            callback: Callback function to remove
        """
        if event_type not in self._subscribers:
            return
        
        with self._lock:
            if callback in self._subscribers[event_type]:
                self._subscribers[event_type].remove(callback)
        
        logger.debug(f"Unsubscribed from {event_type} events") 
   
    def send_notification(
        self,
        type: NotificationType,
        title: str,
        message: str,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Send a notification.
        
        Args:
            type: Type of notification
            title: Notification title
            message: Notification message
            session_id: Optional session ID
            metadata: Optional metadata
            
        Returns:
            Notification ID
        """
        with self._lock:
            self._notification_counter += 1
            notification_id = f"notif_{self._notification_counter}"
        
        notification = Notification(
            id=notification_id,
            type=type,
            title=title,
            message=message,
            timestamp=datetime.now(),
            session_id=session_id,
            metadata=metadata or {}
        )
        
        with self._lock:
            self._notifications.append(notification)
        
        # Log the notification
        log_level = {
            NotificationType.INFO: logging.INFO,
            NotificationType.WARNING: logging.WARNING,
            NotificationType.ERROR: logging.ERROR,
            NotificationType.SUCCESS: logging.INFO,
            NotificationType.PROGRESS: logging.DEBUG
        }.get(type, logging.INFO)
        
        logger.log(log_level, f"Notification [{type.value}]: {title} - {message}")
        
        # Notify subscribers
        self._notify_subscribers('notification', notification)
        
        return notification_id
    
    def create_alert(
        self,
        severity: AlertSeverity,
        title: str,
        description: str,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create an alert for failures or performance issues.
        
        Args:
            severity: Alert severity level
            title: Alert title
            description: Alert description
            session_id: Optional session ID
            metadata: Optional metadata
            
        Returns:
            Alert ID
        """
        with self._lock:
            self._alert_counter += 1
            alert_id = f"alert_{self._alert_counter}"
        
        alert = Alert(
            id=alert_id,
            severity=severity,
            title=title,
            description=description,
            timestamp=datetime.now(),
            session_id=session_id,
            metadata=metadata or {}
        )
        
        with self._lock:
            self._alerts.append(alert)
        
        # Log the alert
        log_level = {
            AlertSeverity.LOW: logging.INFO,
            AlertSeverity.MEDIUM: logging.WARNING,
            AlertSeverity.HIGH: logging.ERROR,
            AlertSeverity.CRITICAL: logging.CRITICAL
        }.get(severity, logging.WARNING)
        
        logger.log(log_level, f"Alert [{severity.value}]: {title} - {description}")
        
        # Notify subscribers
        self._notify_subscribers('alert', alert)
        
        return alert_id
    
    def resolve_alert(self, alert_id: str) -> bool:
        """
        Mark an alert as resolved.
        
        Args:
            alert_id: ID of the alert to resolve
            
        Returns:
            True if alert was found and resolved, False otherwise
        """
        with self._lock:
            for alert in self._alerts:
                if alert.id == alert_id:
                    alert.resolved = True
                    logger.info(f"Alert {alert_id} resolved: {alert.title}")
                    return True
        
        logger.warning(f"Alert {alert_id} not found")
        return False
    
    def start_progress_tracking(
        self,
        session_id: str,
        total_steps: int,
        initial_step_name: str = "Starting..."
    ):
        """
        Start progress tracking for a session.
        
        Args:
            session_id: Unique session identifier
            total_steps: Total number of steps in the process
            initial_step_name: Name of the initial step
        """
        start_time = datetime.now()
        
        progress = ProgressUpdate(
            session_id=session_id,
            current_step=0,
            total_steps=total_steps,
            step_name=initial_step_name,
            progress_percentage=0.0,
            estimated_completion=None,
            elapsed_time=timedelta(0),
            remaining_time=None
        )
        
        with self._lock:
            self._progress_sessions[session_id] = progress
            self._session_start_times[session_id] = start_time
        
        logger.info(f"Started progress tracking for session {session_id} with {total_steps} steps")
        self._notify_subscribers('progress', progress)
    
    def update_progress(
        self,
        session_id: str,
        current_step: int,
        step_name: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Update progress for a session.
        
        Args:
            session_id: Session identifier
            current_step: Current step number (0-based)
            step_name: Name of the current step
            metadata: Optional metadata for the progress update
        """
        if session_id not in self._progress_sessions:
            logger.warning(f"Progress session {session_id} not found")
            return
        
        with self._lock:
            progress = self._progress_sessions[session_id]
            start_time = self._session_start_times[session_id]
            
            # Update progress
            progress.current_step = current_step
            progress.step_name = step_name
            progress.progress_percentage = (current_step / progress.total_steps) * 100
            progress.metadata.update(metadata or {})
            
            # Calculate time estimates
            now = datetime.now()
            progress.elapsed_time = now - start_time
            
            if current_step > 0:
                avg_time_per_step = progress.elapsed_time / current_step
                remaining_steps = progress.total_steps - current_step
                progress.remaining_time = avg_time_per_step * remaining_steps
                progress.estimated_completion = now + progress.remaining_time
        
        logger.debug(
            f"Progress update for {session_id}: Step {current_step}/{progress.total_steps} "
            f"({progress.progress_percentage:.1f}%) - {step_name}"
        )
        
        self._notify_subscribers('progress', progress)
    
    def complete_progress_tracking(self, session_id: str):
        """
        Complete progress tracking for a session.
        
        Args:
            session_id: Session identifier
        """
        if session_id not in self._progress_sessions:
            logger.warning(f"Progress session {session_id} not found")
            return
        
        with self._lock:
            progress = self._progress_sessions[session_id]
            progress.current_step = progress.total_steps
            progress.progress_percentage = 100.0
            progress.step_name = "Completed"
            
            # Calculate final elapsed time
            start_time = self._session_start_times[session_id]
            progress.elapsed_time = datetime.now() - start_time
            progress.remaining_time = timedelta(0)
            progress.estimated_completion = datetime.now()
        
        logger.info(f"Completed progress tracking for session {session_id}")
        self._notify_subscribers('progress', progress)
        
        # Clean up
        with self._lock:
            del self._progress_sessions[session_id]
            del self._session_start_times[session_id]
    
    def get_progress(self, session_id: str) -> Optional[ProgressUpdate]:
        """
        Get current progress for a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Current progress update or None if session not found
        """
        with self._lock:
            return self._progress_sessions.get(session_id)
    
    def get_notifications(
        self,
        session_id: Optional[str] = None,
        type_filter: Optional[NotificationType] = None,
        limit: Optional[int] = None
    ) -> List[Notification]:
        """
        Get notifications with optional filtering.
        
        Args:
            session_id: Filter by session ID
            type_filter: Filter by notification type
            limit: Maximum number of notifications to return
            
        Returns:
            List of notifications
        """
        with self._lock:
            notifications = self._notifications.copy()
        
        # Apply filters
        if session_id:
            notifications = [n for n in notifications if n.session_id == session_id]
        
        if type_filter:
            notifications = [n for n in notifications if n.type == type_filter]
        
        # Sort by timestamp (newest first)
        notifications.sort(key=lambda n: n.timestamp, reverse=True)
        
        # Apply limit
        if limit:
            notifications = notifications[:limit]
        
        return notifications
    
    def get_alerts(
        self,
        session_id: Optional[str] = None,
        severity_filter: Optional[AlertSeverity] = None,
        resolved_filter: Optional[bool] = None,
        limit: Optional[int] = None
    ) -> List[Alert]:
        """
        Get alerts with optional filtering.
        
        Args:
            session_id: Filter by session ID
            severity_filter: Filter by severity level
            resolved_filter: Filter by resolved status
            limit: Maximum number of alerts to return
            
        Returns:
            List of alerts
        """
        with self._lock:
            alerts = self._alerts.copy()
        
        # Apply filters
        if session_id:
            alerts = [a for a in alerts if a.session_id == session_id]
        
        if severity_filter:
            alerts = [a for a in alerts if a.severity == severity_filter]
        
        if resolved_filter is not None:
            alerts = [a for a in alerts if a.resolved == resolved_filter]
        
        # Sort by timestamp (newest first)
        alerts.sort(key=lambda a: a.timestamp, reverse=True)
        
        # Apply limit
        if limit:
            alerts = alerts[:limit]
        
        return alerts
    
    def _notify_subscribers(self, event_type: str, data: Any):
        """
        Notify all subscribers of an event.
        
        Args:
            event_type: Type of event
            data: Event data
        """
        with self._lock:
            subscribers = self._subscribers[event_type].copy()
        
        for callback in subscribers:
            try:
                callback(data)
            except Exception as e:
                logger.error(f"Error notifying subscriber: {e}")
    
    def clear_notifications(self, session_id: Optional[str] = None):
        """
        Clear notifications, optionally filtered by session ID.
        
        Args:
            session_id: Optional session ID to filter by
        """
        with self._lock:
            if session_id:
                self._notifications = [
                    n for n in self._notifications 
                    if n.session_id != session_id
                ]
            else:
                self._notifications.clear()
        
        logger.info(f"Cleared notifications{' for session ' + session_id if session_id else ''}")
    
    def clear_alerts(self, session_id: Optional[str] = None):
        """
        Clear alerts, optionally filtered by session ID.
        
        Args:
            session_id: Optional session ID to filter by
        """
        with self._lock:
            if session_id:
                self._alerts = [
                    a for a in self._alerts 
                    if a.session_id != session_id
                ]
            else:
                self._alerts.clear()
        
        logger.info(f"Cleared alerts{' for session ' + session_id if session_id else ''}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Get overall system status and statistics.
        
        Returns:
            Dictionary containing system status information
        """
        with self._lock:
            active_sessions = len(self._progress_sessions)
            total_notifications = len(self._notifications)
            total_alerts = len(self._alerts)
            unresolved_alerts = len([a for a in self._alerts if not a.resolved])
            
            # Get alert counts by severity
            alert_counts = {}
            for severity in AlertSeverity:
                alert_counts[severity.value] = len([
                    a for a in self._alerts 
                    if a.severity == severity and not a.resolved
                ])
        
        return {
            'active_sessions': active_sessions,
            'total_notifications': total_notifications,
            'total_alerts': total_alerts,
            'unresolved_alerts': unresolved_alerts,
            'alert_counts_by_severity': alert_counts,
            'timestamp': datetime.now().isoformat()
        }
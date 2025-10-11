"""
WebSocket Manager for real-time updates.

This module provides WebSocket connection management and event broadcasting
for the robotics model optimization platform using Socket.IO.
"""

import asyncio
import logging
from typing import Dict, Set, Optional, Any, Callable
from datetime import datetime
from threading import Lock
import socketio

from .notification_service import (
    NotificationService, 
    Notification, 
    Alert, 
    ProgressUpdate
)

logger = logging.getLogger(__name__)


class WebSocketManager:
    """
    Manages WebSocket connections and event broadcasting.
    
    Provides real-time updates to connected clients through Socket.IO,
    integrating with the NotificationService for event propagation.
    """
    
    _instance = None
    
    def __new__(cls):
        """Singleton pattern to ensure single instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize WebSocket manager."""
        if self._initialized:
            return
        
        # Create Socket.IO server with async mode
        self.sio = socketio.AsyncServer(
            async_mode='asgi',
            cors_allowed_origins='*',  # Configure appropriately for production
            logger=False,
            engineio_logger=False,
            ping_timeout=60,
            ping_interval=25
        )
        
        # Track connections and subscriptions
        self._connections: Dict[str, Dict[str, Any]] = {}  # sid -> connection info
        self._session_subscriptions: Dict[str, Set[str]] = {}  # session_id -> set of sids
        self._lock = Lock()
        
        # Notification service reference
        self._notification_service: Optional[NotificationService] = None
        
        # Metrics tracking
        from ..api.middleware import WebSocketMetricsMiddleware
        self._metrics = WebSocketMetricsMiddleware()
        
        # Register Socket.IO event handlers
        self._register_handlers()
        
        self._initialized = True
        logger.info("WebSocketManager initialized")
    
    def _register_handlers(self):
        """Register Socket.IO event handlers."""
        
        @self.sio.event
        async def connect(sid, environ, auth):
            """Handle client connection."""
            await self._handle_connect(sid, environ, auth)
        
        @self.sio.event
        async def disconnect(sid):
            """Handle client disconnection."""
            await self._handle_disconnect(sid)
        
        @self.sio.event
        async def subscribe_session(sid, data):
            """Handle session subscription request."""
            await self._handle_subscribe_session(sid, data)
        
        @self.sio.event
        async def unsubscribe_session(sid, data):
            """Handle session unsubscription request."""
            await self._handle_unsubscribe_session(sid, data)
        
        @self.sio.event
        async def ping(sid):
            """Handle ping for connection health check."""
            await self.sio.emit('pong', {'timestamp': datetime.now().isoformat()}, room=sid)
    
    async def _handle_connect(self, sid: str, environ: Dict, auth: Optional[Dict]):
        """
        Handle new client connection with authentication.
        
        Args:
            sid: Socket.IO session ID
            environ: WSGI environment
            auth: Authentication data containing token
        """
        # Validate authentication token
        user = None
        if auth and 'token' in auth:
            from ..api.auth import get_auth_manager
            auth_manager = get_auth_manager()
            user = auth_manager.verify_websocket_token(auth['token'])
            
            if not user:
                logger.warning(
                    f"WebSocket connection rejected: invalid token",
                    extra={
                        "component": "WebSocketManager",
                        "sid": sid
                    }
                )
                # Reject connection
                return False
        else:
            logger.warning(
                f"WebSocket connection rejected: no authentication token",
                extra={
                    "component": "WebSocketManager",
                    "sid": sid
                }
            )
            # Reject connection
            return False
        
        with self._lock:
            self._connections[sid] = {
                'connected_at': datetime.now(),
                'user_id': user.id,
                'username': user.username,
                'role': user.role,
                'subscriptions': set()
            }
        
        # Track metrics
        self._metrics.on_connect(sid)
        
        logger.info(
            f"Client connected: {sid}",
            extra={
                "component": "WebSocketManager",
                "sid": sid,
                "user_id": user.id,
                "username": user.username,
                "role": user.role
            }
        )
        
        # Send connection confirmation
        await self.sio.emit('connected', {
            'sid': sid,
            'timestamp': datetime.now().isoformat(),
            'message': 'Connected to optimization platform',
            'user': {
                'id': user.id,
                'username': user.username,
                'role': user.role
            }
        }, room=sid)
    
    async def _handle_disconnect(self, sid: str):
        """
        Handle client disconnection.
        
        Args:
            sid: Socket.IO session ID
        """
        # Clean up subscriptions
        with self._lock:
            if sid in self._connections:
                subscriptions = self._connections[sid].get('subscriptions', set())
                
                # Remove from session subscriptions
                for session_id in subscriptions:
                    if session_id in self._session_subscriptions:
                        self._session_subscriptions[session_id].discard(sid)
                        if not self._session_subscriptions[session_id]:
                            del self._session_subscriptions[session_id]
                
                # Remove connection
                del self._connections[sid]
        
        # Track metrics
        self._metrics.on_disconnect(sid)
        
        logger.info(f"Client disconnected: {sid}")
    
    async def _handle_subscribe_session(self, sid: str, data: Dict):
        """
        Handle session subscription request.
        
        Args:
            sid: Socket.IO session ID
            data: Subscription data containing session_id
        """
        session_id = data.get('session_id')
        if not session_id:
            await self.sio.emit('error', {
                'message': 'session_id required for subscription'
            }, room=sid)
            return
        
        with self._lock:
            # Add to connection subscriptions
            if sid in self._connections:
                self._connections[sid]['subscriptions'].add(session_id)
            
            # Add to session subscriptions
            if session_id not in self._session_subscriptions:
                self._session_subscriptions[session_id] = set()
            self._session_subscriptions[session_id].add(sid)
        
        logger.info(f"Client {sid} subscribed to session {session_id}")
        
        await self.sio.emit('subscribed', {
            'session_id': session_id,
            'timestamp': datetime.now().isoformat()
        }, room=sid)
    
    async def _handle_unsubscribe_session(self, sid: str, data: Dict):
        """
        Handle session unsubscription request.
        
        Args:
            sid: Socket.IO session ID
            data: Unsubscription data containing session_id
        """
        session_id = data.get('session_id')
        if not session_id:
            return
        
        with self._lock:
            # Remove from connection subscriptions
            if sid in self._connections:
                self._connections[sid]['subscriptions'].discard(session_id)
            
            # Remove from session subscriptions
            if session_id in self._session_subscriptions:
                self._session_subscriptions[session_id].discard(sid)
                if not self._session_subscriptions[session_id]:
                    del self._session_subscriptions[session_id]
        
        logger.info(f"Client {sid} unsubscribed from session {session_id}")
        
        await self.sio.emit('unsubscribed', {
            'session_id': session_id,
            'timestamp': datetime.now().isoformat()
        }, room=sid)
    
    def setup_notification_handlers(self, notification_service: NotificationService):
        """
        Set up handlers for notification service events.
        
        Subscribes to all NotificationService events and transforms them
        into WebSocket events for connected clients.
        
        Args:
            notification_service: NotificationService instance to subscribe to
        """
        self._notification_service = notification_service
        
        # Subscribe to notification events
        notification_service.subscribe('notification', self._handle_notification)
        notification_service.subscribe('alert', self._handle_alert)
        notification_service.subscribe('progress', self._handle_progress)
        
        logger.info(
            "WebSocketManager connected to NotificationService",
            extra={
                "component": "WebSocketManager",
                "event": "notification_service_connected"
            }
        )
    
    def _handle_notification(self, notification: Notification):
        """
        Handle notification from NotificationService.
        
        Args:
            notification: Notification object
        """
        asyncio.create_task(self._broadcast_notification(notification))
    
    def _handle_alert(self, alert: Alert):
        """
        Handle alert from NotificationService.
        
        Args:
            alert: Alert object
        """
        asyncio.create_task(self._broadcast_alert(alert))
    
    def _handle_progress(self, progress: ProgressUpdate):
        """
        Handle progress update from NotificationService.
        
        Args:
            progress: ProgressUpdate object
        """
        asyncio.create_task(self._broadcast_progress(progress))
    
    async def _broadcast_notification(self, notification: Notification):
        """
        Broadcast notification to relevant clients.
        
        Transforms NotificationService Notification objects into WebSocket
        notification events and broadcasts to appropriate clients.
        
        Args:
            notification: Notification to broadcast
        """
        event_data = {
            'id': notification.id,
            'type': notification.type.value,
            'title': notification.title,
            'message': notification.message,
            'timestamp': notification.timestamp.isoformat(),
            'session_id': notification.session_id,
            'metadata': notification.metadata
        }
        
        try:
            if notification.session_id:
                # Broadcast to session subscribers
                await self._broadcast_to_session(
                    notification.session_id,
                    'notification',
                    event_data
                )
                logger.debug(
                    f"Broadcasted notification to session subscribers",
                    extra={
                        "component": "WebSocketManager",
                        "notification_id": notification.id,
                        "session_id": notification.session_id,
                        "type": notification.type.value
                    }
                )
            else:
                # Broadcast to all connected clients
                await self.sio.emit('notification', event_data)
                logger.debug(
                    f"Broadcasted notification to all clients",
                    extra={
                        "component": "WebSocketManager",
                        "notification_id": notification.id,
                        "type": notification.type.value
                    }
                )
        except Exception as e:
            logger.error(
                f"Failed to broadcast notification: {e}",
                extra={
                    "component": "WebSocketManager",
                    "notification_id": notification.id,
                    "error": str(e)
                },
                exc_info=True
            )
    
    async def _broadcast_alert(self, alert: Alert):
        """
        Broadcast alert to relevant clients.
        
        Transforms NotificationService Alert objects into WebSocket
        alert events and broadcasts to appropriate clients.
        
        Args:
            alert: Alert to broadcast
        """
        event_data = {
            'id': alert.id,
            'severity': alert.severity.value,
            'title': alert.title,
            'description': alert.description,
            'timestamp': alert.timestamp.isoformat(),
            'session_id': alert.session_id,
            'resolved': alert.resolved,
            'metadata': alert.metadata
        }
        
        try:
            if alert.session_id:
                # Broadcast to session subscribers
                await self._broadcast_to_session(
                    alert.session_id,
                    'alert',
                    event_data
                )
                logger.debug(
                    f"Broadcasted alert to session subscribers",
                    extra={
                        "component": "WebSocketManager",
                        "alert_id": alert.id,
                        "session_id": alert.session_id,
                        "severity": alert.severity.value
                    }
                )
            else:
                # Broadcast to all connected clients
                await self.sio.emit('alert', event_data)
                logger.debug(
                    f"Broadcasted alert to all clients",
                    extra={
                        "component": "WebSocketManager",
                        "alert_id": alert.id,
                        "severity": alert.severity.value
                    }
                )
        except Exception as e:
            logger.error(
                f"Failed to broadcast alert: {e}",
                extra={
                    "component": "WebSocketManager",
                    "alert_id": alert.id,
                    "error": str(e)
                },
                exc_info=True
            )
    
    async def _broadcast_progress(self, progress: ProgressUpdate):
        """
        Broadcast progress update to session subscribers.
        
        Transforms NotificationService ProgressUpdate objects into WebSocket
        progress_update events and broadcasts to session subscribers.
        
        Args:
            progress: ProgressUpdate to broadcast
        """
        event_data = {
            'session_id': progress.session_id,
            'current_step': progress.current_step,
            'total_steps': progress.total_steps,
            'step_name': progress.step_name,
            'progress_percentage': progress.progress_percentage,
            'estimated_completion': progress.estimated_completion.isoformat() if progress.estimated_completion else None,
            'elapsed_time': str(progress.elapsed_time) if progress.elapsed_time else None,
            'remaining_time': str(progress.remaining_time) if progress.remaining_time else None,
            'metadata': progress.metadata
        }
        
        try:
            # Broadcast to session subscribers
            await self._broadcast_to_session(
                progress.session_id,
                'progress_update',
                event_data
            )
            
            logger.debug(
                f"Broadcasted progress update",
                extra={
                    "component": "WebSocketManager",
                    "session_id": progress.session_id,
                    "progress_percentage": progress.progress_percentage,
                    "step_name": progress.step_name
                }
            )
        except Exception as e:
            logger.error(
                f"Failed to broadcast progress update: {e}",
                extra={
                    "component": "WebSocketManager",
                    "session_id": progress.session_id,
                    "error": str(e)
                },
                exc_info=True
            )
    
    async def _broadcast_to_session(self, session_id: str, event: str, data: Dict):
        """
        Broadcast event to all clients subscribed to a session.
        
        Handles event delivery failures gracefully by logging errors and
        continuing to deliver to other subscribers.
        
        Args:
            session_id: Session ID
            event: Event name
            data: Event data
        """
        with self._lock:
            subscribers = self._session_subscriptions.get(session_id, set()).copy()
        
        if not subscribers:
            logger.debug(
                f"No subscribers for session",
                extra={
                    "component": "WebSocketManager",
                    "session_id": session_id,
                    "event": event
                }
            )
            return
        
        # Track delivery statistics
        successful_deliveries = 0
        failed_deliveries = 0
        
        # Emit to each subscriber
        for sid in subscribers:
            try:
                await self.sio.emit(event, data, room=sid)
                successful_deliveries += 1
            except Exception as e:
                failed_deliveries += 1
                logger.error(
                    f"Failed to emit event to client",
                    extra={
                        "component": "WebSocketManager",
                        "session_id": session_id,
                        "event": event,
                        "sid": sid,
                        "error": str(e)
                    },
                    exc_info=True
                )
        
        # Log delivery summary
        logger.debug(
            f"Event broadcast complete",
            extra={
                "component": "WebSocketManager",
                "session_id": session_id,
                "event": event,
                "total_subscribers": len(subscribers),
                "successful_deliveries": successful_deliveries,
                "failed_deliveries": failed_deliveries
            }
        )
    
    async def broadcast_session_started(self, session_id: str, model_id: str, 
                                       model_name: str, techniques: list):
        """
        Broadcast session started event.
        
        Notifies all subscribers that a new optimization session has started.
        
        Args:
            session_id: Session ID
            model_id: Model ID
            model_name: Model name
            techniques: List of optimization techniques
        """
        event_data = {
            'session_id': session_id,
            'model_id': model_id,
            'model_name': model_name,
            'techniques': techniques,
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            await self._broadcast_to_session(session_id, 'session_started', event_data)
            logger.info(
                f"Broadcasted session_started event",
                extra={
                    "component": "WebSocketManager",
                    "session_id": session_id,
                    "model_id": model_id,
                    "techniques": techniques
                }
            )
        except Exception as e:
            logger.error(
                f"Failed to broadcast session_started: {e}",
                extra={
                    "component": "WebSocketManager",
                    "session_id": session_id,
                    "error": str(e)
                },
                exc_info=True
            )
    
    async def broadcast_session_completed(self, session_id: str, results: Dict):
        """
        Broadcast session completed event.
        
        Notifies all subscribers that an optimization session has completed successfully.
        
        Args:
            session_id: Session ID
            results: Optimization results
        """
        event_data = {
            'session_id': session_id,
            'results': results,
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            await self._broadcast_to_session(session_id, 'session_completed', event_data)
            logger.info(
                f"Broadcasted session_completed event",
                extra={
                    "component": "WebSocketManager",
                    "session_id": session_id,
                    "results_summary": {
                        k: v for k, v in results.items() 
                        if k in ['size_reduction_percent', 'speed_improvement_percent']
                    }
                }
            )
        except Exception as e:
            logger.error(
                f"Failed to broadcast session_completed: {e}",
                extra={
                    "component": "WebSocketManager",
                    "session_id": session_id,
                    "error": str(e)
                },
                exc_info=True
            )
    
    async def broadcast_session_failed(self, session_id: str, error_message: str, 
                                      error_type: str):
        """
        Broadcast session failed event.
        
        Notifies all subscribers that an optimization session has failed.
        
        Args:
            session_id: Session ID
            error_message: Error message
            error_type: Error type
        """
        event_data = {
            'session_id': session_id,
            'error_message': error_message,
            'error_type': error_type,
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            await self._broadcast_to_session(session_id, 'session_failed', event_data)
            logger.info(
                f"Broadcasted session_failed event",
                extra={
                    "component": "WebSocketManager",
                    "session_id": session_id,
                    "error_type": error_type,
                    "error_message": error_message
                }
            )
        except Exception as e:
            logger.error(
                f"Failed to broadcast session_failed: {e}",
                extra={
                    "component": "WebSocketManager",
                    "session_id": session_id,
                    "error": str(e)
                },
                exc_info=True
            )
    
    async def broadcast_session_cancelled(self, session_id: str):
        """
        Broadcast session cancelled event.
        
        Notifies all subscribers that an optimization session has been cancelled.
        
        Args:
            session_id: Session ID
        """
        event_data = {
            'session_id': session_id,
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            await self._broadcast_to_session(session_id, 'session_cancelled', event_data)
            logger.info(
                f"Broadcasted session_cancelled event",
                extra={
                    "component": "WebSocketManager",
                    "session_id": session_id
                }
            )
        except Exception as e:
            logger.error(
                f"Failed to broadcast session_cancelled: {e}",
                extra={
                    "component": "WebSocketManager",
                    "session_id": session_id,
                    "error": str(e)
                },
                exc_info=True
            )
    
    async def broadcast_system_status(self, status: Dict):
        """
        Broadcast system status to all connected clients.
        
        Sends system health and status information to all connected clients.
        
        Args:
            status: System status data
        """
        event_data = {
            'status': status,
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            await self.sio.emit('system_status', event_data)
            logger.debug(
                "Broadcasted system_status event",
                extra={
                    "component": "WebSocketManager",
                    "status_summary": {
                        k: v for k, v in status.items()
                        if k in ['active_sessions', 'total_alerts', 'unresolved_alerts']
                    }
                }
            )
        except Exception as e:
            logger.error(
                f"Failed to broadcast system_status: {e}",
                extra={
                    "component": "WebSocketManager",
                    "error": str(e)
                },
                exc_info=True
            )
    
    def get_connection_count(self) -> int:
        """
        Get number of connected clients.
        
        Returns:
            Number of connected clients
        """
        with self._lock:
            return len(self._connections)
    
    def get_session_subscriber_count(self, session_id: str) -> int:
        """
        Get number of subscribers for a session.
        
        Args:
            session_id: Session ID
            
        Returns:
            Number of subscribers
        """
        with self._lock:
            return len(self._session_subscriptions.get(session_id, set()))
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get WebSocket manager statistics.
        
        Returns:
            Dictionary containing statistics including connection metrics
        """
        with self._lock:
            total_connections = len(self._connections)
            total_sessions = len(self._session_subscriptions)
            
            # Calculate average subscriptions per connection
            total_subscriptions = sum(
                len(conn['subscriptions']) 
                for conn in self._connections.values()
            )
            avg_subscriptions = (
                total_subscriptions / total_connections 
                if total_connections > 0 else 0
            )
        
        # Get metrics from middleware
        metrics = self._metrics.get_metrics()
        
        return {
            'total_connections': total_connections,
            'total_sessions_with_subscribers': total_sessions,
            'total_subscriptions': total_subscriptions,
            'average_subscriptions_per_connection': avg_subscriptions,
            'notification_service_connected': self._notification_service is not None,
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics
        }
    
    def is_notification_service_connected(self) -> bool:
        """
        Check if NotificationService is connected.
        
        Returns:
            True if NotificationService is connected, False otherwise
        """
        return self._notification_service is not None
    
    def get_notification_service(self) -> Optional[NotificationService]:
        """
        Get the connected NotificationService instance.
        
        Returns:
            NotificationService instance or None if not connected
        """
        return self._notification_service

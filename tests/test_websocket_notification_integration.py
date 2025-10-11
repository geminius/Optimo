"""
Tests for WebSocket and NotificationService integration.

This module tests the integration between WebSocketManager and NotificationService,
ensuring that events are properly transformed and broadcast to connected clients.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch, MagicMock

from src.services.websocket_manager import WebSocketManager
from src.services.notification_service import (
    NotificationService,
    Notification,
    Alert,
    ProgressUpdate,
    NotificationType,
    AlertSeverity
)


@pytest.fixture
def notification_service():
    """Create a NotificationService instance."""
    return NotificationService()


@pytest.fixture
def websocket_manager():
    """Create a WebSocketManager instance."""
    manager = WebSocketManager()
    # Reset singleton state for testing
    manager._connections = {}
    manager._session_subscriptions = {}
    manager._notification_service = None
    return manager


@pytest.fixture
def mock_sio():
    """Create a mock Socket.IO server."""
    mock = AsyncMock()
    mock.emit = AsyncMock()
    return mock


class TestWebSocketNotificationIntegration:
    """Test WebSocket and NotificationService integration."""
    
    def test_setup_notification_handlers(self, websocket_manager, notification_service):
        """Test setting up notification handlers."""
        # Setup handlers
        websocket_manager.setup_notification_handlers(notification_service)
        
        # Verify connection
        assert websocket_manager.is_notification_service_connected()
        assert websocket_manager.get_notification_service() == notification_service
        
        # Verify subscribers were registered
        assert len(notification_service._subscribers['notification']) > 0
        assert len(notification_service._subscribers['alert']) > 0
        assert len(notification_service._subscribers['progress']) > 0
    
    @pytest.mark.asyncio
    async def test_notification_event_transformation(self, websocket_manager, notification_service, mock_sio):
        """Test that notifications are transformed and broadcast correctly."""
        # Replace sio with mock
        websocket_manager.sio = mock_sio
        
        # Setup handlers
        websocket_manager.setup_notification_handlers(notification_service)
        
        # Send a notification
        notification_id = notification_service.send_notification(
            type=NotificationType.INFO,
            title="Test Notification",
            message="This is a test",
            session_id=None
        )
        
        # Wait for async processing
        await asyncio.sleep(0.1)
        
        # Verify emit was called
        assert mock_sio.emit.called
        call_args = mock_sio.emit.call_args
        assert call_args[0][0] == 'notification'
        
        event_data = call_args[0][1]
        assert event_data['id'] == notification_id
        assert event_data['type'] == 'info'
        assert event_data['title'] == "Test Notification"
        assert event_data['message'] == "This is a test"
    
    @pytest.mark.asyncio
    async def test_alert_event_transformation(self, websocket_manager, notification_service, mock_sio):
        """Test that alerts are transformed and broadcast correctly."""
        # Replace sio with mock
        websocket_manager.sio = mock_sio
        
        # Setup handlers
        websocket_manager.setup_notification_handlers(notification_service)
        
        # Create an alert
        alert_id = notification_service.create_alert(
            severity=AlertSeverity.HIGH,
            title="Test Alert",
            description="This is a test alert",
            session_id=None
        )
        
        # Wait for async processing
        await asyncio.sleep(0.1)
        
        # Verify emit was called
        assert mock_sio.emit.called
        call_args = mock_sio.emit.call_args
        assert call_args[0][0] == 'alert'
        
        event_data = call_args[0][1]
        assert event_data['id'] == alert_id
        assert event_data['severity'] == 'high'
        assert event_data['title'] == "Test Alert"
        assert event_data['description'] == "This is a test alert"
    
    @pytest.mark.asyncio
    async def test_progress_event_transformation(self, websocket_manager, notification_service, mock_sio):
        """Test that progress updates are transformed and broadcast correctly."""
        # Replace sio with mock
        websocket_manager.sio = mock_sio
        
        # Setup handlers
        websocket_manager.setup_notification_handlers(notification_service)
        
        # Add a mock subscriber to the session
        session_id = "test_session_123"
        websocket_manager._session_subscriptions[session_id] = {"client_1"}
        
        # Start progress tracking
        notification_service.start_progress_tracking(
            session_id=session_id,
            total_steps=10,
            initial_step_name="Starting"
        )
        
        # Wait for async processing
        await asyncio.sleep(0.1)
        
        # Verify emit was called
        assert mock_sio.emit.called
        call_args = mock_sio.emit.call_args
        assert call_args[0][0] == 'progress_update'
        
        event_data = call_args[0][1]
        assert event_data['session_id'] == session_id
        assert event_data['current_step'] == 0
        assert event_data['total_steps'] == 10
        assert event_data['progress_percentage'] == 0.0
    
    @pytest.mark.asyncio
    async def test_session_specific_notification_broadcast(self, websocket_manager, notification_service, mock_sio):
        """Test that session-specific notifications are broadcast to subscribers only."""
        # Replace sio with mock
        websocket_manager.sio = mock_sio
        
        # Setup handlers
        websocket_manager.setup_notification_handlers(notification_service)
        
        # Add mock subscribers
        session_id = "test_session_123"
        websocket_manager._session_subscriptions[session_id] = {"client_1", "client_2"}
        
        # Send session-specific notification
        notification_service.send_notification(
            type=NotificationType.INFO,
            title="Session Notification",
            message="This is for a specific session",
            session_id=session_id
        )
        
        # Wait for async processing
        await asyncio.sleep(0.1)
        
        # Verify emit was called for each subscriber
        assert mock_sio.emit.called
        assert mock_sio.emit.call_count >= 2  # Once for each subscriber
    
    @pytest.mark.asyncio
    async def test_broadcast_error_handling(self, websocket_manager, notification_service, mock_sio):
        """Test that broadcast errors are handled gracefully."""
        # Replace sio with mock that raises an error
        mock_sio.emit = AsyncMock(side_effect=Exception("Network error"))
        websocket_manager.sio = mock_sio
        
        # Setup handlers
        websocket_manager.setup_notification_handlers(notification_service)
        
        # Send a notification (should not raise exception)
        try:
            notification_service.send_notification(
                type=NotificationType.ERROR,
                title="Test Error",
                message="This should handle errors gracefully"
            )
            
            # Wait for async processing
            await asyncio.sleep(0.1)
            
            # If we get here, error was handled gracefully
            assert True
        except Exception as e:
            pytest.fail(f"Broadcast error was not handled gracefully: {e}")
    
    @pytest.mark.asyncio
    async def test_multiple_event_types_in_sequence(self, websocket_manager, notification_service, mock_sio):
        """Test handling multiple event types in sequence."""
        # Replace sio with mock
        websocket_manager.sio = mock_sio
        
        # Setup handlers
        websocket_manager.setup_notification_handlers(notification_service)
        
        session_id = "test_session_123"
        websocket_manager._session_subscriptions[session_id] = {"client_1"}
        
        # Send multiple events
        notification_service.send_notification(
            type=NotificationType.INFO,
            title="Info",
            message="Info message"
        )
        
        notification_service.create_alert(
            severity=AlertSeverity.MEDIUM,
            title="Alert",
            description="Alert message"
        )
        
        notification_service.start_progress_tracking(
            session_id=session_id,
            total_steps=5
        )
        
        # Wait for async processing
        await asyncio.sleep(0.2)
        
        # Verify all events were emitted
        assert mock_sio.emit.call_count >= 3
        
        # Verify different event types were called
        event_types = [call[0][0] for call in mock_sio.emit.call_args_list]
        assert 'notification' in event_types
        assert 'alert' in event_types
        assert 'progress_update' in event_types
    
    def test_stats_include_notification_service_status(self, websocket_manager, notification_service):
        """Test that stats include NotificationService connection status."""
        # Before connection
        stats = websocket_manager.get_stats()
        assert stats['notification_service_connected'] is False
        
        # After connection
        websocket_manager.setup_notification_handlers(notification_service)
        stats = websocket_manager.get_stats()
        assert stats['notification_service_connected'] is True
    
    @pytest.mark.asyncio
    async def test_session_lifecycle_events(self, websocket_manager, mock_sio):
        """Test session lifecycle event broadcasting."""
        # Replace sio with mock
        websocket_manager.sio = mock_sio
        
        session_id = "test_session_123"
        websocket_manager._session_subscriptions[session_id] = {"client_1"}
        
        # Test session started
        await websocket_manager.broadcast_session_started(
            session_id=session_id,
            model_id="model_123",
            model_name="test_model.pt",
            techniques=["quantization", "pruning"]
        )
        
        assert mock_sio.emit.called
        assert mock_sio.emit.call_args[0][0] == 'session_started'
        
        # Test session completed
        mock_sio.emit.reset_mock()
        await websocket_manager.broadcast_session_completed(
            session_id=session_id,
            results={"size_reduction_percent": 25.0}
        )
        
        assert mock_sio.emit.called
        assert mock_sio.emit.call_args[0][0] == 'session_completed'
        
        # Test session failed
        mock_sio.emit.reset_mock()
        await websocket_manager.broadcast_session_failed(
            session_id=session_id,
            error_message="Test error",
            error_type="TestError"
        )
        
        assert mock_sio.emit.called
        assert mock_sio.emit.call_args[0][0] == 'session_failed'
        
        # Test session cancelled
        mock_sio.emit.reset_mock()
        await websocket_manager.broadcast_session_cancelled(session_id=session_id)
        
        assert mock_sio.emit.called
        assert mock_sio.emit.call_args[0][0] == 'session_cancelled'
    
    @pytest.mark.asyncio
    async def test_system_status_broadcast(self, websocket_manager, mock_sio):
        """Test system status broadcasting to all clients."""
        # Replace sio with mock
        websocket_manager.sio = mock_sio
        
        # Add some connections
        websocket_manager._connections = {
            "client_1": {"connected_at": datetime.now(), "subscriptions": set()},
            "client_2": {"connected_at": datetime.now(), "subscriptions": set()}
        }
        
        # Broadcast system status
        await websocket_manager.broadcast_system_status({
            "active_sessions": 5,
            "total_alerts": 2,
            "unresolved_alerts": 1
        })
        
        # Verify emit was called
        assert mock_sio.emit.called
        call_args = mock_sio.emit.call_args
        assert call_args[0][0] == 'system_status'
        
        event_data = call_args[0][1]
        assert event_data['status']['active_sessions'] == 5
        assert event_data['status']['total_alerts'] == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

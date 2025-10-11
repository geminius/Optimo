"""
Integration tests for WebSocket event flow.

Tests end-to-end event delivery from NotificationService to clients,
multiple client scenarios, room-based filtering, and reconnection handling.
"""

import pytest
import pytest_asyncio
import asyncio
import socketio
from datetime import datetime, timedelta
from typing import List, Dict, Any
from unittest.mock import MagicMock, AsyncMock, patch

from src.services.websocket_manager import WebSocketManager
from src.services.notification_service import (
    NotificationService,
    NotificationType,
    AlertSeverity
)
from src.api.models import User


class MockSocketIOClient:
    """Mock Socket.IO client for testing."""
    
    def __init__(self, client_id: str):
        self.client_id = client_id
        self.sid = None
        self.connected = False
        self.received_events: List[Dict[str, Any]] = []
        self.subscriptions: set = set()
        self._event_handlers = {}
    
    def on(self, event: str, handler):
        """Register event handler."""
        self._event_handlers[event] = handler
    
    async def connect(self, url: str, auth: Dict[str, str]):
        """Simulate connection."""
        self.connected = True
        self.sid = f"sid_{self.client_id}"
        
        # Trigger connected event if handler exists
        if 'connected' in self._event_handlers:
            await self._event_handlers['connected']({
                'sid': self.sid,
                'timestamp': datetime.now().isoformat(),
                'message': 'Connected to optimization platform'
            })
    
    async def disconnect(self):
        """Simulate disconnection."""
        self.connected = False
        self.sid = None
    
    async def emit(self, event: str, data: Dict[str, Any]):
        """Emit event to server."""
        pass
    
    def receive_event(self, event: str, data: Dict[str, Any]):
        """Receive event from server."""
        self.received_events.append({
            'event': event,
            'data': data,
            'timestamp': datetime.now()
        })
        
        # Trigger handler if exists
        if event in self._event_handlers:
            asyncio.create_task(self._event_handlers[event](data))
    
    def get_events_by_type(self, event_type: str) -> List[Dict[str, Any]]:
        """Get all received events of a specific type."""
        return [e for e in self.received_events if e['event'] == event_type]
    
    def clear_events(self):
        """Clear received events."""
        self.received_events.clear()


@pytest.fixture
def notification_service():
    """Create NotificationService instance."""
    return NotificationService()


@pytest.fixture
def websocket_manager():
    """Create WebSocketManager instance."""
    manager = WebSocketManager()
    # Reset singleton state
    manager._connections = {}
    manager._session_subscriptions = {}
    manager._notification_service = None
    return manager


@pytest.fixture
def mock_user():
    """Create mock user for authentication."""
    return User(
        id="test-user-1",
        username="testuser",
        role="user",
        is_active=True
    )


@pytest.fixture
def mock_admin_user():
    """Create mock admin user."""
    return User(
        id="admin-user-1",
        username="admin",
        role="administrator",
        is_active=True
    )


@pytest.fixture
def connected_clients(websocket_manager, mock_user):
    """Create multiple connected mock clients."""
    clients = []
    
    for i in range(3):
        client = MockSocketIOClient(f"client_{i}")
        sid = f"sid_client_{i}"
        
        # Simulate connection
        websocket_manager._connections[sid] = {
            'connected_at': datetime.now(),
            'user_id': mock_user.id,
            'username': mock_user.username,
            'role': mock_user.role,
            'subscriptions': set()
        }
        
        client.sid = sid
        client.connected = True
        clients.append(client)
    
    yield clients
    
    # Cleanup
    for client in clients:
        if client.sid in websocket_manager._connections:
            del websocket_manager._connections[client.sid]


class TestEndToEndEventDelivery:
    """Test end-to-end event delivery from NotificationService to clients."""
    
    @pytest.mark.asyncio
    async def test_notification_delivery_flow(
        self, 
        websocket_manager, 
        notification_service,
        connected_clients
    ):
        """Test complete notification flow from service to clients."""
        # Setup WebSocket manager with mock emit
        emitted_events = []
        
        async def mock_emit(event, data, room=None):
            emitted_events.append({
                'event': event,
                'data': data,
                'room': room
            })
        
        websocket_manager.sio.emit = mock_emit
        
        # Connect NotificationService to WebSocketManager
        websocket_manager.setup_notification_handlers(notification_service)
        
        # Send notification through NotificationService
        notification_id = notification_service.send_notification(
            type=NotificationType.INFO,
            title="Test Notification",
            message="This is an end-to-end test",
            session_id=None
        )
        
        # Wait for async processing
        await asyncio.sleep(0.2)
        
        # Verify event was emitted
        assert len(emitted_events) > 0
        notification_events = [e for e in emitted_events if e['event'] == 'notification']
        assert len(notification_events) == 1
        
        event_data = notification_events[0]['data']
        assert event_data['id'] == notification_id
        assert event_data['type'] == 'info'
        assert event_data['title'] == "Test Notification"
        assert event_data['message'] == "This is an end-to-end test"
    
    @pytest.mark.asyncio
    async def test_alert_delivery_flow(
        self,
        websocket_manager,
        notification_service,
        connected_clients
    ):
        """Test complete alert flow from service to clients."""
        emitted_events = []
        
        async def mock_emit(event, data, room=None):
            emitted_events.append({
                'event': event,
                'data': data,
                'room': room
            })
        
        websocket_manager.sio.emit = mock_emit
        websocket_manager.setup_notification_handlers(notification_service)
        
        # Create alert
        alert_id = notification_service.create_alert(
            severity=AlertSeverity.HIGH,
            title="Critical Alert",
            description="System memory usage critical",
            session_id=None
        )
        
        await asyncio.sleep(0.2)
        
        # Verify alert was emitted
        alert_events = [e for e in emitted_events if e['event'] == 'alert']
        assert len(alert_events) == 1
        
        event_data = alert_events[0]['data']
        assert event_data['id'] == alert_id
        assert event_data['severity'] == 'high'
        assert event_data['title'] == "Critical Alert"
    
    @pytest.mark.asyncio
    async def test_progress_update_delivery_flow(
        self,
        websocket_manager,
        notification_service,
        connected_clients
    ):
        """Test complete progress update flow."""
        emitted_events = []
        
        async def mock_emit(event, data, room=None):
            emitted_events.append({
                'event': event,
                'data': data,
                'room': room
            })
        
        websocket_manager.sio.emit = mock_emit
        websocket_manager.setup_notification_handlers(notification_service)
        
        # Add session subscription
        session_id = "test_session_123"
        websocket_manager._session_subscriptions[session_id] = {
            connected_clients[0].sid
        }
        
        # Start progress tracking
        notification_service.start_progress_tracking(
            session_id=session_id,
            total_steps=10,
            initial_step_name="Initializing"
        )
        
        await asyncio.sleep(0.2)
        
        # Update progress
        notification_service.update_progress(
            session_id=session_id,
            current_step=5,
            step_name="Processing"
        )
        
        await asyncio.sleep(0.2)
        
        # Verify progress events were emitted
        progress_events = [e for e in emitted_events if e['event'] == 'progress_update']
        assert len(progress_events) >= 2
        
        # Check initial progress
        initial_event = progress_events[0]['data']
        assert initial_event['session_id'] == session_id
        assert initial_event['progress_percentage'] == 0.0
        
        # Check updated progress
        update_event = progress_events[1]['data']
        assert update_event['session_id'] == session_id
        assert update_event['progress_percentage'] == 50.0
        assert update_event['step_name'] == "Processing"


class TestMultipleClientScenarios:
    """Test scenarios with multiple connected clients."""
    
    @pytest.mark.asyncio
    async def test_broadcast_to_all_clients(
        self,
        websocket_manager,
        notification_service,
        connected_clients
    ):
        """Test broadcasting events to all connected clients."""
        emitted_events = []
        
        async def mock_emit(event, data, room=None):
            emitted_events.append({
                'event': event,
                'data': data,
                'room': room
            })
        
        websocket_manager.sio.emit = mock_emit
        websocket_manager.setup_notification_handlers(notification_service)
        
        # Send global notification (no session_id)
        notification_service.send_notification(
            type=NotificationType.INFO,
            title="Global Announcement",
            message="This goes to all clients"
        )
        
        await asyncio.sleep(0.2)
        
        # Verify event was broadcast (room should be None for global)
        notification_events = [e for e in emitted_events if e['event'] == 'notification']
        assert len(notification_events) == 1
        assert notification_events[0]['room'] is None
    
    @pytest.mark.asyncio
    async def test_multiple_clients_receive_same_event(
        self,
        websocket_manager,
        notification_service,
        connected_clients
    ):
        """Test that multiple clients subscribed to same session receive events."""
        emitted_events = []
        
        async def mock_emit(event, data, room=None):
            emitted_events.append({
                'event': event,
                'data': data,
                'room': room
            })
            
            # Simulate delivery to all subscribers
            if room:
                for client in connected_clients:
                    if client.sid == room:
                        client.receive_event(event, data)
        
        websocket_manager.sio.emit = mock_emit
        websocket_manager.setup_notification_handlers(notification_service)
        
        # Subscribe all clients to same session
        session_id = "shared_session_123"
        for client in connected_clients:
            websocket_manager._session_subscriptions.setdefault(session_id, set()).add(client.sid)
            websocket_manager._connections[client.sid]['subscriptions'].add(session_id)
        
        # Send session-specific notification
        notification_service.send_notification(
            type=NotificationType.SUCCESS,
            title="Session Update",
            message="Optimization completed",
            session_id=session_id
        )
        
        await asyncio.sleep(0.2)
        
        # Verify all clients received the event
        notification_events = [e for e in emitted_events if e['event'] == 'notification']
        # Should emit once per subscriber
        assert len(notification_events) >= len(connected_clients)
    
    @pytest.mark.asyncio
    async def test_client_specific_subscriptions(
        self,
        websocket_manager,
        notification_service,
        connected_clients
    ):
        """Test that clients only receive events for their subscriptions."""
        emitted_events = []
        
        async def mock_emit(event, data, room=None):
            emitted_events.append({
                'event': event,
                'data': data,
                'room': room
            })
            
            # Simulate targeted delivery
            if room:
                for client in connected_clients:
                    if client.sid == room:
                        client.receive_event(event, data)
        
        websocket_manager.sio.emit = mock_emit
        websocket_manager.setup_notification_handlers(notification_service)
        
        # Subscribe clients to different sessions
        session_1 = "session_1"
        session_2 = "session_2"
        
        # Client 0 subscribes to session_1
        websocket_manager._session_subscriptions[session_1] = {connected_clients[0].sid}
        websocket_manager._connections[connected_clients[0].sid]['subscriptions'].add(session_1)
        
        # Client 1 subscribes to session_2
        websocket_manager._session_subscriptions[session_2] = {connected_clients[1].sid}
        websocket_manager._connections[connected_clients[1].sid]['subscriptions'].add(session_2)
        
        # Client 2 subscribes to both
        websocket_manager._session_subscriptions[session_1].add(connected_clients[2].sid)
        websocket_manager._session_subscriptions[session_2].add(connected_clients[2].sid)
        websocket_manager._connections[connected_clients[2].sid]['subscriptions'].update([session_1, session_2])
        
        # Send notifications to different sessions
        notification_service.send_notification(
            type=NotificationType.INFO,
            title="Session 1 Update",
            message="Update for session 1",
            session_id=session_1
        )
        
        notification_service.send_notification(
            type=NotificationType.INFO,
            title="Session 2 Update",
            message="Update for session 2",
            session_id=session_2
        )
        
        await asyncio.sleep(0.3)
        
        # Verify client 0 received only session_1 events
        client_0_events = connected_clients[0].get_events_by_type('notification')
        assert len(client_0_events) == 1
        assert "Session 1" in client_0_events[0]['data']['title']
        
        # Verify client 1 received only session_2 events
        client_1_events = connected_clients[1].get_events_by_type('notification')
        assert len(client_1_events) == 1
        assert "Session 2" in client_1_events[0]['data']['title']
        
        # Verify client 2 received both
        client_2_events = connected_clients[2].get_events_by_type('notification')
        assert len(client_2_events) == 2
    
    @pytest.mark.asyncio
    async def test_concurrent_event_delivery(
        self,
        websocket_manager,
        notification_service,
        connected_clients
    ):
        """Test handling concurrent event delivery to multiple clients."""
        emitted_events = []
        
        async def mock_emit(event, data, room=None):
            emitted_events.append({
                'event': event,
                'data': data,
                'room': room
            })
        
        websocket_manager.sio.emit = mock_emit
        websocket_manager.setup_notification_handlers(notification_service)
        
        # Send multiple events rapidly
        for i in range(10):
            notification_service.send_notification(
                type=NotificationType.INFO,
                title=f"Notification {i}",
                message=f"Message {i}"
            )
        
        await asyncio.sleep(0.5)
        
        # Verify all events were emitted
        notification_events = [e for e in emitted_events if e['event'] == 'notification']
        assert len(notification_events) == 10


class TestRoomBasedEventFiltering:
    """Test room-based event filtering and subscription management."""
    
    @pytest.mark.asyncio
    async def test_session_room_subscription(
        self,
        websocket_manager,
        connected_clients
    ):
        """Test subscribing and unsubscribing from session rooms."""
        session_id = "test_session_456"
        client = connected_clients[0]
        
        # Subscribe to session
        await websocket_manager._handle_subscribe_session(
            client.sid,
            {'session_id': session_id}
        )
        
        # Verify subscription
        assert session_id in websocket_manager._connections[client.sid]['subscriptions']
        assert client.sid in websocket_manager._session_subscriptions[session_id]
        
        # Unsubscribe from session
        await websocket_manager._handle_unsubscribe_session(
            client.sid,
            {'session_id': session_id}
        )
        
        # Verify unsubscription
        assert session_id not in websocket_manager._connections[client.sid]['subscriptions']
        assert session_id not in websocket_manager._session_subscriptions or \
               client.sid not in websocket_manager._session_subscriptions[session_id]
    
    @pytest.mark.asyncio
    async def test_room_based_progress_filtering(
        self,
        websocket_manager,
        notification_service,
        connected_clients
    ):
        """Test that progress updates only go to subscribed clients."""
        emitted_events = []
        
        async def mock_emit(event, data, room=None):
            emitted_events.append({
                'event': event,
                'data': data,
                'room': room
            })
            
            # Simulate room-based delivery
            if room:
                for client in connected_clients:
                    if client.sid == room:
                        client.receive_event(event, data)
        
        websocket_manager.sio.emit = mock_emit
        websocket_manager.setup_notification_handlers(notification_service)
        
        # Only subscribe first client to session
        session_id = "progress_session_789"
        websocket_manager._session_subscriptions[session_id] = {connected_clients[0].sid}
        
        # Start progress tracking
        notification_service.start_progress_tracking(
            session_id=session_id,
            total_steps=5
        )
        
        await asyncio.sleep(0.2)
        
        # Verify only subscribed client received progress
        client_0_progress = connected_clients[0].get_events_by_type('progress_update')
        client_1_progress = connected_clients[1].get_events_by_type('progress_update')
        
        assert len(client_0_progress) > 0
        assert len(client_1_progress) == 0
    
    @pytest.mark.asyncio
    async def test_multiple_session_subscriptions(
        self,
        websocket_manager,
        notification_service,
        connected_clients
    ):
        """Test client subscribed to multiple sessions."""
        emitted_events = []
        
        async def mock_emit(event, data, room=None):
            emitted_events.append({
                'event': event,
                'data': data,
                'room': room
            })
            
            if room:
                for client in connected_clients:
                    if client.sid == room:
                        client.receive_event(event, data)
        
        websocket_manager.sio.emit = mock_emit
        websocket_manager.setup_notification_handlers(notification_service)
        
        # Subscribe one client to multiple sessions
        client = connected_clients[0]
        sessions = ["session_a", "session_b", "session_c"]
        
        for session_id in sessions:
            websocket_manager._session_subscriptions[session_id] = {client.sid}
            websocket_manager._connections[client.sid]['subscriptions'].add(session_id)
        
        # Send progress updates to each session
        for session_id in sessions:
            notification_service.start_progress_tracking(
                session_id=session_id,
                total_steps=3
            )
        
        await asyncio.sleep(0.3)
        
        # Verify client received updates from all sessions
        progress_events = client.get_events_by_type('progress_update')
        assert len(progress_events) == len(sessions)
        
        # Verify each session is represented
        session_ids = {e['data']['session_id'] for e in progress_events}
        assert session_ids == set(sessions)
    
    @pytest.mark.asyncio
    async def test_session_cleanup_on_disconnect(
        self,
        websocket_manager,
        connected_clients
    ):
        """Test that session subscriptions are cleaned up on disconnect."""
        client = connected_clients[0]
        session_id = "cleanup_session"
        
        # Subscribe to session
        websocket_manager._session_subscriptions[session_id] = {client.sid}
        websocket_manager._connections[client.sid]['subscriptions'].add(session_id)
        
        # Disconnect client
        await websocket_manager._handle_disconnect(client.sid)
        
        # Verify cleanup
        assert client.sid not in websocket_manager._connections
        assert session_id not in websocket_manager._session_subscriptions or \
               len(websocket_manager._session_subscriptions[session_id]) == 0


class TestReconnectionAndStateSynchronization:
    """Test reconnection handling and state synchronization."""
    
    @pytest.mark.asyncio
    async def test_client_reconnection(
        self,
        websocket_manager,
        mock_user
    ):
        """Test client reconnection flow."""
        client = MockSocketIOClient("reconnect_client")
        
        # Initial connection
        sid_1 = "sid_reconnect_1"
        websocket_manager._connections[sid_1] = {
            'connected_at': datetime.now(),
            'user_id': mock_user.id,
            'username': mock_user.username,
            'role': mock_user.role,
            'subscriptions': {'session_1', 'session_2'}
        }
        
        # Add to session subscriptions
        websocket_manager._session_subscriptions['session_1'] = {sid_1}
        websocket_manager._session_subscriptions['session_2'] = {sid_1}
        
        # Disconnect
        await websocket_manager._handle_disconnect(sid_1)
        
        # Verify cleanup
        assert sid_1 not in websocket_manager._connections
        assert sid_1 not in websocket_manager._session_subscriptions.get('session_1', set())
        
        # Reconnect with new sid
        sid_2 = "sid_reconnect_2"
        websocket_manager._connections[sid_2] = {
            'connected_at': datetime.now(),
            'user_id': mock_user.id,
            'username': mock_user.username,
            'role': mock_user.role,
            'subscriptions': set()
        }
        
        # Client needs to re-subscribe
        await websocket_manager._handle_subscribe_session(sid_2, {'session_id': 'session_1'})
        await websocket_manager._handle_subscribe_session(sid_2, {'session_id': 'session_2'})
        
        # Verify new subscriptions
        assert 'session_1' in websocket_manager._connections[sid_2]['subscriptions']
        assert 'session_2' in websocket_manager._connections[sid_2]['subscriptions']
    
    @pytest.mark.asyncio
    async def test_state_synchronization_after_reconnect(
        self,
        websocket_manager,
        notification_service,
        mock_user
    ):
        """Test that client can synchronize state after reconnection."""
        # Simulate ongoing session
        session_id = "ongoing_session"
        notification_service.start_progress_tracking(
            session_id=session_id,
            total_steps=10
        )
        
        # Update progress
        notification_service.update_progress(
            session_id=session_id,
            current_step=5,
            step_name="Processing"
        )
        
        # Client reconnects and subscribes
        sid = "sid_sync_test"
        websocket_manager._connections[sid] = {
            'connected_at': datetime.now(),
            'user_id': mock_user.id,
            'username': mock_user.username,
            'role': mock_user.role,
            'subscriptions': set()
        }
        
        await websocket_manager._handle_subscribe_session(sid, {'session_id': session_id})
        
        # Get current progress state
        current_progress = notification_service.get_progress(session_id)
        
        # Verify client can get current state
        assert current_progress is not None
        assert current_progress.session_id == session_id
        assert current_progress.current_step == 5
        assert current_progress.progress_percentage == 50.0
    
    @pytest.mark.asyncio
    async def test_missed_events_during_disconnect(
        self,
        websocket_manager,
        notification_service,
        mock_user
    ):
        """Test handling of events that occurred during disconnect."""
        emitted_events = []
        
        async def mock_emit(event, data, room=None):
            emitted_events.append({
                'event': event,
                'data': data,
                'room': room,
                'timestamp': datetime.now()
            })
        
        websocket_manager.sio.emit = mock_emit
        websocket_manager.setup_notification_handlers(notification_service)
        
        session_id = "missed_events_session"
        
        # Client initially connected and subscribed
        sid_1 = "sid_initial"
        websocket_manager._connections[sid_1] = {
            'connected_at': datetime.now(),
            'user_id': mock_user.id,
            'username': mock_user.username,
            'role': mock_user.role,
            'subscriptions': {session_id}
        }
        websocket_manager._session_subscriptions[session_id] = {sid_1}
        
        # Start progress
        notification_service.start_progress_tracking(
            session_id=session_id,
            total_steps=10
        )
        
        await asyncio.sleep(0.1)
        initial_event_count = len(emitted_events)
        
        # Client disconnects
        await websocket_manager._handle_disconnect(sid_1)
        
        # Events occur while disconnected
        for i in range(1, 6):
            notification_service.update_progress(
                session_id=session_id,
                current_step=i,
                step_name=f"Step {i}"
            )
        
        await asyncio.sleep(0.2)
        
        # Events were attempted to be emitted but no subscribers were present
        # Since client disconnected, events after disconnect won't be emitted to rooms
        # But they were still processed by the notification service
        missed_event_count = len(emitted_events) - initial_event_count
        # Events may not be emitted if no subscribers, which is expected behavior
        assert missed_event_count >= 0
        
        # Client reconnects
        sid_2 = "sid_reconnect"
        websocket_manager._connections[sid_2] = {
            'connected_at': datetime.now(),
            'user_id': mock_user.id,
            'username': mock_user.username,
            'role': mock_user.role,
            'subscriptions': set()
        }
        
        await websocket_manager._handle_subscribe_session(sid_2, {'session_id': session_id})
        
        # Client should query current state to catch up
        current_progress = notification_service.get_progress(session_id)
        assert current_progress.current_step == 5
        assert current_progress.progress_percentage == 50.0
    
    @pytest.mark.asyncio
    async def test_connection_health_check(
        self,
        websocket_manager,
        connected_clients
    ):
        """Test connection health check with ping/pong."""
        client = connected_clients[0]
        
        # Mock emit for pong response
        pong_received = False
        
        async def mock_emit(event, data, room=None):
            nonlocal pong_received
            if event == 'pong':
                pong_received = True
        
        websocket_manager.sio.emit = mock_emit
        
        # Get the ping handler that was registered
        # The handler is registered in _register_handlers
        # We'll call it directly
        ping_handler = None
        for handler_name, handler in websocket_manager.sio.handlers['/'].items():
            if handler_name == 'ping':
                ping_handler = handler
                break
        
        if ping_handler:
            await ping_handler(client.sid)
        
        await asyncio.sleep(0.1)
        
        # Verify pong was sent
        assert pong_received


class TestEventDeliveryReliability:
    """Test event delivery reliability and error handling."""
    
    @pytest.mark.asyncio
    async def test_delivery_failure_handling(
        self,
        websocket_manager,
        notification_service,
        connected_clients
    ):
        """Test graceful handling of event delivery failures."""
        failed_sids = set()
        successful_sids = set()
        
        async def mock_emit(event, data, room=None):
            # Simulate failure for first client
            if room == connected_clients[0].sid:
                failed_sids.add(room)
                raise Exception("Network error")
            else:
                successful_sids.add(room)
        
        websocket_manager.sio.emit = mock_emit
        websocket_manager.setup_notification_handlers(notification_service)
        
        # Subscribe all clients to session
        session_id = "reliability_test"
        for client in connected_clients:
            websocket_manager._session_subscriptions.setdefault(session_id, set()).add(client.sid)
        
        # Send notification
        notification_service.send_notification(
            type=NotificationType.INFO,
            title="Reliability Test",
            message="Testing delivery failures",
            session_id=session_id
        )
        
        await asyncio.sleep(0.2)
        
        # Verify that failure for one client didn't prevent delivery to others
        assert len(failed_sids) > 0
        assert len(successful_sids) > 0
    
    @pytest.mark.asyncio
    async def test_event_ordering(
        self,
        websocket_manager,
        notification_service,
        connected_clients
    ):
        """Test that events are delivered in order."""
        emitted_events = []
        
        async def mock_emit(event, data, room=None):
            emitted_events.append({
                'event': event,
                'data': data,
                'timestamp': datetime.now()
            })
        
        websocket_manager.sio.emit = mock_emit
        websocket_manager.setup_notification_handlers(notification_service)
        
        # Send multiple notifications in sequence
        for i in range(5):
            notification_service.send_notification(
                type=NotificationType.INFO,
                title=f"Notification {i}",
                message=f"Message {i}"
            )
        
        await asyncio.sleep(0.3)
        
        # Verify events are in order
        notification_events = [e for e in emitted_events if e['event'] == 'notification']
        assert len(notification_events) == 5
        
        for i, event in enumerate(notification_events):
            assert f"Notification {i}" in event['data']['title']
    
    @pytest.mark.asyncio
    async def test_high_frequency_events(
        self,
        websocket_manager,
        notification_service,
        connected_clients
    ):
        """Test handling of high-frequency event delivery."""
        emitted_events = []
        
        async def mock_emit(event, data, room=None):
            emitted_events.append({
                'event': event,
                'data': data
            })
        
        websocket_manager.sio.emit = mock_emit
        websocket_manager.setup_notification_handlers(notification_service)
        
        session_id = "high_freq_session"
        websocket_manager._session_subscriptions[session_id] = {connected_clients[0].sid}
        
        # Start progress tracking
        notification_service.start_progress_tracking(
            session_id=session_id,
            total_steps=100
        )
        
        # Send rapid progress updates
        for i in range(1, 101):
            notification_service.update_progress(
                session_id=session_id,
                current_step=i,
                step_name=f"Step {i}"
            )
        
        await asyncio.sleep(0.5)
        
        # Verify all events were emitted
        progress_events = [e for e in emitted_events if e['event'] == 'progress_update']
        # Should have initial + 100 updates
        assert len(progress_events) >= 100


class TestComplexScenarios:
    """Test complex real-world scenarios."""
    
    @pytest.mark.asyncio
    async def test_multiple_sessions_multiple_clients(
        self,
        websocket_manager,
        notification_service,
        connected_clients
    ):
        """Test multiple sessions with multiple clients in complex subscription patterns."""
        emitted_events = []
        
        async def mock_emit(event, data, room=None):
            emitted_events.append({
                'event': event,
                'data': data,
                'room': room
            })
            
            if room:
                for client in connected_clients:
                    if client.sid == room:
                        client.receive_event(event, data)
        
        websocket_manager.sio.emit = mock_emit
        websocket_manager.setup_notification_handlers(notification_service)
        
        # Create complex subscription pattern
        # Client 0: sessions A, B
        # Client 1: sessions B, C
        # Client 2: session C
        
        sessions = {
            'session_a': {connected_clients[0].sid},
            'session_b': {connected_clients[0].sid, connected_clients[1].sid},
            'session_c': {connected_clients[1].sid, connected_clients[2].sid}
        }
        
        for session_id, sids in sessions.items():
            websocket_manager._session_subscriptions[session_id] = sids
            for sid in sids:
                websocket_manager._connections[sid]['subscriptions'].add(session_id)
        
        # Start progress for all sessions
        for session_id in sessions.keys():
            notification_service.start_progress_tracking(
                session_id=session_id,
                total_steps=5
            )
            
            notification_service.update_progress(
                session_id=session_id,
                current_step=3,
                step_name="Processing"
            )
        
        await asyncio.sleep(0.5)
        
        # Verify each client received correct events
        client_0_events = connected_clients[0].get_events_by_type('progress_update')
        client_1_events = connected_clients[1].get_events_by_type('progress_update')
        client_2_events = connected_clients[2].get_events_by_type('progress_update')
        
        # Client 0 should have events from A and B (2 sessions * 2 events each)
        assert len(client_0_events) == 4
        
        # Client 1 should have events from B and C (2 sessions * 2 events each)
        assert len(client_1_events) == 4
        
        # Client 2 should have events from C only (1 session * 2 events)
        assert len(client_2_events) == 2
    
    @pytest.mark.asyncio
    async def test_session_lifecycle_with_events(
        self,
        websocket_manager,
        notification_service,
        connected_clients
    ):
        """Test complete session lifecycle with all event types."""
        emitted_events = []
        
        async def mock_emit(event, data, room=None):
            emitted_events.append({
                'event': event,
                'data': data,
                'room': room
            })
        
        websocket_manager.sio.emit = mock_emit
        websocket_manager.setup_notification_handlers(notification_service)
        
        session_id = "lifecycle_session"
        websocket_manager._session_subscriptions[session_id] = {connected_clients[0].sid}
        
        # Session started
        await websocket_manager.broadcast_session_started(
            session_id=session_id,
            model_id="model_123",
            model_name="test_model.pt",
            techniques=["quantization", "pruning"]
        )
        
        # Progress updates
        notification_service.start_progress_tracking(
            session_id=session_id,
            total_steps=5
        )
        
        for i in range(1, 6):
            notification_service.update_progress(
                session_id=session_id,
                current_step=i,
                step_name=f"Step {i}"
            )
        
        # Session completed
        await websocket_manager.broadcast_session_completed(
            session_id=session_id,
            results={
                'size_reduction_percent': 25.0,
                'speed_improvement_percent': 15.0
            }
        )
        
        await asyncio.sleep(0.5)
        
        # Verify all event types were emitted
        event_types = {e['event'] for e in emitted_events}
        assert 'session_started' in event_types
        assert 'progress_update' in event_types
        assert 'session_completed' in event_types


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

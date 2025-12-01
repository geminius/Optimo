"""
Unit tests for WebSocketManager.

This module tests the WebSocketManager class, focusing on connection handling,
room subscriptions, event broadcasting, and error handling.
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch, MagicMock

from src.services.websocket_manager import WebSocketManager


@pytest.fixture
def websocket_manager():
    """Create a WebSocketManager instance for testing."""
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


@pytest.fixture
def mock_auth_user():
    """Create a mock authenticated user."""
    user = Mock()
    user.id = "user_123"
    user.username = "testuser"
    user.role = "user"
    return user


@pytest.fixture
def mock_auth_manager(mock_auth_user):
    """Create a mock authentication manager."""
    mock = Mock()
    mock.verify_websocket_token = Mock(return_value=mock_auth_user)
    return mock


class TestWebSocketManagerConnectionHandling:
    """Test connection and disconnection handling (Requirement 4.1)."""
    
    @pytest.mark.asyncio
    async def test_handle_connect_with_valid_token(self, websocket_manager, mock_sio, mock_auth_manager, mock_auth_user):
        """Test successful client connection with valid authentication token."""
        websocket_manager.sio = mock_sio
        
        with patch('src.api.auth.get_auth_manager', return_value=mock_auth_manager):
            # Simulate connection
            sid = "test_sid_123"
            environ = {}
            auth = {"token": "valid_token"}
            
            result = await websocket_manager._handle_connect(sid, environ, auth)
            
            # Verify connection was accepted (not False)
            assert result is not False
            
            # Verify connection was tracked
            assert sid in websocket_manager._connections
            assert websocket_manager._connections[sid]['user_id'] == mock_auth_user.id
            assert websocket_manager._connections[sid]['username'] == mock_auth_user.username
            assert websocket_manager._connections[sid]['role'] == mock_auth_user.role
            
            # Verify connection confirmation was sent
            assert mock_sio.emit.called
            call_args = mock_sio.emit.call_args
            assert call_args[0][0] == 'connected'
            assert call_args[1]['room'] == sid
    
    @pytest.mark.asyncio
    async def test_handle_connect_with_invalid_token(self, websocket_manager, mock_sio):
        """Test connection rejection with invalid authentication token."""
        websocket_manager.sio = mock_sio
        
        mock_auth_manager = Mock()
        mock_auth_manager.verify_websocket_token = Mock(return_value=None)
        
        with patch('src.api.auth.get_auth_manager', return_value=mock_auth_manager):
            sid = "test_sid_123"
            environ = {}
            auth = {"token": "invalid_token"}
            
            result = await websocket_manager._handle_connect(sid, environ, auth)
            
            # Verify connection was rejected
            assert result is False
            
            # Verify connection was not tracked
            assert sid not in websocket_manager._connections
    
    @pytest.mark.asyncio
    async def test_handle_connect_without_token(self, websocket_manager, mock_sio):
        """Test connection rejection when no authentication token is provided."""
        websocket_manager.sio = mock_sio
        
        sid = "test_sid_123"
        environ = {}
        auth = None
        
        result = await websocket_manager._handle_connect(sid, environ, auth)
        
        # Verify connection was rejected
        assert result is False
        
        # Verify connection was not tracked
        assert sid not in websocket_manager._connections

    @pytest.mark.asyncio
    async def test_handle_disconnect_cleans_up_connection(self, websocket_manager, mock_sio):
        """Test that disconnection properly cleans up connection data."""
        websocket_manager.sio = mock_sio
        
        # Set up a connection with subscriptions
        sid = "test_sid_123"
        session_id = "session_456"
        
        websocket_manager._connections[sid] = {
            'connected_at': datetime.now(),
            'user_id': 'user_123',
            'username': 'testuser',
            'role': 'user',
            'subscriptions': {session_id}
        }
        websocket_manager._session_subscriptions[session_id] = {sid}
        
        # Handle disconnect
        await websocket_manager._handle_disconnect(sid)
        
        # Verify connection was removed
        assert sid not in websocket_manager._connections
        
        # Verify session subscription was cleaned up
        assert session_id not in websocket_manager._session_subscriptions
    
    @pytest.mark.asyncio
    async def test_handle_disconnect_with_multiple_subscribers(self, websocket_manager, mock_sio):
        """Test disconnection when multiple clients are subscribed to same session."""
        websocket_manager.sio = mock_sio
        
        # Set up multiple connections to same session
        sid1 = "test_sid_1"
        sid2 = "test_sid_2"
        session_id = "session_456"
        
        websocket_manager._connections[sid1] = {
            'connected_at': datetime.now(),
            'user_id': 'user_1',
            'username': 'user1',
            'role': 'user',
            'subscriptions': {session_id}
        }
        websocket_manager._connections[sid2] = {
            'connected_at': datetime.now(),
            'user_id': 'user_2',
            'username': 'user2',
            'role': 'user',
            'subscriptions': {session_id}
        }
        websocket_manager._session_subscriptions[session_id] = {sid1, sid2}
        
        # Disconnect first client
        await websocket_manager._handle_disconnect(sid1)
        
        # Verify first connection was removed
        assert sid1 not in websocket_manager._connections
        
        # Verify session subscription still exists with second client
        assert session_id in websocket_manager._session_subscriptions
        assert sid2 in websocket_manager._session_subscriptions[session_id]
        assert sid1 not in websocket_manager._session_subscriptions[session_id]
    
    @pytest.mark.asyncio
    async def test_handle_disconnect_nonexistent_connection(self, websocket_manager, mock_sio):
        """Test that disconnecting a non-existent connection doesn't raise errors."""
        websocket_manager.sio = mock_sio
        
        # Try to disconnect a connection that doesn't exist
        sid = "nonexistent_sid"
        
        # Should not raise an exception
        await websocket_manager._handle_disconnect(sid)
        
        # Verify no errors occurred
        assert sid not in websocket_manager._connections


class TestWebSocketManagerSubscriptions:
    """Test room subscription and unsubscription (Requirement 4.1)."""
    
    @pytest.mark.asyncio
    async def test_subscribe_to_session(self, websocket_manager, mock_sio):
        """Test subscribing a client to a session."""
        websocket_manager.sio = mock_sio
        
        # Set up a connection
        sid = "test_sid_123"
        websocket_manager._connections[sid] = {
            'connected_at': datetime.now(),
            'user_id': 'user_123',
            'username': 'testuser',
            'role': 'user',
            'subscriptions': set()
        }
        
        # Subscribe to session
        session_id = "session_456"
        await websocket_manager._handle_subscribe_session(sid, {'session_id': session_id})
        
        # Verify subscription was added to connection
        assert session_id in websocket_manager._connections[sid]['subscriptions']
        
        # Verify subscription was added to session subscriptions
        assert session_id in websocket_manager._session_subscriptions
        assert sid in websocket_manager._session_subscriptions[session_id]
        
        # Verify confirmation was sent
        assert mock_sio.emit.called
        call_args = mock_sio.emit.call_args
        assert call_args[0][0] == 'subscribed'
        assert call_args[0][1]['session_id'] == session_id

    @pytest.mark.asyncio
    async def test_subscribe_without_session_id(self, websocket_manager, mock_sio):
        """Test that subscribing without session_id sends an error."""
        websocket_manager.sio = mock_sio
        
        sid = "test_sid_123"
        websocket_manager._connections[sid] = {
            'connected_at': datetime.now(),
            'user_id': 'user_123',
            'username': 'testuser',
            'role': 'user',
            'subscriptions': set()
        }
        
        # Try to subscribe without session_id
        await websocket_manager._handle_subscribe_session(sid, {})
        
        # Verify error was sent
        assert mock_sio.emit.called
        call_args = mock_sio.emit.call_args
        assert call_args[0][0] == 'error'
        assert 'session_id required' in call_args[0][1]['message']
    
    @pytest.mark.asyncio
    async def test_subscribe_multiple_sessions(self, websocket_manager, mock_sio):
        """Test subscribing a client to multiple sessions."""
        websocket_manager.sio = mock_sio
        
        sid = "test_sid_123"
        websocket_manager._connections[sid] = {
            'connected_at': datetime.now(),
            'user_id': 'user_123',
            'username': 'testuser',
            'role': 'user',
            'subscriptions': set()
        }
        
        # Subscribe to multiple sessions
        session_id_1 = "session_1"
        session_id_2 = "session_2"
        
        await websocket_manager._handle_subscribe_session(sid, {'session_id': session_id_1})
        await websocket_manager._handle_subscribe_session(sid, {'session_id': session_id_2})
        
        # Verify both subscriptions exist
        assert session_id_1 in websocket_manager._connections[sid]['subscriptions']
        assert session_id_2 in websocket_manager._connections[sid]['subscriptions']
        assert sid in websocket_manager._session_subscriptions[session_id_1]
        assert sid in websocket_manager._session_subscriptions[session_id_2]
    
    @pytest.mark.asyncio
    async def test_unsubscribe_from_session(self, websocket_manager, mock_sio):
        """Test unsubscribing a client from a session."""
        websocket_manager.sio = mock_sio
        
        # Set up a connection with subscription
        sid = "test_sid_123"
        session_id = "session_456"
        
        websocket_manager._connections[sid] = {
            'connected_at': datetime.now(),
            'user_id': 'user_123',
            'username': 'testuser',
            'role': 'user',
            'subscriptions': {session_id}
        }
        websocket_manager._session_subscriptions[session_id] = {sid}
        
        # Unsubscribe from session
        await websocket_manager._handle_unsubscribe_session(sid, {'session_id': session_id})
        
        # Verify subscription was removed from connection
        assert session_id not in websocket_manager._connections[sid]['subscriptions']
        
        # Verify subscription was removed from session subscriptions
        assert session_id not in websocket_manager._session_subscriptions
        
        # Verify confirmation was sent
        assert mock_sio.emit.called
        call_args = mock_sio.emit.call_args
        assert call_args[0][0] == 'unsubscribed'
        assert call_args[0][1]['session_id'] == session_id
    
    @pytest.mark.asyncio
    async def test_unsubscribe_with_remaining_subscribers(self, websocket_manager, mock_sio):
        """Test unsubscribing when other clients remain subscribed."""
        websocket_manager.sio = mock_sio
        
        # Set up multiple connections to same session
        sid1 = "test_sid_1"
        sid2 = "test_sid_2"
        session_id = "session_456"
        
        websocket_manager._connections[sid1] = {
            'connected_at': datetime.now(),
            'user_id': 'user_1',
            'username': 'user1',
            'role': 'user',
            'subscriptions': {session_id}
        }
        websocket_manager._connections[sid2] = {
            'connected_at': datetime.now(),
            'user_id': 'user_2',
            'username': 'user2',
            'role': 'user',
            'subscriptions': {session_id}
        }
        websocket_manager._session_subscriptions[session_id] = {sid1, sid2}
        
        # Unsubscribe first client
        await websocket_manager._handle_unsubscribe_session(sid1, {'session_id': session_id})
        
        # Verify first client's subscription was removed
        assert session_id not in websocket_manager._connections[sid1]['subscriptions']
        
        # Verify session subscription still exists with second client
        assert session_id in websocket_manager._session_subscriptions
        assert sid2 in websocket_manager._session_subscriptions[session_id]
        assert sid1 not in websocket_manager._session_subscriptions[session_id]


class TestWebSocketManagerEventBroadcasting:
    """Test event broadcasting to correct clients (Requirement 4.1)."""
    
    @pytest.mark.asyncio
    async def test_broadcast_to_session_subscribers(self, websocket_manager, mock_sio):
        """Test broadcasting events to session subscribers only."""
        websocket_manager.sio = mock_sio
        
        # Set up subscribers
        session_id = "session_123"
        sid1 = "client_1"
        sid2 = "client_2"
        
        websocket_manager._session_subscriptions[session_id] = {sid1, sid2}
        
        # Broadcast event
        event_data = {'message': 'test event'}
        await websocket_manager._broadcast_to_session(session_id, 'test_event', event_data)
        
        # Verify emit was called for each subscriber
        assert mock_sio.emit.call_count == 2
        
        # Verify correct event and data were sent
        for call in mock_sio.emit.call_args_list:
            assert call[0][0] == 'test_event'
            assert call[0][1] == event_data
            assert call[1]['room'] in {sid1, sid2}
    
    @pytest.mark.asyncio
    async def test_broadcast_to_session_with_no_subscribers(self, websocket_manager, mock_sio):
        """Test broadcasting to a session with no subscribers."""
        websocket_manager.sio = mock_sio
        
        # Broadcast to non-existent session
        session_id = "nonexistent_session"
        event_data = {'message': 'test event'}
        
        await websocket_manager._broadcast_to_session(session_id, 'test_event', event_data)
        
        # Verify no emit calls were made
        assert not mock_sio.emit.called
    
    @pytest.mark.asyncio
    async def test_broadcast_session_started(self, websocket_manager, mock_sio):
        """Test broadcasting session_started event."""
        websocket_manager.sio = mock_sio
        
        session_id = "session_123"
        websocket_manager._session_subscriptions[session_id] = {"client_1"}
        
        await websocket_manager.broadcast_session_started(
            session_id=session_id,
            model_id="model_456",
            model_name="test_model.pt",
            techniques=["quantization", "pruning"]
        )
        
        # Verify emit was called
        assert mock_sio.emit.called
        call_args = mock_sio.emit.call_args
        assert call_args[0][0] == 'session_started'
        
        event_data = call_args[0][1]
        assert event_data['session_id'] == session_id
        assert event_data['model_id'] == "model_456"
        assert event_data['model_name'] == "test_model.pt"
        assert event_data['techniques'] == ["quantization", "pruning"]
    
    @pytest.mark.asyncio
    async def test_broadcast_session_completed(self, websocket_manager, mock_sio):
        """Test broadcasting session_completed event."""
        websocket_manager.sio = mock_sio
        
        session_id = "session_123"
        websocket_manager._session_subscriptions[session_id] = {"client_1"}
        
        results = {
            'size_reduction_percent': 25.0,
            'speed_improvement_percent': 15.0
        }
        
        await websocket_manager.broadcast_session_completed(
            session_id=session_id,
            results=results
        )
        
        # Verify emit was called
        assert mock_sio.emit.called
        call_args = mock_sio.emit.call_args
        assert call_args[0][0] == 'session_completed'
        
        event_data = call_args[0][1]
        assert event_data['session_id'] == session_id
        assert event_data['results'] == results
    
    @pytest.mark.asyncio
    async def test_broadcast_session_failed(self, websocket_manager, mock_sio):
        """Test broadcasting session_failed event."""
        websocket_manager.sio = mock_sio
        
        session_id = "session_123"
        websocket_manager._session_subscriptions[session_id] = {"client_1"}
        
        await websocket_manager.broadcast_session_failed(
            session_id=session_id,
            error_message="Test error occurred",
            error_type="TestError"
        )
        
        # Verify emit was called
        assert mock_sio.emit.called
        call_args = mock_sio.emit.call_args
        assert call_args[0][0] == 'session_failed'
        
        event_data = call_args[0][1]
        assert event_data['session_id'] == session_id
        assert event_data['error_message'] == "Test error occurred"
        assert event_data['error_type'] == "TestError"
    
    @pytest.mark.asyncio
    async def test_broadcast_session_cancelled(self, websocket_manager, mock_sio):
        """Test broadcasting session_cancelled event."""
        websocket_manager.sio = mock_sio
        
        session_id = "session_123"
        websocket_manager._session_subscriptions[session_id] = {"client_1"}
        
        await websocket_manager.broadcast_session_cancelled(session_id=session_id)
        
        # Verify emit was called
        assert mock_sio.emit.called
        call_args = mock_sio.emit.call_args
        assert call_args[0][0] == 'session_cancelled'
        
        event_data = call_args[0][1]
        assert event_data['session_id'] == session_id

    @pytest.mark.asyncio
    async def test_broadcast_system_status(self, websocket_manager, mock_sio):
        """Test broadcasting system_status to all clients."""
        websocket_manager.sio = mock_sio
        
        # Add some connections
        websocket_manager._connections = {
            "client_1": {"connected_at": datetime.now(), "subscriptions": set()},
            "client_2": {"connected_at": datetime.now(), "subscriptions": set()}
        }
        
        status = {
            'active_sessions': 5,
            'total_alerts': 2,
            'unresolved_alerts': 1
        }
        
        await websocket_manager.broadcast_system_status(status)
        
        # Verify emit was called (broadcast to all)
        assert mock_sio.emit.called
        call_args = mock_sio.emit.call_args
        assert call_args[0][0] == 'system_status'
        
        event_data = call_args[0][1]
        assert event_data['status'] == status
    
    @pytest.mark.asyncio
    async def test_broadcast_to_multiple_subscribers(self, websocket_manager, mock_sio):
        """Test that events are broadcast to all subscribers of a session."""
        websocket_manager.sio = mock_sio
        
        # Set up multiple subscribers
        session_id = "session_123"
        subscribers = {"client_1", "client_2", "client_3"}
        websocket_manager._session_subscriptions[session_id] = subscribers
        
        event_data = {'progress': 50}
        await websocket_manager._broadcast_to_session(session_id, 'progress_update', event_data)
        
        # Verify emit was called for each subscriber
        assert mock_sio.emit.call_count == len(subscribers)


class TestWebSocketManagerErrorHandling:
    """Test connection cleanup on errors (Requirement 4.6)."""
    
    @pytest.mark.asyncio
    async def test_broadcast_handles_emit_errors_gracefully(self, websocket_manager, mock_sio):
        """Test that broadcast errors are handled gracefully without stopping delivery."""
        # Set up mock to fail on first call, succeed on second
        mock_sio.emit = AsyncMock(side_effect=[Exception("Network error"), None])
        websocket_manager.sio = mock_sio
        
        # Set up multiple subscribers
        session_id = "session_123"
        websocket_manager._session_subscriptions[session_id] = {"client_1", "client_2"}
        
        # Broadcast should not raise exception
        event_data = {'message': 'test'}
        await websocket_manager._broadcast_to_session(session_id, 'test_event', event_data)
        
        # Verify emit was attempted for both clients
        assert mock_sio.emit.call_count == 2
    
    @pytest.mark.asyncio
    async def test_disconnect_handles_missing_subscriptions(self, websocket_manager, mock_sio):
        """Test that disconnect handles connections without subscriptions."""
        websocket_manager.sio = mock_sio
        
        # Set up connection without subscriptions key
        sid = "test_sid_123"
        websocket_manager._connections[sid] = {
            'connected_at': datetime.now(),
            'user_id': 'user_123',
            'username': 'testuser',
            'role': 'user'
            # Note: no 'subscriptions' key
        }
        
        # Should not raise exception
        await websocket_manager._handle_disconnect(sid)
        
        # Verify connection was removed
        assert sid not in websocket_manager._connections
    
    @pytest.mark.asyncio
    async def test_session_event_broadcast_error_handling(self, websocket_manager, mock_sio):
        """Test that session event broadcasts handle errors gracefully."""
        # Set up mock to raise exception
        mock_sio.emit = AsyncMock(side_effect=Exception("Broadcast failed"))
        websocket_manager.sio = mock_sio
        
        session_id = "session_123"
        websocket_manager._session_subscriptions[session_id] = {"client_1"}
        
        # These should not raise exceptions
        await websocket_manager.broadcast_session_started(
            session_id=session_id,
            model_id="model_1",
            model_name="test.pt",
            techniques=["quantization"]
        )
        
        await websocket_manager.broadcast_session_completed(
            session_id=session_id,
            results={}
        )
        
        await websocket_manager.broadcast_session_failed(
            session_id=session_id,
            error_message="error",
            error_type="Error"
        )
        
        # If we reach here, errors were handled gracefully
        assert True
    
    @pytest.mark.asyncio
    async def test_system_status_broadcast_error_handling(self, websocket_manager, mock_sio):
        """Test that system status broadcast handles errors gracefully."""
        # Set up mock to raise exception
        mock_sio.emit = AsyncMock(side_effect=Exception("Broadcast failed"))
        websocket_manager.sio = mock_sio
        
        # Should not raise exception
        await websocket_manager.broadcast_system_status({'status': 'ok'})
        
        # If we reach here, error was handled gracefully
        assert True


class TestWebSocketManagerUtilityMethods:
    """Test utility methods for connection and subscription management."""
    
    def test_get_connection_count(self, websocket_manager):
        """Test getting the number of connected clients."""
        # Initially no connections
        assert websocket_manager.get_connection_count() == 0
        
        # Add connections
        websocket_manager._connections = {
            "client_1": {"connected_at": datetime.now()},
            "client_2": {"connected_at": datetime.now()},
            "client_3": {"connected_at": datetime.now()}
        }
        
        assert websocket_manager.get_connection_count() == 3

    def test_get_session_subscriber_count(self, websocket_manager):
        """Test getting the number of subscribers for a session."""
        session_id = "session_123"
        
        # Initially no subscribers
        assert websocket_manager.get_session_subscriber_count(session_id) == 0
        
        # Add subscribers
        websocket_manager._session_subscriptions[session_id] = {"client_1", "client_2"}
        
        assert websocket_manager.get_session_subscriber_count(session_id) == 2
    
    def test_get_stats(self, websocket_manager):
        """Test getting WebSocket manager statistics."""
        # Set up some connections and subscriptions
        websocket_manager._connections = {
            "client_1": {
                "connected_at": datetime.now(),
                "subscriptions": {"session_1", "session_2"}
            },
            "client_2": {
                "connected_at": datetime.now(),
                "subscriptions": {"session_1"}
            }
        }
        websocket_manager._session_subscriptions = {
            "session_1": {"client_1", "client_2"},
            "session_2": {"client_1"}
        }
        
        stats = websocket_manager.get_stats()
        
        # Verify stats
        assert stats['total_connections'] == 2
        assert stats['total_sessions_with_subscribers'] == 2
        assert stats['total_subscriptions'] == 3  # client_1 has 2, client_2 has 1
        assert stats['average_subscriptions_per_connection'] == 1.5
        assert stats['notification_service_connected'] is False
        assert 'timestamp' in stats
    
    def test_is_notification_service_connected(self, websocket_manager):
        """Test checking if NotificationService is connected."""
        # Initially not connected
        assert websocket_manager.is_notification_service_connected() is False
        
        # Set notification service
        mock_notification_service = Mock()
        websocket_manager._notification_service = mock_notification_service
        
        assert websocket_manager.is_notification_service_connected() is True
    
    def test_get_notification_service(self, websocket_manager):
        """Test getting the NotificationService instance."""
        # Initially None
        assert websocket_manager.get_notification_service() is None
        
        # Set notification service
        mock_notification_service = Mock()
        websocket_manager._notification_service = mock_notification_service
        
        assert websocket_manager.get_notification_service() == mock_notification_service


class TestWebSocketManagerThreadSafety:
    """Test thread-safe operations with concurrent access."""
    
    @pytest.mark.asyncio
    async def test_concurrent_subscriptions(self, websocket_manager, mock_sio):
        """Test that concurrent subscriptions are handled safely."""
        websocket_manager.sio = mock_sio
        
        # Set up connections
        for i in range(5):
            sid = f"client_{i}"
            websocket_manager._connections[sid] = {
                'connected_at': datetime.now(),
                'user_id': f'user_{i}',
                'username': f'user{i}',
                'role': 'user',
                'subscriptions': set()
            }
        
        # Subscribe all clients to same session concurrently
        session_id = "session_123"
        tasks = []
        for i in range(5):
            sid = f"client_{i}"
            task = websocket_manager._handle_subscribe_session(sid, {'session_id': session_id})
            tasks.append(task)
        
        await asyncio.gather(*tasks)
        
        # Verify all subscriptions were recorded
        assert len(websocket_manager._session_subscriptions[session_id]) == 5
        for i in range(5):
            sid = f"client_{i}"
            assert session_id in websocket_manager._connections[sid]['subscriptions']
    
    @pytest.mark.asyncio
    async def test_concurrent_disconnections(self, websocket_manager, mock_sio):
        """Test that concurrent disconnections are handled safely."""
        websocket_manager.sio = mock_sio
        
        # Set up connections with subscriptions
        session_id = "session_123"
        for i in range(5):
            sid = f"client_{i}"
            websocket_manager._connections[sid] = {
                'connected_at': datetime.now(),
                'user_id': f'user_{i}',
                'username': f'user{i}',
                'role': 'user',
                'subscriptions': {session_id}
            }
        websocket_manager._session_subscriptions[session_id] = {f"client_{i}" for i in range(5)}
        
        # Disconnect all clients concurrently
        tasks = []
        for i in range(5):
            sid = f"client_{i}"
            task = websocket_manager._handle_disconnect(sid)
            tasks.append(task)
        
        await asyncio.gather(*tasks)
        
        # Verify all connections were removed
        assert len(websocket_manager._connections) == 0
        assert session_id not in websocket_manager._session_subscriptions


class TestWebSocketManagerSingleton:
    """Test singleton pattern implementation."""
    
    def test_singleton_instance(self):
        """Test that WebSocketManager follows singleton pattern."""
        manager1 = WebSocketManager()
        manager2 = WebSocketManager()
        
        # Both should be the same instance
        assert manager1 is manager2
    
    def test_singleton_state_persistence(self):
        """Test that singleton state persists across instances."""
        manager1 = WebSocketManager()
        manager1._connections["test_client"] = {"test": "data"}
        
        manager2 = WebSocketManager()
        
        # State should be shared
        assert "test_client" in manager2._connections
        assert manager2._connections["test_client"]["test"] == "data"
        
        # Clean up
        manager1._connections.clear()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

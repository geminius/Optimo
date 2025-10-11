"""
WebSocket and NotificationService Integration Demo.

This script demonstrates how the WebSocketManager integrates with
NotificationService to provide real-time updates to connected clients.
"""

import asyncio
import logging
from datetime import datetime

from src.services.websocket_manager import WebSocketManager
from src.services.notification_service import (
    NotificationService,
    NotificationType,
    AlertSeverity
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def demo_integration():
    """Demonstrate WebSocket and NotificationService integration."""
    
    logger.info("=" * 80)
    logger.info("WebSocket and NotificationService Integration Demo")
    logger.info("=" * 80)
    
    # Create instances
    notification_service = NotificationService()
    websocket_manager = WebSocketManager()
    
    # Setup integration
    logger.info("\n1. Setting up integration...")
    websocket_manager.setup_notification_handlers(notification_service)
    
    # Verify connection
    if websocket_manager.is_notification_service_connected():
        logger.info("✓ WebSocketManager successfully connected to NotificationService")
    else:
        logger.error("✗ Failed to connect WebSocketManager to NotificationService")
        return
    
    # Simulate client connections
    logger.info("\n2. Simulating client connections...")
    session_id = "demo_session_123"
    
    # Add mock subscribers (in real scenario, these would be actual WebSocket clients)
    websocket_manager._connections["client_1"] = {
        "connected_at": datetime.now(),
        "user_id": "user_1",
        "subscriptions": {session_id}
    }
    websocket_manager._connections["client_2"] = {
        "connected_at": datetime.now(),
        "user_id": "user_2",
        "subscriptions": {session_id}
    }
    websocket_manager._session_subscriptions[session_id] = {"client_1", "client_2"}
    
    logger.info(f"✓ Simulated 2 clients subscribed to session: {session_id}")
    
    # Demo 1: Send notification
    logger.info("\n3. Sending notification...")
    notification_id = notification_service.send_notification(
        type=NotificationType.INFO,
        title="Optimization Started",
        message="Your model optimization has begun",
        session_id=session_id,
        metadata={"model_id": "model_456"}
    )
    logger.info(f"✓ Notification sent: {notification_id}")
    await asyncio.sleep(0.1)  # Allow async processing
    
    # Demo 2: Create alert
    logger.info("\n4. Creating alert...")
    alert_id = notification_service.create_alert(
        severity=AlertSeverity.MEDIUM,
        title="High Memory Usage",
        description="System memory usage is at 85%",
        session_id=session_id
    )
    logger.info(f"✓ Alert created: {alert_id}")
    await asyncio.sleep(0.1)
    
    # Demo 3: Progress tracking
    logger.info("\n5. Tracking progress...")
    notification_service.start_progress_tracking(
        session_id=session_id,
        total_steps=5,
        initial_step_name="Initializing"
    )
    logger.info("✓ Progress tracking started")
    await asyncio.sleep(0.1)
    
    # Update progress
    for step in range(1, 6):
        notification_service.update_progress(
            session_id=session_id,
            current_step=step,
            step_name=f"Step {step}: Processing",
            metadata={"detail": f"Processing step {step} of 5"}
        )
        logger.info(f"✓ Progress updated: {step}/5 ({step * 20}%)")
        await asyncio.sleep(0.1)
    
    # Complete progress
    notification_service.complete_progress_tracking(session_id)
    logger.info("✓ Progress tracking completed")
    await asyncio.sleep(0.1)
    
    # Demo 4: Session lifecycle events
    logger.info("\n6. Broadcasting session lifecycle events...")
    
    # Session started
    await websocket_manager.broadcast_session_started(
        session_id=session_id,
        model_id="model_456",
        model_name="robotics_vla_model.pt",
        techniques=["quantization", "pruning"]
    )
    logger.info("✓ Session started event broadcast")
    
    # Session completed
    await websocket_manager.broadcast_session_completed(
        session_id=session_id,
        results={
            "size_reduction_percent": 25.0,
            "speed_improvement_percent": 15.0,
            "accuracy_change_percent": -0.5
        }
    )
    logger.info("✓ Session completed event broadcast")
    
    # Demo 5: System status
    logger.info("\n7. Broadcasting system status...")
    system_status = notification_service.get_system_status()
    await websocket_manager.broadcast_system_status(system_status)
    logger.info("✓ System status broadcast")
    
    # Show statistics
    logger.info("\n8. Integration Statistics:")
    ws_stats = websocket_manager.get_stats()
    logger.info(f"   - Total connections: {ws_stats['total_connections']}")
    logger.info(f"   - Sessions with subscribers: {ws_stats['total_sessions_with_subscribers']}")
    logger.info(f"   - Total subscriptions: {ws_stats['total_subscriptions']}")
    logger.info(f"   - NotificationService connected: {ws_stats['notification_service_connected']}")
    
    ns_stats = notification_service.get_system_status()
    logger.info(f"   - Active sessions: {ns_stats['active_sessions']}")
    logger.info(f"   - Total notifications: {ns_stats['total_notifications']}")
    logger.info(f"   - Total alerts: {ns_stats['total_alerts']}")
    logger.info(f"   - Unresolved alerts: {ns_stats['unresolved_alerts']}")
    
    logger.info("\n" + "=" * 80)
    logger.info("Demo completed successfully!")
    logger.info("=" * 80)
    
    # Key takeaways
    logger.info("\nKey Takeaways:")
    logger.info("1. WebSocketManager automatically receives events from NotificationService")
    logger.info("2. Events are transformed and broadcast to appropriate clients")
    logger.info("3. Session-specific events go to subscribed clients only")
    logger.info("4. System-wide events go to all connected clients")
    logger.info("5. All errors are handled gracefully with logging")
    logger.info("6. The integration is fully asynchronous and non-blocking")


async def demo_error_handling():
    """Demonstrate error handling in the integration."""
    
    logger.info("\n" + "=" * 80)
    logger.info("Error Handling Demo")
    logger.info("=" * 80)
    
    notification_service = NotificationService()
    websocket_manager = WebSocketManager()
    websocket_manager.setup_notification_handlers(notification_service)
    
    # Simulate a broadcast error by using a session with no subscribers
    logger.info("\n1. Testing broadcast to session with no subscribers...")
    notification_service.send_notification(
        type=NotificationType.INFO,
        title="Test",
        message="This session has no subscribers",
        session_id="nonexistent_session"
    )
    await asyncio.sleep(0.1)
    logger.info("✓ Handled gracefully - no errors raised")
    
    # Test with invalid data
    logger.info("\n2. Testing with various notification types...")
    for notif_type in NotificationType:
        notification_service.send_notification(
            type=notif_type,
            title=f"Test {notif_type.value}",
            message=f"Testing {notif_type.value} notification"
        )
        await asyncio.sleep(0.05)
    logger.info("✓ All notification types handled correctly")
    
    logger.info("\n" + "=" * 80)
    logger.info("Error handling demo completed!")
    logger.info("=" * 80)


async def main():
    """Run all demos."""
    try:
        await demo_integration()
        await demo_error_handling()
    except Exception as e:
        logger.error(f"Demo failed: {e}", exc_info=True)


if __name__ == "__main__":
    asyncio.run(main())

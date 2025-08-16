"""
Demo script for NotificationService and MonitoringService.

This script demonstrates the notification and monitoring capabilities
of the robotics model optimization platform.
"""

import time
import threading
from datetime import datetime

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.services.notification_service import (
    NotificationService,
    NotificationType,
    AlertSeverity
)
from src.services.monitoring_service import (
    MonitoringService,
    HealthStatus,
    HealthCheck
)


def demo_notification_service():
    """Demonstrate NotificationService functionality."""
    print("=== NotificationService Demo ===")
    
    # Create notification service
    notification_service = NotificationService()
    
    # Set up event subscribers
    def on_notification(notification):
        print(f"📢 Notification: [{notification.type.value}] {notification.title}")
    
    def on_alert(alert):
        print(f"🚨 Alert: [{alert.severity.value}] {alert.title}")
    
    def on_progress(progress):
        print(f"📊 Progress: {progress.session_id} - {progress.progress_percentage:.1f}% - {progress.step_name}")
    
    # Subscribe to events
    notification_service.subscribe('notification', on_notification)
    notification_service.subscribe('alert', on_alert)
    notification_service.subscribe('progress', on_progress)
    
    print("\n1. Sending notifications...")
    notification_service.send_notification(
        NotificationType.INFO,
        "Model Upload Started",
        "OpenVLA model upload initiated",
        session_id="demo_session_1"
    )
    
    notification_service.send_notification(
        NotificationType.SUCCESS,
        "Model Upload Complete",
        "OpenVLA model successfully uploaded and validated",
        session_id="demo_session_1"
    )
    
    print("\n2. Creating alerts...")
    alert_id = notification_service.create_alert(
        AlertSeverity.MEDIUM,
        "High Memory Usage",
        "Memory usage is approaching 80% threshold",
        session_id="demo_session_1"
    )
    
    notification_service.create_alert(
        AlertSeverity.HIGH,
        "Optimization Failed",
        "Quantization optimization failed due to model incompatibility",
        session_id="demo_session_1"
    )
    
    print("\n3. Progress tracking...")
    session_id = "optimization_session_1"
    notification_service.start_progress_tracking(session_id, 5, "Initializing optimization")
    
    steps = [
        "Analyzing model architecture",
        "Identifying optimization opportunities",
        "Applying quantization",
        "Validating optimized model",
        "Generating evaluation report"
    ]
    
    for i, step in enumerate(steps, 1):
        time.sleep(0.5)  # Simulate work
        notification_service.update_progress(session_id, i, step)
    
    notification_service.complete_progress_tracking(session_id)
    
    print("\n4. Resolving alert...")
    notification_service.resolve_alert(alert_id)
    print(f"Alert {alert_id} resolved")
    
    print("\n5. System status...")
    status = notification_service.get_system_status()
    print(f"Active sessions: {status['active_sessions']}")
    print(f"Total notifications: {status['total_notifications']}")
    print(f"Unresolved alerts: {status['unresolved_alerts']}")
    
    return notification_service


def demo_monitoring_service(notification_service):
    """Demonstrate MonitoringService functionality."""
    print("\n\n=== MonitoringService Demo ===")
    
    # Create monitoring service
    monitoring_service = MonitoringService(notification_service)
    
    print("\n1. Collecting performance metrics...")
    current_metrics = monitoring_service._collect_performance_metrics()
    print(f"CPU Usage: {current_metrics.cpu_usage_percent:.1f}%")
    print(f"Memory Usage: {current_metrics.memory_usage_percent:.1f}%")
    print(f"Disk Usage: {current_metrics.disk_usage_percent:.1f}%")
    print(f"Available Memory: {current_metrics.memory_available_mb:.0f} MB")
    print(f"Free Disk Space: {current_metrics.disk_free_gb:.1f} GB")
    
    print("\n2. Running health checks...")
    monitoring_service._run_health_checks()
    health_status = monitoring_service.get_health_status()
    
    for component, health_check in health_status.items():
        status_emoji = {
            HealthStatus.HEALTHY: "✅",
            HealthStatus.WARNING: "⚠️",
            HealthStatus.CRITICAL: "❌",
            HealthStatus.UNKNOWN: "❓"
        }.get(health_check.status, "❓")
        
        print(f"{status_emoji} {component}: {health_check.status.value} - {health_check.message}")
    
    print(f"\nOverall Health: {monitoring_service.get_overall_health().value}")
    
    print("\n3. Registering custom health check...")
    def custom_optimization_health():
        # Simulate checking optimization service health
        return HealthCheck(
            component="optimization_service",
            status=HealthStatus.HEALTHY,
            message="Optimization service is running normally",
            timestamp=datetime.now(),
            details={"active_optimizations": 2, "queue_size": 0}
        )
    
    monitoring_service.register_health_check("optimization_service", custom_optimization_health)
    monitoring_service._run_health_checks()
    
    updated_health = monitoring_service.get_health_status()
    if "optimization_service" in updated_health:
        print(f"✅ Custom health check registered and executed successfully")
    
    print("\n4. Updating performance thresholds...")
    monitoring_service.set_thresholds(
        cpu_warning=70.0,
        cpu_critical=90.0,
        memory_warning=75.0,
        memory_critical=90.0
    )
    print("Performance thresholds updated")
    
    print("\n5. Simulating monitoring loop (short demo)...")
    # Start monitoring for a short period
    monitoring_service.start_monitoring(interval=2)
    print("Monitoring started... (will run for 6 seconds)")
    
    # Let it run for a few cycles
    time.sleep(6)
    
    # Stop monitoring
    monitoring_service.stop_monitoring()
    print("Monitoring stopped")
    
    # Show collected metrics
    metrics_history = monitoring_service.get_metrics_history(hours=1, limit=5)
    print(f"\nCollected {len(metrics_history)} metrics during demo")
    
    return monitoring_service


def demo_integration_scenario():
    """Demonstrate integration scenario with both services."""
    print("\n\n=== Integration Scenario Demo ===")
    print("Simulating a complete optimization workflow with monitoring...")
    
    # Create services
    notification_service = NotificationService()
    monitoring_service = MonitoringService(notification_service)
    
    # Set up logging
    def log_event(event):
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] {event}")
    
    # Subscribe to all events
    notification_service.subscribe('notification', 
        lambda n: log_event(f"📢 {n.title}: {n.message}"))
    notification_service.subscribe('alert', 
        lambda a: log_event(f"🚨 {a.title}: {a.description}"))
    notification_service.subscribe('progress', 
        lambda p: log_event(f"📊 {p.step_name} ({p.progress_percentage:.0f}%)"))
    
    # Start monitoring
    monitoring_service.start_monitoring(interval=3)
    
    # Simulate optimization workflow
    session_id = "integration_demo"
    
    # Start optimization
    notification_service.send_notification(
        NotificationType.INFO,
        "Optimization Started",
        "Starting optimization workflow for OpenVLA model",
        session_id=session_id
    )
    
    # Track progress through optimization steps
    optimization_steps = [
        "Loading model",
        "Analyzing architecture", 
        "Planning optimization strategy",
        "Applying quantization",
        "Running evaluation",
        "Generating report"
    ]
    
    notification_service.start_progress_tracking(session_id, len(optimization_steps))
    
    for i, step in enumerate(optimization_steps):
        time.sleep(1)  # Simulate work
        notification_service.update_progress(session_id, i + 1, step)
        
        # Simulate some issues during optimization
        if step == "Applying quantization":
            notification_service.create_alert(
                AlertSeverity.MEDIUM,
                "Quantization Warning",
                "Some layers may not benefit from quantization",
                session_id=session_id
            )
        elif step == "Running evaluation":
            notification_service.send_notification(
                NotificationType.WARNING,
                "Performance Drop Detected",
                "Optimized model shows 2% accuracy decrease",
                session_id=session_id
            )
    
    # Complete optimization
    notification_service.complete_progress_tracking(session_id)
    notification_service.send_notification(
        NotificationType.SUCCESS,
        "Optimization Complete",
        "Model optimization completed successfully with 40% size reduction",
        session_id=session_id
    )
    
    # Let monitoring run a bit more
    time.sleep(3)
    
    # Stop monitoring
    monitoring_service.stop_monitoring()
    
    # Show final status
    print(f"\n--- Final Status ---")
    system_status = notification_service.get_system_status()
    print(f"Total notifications: {system_status['total_notifications']}")
    print(f"Total alerts: {system_status['total_alerts']}")
    print(f"Unresolved alerts: {system_status['unresolved_alerts']}")
    
    overall_health = monitoring_service.get_overall_health()
    print(f"System health: {overall_health.value}")


def main():
    """Run all demos."""
    print("🤖 Robotics Model Optimization Platform")
    print("Notification and Monitoring Services Demo")
    print("=" * 50)
    
    try:
        # Demo individual services
        notification_service = demo_notification_service()
        monitoring_service = demo_monitoring_service(notification_service)
        
        # Demo integration
        demo_integration_scenario()
        
        print("\n✅ Demo completed successfully!")
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        raise


if __name__ == "__main__":
    main()
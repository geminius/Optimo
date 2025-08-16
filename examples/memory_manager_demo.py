"""
Demo script for MemoryManager functionality.

This script demonstrates how to use the MemoryManager for optimization
history, session persistence, and audit logging.
"""

import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
import tempfile
import shutil

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.services.memory_manager import MemoryManager, AuditLogEntry
from src.models.core import (
    OptimizationSession, OptimizationStep, OptimizationResults,
    OptimizationStatus, ModelMetadata
)


def create_sample_session(session_id: str, model_id: str) -> OptimizationSession:
    """Create a sample optimization session for demonstration."""
    session = OptimizationSession(
        id=session_id,
        model_id=model_id,
        status=OptimizationStatus.RUNNING,
        criteria_name="performance_optimization",
        plan_id=f"plan-{session_id}",
        created_by="demo_user",
        priority=1,
        tags=["robotics", "openvla", "demo"],
        notes="Demo optimization session for MemoryManager testing"
    )
    
    # Add optimization steps
    step1 = OptimizationStep(
        step_id="quantization-step",
        technique="quantization",
        status=OptimizationStatus.COMPLETED,
        start_time=datetime.now() - timedelta(minutes=30),
        end_time=datetime.now() - timedelta(minutes=25),
        duration_seconds=300.0,
        parameters={
            "bits": 8,
            "method": "dynamic",
            "calibration_samples": 1000
        },
        results={
            "size_reduction_percent": 45.2,
            "accuracy_retention": 0.98,
            "inference_speedup": 1.3
        }
    )
    
    step2 = OptimizationStep(
        step_id="pruning-step",
        technique="pruning",
        status=OptimizationStatus.RUNNING,
        start_time=datetime.now() - timedelta(minutes=25),
        parameters={
            "sparsity_ratio": 0.3,
            "structured": True,
            "gradual_pruning": True
        }
    )
    
    session.steps = [step1, step2]
    
    # Add partial results
    session.results = OptimizationResults(
        original_model_size_mb=250.5,
        optimized_model_size_mb=137.3,
        size_reduction_percent=45.2,
        performance_improvements={
            "inference_time_ms": 1.3,
            "memory_usage_reduction": 0.4
        },
        accuracy_metrics={
            "task_accuracy": 0.98,
            "robustness_score": 0.95
        },
        optimization_summary="Quantization completed successfully, pruning in progress",
        techniques_applied=["quantization"],
        rollback_available=True,
        validation_passed=True
    )
    
    return session


def demonstrate_basic_operations():
    """Demonstrate basic MemoryManager operations."""
    print("=== Basic MemoryManager Operations ===")
    
    # Create temporary directory for demo
    temp_dir = tempfile.mkdtemp()
    db_path = Path(temp_dir) / "demo_optimization.db"
    
    try:
        # Initialize MemoryManager
        config = {
            "database_path": str(db_path),
            "backup_interval_hours": 24,
            "retention_days": 90,
            "max_session_size_mb": 50,
            "cache_max_size": 100,
            "cache_ttl_minutes": 30,
            "audit_enabled": True,
            "audit_retention_days": 365
        }
        
        memory_manager = MemoryManager(config)
        print(f"✓ MemoryManager created with database: {db_path}")
        
        # Initialize database
        if memory_manager.initialize():
            print("✓ MemoryManager initialized successfully")
        else:
            print("✗ Failed to initialize MemoryManager")
            return
        
        # Create and store sample sessions
        session1 = create_sample_session("demo-session-001", "openvla-model-v1")
        session2 = create_sample_session("demo-session-002", "openvla-model-v2")
        session2.status = OptimizationStatus.COMPLETED
        
        print(f"\n--- Storing Sessions ---")
        if memory_manager.store_session(session1):
            print(f"✓ Stored session: {session1.id}")
        else:
            print(f"✗ Failed to store session: {session1.id}")
        
        if memory_manager.store_session(session2):
            print(f"✓ Stored session: {session2.id}")
        else:
            print(f"✗ Failed to store session: {session2.id}")
        
        # Retrieve sessions
        print(f"\n--- Retrieving Sessions ---")
        retrieved1 = memory_manager.retrieve_session(session1.id)
        if retrieved1:
            print(f"✓ Retrieved session: {retrieved1.id}")
            print(f"  Status: {retrieved1.status.value}")
            print(f"  Steps: {len(retrieved1.steps)}")
            print(f"  Model ID: {retrieved1.model_id}")
        else:
            print(f"✗ Failed to retrieve session: {session1.id}")
        
        # List sessions
        print(f"\n--- Listing Sessions ---")
        all_sessions = memory_manager.list_sessions()
        print(f"Total sessions: {len(all_sessions)}")
        for session_info in all_sessions:
            print(f"  - {session_info['id']}: {session_info['status']} ({session_info['model_id']})")
        
        # Filter sessions by status
        running_sessions = memory_manager.list_sessions(status_filter=["running"])
        print(f"Running sessions: {len(running_sessions)}")
        
        # Get session statistics
        print(f"\n--- Session Statistics ---")
        stats = memory_manager.get_session_statistics()
        print(f"Total sessions: {stats['total_sessions']}")
        print(f"Status distribution: {stats['status_distribution']}")
        print(f"Total storage size: {stats['total_storage_size_mb']:.2f} MB")
        print(f"Cache size: {stats['cache_size']}")
        
        return memory_manager, session1, session2
        
    except Exception as e:
        print(f"✗ Error in basic operations: {e}")
        return None, None, None
    
    finally:
        # Note: We don't cleanup here as we return the manager for further demos
        pass


def demonstrate_audit_logging(memory_manager, session1):
    """Demonstrate audit logging functionality."""
    print("\n=== Audit Logging Demonstration ===")
    
    try:
        # Log various audit entries
        audit_entries = [
            {
                "session_id": session1.id,
                "event_type": "analysis_started",
                "component": "analysis_agent",
                "details": {
                    "model_path": "/models/openvla-v1.pth",
                    "analysis_type": "architecture_profiling"
                },
                "user_id": "demo_user"
            },
            {
                "session_id": session1.id,
                "event_type": "optimization_decision",
                "component": "planning_agent",
                "details": {
                    "technique_selected": "quantization",
                    "confidence_score": 0.85,
                    "expected_improvement": 0.4
                },
                "user_id": "demo_user"
            },
            {
                "session_id": session1.id,
                "event_type": "step_completed",
                "component": "quantization_agent",
                "details": {
                    "technique": "quantization",
                    "success": True,
                    "metrics": {
                        "size_reduction": 0.45,
                        "accuracy_retention": 0.98
                    }
                }
            }
        ]
        
        print("--- Logging Audit Entries ---")
        for entry in audit_entries:
            success = memory_manager.log_audit_entry(**entry)
            if success:
                print(f"✓ Logged: {entry['event_type']} from {entry['component']}")
            else:
                print(f"✗ Failed to log: {entry['event_type']}")
        
        # Retrieve audit logs
        print(f"\n--- Retrieving Audit Logs ---")
        
        # Get all logs for session
        session_logs = memory_manager.get_audit_logs(session_id=session1.id)
        print(f"Total audit logs for session {session1.id}: {len(session_logs)}")
        
        for log in session_logs:
            print(f"  - {log.timestamp.strftime('%H:%M:%S')}: {log.event_type} ({log.component})")
            if log.details:
                for key, value in log.details.items():
                    print(f"    {key}: {value}")
        
        # Filter by event type
        decision_logs = memory_manager.get_audit_logs(
            session_id=session1.id,
            event_type="optimization_decision"
        )
        print(f"\nOptimization decision logs: {len(decision_logs)}")
        
        # Filter by component
        agent_logs = memory_manager.get_audit_logs(
            session_id=session1.id,
            component="planning_agent"
        )
        print(f"Planning agent logs: {len(agent_logs)}")
        
    except Exception as e:
        print(f"✗ Error in audit logging demo: {e}")


def demonstrate_session_recovery(memory_manager, session1):
    """Demonstrate session recovery functionality."""
    print("\n=== Session Recovery Demonstration ===")
    
    try:
        # Simulate a failed session
        print("--- Simulating Session Failure ---")
        failed_session = session1
        failed_session.status = OptimizationStatus.FAILED
        
        # Add a failed step
        failed_step = OptimizationStep(
            step_id="failed-step",
            technique="distillation",
            status=OptimizationStatus.FAILED,
            start_time=datetime.now() - timedelta(minutes=10),
            end_time=datetime.now() - timedelta(minutes=5),
            error_message="Distillation failed due to teacher model incompatibility"
        )
        failed_session.steps.append(failed_step)
        
        # Store the failed session
        memory_manager.store_session(failed_session)
        print(f"✓ Stored failed session: {failed_session.id}")
        
        # Get recovery information
        print(f"\n--- Getting Recovery Information ---")
        recovery_info = memory_manager.get_session_recovery_info(failed_session.id)
        
        if recovery_info:
            print(f"✓ Recovery info available for session: {recovery_info.session_id}")
            print(f"  Last checkpoint: {recovery_info.last_checkpoint}")
            print(f"  Recoverable steps: {len(recovery_info.recoverable_steps)}")
            print(f"  Status at failure: {recovery_info.status_at_failure}")
            
            for step_id in recovery_info.recoverable_steps:
                print(f"    - {step_id}")
        else:
            print(f"✗ No recovery info available for session: {failed_session.id}")
            return
        
        # Attempt recovery
        print(f"\n--- Attempting Session Recovery ---")
        recovered_session = memory_manager.recover_session(failed_session.id)
        
        if recovered_session:
            print(f"✓ Successfully recovered session: {recovered_session.id}")
            print(f"  New status: {recovered_session.status.value}")
            
            # Check step statuses
            print("  Step statuses after recovery:")
            for step in recovered_session.steps:
                print(f"    - {step.step_id}: {step.status.value}")
        else:
            print(f"✗ Failed to recover session: {failed_session.id}")
        
    except Exception as e:
        print(f"✗ Error in session recovery demo: {e}")


def demonstrate_data_management(memory_manager):
    """Demonstrate data management features."""
    print("\n=== Data Management Demonstration ===")
    
    try:
        # Create backup
        print("--- Creating Database Backup ---")
        backup_path = Path(tempfile.gettempdir()) / "demo_backup.db"
        
        if memory_manager.backup_database(backup_path):
            print(f"✓ Database backup created: {backup_path}")
            print(f"  Backup size: {backup_path.stat().st_size} bytes")
        else:
            print("✗ Failed to create database backup")
        
        # Cleanup old data (demo with very short retention)
        print(f"\n--- Data Cleanup Demonstration ---")
        cleanup_stats = memory_manager.cleanup_old_data(retention_days=0)  # Clean everything for demo
        print(f"Cleanup statistics:")
        print(f"  Sessions deleted: {cleanup_stats.get('sessions_deleted', 0)}")
        print(f"  Audit logs deleted: {cleanup_stats.get('audit_logs_deleted', 0)}")
        
        # Show updated statistics
        print(f"\n--- Updated Statistics ---")
        stats = memory_manager.get_session_statistics()
        print(f"Total sessions after cleanup: {stats['total_sessions']}")
        
        # Restore from backup if we have one
        if backup_path.exists():
            print(f"\n--- Restoring from Backup ---")
            if memory_manager.restore_database(backup_path):
                print("✓ Database restored from backup")
                
                # Verify restoration
                restored_stats = memory_manager.get_session_statistics()
                print(f"Sessions after restore: {restored_stats['total_sessions']}")
            else:
                print("✗ Failed to restore from backup")
        
        # Cleanup backup file
        if backup_path.exists():
            backup_path.unlink()
            print(f"✓ Cleaned up backup file: {backup_path}")
        
    except Exception as e:
        print(f"✗ Error in data management demo: {e}")


def demonstrate_advanced_queries(memory_manager):
    """Demonstrate advanced query capabilities."""
    print("\n=== Advanced Query Demonstration ===")
    
    try:
        # Create sessions with different characteristics for querying
        print("--- Creating Test Data for Queries ---")
        
        # Recent session
        recent_session = create_sample_session("recent-session", "model-recent")
        recent_session.status = OptimizationStatus.COMPLETED
        memory_manager.store_session(recent_session)
        
        # Old session (simulate by creating then updating timestamp)
        old_session = create_sample_session("old-session", "model-old")
        old_session.status = OptimizationStatus.COMPLETED
        memory_manager.store_session(old_session)
        
        print("✓ Created test sessions for querying")
        
        # Query by different criteria
        print(f"\n--- Querying Sessions ---")
        
        # All sessions
        all_sessions = memory_manager.list_sessions()
        print(f"Total sessions: {len(all_sessions)}")
        
        # Completed sessions only
        completed_sessions = memory_manager.list_sessions(
            status_filter=["completed"]
        )
        print(f"Completed sessions: {len(completed_sessions)}")
        
        # Sessions for specific model
        model_sessions = memory_manager.list_sessions(
            model_id_filter="model-recent"
        )
        print(f"Sessions for 'model-recent': {len(model_sessions)}")
        
        # Paginated results
        page1 = memory_manager.list_sessions(limit=2, offset=0)
        page2 = memory_manager.list_sessions(limit=2, offset=2)
        print(f"Page 1 sessions: {len(page1)}")
        print(f"Page 2 sessions: {len(page2)}")
        
        # Audit log queries
        print(f"\n--- Querying Audit Logs ---")
        
        # Recent audit logs
        recent_logs = memory_manager.get_audit_logs(
            start_time=datetime.now() - timedelta(hours=1),
            limit=10
        )
        print(f"Recent audit logs (last hour): {len(recent_logs)}")
        
        # Logs by component
        component_logs = memory_manager.get_audit_logs(
            component="memory_manager",
            limit=5
        )
        print(f"Memory manager logs: {len(component_logs)}")
        
    except Exception as e:
        print(f"✗ Error in advanced queries demo: {e}")


def main():
    """Run the complete MemoryManager demonstration."""
    print("MemoryManager Demonstration")
    print("=" * 50)
    
    # Run basic operations demo
    memory_manager, session1, session2 = demonstrate_basic_operations()
    
    if not memory_manager:
        print("Failed to initialize MemoryManager. Exiting demo.")
        return
    
    try:
        # Run other demonstrations
        demonstrate_audit_logging(memory_manager, session1)
        demonstrate_session_recovery(memory_manager, session1)
        demonstrate_data_management(memory_manager)
        demonstrate_advanced_queries(memory_manager)
        
        print("\n" + "=" * 50)
        print("✓ MemoryManager demonstration completed successfully!")
        print("\nKey features demonstrated:")
        print("  - Session storage and retrieval")
        print("  - Audit logging and querying")
        print("  - Session recovery capabilities")
        print("  - Database backup and restore")
        print("  - Data cleanup and maintenance")
        print("  - Advanced querying and filtering")
        
    except Exception as e:
        print(f"\n✗ Demo failed with error: {e}")
        
    finally:
        # Cleanup
        if memory_manager:
            memory_manager.cleanup()
            print("\n✓ MemoryManager cleanup completed")


if __name__ == "__main__":
    main()
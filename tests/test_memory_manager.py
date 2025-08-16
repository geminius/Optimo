"""
Unit tests for MemoryManager class.
"""

import pytest
import tempfile
import shutil
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import json

from src.services.memory_manager import MemoryManager, AuditLogEntry, SessionRecoveryInfo
from src.models.core import (
    OptimizationSession, OptimizationStep, OptimizationResults,
    OptimizationStatus, ModelMetadata
)


class TestMemoryManager:
    """Test cases for MemoryManager class."""
    
    @pytest.fixture
    def temp_db_path(self):
        """Create temporary database path for testing."""
        temp_dir = tempfile.mkdtemp()
        db_path = Path(temp_dir) / "test_optimization.db"
        yield db_path
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def memory_manager_config(self, temp_db_path):
        """Create test configuration for MemoryManager."""
        return {
            "database_path": str(temp_db_path),
            "backup_interval_hours": 1,
            "retention_days": 30,
            "max_session_size_mb": 10,
            "cache_max_size": 50,
            "cache_ttl_minutes": 15,
            "audit_enabled": True,
            "audit_retention_days": 90
        }
    
    @pytest.fixture
    def memory_manager(self, memory_manager_config):
        """Create MemoryManager instance for testing."""
        manager = MemoryManager(memory_manager_config)
        assert manager.initialize()
        yield manager
        manager.cleanup()
    
    @pytest.fixture
    def sample_session(self):
        """Create sample optimization session for testing."""
        session = OptimizationSession(
            id="test-session-123",
            model_id="test-model-456",
            status=OptimizationStatus.PENDING,
            criteria_name="test-criteria",
            plan_id="test-plan-789",
            created_by="test-user",
            priority=1,
            tags=["test", "robotics"],
            notes="Test session for unit tests"
        )
        
        # Add some steps
        step1 = OptimizationStep(
            step_id="step-1",
            technique="quantization",
            status=OptimizationStatus.COMPLETED,
            start_time=datetime.now() - timedelta(minutes=10),
            end_time=datetime.now() - timedelta(minutes=5),
            duration_seconds=300.0,
            parameters={"bits": 8, "method": "dynamic"},
            results={"size_reduction": 0.5, "accuracy_loss": 0.02}
        )
        
        step2 = OptimizationStep(
            step_id="step-2",
            technique="pruning",
            status=OptimizationStatus.FAILED,
            start_time=datetime.now() - timedelta(minutes=5),
            end_time=datetime.now() - timedelta(minutes=2),
            duration_seconds=180.0,
            parameters={"sparsity": 0.3, "structured": True},
            error_message="Pruning failed due to incompatible architecture"
        )
        
        session.steps = [step1, step2]
        
        # Add results
        session.results = OptimizationResults(
            original_model_size_mb=100.0,
            optimized_model_size_mb=50.0,
            size_reduction_percent=50.0,
            performance_improvements={"inference_time": 0.3},
            accuracy_metrics={"accuracy": 0.95},
            optimization_summary="Quantization successful, pruning failed",
            techniques_applied=["quantization"],
            rollback_available=True,
            validation_passed=True
        )
        
        return session
    
    def test_initialization(self, memory_manager_config, temp_db_path):
        """Test MemoryManager initialization."""
        manager = MemoryManager(memory_manager_config)
        
        # Test successful initialization
        assert manager.initialize()
        assert temp_db_path.exists()
        
        # Verify database schema
        conn = sqlite3.connect(temp_db_path)
        cursor = conn.cursor()
        
        # Check tables exist
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        assert "optimization_sessions" in tables
        assert "audit_logs" in tables
        
        # Check indexes exist
        cursor.execute("SELECT name FROM sqlite_master WHERE type='index'")
        indexes = [row[0] for row in cursor.fetchall()]
        assert any("idx_sessions_status" in idx for idx in indexes)
        assert any("idx_audit_session_id" in idx for idx in indexes)
        
        conn.close()
        manager.cleanup()
    
    def test_store_and_retrieve_session(self, memory_manager, sample_session):
        """Test storing and retrieving optimization sessions."""
        # Store session
        assert memory_manager.store_session(sample_session)
        
        # Retrieve session
        retrieved_session = memory_manager.retrieve_session(sample_session.id)
        assert retrieved_session is not None
        assert retrieved_session.id == sample_session.id
        assert retrieved_session.model_id == sample_session.model_id
        assert retrieved_session.status == sample_session.status
        assert retrieved_session.criteria_name == sample_session.criteria_name
        assert len(retrieved_session.steps) == len(sample_session.steps)
        assert retrieved_session.results is not None
        
        # Test cache functionality
        cached_session = memory_manager.retrieve_session(sample_session.id)
        assert cached_session is not None
        assert cached_session.id == sample_session.id
    
    def test_retrieve_nonexistent_session(self, memory_manager):
        """Test retrieving non-existent session."""
        result = memory_manager.retrieve_session("nonexistent-session")
        assert result is None
    
    def test_list_sessions(self, memory_manager, sample_session):
        """Test listing sessions with filtering."""
        # Store multiple sessions
        session1 = sample_session
        session1.status = OptimizationStatus.COMPLETED
        memory_manager.store_session(session1)
        
        session2 = OptimizationSession(
            id="test-session-456",
            model_id="test-model-789",
            status=OptimizationStatus.RUNNING,
            criteria_name="different-criteria"
        )
        memory_manager.store_session(session2)
        
        # List all sessions
        all_sessions = memory_manager.list_sessions()
        assert len(all_sessions) == 2
        
        # Filter by status
        completed_sessions = memory_manager.list_sessions(
            status_filter=["completed"]
        )
        assert len(completed_sessions) == 1
        assert completed_sessions[0]["id"] == session1.id
        
        # Filter by model ID
        model_sessions = memory_manager.list_sessions(
            model_id_filter="test-model-789"
        )
        assert len(model_sessions) == 1
        assert model_sessions[0]["id"] == session2.id
        
        # Test pagination
        limited_sessions = memory_manager.list_sessions(limit=1)
        assert len(limited_sessions) == 1
    
    def test_delete_session(self, memory_manager, sample_session):
        """Test deleting optimization sessions."""
        # Store session
        memory_manager.store_session(sample_session)
        
        # Verify it exists
        assert memory_manager.retrieve_session(sample_session.id) is not None
        
        # Delete session
        assert memory_manager.delete_session(sample_session.id)
        
        # Verify it's gone
        assert memory_manager.retrieve_session(sample_session.id) is None
        
        # Test deleting non-existent session
        assert memory_manager.delete_session("nonexistent") is True  # Should not fail
    
    def test_session_recovery_info(self, memory_manager, sample_session):
        """Test getting session recovery information."""
        # Store session with mixed step statuses
        memory_manager.store_session(sample_session)
        
        # Get recovery info
        recovery_info = memory_manager.get_session_recovery_info(sample_session.id)
        assert recovery_info is not None
        assert recovery_info.session_id == sample_session.id
        assert len(recovery_info.recoverable_steps) == 1  # Only completed step
        assert "step-1" in recovery_info.recoverable_steps
        assert recovery_info.status_at_failure == OptimizationStatus.PENDING.value
        
        # Test non-existent session
        recovery_info = memory_manager.get_session_recovery_info("nonexistent")
        assert recovery_info is None
    
    def test_recover_session(self, memory_manager, sample_session):
        """Test session recovery functionality."""
        # Set session to failed status
        sample_session.status = OptimizationStatus.FAILED
        memory_manager.store_session(sample_session)
        
        # Recover session
        recovered_session = memory_manager.recover_session(sample_session.id)
        assert recovered_session is not None
        assert recovered_session.status == OptimizationStatus.PENDING
        
        # Check that failed steps are reset
        failed_steps = [step for step in recovered_session.steps 
                      if step.status == OptimizationStatus.FAILED]
        assert len(failed_steps) == 0  # Should be reset to PENDING
        
        # Test recovering non-existent session
        recovered = memory_manager.recover_session("nonexistent")
        assert recovered is None
    
    def test_audit_logging(self, memory_manager):
        """Test audit logging functionality."""
        session_id = "test-audit-session"
        
        # Log audit entry
        assert memory_manager.log_audit_entry(
            session_id=session_id,
            event_type="test_event",
            component="test_component",
            details={"test_key": "test_value"},
            user_id="test_user"
        )
        
        # Retrieve audit logs
        audit_logs = memory_manager.get_audit_logs(session_id=session_id)
        assert len(audit_logs) == 1
        
        log_entry = audit_logs[0]
        assert log_entry.session_id == session_id
        assert log_entry.event_type == "test_event"
        assert log_entry.component == "test_component"
        assert log_entry.details["test_key"] == "test_value"
        assert log_entry.user_id == "test_user"
        
        # Test filtering by event type
        filtered_logs = memory_manager.get_audit_logs(
            session_id=session_id,
            event_type="test_event"
        )
        assert len(filtered_logs) == 1
        
        # Test filtering by component
        component_logs = memory_manager.get_audit_logs(
            component="test_component"
        )
        assert len(component_logs) == 1
    
    def test_audit_logging_disabled(self, memory_manager_config, temp_db_path):
        """Test audit logging when disabled."""
        memory_manager_config["audit_enabled"] = False
        manager = MemoryManager(memory_manager_config)
        manager.initialize()
        
        # Should still return True but not actually log
        assert manager.log_audit_entry(
            session_id="test",
            event_type="test",
            component="test",
            details={}
        )
        
        manager.cleanup()
    
    def test_session_statistics(self, memory_manager, sample_session):
        """Test getting session statistics."""
        # Store some sessions
        session1 = sample_session
        session1.status = OptimizationStatus.COMPLETED
        memory_manager.store_session(session1)
        
        session2 = OptimizationSession(
            id="test-session-456",
            model_id="test-model-789",
            status=OptimizationStatus.RUNNING
        )
        memory_manager.store_session(session2)
        
        # Get statistics
        stats = memory_manager.get_session_statistics()
        assert stats["total_sessions"] == 2
        assert "completed" in stats["status_distribution"]
        assert "running" in stats["status_distribution"]
        assert stats["total_storage_size_mb"] > 0
        assert "cache_size" in stats
        assert "database_path" in stats
    
    def test_cleanup_old_data(self, memory_manager, sample_session):
        """Test cleanup of old data."""
        # Create old session (simulate by modifying created_at)
        old_session = sample_session
        old_session.status = OptimizationStatus.COMPLETED
        memory_manager.store_session(old_session)
        
        # Manually update the created_at to be old
        with memory_manager.get_connection() as conn:
            cursor = conn.cursor()
            old_date = (datetime.now() - timedelta(days=100)).isoformat()
            cursor.execute(
                "UPDATE optimization_sessions SET created_at = ? WHERE id = ?",
                (old_date, old_session.id)
            )
            conn.commit()
        
        # Run cleanup with short retention
        cleanup_stats = memory_manager.cleanup_old_data(retention_days=30)
        assert cleanup_stats["sessions_deleted"] == 1
        assert cleanup_stats["retention_days"] == 30
        
        # Verify session is deleted
        assert memory_manager.retrieve_session(old_session.id) is None
    
    def test_database_backup_and_restore(self, memory_manager, sample_session, temp_db_path):
        """Test database backup and restore functionality."""
        # Store session
        memory_manager.store_session(sample_session)
        
        # Create backup
        backup_path = temp_db_path.parent / "test_backup.db"
        assert memory_manager.backup_database(backup_path)
        assert backup_path.exists()
        
        # Delete original session
        memory_manager.delete_session(sample_session.id)
        assert memory_manager.retrieve_session(sample_session.id) is None
        
        # Restore from backup
        assert memory_manager.restore_database(backup_path)
        
        # Verify session is restored
        restored_session = memory_manager.retrieve_session(sample_session.id)
        assert restored_session is not None
        assert restored_session.id == sample_session.id
    
    def test_cache_functionality(self, memory_manager, sample_session):
        """Test session caching functionality."""
        # Store session
        memory_manager.store_session(sample_session)
        
        # First retrieval should hit database
        session1 = memory_manager.retrieve_session(sample_session.id)
        assert session1 is not None
        
        # Second retrieval should hit cache
        session2 = memory_manager.retrieve_session(sample_session.id)
        assert session2 is not None
        assert session2.id == session1.id
        
        # Verify cache contains the session
        assert sample_session.id in memory_manager._session_cache
    
    def test_cache_expiration(self, memory_manager_config, temp_db_path, sample_session):
        """Test cache expiration functionality."""
        # Set very short cache TTL
        memory_manager_config["cache_ttl_minutes"] = 0.01  # ~0.6 seconds
        manager = MemoryManager(memory_manager_config)
        manager.initialize()
        
        try:
            # Store and retrieve session
            manager.store_session(sample_session)
            session1 = manager.retrieve_session(sample_session.id)
            assert session1 is not None
            
            # Wait for cache to expire
            import time
            time.sleep(1)
            
            # Update session timestamp to simulate age
            sample_session.updated_at = datetime.now() - timedelta(minutes=5)
            manager._update_session_cache(sample_session)
            
            # Next retrieval should hit database (cache expired)
            session2 = manager.retrieve_session(sample_session.id)
            assert session2 is not None
            
        finally:
            manager.cleanup()
    
    def test_concurrent_access(self, memory_manager, sample_session):
        """Test concurrent access to memory manager."""
        import threading
        import time
        
        results = []
        errors = []
        
        def store_and_retrieve(session_id_suffix):
            try:
                # Create unique session
                session = OptimizationSession(
                    id=f"concurrent-session-{session_id_suffix}",
                    model_id="concurrent-model",
                    status=OptimizationStatus.PENDING
                )
                
                # Store session
                assert memory_manager.store_session(session)
                
                # Small delay to simulate processing
                time.sleep(0.01)
                
                # Retrieve session
                retrieved = memory_manager.retrieve_session(session.id)
                assert retrieved is not None
                results.append(retrieved.id)
                
            except Exception as e:
                errors.append(str(e))
        
        # Create multiple threads
        threads = []
        for i in range(10):
            thread = threading.Thread(target=store_and_retrieve, args=(i,))
            threads.append(thread)
        
        # Start all threads
        for thread in threads:
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Verify results
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) == 10
        assert len(set(results)) == 10  # All unique session IDs
    
    def test_serialization_edge_cases(self, memory_manager):
        """Test serialization of edge cases."""
        # Session with None values
        session = OptimizationSession(
            id="edge-case-session",
            model_id="edge-model",
            status=OptimizationStatus.PENDING,
            plan_id=None,
            started_at=None,
            completed_at=None,
            results=None
        )
        
        # Should handle None values gracefully
        assert memory_manager.store_session(session)
        retrieved = memory_manager.retrieve_session(session.id)
        assert retrieved is not None
        assert retrieved.plan_id is None
        assert retrieved.started_at is None
        assert retrieved.results is None
    
    def test_error_handling(self, memory_manager_config):
        """Test error handling in various scenarios."""
        # Test with invalid database path
        invalid_config = memory_manager_config.copy()
        invalid_config["database_path"] = "/invalid/path/that/does/not/exist/db.sqlite"
        
        manager = MemoryManager(invalid_config)
        # Should handle initialization failure gracefully
        assert not manager.initialize()
    
    def test_large_session_handling(self, memory_manager):
        """Test handling of large sessions."""
        # Create session with large data
        large_session = OptimizationSession(
            id="large-session",
            model_id="large-model",
            status=OptimizationStatus.PENDING,
            notes="x" * 1000000  # 1MB of text
        )
        
        # Should handle large sessions (within limits)
        assert memory_manager.store_session(large_session)
        retrieved = memory_manager.retrieve_session(large_session.id)
        assert retrieved is not None
        assert len(retrieved.notes) == 1000000


if __name__ == "__main__":
    pytest.main([__file__])
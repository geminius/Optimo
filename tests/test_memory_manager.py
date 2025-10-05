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
            created_by="test-user",
            priority=1,
            tags=["test", "robotics"]
        )
        
        # Add some steps
        step1 = OptimizationStep(
            step_id="step-1",
            technique="quantization",
            status="completed",
            parameters={"bits": 8, "method": "dynamic"}
        )
        
        step2 = OptimizationStep(
            step_id="step-2",
            technique="pruning",
            status="failed",
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
            sample_session.created_at = datetime.now() - timedelta(minutes=5)
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
            started_at=None,
            completed_at=None,
            results=None
        )
        
        # Should handle None values gracefully
        assert memory_manager.store_session(session)
        retrieved = memory_manager.retrieve_session(session.id)
        assert retrieved is not None
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
        # Create session with large data (using criteria_name for large text)
        large_session = OptimizationSession(
            id="large-session",
            model_id="large-model",
            status=OptimizationStatus.PENDING,
            criteria_name="x" * 1000000  # 1MB of text
        )
        
        # Should handle large sessions (within limits)
        assert memory_manager.store_session(large_session)
        retrieved = memory_manager.retrieve_session(large_session.id)
        assert retrieved is not None
        assert len(retrieved.criteria_name) == 1000000
    
    def test_get_connection_context_manager(self, memory_manager):
        """Test database connection context manager."""
        # Test successful connection
        with memory_manager.get_connection() as conn:
            assert conn is not None
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            result = cursor.fetchone()
            assert result[0] == 1
        
        # Connection should still be available after context exit
        # (thread-local connections are reused)
        with memory_manager.get_connection() as conn2:
            assert conn2 is not None
    
    def test_audit_logs_time_filtering(self, memory_manager):
        """Test audit log filtering by time range."""
        session_id = "time-filter-test"
        
        # Log entries at different times
        now = datetime.now()
        
        # Log first entry
        memory_manager.log_audit_entry(
            session_id=session_id,
            event_type="early_event",
            component="test",
            details={"timestamp": "early"}
        )
        
        # Simulate time passing
        import time
        time.sleep(0.1)
        
        # Log second entry
        memory_manager.log_audit_entry(
            session_id=session_id,
            event_type="late_event", 
            component="test",
            details={"timestamp": "late"}
        )
        
        # Get all logs
        all_logs = memory_manager.get_audit_logs(session_id=session_id)
        assert len(all_logs) == 2
        
        # Filter by start time (should get only the later entry)
        start_time = now + timedelta(seconds=0.05)
        filtered_logs = memory_manager.get_audit_logs(
            session_id=session_id,
            start_time=start_time
        )
        assert len(filtered_logs) == 1
        assert filtered_logs[0].event_type == "late_event"
        
        # Filter by end time (should get only the earlier entry)
        end_time = now + timedelta(seconds=0.05)
        filtered_logs = memory_manager.get_audit_logs(
            session_id=session_id,
            end_time=end_time
        )
        assert len(filtered_logs) == 1
        assert filtered_logs[0].event_type == "early_event"
    
    def test_session_update_workflow(self, memory_manager, sample_session):
        """Test updating existing sessions."""
        # Store initial session
        assert memory_manager.store_session(sample_session)
        
        # Modify session
        sample_session.status = OptimizationStatus.RUNNING
        sample_session.criteria_name = "Updated criteria"
        
        # Store updated session (should replace)
        assert memory_manager.store_session(sample_session)
        
        # Retrieve and verify updates
        retrieved = memory_manager.retrieve_session(sample_session.id)
        assert retrieved is not None
        assert retrieved.status == OptimizationStatus.RUNNING
        assert retrieved.criteria_name == "Updated criteria"
    
    def test_session_recovery_edge_cases(self, memory_manager):
        """Test session recovery edge cases."""
        # Test recovery of session with no completed steps
        session = OptimizationSession(
            id="no-completed-steps",
            model_id="test-model",
            status=OptimizationStatus.FAILED
        )
        
        # Add only failed steps
        failed_step = OptimizationStep(
            step_id="failed-step",
            technique="test",
            status="failed",
            error_message="Test failure"
        )
        session.steps = [failed_step]
        
        memory_manager.store_session(session)
        
        # Recovery info should indicate no recoverable steps
        recovery_info = memory_manager.get_session_recovery_info(session.id)
        assert recovery_info is not None
        assert len(recovery_info.recoverable_steps) == 0
        
        # Recovery should still work (reset to pending)
        recovered = memory_manager.recover_session(session.id)
        assert recovered is not None
        assert recovered.status == OptimizationStatus.PENDING
    
    def test_datetime_serialization_edge_cases(self, memory_manager):
        """Test datetime serialization with various formats."""
        # Create session with various datetime formats
        now = datetime.now()
        session = OptimizationSession(
            id="datetime-test",
            model_id="test-model",
            status=OptimizationStatus.PENDING,
            created_at=now
        )
        
        # Add step with basic fields (OptimizationStep doesn't have start_time/end_time)
        step = OptimizationStep(
            step_id="datetime-step",
            technique="test",
            status="completed"
        )
        session.steps = [step]
        
        # Store and retrieve
        assert memory_manager.store_session(session)
        retrieved = memory_manager.retrieve_session(session.id)
        
        assert retrieved is not None
        assert retrieved.created_at == now
        assert len(retrieved.steps) == 1
        assert retrieved.steps[0].step_id == "datetime-step"
    
    def test_database_schema_validation(self, memory_manager, temp_db_path):
        """Test database schema is correctly created."""
        # Check that all required tables and indexes exist
        with memory_manager.get_connection() as conn:
            cursor = conn.cursor()
            
            # Check tables
            cursor.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name IN ('optimization_sessions', 'audit_logs')
            """)
            tables = [row[0] for row in cursor.fetchall()]
            assert 'optimization_sessions' in tables
            assert 'audit_logs' in tables
            
            # Check indexes
            cursor.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='index' AND name LIKE 'idx_%'
            """)
            indexes = [row[0] for row in cursor.fetchall()]
            expected_indexes = [
                'idx_sessions_status',
                'idx_sessions_model_id', 
                'idx_sessions_created_at',
                'idx_audit_session_id',
                'idx_audit_timestamp',
                'idx_audit_event_type'
            ]
            
            for expected_idx in expected_indexes:
                assert expected_idx in indexes, f"Missing index: {expected_idx}"
    
    def test_session_list_pagination(self, memory_manager):
        """Test session listing with pagination."""
        # Create multiple sessions
        sessions = []
        for i in range(15):
            session = OptimizationSession(
                id=f"pagination-session-{i:02d}",
                model_id=f"model-{i}",
                status=OptimizationStatus.COMPLETED if i % 2 == 0 else OptimizationStatus.FAILED
            )
            sessions.append(session)
            memory_manager.store_session(session)
        
        # Test pagination
        page1 = memory_manager.list_sessions(limit=5, offset=0)
        assert len(page1) == 5
        
        page2 = memory_manager.list_sessions(limit=5, offset=5)
        assert len(page2) == 5
        
        page3 = memory_manager.list_sessions(limit=5, offset=10)
        assert len(page3) == 5
        
        # Verify no overlap
        page1_ids = {s['id'] for s in page1}
        page2_ids = {s['id'] for s in page2}
        page3_ids = {s['id'] for s in page3}
        
        assert len(page1_ids & page2_ids) == 0
        assert len(page2_ids & page3_ids) == 0
        assert len(page1_ids & page3_ids) == 0
    
    def test_audit_log_metadata_handling(self, memory_manager):
        """Test audit log metadata functionality."""
        session_id = "metadata-test"
        
        # Log entry with metadata
        assert memory_manager.log_audit_entry(
            session_id=session_id,
            event_type="test_with_metadata",
            component="test_component",
            details={"key": "value"},
            user_id="test_user"
        )
        
        # Retrieve and verify metadata
        logs = memory_manager.get_audit_logs(session_id=session_id)
        assert len(logs) == 1
        
        log_entry = logs[0]
        assert log_entry.session_id == session_id
        assert log_entry.event_type == "test_with_metadata"
        assert log_entry.component == "test_component"
        assert log_entry.details["key"] == "value"
        assert log_entry.user_id == "test_user"
        assert isinstance(log_entry.metadata, dict)
    
    def test_cache_size_limit(self, memory_manager_config, temp_db_path):
        """Test cache size limit enforcement."""
        # Set small cache size
        memory_manager_config["cache_max_size"] = 3
        manager = MemoryManager(memory_manager_config)
        manager.initialize()
        
        try:
            # Store more sessions than cache can hold
            sessions = []
            for i in range(5):
                session = OptimizationSession(
                    id=f"cache-limit-{i}",
                    model_id=f"model-{i}",
                    status=OptimizationStatus.PENDING
                )
                sessions.append(session)
                manager.store_session(session)
            
            # Cache should only hold the last 3 sessions
            assert len(manager._session_cache) <= 3
            
            # Verify most recent sessions are in cache
            for i in range(2, 5):  # Last 3 sessions
                assert f"cache-limit-{i}" in manager._session_cache
            
        finally:
            manager.cleanup()
    
    def test_backup_restore_error_handling(self, memory_manager, temp_db_path):
        """Test backup and restore error handling."""
        # Test restore with non-existent backup
        non_existent_backup = temp_db_path.parent / "non_existent_backup.db"
        assert not memory_manager.restore_database(non_existent_backup)
        
        # Test backup to invalid path (should handle gracefully)
        invalid_backup_path = Path("/invalid/path/backup.db")
        assert not memory_manager.backup_database(invalid_backup_path)
    
    def test_session_statistics_edge_cases(self, memory_manager):
        """Test session statistics with edge cases."""
        # Get stats with empty database
        stats = memory_manager.get_session_statistics()
        assert stats["total_sessions"] == 0
        assert stats["total_storage_size_mb"] == 0
        assert stats["recent_sessions_7_days"] == 0
        assert stats["average_duration_seconds"] is None
        
        # Add session with no duration (not started/completed)
        session = OptimizationSession(
            id="no-duration-session",
            model_id="test-model",
            status=OptimizationStatus.PENDING
        )
        memory_manager.store_session(session)
        
        # Stats should handle sessions without duration
        stats = memory_manager.get_session_statistics()
        assert stats["total_sessions"] == 1
        assert stats["average_duration_seconds"] is None  # No completed sessions with duration


if __name__ == "__main__":
    pytest.main([__file__])
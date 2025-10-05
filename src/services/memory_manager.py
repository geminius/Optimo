"""
Memory Manager for optimization history and session persistence.

This module provides the MemoryManager class that handles persistent storage
of optimization sessions, audit logging, and session recovery capabilities.
"""

import logging
import sqlite3
import json
import pickle
import hashlib
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
import threading
import uuid
from contextlib import contextmanager

from ..models.core import (
    OptimizationSession, AnalysisReport, EvaluationReport, 
    OptimizationStatus, ModelMetadata, OptimizationStep,
    OptimizationResults
)


logger = logging.getLogger(__name__)


@dataclass
class AuditLogEntry:
    """Audit log entry for optimization decisions and results."""
    id: str
    session_id: str
    timestamp: datetime
    event_type: str  # "session_created", "step_started", "step_completed", "decision_made", etc.
    component: str   # "analysis_agent", "planning_agent", "optimization_agent", etc.
    details: Dict[str, Any]
    user_id: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class SessionRecoveryInfo:
    """Information needed to recover a session."""
    session_id: str
    last_checkpoint: datetime
    recoverable_steps: List[str]
    recovery_data: Dict[str, Any]
    status_at_failure: str
    error_context: Optional[str] = None


class MemoryManager:
    """
    Manages optimization history, session state, and persistent storage.
    
    Responsibilities:
    - Store and retrieve optimization sessions
    - Maintain audit logs for all optimization decisions
    - Provide session recovery and continuation capabilities
    - Manage database connections and transactions
    - Handle data serialization and deserialization
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Database configuration
        self.db_path = Path(config.get("database_path", "optimization_history.db"))
        self.backup_interval_hours = config.get("backup_interval_hours", 24)
        self.retention_days = config.get("retention_days", 90)
        self.max_session_size_mb = config.get("max_session_size_mb", 100)
        
        # Thread safety
        self._lock = threading.RLock()
        self._connection_pool = {}
        self._thread_local = threading.local()
        
        # Cache for frequently accessed data
        self._session_cache: Dict[str, OptimizationSession] = {}
        self._cache_max_size = config.get("cache_max_size", 100)
        self._cache_ttl_minutes = config.get("cache_ttl_minutes", 30)
        
        # Audit logging
        self.audit_enabled = config.get("audit_enabled", True)
        self.audit_retention_days = config.get("audit_retention_days", 365)
        
        self.logger.info("MemoryManager initialized")
    
    def initialize(self) -> bool:
        """Initialize the memory manager and database."""
        try:
            self.logger.info("Initializing MemoryManager")
            
            # Create database directory if it doesn't exist
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Initialize database schema
            self._initialize_database()
            
            # Start background maintenance tasks
            self._start_maintenance_tasks()
            
            self.logger.info("MemoryManager initialization completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize MemoryManager: {e}")
            return False
    
    def cleanup(self) -> None:
        """Clean up resources and close database connections."""
        try:
            self.logger.info("Cleaning up MemoryManager")
            
            # Close all database connections
            with self._lock:
                for conn in self._connection_pool.values():
                    conn.close()
                self._connection_pool.clear()
            
            # Clear cache
            self._session_cache.clear()
            
            self.logger.info("MemoryManager cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Error during MemoryManager cleanup: {e}")
    
    @contextmanager
    def get_connection(self):
        """Get a database connection with automatic cleanup."""
        conn = None
        try:
            conn = self._get_db_connection()
            yield conn
        finally:
            # Don't close thread-local connections here
            # They will be closed during cleanup
            pass
    
    def store_session(self, session: OptimizationSession) -> bool:
        """
        Store optimization session in persistent storage.
        
        Args:
            session: OptimizationSession to store
            
        Returns:
            True if stored successfully
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Serialize session data
                session_data = self._serialize_session(session)
                session_size = len(session_data)
                
                # Check size limit
                if session_size > self.max_session_size_mb * 1024 * 1024:
                    self.logger.warning(f"Session {session.id} exceeds size limit: {session_size / (1024*1024):.2f} MB")
                
                # Insert or update session
                cursor.execute("""
                    INSERT OR REPLACE INTO optimization_sessions 
                    (id, model_id, status, criteria_name, created_at, 
                     started_at, completed_at, created_by, priority, tags, session_data, data_size)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    session.id,
                    session.model_id,
                    session.status.value,
                    session.criteria_name,
                    session.created_at.isoformat(),
                    session.started_at.isoformat() if session.started_at else None,
                    session.completed_at.isoformat() if session.completed_at else None,
                    session.created_by,
                    session.priority,
                    json.dumps(session.tags),
                    session_data,
                    session_size
                ))
                
                conn.commit()
                
                # Update cache
                self._update_session_cache(session)
                
                # Log audit entry
                if self.audit_enabled:
                    self._log_audit_entry(
                        session_id=session.id,
                        event_type="session_stored",
                        component="memory_manager",
                        details={
                            "status": session.status.value,
                            "data_size": session_size,
                            "steps_count": len(session.steps)
                        }
                    )
                
                self.logger.debug(f"Stored session {session.id} ({session_size} bytes)")
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to store session {session.id}: {e}")
            return False
    
    def retrieve_session(self, session_id: str) -> Optional[OptimizationSession]:
        """
        Retrieve optimization session from storage.
        
        Args:
            session_id: ID of session to retrieve
            
        Returns:
            OptimizationSession if found, None otherwise
        """
        # Check cache first
        cached_session = self._get_cached_session(session_id)
        if cached_session:
            return cached_session
        
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT session_data FROM optimization_sessions 
                    WHERE id = ?
                """, (session_id,))
                
                row = cursor.fetchone()
                if not row:
                    return None
                
                # Deserialize session data
                session = self._deserialize_session(row[0])
                
                # Update cache
                self._update_session_cache(session)
                
                self.logger.debug(f"Retrieved session {session_id}")
                return session
                
        except Exception as e:
            self.logger.error(f"Failed to retrieve session {session_id}: {e}")
            return None
    
    def list_sessions(
        self, 
        status_filter: Optional[List[str]] = None,
        model_id_filter: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        List optimization sessions with optional filtering.
        
        Args:
            status_filter: List of statuses to filter by
            model_id_filter: Model ID to filter by
            limit: Maximum number of sessions to return
            offset: Number of sessions to skip
            
        Returns:
            List of session metadata dictionaries
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Build query with filters
                query = """
                    SELECT id, model_id, status, criteria_name, created_at,
                           started_at, completed_at, created_by, priority, tags, data_size
                    FROM optimization_sessions
                    WHERE 1=1
                """
                params = []
                
                if status_filter:
                    placeholders = ','.join(['?' for _ in status_filter])
                    query += f" AND status IN ({placeholders})"
                    params.extend(status_filter)
                
                if model_id_filter:
                    query += " AND model_id = ?"
                    params.append(model_id_filter)
                
                query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
                params.extend([limit, offset])
                
                cursor.execute(query, params)
                rows = cursor.fetchall()
                
                sessions = []
                for row in rows:
                    session_info = {
                        "id": row[0],
                        "model_id": row[1],
                        "status": row[2],
                        "criteria_name": row[3],
                        "created_at": row[4],
                        "started_at": row[5],
                        "completed_at": row[6],
                        "created_by": row[7],
                        "priority": row[8],
                        "tags": json.loads(row[9]) if row[9] else [],
                        "data_size": row[10]
                    }
                    sessions.append(session_info)
                
                return sessions
                
        except Exception as e:
            self.logger.error(f"Failed to list sessions: {e}")
            return []
    
    def delete_session(self, session_id: str) -> bool:
        """
        Delete optimization session from storage.
        
        Args:
            session_id: ID of session to delete
            
        Returns:
            True if deleted successfully
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Delete session
                cursor.execute("DELETE FROM optimization_sessions WHERE id = ?", (session_id,))
                
                # Delete related audit logs
                cursor.execute("DELETE FROM audit_logs WHERE session_id = ?", (session_id,))
                
                conn.commit()
                
                # Remove from cache
                self._session_cache.pop(session_id, None)
                
                # Log audit entry
                if self.audit_enabled:
                    self._log_audit_entry(
                        session_id=session_id,
                        event_type="session_deleted",
                        component="memory_manager",
                        details={"deleted_at": datetime.now().isoformat()}
                    )
                
                self.logger.info(f"Deleted session {session_id}")
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to delete session {session_id}: {e}")
            return False
    
    def get_session_recovery_info(self, session_id: str) -> Optional[SessionRecoveryInfo]:
        """
        Get recovery information for a failed or interrupted session.
        
        Args:
            session_id: ID of session to get recovery info for
            
        Returns:
            SessionRecoveryInfo if session can be recovered, None otherwise
        """
        try:
            session = self.retrieve_session(session_id)
            if not session:
                return None
            
            # Determine recoverable steps
            recoverable_steps = []
            last_checkpoint = session.created_at
            
            for step in session.steps:
                if step.status == "completed":
                    recoverable_steps.append(step.step_id)
                    # OptimizationStep doesn't have end_time, use created_at as fallback
                    last_checkpoint = session.created_at
                elif step.status in ["failed", "cancelled"]:
                    break  # Stop at first failed step
            
            # Always return recovery info, even if no steps are recoverable
            # This allows for session status reset even without completed steps
            
            recovery_info = SessionRecoveryInfo(
                session_id=session_id,
                last_checkpoint=last_checkpoint,
                recoverable_steps=recoverable_steps,
                recovery_data={
                    "completed_steps": len(recoverable_steps),
                    "total_steps": len(session.steps),
                    "last_status": session.status.value
                },
                status_at_failure=session.status.value
            )
            
            return recovery_info
            
        except Exception as e:
            self.logger.error(f"Failed to get recovery info for session {session_id}: {e}")
            return None
    
    def recover_session(self, session_id: str) -> Optional[OptimizationSession]:
        """
        Recover a failed or interrupted session.
        
        Args:
            session_id: ID of session to recover
            
        Returns:
            Recovered OptimizationSession if successful, None otherwise
        """
        try:
            recovery_info = self.get_session_recovery_info(session_id)
            if not recovery_info:
                self.logger.warning(f"Session {session_id} cannot be recovered")
                return None
            
            session = self.retrieve_session(session_id)
            if not session:
                return None
            
            # Reset session status for recovery
            session.status = OptimizationStatus.PENDING
            session.created_at = datetime.now()
            
            # Reset failed/cancelled steps
            for step in session.steps:
                if step.status in ["failed", "cancelled"]:
                    step.status = "pending"
                    step.error_message = None
            
            # Store recovered session
            if self.store_session(session):
                # Log audit entry
                if self.audit_enabled:
                    self._log_audit_entry(
                        session_id=session_id,
                        event_type="session_recovered",
                        component="memory_manager",
                        details={
                            "recovery_point": recovery_info.last_checkpoint.isoformat(),
                            "recoverable_steps": len(recovery_info.recoverable_steps)
                        }
                    )
                
                self.logger.info(f"Recovered session {session_id}")
                return session
            
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to recover session {session_id}: {e}")
            return None
    
    def log_audit_entry(
        self,
        session_id: str,
        event_type: str,
        component: str,
        details: Dict[str, Any],
        user_id: Optional[str] = None
    ) -> bool:
        """
        Log an audit entry for optimization decisions and results.
        
        Args:
            session_id: ID of related session
            event_type: Type of event being logged
            component: Component that generated the event
            details: Event details and metadata
            user_id: Optional user ID
            
        Returns:
            True if logged successfully
        """
        if not self.audit_enabled:
            return True
        
        return self._log_audit_entry(session_id, event_type, component, details, user_id)
    
    def get_audit_logs(
        self,
        session_id: Optional[str] = None,
        event_type: Optional[str] = None,
        component: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 1000
    ) -> List[AuditLogEntry]:
        """
        Retrieve audit logs with optional filtering.
        
        Args:
            session_id: Filter by session ID
            event_type: Filter by event type
            component: Filter by component
            start_time: Filter by start time
            end_time: Filter by end time
            limit: Maximum number of entries to return
            
        Returns:
            List of AuditLogEntry objects
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Build query with filters
                query = """
                    SELECT id, session_id, timestamp, event_type, component, details, user_id, metadata
                    FROM audit_logs
                    WHERE 1=1
                """
                params = []
                
                if session_id:
                    query += " AND session_id = ?"
                    params.append(session_id)
                
                if event_type:
                    query += " AND event_type = ?"
                    params.append(event_type)
                
                if component:
                    query += " AND component = ?"
                    params.append(component)
                
                if start_time:
                    query += " AND timestamp >= ?"
                    params.append(start_time.isoformat())
                
                if end_time:
                    query += " AND timestamp <= ?"
                    params.append(end_time.isoformat())
                
                query += " ORDER BY timestamp DESC LIMIT ?"
                params.append(limit)
                
                cursor.execute(query, params)
                rows = cursor.fetchall()
                
                audit_logs = []
                for row in rows:
                    audit_log = AuditLogEntry(
                        id=row[0],
                        session_id=row[1],
                        timestamp=datetime.fromisoformat(row[2]),
                        event_type=row[3],
                        component=row[4],
                        details=json.loads(row[5]) if row[5] else {},
                        user_id=row[6],
                        metadata=json.loads(row[7]) if row[7] else {}
                    )
                    audit_logs.append(audit_log)
                
                return audit_logs
                
        except Exception as e:
            self.logger.error(f"Failed to retrieve audit logs: {e}")
            return []
    
    def get_session_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about stored sessions.
        
        Returns:
            Dictionary containing session statistics
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Total sessions
                cursor.execute("SELECT COUNT(*) FROM optimization_sessions")
                total_sessions = cursor.fetchone()[0]
                
                # Sessions by status
                cursor.execute("""
                    SELECT status, COUNT(*) 
                    FROM optimization_sessions 
                    GROUP BY status
                """)
                status_counts = dict(cursor.fetchall())
                
                # Total data size
                cursor.execute("SELECT SUM(data_size) FROM optimization_sessions")
                total_size = cursor.fetchone()[0] or 0
                
                # Recent activity (last 7 days)
                week_ago = (datetime.now() - timedelta(days=7)).isoformat()
                cursor.execute("""
                    SELECT COUNT(*) FROM optimization_sessions 
                    WHERE created_at >= ?
                """, (week_ago,))
                recent_sessions = cursor.fetchone()[0]
                
                # Average session duration for completed sessions
                cursor.execute("""
                    SELECT AVG(
                        (julianday(completed_at) - julianday(started_at)) * 24 * 60 * 60
                    ) FROM optimization_sessions 
                    WHERE status = 'completed' AND started_at IS NOT NULL AND completed_at IS NOT NULL
                """)
                avg_duration = cursor.fetchone()[0]
                
                return {
                    "total_sessions": total_sessions,
                    "status_distribution": status_counts,
                    "total_storage_size_mb": total_size / (1024 * 1024),
                    "recent_sessions_7_days": recent_sessions,
                    "average_duration_seconds": avg_duration,
                    "cache_size": len(self._session_cache),
                    "database_path": str(self.db_path)
                }
                
        except Exception as e:
            self.logger.error(f"Failed to get session statistics: {e}")
            return {}
    
    def cleanup_old_data(self, retention_days: Optional[int] = None) -> Dict[str, int]:
        """
        Clean up old sessions and audit logs based on retention policy.
        
        Args:
            retention_days: Override default retention period
            
        Returns:
            Dictionary with cleanup statistics
        """
        if retention_days is None:
            retention_days = self.retention_days
        
        cutoff_date = (datetime.now() - timedelta(days=retention_days)).isoformat()
        
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Count sessions to be deleted
                cursor.execute("""
                    SELECT COUNT(*) FROM optimization_sessions 
                    WHERE created_at < ? AND status IN ('completed', 'failed', 'cancelled')
                """, (cutoff_date,))
                sessions_to_delete = cursor.fetchone()[0]
                
                # Delete old sessions
                cursor.execute("""
                    DELETE FROM optimization_sessions 
                    WHERE created_at < ? AND status IN ('completed', 'failed', 'cancelled')
                """, (cutoff_date,))
                
                # Count audit logs to be deleted
                audit_cutoff = (datetime.now() - timedelta(days=self.audit_retention_days)).isoformat()
                cursor.execute("""
                    SELECT COUNT(*) FROM audit_logs 
                    WHERE timestamp < ?
                """, (audit_cutoff,))
                audit_logs_to_delete = cursor.fetchone()[0]
                
                # Delete old audit logs
                cursor.execute("DELETE FROM audit_logs WHERE timestamp < ?", (audit_cutoff,))
                
                conn.commit()
                
                # Clear cache of deleted sessions
                self._session_cache.clear()
                
                cleanup_stats = {
                    "sessions_deleted": sessions_to_delete,
                    "audit_logs_deleted": audit_logs_to_delete,
                    "retention_days": retention_days,
                    "audit_retention_days": self.audit_retention_days
                }
                
                self.logger.info(f"Cleanup completed: {cleanup_stats}")
                return cleanup_stats
                
        except Exception as e:
            self.logger.error(f"Failed to cleanup old data: {e}")
            return {"error": str(e)}
    
    def _initialize_database(self) -> None:
        """Initialize database schema."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Create optimization_sessions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS optimization_sessions (
                    id TEXT PRIMARY KEY,
                    model_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    criteria_name TEXT,
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    completed_at TEXT,
                    created_by TEXT,
                    priority INTEGER DEFAULT 1,
                    tags TEXT,
                    session_data BLOB NOT NULL,
                    data_size INTEGER DEFAULT 0,
                    FOREIGN KEY (model_id) REFERENCES models(id)
                )
            """)
            
            # Create audit_logs table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS audit_logs (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    component TEXT NOT NULL,
                    details TEXT,
                    user_id TEXT,
                    metadata TEXT,
                    FOREIGN KEY (session_id) REFERENCES optimization_sessions(id)
                )
            """)
            
            # Create indexes for better query performance
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_sessions_status 
                ON optimization_sessions(status)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_sessions_model_id 
                ON optimization_sessions(model_id)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_sessions_created_at 
                ON optimization_sessions(created_at)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_audit_session_id 
                ON audit_logs(session_id)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_audit_timestamp 
                ON audit_logs(timestamp)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_audit_event_type 
                ON audit_logs(event_type)
            """)
            
            conn.commit()
            self.logger.info("Database schema initialized")
    
    def _get_db_connection(self) -> sqlite3.Connection:
        """Get a database connection for the current thread."""
        thread_id = threading.get_ident()
        
        if not hasattr(self._thread_local, 'connection'):
            self._thread_local.connection = sqlite3.connect(
                self.db_path,
                timeout=30.0,
                check_same_thread=False
            )
            # Enable WAL mode for better concurrency
            self._thread_local.connection.execute("PRAGMA journal_mode=WAL")
            self._thread_local.connection.execute("PRAGMA synchronous=NORMAL")
            self._thread_local.connection.execute("PRAGMA cache_size=10000")
            
        return self._thread_local.connection
    
    def _serialize_session(self, session: OptimizationSession) -> bytes:
        """Serialize optimization session to bytes."""
        try:
            # Convert session to dictionary
            session_dict = asdict(session)
            
            # Handle datetime objects
            session_dict = self._convert_datetimes_to_iso(session_dict)
            
            # Serialize using pickle for efficiency
            return pickle.dumps(session_dict)
            
        except Exception as e:
            self.logger.error(f"Failed to serialize session {session.id}: {e}")
            raise
    
    def _deserialize_session(self, data: bytes) -> OptimizationSession:
        """Deserialize optimization session from bytes."""
        try:
            # Deserialize using pickle
            session_dict = pickle.loads(data)
            
            # Convert ISO strings back to datetime objects
            session_dict = self._convert_iso_to_datetimes(session_dict)
            
            # Reconstruct OptimizationSession object
            session = self._dict_to_session(session_dict)
            
            return session
            
        except Exception as e:
            self.logger.error(f"Failed to deserialize session data: {e}")
            raise
    
    def _convert_datetimes_to_iso(self, obj: Any) -> Any:
        """Recursively convert datetime objects to ISO strings."""
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, dict):
            return {k: self._convert_datetimes_to_iso(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_datetimes_to_iso(item) for item in obj]
        else:
            return obj
    
    def _convert_iso_to_datetimes(self, obj: Any) -> Any:
        """Recursively convert ISO strings to datetime objects."""
        if isinstance(obj, str):
            # Try to parse as datetime
            try:
                if 'T' in obj and (obj.endswith('Z') or '+' in obj[-6:] or obj.count(':') >= 2):
                    return datetime.fromisoformat(obj.replace('Z', '+00:00'))
            except ValueError:
                pass
            return obj
        elif isinstance(obj, dict):
            return {k: self._convert_iso_to_datetimes(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_iso_to_datetimes(item) for item in obj]
        else:
            return obj
    
    def _dict_to_session(self, session_dict: Dict[str, Any]) -> OptimizationSession:
        """Convert dictionary to OptimizationSession object."""
        # Convert status enum
        if 'status' in session_dict and isinstance(session_dict['status'], str):
            session_dict['status'] = OptimizationStatus(session_dict['status'])
        
        # Convert steps
        if 'steps' in session_dict:
            steps = []
            for step_dict in session_dict['steps']:
                if isinstance(step_dict['status'], str):
                    step_dict['status'] = OptimizationStatus(step_dict['status'])
                step = OptimizationStep(**step_dict)
                steps.append(step)
            session_dict['steps'] = steps
        
        # Convert results if present
        if 'results' in session_dict and session_dict['results']:
            results_dict = session_dict['results']
            session_dict['results'] = OptimizationResults(**results_dict)
        
        return OptimizationSession(**session_dict)
    
    def _update_session_cache(self, session: OptimizationSession) -> None:
        """Update session cache with size limit."""
        with self._lock:
            # Remove oldest entries if cache is full
            if len(self._session_cache) >= self._cache_max_size:
                # Remove oldest entry (simple FIFO)
                oldest_key = next(iter(self._session_cache))
                del self._session_cache[oldest_key]
            
            self._session_cache[session.id] = session
    
    def _get_cached_session(self, session_id: str) -> Optional[OptimizationSession]:
        """Get session from cache if available and not expired."""
        with self._lock:
            if session_id in self._session_cache:
                session = self._session_cache[session_id]
                
                # Check if cache entry is still valid (simple TTL check)
                cache_age = datetime.now() - session.created_at
                if cache_age.total_seconds() < self._cache_ttl_minutes * 60:
                    return session
                else:
                    # Remove expired entry
                    del self._session_cache[session_id]
            
            return None
    
    def _log_audit_entry(
        self,
        session_id: str,
        event_type: str,
        component: str,
        details: Dict[str, Any],
        user_id: Optional[str] = None
    ) -> bool:
        """Internal method to log audit entry."""
        try:
            audit_entry = AuditLogEntry(
                id=str(uuid.uuid4()),
                session_id=session_id,
                timestamp=datetime.now(),
                event_type=event_type,
                component=component,
                details=details,
                user_id=user_id
            )
            
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO audit_logs 
                    (id, session_id, timestamp, event_type, component, details, user_id, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    audit_entry.id,
                    audit_entry.session_id,
                    audit_entry.timestamp.isoformat(),
                    audit_entry.event_type,
                    audit_entry.component,
                    json.dumps(audit_entry.details),
                    audit_entry.user_id,
                    json.dumps(audit_entry.metadata)
                ))
                
                conn.commit()
                
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to log audit entry: {e}")
            return False
    
    def _start_maintenance_tasks(self) -> None:
        """Start background maintenance tasks."""
        # This would typically start background threads for:
        # - Periodic cleanup of old data
        # - Database backup
        # - Cache maintenance
        # For now, we'll just log that maintenance is available
        self.logger.info("Maintenance tasks initialized (cleanup available on demand)")
    
    def backup_database(self, backup_path: Optional[Path] = None) -> bool:
        """
        Create a backup of the database.
        
        Args:
            backup_path: Path for backup file (auto-generated if None)
            
        Returns:
            True if backup successful
        """
        if backup_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = self.db_path.parent / f"optimization_history_backup_{timestamp}.db"
        
        try:
            with self.get_connection() as conn:
                backup_conn = sqlite3.connect(backup_path)
                conn.backup(backup_conn)
                backup_conn.close()
            
            self.logger.info(f"Database backup created: {backup_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create database backup: {e}")
            return False
    
    def restore_database(self, backup_path: Path) -> bool:
        """
        Restore database from backup.
        
        Args:
            backup_path: Path to backup file
            
        Returns:
            True if restore successful
        """
        if not backup_path.exists():
            self.logger.error(f"Backup file not found: {backup_path}")
            return False
        
        try:
            # Close existing connections
            with self._lock:
                for conn in self._connection_pool.values():
                    conn.close()
                self._connection_pool.clear()
            
            # Clear thread-local connections
            if hasattr(self._thread_local, 'connection'):
                self._thread_local.connection.close()
                delattr(self._thread_local, 'connection')
            
            # Replace current database with backup
            import shutil
            shutil.copy2(backup_path, self.db_path)
            
            # Clear cache
            self._session_cache.clear()
            
            self.logger.info(f"Database restored from backup: {backup_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to restore database from backup: {e}")
            return False
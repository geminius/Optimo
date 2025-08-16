"""
Recovery mechanisms for handling failures and rollbacks.
"""

import logging
import traceback
from typing import Dict, Any, Optional, List, Callable, Union
from datetime import datetime
from enum import Enum
import torch
import copy

from .exceptions import PlatformError, ErrorCategory, ErrorSeverity


logger = logging.getLogger(__name__)


class RecoveryStrategy(Enum):
    """Available recovery strategies."""
    ROLLBACK = "rollback"
    RETRY = "retry"
    GRACEFUL_DEGRADATION = "graceful_degradation"
    FAIL_FAST = "fail_fast"
    CONTINUE_PARTIAL = "continue_partial"


class RecoveryAction:
    """Represents a recovery action to be taken."""
    
    def __init__(
        self,
        strategy: RecoveryStrategy,
        description: str,
        action: Callable[[], bool],
        priority: int = 0,
        conditions: Optional[List[Callable[[], bool]]] = None
    ):
        self.strategy = strategy
        self.description = description
        self.action = action
        self.priority = priority  # Higher priority actions are tried first
        self.conditions = conditions or []
    
    def can_execute(self) -> bool:
        """Check if this recovery action can be executed."""
        return all(condition() for condition in self.conditions)
    
    def execute(self) -> bool:
        """Execute the recovery action."""
        try:
            logger.info(f"Executing recovery action: {self.description}")
            return self.action()
        except Exception as e:
            logger.error(f"Recovery action failed: {self.description} - {e}")
            return False


class RecoveryManager:
    """Manages recovery strategies and actions for different failure scenarios."""
    
    def __init__(self):
        self.recovery_actions: Dict[ErrorCategory, List[RecoveryAction]] = {}
        self.recovery_history: List[Dict[str, Any]] = []
        self.max_history_size = 100
    
    def register_recovery_action(
        self,
        category: ErrorCategory,
        action: RecoveryAction
    ) -> None:
        """Register a recovery action for a specific error category."""
        if category not in self.recovery_actions:
            self.recovery_actions[category] = []
        
        self.recovery_actions[category].append(action)
        # Sort by priority (descending)
        self.recovery_actions[category].sort(key=lambda x: x.priority, reverse=True)
    
    def handle_error(
        self,
        error: Exception,
        context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Handle an error by attempting appropriate recovery actions.
        
        Returns:
            True if recovery was successful, False otherwise
        """
        context = context or {}
        
        # Determine error category
        if isinstance(error, PlatformError):
            category = error.category
        else:
            category = self._categorize_error(error)
        
        logger.error(f"Handling error in category {category.value}: {error}")
        
        # Get applicable recovery actions
        actions = self.recovery_actions.get(category, [])
        
        recovery_record = {
            "timestamp": datetime.now(),
            "error": str(error),
            "category": category.value,
            "context": context,
            "actions_attempted": [],
            "success": False
        }
        
        # Try recovery actions in priority order
        for action in actions:
            if action.can_execute():
                recovery_record["actions_attempted"].append({
                    "strategy": action.strategy.value,
                    "description": action.description,
                    "timestamp": datetime.now()
                })
                
                try:
                    if action.execute():
                        logger.info(f"Recovery successful using strategy: {action.strategy.value}")
                        recovery_record["success"] = True
                        recovery_record["successful_action"] = action.description
                        self._add_to_history(recovery_record)
                        return True
                except Exception as recovery_error:
                    logger.error(f"Recovery action failed: {recovery_error}")
                    recovery_record["actions_attempted"][-1]["error"] = str(recovery_error)
        
        logger.error(f"All recovery attempts failed for error: {error}")
        self._add_to_history(recovery_record)
        return False
    
    def _categorize_error(self, error: Exception) -> ErrorCategory:
        """Categorize a generic exception."""
        error_type = type(error).__name__
        error_message = str(error).lower()
        
        if any(keyword in error_message for keyword in ["file", "path", "directory", "disk"]):
            return ErrorCategory.STORAGE
        elif any(keyword in error_message for keyword in ["network", "connection", "timeout", "http"]):
            return ErrorCategory.NETWORK
        elif any(keyword in error_message for keyword in ["memory", "resource", "system"]):
            return ErrorCategory.SYSTEM
        elif any(keyword in error_message for keyword in ["model", "load", "format"]):
            return ErrorCategory.MODEL_LOADING
        elif any(keyword in error_message for keyword in ["config", "parameter", "setting"]):
            return ErrorCategory.CONFIGURATION
        else:
            return ErrorCategory.SYSTEM
    
    def _add_to_history(self, record: Dict[str, Any]) -> None:
        """Add recovery record to history."""
        self.recovery_history.append(record)
        
        # Maintain history size limit
        if len(self.recovery_history) > self.max_history_size:
            self.recovery_history = self.recovery_history[-self.max_history_size:]
    
    def get_recovery_history(self) -> List[Dict[str, Any]]:
        """Get recovery history."""
        return self.recovery_history.copy()
    
    def clear_history(self) -> None:
        """Clear recovery history."""
        self.recovery_history.clear()


class ModelRecoveryManager:
    """Specialized recovery manager for model-related operations."""
    
    def __init__(self):
        self.model_snapshots: Dict[str, List[Dict[str, Any]]] = {}
        self.max_snapshots_per_model = 5
    
    def create_model_snapshot(
        self,
        model_id: str,
        model: torch.nn.Module,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create a snapshot of a model for recovery purposes."""
        snapshot_id = f"{model_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        snapshot = {
            "snapshot_id": snapshot_id,
            "model_id": model_id,
            "state_dict": copy.deepcopy(model.state_dict()),
            "model_class": model.__class__.__name__,
            "timestamp": datetime.now(),
            "metadata": metadata or {}
        }
        
        if model_id not in self.model_snapshots:
            self.model_snapshots[model_id] = []
        
        self.model_snapshots[model_id].append(snapshot)
        
        # Maintain snapshot limit
        if len(self.model_snapshots[model_id]) > self.max_snapshots_per_model:
            self.model_snapshots[model_id] = self.model_snapshots[model_id][-self.max_snapshots_per_model:]
        
        logger.info(f"Created model snapshot: {snapshot_id}")
        return snapshot_id
    
    def restore_model_from_snapshot(
        self,
        model: torch.nn.Module,
        model_id: str,
        snapshot_id: Optional[str] = None
    ) -> bool:
        """
        Restore a model from a snapshot.
        
        Args:
            model: Model to restore
            model_id: ID of the model
            snapshot_id: Specific snapshot ID, or None for latest
        
        Returns:
            True if restoration was successful
        """
        if model_id not in self.model_snapshots:
            logger.error(f"No snapshots found for model: {model_id}")
            return False
        
        snapshots = self.model_snapshots[model_id]
        if not snapshots:
            logger.error(f"No snapshots available for model: {model_id}")
            return False
        
        # Find the snapshot to restore
        if snapshot_id is None:
            # Use latest snapshot
            snapshot = snapshots[-1]
        else:
            snapshot = next((s for s in snapshots if s["snapshot_id"] == snapshot_id), None)
            if snapshot is None:
                logger.error(f"Snapshot not found: {snapshot_id}")
                return False
        
        try:
            model.load_state_dict(snapshot["state_dict"])
            logger.info(f"Successfully restored model from snapshot: {snapshot['snapshot_id']}")
            return True
        except Exception as e:
            logger.error(f"Failed to restore model from snapshot: {e}")
            return False
    
    def list_snapshots(self, model_id: str) -> List[Dict[str, Any]]:
        """List available snapshots for a model."""
        if model_id not in self.model_snapshots:
            return []
        
        return [
            {
                "snapshot_id": s["snapshot_id"],
                "timestamp": s["timestamp"],
                "metadata": s["metadata"]
            }
            for s in self.model_snapshots[model_id]
        ]
    
    def cleanup_snapshots(self, model_id: str, keep_latest: int = 1) -> None:
        """Clean up old snapshots, keeping only the specified number of latest ones."""
        if model_id not in self.model_snapshots:
            return
        
        snapshots = self.model_snapshots[model_id]
        if len(snapshots) <= keep_latest:
            return
        
        # Keep only the latest snapshots
        self.model_snapshots[model_id] = snapshots[-keep_latest:]
        logger.info(f"Cleaned up snapshots for model {model_id}, kept {keep_latest} latest")


class GracefulDegradationManager:
    """Manages graceful degradation when some optimization techniques fail."""
    
    def __init__(self):
        self.fallback_strategies: Dict[str, List[str]] = {}
        self.technique_priorities: Dict[str, int] = {}
    
    def register_fallback_strategy(
        self,
        primary_technique: str,
        fallback_techniques: List[str],
        priority: int = 0
    ) -> None:
        """Register fallback techniques for a primary optimization technique."""
        self.fallback_strategies[primary_technique] = fallback_techniques
        self.technique_priorities[primary_technique] = priority
    
    def get_fallback_techniques(
        self,
        failed_technique: str,
        available_techniques: List[str]
    ) -> List[str]:
        """Get fallback techniques for a failed optimization technique."""
        fallbacks = self.fallback_strategies.get(failed_technique, [])
        
        # Filter to only include available techniques
        available_fallbacks = [t for t in fallbacks if t in available_techniques]
        
        # Sort by priority
        available_fallbacks.sort(
            key=lambda t: self.technique_priorities.get(t, 0),
            reverse=True
        )
        
        return available_fallbacks
    
    def create_degraded_plan(
        self,
        original_plan: List[str],
        failed_techniques: List[str],
        available_techniques: List[str]
    ) -> List[str]:
        """Create a degraded optimization plan when some techniques fail."""
        degraded_plan = []
        
        for technique in original_plan:
            if technique not in failed_techniques:
                # Technique is still available
                degraded_plan.append(technique)
            else:
                # Find fallback techniques
                fallbacks = self.get_fallback_techniques(technique, available_techniques)
                if fallbacks:
                    # Use the highest priority fallback
                    degraded_plan.append(fallbacks[0])
                    logger.info(f"Replaced failed technique {technique} with {fallbacks[0]}")
                else:
                    logger.warning(f"No fallback available for failed technique: {technique}")
        
        return degraded_plan


# Global recovery manager instance
recovery_manager = RecoveryManager()
model_recovery_manager = ModelRecoveryManager()
degradation_manager = GracefulDegradationManager()


def setup_default_recovery_actions():
    """Set up default recovery actions for common error categories."""
    
    # Model loading recovery actions
    def retry_model_loading():
        logger.info("Attempting to retry model loading with different parameters")
        return True  # This would be implemented with actual retry logic
    
    recovery_manager.register_recovery_action(
        ErrorCategory.MODEL_LOADING,
        RecoveryAction(
            strategy=RecoveryStrategy.RETRY,
            description="Retry model loading with adjusted parameters",
            action=retry_model_loading,
            priority=10
        )
    )
    
    # System error recovery actions
    def free_memory():
        logger.info("Attempting to free memory")
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return True
    
    recovery_manager.register_recovery_action(
        ErrorCategory.SYSTEM,
        RecoveryAction(
            strategy=RecoveryStrategy.GRACEFUL_DEGRADATION,
            description="Free memory and reduce resource usage",
            action=free_memory,
            priority=5
        )
    )
    
    # Network error recovery actions
    def wait_and_retry():
        logger.info("Waiting before network retry")
        import time
        time.sleep(5)
        return True
    
    recovery_manager.register_recovery_action(
        ErrorCategory.NETWORK,
        RecoveryAction(
            strategy=RecoveryStrategy.RETRY,
            description="Wait and retry network operation",
            action=wait_and_retry,
            priority=8
        )
    )
    
    # Set up default fallback strategies
    degradation_manager.register_fallback_strategy(
        "quantization_4bit",
        ["quantization_8bit", "pruning_structured"],
        priority=10
    )
    
    degradation_manager.register_fallback_strategy(
        "quantization_8bit",
        ["pruning_structured", "pruning_unstructured"],
        priority=8
    )
    
    degradation_manager.register_fallback_strategy(
        "pruning_structured",
        ["pruning_unstructured"],
        priority=6
    )


# Initialize default recovery actions
setup_default_recovery_actions()
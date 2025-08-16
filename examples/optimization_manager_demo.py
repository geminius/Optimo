"""
Demo script for OptimizationManager orchestrator.

This script demonstrates how to use the OptimizationManager to coordinate
the complete optimization workflow for robotics models.
"""

import torch
import torch.nn as nn
import tempfile
import os
import time
import logging
from typing import Dict, Any

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.services.optimization_manager import OptimizationManager, ProgressUpdate
from src.config.optimization_criteria import (
    OptimizationCriteria, PerformanceMetric, PerformanceThreshold,
    OptimizationConstraints, OptimizationTechnique
)


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DemoRoboticsModel(nn.Module):
    """Demo robotics model for optimization testing."""
    
    def __init__(self, input_size=512, hidden_size=256, output_size=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
        
        self.attention = nn.MultiheadAttention(output_size, num_heads=8)
        
        self.decoder = nn.Sequential(
            nn.Linear(output_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32)  # Action space
        )
    
    def forward(self, x):
        # Encode input
        encoded = self.encoder(x)
        
        # Apply attention (simplified)
        attended, _ = self.attention(encoded.unsqueeze(0), encoded.unsqueeze(0), encoded.unsqueeze(0))
        attended = attended.squeeze(0)
        
        # Decode to actions
        actions = self.decoder(attended)
        return actions


def create_demo_model() -> str:
    """Create a demo model and save it to a temporary file."""
    model = DemoRoboticsModel()
    
    # Create temporary file
    temp_file = tempfile.NamedTemporaryFile(suffix='.pt', delete=False)
    temp_path = temp_file.name
    temp_file.close()
    
    # Save model
    torch.save(model, temp_path)
    logger.info(f"Created demo model at: {temp_path}")
    
    return temp_path


def create_optimization_criteria() -> OptimizationCriteria:
    """Create optimization criteria for the demo."""
    
    # Define performance thresholds
    thresholds = [
        PerformanceThreshold(
            metric=PerformanceMetric.INFERENCE_TIME,
            max_value=50.0,  # Max 50ms inference time
            description="Maximum acceptable inference time"
        ),
        PerformanceThreshold(
            metric=PerformanceMetric.MODEL_SIZE,
            max_value=100.0,  # Max 100MB model size
            description="Maximum acceptable model size"
        ),
        PerformanceThreshold(
            metric=PerformanceMetric.ACCURACY,
            min_value=0.90,  # Minimum 90% accuracy
            description="Minimum acceptable accuracy"
        )
    ]
    
    # Define optimization constraints
    constraints = OptimizationConstraints(
        preserve_accuracy_threshold=0.95,  # Preserve 95% of original accuracy
        allowed_techniques=[
            OptimizationTechnique.QUANTIZATION,
            OptimizationTechnique.PRUNING
        ],
        max_optimization_time_minutes=60
    )
    
    # Create criteria
    criteria = OptimizationCriteria(
        name="robotics_edge_deployment",
        description="Optimization for edge deployment of robotics models",
        target_deployment="edge",
        priority_weights={
            PerformanceMetric.MODEL_SIZE: 0.4,
            PerformanceMetric.INFERENCE_TIME: 0.4,
            PerformanceMetric.ACCURACY: 0.2
        },
        performance_thresholds=thresholds,
        constraints=constraints
    )
    
    return criteria


def create_manager_config() -> Dict[str, Any]:
    """Create configuration for OptimizationManager."""
    return {
        "max_concurrent_sessions": 2,
        "auto_rollback_on_failure": True,
        "snapshot_frequency": 1,
        "session_timeout_minutes": 120,
        
        # Agent configurations
        "analysis_agent": {
            "profiling_samples": 50,
            "warmup_samples": 5
        },
        "planning_agent": {
            "max_plan_steps": 3,
            "risk_tolerance": 0.7,
            "min_impact_threshold": 0.1
        },
        "evaluation_agent": {
            "benchmark_samples": 50,
            "accuracy_threshold": 0.95,
            "timeout_seconds": 180
        },
        "quantization_agent": {
            "preserve_modules": ["decoder"]
        },
        "pruning_agent": {
            "preserve_modules": ["attention", "decoder"]
        }
    }


def progress_callback(session_id: str, update: ProgressUpdate):
    """Callback function to handle progress updates."""
    logger.info(f"Session {session_id}: {update.status.value} - "
                f"{update.progress_percentage:.1f}% - {update.current_step}")
    
    if update.message:
        logger.info(f"  Message: {update.message}")
    
    if update.estimated_remaining_minutes:
        logger.info(f"  Estimated remaining: {update.estimated_remaining_minutes} minutes")


def demonstrate_basic_optimization():
    """Demonstrate basic optimization workflow."""
    logger.info("=== Basic Optimization Demo ===")
    
    # Create demo model and criteria
    model_path = create_demo_model()
    criteria = create_optimization_criteria()
    config = create_manager_config()
    
    try:
        # Create and initialize OptimizationManager
        manager = OptimizationManager(config)
        
        if not manager.initialize():
            logger.error("Failed to initialize OptimizationManager")
            return
        
        logger.info("OptimizationManager initialized successfully")
        
        # Add progress callback
        manager.add_progress_callback(progress_callback)
        
        # Start optimization session
        logger.info("Starting optimization session...")
        session_id = manager.start_optimization_session(model_path, criteria)
        logger.info(f"Started session: {session_id}")
        
        # Monitor progress
        start_time = time.time()
        timeout = 300  # 5 minutes timeout
        
        while time.time() - start_time < timeout:
            try:
                status = manager.get_session_status(session_id)
                current_status = status["status"]
                
                logger.info(f"Current status: {current_status} - "
                           f"{status['progress_percentage']:.1f}%")
                
                if current_status in ["completed", "failed", "cancelled"]:
                    break
                
                time.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                logger.error(f"Error getting session status: {e}")
                break
        
        # Get final status
        try:
            final_status = manager.get_session_status(session_id)
            logger.info(f"Final session status: {final_status}")
            
            if final_status["status"] == "completed":
                logger.info("✅ Optimization completed successfully!")
            else:
                logger.warning(f"⚠️ Optimization ended with status: {final_status['status']}")
                if final_status.get("error_message"):
                    logger.error(f"Error: {final_status['error_message']}")
        
        except Exception as e:
            logger.error(f"Error getting final status: {e}")
        
        # Cleanup
        manager.cleanup()
        
    finally:
        # Clean up temporary model file
        if os.path.exists(model_path):
            os.unlink(model_path)
            logger.info(f"Cleaned up temporary model file: {model_path}")


def demonstrate_session_management():
    """Demonstrate session management features."""
    logger.info("=== Session Management Demo ===")
    
    model_path = create_demo_model()
    criteria = create_optimization_criteria()
    config = create_manager_config()
    
    try:
        manager = OptimizationManager(config)
        
        if not manager.initialize():
            logger.error("Failed to initialize OptimizationManager")
            return
        
        # Start multiple sessions
        session_ids = []
        for i in range(2):
            session_id = manager.start_optimization_session(
                model_path, criteria, f"demo_session_{i}"
            )
            session_ids.append(session_id)
            logger.info(f"Started session {i}: {session_id}")
        
        # List active sessions
        active_sessions = manager.get_active_sessions()
        logger.info(f"Active sessions: {active_sessions}")
        
        # Wait a bit for sessions to start
        time.sleep(2)
        
        # Demonstrate pause/resume
        first_session = session_ids[0]
        logger.info(f"Pausing session: {first_session}")
        manager.pause_session(first_session)
        
        time.sleep(1)
        
        logger.info(f"Resuming session: {first_session}")
        manager.resume_session(first_session)
        
        # Demonstrate cancellation
        second_session = session_ids[1]
        logger.info(f"Cancelling session: {second_session}")
        manager.cancel_session(second_session)
        
        # Check agent status
        agent_status = manager.get_optimization_agents_status()
        logger.info(f"Optimization agents status: {list(agent_status.keys())}")
        
        # Cleanup
        manager.cleanup()
        
    finally:
        if os.path.exists(model_path):
            os.unlink(model_path)


def demonstrate_snapshot_rollback():
    """Demonstrate snapshot and rollback functionality."""
    logger.info("=== Snapshot and Rollback Demo ===")
    
    model_path = create_demo_model()
    criteria = create_optimization_criteria()
    config = create_manager_config()
    
    try:
        manager = OptimizationManager(config)
        
        if not manager.initialize():
            logger.error("Failed to initialize OptimizationManager")
            return
        
        # Start session
        session_id = manager.start_optimization_session(model_path, criteria)
        logger.info(f"Started session: {session_id}")
        
        # Wait for some progress
        time.sleep(5)
        
        # Check for snapshots
        snapshots = manager.get_session_snapshots(session_id)
        logger.info(f"Available snapshots: {len(snapshots)}")
        
        for i, snapshot in enumerate(snapshots):
            logger.info(f"  Snapshot {i}: {snapshot['description']} "
                       f"at {snapshot['timestamp']}")
        
        # Demonstrate rollback if snapshots exist
        if snapshots:
            logger.info("Demonstrating rollback to first snapshot...")
            rollback_success = manager.rollback_session(session_id, 0)
            
            if rollback_success:
                logger.info("✅ Rollback successful")
            else:
                logger.warning("⚠️ Rollback failed")
        
        # Cleanup
        manager.cleanup()
        
    finally:
        if os.path.exists(model_path):
            os.unlink(model_path)


def main():
    """Run all demonstrations."""
    logger.info("Starting OptimizationManager demonstrations...")
    
    try:
        # Run demonstrations
        demonstrate_basic_optimization()
        print("\n" + "="*50 + "\n")
        
        demonstrate_session_management()
        print("\n" + "="*50 + "\n")
        
        demonstrate_snapshot_rollback()
        
        logger.info("All demonstrations completed!")
        
    except Exception as e:
        logger.error(f"Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
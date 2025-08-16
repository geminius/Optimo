"""Minimal models package for testing."""

# Expose commonly used classes for convenience
from .core import (
    ModelMetadata,
    OptimizationSession,
    SessionSnapshot,
    WorkflowState,
    WorkflowStatus,
    AnalysisReport,
    OptimizationPlan,
    OptimizationResults,
    EvaluationReport,
    OptimizationStatus,
    SessionStatus,
    ProgressUpdate,
    SessionContext,
    ArchitectureSummary,
    PerformanceProfile,
    OptimizationOpportunity,
    Recommendation,
    BenchmarkResult,
    PerformanceMetrics,
    ComparisonResult,
    ValidationStatus,
)

from .store import ModelStore

__all__ = [
    "ModelMetadata",
    "OptimizationSession",
    "SessionSnapshot",
    "WorkflowState",
    "WorkflowStatus",
    "AnalysisReport",
    "OptimizationPlan",
    "OptimizationResults",
    "EvaluationReport",
    "OptimizationStatus",
    "SessionStatus",
    "ProgressUpdate",
    "SessionContext",
    "ArchitectureSummary",
    "PerformanceProfile",
    "OptimizationOpportunity",
    "Recommendation",
    "BenchmarkResult",
    "PerformanceMetrics",
    "ComparisonResult",
    "ValidationStatus",
    "ModelStore",
]


---
inclusion: always
---
---
inclusion: always
---

# Robotics Model Optimization Platform

## Product Context

This platform autonomously optimizes robotics models (OpenVLA, Vision-Language-Action models) using specialized AI agents. The system analyzes, plans, optimizes, and evaluates models with minimal human intervention, targeting edge deployment scenarios.

## Agent Architecture

The platform uses four specialized agents with distinct responsibilities:

1. **Analysis Agent** (`src/agents/analysis/`) - Profiles models, detects architecture patterns, identifies optimization opportunities
2. **Planning Agent** (`src/agents/planning/`) - Selects optimization strategies based on model characteristics and user-defined constraints
3. **Optimization Agents** (`src/agents/optimization/`) - Execute specific techniques:
   - Quantization (INT8, INT4, GPTQ, bitsandbytes)
   - Pruning (structured/unstructured, magnitude-based)
   - Distillation (teacher-student compression)
   - Architecture Search (NAS)
   - Compression (general size reduction)
4. **Evaluation Agent** (`src/agents/evaluation/`) - Validates results against accuracy, size, and latency criteria

## Optimization Priorities

When making optimization decisions, prioritize in this order:

1. **Real-time inference requirements** - Robotics models must meet strict latency constraints
2. **Edge device constraints** - Memory and compute limitations are hard boundaries
3. **Task-specific accuracy** - Never sacrifice accuracy below user-defined thresholds
4. **Deployment environment** - Consider target hardware capabilities

## Model Handling Rules

**Supported Formats**: PyTorch (.pt, .pth), TensorFlow, ONNX, SafeTensors

- Always validate model format before processing
- Provide actionable error messages for unsupported formats
- Never assume model structure - use Analysis Agent to detect architecture
- Avoid hardcoded model configurations - let agents make decisions based on profiling

## User Experience Mandates

**Progress Transparency**: All long-running operations MUST emit real-time progress events via WebSocket. Use `self.emit_progress()` in agent implementations.

**Error Recovery**: Failed optimizations MUST rollback to last known good state. Use recovery strategies from `src/utils/recovery.py`.

**Evaluation Clarity**: Results MUST include before/after metrics (accuracy, size, latency, inference time) for comparison.

## Standard Workflow

```
Upload → Analysis → Planning → Optimization → Evaluation → Results
```

Users define optimization criteria (target size, accuracy threshold, latency requirements). The system autonomously selects and applies appropriate techniques. All optimization attempts are logged with full metrics for history tracking and comparison.

## API Conventions

- **REST endpoints** for CRUD operations and job submission
- **WebSocket connections** for real-time progress updates
- **Async/await** required for all I/O operations
- **Structured errors** with actionable messages (use custom exceptions from `src/utils/exceptions.py`)
- **OpenAPI documentation** auto-generated from FastAPI schemas

## Key Behaviors

- Agents should make intelligent decisions based on model characteristics, not require manual configuration per model
- Always log optimization attempts with full context for debugging and analysis
- Emit progress events at meaningful milestones (0%, 25%, 50%, 75%, 100%)
- Validate optimization results against user-defined criteria before marking as successful
- Store model versions and metadata for rollback capability
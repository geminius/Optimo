---
inclusion: always
---
---
inclusion: always
---

# Robotics Model Optimization Platform

## What This Platform Does

Autonomously optimizes robotics models (OpenVLA, Vision-Language-Action models) for edge deployment using specialized AI agents. The system analyzes, plans, optimizes, and evaluates models with minimal human intervention.

## Agent System Architecture

Four specialized agents work in sequence:

1. **Analysis Agent** (`src/agents/analysis/agent.py`) - Profiles models, detects architecture patterns, identifies optimization opportunities
2. **Planning Agent** (`src/agents/planning/`) - Selects optimization strategies based on model characteristics and constraints
3. **Optimization Agents** (`src/agents/optimization/`) - Five specialized techniques:
   - `quantization.py` - INT8, INT4, GPTQ, bitsandbytes quantization
   - `pruning.py` - Structured/unstructured, magnitude-based pruning
   - `distillation.py` - Teacher-student knowledge distillation
   - `architecture_search.py` - Neural Architecture Search (NAS)
   - `compression.py` - General model compression
4. **Evaluation Agent** (`src/agents/evaluation/agent.py`) - Validates results against accuracy, size, and latency criteria

All agents inherit from `BaseOptimizationAgent` or `BaseAgent` in `src/agents/base.py`.

## Critical Optimization Priorities

When implementing or modifying optimization logic, prioritize in this order:

1. **Real-time inference** - Robotics models have strict latency constraints (typically <100ms)
2. **Edge device limits** - Memory and compute are hard boundaries, not suggestions
3. **Accuracy thresholds** - Never drop below user-defined minimum accuracy
4. **Target hardware** - Optimize for specific deployment environment (NVIDIA Jetson, mobile, etc.)

## Model Handling Requirements

**Supported Formats**: PyTorch (.pt, .pth), TensorFlow, ONNX, SafeTensors

When working with models:
- Always validate format before processing using `src/models/validation.py`
- Never assume model structure - use Analysis Agent to detect architecture
- Avoid hardcoded configurations - let agents decide based on profiling
- Provide actionable error messages for unsupported formats (use custom exceptions from `src/utils/exceptions.py`)

## Mandatory User Experience Patterns

### Progress Transparency
All long-running operations MUST emit real-time progress via WebSocket:
```python
self.emit_progress({"status": "processing", "progress": 0.5, "message": "Applying quantization"})
```
Emit at meaningful milestones: 0%, 25%, 50%, 75%, 100%

### Error Recovery
Failed optimizations MUST rollback to last known good state:
- Use recovery strategies from `src/utils/recovery.py`
- Store model versions in `models/versions/` for rollback
- Log all state changes for debugging

### Evaluation Clarity
Results MUST include before/after comparison metrics:
- Accuracy (absolute and relative change)
- Model size (MB, compression ratio)
- Latency (ms, speedup factor)
- Inference time (ms per sample)

## Standard Workflow

```
Upload → Analysis → Planning → Optimization → Evaluation → Results
```

Users define optimization criteria (target size, accuracy threshold, latency requirements). The system autonomously selects and applies techniques. All attempts are logged with full metrics in `data/optimization_history.db`.

## API Design Patterns

### REST Endpoints
- CRUD operations and job submission
- All I/O operations MUST use async/await
- Use dependency injection via `src/api/dependencies.py`
- Request/response models in `src/api/models.py`

### WebSocket Connections
- Real-time progress updates via `src/services/websocket_manager.py`
- Event types defined in `src/api/websocket_events.py`
- Emit events for: progress, status changes, errors, completion

### Error Handling
- Use custom exceptions from `src/utils/exceptions.py` (OptimizationError, ValidationError, ResourceError)
- Apply retry decorators from `src/utils/retry.py` for transient failures
- Return structured error responses with actionable messages
- OpenAPI documentation auto-generated from FastAPI schemas

## Agent Implementation Rules

When creating or modifying agents:

1. **Inherit from base classes** - Use `BaseOptimizationAgent` or `BaseAgent` from `src/agents/base.py`
2. **Implement async methods** - All optimize/process methods must be async
3. **Emit progress events** - Call `self.emit_progress()` at key milestones
4. **Make intelligent decisions** - Agents should adapt to model characteristics, not require manual configuration
5. **Log with context** - Use structured logging: `logger.info("message", extra={"component": "AgentName", "model_id": id})`
6. **Validate results** - Check optimization results against user-defined criteria before marking successful

## Service Layer Patterns

Services in `src/services/` use singleton pattern:
- `OptimizationManager` - Orchestrates optimization workflows
- `MemoryManager` - Tracks and manages system resources
- `NotificationService` - Handles event broadcasting
- `MonitoringService` - Collects metrics and health data
- `WebSocketManager` - Manages WebSocket connections

Access services via dependency injection, never instantiate directly.

## Data Models

Core models in `src/models/core.py`:
- `ModelData` - Represents uploaded models with metadata
- `OptimizationConfig` - User-defined optimization criteria and constraints
- `OptimizationResult` - Results with before/after metrics

Always use these typed models for consistency across the platform.
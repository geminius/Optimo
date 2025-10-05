---
inclusion: always
---
---
inclusion: always
---

# Robotics Model Optimization Platform

## Product Mission

Autonomous optimization of robotics models (OpenVLA, Vision-Language-Action models) using specialized AI agents that analyze, plan, optimize, and evaluate with minimal human intervention.

## Core Design Principles

**Autonomous Operation** - Agents make intelligent optimization decisions based on model characteristics and predefined criteria. Avoid requiring manual configuration for each model.

**Agent-Driven Architecture** - Four specialized agents with clear boundaries:
- **Analysis Agent**: Model profiling and characteristic detection
- **Planning Agent**: Strategy selection based on constraints
- **Optimization Agents**: Execute techniques (quantization, pruning, distillation, architecture search, compression)
- **Evaluation Agent**: Validate results against criteria

**Robotics-First Constraints** - Prioritize:
- Real-time inference requirements
- Edge device limitations (memory, compute)
- Task-specific accuracy thresholds
- Deployment environment constraints

## Optimization Techniques

- **Quantization**: INT8, INT4, GPTQ, bitsandbytes
- **Pruning**: Structured/unstructured, magnitude-based
- **Knowledge Distillation**: Teacher-student compression
- **Architecture Search**: Automated NAS
- **Compression**: General size reduction

## Model Format Support

Supported: PyTorch (.pt, .pth), TensorFlow, ONNX, SafeTensors

**Always validate format before processing** and provide actionable error messages for unsupported formats.

## User Experience Requirements

**Progress Transparency** - Emit real-time progress events via WebSocket for all long-running operations. Users must see status updates during optimization.

**Error Recovery** - Failed optimizations must rollback to last known good state. Never leave models corrupted or partially optimized.

**Evaluation Clarity** - Results must include before/after metrics (accuracy, size, latency) for informed decision-making.

## API Design Rules

- REST for CRUD and job submission
- WebSocket for real-time progress
- Async/await for all I/O operations
- Structured error responses with actionable messages
- Auto-generated OpenAPI docs from FastAPI schemas

## Primary Workflow

**Upload → Analysis → Planning → Optimization → Evaluation → Results**

Users define optimization criteria (target size, accuracy threshold, latency requirements). All attempts are logged with full metrics for comparison and history tracking.
---
inclusion: always
---
---
inclusion: always
---

# Architecture & Code Structure

## Layered Architecture

The codebase follows strict layer separation. Never bypass layers or create circular dependencies:

1. **Agent Layer** (`src/agents/`) - All agents inherit from `BaseOptimizationAgent` or `BaseAgent`
2. **Service Layer** (`src/services/`) - Singleton managers (MemoryManager, OptimizationManager, NotificationService, MonitoringService)
3. **Integration Layer** (`src/integration/`) - Dependency injection and component orchestration
4. **API Layer** (`src/api/`) - FastAPI endpoints with async/await for all I/O operations

## Foundation Files (Always Check Before Creating Components)

Before implementing new functionality, check these files for existing patterns:

- `src/agents/base.py` - Base classes with progress emission and error handling
- `src/models/core.py` - Core data models (ModelData, OptimizationResult, OptimizationConfig)
- `src/utils/exceptions.py` - Custom exceptions (OptimizationError, ValidationError, ResourceError)
- `src/api/models.py` - Pydantic request/response schemas
- `src/config/optimization_criteria.py` - Optimization configuration and constraints

## Implementation Patterns

### Creating a New Agent

Must inherit from base class, implement async methods, and emit progress:

```python
from src.agents.base import BaseOptimizationAgent
from src.models.core import ModelData, OptimizationResult

class NewAgent(BaseOptimizationAgent):
    async def optimize(self, model_data: ModelData) -> OptimizationResult:
        self.emit_progress({"status": "starting", "progress": 0.0})
        # Implementation
        self.emit_progress({"status": "complete", "progress": 1.0})
        return result
```

### Creating a New Service

Use singleton pattern to ensure single instance:

```python
class NewService:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            # Initialize once
        return cls._instance
```

### Creating a New API Endpoint

Use FastAPI dependency injection and typed models:

```python
from fastapi import APIRouter, Depends
from src.api.dependencies import get_optimization_manager
from src.api.models import RequestModel, ResponseModel

router = APIRouter(prefix="/api/v1", tags=["category"])

@router.post("/endpoint", response_model=ResponseModel)
async def endpoint(
    request: RequestModel,
    manager = Depends(get_optimization_manager)
):
    return await manager.process(request)
```

## Naming Conventions

- **Python files**: `snake_case.py`
- **Python classes**: `PascalCase`
- **Python functions/variables**: `snake_case`
- **Python constants**: `UPPER_SNAKE_CASE`
- **React components**: `PascalCase.tsx`
- **React tests**: `ComponentName.test.tsx`
- **Backend tests**: `test_module_name.py`

## Import Order

Follow isort conventions (enforced by tooling):

1. Standard library imports
2. Third-party package imports
3. Local application imports (relative)

## Error Handling

- Use custom exceptions from `src/utils/exceptions.py` - never raise generic Exception
- Apply retry decorators from `src/utils/retry.py` for transient failures
- Implement recovery strategies from `src/utils/recovery.py` for rollback capability
- Use structured logging with context: `logger.info("message", extra={"component": "ComponentName", "model_id": model_id})`

## Testing Structure

- **Backend tests**: Mirror `src/` structure in `tests/` directory
- Use shared fixtures from `tests/conftest.py`
- Test files must start with `test_` prefix
- Integration tests go in `tests/integration/`
- Performance tests go in `tests/performance/`

## Frontend Structure

- `src/components/` - Reusable UI components
- `src/pages/` - Route-level page components
- `src/services/api.ts` - Centralized API client (all backend calls go through this)
- `src/types/index.ts` - TypeScript type definitions
- `src/contexts/` - React context providers (WebSocket, state management)
- `src/tests/` - Component tests with React Testing Library

## Critical Rules

- All async operations MUST use async/await syntax
- All agents MUST emit progress events via `self.emit_progress()`
- All services MUST use singleton pattern
- All API endpoints MUST use dependency injection
- All functions MUST have type hints (Python) or TypeScript types
- Never hardcode paths - use configuration or environment variables
- Never commit sensitive data - use environment variables
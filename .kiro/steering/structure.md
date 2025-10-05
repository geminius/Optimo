---
inclusion: always
---
---
inclusion: always
---

# Architecture & Code Structure

## Layered Architecture (Strict Separation)

1. **Agent Layer** (`src/agents/`) - Inherit from `BaseOptimizationAgent` or `BaseAgent`
2. **Service Layer** (`src/services/`) - Singleton pattern for managers (MemoryManager, OptimizationManager, etc.)
3. **Integration Layer** (`src/integration/`) - Dependency injection for component wiring
4. **API Layer** (`src/api/`) - FastAPI endpoints, all I/O must be async/await

## Foundation Files (Check Before Creating New Components)

- `src/agents/base.py` - Base classes for all agents
- `src/models/core.py` - Core data models (ModelData, OptimizationResult, etc.)
- `src/utils/exceptions.py` - Custom exceptions (OptimizationError, ValidationError, etc.)
- `src/api/models.py` - Pydantic request/response models
- `src/config/optimization_criteria.py` - Optimization configuration schemas

## Required Patterns

**New Agent** - Must inherit from base, use async/await, emit progress events:
```python
from src.agents.base import BaseOptimizationAgent

class NewAgent(BaseOptimizationAgent):
    async def optimize(self, model_data: ModelData) -> OptimizationResult:
        self.emit_progress({"status": "processing", "progress": 0.5})
        # Implementation
```

**New Service** - Use singleton pattern:
```python
class NewService:
    _instance = None
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
```

**New API Endpoint** - Use dependency injection, typed responses:
```python
from fastapi import APIRouter, Depends
from src.api.dependencies import get_service

router = APIRouter(prefix="/api/v1", tags=["category"])

@router.post("/endpoint", response_model=ResponseModel)
async def endpoint(request: RequestModel, service = Depends(get_service)):
    return await service.process(request)
```

## Naming Conventions

- Python: `snake_case.py`, classes `PascalCase`, functions `snake_case`, constants `UPPER_SNAKE_CASE`
- React: Components `PascalCase.tsx`, tests `*.test.tsx`
- Tests: `test_*.py` (backend)

## Import Order (Enforced by isort)

1. Standard library
2. Third-party packages
3. Local imports (relative)

## Error Handling Rules

- Use custom exceptions from `src/utils/exceptions.py`
- Structured logging: `logger.info("message", extra={"component": "ComponentName"})`
- Apply retry decorators from `src/utils/retry.py` for transient failures
- Implement recovery strategies from `src/utils/recovery.py`

## Directory Structure

**Backend**: Tests mirror `src/` structure in `tests/` (use fixtures from `tests/conftest.py`)

**Frontend**:
- `components/` - Reusable UI components
- `pages/` - Route-level pages
- `services/api.ts` - Centralized API client
- `types/index.ts` - TypeScript definitions
- `contexts/` - React context providers
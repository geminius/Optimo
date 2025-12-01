---
inclusion: always
---

# Architecture & Code Structure

## Layered Architecture

Strict layer separation - never bypass layers or create circular dependencies:

1. **Agent Layer** (`src/agents/`) - Inherit from `BaseOptimizationAgent` or `BaseAgent`
2. **Service Layer** (`src/services/`) - Singleton managers for orchestration
3. **Integration Layer** (`src/integration/`) - Dependency injection and component wiring
4. **API Layer** (`src/api/`) - FastAPI endpoints with async/await

## Foundation Files - Check Before Creating New Components

- `src/agents/base.py` - Base agent classes with progress emission and error handling
- `src/models/core.py` - Core data models (ModelData, OptimizationResult, OptimizationConfig)
- `src/utils/exceptions.py` - Custom exceptions (OptimizationError, ValidationError, ResourceError)
- `src/api/models.py` - Pydantic request/response schemas
- `src/api/dependencies.py` - Dependency injection functions
- `src/config/optimization_criteria.py` - Optimization configuration and constraints

## Backend Implementation Patterns

### Agent Pattern
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

### Service Pattern (Singleton)
```python
class NewService:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
```

### API Endpoint Pattern
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

## Frontend Structure

- `frontend/src/components/` - Reusable UI components
- `frontend/src/pages/` - Route-level page components
- `frontend/src/services/api.ts` - Centralized API client (all backend calls go through this)
- `frontend/src/services/auth.ts` - Authentication service
- `frontend/src/types/` - TypeScript type definitions
- `frontend/src/contexts/` - React context providers (AuthContext, WebSocketContext)
- `frontend/src/hooks/` - Custom React hooks
- `frontend/src/tests/` - Component tests with React Testing Library

### Frontend Patterns

- Functional components with hooks only (no class components)
- All API calls through `src/services/api.ts`
- Props and state must be explicitly typed
- Use AuthContext for authentication state
- Use WebSocketContext for real-time updates

## Naming Conventions

**Python:**
- Files: `snake_case.py`
- Classes: `PascalCase`
- Functions/variables: `snake_case`
- Constants: `UPPER_SNAKE_CASE`

**TypeScript/React:**
- Components: `PascalCase.tsx`
- Hooks: `useCamelCase.ts`
- Utilities: `camelCase.ts`
- Types: `PascalCase` interfaces/types

**Tests:**
- Backend: `test_module_name.py`
- Frontend: `ComponentName.test.tsx`

## Import Order

Follow isort conventions (enforced):

1. Standard library imports
2. Third-party package imports
3. Local application imports

## Error Handling

- Use custom exceptions from `src/utils/exceptions.py` - never raise generic `Exception`
- Apply retry decorators from `src/utils/retry.py` for transient failures
- Implement recovery strategies from `src/utils/recovery.py` for rollback
- Use structured logging: `logger.info("msg", extra={"component": "Name", "context": value})`

## Testing Structure

**Backend:**
- Mirror `src/` structure in `tests/` directory
- Use shared fixtures from `tests/conftest.py`
- Prefix test files with `test_`
- Integration tests: `tests/integration/`
- Performance tests: `tests/performance/`

**Frontend:**
- Colocate tests in `frontend/src/tests/`
- Use React Testing Library
- Mock API calls via `jest.mock()`
- Test user interactions, not implementation details

## Critical Rules

**Backend:**
- All I/O operations MUST use async/await
- All agents MUST emit progress via `self.emit_progress()`
- All services MUST use singleton pattern
- All API endpoints MUST use dependency injection
- All functions MUST have type hints

**Frontend:**
- All components MUST be functional with hooks
- All API calls MUST go through `services/api.ts`
- All props and state MUST be explicitly typed
- Use contexts for global state (auth, websocket)

**General:**
- Never hardcode paths - use config or environment variables
- Never commit sensitive data - use `.env` files
- Never bypass layer architecture
- Always check foundation files before creating new patterns
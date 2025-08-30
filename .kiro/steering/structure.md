---
inclusion: always
---

# Architecture & Code Structure Guide

## Layered Architecture Pattern
Follow strict separation of concerns across 4 layers:
- **Agent Layer** (`src/agents/`): Autonomous optimization agents - inherit from base classes
- **Service Layer** (`src/services/`): Business logic - use singleton pattern for managers
- **Integration Layer** (`src/integration/`): Component wiring - dependency injection pattern
- **API Layer** (`src/api/`): FastAPI endpoints - async/await for all I/O operations

## Critical File Locations
- **Agent base classes**: `src/agents/base.py` - Always inherit from these
- **Core data models**: `src/models/core.py` - Use existing schemas
- **Custom exceptions**: `src/utils/exceptions.py` - Extend these for error handling
- **API models**: `src/api/models.py` - Pydantic schemas for API contracts
- **Configuration**: `src/config/optimization_criteria.py` - Platform optimization settings

## Code Patterns to Follow

### Agent Implementation
```python
from src.agents.base import BaseOptimizationAgent
from src.utils.exceptions import OptimizationError

class NewAgent(BaseOptimizationAgent):
    async def optimize(self, model_data: ModelData) -> OptimizationResult:
        # Always use async/await for I/O operations
        # Always handle errors with custom exceptions
        # Always emit progress updates via self.emit_progress()
```

### Service Implementation
```python
from typing import Optional
import logging

class NewService:
    _instance: Optional['NewService'] = None  # Singleton pattern
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    async def process(self) -> None:
        # Use structured logging
        logging.info("Processing started", extra={"component": "NewService"})
```

### API Endpoint Pattern
```python
from fastapi import APIRouter, Depends, HTTPException
from src.api.models import RequestModel, ResponseModel
from src.api.dependencies import get_service

router = APIRouter(prefix="/api/v1", tags=["category"])

@router.post("/endpoint", response_model=ResponseModel)
async def endpoint(
    request: RequestModel,
    service = Depends(get_service)
) -> ResponseModel:
    # Always use dependency injection
    # Always return proper response models
```

## Naming Conventions (Strictly Enforced)
- **Python files**: `snake_case.py`
- **Classes**: `PascalCase` (e.g., `OptimizationManager`)
- **Functions/methods**: `snake_case` (e.g., `optimize_model`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `MAX_RETRIES`)
- **React components**: `PascalCase.tsx` (e.g., `ModelUpload.tsx`)
- **Test files**: `test_*.py` or `*.test.tsx`

## Import Order (Required)
```python
# 1. Standard library
import os
import logging
from typing import Dict, List, Optional

# 2. Third-party packages
import torch
from fastapi import FastAPI

# 3. Local imports (relative)
from .base import BaseAgent
from ..utils.exceptions import OptimizationError
```

## Error Handling Rules
- Use custom exceptions from `src/utils/exceptions.py`
- Always log errors with structured logging
- Implement retry logic using `src/utils/retry.py`
- Use recovery strategies from `src/utils/recovery.py`

## Testing Requirements
- Place tests in `/tests/` with matching structure to `/src/`
- Use pytest fixtures from `tests/conftest.py`
- Integration tests go in `tests/integration/`
- Performance tests go in `tests/performance/`

## Frontend Structure (React/TypeScript)
- Components in `frontend/src/components/` - reusable UI elements
- Pages in `frontend/src/pages/` - route-level components
- API calls in `frontend/src/services/api.ts` - centralized HTTP client
- Types in `frontend/src/types/index.ts` - TypeScript definitions
- Context providers in `frontend/src/contexts/` - state management
---
inclusion: always
---

# Technology Stack & Development Standards

## Core Technologies

**Backend**: Python 3.8+, FastAPI, PyTorch, Uvicorn ASGI server
**Frontend**: React 18 + TypeScript, Ant Design, Socket.io client  
**Testing**: pytest (backend), Jest + React Testing Library (frontend)
**Deployment**: Docker with docker-compose orchestration

## Critical Dependencies

**ML/Optimization**: PyTorch (primary framework), transformers (Hugging Face), bitsandbytes, auto-gptq, optimum
**System Monitoring**: psutil for resource metrics
**API Documentation**: FastAPI auto-generates OpenAPI/Swagger specs at `/docs` and `/redoc`

## Development Commands

Backend API server:
```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

Test suites:
```bash
python run_tests.py --suite unit              # Unit tests only
python run_tests.py --suite integration       # Integration tests
python run_tests.py --suite performance       # Performance benchmarks
```

Frontend development:
```bash
cd frontend && npm start                      # Dev server on port 3000
cd frontend && npm test                       # Run Jest tests
cd frontend && npm run build                  # Production build
```

Docker environments:
```bash
docker-compose -f deploy/docker-compose.dev.yml up     # Development
docker-compose -f deploy/docker-compose.prod.yml up    # Production
```

## Python Code Standards

**Type Hints**: Required for all function signatures, parameters, and return types
```python
async def process_model(model_id: str, config: OptimizationConfig) -> OptimizationResult:
    pass
```

**Async/Await**: MUST be used for all I/O operations (database, file system, external APIs, HTTP requests)
```python
# Correct
async def load_model(path: str) -> ModelData:
    async with aiofiles.open(path, 'rb') as f:
        data = await f.read()
    return ModelData(data)

# Incorrect - blocking I/O
def load_model(path: str) -> ModelData:
    with open(path, 'rb') as f:
        data = f.read()
    return ModelData(data)
```

**Structured Logging**: Always include context with extra parameter
```python
logger.info("Starting optimization", extra={
    "component": "QuantizationAgent",
    "model_id": model_id,
    "technique": "INT8"
})
```

**Import Order**: Enforced by isort
1. Standard library (os, sys, typing)
2. Third-party packages (torch, fastapi, pytest)
3. Local application imports (src.agents, src.models)

**Formatting**: Black and isort are enforced - code must pass these checks

## TypeScript/React Code Standards

**Strict Mode**: TypeScript strict mode is enabled - all types must be explicit

**Component Pattern**: Functional components with hooks only
```typescript
// Correct
const Dashboard: React.FC = () => {
  const [data, setData] = useState<DashboardData | null>(null);
  return <div>{data?.title}</div>;
};

// Incorrect - class components not allowed
class Dashboard extends React.Component { }
```

**API Calls**: MUST go through centralized `frontend/src/services/api.ts`
```typescript
// Correct
import { api } from '../services/api';
const result = await api.post('/optimize', data);

// Incorrect - direct fetch calls
const result = await fetch('http://localhost:8000/api/v1/optimize', { ... });
```

**Type Definitions**: All props, state, and API responses must be typed
```typescript
interface DashboardProps {
  userId: string;
  onUpdate: (data: OptimizationResult) => void;
}

const Dashboard: React.FC<DashboardProps> = ({ userId, onUpdate }) => {
  // Implementation
};
```

**Real-Time Updates**: Use WebSocket connections via `WebSocketContext`, not polling
```typescript
const { socket } = useWebSocket();
useEffect(() => {
  socket?.on('optimization:progress', handleProgress);
  return () => socket?.off('optimization:progress', handleProgress);
}, [socket]);
```

## Testing Standards

**Backend Tests**:
- Location: `tests/` directory mirroring `src/` structure
- Naming: `test_<module_name>.py` (e.g., `test_quantization_agent.py`)
- Fixtures: Use shared fixtures from `tests/conftest.py`
- Async tests: Use `@pytest.mark.asyncio` decorator
- Mocking: Use `pytest-mock` for external dependencies

**Frontend Tests**:
- Location: `frontend/src/tests/` directory
- Naming: `<ComponentName>.test.tsx` (e.g., `Dashboard.test.tsx`)
- Library: React Testing Library (test user behavior, not implementation)
- Mocking: Use `jest.mock()` for API calls and external dependencies
- Coverage: Aim for >80% coverage on critical paths

**Test Organization**:
- Unit tests: Test individual functions/components in isolation
- Integration tests: `tests/integration/` - test component interactions
- Performance tests: `tests/performance/` - benchmark optimization operations
- E2E tests: `frontend/src/tests/e2e/` - full user workflows

## Environment Configuration

**Never Hardcode**:
- File paths → Use `config/default.json` or environment variables
- API endpoints → Use environment-specific configs (`.env`, `.env.production`)
- Credentials/secrets → Use `.env` files, never commit to git
- Port numbers → Use environment variables with defaults

**Configuration Hierarchy** (highest to lowest priority):
1. Environment variables (`.env` files)
2. `config/default.json`
3. Code defaults (fallback values)

**Environment Files**:
- `.env.example` - Template with all required variables (commit this)
- `.env` - Local development values (never commit)
- `.env.production` - Production values (never commit)

## Critical Rules for AI Assistants

**When Writing Python Code**:
- Always add type hints to function signatures
- Use async/await for any I/O operation
- Import custom exceptions from `src/utils/exceptions.py`, never raise generic `Exception`
- Add structured logging with component context
- Follow the singleton pattern for services in `src/services/`

**When Writing TypeScript/React Code**:
- Use functional components with hooks exclusively
- Route all API calls through `frontend/src/services/api.ts`
- Explicitly type all props, state, and API responses
- Use contexts (`AuthContext`, `WebSocketContext`) for global state
- Test user interactions with React Testing Library, not implementation details

**When Running Commands**:
- Backend server: Use `uvicorn src.api.main:app --reload` for development
- Tests: Use `python run_tests.py --suite <type>` not direct pytest
- Frontend: Run from `frontend/` directory with `npm start` or `npm test`

**When Checking Code Quality**:
- Use `getDiagnostics` tool to check for TypeScript/linting errors
- Python code must pass Black and isort formatting
- All tests must pass before considering work complete
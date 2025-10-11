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
**API Documentation**: FastAPI auto-generates OpenAPI/Swagger specs

## Development Commands

Start backend API server:
```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

Run test suites:
```bash
python run_tests.py --suite unit|integration|performance
```

Start frontend development server:
```bash
cd frontend && npm start
```

Docker environments:
```bash
docker-compose -f deploy/docker-compose.dev.yml up    # Development
docker-compose -f deploy/docker-compose.prod.yml up   # Production
```

## Code Standards

**Python**:
- Type hints required for all function signatures
- Format with Black and isort (enforced)
- All I/O operations MUST use async/await syntax
- Use structured logging with context: `logger.info("msg", extra={"component": "Name"})`

**TypeScript/React**:
- Strict TypeScript mode enabled
- Functional components with hooks (no class components)
- All API calls through centralized `src/services/api.ts`
- Props and state must be explicitly typed

**Async Patterns**:
- Backend: async/await for all database, file I/O, and external API calls
- Frontend: WebSocket connections for real-time updates, not polling
- All agent operations are async and emit progress events

**Testing Requirements**:
- Backend tests mirror `src/` structure in `tests/`
- Test files prefixed with `test_`
- Use shared fixtures from `tests/conftest.py`
- Frontend tests colocated in `src/tests/`

## Environment Configuration

Never hardcode:
- File paths (use config or environment variables)
- API endpoints (use environment-specific configs)
- Credentials or secrets (use .env files, never commit)

Configuration hierarchy:
1. Environment variables (highest priority)
2. `config/default.json`
3. Code defaults (lowest priority)
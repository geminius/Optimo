---
inclusion: always
---

# Technology Stack

## Core Stack
- **Backend**: Python 3.8+ with FastAPI, PyTorch, Uvicorn ASGI server
- **Frontend**: React 18 + TypeScript, Ant Design components, Socket.io for WebSockets
- **Testing**: pytest (backend), Jest + React Testing Library (frontend)
- **Deployment**: Docker with docker-compose for multi-container orchestration

## Critical Libraries
- **Quantization**: bitsandbytes, auto-gptq, optimum
- **ML Framework**: PyTorch (primary), transformers (Hugging Face models)
- **Monitoring**: psutil for system metrics
- **API**: FastAPI with automatic OpenAPI/Swagger documentation

## Key Commands

**Start API server**: `uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload`

**Run tests**: `python run_tests.py` (supports `--suite unit|integration|performance`)

**Frontend dev**: `cd frontend && npm start`

**Docker**: `docker-compose -f docker-compose.dev.yml up` (dev) or `docker-compose.prod.yml` (prod)

## Code Quality Standards
- Format with Black and isort before committing
- Type hints required for all functions (checked with MyPy)
- All async I/O operations must use async/await
- WebSocket connections for real-time progress updates
---
description: Repository Information Overview
alwaysApply: true
---

# Robotics Model Optimization Platform

## Summary

This is a comprehensive AI agentic platform for automatically optimizing robotics models (e.g., OpenVLA). The system employs multi-agent architecture with specialized agents for analysis, planning, optimization, and evaluation. It supports quantization, pruning, knowledge distillation, architecture search, and compression techniques. The platform is built with a Python/FastAPI backend, React/TypeScript frontend, PostgreSQL database, Redis cache, and Docker containerization.

## Repository Structure

The project is organized as a multi-service application:

- **`src/`** - Python backend with agents, API, services, and configuration
- **`frontend/`** - React/TypeScript web interface
- **`tests/`** - Comprehensive test suite (unit, integration, performance, stress)
- **`deploy/`** - Docker and deployment configurations
- **`config/`** - Configuration files (default.json, optimization_criteria.json)
- **`examples/`** - Demo scripts for various agents and features
- **`docs/`** - Documentation and deployment guides
- **`scripts/`** - Deployment and utility scripts

## Backend

### Language & Runtime
**Language**: Python  
**Version**: 3.8+  
**Runtime**: Python 3.9 (Docker base image)  
**Build System**: setuptools  
**Package Manager**: pip

### Main Dependencies
- **Framework**: FastAPI 0.68.0+, Uvicorn 0.15.0+
- **ML/Optimization**: PyTorch 2.0.0+, Transformers 4.20.0+, bitsandbytes 0.35.0+, auto-gptq, optimum
- **Database**: SQLAlchemy 1.4.0+, Alembic 1.7.0+
- **API/WebSocket**: python-socketio 5.7.0+, aiohttp 3.8.0+, PyJWT 2.0.0+
- **Monitoring**: psutil 5.8.0+, watchdog 2.1.0+
- **Dev**: black, flake8, mypy, pytest, pytest-asyncio, pytest-cov

### Build & Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Install as editable package
pip install -e .

# Run main platform
python -m src.main

# Run API server only
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

# With custom config
python -m src.main --config config.json
```

### Main Entry Points

- **`src/main.py`** - Complete integrated platform entry point (RoboticsOptimizationPlatform class)
- **`src/api/main.py`** - FastAPI application for API-only mode

## Frontend

### Language & Runtime
**Language**: TypeScript/React  
**Node.js**: 18+ (Docker base image)  
**Package Manager**: npm  
**Build Tool**: react-scripts (Create React App)

### Main Dependencies
- **Framework**: React 18.2.0+, React Router DOM 6.3.0+
- **UI**: Ant Design 4.24.0+, Recharts 2.5.0+ (charts)
- **HTTP**: Axios 0.27.2+
- **WebSocket**: socket.io-client 4.5.0+
- **Auth**: jwt-decode 4.0.0+
- **Dev**: TypeScript 4.7.4+, Jest, React Testing Library

### Build & Installation

```bash
cd frontend

# Install dependencies
npm install

# Start development server
npm start

# Build for production
npm run build

# Run tests
npm test

# Run tests with coverage
npm run test:coverage

# E2E tests
npm run test:e2e
```

## Database & Cache

**Database**: PostgreSQL 15 (Docker image: postgres:15-alpine)  
**Cache**: Redis 7 (Docker image: redis:7-alpine)  
**Connection**: SQLAlchemy ORM with Alembic migrations

## Docker Configuration

### API Service
**Dockerfile**: `deploy/Dockerfile`
- **Base**: python:3.9-slim
- **Port**: 8000
- **Health Check**: HTTP `/health` endpoint (30s interval)
- **Entry Point**: uvicorn `src.api.main:app`

### Worker Service
**Dockerfile**: `deploy/Dockerfile.worker`
- Async optimization workers (2 replicas by default)
- Same base image and dependencies as API

### Frontend Service
**Dockerfile**: `frontend/Dockerfile`
- **Multi-stage build**: Node 18 → nginx:alpine
- **Port**: 80 (nginx reverse proxy)
- **Build**: npm build → nginx serving

### Docker Compose
**File**: `deploy/docker-compose.yml`
**Services**:
- PostgreSQL (5432)
- Redis (6379)
- API (8000)
- Worker (scalable)
- Frontend (3000)
- Nginx (80/443)

**Environment Variables** (from `.env`):
- `POSTGRES_DB`, `POSTGRES_USER`, `POSTGRES_PASSWORD`
- `ENVIRONMENT` (development/production)
- `REACT_APP_API_URL`, `REACT_APP_WS_URL`

## Testing

**Framework**: pytest 6.2.0+  
**Async Support**: pytest-asyncio 0.18.0+  
**Coverage**: pytest-cov 2.12.0+ (80% minimum)  

### Test Structure
- **Unit Tests**: `tests/test_*.py` (agents, services, models)
- **Integration Tests**: `tests/integration/`
- **Performance Tests**: `tests/performance/` (benchmarks)
- **Stress Tests**: `tests/stress/` (concurrent sessions, load)
- **Test Data**: `tests/data/` (fixtures and generation)

### Configuration
**File**: `pytest.ini`
- Test markers: unit, integration, performance, stress, slow, gpu, network, asyncio
- Coverage report: HTML + XML
- Timeout: 300 seconds per test
- Output: junit XML, coverage HTML

### Run Commands

```bash
# All tests
pytest

# By category
pytest -m unit
pytest -m integration
pytest -m performance

# With coverage
pytest --cov=src --cov-report=html

# Quick mode (skip slow tests)
pytest --quick

# Using test runner
python3 run_tests.py --suite all
python3 run_tests.py --suite unit
```

## Configuration

### Environment Setup
1. Copy `.env.example` to `.env` (root) and `frontend/.env.example` to `frontend/.env`
2. Update `JWT_SECRET_KEY` with secure random string
3. Configure API/WebSocket URLs and database connections

### Configuration Files
- **`config/default.json`** - Default platform settings (logging, monitoring, optimization manager)
- **`config/optimization_criteria.json`** - Optimization constraints and techniques

### Key Environment Variables
- `JWT_SECRET_KEY` - JWT signing key (required)
- `DATABASE_URL` - Database connection (PostgreSQL, MySQL, SQLite)
- `REDIS_URL` - Redis connection for caching
- `API_PORT` - API server port (default: 8000)
- `LOG_LEVEL` - Logging level (INFO, DEBUG, etc.)
- `MAX_CONCURRENT_SESSIONS` - Concurrent optimization limit (default: 5)

## Build & Deployment

### Makefile Commands

```bash
make build           # Build all Docker images
make deploy          # Deploy to development
make deploy-prod     # Deploy to production
make test            # Run deployment tests
make health          # Check service health
make logs            # Show all service logs
make restart         # Restart all services
make scale-workers   # Scale worker services
```

### Entry Points
- **Backend API**: `http://localhost:8000` (Swagger UI: `/docs`)
- **Frontend UI**: `http://localhost:3000`
- **WebSocket**: `ws://localhost:8000`

### Database & Cache
- **PostgreSQL**: `localhost:5432`
- **Redis**: `localhost:6379`

## Agents & Optimization

The platform implements specialized agents:
- **AnalysisAgent** - Model profiling and analysis
- **PlanningAgent** - Optimization strategy determination
- **QuantizationAgent** - 4-bit, 8-bit, dynamic quantization
- **PruningAgent** - Structured/unstructured pruning
- **DistillationAgent** - Knowledge distillation
- **ArchitectureSearchAgent** - Neural architecture search
- **CompressionAgent** - Tensor decomposition
- **EvaluationAgent** - Benchmarking and testing

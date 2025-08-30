# Technology Stack

## Backend
- **Python 3.8+**: Core platform language
- **FastAPI**: REST API framework with automatic OpenAPI documentation
- **PyTorch**: Primary ML framework for model optimization
- **SQLAlchemy + Alembic**: Database ORM and migrations
- **Uvicorn**: ASGI server for FastAPI

## Frontend
- **React 18**: UI framework with TypeScript
- **Ant Design**: UI component library
- **React Router**: Client-side routing
- **Axios**: HTTP client for API communication
- **Socket.io**: Real-time WebSocket communication
- **Recharts**: Data visualization

## Key Libraries
- **bitsandbytes**: Advanced quantization techniques
- **transformers**: Hugging Face model support
- **auto-gptq**: GPTQ quantization
- **optimum**: Model optimization utilities
- **psutil**: System monitoring
- **pytest**: Testing framework

## Development Tools
- **Black**: Code formatting
- **Flake8**: Linting
- **MyPy**: Type checking
- **Docker**: Containerization
- **Make**: Build automation

## Common Commands

### Development Setup
```bash
# Install dependencies
pip install -r requirements.txt
pip install -e .

# Install frontend dependencies
cd frontend && npm install
```

### Running the Platform
```bash
# Complete platform
python -m src.main

# API server only
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

# Frontend development server
cd frontend && npm start
```

### Testing
```bash
# Run all tests
python run_tests.py

# Run specific test suites
python run_tests.py --suite unit
python run_tests.py --suite integration
python run_tests.py --suite performance

# Frontend tests
cd frontend && npm test
```

### Docker Deployment
```bash
# Development environment
docker-compose -f docker-compose.dev.yml up

# Production environment
docker-compose -f docker-compose.prod.yml up
```

### Code Quality
```bash
# Format code
black src/ tests/
isort src/ tests/

# Run linting
flake8 src/ tests/
mypy src/

# Pre-commit hooks
pre-commit install
```
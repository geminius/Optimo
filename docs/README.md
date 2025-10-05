# Robotics Model Optimization Platform - Documentation

Welcome to the documentation for the Robotics Model Optimization Platform.

## Quick Links

- [Main README](../README.md) - Project overview and setup
- [Deployment Guide](../DEPLOYMENT.md) - Deployment instructions



### API Documentation
- OpenAPI documentation available at `/docs` when running the API server
- Run `uvicorn src.api.main:app --reload` and visit `http://localhost:8000/docs`

## Project Structure

```
.
├── src/                    # Source code
│   ├── agents/            # Optimization agents
│   ├── api/               # REST API
│   ├── config/            # Configuration
│   ├── models/            # Data models
│   ├── services/          # Business logic services
│   └── utils/             # Utilities
├── tests/                 # Test suite
│   ├── integration/       # Integration tests
│   ├── performance/       # Performance benchmarks
│   ├── stress/           # Stress tests
│   ├── data/             # Test data generation
│   └── automation/       # Test automation
├── frontend/             # React web interface
├── docs/                 # Documentation (you are here)
├── config/               # Configuration files
├── scripts/              # Utility scripts
└── examples/             # Example usage
```

## Key Features

### Autonomous Optimization
- Automatic model analysis and optimization
- Multi-agent architecture
- Support for multiple optimization techniques

### Optimization Techniques
- Quantization (4-bit, 8-bit, AWQ, SmoothQuant)
- Pruning (structured and unstructured)
- Knowledge Distillation
- Neural Architecture Search
- Tensor Compression

### Model Support
- PyTorch models
- TensorFlow models
- ONNX models
- SafeTensors format

### Monitoring & Evaluation
- Real-time progress tracking
- Comprehensive performance metrics
- Automatic rollback on failures
- Detailed evaluation reports

## Running the Platform

### Development Mode
```bash
# Start the complete platform
python -m src.main

# Or start components separately
uvicorn src.api.main:app --reload  # API server
cd frontend && npm start            # Web interface
```

### Production Mode
```bash
# Using Docker Compose
docker-compose -f docker-compose.prod.yml up
```

### Running Tests

```bash
# Run all tests
python3 run_tests.py --suite all

# Run specific test suites
python3 run_tests.py --suite unit
python3 run_tests.py --suite integration
python3 run_tests.py --suite performance
python3 run_tests.py --suite stress

# Generate specific report formats
python3 run_tests.py --suite all --formats json html txt

# Custom output directory
python3 run_tests.py --suite all --output-dir custom_results
```

#### Using Pytest Directly

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/integration/test_end_to_end_workflows.py

# Run with markers
pytest -m integration
pytest -m performance
pytest -m stress

# Quick mode (skip slow tests)
pytest --quick

# With coverage
pytest --cov=src --cov-report=html
```

#### Test Structure

The test suite is organized into multiple categories:

- **Unit Tests** (`tests/test_*.py`) - Individual component testing with mocks
- **Integration Tests** (`tests/integration/`) - End-to-end workflow testing
- **Performance Tests** (`tests/performance/`) - Optimization speed and accuracy benchmarks
- **Stress Tests** (`tests/stress/`) - Concurrent session and resource testing
- **Test Data** (`tests/data/`) - Test data generation utilities

#### Supported Model Types

1. **CNN** - Convolutional Neural Networks
2. **Transformer** - Attention-based models
3. **ResNet** - Residual networks
4. **LSTM** - Recurrent networks
5. **MLP** - Multi-layer perceptrons
6. **Robotics VLA** - Vision-Language-Action models

#### Test Configuration

- **Pytest Config**: See `pytest.ini` at project root
- **Test Fixtures**: See `tests/conftest.py` for available fixtures
- **Custom Options**: `--quick`, `--gpu`, `--stress`

## Configuration

### Optimization Criteria
Configure optimization criteria in `src/config/optimization_criteria.py`:
- Performance thresholds
- Accuracy constraints
- Allowed techniques
- Target deployment environment

### API Configuration
Configure API settings in environment variables or `.env` files:
- `API_HOST`: API server host (default: 0.0.0.0)
- `API_PORT`: API server port (default: 8000)
- `DATABASE_URL`: Database connection string
- `LOG_LEVEL`: Logging level (default: INFO)

## Architecture

### Agent Layer
- **AnalysisAgent**: Model analysis and profiling
- **PlanningAgent**: Optimization strategy planning
- **OptimizationAgents**: Execute optimization techniques
- **EvaluationAgent**: Performance evaluation

### Service Layer
- **OptimizationManager**: Orchestrates optimization workflow
- **MemoryManager**: Session and history management
- **NotificationService**: Real-time status updates
- **ModelStore**: Model storage and versioning

### API Layer
- REST API with FastAPI
- WebSocket support for real-time updates
- OpenAPI documentation
- Authentication and authorization

### Web Interface
- React-based UI
- Real-time progress monitoring
- Optimization history visualization
- Configuration management

## Development Workflow

### Adding New Features
1. Create feature branch
2. Implement feature with tests
3. Run test suite: `python3 run_tests.py --suite all`
4. Update documentation
5. Submit pull request

### Code Quality
```bash
# Format code
black src/ tests/
isort src/ tests/

# Run linting
flake8 src/ tests/
mypy src/

# Run tests with coverage
pytest --cov=src --cov-report=html
```

## Troubleshooting

### Common Issues

**Import Errors**
```bash
pip install -r requirements.txt
pip install -e .
```

**Database Connection Issues**
Check `DATABASE_URL` environment variable and ensure database is running.

**CUDA Not Available**
GPU-based optimizations will fall back to CPU. Install CUDA toolkit for GPU support.

**Port Already in Use**
Change port in configuration or stop conflicting service.

## Contributing

We welcome contributions! Please:
1. Follow the existing code style
2. Add tests for new features
3. Update documentation
4. Ensure all tests pass

## Support

For issues, questions, or contributions:
1. Check this documentation
2. Review existing issues
3. Create a new issue with detailed information

## License

[Add license information here]

## Acknowledgments

Built with:
- PyTorch for deep learning
- FastAPI for REST API
- React for web interface
- Docker for containerization

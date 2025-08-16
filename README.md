# Robotics Model Optimization Platform

An AI agentic platform for automatically optimizing robotics models like OpenVLA. The platform employs autonomous agents that can analyze models, determine optimization opportunities based on predefined criteria, and execute optimizations while providing comprehensive evaluation capabilities.

## 🚀 Features

### Core Capabilities
- **🤖 Multi-Agent Architecture**: Specialized agents for analysis, optimization, planning, and evaluation
- **⚙️ Autonomous Optimization**: Fully automated model optimization with minimal human intervention
- **📊 Comprehensive Evaluation**: Automated testing and benchmarking of optimized models
- **🔧 Configurable Criteria**: Flexible optimization constraints and performance thresholds
- **📈 Real-time Monitoring**: Live progress tracking and system health monitoring
- **🔄 Error Recovery**: Automatic rollback and retry mechanisms for failed optimizations

### Optimization Techniques
- **Quantization**: 4-bit, 8-bit, and dynamic quantization using bitsandbytes, AWQ, SmoothQuant
- **Pruning**: Structured and unstructured pruning with various sparsity patterns
- **Knowledge Distillation**: Model compression through teacher-student training
- **Architecture Search**: Neural architecture search for optimal model structures
- **Compression**: Tensor decomposition and other compression techniques

### Model Support
- **Formats**: PyTorch (.pth, .pt), TensorFlow (.pb, .h5), ONNX (.onnx), SafeTensors (.safetensors)
- **Architectures**: Vision-Language-Action models (OpenVLA), Transformer models, CNNs, and custom architectures
- **Frameworks**: Native support for PyTorch, TensorFlow, and ONNX Runtime

## 🏗️ Architecture

The platform follows a layered architecture with autonomous agents:

```
┌─────────────────────────────────────────────────────────────┐
│                    Web Interface & API                      │
├─────────────────────────────────────────────────────────────┤
│              Integration & Orchestration Layer              │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐│
│  │ PlatformIntegr. │ │ WorkflowOrch.   │ │ MonitoringInteg.││
│  └─────────────────┘ └─────────────────┘ └─────────────────┘│
├─────────────────────────────────────────────────────────────┤
│                   Service Layer                             │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐│
│  │ OptimizationMgr │ │ ModelStore      │ │ MemoryManager   ││
│  └─────────────────┘ └─────────────────┘ └─────────────────┘│
├─────────────────────────────────────────────────────────────┤
│                    Agent Layer                              │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐│
│  │ AnalysisAgent   │ │ PlanningAgent   │ │ EvaluationAgent ││
│  └─────────────────┘ └─────────────────┘ └─────────────────┘│
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐│
│  │ QuantizationAgt │ │ PruningAgent    │ │ DistillationAgt ││
│  └─────────────────┘ └─────────────────┘ └─────────────────┘│
└─────────────────────────────────────────────────────────────┘
```

## 📁 Project Structure

```
├── src/
│   ├── agents/              # Autonomous agent implementations
│   │   ├── analysis/        # Model analysis and profiling
│   │   ├── evaluation/      # Model evaluation and benchmarking
│   │   ├── optimization/    # Optimization technique agents
│   │   ├── planning/        # Optimization strategy planning
│   │   └── base.py          # Base agent interfaces and protocols
│   ├── api/                 # REST API and web interface
│   │   ├── main.py          # FastAPI application
│   │   ├── models.py        # API data models
│   │   ├── auth.py          # Authentication and authorization
│   │   └── monitoring.py    # API monitoring endpoints
│   ├── integration/         # Platform integration layer
│   │   ├── platform_integration.py    # Component wiring and lifecycle
│   │   ├── workflow_orchestrator.py   # End-to-end workflow management
│   │   ├── logging_integration.py     # Centralized logging
│   │   └── monitoring_integration.py  # System monitoring
│   ├── services/            # Core service implementations
│   │   ├── optimization_manager.py    # Central optimization coordinator
│   │   ├── model_store.py             # Model storage and versioning
│   │   ├── memory_manager.py          # Session and state management
│   │   ├── notification_service.py    # Event notifications
│   │   └── monitoring_service.py      # System health monitoring
│   ├── models/              # Data models and schemas
│   │   ├── core.py          # Core data structures
│   │   ├── validation.py    # Data validation logic
│   │   └── store.py         # Model storage abstractions
│   ├── config/              # Configuration management
│   │   └── optimization_criteria.py   # Optimization criteria definitions
│   ├── utils/               # Utility modules
│   │   ├── exceptions.py    # Custom exception classes
│   │   ├── retry.py         # Retry mechanisms
│   │   └── recovery.py      # Error recovery strategies
│   └── main.py              # Main platform entry point
├── frontend/                # React-based web interface
│   ├── src/
│   │   ├── components/      # React components
│   │   ├── pages/           # Application pages
│   │   ├── services/        # API client services
│   │   └── contexts/        # React contexts
├── tests/                   # Comprehensive test suite
│   ├── integration/         # Integration and end-to-end tests
│   ├── performance/         # Performance benchmarks
│   ├── stress/              # Stress and load tests
│   ├── automation/          # Test automation framework
│   └── data/                # Test data generation
├── examples/                # Usage examples and demos
├── scripts/                 # Deployment and utility scripts
└── docs/                    # Documentation
```

## 🚀 Quick Start

### Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd robotics-model-optimization-platform
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
pip install -e .
```

3. **Install optional dependencies** (for specific optimization techniques):
```bash
# For advanced quantization
pip install bitsandbytes transformers

# For ONNX support
pip install onnx onnxruntime

# For TensorFlow support
pip install tensorflow
```

### Running the Platform

#### Option 1: Complete Platform (Recommended)
```bash
# Start the complete integrated platform
python -m src.main

# With custom configuration
python -m src.main --config config.json

# Test with a specific model
python -m src.main --test-workflow path/to/your/model.pth
```

#### Option 2: API Server Only
```bash
# Start just the API server
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

#### Option 3: Web Interface
```bash
# Start the frontend (in a separate terminal)
cd frontend
npm install
npm start
```

### Basic Usage

#### Programmatic API
```python
from src.main import RoboticsOptimizationPlatform
from src.config.optimization_criteria import OptimizationCriteria, OptimizationConstraints, OptimizationTechnique

# Create and start platform
platform = RoboticsOptimizationPlatform()
await platform.start()

# Define optimization criteria
constraints = OptimizationConstraints(
    preserve_accuracy_threshold=0.95,
    max_size_reduction_percent=50.0,
    allowed_techniques=[OptimizationTechnique.QUANTIZATION, OptimizationTechnique.PRUNING]
)

criteria = OptimizationCriteria(
    name="robotics_optimization",
    description="Optimize for robotics deployment",
    constraints=constraints,
    target_deployment="edge_device"
)

# Execute optimization
result = await platform.optimize_model(
    model_path="path/to/your/model.pth",
    criteria=criteria,
    user_id="researcher_123"
)

# Check results
if result["success"]:
    print(f"Optimization completed! Session ID: {result['session_id']}")
    print(f"Execution time: {result['execution_time_seconds']:.2f}s")
else:
    print(f"Optimization failed: {result['error_message']}")

# Shutdown platform
await platform.stop()
```

#### REST API
```bash
# Upload a model
curl -X POST "http://localhost:8000/models/upload" \
  -H "Authorization: Bearer <token>" \
  -F "file=@model.pth" \
  -F "name=MyRoboticsModel"

# Start optimization
curl -X POST "http://localhost:8000/optimize" \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "model-uuid",
    "criteria_name": "robotics_optimization",
    "target_accuracy_threshold": 0.95,
    "optimization_techniques": ["quantization", "pruning"]
  }'

# Check optimization status
curl "http://localhost:8000/sessions/{session_id}/status" \
  -H "Authorization: Bearer <token>"

# Get results
curl "http://localhost:8000/sessions/{session_id}/results" \
  -H "Authorization: Bearer <token>"
```

## ⚙️ Configuration

The platform uses JSON configuration files for customization:

```json
{
  "logging": {
    "level": "INFO",
    "log_dir": "logs",
    "json_format": true,
    "console_output": true
  },
  "monitoring": {
    "monitoring_interval_seconds": 30,
    "health_check_interval_seconds": 60,
    "alert_thresholds": {
      "cpu_percent": 80.0,
      "memory_percent": 85.0,
      "disk_percent": 90.0
    }
  },
  "optimization_manager": {
    "max_concurrent_sessions": 5,
    "auto_rollback_on_failure": true,
    "session_timeout_minutes": 240
  },
  "quantization_agent": {
    "default_bits": 8,
    "enable_dynamic": true
  },
  "pruning_agent": {
    "default_sparsity": 0.5,
    "enable_structured": true
  }
}
```

## 🧪 Testing

### Run All Tests
```bash
# Run comprehensive test suite
python run_tests.py

# Run specific test categories
python run_tests.py --suite unit
python run_tests.py --suite integration
python run_tests.py --suite performance

# Quick test subset
python run_tests.py --quick
```

### Integration Validation
```bash
# Validate platform integration
python validate_integration.py

# Run integration test suite
python scripts/run_integration_tests.py
```

### Performance Benchmarks
```bash
# Run performance benchmarks
python -m pytest tests/performance/ -v

# Run stress tests
python -m pytest tests/stress/ -v
```

## 📊 Monitoring and Observability

The platform includes comprehensive monitoring capabilities:

- **System Metrics**: CPU, memory, disk, and network usage
- **Component Health**: Real-time health checking of all agents and services
- **Performance Tracking**: Optimization execution times and resource usage
- **Error Monitoring**: Automatic error detection and alerting
- **Audit Logging**: Complete audit trail of all optimization activities

Access monitoring dashboards at `http://localhost:8000/monitoring` when running the API server.

## 🔧 Development

### Setting up Development Environment
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Run code formatting
black src/ tests/
isort src/ tests/

# Run linting
flake8 src/ tests/
mypy src/
```

### Adding New Optimization Techniques

1. **Create Agent Class**:
```python
from src.agents.base import BaseOptimizationAgent

class MyOptimizationAgent(BaseOptimizationAgent):
    def can_optimize(self, model: torch.nn.Module) -> bool:
        # Implementation here
        pass
    
    def optimize(self, model: torch.nn.Module, config: Dict) -> OptimizedModel:
        # Implementation here
        pass
```

2. **Register Agent**:
Add your agent to the platform configuration and integration layer.

3. **Add Tests**:
Create comprehensive tests in `tests/test_my_optimization_agent.py`.

### Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes and add tests
4. Run the test suite (`python run_tests.py`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## 📚 Documentation

- **API Documentation**: Available at `http://localhost:8000/docs` when running the server
- **Architecture Guide**: See `docs/architecture.md`
- **Agent Development**: See `docs/agent_development.md`
- **Deployment Guide**: See `DEPLOYMENT.md`
- **Integration Summary**: See `INTEGRATION_SUMMARY.md`

## 🚀 Deployment

### Docker Deployment
```bash
# Development environment
docker-compose -f docker-compose.dev.yml up

# Production environment
docker-compose -f docker-compose.prod.yml up
```

### Kubernetes Deployment
```bash
# Apply Kubernetes manifests
kubectl apply -f k8s/

# Check deployment status
kubectl get pods -l app=robotics-optimization-platform
```

See `DEPLOYMENT.md` for detailed deployment instructions.

## 🤝 Support

- **Issues**: Report bugs and feature requests on GitHub Issues
- **Discussions**: Join community discussions on GitHub Discussions
- **Documentation**: Comprehensive docs available in the `docs/` directory
- **Examples**: Check the `examples/` directory for usage examples

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built upon the OpenVLA planner prototype
- Inspired by modern MLOps and model optimization practices
- Thanks to the robotics and AI community for feedback and contributions

## 🔗 Related Projects

- [OpenVLA](https://github.com/openvla/openvla): Open-source Vision-Language-Action models
- [bitsandbytes](https://github.com/TimDettmers/bitsandbytes): Efficient quantization library
- [ONNX](https://onnx.ai/): Open Neural Network Exchange format

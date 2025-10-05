# Testing Documentation

This directory contains comprehensive documentation for the Robotics Model Optimization Platform testing suite.

## Quick Start

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

### Using Pytest Directly

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

## Documentation Files

### [TEST_SUITE_SUMMARY.md](./TEST_SUITE_SUMMARY.md)
Comprehensive overview of the testing suite implementation, including:
- All test components and their features
- Test coverage details
- Model types and scenarios supported
- Execution instructions

### [TASK_17_VERIFICATION.md](./TASK_17_VERIFICATION.md)
Detailed verification report confirming all testing components are implemented:
- Subtask verification checklist
- File structure verification
- Requirements coverage mapping
- Implementation status

### [TASK_STATUS_CONFIRMATION.md](./TASK_STATUS_CONFIRMATION.md)
Status confirmation report with:
- Current task completion status
- Line count verification
- File existence verification
- Overall project status

## Test Structure

```
tests/
├── integration/          # End-to-end workflow tests
├── performance/          # Performance benchmarks
├── stress/              # Concurrent session stress tests
├── data/                # Test data generation
├── automation/          # Automated test execution
├── conftest.py          # Pytest configuration
└── test_*.py            # Unit tests
```

## Test Suites

### Unit Tests
- Individual component testing
- Mock-based isolation
- Fast execution
- Located in `tests/test_*.py`

### Integration Tests
- End-to-end workflow testing
- Multi-component interaction
- Real optimization scenarios
- Located in `tests/integration/`

### Performance Tests
- Optimization speed benchmarks
- Accuracy retention metrics
- Memory usage tracking
- Located in `tests/performance/`

### Stress Tests
- Concurrent session handling
- Memory pressure testing
- Resource monitoring
- Located in `tests/stress/`

## Test Coverage

- **Integration Tests:** 6+ test scenarios
- **Performance Benchmarks:** 6+ benchmarks across 3 model types
- **Stress Tests:** 5+ stress scenarios
- **Test Data Generation:** 7+ scenarios, 6 model types
- **Total Test Code:** 2,625+ lines

## Model Types Supported

1. **CNN** - Convolutional Neural Networks
2. **Transformer** - Attention-based models
3. **ResNet** - Residual networks
4. **LSTM** - Recurrent networks
5. **MLP** - Multi-layer perceptrons
6. **Robotics VLA** - Vision-Language-Action models

## Test Scenarios

- Simple optimizations (conservative settings)
- Moderate optimizations (balanced settings)
- Complex optimizations (multiple techniques)
- Edge cases (impossible constraints, tiny models)
- Robotics-specific scenarios

## Report Formats

### JSON Reports
Machine-readable format for CI/CD integration and automated analysis.

### HTML Reports
Web-viewable format with interactive visualizations and detailed breakdowns.

### Text Reports
Human-readable format for quick review and terminal output.

## Configuration

### Pytest Configuration
See `pytest.ini` at project root for pytest settings.

### Test Fixtures
See `tests/conftest.py` for available fixtures and test utilities.

### Custom Options
- `--quick`: Skip slow tests
- `--gpu`: Run GPU tests (requires CUDA)
- `--stress`: Run stress tests

## Continuous Integration

The test suite is designed for CI/CD integration:

```yaml
# Example CI configuration
test:
  script:
    - python3 run_tests.py --suite all --formats json
    - pytest --cov=src --cov-report=xml
```

## Troubleshooting

### Missing Dependencies
```bash
pip install -r requirements.txt
```

### CUDA Not Available
GPU tests will be automatically skipped if CUDA is not available.

### Slow Tests
Use `--quick` flag to skip slow performance and stress tests:
```bash
pytest --quick
```

### Memory Issues
Reduce concurrent sessions in stress tests or run tests individually:
```bash
python3 run_tests.py --suite unit
python3 run_tests.py --suite integration
```

## Contributing

When adding new tests:
1. Follow existing test structure and naming conventions
2. Add appropriate markers (`@pytest.mark.integration`, etc.)
3. Update test documentation
4. Ensure tests are isolated and repeatable
5. Add fixtures to `conftest.py` if needed

## Support

For issues or questions about testing:
1. Check this documentation
2. Review test examples in `tests/`
3. Check pytest output for detailed error messages
4. Review test logs in `test_results/` directory

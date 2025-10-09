# Test Runner Guide - Optimo Platform

## ✅ Test Runner Fixed!

The automated test runner now works correctly. Issues fixed:
1. Pytest glob pattern wasn't being expanded properly
2. Output buffering was causing hangs
3. Timeout was too short for large test suites
4. **Ctrl+C now works** - You can interrupt tests at any time

## Usage

### Activate Virtual Environment
```bash
source venv/bin/activate
```

### Run Test Suites

**Note**: Press **Ctrl+C** at any time to cancel test execution. Partial results will be saved.

```bash
# Run unit tests (689 tests - takes 5-10 minutes)
python run_tests.py --suite unit

# Run integration tests
python run_tests.py --suite integration

# Run performance benchmarks
python run_tests.py --suite performance

# Run stress tests
python run_tests.py --suite stress

# Run all test suites
python run_tests.py --suite all
```

### Options

```bash
# Specify output directory
python run_tests.py --suite unit --output-dir my_results

# Choose report formats
python run_tests.py --suite unit --formats json html txt

# Run in parallel (where supported)
python run_tests.py --suite unit --parallel
```

## Test Results

Results are saved in `test_results/` directory with:
- **JSON format**: Machine-readable results
- **HTML format**: Browser-viewable report
- **TXT format**: Human-readable summary

## Test Suite Overview

### Unit Tests (689 tests)
- Base agents and optimization agents
- Core services (memory, storage, monitoring)
- API endpoints
- Data models and configuration
- Error handling

**Expected Duration**: 5-10 minutes

### Integration Tests
- End-to-end workflows
- Platform initialization
- API integration
- Requirements validation

**Expected Duration**: 20-30 minutes

### Performance Tests
- Optimization benchmarks
- Speed and memory metrics
- Accuracy retention tests

**Expected Duration**: 30-60 minutes

### Stress Tests
- Concurrent operations
- Memory pressure
- Resource limits

**Expected Duration**: 30-60 minutes

## Known Test Failures

Some tests may fail due to:
1. **API authentication** (401 errors) - Expected in test environment
2. **OptimizationCriteria parameter mismatches** - Schema changes
3. **Mock configuration issues** - Test setup problems

These are test-level issues, not test runner issues.

## Troubleshooting

### Tests taking too long
- Unit tests with 689 tests take 5-10 minutes
- This is normal for comprehensive test suites
- Use `--suite unit` for faster feedback than `--suite all`
- **Press Ctrl+C to cancel** if you need to stop early

### Can't stop tests
- **Fixed!** Ctrl+C now works properly
- Tests will terminate gracefully
- Partial results are saved automatically

### Timeout errors
- Default timeout is 30 minutes per suite
- Increase if needed by modifying `timeout` parameter in `test_runner.py`

### Import errors
- Ensure venv is activated: `source venv/bin/activate`
- Verify dependencies: `pip install -r requirements.txt requirements_test.txt`

## Direct Pytest Alternative

For faster iteration on specific tests:

```bash
# Single file
pytest tests/test_models.py -v

# Specific test
pytest tests/test_models.py::TestModelMetadata::test_valid_model_metadata -v

# With coverage
pytest tests/test_optimization_manager.py --cov=src.services -v
```

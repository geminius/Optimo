#!/bin/bash
# Run load tests and generate performance report

set -e

echo "=========================================="
echo "Running Load Tests"
echo "=========================================="
echo ""

# Check if API server is running
if ! curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "⚠️  API server is not running on http://localhost:8000"
    echo "Please start the server first:"
    echo "  uvicorn src.api.main:app --host 0.0.0.0 --port 8000"
    exit 1
fi

echo "✓ API server is running"
echo ""

# Create results directory
mkdir -p test_results/load_tests

# Run load tests
echo "Running load tests..."
pytest tests/performance/test_load_testing.py -v -s --tb=short \
    --json-report --json-report-file=test_results/load_tests/results.json \
    2>&1 | tee test_results/load_tests/output.log

# Check if tests passed
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo ""
    echo "✓ All load tests passed"
else
    echo ""
    echo "⚠️  Some load tests failed"
fi

echo ""
echo "=========================================="
echo "Load Test Results"
echo "=========================================="
echo ""
echo "Results saved to: test_results/load_tests/"
echo "  - results.json: Test results in JSON format"
echo "  - output.log: Full test output"
echo ""

# Generate performance report if analyzer script exists
if [ -f "scripts/analyze_load_test_results.py" ]; then
    echo "Generating performance report..."
    python scripts/analyze_load_test_results.py test_results/load_tests/results.json
    echo ""
fi

echo "=========================================="
echo "Load Testing Complete"
echo "=========================================="

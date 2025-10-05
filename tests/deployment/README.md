# Deployment Validation Tests

This directory contains comprehensive deployment validation tests for the Robotics Model Optimization Platform. These tests validate that the deployment meets requirements 6.1 and 6.2, ensuring proper model format support and deployment readiness.

## Test Structure

### Core Test Files

- **`test_deployment_validation.py`** - Basic deployment health and functionality tests
- **`test_production_deployment.py`** - Production-specific configuration and security tests  
- **`test_deployment_integration.py`** - End-to-end deployment integration tests
- **`conftest.py`** - Pytest configuration and fixtures for deployment tests

### Test Categories

#### 1. Basic Deployment Validation (`test_deployment_validation.py`)

**TestDeploymentValidation**
- Docker services running and healthy
- Service health checks (API, database, Redis)
- API endpoints accessible
- Frontend accessibility
- Database schema validation
- Authentication endpoints
- File upload functionality
- WebSocket connectivity
- Nginx proxy configuration
- Resource limits compliance
- Log accessibility
- Environment configuration

**TestModelFormatSupport** (Requirement 6.1)
- PyTorch model upload support (.pth, .pt)
- ONNX model upload support (.onnx)
- TensorFlow model upload support (.h5)
- Unsupported format error handling

**TestOptimizationTechniqueSupport** (Requirement 6.3)
- Quantization technique availability
- Pruning technique availability
- Knowledge distillation availability
- Architecture search availability

**TestDeploymentPerformance**
- API response time validation
- Frontend load time validation
- Database query performance

**TestDeploymentSecurity**
- Default credentials validation
- Service isolation checks
- Port exposure validation

#### 2. Production Deployment Tests (`test_production_deployment.py`)

**TestProductionDeployment**
- Production environment detection
- Debug mode disabled
- SSL configuration
- Resource limits enforcement
- Service replica configuration
- Logging configuration
- Security headers validation

**TestDevelopmentDeployment**
- Development environment detection
- Debug endpoints availability
- Hot reload functionality
- Database port exposure

**TestDeploymentBackup**
- Backup script functionality
- Backup data validation
- Database recovery capability

**TestDeploymentUpgrade**
- Rolling update capability
- Configuration update handling
- Data migration readiness

#### 3. Integration Tests (`test_deployment_integration.py`)

**TestEndToEndDeployment**
- Complete optimization workflow
- WebSocket progress updates
- Concurrent optimization handling
- Error handling and recovery
- System resource monitoring
- API rate limiting
- Data persistence across restarts
- Backup and restore workflow

**TestDeploymentScalability**
- Horizontal scaling
- Load balancing functionality
- Resource utilization under load

## Running the Tests

### Using the Test Runner Script

The recommended way to run deployment validation tests is using the comprehensive test runner:

```bash
# Run all deployment validation tests
python scripts/run_deployment_validation.py

# Run tests for specific environment
python scripts/run_deployment_validation.py --environment production

# Run specific test suite
python scripts/run_deployment_validation.py --suite validation
python scripts/run_deployment_validation.py --suite production
python scripts/run_deployment_validation.py --suite integration

# Enable verbose output
python scripts/run_deployment_validation.py --verbose
```

### Using Pytest Directly

You can also run individual test files directly:

```bash
# Basic deployment validation
pytest tests/deployment/test_deployment_validation.py -v

# Production-specific tests
pytest tests/deployment/test_production_deployment.py -v

# Integration tests
pytest tests/deployment/test_deployment_integration.py -v

# Run all deployment tests
pytest tests/deployment/ -v
```

### Using the Validation Script

The deployment validation is integrated into the main validation script:

```bash
# Run complete deployment validation
./scripts/validate-deployment.sh development
./scripts/validate-deployment.sh production
```

## Environment Configuration

### Environment Variables

The tests use the following environment variables:

- `ENVIRONMENT` - Deployment environment (development/production/testing)
- `API_URL` - API service URL (default: http://localhost:8000)
- `FRONTEND_URL` - Frontend URL (default: http://localhost:3000)
- `WS_URL` - WebSocket URL (default: ws://localhost:8000/ws)
- `DB_HOST` - Database host (default: localhost)
- `DB_PORT` - Database port (default: 5432)
- `POSTGRES_DB` - Database name
- `POSTGRES_USER` - Database user
- `POSTGRES_PASSWORD` - Database password
- `REDIS_URL` - Redis connection URL

### Test Markers

The tests use pytest markers for categorization:

- `@pytest.mark.deployment` - General deployment tests
- `@pytest.mark.production` - Production-only tests
- `@pytest.mark.development` - Development-only tests
- `@pytest.mark.slow` - Slow-running tests

## Prerequisites

### System Requirements

- Docker and Docker Compose installed
- Python 3.8+ with required dependencies
- Services deployed and running

### Required Dependencies

Install test dependencies:

```bash
pip install -r requirements_test.txt
```

Key dependencies for deployment tests:
- `requests` - HTTP client for API testing
- `websocket-client` - WebSocket connectivity testing
- `psycopg2-binary` - PostgreSQL database testing
- `redis` - Redis connectivity testing
- `torch` - PyTorch model creation for testing
- `onnx` - ONNX model format testing
- `tensorflow` - TensorFlow model format testing

## Test Reports

### JSON Reports

The test runner generates detailed JSON reports:

- `test_results/deployment_validation_report.json` - Basic validation results
- `test_results/production_deployment_report.json` - Production test results
- `test_results/deployment_integration_report.json` - Integration test results
- `test_results/deployment_validation_summary.json` - Overall summary

### Console Output

Tests provide detailed console output with:
- ✅ Success indicators
- ❌ Failure indicators  
- ⚠️ Warning indicators
- 📊 Summary statistics
- 🔍 Diagnostic information

## Troubleshooting

### Common Issues

1. **Services Not Ready**
   - Ensure all Docker services are running: `docker-compose ps`
   - Wait for health checks to pass: `docker-compose logs`
   - Check service connectivity manually

2. **Database Connection Failures**
   - Verify database is running: `docker-compose exec db pg_isready`
   - Check connection parameters in environment variables
   - Ensure database initialization completed

3. **Redis Connection Failures**
   - Verify Redis is running: `docker-compose exec redis redis-cli ping`
   - Check Redis URL configuration

4. **Model Upload Tests Failing**
   - Ensure upload directory exists and is writable
   - Check file permissions
   - Verify API authentication if required

5. **WebSocket Tests Failing**
   - Check if WebSocket endpoint requires authentication
   - Verify WebSocket URL configuration
   - Check firewall/proxy settings

### Debug Mode

Enable verbose output for detailed debugging:

```bash
python scripts/run_deployment_validation.py --verbose
```

### Manual Verification

You can manually verify deployment health:

```bash
# Check service status
docker-compose ps

# Check API health
curl http://localhost:8000/health

# Check frontend
curl http://localhost:3000

# Check database
docker-compose exec db pg_isready -U postgres

# Check Redis
docker-compose exec redis redis-cli ping
```

## Integration with CI/CD

These tests are designed to be integrated into CI/CD pipelines:

```yaml
# Example GitHub Actions step
- name: Validate Deployment
  run: |
    python scripts/run_deployment_validation.py --environment production
  env:
    ENVIRONMENT: production
    API_URL: https://api.example.com
    FRONTEND_URL: https://app.example.com
```

## Contributing

When adding new deployment validation tests:

1. Follow the existing test structure and naming conventions
2. Use appropriate pytest markers
3. Include proper error handling and cleanup
4. Add documentation for new test categories
5. Update this README with new test descriptions
6. Ensure tests work in both development and production environments

## Requirements Validation

These tests specifically validate:

- **Requirement 6.1**: Model format support (PyTorch, TensorFlow, ONNX)
- **Requirement 6.2**: Model type identification and optimization technique selection
- **Deployment Readiness**: All services healthy and functional
- **Security**: Proper authentication and security headers
- **Performance**: Acceptable response times and resource usage
- **Scalability**: Horizontal scaling and load balancing
- **Reliability**: Error handling and recovery mechanisms
# Robotics Model Optimization Platform API

REST API for the AI agentic platform that automatically optimizes robotics models like OpenVLA.

## Features

- **Model Management**: Upload, list, and manage robotics models
- **Automatic Optimization**: AI agents analyze and optimize models using various techniques
- **Real-time Monitoring**: Track optimization progress and system metrics
- **Authentication & Authorization**: Secure access with JWT tokens
- **Comprehensive Documentation**: OpenAPI/Swagger documentation
- **Integration Testing**: Full test suite for API endpoints

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Start the API Server

```bash
# Using the startup script
python src/api/run.py

# Or using uvicorn directly
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

### 3. Access Documentation

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI JSON**: http://localhost:8000/openapi.json

### 4. Run Demo

```bash
python examples/api_demo.py
```

## Authentication

All endpoints (except `/health` and `/auth/login`) require authentication using Bearer tokens.

### Login

```bash
curl -X POST "http://localhost:8000/auth/login" \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "admin"}'
```

### Use Token

```bash
curl -X GET "http://localhost:8000/models" \
  -H "Authorization: Bearer YOUR_TOKEN_HERE"
```

## API Endpoints

### Health & System

- `GET /health` - Basic health check
- `GET /monitoring/system` - Detailed system status and metrics
- `GET /monitoring/sessions/metrics` - Active session metrics

### Authentication

- `POST /auth/login` - User login

### Model Management

- `POST /models/upload` - Upload a model file
- `GET /models` - List uploaded models
- `DELETE /models/{model_id}` - Delete a model

### Optimization

- `POST /optimize` - Start optimization session
- `GET /sessions` - List optimization sessions
- `GET /sessions/{session_id}/status` - Get session status
- `POST /sessions/{session_id}/cancel` - Cancel session
- `POST /sessions/{session_id}/rollback` - Rollback session

### Results

- `GET /sessions/{session_id}/results` - Get optimization results

## Usage Examples

### Upload a Model

```python
import requests

# Login
login_response = requests.post("http://localhost:8000/auth/login", 
                              json={"username": "admin", "password": "admin"})
token = login_response.json()["access_token"]

headers = {"Authorization": f"Bearer {token}"}

# Upload model
with open("my_model.pt", "rb") as f:
    files = {"file": ("my_model.pt", f, "application/octet-stream")}
    data = {"name": "My OpenVLA Model", "description": "Fine-tuned for manipulation"}
    
    response = requests.post("http://localhost:8000/models/upload", 
                           headers=headers, files=files, data=data)

model_id = response.json()["model_id"]
```

### Start Optimization

```python
optimization_request = {
    "model_id": model_id,
    "criteria_name": "balanced_optimization",
    "target_accuracy_threshold": 0.95,
    "max_size_reduction_percent": 50.0,
    "optimization_techniques": ["quantization", "pruning"]
}

response = requests.post("http://localhost:8000/optimize", 
                        headers=headers, json=optimization_request)

session_id = response.json()["session_id"]
```

### Monitor Progress

```python
import time

while True:
    response = requests.get(f"http://localhost:8000/sessions/{session_id}/status", 
                           headers=headers)
    status = response.json()
    
    print(f"Status: {status['status']} | Progress: {status['progress_percentage']}%")
    
    if status['status'] in ['completed', 'failed', 'cancelled']:
        break
    
    time.sleep(5)
```

### Get Results

```python
response = requests.get(f"http://localhost:8000/sessions/{session_id}/results", 
                       headers=headers)
results = response.json()

print(f"Optimization Summary: {results['optimization_summary']}")
print(f"Performance Improvements: {results['performance_improvements']}")
```

## Configuration

### Environment Variables

- `API_HOST`: Server host (default: 0.0.0.0)
- `API_PORT`: Server port (default: 8000)
- `API_RELOAD`: Enable auto-reload (default: true)
- `API_LOG_LEVEL`: Logging level (default: info)

### File Uploads

- **Max file size**: 500MB
- **Supported formats**: .pt, .pth, .onnx, .pb, .h5, .safetensors
- **Upload directory**: `uploads/`

## Testing

### Run Tests

```bash
# Run all API tests
pytest tests/test_api.py -v

# Run with coverage
pytest tests/test_api.py --cov=src.api --cov-report=html
```

### Test Categories

- **Authentication Tests**: Login, token validation, permissions
- **Model Management Tests**: Upload, list, delete models
- **Optimization Tests**: Start, monitor, cancel optimizations
- **Error Handling Tests**: Invalid requests, unauthorized access
- **Integration Tests**: End-to-end workflows

## Security Considerations

### Production Deployment

1. **Change Default Credentials**: Update default admin password
2. **Use Strong Secret Key**: Replace JWT secret key
3. **Enable HTTPS**: Use SSL/TLS certificates
4. **Configure CORS**: Restrict allowed origins
5. **Rate Limiting**: Implement request rate limiting
6. **Input Validation**: Validate all user inputs
7. **File Upload Security**: Scan uploaded files for malware

### Authentication

- JWT tokens expire after 60 minutes
- Tokens are revoked on logout
- Role-based access control (admin vs user)
- Password hashing using SHA-256 (upgrade to bcrypt in production)

## Monitoring & Logging

### System Metrics

- CPU and memory usage
- Disk space utilization
- GPU utilization (if available)
- Active optimization sessions
- Model count and storage

### Logging

- Request/response logging
- Error tracking and debugging
- Optimization progress logging
- Security event logging

## Error Handling

### HTTP Status Codes

- `200`: Success
- `400`: Bad Request (validation errors)
- `401`: Unauthorized (invalid token)
- `403`: Forbidden (insufficient permissions)
- `404`: Not Found (resource doesn't exist)
- `413`: Payload Too Large (file too big)
- `422`: Unprocessable Entity (validation errors)
- `500`: Internal Server Error
- `503`: Service Unavailable

### Error Response Format

```json
{
  "error": "validation_error",
  "message": "Invalid request parameters",
  "details": "model_id is required"
}
```

## Development

### Project Structure

```
src/api/
├── main.py              # FastAPI application
├── models.py            # Pydantic models
├── auth.py              # Authentication logic
├── dependencies.py      # FastAPI dependencies
├── monitoring.py        # Monitoring endpoints
├── openapi_config.py    # OpenAPI configuration
├── run.py              # Startup script
└── README.md           # This file
```

### Adding New Endpoints

1. Define Pydantic models in `models.py`
2. Add endpoint logic to appropriate router
3. Update OpenAPI configuration
4. Write tests in `tests/test_api.py`
5. Update documentation

### Code Style

- Follow PEP 8 style guidelines
- Use type hints for all functions
- Document all endpoints with docstrings
- Include request/response examples

## Troubleshooting

### Common Issues

1. **Connection Refused**: Check if server is running on correct port
2. **401 Unauthorized**: Verify token is valid and not expired
3. **File Upload Fails**: Check file size and format restrictions
4. **Optimization Hangs**: Check system resources and agent status

### Debug Mode

```bash
# Enable debug logging
API_LOG_LEVEL=debug python src/api/run.py

# Check system status
curl http://localhost:8000/monitoring/system
```

### Health Checks

```bash
# Basic health check
curl http://localhost:8000/health

# Detailed health check (requires auth)
curl -H "Authorization: Bearer TOKEN" http://localhost:8000/monitoring/health/detailed
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

MIT License - see LICENSE file for details.
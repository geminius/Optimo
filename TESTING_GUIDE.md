# Testing Guide - API Server & Frontend with Real Model

This guide helps you test the complete platform with a real robotics model.

## Quick Start

### 1. Start the API Server

```bash
# Option A: Start API server only
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

# Option B: Start complete platform (includes API)
python -m src.main
```

The API server will be available at: http://localhost:8000
- API Documentation: http://localhost:8000/docs
- Health Check: http://localhost:8000/health

### 2. Run the Test Script

```bash
# Run the automated test script
python test_real_model.py
```

This script will:
1. ✅ Create a sample robotics VLA model
2. ✅ Check if API server is running
3. ✅ Authenticate with the API
4. ✅ Upload the model
5. ✅ Start an optimization session
6. ✅ Monitor progress in real-time
7. ✅ Display results
8. ✅ Provide frontend testing instructions

### 3. Start the Frontend (Optional)

```bash
cd frontend
npm install
npm start
```

The frontend will be available at: http://localhost:3000

## Manual API Testing

### Authentication

```bash
# Login
curl -X POST "http://localhost:8000/auth/login" \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "admin"}'

# Response includes access_token
```

### Upload Model

```bash
# Upload a model file
curl -X POST "http://localhost:8000/models/upload" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -F "file=@test_robotics_model.pt" \
  -F "name=Test Model" \
  -F "description=Test robotics model" \
  -F "tags=test,robotics"

# Response includes model_id
```

### Start Optimization

```bash
# Start optimization session
curl -X POST "http://localhost:8000/optimize" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "YOUR_MODEL_ID",
    "criteria_name": "edge_robotics_deployment",
    "target_accuracy_threshold": 0.95,
    "optimization_techniques": ["quantization", "pruning"]
  }'

# Response includes session_id
```

### Monitor Progress

```bash
# Check session status
curl "http://localhost:8000/sessions/YOUR_SESSION_ID/status" \
  -H "Authorization: Bearer YOUR_TOKEN"

# Get results (when completed)
curl "http://localhost:8000/sessions/YOUR_SESSION_ID/results" \
  -H "Authorization: Bearer YOUR_TOKEN"
```

### List Resources

```bash
# List all models
curl "http://localhost:8000/models" \
  -H "Authorization: Bearer YOUR_TOKEN"

# List all sessions
curl "http://localhost:8000/sessions" \
  -H "Authorization: Bearer YOUR_TOKEN"

# Get dashboard statistics
curl "http://localhost:8000/api/v1/dashboard/stats" \
  -H "Authorization: Bearer YOUR_TOKEN"
```

## Frontend Testing

### 1. Login
- Navigate to http://localhost:3000
- Username: `admin`
- Password: `admin`

### 2. Test Features

#### Upload Model
1. Click "Upload Model" button
2. Select a model file (.pt, .pth, .onnx, etc.)
3. Fill in model details
4. Click "Upload"
5. Verify model appears in the list

#### Start Optimization
1. Select a model from the list
2. Click "Optimize"
3. Configure optimization criteria:
   - Target accuracy threshold
   - Optimization techniques
   - Deployment target
4. Click "Start Optimization"
5. Watch real-time progress updates

#### Monitor Progress
1. View active sessions in the dashboard
2. Click on a session to see details
3. Watch progress bar and status updates
4. Check WebSocket connection in browser console (F12)

#### View Results
1. Wait for optimization to complete
2. Click on completed session
3. Review metrics:
   - Model size reduction
   - Performance improvements
   - Validation status
4. Download optimized model (if available)

### 3. WebSocket Testing

Open browser console (F12) and look for:
```
Socket.IO connected
Subscribed to session: YOUR_SESSION_ID
Progress update: { status: "running", progress: 45.0, ... }
```

## Testing with Your Own Model

### Supported Model Formats
- PyTorch: `.pt`, `.pth`
- ONNX: `.onnx`
- TensorFlow: `.pb`, `.h5`
- SafeTensors: `.safetensors`

### Using Your Model

```python
# Option 1: Use the test script with your model
python test_real_model.py

# Then manually upload your model through the API or frontend

# Option 2: Programmatic approach
from pathlib import Path
import requests

# Upload your model
with open("path/to/your/model.pt", "rb") as f:
    files = {"file": ("model.pt", f, "application/octet-stream")}
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.post(
        "http://localhost:8000/models/upload",
        files=files,
        headers=headers
    )
    model_id = response.json()["model_id"]

# Start optimization
response = requests.post(
    "http://localhost:8000/optimize",
    json={
        "model_id": model_id,
        "criteria_name": "edge_robotics_deployment",
        "target_accuracy_threshold": 0.95,
        "optimization_techniques": ["quantization", "pruning"]
    },
    headers=headers
)
session_id = response.json()["session_id"]
```

## Troubleshooting

### API Server Not Starting

```bash
# Check if port 8000 is already in use
lsof -i :8000

# Kill existing process if needed
kill -9 PID

# Check logs
tail -f logs/api.log
```

### Frontend Not Connecting

1. Check API server is running: http://localhost:8000/health
2. Check CORS configuration in `.env` or environment variables
3. Verify WebSocket connection in browser console
4. Check frontend environment variables (`.env.local`)

### Optimization Fails

1. Check model format is supported
2. Verify model file is not corrupted
3. Check available system resources (RAM, GPU)
4. Review error message in session status
5. Check logs: `tail -f logs/optimization.log`

### WebSocket Not Working

1. Verify API server supports WebSocket (Socket.IO)
2. Check CORS allows WebSocket upgrade
3. Verify frontend Socket.IO client version matches server
4. Check browser console for connection errors

## Performance Testing

### Load Testing

```bash
# Install load testing tool
pip install locust

# Run load test
locust -f tests/load/locustfile.py --host=http://localhost:8000
```

### Stress Testing

```bash
# Run stress tests
pytest tests/stress/ -v

# Test concurrent sessions
pytest tests/stress/test_concurrent_sessions.py -v
```

## Integration Testing

```bash
# Run complete integration test suite
pytest tests/integration/ -v

# Run specific workflow test
pytest tests/integration/test_real_optimization_workflow.py -v -s

# Run with coverage
pytest tests/integration/ --cov=src --cov-report=html
```

## Monitoring

### Check System Health

```bash
# API health check
curl http://localhost:8000/health

# Detailed monitoring
curl http://localhost:8000/api/v1/monitoring/health

# System metrics
curl http://localhost:8000/metrics
```

### View Logs

```bash
# API logs
tail -f logs/api.log

# Optimization logs
tail -f logs/optimization.log

# All logs
tail -f logs/*.log
```

## Next Steps

1. ✅ Test with sample model (automated script)
2. ✅ Test with your own robotics model
3. ✅ Explore frontend features
4. ✅ Test WebSocket real-time updates
5. ✅ Review optimization results
6. ✅ Test error handling (invalid models, etc.)
7. ✅ Load test with multiple concurrent sessions
8. ✅ Deploy to production environment

## Additional Resources

- **API Documentation**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **Main README**: [README.md](README.md)
- **Deployment Guide**: [deploy/DEPLOYMENT.md](deploy/DEPLOYMENT.md)
- **Architecture**: [docs/STRUCTURE.md](docs/STRUCTURE.md)

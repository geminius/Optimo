# Environment Variables Reference

## Overview

This document lists all environment variables used by the Robotics Model Optimization Platform.

## Backend Environment Variables

Create a `.env` file in the project root directory.

### Authentication (Required)

```bash
# JWT Secret Key - CHANGE IN PRODUCTION!
# Generate with: python -c "import secrets; print(secrets.token_urlsafe(32))"
JWT_SECRET_KEY=your-secret-key-here-min-32-characters-long

# JWT Algorithm (default: HS256)
JWT_ALGORITHM=HS256

# Token expiration in minutes (default: 60)
JWT_ACCESS_TOKEN_EXPIRE_MINUTES=60
```

### API Configuration

```bash
# API server host (default: 0.0.0.0)
API_HOST=0.0.0.0

# API server port (default: 8000)
API_PORT=8000

# CORS allowed origins (comma-separated)
CORS_ORIGINS=http://localhost:3000,http://localhost:3001

# Enable API documentation (default: true)
ENABLE_API_DOCS=true
```

### Database Configuration

```bash
# Database URL (SQLite default)
DATABASE_URL=sqlite:///./data/optimization_history.db

# For PostgreSQL:
# DATABASE_URL=postgresql://user:password@localhost/dbname

# For MySQL:
# DATABASE_URL=mysql://user:password@localhost/dbname
```

### Logging Configuration

```bash
# Log level: DEBUG, INFO, WARNING, ERROR, CRITICAL (default: INFO)
LOG_LEVEL=INFO

# Log directory (default: logs)
LOG_DIR=logs

# Enable JSON logging (default: false)
JSON_LOGGING=false

# Enable console output (default: true)
CONSOLE_LOGGING=true
```

### Storage Configuration

```bash
# Model storage directory (default: models)
MODEL_STORAGE_DIR=models

# Upload directory (default: uploads)
UPLOAD_DIR=uploads

# Maximum upload size in MB (default: 500)
MAX_UPLOAD_SIZE_MB=500
```

### Optimization Configuration

```bash
# Maximum concurrent optimization sessions (default: 5)
MAX_CONCURRENT_SESSIONS=5

# Session timeout in minutes (default: 240)
SESSION_TIMEOUT_MINUTES=240

# Enable auto-rollback on failure (default: true)
AUTO_ROLLBACK_ON_FAILURE=true
```

### Monitoring Configuration

```bash
# Monitoring interval in seconds (default: 30)
MONITORING_INTERVAL_SECONDS=30

# Health check interval in seconds (default: 60)
HEALTH_CHECK_INTERVAL_SECONDS=60

# CPU alert threshold percentage (default: 80)
CPU_ALERT_THRESHOLD=80

# Memory alert threshold percentage (default: 85)
MEMORY_ALERT_THRESHOLD=85

# Disk alert threshold percentage (default: 90)
DISK_ALERT_THRESHOLD=90
```

### WebSocket Configuration

```bash
# WebSocket ping interval in seconds (default: 25)
WS_PING_INTERVAL=25

# WebSocket ping timeout in seconds (default: 60)
WS_PING_TIMEOUT=60

# Enable WebSocket compression (default: true)
WS_COMPRESSION=true
```

## Frontend Environment Variables

Create a `frontend/.env` file in the frontend directory.

### Required Variables

```bash
# Backend API URL (required)
REACT_APP_API_URL=http://localhost:8000

# WebSocket server URL (required)
REACT_APP_WS_URL=http://localhost:8000
```

### Optional Variables

```bash
# Enable debug mode (default: false)
REACT_APP_DEBUG=false

# API request timeout in milliseconds (default: 30000)
REACT_APP_API_TIMEOUT=30000

# Enable service worker (default: false)
REACT_APP_ENABLE_SERVICE_WORKER=false
```

## Production Configuration

### Backend Production `.env`

```bash
# Authentication - Use strong secret key!
JWT_SECRET_KEY=<generate-with-openssl-rand-base64-32>
JWT_ALGORITHM=HS256
JWT_ACCESS_TOKEN_EXPIRE_MINUTES=30

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
CORS_ORIGINS=https://yourplatform.com,https://www.yourplatform.com

# Database - Use production database
DATABASE_URL=postgresql://user:password@db-host:5432/production_db

# Logging
LOG_LEVEL=WARNING
LOG_DIR=/var/log/robotics-platform
JSON_LOGGING=true
CONSOLE_LOGGING=false

# Storage
MODEL_STORAGE_DIR=/data/models
UPLOAD_DIR=/data/uploads
MAX_UPLOAD_SIZE_MB=1000

# Optimization
MAX_CONCURRENT_SESSIONS=10
SESSION_TIMEOUT_MINUTES=480
AUTO_ROLLBACK_ON_FAILURE=true

# Monitoring
MONITORING_INTERVAL_SECONDS=60
HEALTH_CHECK_INTERVAL_SECONDS=120
CPU_ALERT_THRESHOLD=70
MEMORY_ALERT_THRESHOLD=80
DISK_ALERT_THRESHOLD=85
```

### Frontend Production `.env`

```bash
# Production API URLs
REACT_APP_API_URL=https://api.yourplatform.com
REACT_APP_WS_URL=https://api.yourplatform.com

# Production settings
REACT_APP_DEBUG=false
REACT_APP_API_TIMEOUT=60000
REACT_APP_ENABLE_SERVICE_WORKER=true
```

## Docker Environment Variables

When using Docker, pass environment variables via:

### docker-compose.yml

```yaml
services:
  backend:
    environment:
      - JWT_SECRET_KEY=${JWT_SECRET_KEY}
      - DATABASE_URL=${DATABASE_URL}
      - LOG_LEVEL=${LOG_LEVEL}
    env_file:
      - .env
  
  frontend:
    environment:
      - REACT_APP_API_URL=${REACT_APP_API_URL}
      - REACT_APP_WS_URL=${REACT_APP_WS_URL}
```

### Docker Run

```bash
docker run -d \
  -e JWT_SECRET_KEY="your-secret-key" \
  -e DATABASE_URL="postgresql://..." \
  -e LOG_LEVEL="INFO" \
  -p 8000:8000 \
  robotics-platform-backend
```

## Kubernetes Environment Variables

### ConfigMap

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: platform-config
data:
  API_HOST: "0.0.0.0"
  API_PORT: "8000"
  LOG_LEVEL: "INFO"
  MAX_CONCURRENT_SESSIONS: "10"
```

### Secret

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: platform-secrets
type: Opaque
stringData:
  JWT_SECRET_KEY: "your-secret-key-here"
  DATABASE_URL: "postgresql://user:password@host/db"
```

### Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: platform-backend
spec:
  template:
    spec:
      containers:
      - name: backend
        envFrom:
        - configMapRef:
            name: platform-config
        - secretRef:
            name: platform-secrets
```

## Environment Variable Validation

The platform validates required environment variables on startup:

### Backend Validation

```python
# Required variables
required_vars = [
    'JWT_SECRET_KEY',
]

# Validation on startup
for var in required_vars:
    if not os.getenv(var):
        raise ValueError(f"Required environment variable {var} is not set")
```

### Frontend Validation

```typescript
// Required variables
const requiredVars = [
  'REACT_APP_API_URL',
  'REACT_APP_WS_URL',
];

// Validation on build
requiredVars.forEach(varName => {
  if (!process.env[varName]) {
    throw new Error(`Required environment variable ${varName} is not set`);
  }
});
```

## Security Best Practices

### DO

✅ Use strong, randomly generated secret keys
✅ Store secrets in environment variables, not code
✅ Use different secrets for development and production
✅ Rotate secrets regularly
✅ Use `.env` files for local development
✅ Use secret management systems in production (AWS Secrets Manager, HashiCorp Vault, etc.)
✅ Restrict file permissions on `.env` files (chmod 600)

### DON'T

❌ Commit `.env` files to version control
❌ Use default or weak secret keys
❌ Share secrets between environments
❌ Log secret values
❌ Hardcode secrets in code
❌ Use the same secrets across multiple applications

## Troubleshooting

### Variables Not Loading

**Problem**: Environment variables not being read

**Solutions**:
1. Verify `.env` file exists in correct location
2. Restart application after changing `.env`
3. Check file permissions (should be readable)
4. Verify variable names are correct (case-sensitive)
5. For frontend, ensure variables start with `REACT_APP_`

### Invalid Secret Key

**Problem**: JWT authentication fails with "Invalid secret key"

**Solutions**:
1. Generate new secret key: `openssl rand -base64 32`
2. Ensure secret key is at least 32 characters
3. Verify no extra whitespace in secret key
4. Check secret key is same on all backend instances

### CORS Errors

**Problem**: Frontend can't connect to backend

**Solutions**:
1. Add frontend URL to `CORS_ORIGINS`
2. Ensure URLs match exactly (including protocol and port)
3. Restart backend after changing CORS settings
4. Check for typos in URLs

## Example .env Files

### Backend `.env.example`

```bash
# Copy this file to .env and update values

# Authentication (REQUIRED - change in production!)
JWT_SECRET_KEY=change-this-to-a-secure-random-string-min-32-chars
JWT_ALGORITHM=HS256
JWT_ACCESS_TOKEN_EXPIRE_MINUTES=60

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
CORS_ORIGINS=http://localhost:3000

# Database
DATABASE_URL=sqlite:///./data/optimization_history.db

# Logging
LOG_LEVEL=INFO
LOG_DIR=logs

# Storage
MODEL_STORAGE_DIR=models
UPLOAD_DIR=uploads
MAX_UPLOAD_SIZE_MB=500

# Optimization
MAX_CONCURRENT_SESSIONS=5
SESSION_TIMEOUT_MINUTES=240
AUTO_ROLLBACK_ON_FAILURE=true
```

### Frontend `.env.example`

```bash
# Copy this file to .env and update values

# Backend API URL (REQUIRED)
REACT_APP_API_URL=http://localhost:8000

# WebSocket URL (REQUIRED)
REACT_APP_WS_URL=http://localhost:8000

# Optional settings
REACT_APP_DEBUG=false
REACT_APP_API_TIMEOUT=30000
```

## Additional Resources

- [Authentication Guide](AUTHENTICATION.md)
- [Deployment Guide](../DEPLOYMENT.md)
- [Configuration Documentation](../config/README.md)
- [Twelve-Factor App - Config](https://12factor.net/config)

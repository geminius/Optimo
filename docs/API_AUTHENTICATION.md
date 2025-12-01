# API Authentication Quick Reference

## Overview

All API endpoints require JWT authentication except:
- `POST /auth/login` - Login endpoint
- `GET /health` - Health check endpoint

## Quick Start

### 1. Login

```bash
curl -X POST "http://localhost:8000/auth/login" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=admin&password=admin123"
```

**Response**:
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 3600,
  "user": {
    "id": "550e8400-e29b-41d4-a716-446655440000",
    "username": "admin",
    "role": "admin"
  }
}
```

### 2. Use Token in Requests

Include the token in the `Authorization` header with `Bearer` prefix:

```bash
curl -X GET "http://localhost:8000/models" \
  -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
```

## Authentication Endpoints

### POST /auth/login

Authenticate user and receive JWT token.

**Request**:
- **Content-Type**: `application/x-www-form-urlencoded`
- **Body**:
  - `username` (string, required): Username
  - `password` (string, required): Password

**Response** (200 OK):
```json
{
  "access_token": "string",
  "token_type": "bearer",
  "expires_in": 3600,
  "user": {
    "id": "string",
    "username": "string",
    "role": "string",
    "email": "string"
  }
}
```

**Errors**:
- `401 Unauthorized`: Invalid credentials
- `422 Unprocessable Entity`: Missing or invalid parameters

### GET /auth/me

Get current authenticated user information.

**Request**:
- **Headers**: `Authorization: Bearer <token>`

**Response** (200 OK):
```json
{
  "id": "string",
  "username": "string",
  "role": "string",
  "email": "string"
}
```

**Errors**:
- `401 Unauthorized`: Invalid or expired token

## Protected Endpoints

All other endpoints require authentication. Include the token in the `Authorization` header:

```
Authorization: Bearer <your-jwt-token>
```

### Example: Upload Model

```bash
TOKEN="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."

curl -X POST "http://localhost:8000/models/upload" \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@model.pth" \
  -F "name=MyModel"
```

### Example: Start Optimization

```bash
TOKEN="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."

curl -X POST "http://localhost:8000/optimize" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "model-uuid",
    "criteria_name": "robotics_optimization",
    "target_accuracy_threshold": 0.95
  }'
```

### Example: Get Optimization Status

```bash
TOKEN="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."

curl -X GET "http://localhost:8000/sessions/{session_id}/status" \
  -H "Authorization: Bearer $TOKEN"
```

## Common Response Codes

### Authentication Errors

- **401 Unauthorized**: Token is missing, invalid, or expired
  - **Solution**: Login again to get a new token
  
- **403 Forbidden**: User doesn't have permission for this resource
  - **Solution**: Check user role and endpoint requirements

### Success Codes

- **200 OK**: Request successful
- **201 Created**: Resource created successfully
- **204 No Content**: Request successful, no content to return

## Token Information

### Token Expiration

- **Default**: 60 minutes
- **Configurable**: Set `JWT_ACCESS_TOKEN_EXPIRE_MINUTES` in backend `.env`
- **Behavior**: Expired tokens return 401 Unauthorized

### Token Format

JWT tokens consist of three parts separated by dots:
```
header.payload.signature
```

Example:
```
eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJhZG1pbiIsInVzZXJfaWQiOiI1NTBlODQwMC1lMjliLTQxZDQtYTcxNi00NDY2NTU0NDAwMDAiLCJyb2xlIjoiYWRtaW4iLCJleHAiOjE2OTk1NjQ4MDB9.signature
```

### Token Payload

Decoded payload contains:
```json
{
  "sub": "admin",
  "user_id": "550e8400-e29b-41d4-a716-446655440000",
  "role": "admin",
  "exp": 1699564800
}
```

## WebSocket Authentication

WebSocket connections also require authentication.

### JavaScript/TypeScript

```typescript
import io from 'socket.io-client';

const token = 'your-jwt-token';

const socket = io('http://localhost:8000', {
  auth: {
    token: token
  },
  transports: ['websocket']
});

socket.on('connect', () => {
  console.log('Connected with authentication');
});

socket.on('connect_error', (error) => {
  console.error('Connection failed:', error.message);
});
```

### Python

```python
import socketio

token = 'your-jwt-token'

sio = socketio.Client()

@sio.event
def connect():
    print('Connected with authentication')

@sio.event
def connect_error(data):
    print('Connection failed:', data)

sio.connect('http://localhost:8000', auth={'token': token})
```

## Testing with Postman

### 1. Create Login Request

- **Method**: POST
- **URL**: `http://localhost:8000/auth/login`
- **Headers**: `Content-Type: application/x-www-form-urlencoded`
- **Body** (x-www-form-urlencoded):
  - `username`: admin
  - `password`: admin123

### 2. Save Token

After login, copy the `access_token` from the response.

### 3. Use Token in Requests

For all other requests:
- **Headers**: Add `Authorization: Bearer <paste-token-here>`

### 4. Automate with Postman Variables

1. In login request, add to **Tests** tab:
```javascript
pm.test("Login successful", function () {
    var jsonData = pm.response.json();
    pm.environment.set("auth_token", jsonData.access_token);
});
```

2. In other requests, use:
```
Authorization: Bearer {{auth_token}}
```

## Testing with Python Requests

```python
import requests

# Login
login_response = requests.post(
    'http://localhost:8000/auth/login',
    data={
        'username': 'admin',
        'password': 'admin123'
    }
)

token = login_response.json()['access_token']

# Use token in requests
headers = {
    'Authorization': f'Bearer {token}'
}

# Upload model
with open('model.pth', 'rb') as f:
    files = {'file': f}
    data = {'name': 'MyModel'}
    response = requests.post(
        'http://localhost:8000/models/upload',
        headers=headers,
        files=files,
        data=data
    )

# Start optimization
response = requests.post(
    'http://localhost:8000/optimize',
    headers=headers,
    json={
        'model_id': 'model-uuid',
        'criteria_name': 'robotics_optimization',
        'target_accuracy_threshold': 0.95
    }
)
```

## Security Best Practices

### DO

✅ Use HTTPS in production
✅ Store tokens securely
✅ Set reasonable expiration times
✅ Validate tokens on every request
✅ Use environment variables for secrets
✅ Implement rate limiting on login endpoint

### DON'T

❌ Share tokens between users
❌ Log tokens in plain text
❌ Use weak secret keys
❌ Transmit tokens over HTTP in production
❌ Store tokens in version control
❌ Use very long expiration times

## Troubleshooting

### "401 Unauthorized"

**Causes**:
- Token missing from request
- Token expired
- Invalid token format
- Wrong secret key

**Solutions**:
1. Verify token is included: `Authorization: Bearer <token>`
2. Check token hasn't expired (60 minutes default)
3. Login again to get new token
4. Verify `Bearer` prefix is included

### "403 Forbidden"

**Causes**:
- User doesn't have required permissions
- Endpoint requires admin role

**Solutions**:
1. Check user role in token payload
2. Use admin account if needed
3. Verify endpoint permissions

### "422 Unprocessable Entity"

**Causes**:
- Missing required parameters
- Invalid parameter format

**Solutions**:
1. Check request body format
2. Verify all required fields are included
3. Check parameter types match API spec

## Additional Resources

- [Full Authentication Guide](AUTHENTICATION.md)
- [API Documentation](http://localhost:8000/docs) (when server is running)
- [JWT.io](https://jwt.io/) - JWT token debugger
- [FastAPI Security](https://fastapi.tiangolo.com/tutorial/security/)

## Support

For issues or questions:
1. Check [AUTHENTICATION.md](AUTHENTICATION.md) for detailed troubleshooting
2. Review API documentation at `/docs`
3. Check backend logs for errors
4. Create an issue with detailed error information

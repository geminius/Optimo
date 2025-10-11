# API Documentation Guide

## Overview

The Robotics Model Optimization Platform API provides comprehensive OpenAPI documentation for all endpoints, WebSocket events, and authentication mechanisms. This guide helps you navigate and use the documentation effectively.

## Accessing Documentation

### Swagger UI (Interactive)
**URL**: http://localhost:8000/docs

**Features**:
- Interactive API testing
- Try endpoints directly from browser
- View request/response schemas
- See example payloads
- Test authentication

**Best For**:
- Testing endpoints
- Exploring the API
- Quick prototyping
- Learning the API

### ReDoc (Reference)
**URL**: http://localhost:8000/redoc

**Features**:
- Clean, readable layout
- Better for reading
- Printable format
- Search functionality
- Organized by tags

**Best For**:
- Reading documentation
- Reference material
- Printing documentation
- Sharing with team

### OpenAPI JSON
**URL**: http://localhost:8000/openapi.json

**Features**:
- Machine-readable specification
- OpenAPI 3.0 format
- Complete API definition

**Best For**:
- Code generation
- API testing tools
- Integration with tools
- Automated testing

## Documentation Structure

### Main Sections

1. **Overview**
   - Platform description
   - Key features
   - Workflow overview
   - Supported model types

2. **Authentication**
   - How to obtain tokens
   - Using tokens in requests
   - Token expiration handling
   - Role-based access control

3. **Endpoints by Category**
   - Health: System status
   - Authentication: Login/logout
   - Models: Upload and management
   - Dashboard: Statistics and metrics
   - Configuration: Optimization settings
   - Optimization: Start and manage
   - Sessions: Monitor progress
   - Results: Retrieve outcomes

4. **WebSocket Events**
   - Connection management
   - Event types and schemas
   - Subscription system
   - Code examples

5. **Error Handling**
   - Error response format
   - Common error codes
   - Troubleshooting guide

## Quick Start

### 1. Authentication

```bash
# Login to get token
curl -X POST http://localhost:8000/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "admin"}'

# Response includes access_token
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 3600
}
```

### 2. Using Authenticated Endpoints

```bash
# Use token in Authorization header
curl -X GET http://localhost:8000/dashboard/stats \
  -H "Authorization: Bearer YOUR_TOKEN_HERE"
```

### 3. WebSocket Connection

```javascript
import io from 'socket.io-client';

const socket = io('http://localhost:8000', {
  path: '/socket.io',
  auth: { token: 'YOUR_TOKEN_HERE' }
});

socket.on('connect', () => {
  console.log('Connected');
  socket.emit('subscribe_session', { session_id: 'session_id' });
});

socket.on('progress_update', (data) => {
  console.log('Progress:', data.progress_percentage);
});
```

## Key Endpoints

### Dashboard Statistics
```
GET /dashboard/stats
```
Returns aggregate metrics about the platform.

**Response**:
- Total models
- Active optimizations
- Completed optimizations
- Average size reduction
- Average speed improvement

### List Sessions
```
GET /optimization/sessions?status=running&limit=50
```
Returns paginated list of optimization sessions.

**Query Parameters**:
- `status`: Filter by status (running, completed, failed, cancelled)
- `model_id`: Filter by model
- `start_date`: Filter by creation date
- `end_date`: Filter by completion date
- `skip`: Pagination offset
- `limit`: Items per page (max 100)

### Get Configuration
```
GET /config/optimization-criteria
```
Returns current optimization criteria configuration.

### Update Configuration (Admin Only)
```
PUT /config/optimization-criteria
```
Updates optimization criteria configuration.

**Requires**: Administrator role

## WebSocket Events

### Session Events

**session_started**
- Emitted when optimization begins
- Includes session ID, model info, techniques

**progress_update**
- Emitted every 5-10 seconds during optimization
- Includes progress percentage, current step, time estimates

**session_completed**
- Emitted when optimization succeeds
- Includes results and metrics

**session_failed**
- Emitted when optimization fails
- Includes error message and type

### Subscription

```javascript
// Subscribe to session updates
socket.emit('subscribe_session', { session_id: 'session_abc123' });

// Unsubscribe
socket.emit('unsubscribe_session', { session_id: 'session_abc123' });
```

## Error Handling

### Common Error Codes

**400 Bad Request**
- Invalid query parameters
- Validation errors
- Malformed request

**401 Unauthorized**
- Missing authentication token
- Invalid token
- Expired token

**403 Forbidden**
- Insufficient permissions
- Admin-only endpoint

**404 Not Found**
- Resource doesn't exist
- Invalid session ID

**500 Internal Server Error**
- Server-side error
- Service unavailable

### Error Response Format

```json
{
  "error": "ErrorType",
  "message": "Human-readable message",
  "details": {
    "field": "parameter_name",
    "reason": "specific_issue"
  },
  "timestamp": "2024-01-01T12:00:00Z",
  "request_id": "req_abc123"
}
```

## Best Practices

### Authentication
1. Store tokens securely (httpOnly cookies preferred)
2. Check token expiration before requests
3. Handle 401 errors gracefully
4. Never expose tokens in logs or URLs

### API Usage
1. Use pagination for list endpoints
2. Filter results to reduce response size
3. Handle errors with user-friendly messages
4. Implement retry logic for transient failures

### WebSocket
1. Implement reconnection logic
2. Unsubscribe when done monitoring
3. Handle connection errors
4. Validate event data before using

### Performance
1. Cache dashboard statistics (30s TTL)
2. Use WebSocket for real-time updates (don't poll)
3. Limit concurrent requests
4. Use appropriate pagination limits

## Code Examples

### JavaScript/TypeScript

```typescript
// API Client with authentication
class APIClient {
  private baseURL = 'http://localhost:8000';
  private token: string;

  constructor(token: string) {
    this.token = token;
  }

  async getDashboardStats() {
    const response = await fetch(`${this.baseURL}/dashboard/stats`, {
      headers: {
        'Authorization': `Bearer ${this.token}`
      }
    });

    if (!response.ok) {
      throw new Error(`API error: ${response.status}`);
    }

    return await response.json();
  }

  async listSessions(filters: {
    status?: string;
    model_id?: string;
    skip?: number;
    limit?: number;
  }) {
    const params = new URLSearchParams();
    Object.entries(filters).forEach(([key, value]) => {
      if (value !== undefined) {
        params.append(key, String(value));
      }
    });

    const response = await fetch(
      `${this.baseURL}/optimization/sessions?${params}`,
      {
        headers: {
          'Authorization': `Bearer ${this.token}`
        }
      }
    );

    if (!response.ok) {
      throw new Error(`API error: ${response.status}`);
    }

    return await response.json();
  }
}
```

### Python

```python
import requests
from typing import Optional, Dict, Any

class APIClient:
    def __init__(self, base_url: str, token: str):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'Bearer {token}'
        })
    
    def get_dashboard_stats(self) -> Dict[str, Any]:
        response = self.session.get(f'{self.base_url}/dashboard/stats')
        response.raise_for_status()
        return response.json()
    
    def list_sessions(
        self,
        status: Optional[str] = None,
        model_id: Optional[str] = None,
        skip: int = 0,
        limit: int = 50
    ) -> Dict[str, Any]:
        params = {
            'skip': skip,
            'limit': limit
        }
        
        if status:
            params['status'] = status
        if model_id:
            params['model_id'] = model_id
        
        response = self.session.get(
            f'{self.base_url}/optimization/sessions',
            params=params
        )
        response.raise_for_status()
        return response.json()

# Usage
client = APIClient('http://localhost:8000', 'YOUR_TOKEN')
stats = client.get_dashboard_stats()
sessions = client.list_sessions(status='running')
```

## Troubleshooting

### Cannot Access Documentation

**Problem**: `/docs` returns 404

**Solution**:
1. Verify server is running: `curl http://localhost:8000/health`
2. Check server logs for errors
3. Ensure FastAPI app is configured correctly

### Authentication Fails

**Problem**: Getting 401 errors

**Solution**:
1. Verify token is valid
2. Check token hasn't expired (1 hour lifetime)
3. Ensure Authorization header format: `Bearer TOKEN`
4. Login again to get fresh token

### WebSocket Won't Connect

**Problem**: Cannot establish WebSocket connection

**Solution**:
1. Verify server supports WebSocket
2. Check firewall/proxy settings
3. Include authentication token
4. Use correct path: `/socket.io`

### No Events Received

**Problem**: Connected but not receiving events

**Solution**:
1. Emit `subscribe_session` after connecting
2. Verify session ID is correct
3. Check user has permission to view session
4. Ensure event listeners are registered

## Additional Resources

- **API Source Code**: `src/api/`
- **WebSocket Events**: `src/api/websocket_events.py`
- **Configuration**: `src/api/openapi_config.py`
- **Examples**: `examples/` directory

## Support

For issues or questions:
1. Check this documentation
2. Review error messages and request IDs
3. Check server logs
4. Consult troubleshooting guides
5. Contact platform administrators

## Version Information

- **API Version**: 1.0.0
- **OpenAPI Version**: 3.0.0
- **Documentation Last Updated**: 2024-01-11

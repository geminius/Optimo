# Authentication Guide

## Overview

The Robotics Model Optimization Platform uses JWT (JSON Web Token) based authentication to secure API endpoints and WebSocket connections. This guide covers authentication setup, configuration, and troubleshooting.

## Table of Contents

- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Configuration](#configuration)
- [API Authentication](#api-authentication)
- [Frontend Authentication](#frontend-authentication)
- [WebSocket Authentication](#websocket-authentication)
- [Security Best Practices](#security-best-practices)
- [Troubleshooting](#troubleshooting)

## Quick Start

### Default Credentials

For development and testing:
- **Username**: `admin`
- **Password**: `admin123`

⚠️ **Important**: Change these credentials in production!

### Environment Setup

1. **Backend Configuration** (root `.env` file):
```bash
# JWT Configuration
JWT_SECRET_KEY=your-secret-key-here-change-in-production
JWT_ALGORITHM=HS256
JWT_ACCESS_TOKEN_EXPIRE_MINUTES=60

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
```

2. **Frontend Configuration** (`frontend/.env` file):
```bash
# Backend API URL
REACT_APP_API_URL=http://localhost:8000

# WebSocket URL
REACT_APP_WS_URL=http://localhost:8000
```

3. **Start the servers**:
```bash
# Terminal 1: Start backend
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2: Start frontend
cd frontend
npm start
```

4. **Access the application**:
- Open browser to `http://localhost:3000`
- Login with default credentials
- You'll be redirected to the dashboard

## Architecture

### Authentication Flow

```
┌─────────┐         ┌──────────┐         ┌─────────┐
│ Browser │         │ Frontend │         │ Backend │
└────┬────┘         └────┬─────┘         └────┬────┘
     │                   │                     │
     │  1. Enter creds   │                     │
     ├──────────────────>│                     │
     │                   │  2. POST /auth/login│
     │                   ├────────────────────>│
     │                   │                     │
     │                   │  3. JWT token       │
     │                   │<────────────────────┤
     │                   │                     │
     │  4. Store token   │                     │
     │   in localStorage │                     │
     │<──────────────────┤                     │
     │                   │                     │
     │  5. API requests  │                     │
     │   with token      │  6. Authenticated   │
     ├──────────────────>├────────────────────>│
     │                   │     requests        │
     │                   │                     │
```

### Components

1. **Backend (`src/api/auth.py`)**
   - JWT token generation and validation
   - User authentication endpoint
   - Password hashing and verification
   - Token expiration management

2. **Frontend (`frontend/src/`)**
   - `services/auth.ts` - Authentication service
   - `contexts/AuthContext.tsx` - Global auth state
   - `components/auth/LoginPage.tsx` - Login UI
   - `components/auth/ProtectedRoute.tsx` - Route guards
   - `hooks/useAuth.ts` - Authentication hook

3. **API Interceptors (`frontend/src/services/api.ts`)**
   - Automatically add Authorization header
   - Handle 401/403 responses
   - Redirect to login on auth failure

## Configuration

### Backend JWT Settings

Configure in root `.env` file or environment variables:

```bash
# Secret key for signing tokens (REQUIRED - change in production!)
JWT_SECRET_KEY=your-very-secret-key-min-32-characters-long

# Algorithm for token signing (default: HS256)
JWT_ALGORITHM=HS256

# Token expiration time in minutes (default: 60)
JWT_ACCESS_TOKEN_EXPIRE_MINUTES=60

# Optional: Refresh token settings
JWT_REFRESH_TOKEN_EXPIRE_DAYS=7
```

### Frontend API Configuration

Configure in `frontend/.env`:

```bash
# Backend API base URL
REACT_APP_API_URL=http://localhost:8000

# WebSocket server URL (usually same as API URL)
REACT_APP_WS_URL=http://localhost:8000
```

### Production Configuration

For production deployments:

1. **Generate a secure secret key**:
```bash
# Python
python -c "import secrets; print(secrets.token_urlsafe(32))"

# OpenSSL
openssl rand -base64 32
```

2. **Update environment variables**:
```bash
# Backend
JWT_SECRET_KEY=<generated-secret-key>
JWT_ACCESS_TOKEN_EXPIRE_MINUTES=30  # Shorter for production

# Frontend
REACT_APP_API_URL=https://api.yourplatform.com
REACT_APP_WS_URL=https://api.yourplatform.com
```

3. **Enable HTTPS**:
- Use HTTPS for all production deployments
- Configure SSL certificates
- Update CORS settings for production domains

## API Authentication

### Login Endpoint

**Endpoint**: `POST /auth/login`

**Request**:
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
    "role": "admin",
    "email": "admin@example.com"
  }
}
```

### Authenticated Requests

Include the token in the `Authorization` header:

```bash
curl -X GET "http://localhost:8000/models" \
  -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
```

### Token Structure

JWT tokens contain:
- **Header**: Algorithm and token type
- **Payload**: User ID, username, role, expiration
- **Signature**: Cryptographic signature

Example decoded payload:
```json
{
  "sub": "admin",
  "user_id": "550e8400-e29b-41d4-a716-446655440000",
  "role": "admin",
  "exp": 1699564800
}
```

## Frontend Authentication

### Login Flow

1. **User enters credentials** on login page
2. **AuthService.login()** sends credentials to backend
3. **Backend validates** and returns JWT token
4. **AuthContext** stores token in localStorage
5. **User redirected** to dashboard
6. **All API requests** include token automatically

### Using Authentication in Components

```typescript
import { useAuth } from '../hooks/useAuth';

function MyComponent() {
  const { user, isAuthenticated, login, logout } = useAuth();

  if (!isAuthenticated) {
    return <div>Please log in</div>;
  }

  return (
    <div>
      <p>Welcome, {user.username}!</p>
      <button onClick={logout}>Logout</button>
    </div>
  );
}
```

### Protected Routes

Routes are automatically protected using `ProtectedRoute` component:

```typescript
<Route
  path="/dashboard"
  element={
    <ProtectedRoute>
      <Dashboard />
    </ProtectedRoute>
  }
/>
```

### Token Storage

Tokens are stored in browser localStorage:
- **Key**: `auth`
- **Value**: JSON object with token, user, and expiration
- **Persistence**: Survives page refresh
- **Security**: Accessible to JavaScript (XSS risk - mitigated by input sanitization)

### Session Management

- **Token Expiration**: Checked on every route change
- **Automatic Logout**: When token expires
- **Session Warning**: Shown 5 minutes before expiration
- **Remember Me**: Optional persistent sessions

## WebSocket Authentication

### Connection Setup

WebSocket connections include the JWT token:

```typescript
import io from 'socket.io-client';
import { AuthService } from './services/auth';

const socket = io('http://localhost:8000', {
  auth: {
    token: AuthService.getToken()
  },
  transports: ['websocket']
});
```

### Authentication Errors

Handle WebSocket authentication failures:

```typescript
socket.on('connect_error', (error) => {
  if (error.message === 'Authentication error') {
    // Token invalid or expired
    AuthService.removeToken();
    window.location.href = '/login';
  }
});
```

### Backend Validation

Backend validates WebSocket tokens on connection:

```python
@sio.event
async def connect(sid, environ, auth):
    token = auth.get('token')
    if not token:
        raise ConnectionRefusedError('Authentication required')
    
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        # Connection accepted
    except JWTError:
        raise ConnectionRefusedError('Invalid token')
```

## Security Best Practices

### Production Checklist

- [ ] Change default credentials
- [ ] Generate secure JWT secret key (min 32 characters)
- [ ] Use HTTPS for all connections
- [ ] Set shorter token expiration (15-30 minutes)
- [ ] Implement token refresh mechanism
- [ ] Enable CORS only for trusted domains
- [ ] Use httpOnly cookies (requires backend changes)
- [ ] Implement rate limiting on login endpoint
- [ ] Add account lockout after failed attempts
- [ ] Log all authentication events
- [ ] Monitor for suspicious activity

### Token Security

**DO**:
- ✅ Use HTTPS in production
- ✅ Set reasonable expiration times
- ✅ Validate tokens on every request
- ✅ Use strong secret keys
- ✅ Implement token refresh
- ✅ Clear tokens on logout

**DON'T**:
- ❌ Store tokens in cookies without httpOnly flag
- ❌ Use weak or default secret keys
- ❌ Set very long expiration times
- ❌ Share tokens between users
- ❌ Log tokens in plain text
- ❌ Transmit tokens over HTTP

### XSS Protection

- All user inputs are sanitized
- React automatically escapes content
- Content Security Policy headers recommended
- Avoid `dangerouslySetInnerHTML`

### CSRF Protection

- JWT tokens in Authorization header (not cookies)
- Not vulnerable to CSRF attacks
- No CSRF tokens needed

## Troubleshooting

### Login Issues

#### Problem: "Invalid username or password"

**Possible Causes**:
- Incorrect credentials
- Backend not running
- Database connection issues

**Solutions**:
1. Verify credentials (default: admin/admin123)
2. Check backend is running: `curl http://localhost:8000/health`
3. Check backend logs for errors
4. Verify database is accessible

#### Problem: "Unable to connect to server"

**Possible Causes**:
- Backend not running
- Wrong API URL
- Network issues
- CORS configuration

**Solutions**:
1. Start backend: `uvicorn src.api.main:app --reload`
2. Verify `REACT_APP_API_URL` in `frontend/.env`
3. Check network connectivity
4. Review CORS settings in backend

#### Problem: Login button does nothing

**Possible Causes**:
- JavaScript errors
- Form validation failing
- Network request blocked

**Solutions**:
1. Open browser console (F12) and check for errors
2. Verify form fields are filled correctly
3. Check Network tab for failed requests
4. Disable browser extensions that might block requests

### Token Issues

#### Problem: "Session expired, please log in again"

**Possible Causes**:
- Token expired (60 minutes default)
- Token invalidated
- System time mismatch

**Solutions**:
1. Log in again to get new token
2. Use "Remember Me" for longer sessions
3. Check system time is correct
4. Consider increasing `JWT_ACCESS_TOKEN_EXPIRE_MINUTES`

#### Problem: Token not persisting across page refresh

**Possible Causes**:
- localStorage disabled
- Private/incognito mode
- Browser extension blocking storage

**Solutions**:
1. Enable localStorage in browser settings
2. Disable private/incognito mode
3. Disable blocking browser extensions
4. Check browser console for storage errors

#### Problem: Automatic logout immediately after login

**Possible Causes**:
- Token expiration time in past
- System time mismatch
- Token validation failing

**Solutions**:
1. Check system time is correct
2. Verify `JWT_ACCESS_TOKEN_EXPIRE_MINUTES` is positive
3. Check backend logs for token validation errors
4. Regenerate JWT secret key

### API Request Issues

#### Problem: 401 Unauthorized on API requests

**Possible Causes**:
- Token expired
- Token not included in request
- Invalid token format

**Solutions**:
1. Check token is in localStorage: `localStorage.getItem('auth')`
2. Verify Authorization header: Check Network tab in browser
3. Log out and log in again
4. Check token format: `Bearer <token>`

#### Problem: 403 Forbidden on API requests

**Possible Causes**:
- Insufficient permissions
- Wrong user role
- Endpoint requires admin access

**Solutions**:
1. Check user role in token payload
2. Verify endpoint permissions
3. Use admin account if needed
4. Check backend authorization logic

### WebSocket Issues

#### Problem: WebSocket shows "Disconnected"

**Possible Causes**:
- Backend WebSocket server not running
- Wrong WebSocket URL
- Token not included in connection
- Authentication failed

**Solutions**:
1. Verify backend is running with WebSocket support
2. Check `REACT_APP_WS_URL` in `frontend/.env`
3. Verify token is valid: Check localStorage
4. Review browser console for WebSocket errors
5. Check backend logs for connection errors

#### Problem: No real-time updates

**Possible Causes**:
- WebSocket not connected
- Not subscribed to events
- Backend not emitting events

**Solutions**:
1. Check connection status in header
2. Verify WebSocket connection in Network tab
3. Check event subscriptions in code
4. Review backend event emission logic

### CORS Issues

#### Problem: CORS errors in browser console

**Error Message**: "Access to XMLHttpRequest at 'http://localhost:8000' from origin 'http://localhost:3000' has been blocked by CORS policy"

**Solutions**:
1. **Backend CORS Configuration** (`src/api/main.py`):
```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

2. **Production CORS**:
```python
allow_origins=[
    "https://yourplatform.com",
    "https://www.yourplatform.com"
]
```

3. **Development CORS** (allow all - not for production):
```python
allow_origins=["*"]
```

### Environment Variable Issues

#### Problem: Environment variables not loading

**Solutions**:
1. **Frontend**: Restart development server after changing `.env`
2. **Backend**: Restart uvicorn after changing `.env`
3. **Verify file location**: `.env` in root for backend, `frontend/.env` for frontend
4. **Check file name**: Must be exactly `.env` (not `.env.local` or `.env.development`)
5. **Prefix**: Frontend variables must start with `REACT_APP_`

### Database Issues

#### Problem: Authentication fails with database errors

**Solutions**:
1. Check database connection string
2. Verify database is running
3. Check user table exists and has data
4. Review backend logs for database errors

## Testing Authentication

### Manual Testing

1. **Test Login**:
```bash
curl -X POST "http://localhost:8000/auth/login" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=admin&password=admin123"
```

2. **Test Protected Endpoint**:
```bash
TOKEN="<your-token-here>"
curl -X GET "http://localhost:8000/models" \
  -H "Authorization: Bearer $TOKEN"
```

3. **Test Invalid Token**:
```bash
curl -X GET "http://localhost:8000/models" \
  -H "Authorization: Bearer invalid-token"
# Should return 401
```

### Automated Testing

Frontend authentication tests are located in `frontend/src/tests/`:
- `AuthService.test.ts` - Authentication service tests
- `AuthContext.test.tsx` - Auth context tests
- `ProtectedRoute.test.tsx` - Route protection tests
- `UserMenu.test.tsx` - User menu tests

Run tests:
```bash
cd frontend
npm test
```

## Additional Resources

- [JWT.io](https://jwt.io/) - JWT token debugger
- [FastAPI Security](https://fastapi.tiangolo.com/tutorial/security/) - FastAPI security documentation
- [React Authentication](https://reactjs.org/docs/context.html) - React context for auth
- [OWASP Authentication Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Authentication_Cheat_Sheet.html)

## Support

For additional help:
1. Check this documentation
2. Review backend logs: `logs/platform.log`
3. Check browser console for frontend errors
4. Review API documentation: `http://localhost:8000/docs`
5. Create an issue on GitHub with detailed error information

# Implementation Plan - Frontend Authentication

## Task Overview

This implementation plan breaks down the frontend authentication feature into discrete, manageable coding tasks. Each task builds incrementally on previous work, following test-driven development principles where appropriate.

---

## Phase 1: Core Authentication Infrastructure

- [x] 1. Create authentication types and interfaces
  - Create `frontend/src/types/auth.ts` with TypeScript interfaces
  - Define User, AuthState, LoginResponse, and AuthContextType interfaces
  - Export all types for use across the application
  - _Requirements: 1.1, 1.2, 2.1_

- [x] 2. Implement AuthService for token management
  - [x] 2.1 Create `frontend/src/services/auth.ts` with AuthService class
    - Implement login() method to call backend /auth/login endpoint
    - Implement token storage methods (getToken, setToken, removeToken)
    - Implement token validation (isTokenValid, getTokenExpiration)
    - Add JWT token decoding to extract expiration time
    - _Requirements: 1.2, 2.1, 2.2, 2.3_
  
  - [x] 2.2 Add localStorage integration
    - Implement secure token storage in localStorage
    - Add methods to persist and retrieve auth state
    - Handle JSON serialization/deserialization
    - _Requirements: 2.1, 2.2_

- [x] 3. Create AuthContext for state management
  - [x] 3.1 Create `frontend/src/contexts/AuthContext.tsx`
    - Implement AuthProvider component with useState and useEffect
    - Create context with user, token, isAuthenticated, isLoading states
    - Implement login function that calls AuthService
    - Implement logout function that clears state and storage
    - _Requirements: 1.2, 1.3, 2.2, 2.5_
  
  - [x] 3.2 Add token persistence on app load
    - Check localStorage for existing token on mount
    - Validate token expiration
    - Restore user state if token is valid
    - Set isLoading to false after check completes
    - _Requirements: 2.2, 2.3_
  
  - [x] 3.3 Implement token expiration handling
    - Add useEffect to check token expiration periodically
    - Automatically logout when token expires
    - Clear state and redirect to login
    - _Requirements: 2.4, 7.3_

- [x] 4. Create custom useAuth hook
  - Create `frontend/src/hooks/useAuth.ts`
  - Export hook that returns AuthContext value
  - Add error handling for context usage outside provider
  - _Requirements: 1.2, 1.3_

---

## Phase 2: Login Interface

- [x] 5. Create LoginPage component
  - [x] 5.1 Create `frontend/src/components/auth/LoginPage.tsx`
    - Build login form with Ant Design Form component
    - Add username and password input fields
    - Add "Remember Me" checkbox
    - Add submit button with loading state
    - _Requirements: 1.1, 1.2_
  
  - [x] 5.2 Implement form validation
    - Add required field validation
    - Add minimum length validation for password
    - Display validation errors inline
    - _Requirements: 1.1_
  
  - [x] 5.3 Add login submission logic
    - Call useAuth().login() on form submit
    - Handle loading state during authentication
    - Display error messages from backend
    - Redirect to dashboard on success
    - _Requirements: 1.2, 1.3, 1.4, 7.1, 7.2_
  
  - [x] 5.4 Style login page
    - Center form on page
    - Add platform logo and branding
    - Make responsive for mobile devices
    - Match existing design system
    - _Requirements: 1.1_

- [x] 6. Update routing for login page
  - Update `frontend/src/App.tsx` to add /login route
  - Wrap app with AuthProvider
  - Add redirect logic for authenticated users on /login
  - _Requirements: 1.5, 5.5_

---

## Phase 3: API Integration

- [x] 7. Add authentication interceptors to API service
  - [x] 7.1 Update `frontend/src/services/api.ts`
    - Add axios request interceptor to include Authorization header
    - Get token from AuthService and add Bearer prefix
    - Only add header if token exists
    - _Requirements: 3.1, 3.4_
  
  - [x] 7.2 Add response interceptor for auth errors
    - Intercept 401 responses
    - Clear token and redirect to login
    - Show appropriate error message
    - _Requirements: 3.2, 7.3_
  
  - [x] 7.3 Handle 403 Forbidden responses
    - Intercept 403 responses
    - Display permission denied message
    - Do not redirect to login
    - _Requirements: 3.3, 7.4_

---

## Phase 4: Route Protection

- [x] 8. Create ProtectedRoute component
  - [x] 8.1 Create `frontend/src/components/auth/ProtectedRoute.tsx`
    - Check authentication status using useAuth hook
    - Show loading spinner while checking auth
    - Redirect to /login if not authenticated
    - Render children if authenticated
    - _Requirements: 5.1, 5.2, 5.4_
  
  - [x] 8.2 Add role-based access control
    - Accept optional requiredRole prop
    - Check user role against required role
    - Show "Access Denied" message if role doesn't match
    - _Requirements: 5.1, 5.2, 6.5_

- [x] 9. Wrap protected routes
  - Update `frontend/src/App.tsx` to wrap routes with ProtectedRoute
  - Protect dashboard, upload, history, and configuration routes
  - Keep login route public
  - _Requirements: 5.1, 5.2_

---

## Phase 5: WebSocket Authentication

- [x] 10. Update WebSocket connection with authentication
  - [x] 10.1 Update `frontend/src/contexts/WebSocketContext.tsx`
    - Import AuthService to get token
    - Add auth option to socket.io connection with token
    - Only connect if token exists
    - _Requirements: 4.1, 4.4_
  
  - [x] 10.2 Handle WebSocket authentication errors
    - Listen for connect_error event
    - Check if error is authentication-related
    - Clear token and redirect to login if auth fails
    - Show connection status in UI
    - _Requirements: 4.3, 4.4, 7.5_
  
  - [x] 10.3 Reconnect WebSocket after login
    - Disconnect existing socket on logout
    - Reconnect socket after successful login
    - Update connection status in UI
    - _Requirements: 4.2, 4.3_

---

## Phase 6: User Interface Updates

- [x] 11. Create UserMenu component
  - [x] 11.1 Create `frontend/src/components/auth/UserMenu.tsx`
    - Display username from auth context
    - Add dropdown menu with user options
    - Add logout button
    - Show user role badge
    - _Requirements: 6.1, 6.2, 6.5_
  
  - [x] 11.2 Implement logout functionality
    - Call useAuth().logout() on button click
    - Clear all state and storage
    - Redirect to login page
    - Show confirmation message
    - _Requirements: 6.3_

- [x] 12. Update Header component
  - Update `frontend/src/components/layout/Header.tsx`
  - Add UserMenu component to header
  - Show connection status (WebSocket)
  - Hide user menu when not authenticated
  - _Requirements: 6.1, 6.2, 6.4, 7.5_

---

## Phase 7: Error Handling and Polish

- [x] 13. Implement comprehensive error handling
  - [x] 13.1 Add error state to AuthContext
    - Add error field to AuthState
    - Set error on login failure
    - Clear error on successful login
    - _Requirements: 7.1, 7.2_
  
  - [x] 13.2 Create error message mapping
    - Map backend error codes to user-friendly messages
    - Handle network errors separately
    - Display errors using Ant Design message component
    - _Requirements: 7.1, 7.2, 7.3, 7.4_

- [x] 14. Add session timeout warning
  - Create notification component for token expiration warning
  - Show warning 5 minutes before token expires
  - Provide option to extend session
  - Auto-logout when token expires
  - _Requirements: 2.4, 8.2_

- [x] 15. Implement "Remember Me" functionality
  - Store rememberMe preference in localStorage
  - Clear token on browser close if not checked
  - Use sessionStorage instead of localStorage when not remembered
  - _Requirements: 8.5_

---

## Phase 8: Testing and Documentation

- [x] 16. Write unit tests for authentication
  - [x]* 16.1 Test AuthService methods
    - Test login success and failure
    - Test token storage and retrieval
    - Test token validation
    - _Requirements: All_
  
  - [x]* 16.2 Test AuthContext
    - Test login flow
    - Test logout flow
    - Test token persistence
    - Test token expiration
    - _Requirements: All_
  
  - [x]* 16.3 Test ProtectedRoute component
    - Test redirect when not authenticated
    - Test access when authenticated
    - Test role-based access
    - _Requirements: 5.1, 5.2_

- [x] 17. Update documentation
  - Update README with authentication setup instructions
  - Document environment variables for API URL
  - Add troubleshooting section for common auth issues
  - Update API documentation with auth requirements
  - _Requirements: All_

---

## Verification Checklist

After implementation, verify:

- [x] User can log in with valid credentials
- [x] Invalid credentials show error message
- [x] Token is stored in localStorage
- [x] Token persists across page refresh
- [x] Protected routes redirect to login when not authenticated
- [x] API requests include Authorization header
- [x] 401 responses redirect to login
- [x] 403 responses show error message
- [x] WebSocket connects with authentication
- [x] User menu shows username and logout button
- [x] Logout clears token and redirects to login
- [x] Token expiration triggers automatic logout
- [x] All error scenarios show appropriate messages
- [x] UI is responsive and matches design system
- [x] No console errors or warnings

---

## Dependencies

- axios (already installed)
- react-router-dom (already installed)
- antd (already installed)
- socket.io-client (already installed)
- jwt-decode (needs installation: `npm install jwt-decode`)

---

## Estimated Effort

- Phase 1: 4-6 hours
- Phase 2: 3-4 hours
- Phase 3: 2-3 hours
- Phase 4: 2-3 hours
- Phase 5: 2-3 hours
- Phase 6: 2-3 hours
- Phase 7: 3-4 hours
- Phase 8: 4-6 hours

**Total: 22-32 hours** (3-4 days of focused development)

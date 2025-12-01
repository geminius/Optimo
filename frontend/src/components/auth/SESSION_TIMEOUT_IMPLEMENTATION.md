# Session Timeout Warning Implementation

## Overview

The Session Timeout Warning feature provides users with advance notice when their authentication token is about to expire, giving them the option to extend their session or logout gracefully.

## Components

### SessionTimeoutWarning Component

**Location:** `frontend/src/components/auth/SessionTimeoutWarning.tsx`

**Purpose:** Monitors token expiration and displays a modal warning dialog when the session is about to expire.

**Key Features:**

1. **Automatic Monitoring**: Checks token expiration every second when user is authenticated
2. **Configurable Warning Time**: Default 5 minutes before expiration (configurable via props)
3. **Visual Countdown**: Shows remaining time in MM:SS format with progress bar
4. **User Actions**: 
   - Extend Session: Attempts to refresh the token
   - Logout Now: Immediately ends the session
5. **Auto-Logout**: Automatically logs out when token expires

## Implementation Details

### Token Expiration Monitoring

The component uses a `setInterval` that runs every second to:
- Get time remaining until token expiration
- Check if warning threshold is reached
- Update the countdown display
- Auto-logout when token expires

```typescript
useEffect(() => {
  if (!isAuthenticated) return;

  const checkInterval = setInterval(() => {
    const remaining = AuthService.getTimeUntilExpiration();
    
    if (remaining === 0) {
      logout(); // Auto-logout
      return;
    }

    if (AuthService.isTokenExpiringSoon(warningMinutes)) {
      setShowWarning(true);
      setTimeRemaining(remaining);
    }
  }, 1000);

  return () => clearInterval(checkInterval);
}, [isAuthenticated, warningMinutes, logout]);
```

### Visual Feedback

The component provides rich visual feedback:

1. **Progress Bar**: Color-coded based on time remaining
   - Green (>50% time remaining)
   - Orange (25-50% time remaining)
   - Red (<25% time remaining)

2. **Countdown Timer**: Large, prominent display of time remaining in MM:SS format

3. **Modal Dialog**: Non-dismissible modal that requires user action

### Session Extension

When the user clicks "Extend Session":

1. Calls `refreshToken()` from AuthContext
2. Shows loading state on the button
3. On success: Closes the warning dialog
4. On failure: Logs out the user (forces re-authentication)

**Note:** The current implementation doesn't have a backend refresh endpoint, so the `refreshToken()` function validates the existing token. In production, this should call a `/auth/refresh` endpoint.

## Integration

### App.tsx Integration

The component is added at the root level inside the AuthProvider:

```tsx
<AuthProvider>
  <SessionTimeoutWarning warningMinutes={5} />
  <Routes>
    {/* ... routes ... */}
  </Routes>
</AuthProvider>
```

This ensures:
- Component has access to authentication context
- Warning is available on all pages
- Component lifecycle matches authentication state

### AuthService Methods Used

The component relies on these AuthService methods:

- `getTimeUntilExpiration()`: Returns milliseconds until token expires
- `isTokenExpiringSoon(minutes)`: Returns true if token expires within specified minutes
- `isTokenValid()`: Validates token is not expired

## Configuration

### Props

```typescript
interface SessionTimeoutWarningProps {
  warningMinutes?: number; // Default: 5
}
```

**Example Usage:**

```tsx
// Show warning 10 minutes before expiration
<SessionTimeoutWarning warningMinutes={10} />

// Use default (5 minutes)
<SessionTimeoutWarning />
```

## Requirements Satisfied

This implementation satisfies the following requirements:

- **Requirement 2.4**: Token expiration triggers automatic logout
- **Requirement 8.2**: System warns user when token is about to expire

## Testing

Test file: `frontend/src/tests/SessionTimeoutWarning.test.tsx`

**Test Coverage:**
- Warning not shown when token has plenty of time
- Warning shown when token is expiring soon
- Time formatting is correct
- Component doesn't render for unauthenticated users
- Extend session button functionality
- Logout button functionality

## Future Enhancements

1. **Backend Token Refresh**: Implement actual token refresh endpoint
2. **Configurable Actions**: Allow custom actions in the warning dialog
3. **Sound/Visual Alerts**: Add optional audio or browser notifications
4. **Session Activity Tracking**: Extend session automatically on user activity
5. **Remember User Preference**: Store user's preference for session extension

## Security Considerations

1. **Non-Dismissible Modal**: User must take action (extend or logout)
2. **Automatic Logout**: Ensures expired tokens cannot be used
3. **Secure Token Storage**: Tokens remain in localStorage/sessionStorage
4. **No Token Exposure**: Token is never displayed in the UI

## Browser Compatibility

- Modern browsers with ES6+ support
- Requires localStorage/sessionStorage support
- Modal uses Ant Design components (cross-browser compatible)

/**
 * Integration test for automatic logout on token expiration
 * 
 * This test verifies that when a JWT token expires, the system automatically
 * logs out the user and redirects them to the login page.
 * 
 * Validates Requirements: 2.4, 7.3
 */

import React from 'react';
import { render, screen, waitFor, act } from '@testing-library/react';
import { MemoryRouter, Routes, Route, useLocation } from 'react-router-dom';
import { AuthProvider } from '../contexts/AuthContext';
import { useAuth } from '../hooks/useAuth';
import AuthService from '../services/auth';

// Mock AuthService
jest.mock('../services/auth');

// Mock react-router-dom's useNavigate
const mockNavigate = jest.fn();
jest.mock('react-router-dom', () => ({
  ...jest.requireActual('react-router-dom'),
  useNavigate: () => mockNavigate,
}));

const mockedAuthService = AuthService as jest.Mocked<typeof AuthService>;

// Test component that displays current auth state and location
const TestComponent: React.FC = () => {
  const { isAuthenticated, user, error } = useAuth();
  const location = useLocation();

  return (
    <div>
      <div data-testid="auth-status">
        {isAuthenticated ? 'Authenticated' : 'Not Authenticated'}
      </div>
      <div data-testid="user-info">
        {user ? `User: ${user.username}` : 'No User'}
      </div>
      <div data-testid="current-path">{location.pathname}</div>
      {error && <div data-testid="error-message">{error}</div>}
    </div>
  );
};

// Test app wrapper
const TestApp: React.FC<{ initialPath?: string }> = ({ initialPath = '/dashboard' }) => {
  return (
    <MemoryRouter initialEntries={[initialPath]}>
      <AuthProvider>
        <Routes>
          <Route path="/dashboard" element={<TestComponent />} />
          <Route path="/login" element={<div data-testid="login-page">Login Page</div>} />
        </Routes>
      </AuthProvider>
    </MemoryRouter>
  );
};

describe('Token Expiration Auto Logout Integration Test', () => {
  const mockUser = {
    id: '123',
    username: 'testuser',
    role: 'user',
    email: 'test@example.com',
  };

  beforeEach(() => {
    jest.clearAllMocks();
    localStorage.clear();
    sessionStorage.clear();
    jest.useFakeTimers();
    mockNavigate.mockClear();
    
    // Mock console methods to avoid noise in test output
    jest.spyOn(console, 'error').mockImplementation(() => {});
    jest.spyOn(console, 'log').mockImplementation(() => {});
  });

  afterEach(() => {
    jest.useRealTimers();
    jest.restoreAllMocks();
  });

  it('should automatically logout when token expires', async () => {
    // Arrange - Start with valid token
    mockedAuthService.isTokenValid
      .mockReturnValueOnce(true)  // Initial mount check
      .mockReturnValueOnce(true)  // First expiration check (30s)
      .mockReturnValueOnce(false); // Second expiration check (60s) - token expired

    mockedAuthService.getToken.mockReturnValue('valid-token');
    mockedAuthService.getUser.mockReturnValue(mockUser);
    mockedAuthService.removeToken.mockImplementation(() => {});

    // Act - Render app with authenticated user
    render(<TestApp />);

    // Wait for initial auth check to complete
    await waitFor(() => {
      expect(screen.getByTestId('auth-status')).toHaveTextContent('Authenticated');
    });

    // Verify user is authenticated
    expect(screen.getByTestId('user-info')).toHaveTextContent('User: testuser');

    // Advance time by 30 seconds (first check - token still valid)
    act(() => {
      jest.advanceTimersByTime(30000);
    });

    // User should still be authenticated
    await waitFor(() => {
      expect(screen.getByTestId('auth-status')).toHaveTextContent('Authenticated');
    });

    // Advance time by another 30 seconds (second check - token expired)
    act(() => {
      jest.advanceTimersByTime(30000);
    });

    // Assert - User should be automatically logged out
    await waitFor(() => {
      expect(screen.getByTestId('auth-status')).toHaveTextContent('Not Authenticated');
    }, { timeout: 3000 });

    // Verify token was removed
    expect(mockedAuthService.removeToken).toHaveBeenCalled();

    // Verify error message is displayed
    expect(screen.getByTestId('error-message')).toHaveTextContent('Session expired');

    // Verify user info is cleared
    expect(screen.getByTestId('user-info')).toHaveTextContent('No User');
  });

  it('should redirect to login page after token expires', async () => {
    // Arrange
    mockedAuthService.isTokenValid
      .mockReturnValueOnce(true)  // Initial check
      .mockReturnValueOnce(false); // First expiration check - expired

    mockedAuthService.getToken.mockReturnValue('expired-token');
    mockedAuthService.getUser.mockReturnValue(mockUser);
    mockedAuthService.removeToken.mockImplementation(() => {});

    // Act - Render app
    render(<TestApp />);

    // Wait for initial auth
    await waitFor(() => {
      expect(screen.getByTestId('auth-status')).toHaveTextContent('Authenticated');
    });

    // Advance time to trigger expiration check
    act(() => {
      jest.advanceTimersByTime(30000);
    });

    // Assert - Should call navigate to redirect to login page
    await waitFor(() => {
      expect(mockNavigate).toHaveBeenCalledWith('/login');
    }, { timeout: 3000 });
  });

  it('should not check expiration when user is not authenticated', async () => {
    // Arrange - No valid token
    mockedAuthService.isTokenValid.mockReturnValue(false);
    mockedAuthService.getToken.mockReturnValue(null);
    mockedAuthService.getUser.mockReturnValue(null);

    // Act - Render app
    render(<TestApp />);

    // Wait for initial check
    await waitFor(() => {
      expect(screen.getByTestId('auth-status')).toHaveTextContent('Not Authenticated');
    });

    // Clear mock calls from initial check
    mockedAuthService.isTokenValid.mockClear();

    // Advance time
    act(() => {
      jest.advanceTimersByTime(60000);
    });

    // Assert - Should not call isTokenValid when not authenticated
    expect(mockedAuthService.isTokenValid).not.toHaveBeenCalled();
  });

  it('should continue checking token validity at 30-second intervals', async () => {
    // Arrange - Token remains valid for multiple checks
    mockedAuthService.isTokenValid.mockReturnValue(true);
    mockedAuthService.getToken.mockReturnValue('valid-token');
    mockedAuthService.getUser.mockReturnValue(mockUser);

    // Act - Render app
    render(<TestApp />);

    // Wait for initial auth
    await waitFor(() => {
      expect(screen.getByTestId('auth-status')).toHaveTextContent('Authenticated');
    });

    // Clear initial calls
    mockedAuthService.isTokenValid.mockClear();

    // Advance time by 90 seconds (should trigger 3 checks)
    act(() => {
      jest.advanceTimersByTime(90000);
    });

    // Assert - Should have checked token validity 3 times
    await waitFor(() => {
      expect(mockedAuthService.isTokenValid).toHaveBeenCalledTimes(3);
    });

    // User should still be authenticated
    expect(screen.getByTestId('auth-status')).toHaveTextContent('Authenticated');
  });

  it('should handle token expiration during active session', async () => {
    // Arrange - Simulate a token that expires mid-session
    let checkCount = 0;
    mockedAuthService.isTokenValid.mockImplementation(() => {
      checkCount++;
      // Token is valid for first 5 checks, then expires
      return checkCount <= 5;
    });

    mockedAuthService.getToken.mockReturnValue('session-token');
    mockedAuthService.getUser.mockReturnValue(mockUser);
    mockedAuthService.removeToken.mockImplementation(() => {});

    // Act - Render app
    render(<TestApp />);

    // Wait for initial auth
    await waitFor(() => {
      expect(screen.getByTestId('auth-status')).toHaveTextContent('Authenticated');
    });

    // Advance time by 150 seconds (5 checks at 30s intervals)
    act(() => {
      jest.advanceTimersByTime(150000);
    });

    // User should still be authenticated after 5 checks
    expect(screen.getByTestId('auth-status')).toHaveTextContent('Authenticated');

    // Advance time by another 30 seconds (6th check - token expires)
    act(() => {
      jest.advanceTimersByTime(30000);
    });

    // Assert - User should be logged out
    await waitFor(() => {
      expect(screen.getByTestId('auth-status')).toHaveTextContent('Not Authenticated');
    }, { timeout: 3000 });

    expect(mockedAuthService.removeToken).toHaveBeenCalled();
  });

  it('should display appropriate error message on token expiration', async () => {
    // Arrange
    mockedAuthService.isTokenValid
      .mockReturnValueOnce(true)
      .mockReturnValueOnce(false);

    mockedAuthService.getToken.mockReturnValue('token');
    mockedAuthService.getUser.mockReturnValue(mockUser);

    // Act
    render(<TestApp />);

    await waitFor(() => {
      expect(screen.getByTestId('auth-status')).toHaveTextContent('Authenticated');
    });

    // Trigger expiration
    act(() => {
      jest.advanceTimersByTime(30000);
    });

    // Assert - Error message should be user-friendly
    await waitFor(() => {
      const errorMessage = screen.getByTestId('error-message');
      expect(errorMessage).toHaveTextContent(/session expired/i);
      expect(errorMessage).toHaveTextContent(/please log in again/i);
    }, { timeout: 3000 });
  });
});

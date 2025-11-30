/**
 * Unit tests for AuthContext
 * 
 * Tests authentication context including login flow, logout flow,
 * token persistence, and token expiration handling.
 */

import React from 'react';
import { renderHook, act, waitFor } from '@testing-library/react';
import { BrowserRouter } from 'react-router-dom';
import { AuthProvider } from '../contexts/AuthContext';
import { useAuth } from '../hooks/useAuth';
import AuthService from '../services/auth';

// Mock dependencies
jest.mock('../services/auth');
jest.mock('react-router-dom', () => ({
  ...jest.requireActual('react-router-dom'),
  useNavigate: () => jest.fn(),
}));

const mockedAuthService = AuthService as jest.Mocked<typeof AuthService>;

describe('AuthContext', () => {
  const mockUser = {
    id: '123',
    username: 'testuser',
    role: 'user',
    email: 'test@example.com',
  };

  const mockLoginResponse = {
    access_token: 'mock-jwt-token',
    token_type: 'bearer',
    expires_in: 3600,
    user: mockUser,
  };

  // Wrapper component for testing hooks
  const wrapper = ({ children }: { children: React.ReactNode }) => (
    <BrowserRouter>
      <AuthProvider>{children}</AuthProvider>
    </BrowserRouter>
  );

  beforeEach(() => {
    jest.clearAllMocks();
    localStorage.clear();
    sessionStorage.clear();
    
    // Mock console methods
    jest.spyOn(console, 'error').mockImplementation(() => {});
    jest.spyOn(console, 'log').mockImplementation(() => {});
    
    // Default mock implementations
    mockedAuthService.isTokenValid.mockReturnValue(false);
    mockedAuthService.getToken.mockReturnValue(null);
    mockedAuthService.getUser.mockReturnValue(null);
  });

  afterEach(() => {
    jest.restoreAllMocks();
  });

  describe('initial state', () => {
    it('should have correct initial state when no token exists', async () => {
      // Arrange
      mockedAuthService.isTokenValid.mockReturnValue(false);

      // Act
      const { result } = renderHook(() => useAuth(), { wrapper });

      // Wait for initial auth check to complete
      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
      });

      // Assert
      expect(result.current.user).toBeNull();
      expect(result.current.token).toBeNull();
      expect(result.current.isAuthenticated).toBe(false);
      expect(result.current.error).toBeNull();
    });

    it('should restore auth state from valid stored token', async () => {
      // Arrange
      mockedAuthService.isTokenValid.mockReturnValue(true);
      mockedAuthService.getToken.mockReturnValue('valid-token');
      mockedAuthService.getUser.mockReturnValue(mockUser);

      // Act
      const { result } = renderHook(() => useAuth(), { wrapper });

      // Wait for auth restoration
      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
      });

      // Assert
      expect(result.current.user).toEqual(mockUser);
      expect(result.current.token).toBe('valid-token');
      expect(result.current.isAuthenticated).toBe(true);
    });

    it('should clear invalid token on mount', async () => {
      // Arrange
      mockedAuthService.isTokenValid.mockReturnValue(false);
      mockedAuthService.getToken.mockReturnValue('expired-token');

      // Act
      const { result } = renderHook(() => useAuth(), { wrapper });

      // Wait for auth check
      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
      });

      // Assert
      expect(mockedAuthService.removeToken).toHaveBeenCalled();
      expect(result.current.isAuthenticated).toBe(false);
    });
  });

  describe('login flow', () => {
    it('should successfully login with valid credentials', async () => {
      // Arrange
      mockedAuthService.login.mockResolvedValue(mockLoginResponse);
      mockedAuthService.setToken.mockImplementation(() => {});

      const { result } = renderHook(() => useAuth(), { wrapper });

      // Wait for initial load
      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
      });

      // Act
      await act(async () => {
        await result.current.login('testuser', 'password123', true);
      });

      // Assert
      expect(mockedAuthService.login).toHaveBeenCalledWith('testuser', 'password123');
      expect(mockedAuthService.setToken).toHaveBeenCalledWith(
        mockLoginResponse.access_token,
        mockLoginResponse.user,
        mockLoginResponse.expires_in,
        true
      );
      expect(result.current.user).toEqual(mockUser);
      expect(result.current.token).toBe(mockLoginResponse.access_token);
      expect(result.current.isAuthenticated).toBe(true);
      expect(result.current.error).toBeNull();
    });

    it('should handle login failure with error message', async () => {
      // Arrange
      const errorMessage = 'Invalid credentials';
      mockedAuthService.login.mockRejectedValue(new Error(errorMessage));

      const { result } = renderHook(() => useAuth(), { wrapper });

      // Wait for initial load
      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
      });

      // Act & Assert
      await act(async () => {
        await expect(result.current.login('testuser', 'wrongpassword')).rejects.toThrow();
      });

      expect(result.current.user).toBeNull();
      expect(result.current.token).toBeNull();
      expect(result.current.isAuthenticated).toBe(false);
      expect(result.current.error).toBeTruthy();
    });

    it('should set loading state during login', async () => {
      // Arrange
      let resolveLogin: (value: any) => void;
      const loginPromise = new Promise((resolve) => {
        resolveLogin = resolve;
      });
      mockedAuthService.login.mockReturnValue(loginPromise as any);

      const { result } = renderHook(() => useAuth(), { wrapper });

      // Wait for initial load
      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
      });

      // Act - Start login
      act(() => {
        result.current.login('testuser', 'password123');
      });

      // Assert - Should be loading
      await waitFor(() => {
        expect(result.current.isLoading).toBe(true);
      });

      // Complete login
      act(() => {
        resolveLogin!(mockLoginResponse);
      });

      // Assert - Should finish loading
      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
      });
    });

    it('should clear previous errors on new login attempt', async () => {
      // Arrange
      mockedAuthService.login
        .mockRejectedValueOnce(new Error('First error'))
        .mockResolvedValueOnce(mockLoginResponse);

      const { result } = renderHook(() => useAuth(), { wrapper });

      // Wait for initial load
      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
      });

      // First login attempt (fails)
      await act(async () => {
        try {
          await result.current.login('testuser', 'wrongpassword');
        } catch (e) {
          // Expected error
        }
      });

      expect(result.current.error).toBeTruthy();

      // Second login attempt (succeeds)
      await act(async () => {
        await result.current.login('testuser', 'correctpassword');
      });

      // Assert - Error should be cleared
      expect(result.current.error).toBeNull();
      expect(result.current.isAuthenticated).toBe(true);
    });

    it('should emit auth:login event on successful login', async () => {
      // Arrange
      mockedAuthService.login.mockResolvedValue(mockLoginResponse);
      const eventListener = jest.fn();
      window.addEventListener('auth:login', eventListener);

      const { result } = renderHook(() => useAuth(), { wrapper });

      // Wait for initial load
      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
      });

      // Act
      await act(async () => {
        await result.current.login('testuser', 'password123');
      });

      // Assert
      expect(eventListener).toHaveBeenCalled();

      // Cleanup
      window.removeEventListener('auth:login', eventListener);
    });

    it('should respect rememberMe parameter', async () => {
      // Arrange
      mockedAuthService.login.mockResolvedValue(mockLoginResponse);

      const { result } = renderHook(() => useAuth(), { wrapper });

      // Wait for initial load
      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
      });

      // Act - Login with rememberMe = false
      await act(async () => {
        await result.current.login('testuser', 'password123', false);
      });

      // Assert
      expect(mockedAuthService.setToken).toHaveBeenCalledWith(
        expect.any(String),
        expect.any(Object),
        expect.any(Number),
        false // rememberMe should be false
      );
    });
  });

  describe('logout flow', () => {
    it('should clear auth state on logout', async () => {
      // Arrange
      mockedAuthService.login.mockResolvedValue(mockLoginResponse);
      mockedAuthService.logout.mockImplementation(() => {});

      const { result } = renderHook(() => useAuth(), { wrapper });

      // Wait for initial load
      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
      });

      // Login first
      await act(async () => {
        await result.current.login('testuser', 'password123');
      });

      expect(result.current.isAuthenticated).toBe(true);

      // Act - Logout
      act(() => {
        result.current.logout();
      });

      // Assert
      expect(mockedAuthService.logout).toHaveBeenCalled();
      expect(result.current.user).toBeNull();
      expect(result.current.token).toBeNull();
      expect(result.current.isAuthenticated).toBe(false);
    });

    it('should emit auth:logout event on logout', () => {
      // Arrange
      const eventListener = jest.fn();
      window.addEventListener('auth:logout', eventListener);

      const { result } = renderHook(() => useAuth(), { wrapper });

      // Act
      act(() => {
        result.current.logout();
      });

      // Assert
      expect(eventListener).toHaveBeenCalled();

      // Cleanup
      window.removeEventListener('auth:logout', eventListener);
    });
  });

  describe('token persistence', () => {
    it('should persist token across page reloads', async () => {
      // Arrange - First render with valid token
      mockedAuthService.isTokenValid.mockReturnValue(true);
      mockedAuthService.getToken.mockReturnValue('persisted-token');
      mockedAuthService.getUser.mockReturnValue(mockUser);

      // Act - First render
      const { result: result1, unmount } = renderHook(() => useAuth(), { wrapper });

      await waitFor(() => {
        expect(result1.current.isLoading).toBe(false);
      });

      expect(result1.current.isAuthenticated).toBe(true);

      // Unmount (simulate page reload)
      unmount();

      // Second render (simulating page reload)
      const { result: result2 } = renderHook(() => useAuth(), { wrapper });

      await waitFor(() => {
        expect(result2.current.isLoading).toBe(false);
      });

      // Assert - Should still be authenticated
      expect(result2.current.isAuthenticated).toBe(true);
      expect(result2.current.user).toEqual(mockUser);
      expect(result2.current.token).toBe('persisted-token');
    });

    it('should not persist invalid token', async () => {
      // Arrange
      mockedAuthService.isTokenValid.mockReturnValue(false);
      mockedAuthService.getToken.mockReturnValue('invalid-token');

      // Act
      const { result } = renderHook(() => useAuth(), { wrapper });

      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
      });

      // Assert
      expect(mockedAuthService.removeToken).toHaveBeenCalled();
      expect(result.current.isAuthenticated).toBe(false);
    });
  });

  describe('token expiration', () => {
    beforeEach(() => {
      jest.useFakeTimers();
    });

    afterEach(() => {
      jest.useRealTimers();
    });

    it('should check token expiration periodically when authenticated', async () => {
      // Arrange
      mockedAuthService.login.mockResolvedValue(mockLoginResponse);
      mockedAuthService.isTokenValid
        .mockReturnValueOnce(true) // Initial check
        .mockReturnValueOnce(true) // First interval check
        .mockReturnValueOnce(false); // Second interval check (expired)

      const { result } = renderHook(() => useAuth(), { wrapper });

      // Wait for initial load
      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
      });

      // Login
      await act(async () => {
        await result.current.login('testuser', 'password123');
      });

      expect(result.current.isAuthenticated).toBe(true);

      // Act - Advance time by 30 seconds (first check)
      act(() => {
        jest.advanceTimersByTime(30000);
      });

      // Still authenticated
      expect(result.current.isAuthenticated).toBe(true);

      // Act - Advance time by another 30 seconds (token expires)
      act(() => {
        jest.advanceTimersByTime(30000);
      });

      // Assert - Should be logged out
      await waitFor(() => {
        expect(result.current.isAuthenticated).toBe(false);
      });
    });

    it('should not check expiration when not authenticated', async () => {
      // Arrange
      mockedAuthService.isTokenValid.mockReturnValue(false);

      const { result } = renderHook(() => useAuth(), { wrapper });

      // Wait for initial load
      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
      });

      expect(result.current.isAuthenticated).toBe(false);

      // Clear previous calls
      mockedAuthService.isTokenValid.mockClear();

      // Act - Advance time
      act(() => {
        jest.advanceTimersByTime(60000);
      });

      // Assert - Should not check token validity when not authenticated
      expect(mockedAuthService.isTokenValid).not.toHaveBeenCalled();
    });

    it('should set error message when token expires', async () => {
      // Arrange
      mockedAuthService.login.mockResolvedValue(mockLoginResponse);
      mockedAuthService.isTokenValid
        .mockReturnValueOnce(true)
        .mockReturnValueOnce(false);

      const { result } = renderHook(() => useAuth(), { wrapper });

      // Wait for initial load
      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
      });

      // Login
      await act(async () => {
        await result.current.login('testuser', 'password123');
      });

      // Act - Token expires
      act(() => {
        jest.advanceTimersByTime(30000);
      });

      // Assert
      await waitFor(() => {
        expect(result.current.error).toContain('Session expired');
      });
    });
  });

  describe('refreshToken', () => {
    it('should validate token on refresh attempt', async () => {
      // Arrange
      mockedAuthService.isTokenValid.mockReturnValue(true);

      const { result } = renderHook(() => useAuth(), { wrapper });

      // Wait for initial load
      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
      });

      // Act
      await act(async () => {
        await result.current.refreshToken();
      });

      // Assert
      expect(mockedAuthService.isTokenValid).toHaveBeenCalled();
    });

    it('should throw error if token is invalid during refresh', async () => {
      // Arrange
      mockedAuthService.isTokenValid.mockReturnValue(false);

      const { result } = renderHook(() => useAuth(), { wrapper });

      // Wait for initial load
      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
      });

      // Act & Assert
      await act(async () => {
        await expect(result.current.refreshToken()).rejects.toThrow();
      });
    });
  });
});

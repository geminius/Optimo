/**
 * Logout Integration Tests
 * 
 * Comprehensive tests to verify that logout:
 * 1. Clears the token from storage (localStorage and sessionStorage)
 * 2. Redirects to the login page
 * 3. Clears all authentication state
 */

import React from 'react';
import { renderHook, act, waitFor } from '@testing-library/react';
import { BrowserRouter } from 'react-router-dom';
import { AuthProvider } from '../contexts/AuthContext';
import { useAuth } from '../hooks/useAuth';
import AuthService from '../services/auth';

// Mock AuthService
jest.mock('../services/auth');

const mockedAuthService = AuthService as jest.Mocked<typeof AuthService>;

// Mock react-router-dom navigate
const mockNavigate = jest.fn();
jest.mock('react-router-dom', () => ({
  ...jest.requireActual('react-router-dom'),
  useNavigate: () => mockNavigate,
}));



describe('Logout Integration Tests', () => {
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
    mockNavigate.mockClear();
    localStorage.clear();
    sessionStorage.clear();

    // Mock console methods
    jest.spyOn(console, 'error').mockImplementation(() => {});
    jest.spyOn(console, 'log').mockImplementation(() => {});

    // Default mock implementations
    mockedAuthService.isTokenValid.mockReturnValue(false);
    mockedAuthService.getToken.mockReturnValue(null);
    mockedAuthService.getUser.mockReturnValue(null);
    mockedAuthService.setToken.mockImplementation((token, user, expiresIn, rememberMe) => {
      const authData = {
        token,
        user,
        expiresAt: new Date(Date.now() + expiresIn * 1000).toISOString(),
        rememberMe,
      };
      const storage = rememberMe ? localStorage : sessionStorage;
      storage.setItem('auth', JSON.stringify(authData));
    });
    mockedAuthService.removeToken.mockImplementation(() => {
      localStorage.removeItem('auth');
      sessionStorage.removeItem('auth');
    });
    mockedAuthService.logout.mockImplementation(() => {
      localStorage.removeItem('auth');
      sessionStorage.removeItem('auth');
    });
  });

  afterEach(() => {
    jest.restoreAllMocks();
  });

  test('logout clears token from localStorage', async () => {
    // Arrange - Set up authenticated state
    const authData = {
      token: 'test-token',
      user: mockUser,
      expiresAt: new Date(Date.now() + 3600000).toISOString(),
      rememberMe: true,
    };
    localStorage.setItem('auth', JSON.stringify(authData));

    mockedAuthService.login.mockResolvedValue(mockLoginResponse);

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

    // Verify token exists before logout
    expect(localStorage.getItem('auth')).toBeTruthy();

    // Act - Logout
    act(() => {
      result.current.logout();
    });

    // Assert - Token should be cleared from localStorage
    expect(mockedAuthService.logout).toHaveBeenCalled();
    expect(localStorage.getItem('auth')).toBeNull();
  });

  test('logout clears token from sessionStorage', async () => {
    // Arrange - Set up authenticated state with sessionStorage
    mockedAuthService.login.mockResolvedValue(mockLoginResponse);

    const { result } = renderHook(() => useAuth(), { wrapper });

    // Wait for initial load
    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    // Login with rememberMe = false (stores in sessionStorage)
    await act(async () => {
      await result.current.login('testuser', 'password123', false);
    });

    expect(result.current.isAuthenticated).toBe(true);

    // Verify token exists in sessionStorage before logout
    expect(sessionStorage.getItem('auth')).toBeTruthy();

    // Act - Logout
    act(() => {
      result.current.logout();
    });

    // Assert - Token should be cleared from sessionStorage
    expect(mockedAuthService.logout).toHaveBeenCalled();
    expect(sessionStorage.getItem('auth')).toBeNull();
  });

  test('logout redirects to login page', async () => {
    // Arrange
    mockedAuthService.login.mockResolvedValue(mockLoginResponse);

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
    expect(mockNavigate).toHaveBeenCalledWith('/dashboard');

    // Clear previous navigate calls
    mockNavigate.mockClear();

    // Act - Logout
    act(() => {
      result.current.logout();
    });

    // Assert - Should redirect to login page
    expect(mockNavigate).toHaveBeenCalledWith('/login');
  });

  test('logout clears all authentication state', async () => {
    // Arrange
    mockedAuthService.login.mockResolvedValue(mockLoginResponse);

    const { result } = renderHook(() => useAuth(), { wrapper });

    // Wait for initial load
    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    // Login first
    await act(async () => {
      await result.current.login('testuser', 'password123');
    });

    // Verify authenticated state
    expect(result.current.isAuthenticated).toBe(true);
    expect(result.current.user).toEqual(mockUser);
    expect(result.current.token).toBe(mockLoginResponse.access_token);

    // Act - Logout
    act(() => {
      result.current.logout();
    });

    // Assert - All auth state should be cleared
    expect(mockedAuthService.logout).toHaveBeenCalled();
    expect(result.current.user).toBeNull();
    expect(result.current.token).toBeNull();
    expect(result.current.isAuthenticated).toBe(false);
  });

  test('logout emits auth:logout event', async () => {
    // Arrange
    mockedAuthService.login.mockResolvedValue(mockLoginResponse);

    const eventListener = jest.fn();
    window.addEventListener('auth:logout', eventListener);

    const { result } = renderHook(() => useAuth(), { wrapper });

    // Wait for initial load
    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    // Login first
    await act(async () => {
      await result.current.login('testuser', 'password123');
    });

    // Act - Logout
    act(() => {
      result.current.logout();
    });

    // Assert - Event should be emitted
    expect(eventListener).toHaveBeenCalled();

    // Cleanup
    window.removeEventListener('auth:logout', eventListener);
  });

  test('logout calls AuthService.logout method', async () => {
    // Arrange
    mockedAuthService.login.mockResolvedValue(mockLoginResponse);

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

    // Assert - AuthService.logout should be called
    expect(mockedAuthService.logout).toHaveBeenCalled();
  });

  test('logout clears both localStorage and sessionStorage', async () => {
    // Arrange - Manually set tokens in both storages (edge case scenario)
    const authData = {
      token: 'test-token',
      user: mockUser,
      expiresAt: new Date(Date.now() + 3600000).toISOString(),
      rememberMe: true,
    };
    localStorage.setItem('auth', JSON.stringify(authData));
    sessionStorage.setItem('auth', JSON.stringify(authData));

    mockedAuthService.login.mockResolvedValue(mockLoginResponse);

    const { result } = renderHook(() => useAuth(), { wrapper });

    // Wait for initial load
    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
    });

    // Login first (this will overwrite one of the storages, but we manually set both)
    await act(async () => {
      await result.current.login('testuser', 'password123');
    });

    // Manually ensure both storages have data (simulating edge case)
    localStorage.setItem('auth', JSON.stringify(authData));
    sessionStorage.setItem('auth', JSON.stringify(authData));

    // Verify tokens exist in both storages before logout
    expect(localStorage.getItem('auth')).toBeTruthy();
    expect(sessionStorage.getItem('auth')).toBeTruthy();

    // Act - Logout
    act(() => {
      result.current.logout();
    });

    // Assert - Both storages should be cleared
    expect(mockedAuthService.logout).toHaveBeenCalled();
    expect(localStorage.getItem('auth')).toBeNull();
    expect(sessionStorage.getItem('auth')).toBeNull();
  });
});

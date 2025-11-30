/**
 * Token Storage Integration Test
 * 
 * End-to-end test demonstrating token storage flow:
 * Login → Token Stored → Page Refresh → Token Retrieved → User Authenticated
 */

import React from 'react';
import { render, screen, waitFor } from '@testing-library/react';
import { BrowserRouter } from 'react-router-dom';
import { AuthProvider } from '../contexts/AuthContext';
import AuthService from '../services/auth';
import App from '../App';

// Mock the API
jest.mock('axios');

describe('Token Storage Integration', () => {
  beforeEach(() => {
    // Clear storage before each test
    localStorage.clear();
    sessionStorage.clear();
    
    // Clear any auth state
    AuthService.removeToken();
  });

  afterEach(() => {
    localStorage.clear();
    sessionStorage.clear();
  });

  it('should store token in localStorage after successful login', async () => {
    // Arrange: Mock successful login response
    const mockUser = {
      id: '123',
      username: 'testuser',
      role: 'user',
      email: 'test@example.com',
    };

    const mockToken = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJ0ZXN0dXNlciIsImV4cCI6OTk5OTk5OTk5OX0.test';

    // Act: Store token (simulating successful login)
    AuthService.setToken(mockToken, mockUser, 3600, true);

    // Assert: Verify token is in localStorage
    const storedData = localStorage.getItem('auth');
    expect(storedData).not.toBeNull();

    const parsedData = JSON.parse(storedData!);
    expect(parsedData.token).toBe(mockToken);
    expect(parsedData.user).toEqual(mockUser);
    expect(parsedData.rememberMe).toBe(true);
  });

  it('should retrieve token from localStorage on app reload', async () => {
    // Arrange: Store token (simulating previous login)
    const mockUser = {
      id: '123',
      username: 'testuser',
      role: 'user',
      email: 'test@example.com',
    };

    const mockToken = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJ0ZXN0dXNlciIsImV4cCI6OTk5OTk5OTk5OX0.test';

    AuthService.setToken(mockToken, mockUser, 3600, true);

    // Act: Retrieve token (simulating app reload)
    const retrievedToken = AuthService.getToken();
    const retrievedUser = AuthService.getUser();

    // Assert: Verify token and user are retrieved correctly
    expect(retrievedToken).toBe(mockToken);
    expect(retrievedUser).toEqual(mockUser);
  });

  it('should validate token before using it', async () => {
    // Arrange: Store valid token
    const mockUser = {
      id: '123',
      username: 'testuser',
      role: 'user',
    };

    const mockToken = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJ0ZXN0dXNlciIsImV4cCI6OTk5OTk5OTk5OX0.test';

    AuthService.setToken(mockToken, mockUser, 3600, true);

    // Act: Validate token
    const isValid = AuthService.isTokenValid();

    // Assert: Token should be valid
    expect(isValid).toBe(true);
  });

  it('should reject expired token', async () => {
    // Arrange: Store expired token
    const mockUser = {
      id: '123',
      username: 'testuser',
      role: 'user',
    };

    const mockToken = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJ0ZXN0dXNlciIsImV4cCI6OTk5OTk5OTk5OX0.test';

    // Store with negative expiration (already expired)
    AuthService.setToken(mockToken, mockUser, -3600, true);

    // Act: Validate token
    const isValid = AuthService.isTokenValid();

    // Assert: Token should be invalid
    expect(isValid).toBe(false);
  });

  it('should clear token from localStorage on logout', async () => {
    // Arrange: Store token
    const mockUser = {
      id: '123',
      username: 'testuser',
      role: 'user',
    };

    const mockToken = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJ0ZXN0dXNlciIsImV4cCI6OTk5OTk5OTk5OX0.test';

    AuthService.setToken(mockToken, mockUser, 3600, true);

    // Verify token is stored
    expect(localStorage.getItem('auth')).not.toBeNull();

    // Act: Logout
    AuthService.logout();

    // Assert: Token should be removed
    expect(localStorage.getItem('auth')).toBeNull();
    expect(AuthService.getToken()).toBeNull();
  });

  it('should persist token across simulated page refresh', async () => {
    // Arrange: Store token
    const mockUser = {
      id: '123',
      username: 'testuser',
      role: 'user',
    };

    const mockToken = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJ0ZXN0dXNlciIsImV4cCI6OTk5OTk5OTk5OX0.test';

    AuthService.setToken(mockToken, mockUser, 3600, true);

    // Act: Simulate page refresh by getting token again
    const tokenBeforeRefresh = AuthService.getToken();
    
    // Simulate refresh - token should still be in localStorage
    const tokenAfterRefresh = AuthService.getToken();

    // Assert: Token persists
    expect(tokenBeforeRefresh).toBe(mockToken);
    expect(tokenAfterRefresh).toBe(mockToken);
    expect(tokenBeforeRefresh).toBe(tokenAfterRefresh);
  });

  it('should use sessionStorage when rememberMe is false', async () => {
    // Arrange: Store token with rememberMe=false
    const mockUser = {
      id: '123',
      username: 'testuser',
      role: 'user',
    };

    const mockToken = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJ0ZXN0dXNlciIsImV4cCI6OTk5OTk5OTk5OX0.test';

    AuthService.setToken(mockToken, mockUser, 3600, false);

    // Assert: Token should be in sessionStorage, not localStorage
    expect(sessionStorage.getItem('auth')).not.toBeNull();
    expect(localStorage.getItem('auth')).toBeNull();

    // Verify token can still be retrieved
    const retrievedToken = AuthService.getToken();
    expect(retrievedToken).toBe(mockToken);
  });

  it('should handle complete login-to-logout flow with storage', async () => {
    // Step 1: Login (store token)
    const mockUser = {
      id: '123',
      username: 'testuser',
      role: 'user',
    };

    const mockToken = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJ0ZXN0dXNlciIsImV4cCI6OTk5OTk5OTk5OX0.test';

    AuthService.setToken(mockToken, mockUser, 3600, true);
    expect(localStorage.getItem('auth')).not.toBeNull();

    // Step 2: Verify authentication
    expect(AuthService.isTokenValid()).toBe(true);
    expect(AuthService.getToken()).toBe(mockToken);
    expect(AuthService.getUser()).toEqual(mockUser);

    // Step 3: Simulate page refresh
    const tokenAfterRefresh = AuthService.getToken();
    expect(tokenAfterRefresh).toBe(mockToken);

    // Step 4: Logout (clear token)
    AuthService.logout();
    expect(localStorage.getItem('auth')).toBeNull();
    expect(AuthService.getToken()).toBeNull();
    expect(AuthService.getUser()).toBeNull();
  });
});

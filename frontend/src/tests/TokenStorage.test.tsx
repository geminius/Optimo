/**
 * Token Storage Verification Tests
 * 
 * Tests to verify that JWT tokens are properly stored in localStorage
 * after successful login and retrieved on app reload.
 */

import AuthService from '../services/auth';
import { User } from '../types/auth';

describe('Token Storage in localStorage', () => {
  const mockUser: User = {
    id: '123',
    username: 'testuser',
    role: 'user',
    email: 'test@example.com',
  };

  const mockToken = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJ0ZXN0dXNlciIsImV4cCI6OTk5OTk5OTk5OX0.test';
  const expiresIn = 3600; // 1 hour

  beforeEach(() => {
    // Clear storage before each test
    localStorage.clear();
    sessionStorage.clear();
  });

  afterEach(() => {
    // Clean up after each test
    localStorage.clear();
    sessionStorage.clear();
  });

  describe('Token Storage with rememberMe=true', () => {
    it('should store token in localStorage when rememberMe is true', () => {
      // Store token with rememberMe=true
      AuthService.setToken(mockToken, mockUser, expiresIn, true);

      // Verify token is in localStorage
      const storedData = localStorage.getItem('auth');
      expect(storedData).not.toBeNull();

      // Parse and verify the stored data
      const parsedData = JSON.parse(storedData!);
      expect(parsedData.token).toBe(mockToken);
      expect(parsedData.user).toEqual(mockUser);
      expect(parsedData.rememberMe).toBe(true);
      expect(parsedData.expiresAt).toBeDefined();
    });

    it('should retrieve token from localStorage', () => {
      // Store token
      AuthService.setToken(mockToken, mockUser, expiresIn, true);

      // Retrieve token
      const retrievedToken = AuthService.getToken();
      expect(retrievedToken).toBe(mockToken);
    });

    it('should retrieve user from localStorage', () => {
      // Store token
      AuthService.setToken(mockToken, mockUser, expiresIn, true);

      // Retrieve user
      const retrievedUser = AuthService.getUser();
      expect(retrievedUser).toEqual(mockUser);
    });

    it('should persist token across page reloads (simulated)', () => {
      // Store token
      AuthService.setToken(mockToken, mockUser, expiresIn, true);

      // Simulate page reload by creating a new instance check
      // (In real scenario, the service would be re-instantiated)
      const token = AuthService.getToken();
      const user = AuthService.getUser();

      expect(token).toBe(mockToken);
      expect(user).toEqual(mockUser);
    });
  });

  describe('Token Storage with rememberMe=false', () => {
    it('should store token in sessionStorage when rememberMe is false', () => {
      // Store token with rememberMe=false
      AuthService.setToken(mockToken, mockUser, expiresIn, false);

      // Verify token is in sessionStorage, not localStorage
      const sessionData = sessionStorage.getItem('auth');
      const localData = localStorage.getItem('auth');

      expect(sessionData).not.toBeNull();
      expect(localData).toBeNull();

      // Parse and verify the stored data
      const parsedData = JSON.parse(sessionData!);
      expect(parsedData.token).toBe(mockToken);
      expect(parsedData.user).toEqual(mockUser);
      expect(parsedData.rememberMe).toBe(false);
    });

    it('should retrieve token from sessionStorage when rememberMe is false', () => {
      // Store token in sessionStorage
      AuthService.setToken(mockToken, mockUser, expiresIn, false);

      // Retrieve token
      const retrievedToken = AuthService.getToken();
      expect(retrievedToken).toBe(mockToken);
    });
  });

  describe('Token Removal', () => {
    it('should remove token from localStorage on logout', () => {
      // Store token
      AuthService.setToken(mockToken, mockUser, expiresIn, true);

      // Verify it's stored
      expect(localStorage.getItem('auth')).not.toBeNull();

      // Remove token
      AuthService.removeToken();

      // Verify it's removed
      expect(localStorage.getItem('auth')).toBeNull();
      expect(AuthService.getToken()).toBeNull();
    });

    it('should remove token from both localStorage and sessionStorage', () => {
      // Store in both (edge case)
      localStorage.setItem('auth', JSON.stringify({ token: mockToken, user: mockUser }));
      sessionStorage.setItem('auth', JSON.stringify({ token: mockToken, user: mockUser }));

      // Remove token
      AuthService.removeToken();

      // Verify both are cleared
      expect(localStorage.getItem('auth')).toBeNull();
      expect(sessionStorage.getItem('auth')).toBeNull();
    });
  });

  describe('Token Validation', () => {
    it('should validate stored token correctly', () => {
      // Store valid token
      AuthService.setToken(mockToken, mockUser, expiresIn, true);

      // Validate token
      const isValid = AuthService.isTokenValid();
      expect(isValid).toBe(true);
    });

    it('should return false for missing token', () => {
      // Don't store any token
      const isValid = AuthService.isTokenValid();
      expect(isValid).toBe(false);
    });

    it('should return false for expired token', () => {
      // Store token with negative expiration (already expired)
      AuthService.setToken(mockToken, mockUser, -3600, true);

      // Validate token
      const isValid = AuthService.isTokenValid();
      expect(isValid).toBe(false);
    });
  });

  describe('Token Expiration', () => {
    it('should calculate token expiration correctly', () => {
      // Store token
      AuthService.setToken(mockToken, mockUser, expiresIn, true);

      // Get expiration
      const expiration = AuthService.getTokenExpiration();
      expect(expiration).not.toBeNull();
      expect(expiration).toBeInstanceOf(Date);

      // Verify expiration is in the future
      expect(expiration!.getTime()).toBeGreaterThan(Date.now());
    });

    it('should return null expiration for missing token', () => {
      const expiration = AuthService.getTokenExpiration();
      expect(expiration).toBeNull();
    });
  });

  describe('Storage Format', () => {
    it('should store data in correct JSON format', () => {
      AuthService.setToken(mockToken, mockUser, expiresIn, true);

      const storedData = localStorage.getItem('auth');
      expect(storedData).not.toBeNull();

      // Verify it's valid JSON
      expect(() => JSON.parse(storedData!)).not.toThrow();

      // Verify structure
      const parsed = JSON.parse(storedData!);
      expect(parsed).toHaveProperty('token');
      expect(parsed).toHaveProperty('user');
      expect(parsed).toHaveProperty('expiresAt');
      expect(parsed).toHaveProperty('rememberMe');
    });

    it('should handle JSON parsing errors gracefully', () => {
      // Store invalid JSON
      localStorage.setItem('auth', 'invalid-json');

      // Should return null instead of throwing
      const token = AuthService.getToken();
      expect(token).toBeNull();
    });
  });

  describe('Edge Cases', () => {
    it('should handle storage quota exceeded gracefully', () => {
      // This is hard to test without actually filling storage
      // but we can verify the method doesn't throw
      expect(() => {
        AuthService.setToken(mockToken, mockUser, expiresIn, true);
      }).not.toThrow();
    });

    it('should handle missing localStorage gracefully', () => {
      // Mock localStorage being unavailable
      const originalLocalStorage = window.localStorage;
      Object.defineProperty(window, 'localStorage', {
        value: undefined,
        writable: true,
      });

      // Should not throw
      expect(() => {
        AuthService.getToken();
      }).not.toThrow();

      // Restore
      Object.defineProperty(window, 'localStorage', {
        value: originalLocalStorage,
        writable: true,
      });
    });
  });
});

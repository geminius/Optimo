/**
 * Remember Me Functionality Tests
 * 
 * Tests for Task 15: Implement "Remember Me" functionality
 * Verifies that rememberMe preference controls storage location
 */

import AuthService from '../services/auth';
import { User } from '../types/auth';

describe('Remember Me Functionality', () => {
  const mockUser: User = {
    id: '1',
    username: 'testuser',
    role: 'user',
    email: 'test@example.com',
  };

  const mockToken = 'mock.jwt.token';
  const mockExpiresIn = 3600; // 1 hour

  beforeEach(() => {
    // Clear all storage before each test
    localStorage.clear();
    sessionStorage.clear();
  });

  afterEach(() => {
    // Clean up after each test
    localStorage.clear();
    sessionStorage.clear();
  });

  describe('Token Storage with Remember Me', () => {
    it('should store token in localStorage when rememberMe is true', () => {
      // Store with rememberMe = true
      AuthService.setToken(mockToken, mockUser, mockExpiresIn, true);

      // Verify token is in localStorage
      const storedData = localStorage.getItem('auth');
      expect(storedData).not.toBeNull();
      
      const parsedData = JSON.parse(storedData!);
      expect(parsedData.token).toBe(mockToken);
      expect(parsedData.user).toEqual(mockUser);
      expect(parsedData.rememberMe).toBe(true);

      // Verify token is NOT in sessionStorage
      const sessionData = sessionStorage.getItem('auth');
      expect(sessionData).toBeNull();
    });

    it('should store token in sessionStorage when rememberMe is false', () => {
      // Store with rememberMe = false
      AuthService.setToken(mockToken, mockUser, mockExpiresIn, false);

      // Verify token is in sessionStorage
      const storedData = sessionStorage.getItem('auth');
      expect(storedData).not.toBeNull();
      
      const parsedData = JSON.parse(storedData!);
      expect(parsedData.token).toBe(mockToken);
      expect(parsedData.user).toEqual(mockUser);
      expect(parsedData.rememberMe).toBe(false);

      // Verify token is NOT in localStorage
      const localData = localStorage.getItem('auth');
      expect(localData).toBeNull();
    });

    it('should default to rememberMe true when parameter is omitted', () => {
      // Store without specifying rememberMe (should default to true)
      AuthService.setToken(mockToken, mockUser, mockExpiresIn);

      // Verify token is in localStorage
      const storedData = localStorage.getItem('auth');
      expect(storedData).not.toBeNull();
      
      const parsedData = JSON.parse(storedData!);
      expect(parsedData.rememberMe).toBe(true);
    });
  });

  describe('Token Retrieval', () => {
    it('should retrieve token from localStorage when present', () => {
      // Store in localStorage
      AuthService.setToken(mockToken, mockUser, mockExpiresIn, true);

      // Retrieve token
      const retrievedToken = AuthService.getToken();
      expect(retrievedToken).toBe(mockToken);

      const retrievedUser = AuthService.getUser();
      expect(retrievedUser).toEqual(mockUser);
    });

    it('should retrieve token from sessionStorage when present', () => {
      // Store in sessionStorage
      AuthService.setToken(mockToken, mockUser, mockExpiresIn, false);

      // Retrieve token
      const retrievedToken = AuthService.getToken();
      expect(retrievedToken).toBe(mockToken);

      const retrievedUser = AuthService.getUser();
      expect(retrievedUser).toEqual(mockUser);
    });

    it('should prioritize localStorage over sessionStorage', () => {
      const localToken = 'local.token';
      const sessionToken = 'session.token';

      // Store different tokens in both storages
      localStorage.setItem('auth', JSON.stringify({
        token: localToken,
        user: mockUser,
        expiresAt: new Date(Date.now() + 3600000).toISOString(),
        rememberMe: true,
      }));

      sessionStorage.setItem('auth', JSON.stringify({
        token: sessionToken,
        user: mockUser,
        expiresAt: new Date(Date.now() + 3600000).toISOString(),
        rememberMe: false,
      }));

      // Should retrieve from localStorage first
      const retrievedToken = AuthService.getToken();
      expect(retrievedToken).toBe(localToken);
    });
  });

  describe('Token Removal', () => {
    it('should clear token from both localStorage and sessionStorage', () => {
      // Store in both storages
      localStorage.setItem('auth', JSON.stringify({
        token: 'local.token',
        user: mockUser,
        expiresAt: new Date(Date.now() + 3600000).toISOString(),
        rememberMe: true,
      }));

      sessionStorage.setItem('auth', JSON.stringify({
        token: 'session.token',
        user: mockUser,
        expiresAt: new Date(Date.now() + 3600000).toISOString(),
        rememberMe: false,
      }));

      // Remove token
      AuthService.removeToken();

      // Verify both storages are cleared
      expect(localStorage.getItem('auth')).toBeNull();
      expect(sessionStorage.getItem('auth')).toBeNull();
    });
  });

  describe('Session Persistence Behavior', () => {
    it('should simulate browser close clearing sessionStorage but not localStorage', () => {
      // Store with rememberMe = false (sessionStorage)
      AuthService.setToken(mockToken, mockUser, mockExpiresIn, false);
      
      // Verify token is in sessionStorage
      expect(sessionStorage.getItem('auth')).not.toBeNull();

      // Simulate browser close by clearing sessionStorage
      sessionStorage.clear();

      // Token should be gone
      expect(AuthService.getToken()).toBeNull();
    });

    it('should persist token in localStorage across simulated browser restarts', () => {
      // Store with rememberMe = true (localStorage)
      AuthService.setToken(mockToken, mockUser, mockExpiresIn, true);
      
      // Verify token is in localStorage
      expect(localStorage.getItem('auth')).not.toBeNull();

      // Simulate browser close by clearing sessionStorage (localStorage persists)
      sessionStorage.clear();

      // Token should still be available
      expect(AuthService.getToken()).toBe(mockToken);
    });
  });

  describe('Remember Me Preference Storage', () => {
    it('should store rememberMe preference with auth data', () => {
      // Store with rememberMe = true
      AuthService.setToken(mockToken, mockUser, mockExpiresIn, true);
      
      const storedData = localStorage.getItem('auth');
      const parsedData = JSON.parse(storedData!);
      
      expect(parsedData.rememberMe).toBe(true);
    });

    it('should store rememberMe = false preference with auth data', () => {
      // Store with rememberMe = false
      AuthService.setToken(mockToken, mockUser, mockExpiresIn, false);
      
      const storedData = sessionStorage.getItem('auth');
      const parsedData = JSON.parse(storedData!);
      
      expect(parsedData.rememberMe).toBe(false);
    });
  });
});

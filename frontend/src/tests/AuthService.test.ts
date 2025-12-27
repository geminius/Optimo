/**
 * Unit tests for AuthService
 * 
 * Tests authentication service methods including login, token storage,
 * token validation, and token expiration handling.
 */

import AuthService from '../services/auth';
import axios from 'axios';
import { jwtDecode } from 'jwt-decode';

// Mock dependencies
jest.mock('axios');
jest.mock('jwt-decode');

const mockedAxios = axios as jest.Mocked<typeof axios>;
const mockedJwtDecode = jwtDecode as jest.MockedFunction<typeof jwtDecode>;

describe('AuthService', () => {
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

  const mockTokenPayload = {
    sub: 'testuser',
    exp: Math.floor(Date.now() / 1000) + 3600, // Expires in 1 hour
    role: 'user',
  };

  beforeEach(() => {
    // Clear all mocks before each test
    jest.clearAllMocks();
    
    // Clear localStorage and sessionStorage
    localStorage.clear();
    sessionStorage.clear();
    
    // Reset console.error mock
    jest.spyOn(console, 'error').mockImplementation(() => {});
  });

  afterEach(() => {
    jest.restoreAllMocks();
  });

  describe('login', () => {
    it('should successfully login with valid credentials', async () => {
      // Arrange
      mockedAxios.post.mockResolvedValue({ data: mockLoginResponse });

      // Act
      const result = await AuthService.login('testuser', 'password123');

      // Assert
      expect(mockedAxios.post).toHaveBeenCalledWith(
        'http://localhost:8000/auth/login',
        { username: 'testuser', password: 'password123' },
        expect.objectContaining({
          headers: {
            'Content-Type': 'application/json',
          },
        })
      );
      expect(result).toEqual(mockLoginResponse);
    });

    it('should throw error on login failure with invalid credentials', async () => {
      // Arrange
      const errorResponse = {
        response: {
          status: 401,
          data: { detail: 'Invalid credentials' },
        },
      };
      mockedAxios.post.mockRejectedValue(errorResponse);

      // Act & Assert
      await expect(AuthService.login('testuser', 'wrongpassword')).rejects.toThrow();
      expect(mockedAxios.post).toHaveBeenCalled();
    });

    it('should throw error on network failure', async () => {
      // Arrange
      mockedAxios.post.mockRejectedValue(new Error('Network Error'));

      // Act & Assert
      await expect(AuthService.login('testuser', 'password123')).rejects.toThrow();
    });

    it('should send credentials as JSON data', async () => {
      // Arrange
      mockedAxios.post.mockResolvedValue({ data: mockLoginResponse });

      // Act
      await AuthService.login('testuser', 'password123');

      // Assert
      const callArgs = mockedAxios.post.mock.calls[0];
      const requestData = callArgs[1];
      
      // Check that it's a plain object with the right data
      expect(requestData).toEqual({
        username: 'testuser',
        password: 'password123'
      });
      
      // Check headers
      const config = callArgs[2];
      expect(config.headers['Content-Type']).toBe('application/json');
    });
  });

  describe('token storage', () => {
    it('should store token in localStorage when rememberMe is true', () => {
      // Act
      AuthService.setToken(mockLoginResponse.access_token, mockUser, 3600, true);

      // Assert
      const stored = localStorage.getItem('auth');
      expect(stored).not.toBeNull();
      
      const parsedData = JSON.parse(stored!);
      expect(parsedData.token).toBe(mockLoginResponse.access_token);
      expect(parsedData.user).toEqual(mockUser);
      expect(parsedData.rememberMe).toBe(true);
    });

    it('should store token in sessionStorage when rememberMe is false', () => {
      // Act
      AuthService.setToken(mockLoginResponse.access_token, mockUser, 3600, false);

      // Assert
      const stored = sessionStorage.getItem('auth');
      expect(stored).not.toBeNull();
      
      const parsedData = JSON.parse(stored!);
      expect(parsedData.token).toBe(mockLoginResponse.access_token);
      expect(parsedData.rememberMe).toBe(false);
    });

    it('should calculate correct expiration time', () => {
      // Arrange
      const expiresIn = 3600; // 1 hour
      const beforeTime = Date.now();

      // Act
      AuthService.setToken(mockLoginResponse.access_token, mockUser, expiresIn, true);

      // Assert
      const stored = localStorage.getItem('auth');
      const parsedData = JSON.parse(stored!);
      const expiresAt = new Date(parsedData.expiresAt).getTime();
      const afterTime = Date.now();

      // Should be approximately 1 hour from now
      expect(expiresAt).toBeGreaterThanOrEqual(beforeTime + expiresIn * 1000);
      expect(expiresAt).toBeLessThanOrEqual(afterTime + expiresIn * 1000);
    });

    it('should retrieve token from localStorage', () => {
      // Arrange
      AuthService.setToken(mockLoginResponse.access_token, mockUser, 3600, true);

      // Act
      const token = AuthService.getToken();

      // Assert
      expect(token).toBe(mockLoginResponse.access_token);
    });

    it('should retrieve token from sessionStorage if not in localStorage', () => {
      // Arrange
      AuthService.setToken(mockLoginResponse.access_token, mockUser, 3600, false);

      // Act
      const token = AuthService.getToken();

      // Assert
      expect(token).toBe(mockLoginResponse.access_token);
    });

    it('should return null when no token is stored', () => {
      // Act
      const token = AuthService.getToken();

      // Assert
      expect(token).toBeNull();
    });

    it('should remove token from both storages', () => {
      // Arrange
      localStorage.setItem('auth', JSON.stringify({ token: 'test-token' }));
      sessionStorage.setItem('auth', JSON.stringify({ token: 'test-token' }));

      // Act
      AuthService.removeToken();

      // Assert
      expect(localStorage.getItem('auth')).toBeNull();
      expect(sessionStorage.getItem('auth')).toBeNull();
    });
  });

  describe('token validation', () => {
    it('should return true for valid non-expired token', () => {
      // Arrange
      const futureExp = Math.floor(Date.now() / 1000) + 3600; // 1 hour from now
      mockedJwtDecode.mockReturnValue({ ...mockTokenPayload, exp: futureExp });
      
      AuthService.setToken(mockLoginResponse.access_token, mockUser, 3600, true);

      // Act
      const isValid = AuthService.isTokenValid();

      // Assert
      expect(isValid).toBe(true);
    });

    it('should return false for expired token based on stored expiration', () => {
      // Arrange
      const pastExp = Math.floor(Date.now() / 1000) - 3600; // 1 hour ago
      mockedJwtDecode.mockReturnValue({ ...mockTokenPayload, exp: pastExp });
      
      // Store token with past expiration
      const authData = {
        token: mockLoginResponse.access_token,
        user: mockUser,
        expiresAt: new Date(Date.now() - 3600000).toISOString(), // 1 hour ago
        rememberMe: true,
      };
      localStorage.setItem('auth', JSON.stringify(authData));

      // Act
      const isValid = AuthService.isTokenValid();

      // Assert
      expect(isValid).toBe(false);
    });

    it('should return false for expired token based on JWT exp claim', () => {
      // Arrange
      const pastExp = Math.floor(Date.now() / 1000) - 100; // Expired
      mockedJwtDecode.mockReturnValue({ ...mockTokenPayload, exp: pastExp });
      
      // Store with future expiration but JWT is expired
      AuthService.setToken(mockLoginResponse.access_token, mockUser, 3600, true);

      // Act
      const isValid = AuthService.isTokenValid();

      // Assert
      expect(isValid).toBe(false);
    });

    it('should return false when no token is stored', () => {
      // Act
      const isValid = AuthService.isTokenValid();

      // Assert
      expect(isValid).toBe(false);
    });

    it('should return false when token decode fails', () => {
      // Arrange
      mockedJwtDecode.mockImplementation(() => {
        throw new Error('Invalid token');
      });
      
      AuthService.setToken('invalid-token', mockUser, 3600, true);

      // Act
      const isValid = AuthService.isTokenValid();

      // Assert
      expect(isValid).toBe(false);
    });

    it('should handle corrupted stored data gracefully', () => {
      // Arrange
      localStorage.setItem('auth', 'invalid-json');

      // Act
      const isValid = AuthService.isTokenValid();

      // Assert
      expect(isValid).toBe(false);
    });
  });

  describe('token expiration', () => {
    it('should return correct expiration date from JWT', () => {
      // Arrange
      const futureExp = Math.floor(Date.now() / 1000) + 3600;
      mockedJwtDecode.mockReturnValue({ ...mockTokenPayload, exp: futureExp });
      
      AuthService.setToken(mockLoginResponse.access_token, mockUser, 3600, true);

      // Act
      const expiration = AuthService.getTokenExpiration();

      // Assert
      expect(expiration).not.toBeNull();
      expect(expiration!.getTime()).toBeCloseTo(futureExp * 1000, -3);
    });

    it('should return null when no token is stored', () => {
      // Act
      const expiration = AuthService.getTokenExpiration();

      // Assert
      expect(expiration).toBeNull();
    });

    it('should fallback to stored expiration if JWT decode fails', () => {
      // Arrange
      mockedJwtDecode.mockImplementation(() => {
        throw new Error('Decode error');
      });
      
      AuthService.setToken(mockLoginResponse.access_token, mockUser, 3600, true);

      // Act
      const expiration = AuthService.getTokenExpiration();

      // Assert
      expect(expiration).not.toBeNull();
      expect(expiration).toBeInstanceOf(Date);
    });

    it('should calculate time until expiration correctly', () => {
      // Arrange
      const futureExp = Math.floor(Date.now() / 1000) + 3600; // 1 hour
      mockedJwtDecode.mockReturnValue({ ...mockTokenPayload, exp: futureExp });
      
      AuthService.setToken(mockLoginResponse.access_token, mockUser, 3600, true);

      // Act
      const timeRemaining = AuthService.getTimeUntilExpiration();

      // Assert
      expect(timeRemaining).toBeGreaterThan(0);
      expect(timeRemaining).toBeLessThanOrEqual(3600000); // Less than or equal to 1 hour in ms
    });

    it('should return 0 for expired token', () => {
      // Arrange
      const pastExp = Math.floor(Date.now() / 1000) - 100;
      mockedJwtDecode.mockReturnValue({ ...mockTokenPayload, exp: pastExp });
      
      const authData = {
        token: mockLoginResponse.access_token,
        user: mockUser,
        expiresAt: new Date(Date.now() - 100000).toISOString(),
        rememberMe: true,
      };
      localStorage.setItem('auth', JSON.stringify(authData));

      // Act
      const timeRemaining = AuthService.getTimeUntilExpiration();

      // Assert
      expect(timeRemaining).toBe(0);
    });

    it('should detect token expiring soon', () => {
      // Arrange
      const soonExp = Math.floor(Date.now() / 1000) + 240; // 4 minutes
      mockedJwtDecode.mockReturnValue({ ...mockTokenPayload, exp: soonExp });
      
      AuthService.setToken(mockLoginResponse.access_token, mockUser, 240, true);

      // Act
      const isExpiringSoon = AuthService.isTokenExpiringSoon(5); // Within 5 minutes

      // Assert
      expect(isExpiringSoon).toBe(true);
    });

    it('should not detect token expiring soon when plenty of time remains', () => {
      // Arrange
      const futureExp = Math.floor(Date.now() / 1000) + 3600; // 1 hour
      mockedJwtDecode.mockReturnValue({ ...mockTokenPayload, exp: futureExp });
      
      AuthService.setToken(mockLoginResponse.access_token, mockUser, 3600, true);

      // Act
      const isExpiringSoon = AuthService.isTokenExpiringSoon(5); // Within 5 minutes

      // Assert
      expect(isExpiringSoon).toBe(false);
    });
  });

  describe('user management', () => {
    it('should retrieve stored user information', () => {
      // Arrange
      AuthService.setToken(mockLoginResponse.access_token, mockUser, 3600, true);

      // Act
      const user = AuthService.getUser();

      // Assert
      expect(user).toEqual(mockUser);
    });

    it('should return null when no user is stored', () => {
      // Act
      const user = AuthService.getUser();

      // Assert
      expect(user).toBeNull();
    });
  });

  describe('logout', () => {
    it('should clear all authentication data', () => {
      // Arrange
      AuthService.setToken(mockLoginResponse.access_token, mockUser, 3600, true);
      expect(AuthService.getToken()).not.toBeNull();

      // Act
      AuthService.logout();

      // Assert
      expect(AuthService.getToken()).toBeNull();
      expect(AuthService.getUser()).toBeNull();
    });
  });

  describe('token decoding', () => {
    it('should decode valid JWT token', () => {
      // Arrange
      mockedJwtDecode.mockReturnValue(mockTokenPayload);

      // Act
      const decoded = AuthService.decodeToken('valid-token');

      // Assert
      expect(decoded).toEqual(mockTokenPayload);
      expect(mockedJwtDecode).toHaveBeenCalledWith('valid-token');
    });

    it('should return null for invalid token', () => {
      // Arrange
      mockedJwtDecode.mockImplementation(() => {
        throw new Error('Invalid token');
      });

      // Act
      const decoded = AuthService.decodeToken('invalid-token');

      // Assert
      expect(decoded).toBeNull();
    });
  });
});

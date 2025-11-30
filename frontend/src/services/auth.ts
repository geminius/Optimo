/**
 * Authentication Service
 * 
 * Handles authentication API calls, token management, and token validation.
 * Provides methods for login, logout, token storage, and token expiration checking.
 */

import axios from 'axios';
import { jwtDecode } from 'jwt-decode';
import { LoginResponse, LoginRequest, User, StoredAuth, TokenPayload } from '../types/auth';
import ErrorHandler from '../utils/errorHandler';
import { logger } from '../utils/logger';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';
const AUTH_STORAGE_KEY = 'auth';

/**
 * AuthService class for managing authentication operations
 */
class AuthService {
  private static instance: AuthService;

  private constructor() {}

  /**
   * Get singleton instance of AuthService
   */
  public static getInstance(): AuthService {
    if (!AuthService.instance) {
      AuthService.instance = new AuthService();
    }
    return AuthService.instance;
  }

  /**
   * Login user with username and password
   * @param username - User's username
   * @param password - User's password
   * @returns Promise with login response containing token and user info
   */
  async login(username: string, password: string): Promise<LoginResponse> {
    try {
      const loginData: LoginRequest = { username, password };

      const response = await axios.post<LoginResponse>(
        `${API_BASE_URL}/auth/login`,
        loginData,
        {
          headers: {
            'Content-Type': 'application/json',
          },
        }
      );

      return response.data;
    } catch (error) {
      // Use ErrorHandler to parse and throw appropriate error
      const errorDetails = ErrorHandler.parseError(error);
      ErrorHandler.logError(errorDetails, { component: 'AuthService', method: 'login' });
      throw new Error(errorDetails.message);
    }
  }

  /**
   * Get stored authentication token
   * @returns Token string or null if not found
   */
  getToken(): string | null {
    try {
      const storedAuth = this.getStoredAuth();
      return storedAuth?.token || null;
    } catch (error) {
      logger.error('Error getting token:', error);
      return null;
    }
  }

  /**
   * Store authentication token and user data
   * @param token - JWT token
   * @param user - User information
   * @param expiresIn - Token expiration time in seconds
   * @param rememberMe - Whether to persist session
   */
  setToken(token: string, user: User, expiresIn: number, rememberMe: boolean = true): void {
    try {
      const expiresAt = new Date(Date.now() + expiresIn * 1000).toISOString();
      
      const authData: StoredAuth = {
        token,
        user,
        expiresAt,
        rememberMe,
      };

      const storage = rememberMe ? localStorage : sessionStorage;
      storage.setItem(AUTH_STORAGE_KEY, JSON.stringify(authData));
    } catch (error) {
      logger.error('Error storing token:', error);
      throw new Error('Failed to store authentication data');
    }
  }

  /**
   * Remove authentication token and user data
   */
  removeToken(): void {
    try {
      localStorage.removeItem(AUTH_STORAGE_KEY);
      sessionStorage.removeItem(AUTH_STORAGE_KEY);
    } catch (error) {
      logger.error('Error removing token:', error);
    }
  }

  /**
   * Check if stored token is valid and not expired
   * @returns True if token is valid, false otherwise
   */
  isTokenValid(): boolean {
    try {
      const storedAuth = this.getStoredAuth();
      
      if (!storedAuth || !storedAuth.token) {
        return false;
      }

      // Check if token is expired based on stored expiration
      const expiresAt = new Date(storedAuth.expiresAt);
      const now = new Date();
      
      if (now >= expiresAt) {
        return false;
      }

      // Additional validation: decode JWT and check exp claim
      try {
        const decoded = jwtDecode<TokenPayload>(storedAuth.token);
        const tokenExpiration = decoded.exp * 1000; // Convert to milliseconds
        
        if (Date.now() >= tokenExpiration) {
          return false;
        }
      } catch (decodeError) {
        logger.error('Error decoding token:', decodeError);
        return false;
      }

      return true;
    } catch (error) {
      logger.error('Error validating token:', error);
      return false;
    }
  }

  /**
   * Get token expiration date
   * @returns Date object representing token expiration or null if not available
   */
  getTokenExpiration(): Date | null {
    try {
      const storedAuth = this.getStoredAuth();
      
      if (!storedAuth || !storedAuth.token) {
        return null;
      }

      // Try to get expiration from JWT token
      try {
        const decoded = jwtDecode<TokenPayload>(storedAuth.token);
        return new Date(decoded.exp * 1000);
      } catch (decodeError) {
        // Fallback to stored expiration
        return new Date(storedAuth.expiresAt);
      }
    } catch (error) {
      logger.error('Error getting token expiration:', error);
      return null;
    }
  }

  /**
   * Get stored user information
   * @returns User object or null if not found
   */
  getUser(): User | null {
    try {
      const storedAuth = this.getStoredAuth();
      return storedAuth?.user || null;
    } catch (error) {
      logger.error('Error getting user:', error);
      return null;
    }
  }

  /**
   * Get complete stored authentication data
   * @returns StoredAuth object or null if not found
   */
  private getStoredAuth(): StoredAuth | null {
    try {
      // Check localStorage first
      let authData = localStorage.getItem(AUTH_STORAGE_KEY);
      
      // If not in localStorage, check sessionStorage
      if (!authData) {
        authData = sessionStorage.getItem(AUTH_STORAGE_KEY);
      }

      if (!authData) {
        return null;
      }

      return JSON.parse(authData) as StoredAuth;
    } catch (error) {
      logger.error('Error parsing stored auth data:', error);
      return null;
    }
  }

  /**
   * Logout user by removing all stored authentication data
   */
  logout(): void {
    this.removeToken();
  }

  /**
   * Decode JWT token to extract payload
   * @param token - JWT token string
   * @returns Decoded token payload or null if invalid
   */
  decodeToken(token: string): TokenPayload | null {
    try {
      return jwtDecode<TokenPayload>(token);
    } catch (error) {
      logger.error('Error decoding token:', error);
      return null;
    }
  }

  /**
   * Get time remaining until token expires (in milliseconds)
   * @returns Milliseconds until expiration or 0 if expired/invalid
   */
  getTimeUntilExpiration(): number {
    const expiration = this.getTokenExpiration();
    if (!expiration) {
      return 0;
    }

    const timeRemaining = expiration.getTime() - Date.now();
    return Math.max(0, timeRemaining);
  }

  /**
   * Check if token will expire soon (within specified minutes)
   * @param minutes - Number of minutes to check
   * @returns True if token expires within specified time
   */
  isTokenExpiringSoon(minutes: number = 5): boolean {
    const timeRemaining = this.getTimeUntilExpiration();
    const thresholdMs = minutes * 60 * 1000;
    return timeRemaining > 0 && timeRemaining <= thresholdMs;
  }
}

// Export singleton instance
export default AuthService.getInstance();

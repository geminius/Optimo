/**
 * Authentication Context
 * 
 * Provides centralized authentication state management using React Context API.
 * Manages user authentication state, token persistence, and automatic token expiration handling.
 */

import React, { createContext, useState, useEffect, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import { AuthContextType, AuthState } from '../types/auth';
import AuthService from '../services/auth';
import ErrorHandler from '../utils/errorHandler';
import { logger } from '../utils/logger';

// Create the authentication context
const AuthContext = createContext<AuthContextType | undefined>(undefined);

/**
 * AuthProvider component that wraps the application and provides auth state
 */
export const AuthProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const navigate = useNavigate();

  // Authentication state
  const [authState, setAuthState] = useState<AuthState>({
    user: null,
    token: null,
    isAuthenticated: false,
    isLoading: true,
    error: null,
  });

  /**
   * Login function that authenticates user with backend
   * @param username - User's username
   * @param password - User's password
   * @param rememberMe - Whether to persist session across browser restarts
   */
  const login = useCallback(async (username: string, password: string, rememberMe: boolean = true): Promise<void> => {
    try {
      // Clear any previous errors
      setAuthState(prev => ({ ...prev, error: null, isLoading: true }));

      // Call AuthService to authenticate
      const response = await AuthService.login(username, password);

      // Store token and user data with rememberMe preference
      AuthService.setToken(response.access_token, response.user, response.expires_in, rememberMe);

      // Update state
      setAuthState({
        user: response.user,
        token: response.access_token,
        isAuthenticated: true,
        isLoading: false,
        error: null,
      });

      // Emit custom event for WebSocket to reconnect
      window.dispatchEvent(new CustomEvent('auth:login'));

      // Show success message
      ErrorHandler.showSuccess('Login successful');

      // Navigate to dashboard on successful login
      navigate('/dashboard');
    } catch (error) {
      // Parse error using ErrorHandler
      const errorDetails = ErrorHandler.handleAuthError(error, false);
      
      setAuthState(prev => ({
        ...prev,
        user: null,
        token: null,
        isAuthenticated: false,
        isLoading: false,
        error: errorDetails.message,
      }));

      throw error;
    }
  }, [navigate]);

  /**
   * Logout function that clears authentication state and storage
   */
  const logout = useCallback(() => {
    // Emit custom event for WebSocket to disconnect
    window.dispatchEvent(new CustomEvent('auth:logout'));

    // Clear token from storage
    AuthService.logout();

    // Clear state
    setAuthState({
      user: null,
      token: null,
      isAuthenticated: false,
      isLoading: false,
      error: null,
    });

    // Show info message
    ErrorHandler.showInfo('Logged out successfully');

    // Redirect to login page
    navigate('/login');
  }, [navigate]);

  /**
   * Refresh token function
   * Note: Since the backend doesn't currently support token refresh,
   * this will prompt the user to re-login by clearing the session.
   * In a production system, this would call a /auth/refresh endpoint.
   */
  const refreshToken = useCallback(async (): Promise<void> => {
    try {
      // TODO: When backend supports token refresh, implement it here
      // For now, we'll just validate the current token
      // If it's still valid, we can continue; otherwise, force re-login
      
      if (!AuthService.isTokenValid()) {
        throw new Error('Token is no longer valid');
      }

      // In a real implementation, this would be:
      // const response = await axios.post('/auth/refresh', { token: currentToken });
      // AuthService.setToken(response.data.access_token, response.data.user, response.data.expires_in, true);
      
      ErrorHandler.showInfo('Session extended successfully');
    } catch (error) {
      ErrorHandler.showError('Unable to extend session. Please log in again.');
      throw error;
    }
  }, []);

  /**
   * Check and restore authentication state on app load
   * Sub-task 3.2: Add token persistence on app load
   */
  useEffect(() => {
    const restoreAuthState = () => {
      try {
        // Check if token exists and is valid
        if (AuthService.isTokenValid()) {
          const token = AuthService.getToken();
          const user = AuthService.getUser();

          if (token && user) {
            // Restore authentication state
            setAuthState({
              user,
              token,
              isAuthenticated: true,
              isLoading: false,
              error: null,
            });
            return;
          }
        }

        // Token is invalid or doesn't exist
        AuthService.removeToken();
        setAuthState(prev => ({
          ...prev,
          user: null,
          token: null,
          isAuthenticated: false,
          isLoading: false,
        }));
      } catch (error) {
        logger.error('Error restoring auth state:', error);
        setAuthState(prev => ({
          ...prev,
          user: null,
          token: null,
          isAuthenticated: false,
          isLoading: false,
        }));
      }
    };

    restoreAuthState();
  }, []);

  /**
   * Periodically check token expiration and auto-logout if expired
   * Sub-task 3.3: Implement token expiration handling
   */
  useEffect(() => {
    // Only set up expiration check if user is authenticated
    if (!authState.isAuthenticated) {
      return;
    }

    // Check token expiration every 30 seconds
    const expirationCheckInterval = setInterval(() => {
      if (!AuthService.isTokenValid()) {
        logger.log('Token expired, logging out...');
        
        // Clear token and state
        AuthService.removeToken();
        setAuthState({
          user: null,
          token: null,
          isAuthenticated: false,
          isLoading: false,
          error: 'Session expired, please log in again',
        });

        // Redirect to login
        navigate('/login');
      }
    }, 30000); // Check every 30 seconds

    // Cleanup interval on unmount or when auth state changes
    return () => {
      clearInterval(expirationCheckInterval);
    };
  }, [authState.isAuthenticated, navigate]);

  // Context value
  const contextValue: AuthContextType = {
    user: authState.user,
    token: authState.token,
    isAuthenticated: authState.isAuthenticated,
    isLoading: authState.isLoading,
    error: authState.error,
    login,
    logout,
    refreshToken,
  };

  return (
    <AuthContext.Provider value={contextValue}>
      {children}
    </AuthContext.Provider>
  );
};

export default AuthContext;

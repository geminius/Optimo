/**
 * Error Scenarios Test
 * 
 * Comprehensive test to verify all error scenarios show appropriate messages
 * as specified in Requirement 7: Error Handling and User Feedback
 */

import { render, screen, waitFor } from '@testing-library/react';
import { message } from 'antd';
import axios, { AxiosError } from 'axios';
import ErrorHandler, { ERROR_MESSAGES, ErrorType } from '../utils/errorHandler';

// Mock Ant Design message component
jest.mock('antd', () => ({
  ...jest.requireActual('antd'),
  message: {
    error: jest.fn(),
    success: jest.fn(),
    warning: jest.fn(),
    info: jest.fn(),
  },
}));

// Helper function to create Axios-like errors
const createAxiosError = (status: number, detail: string, code?: string): any => {
  const error: any = new Error('Axios Error');
  error.isAxiosError = true;
  error.response = {
    status,
    data: { detail },
  };
  if (code) {
    error.code = code;
  }
  return error;
};

describe('Error Scenarios - All error scenarios show appropriate messages', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    // Mock axios.isAxiosError to properly detect our test errors
    jest.spyOn(axios, 'isAxiosError').mockImplementation((error: any) => {
      return error && error.isAxiosError === true;
    });
  });

  afterEach(() => {
    jest.restoreAllMocks();
  });

  describe('Requirement 7.1: Invalid Credentials Error', () => {
    it('should display "Invalid username or password" for invalid credentials', () => {
      const error = createAxiosError(401, 'Invalid credentials');

      const errorDetails = ErrorHandler.parseError(error);
      
      expect(errorDetails.type).toBe(ErrorType.AUTHENTICATION);
      expect(errorDetails.message).toBe(ERROR_MESSAGES.AUTH_INVALID_CREDENTIALS);
      expect(errorDetails.statusCode).toBe(401);
    });

    it('should display error message when login fails with incorrect password', () => {
      const error = createAxiosError(401, 'Incorrect username or password');

      ErrorHandler.showError(error);
      
      expect(message.error).toHaveBeenCalledWith(
        ERROR_MESSAGES.AUTH_INVALID_CREDENTIALS,
        5
      );
    });
  });

  describe('Requirement 7.2: Network Error', () => {
    it('should display "Unable to connect to server" for network errors', () => {
      const error: any = new Error('Network Error');
      error.isAxiosError = true;
      error.code = 'ERR_NETWORK';

      const errorDetails = ErrorHandler.parseError(error);
      
      expect(errorDetails.type).toBe(ErrorType.NETWORK);
      expect(errorDetails.message).toBe(ERROR_MESSAGES.NETWORK_CONNECTION_FAILED);
    });

    it('should display network error message when server is unreachable', () => {
      const error: any = new Error('Network Error');
      error.isAxiosError = true;

      ErrorHandler.showError(error);
      
      expect(message.error).toHaveBeenCalledWith(
        ERROR_MESSAGES.NETWORK_CONNECTION_FAILED,
        5
      );
    });

    it('should display timeout error for request timeouts', () => {
      const error: any = new Error('timeout of 5000ms exceeded');
      error.isAxiosError = true;
      error.code = 'ECONNABORTED';

      const errorDetails = ErrorHandler.parseError(error);
      
      expect(errorDetails.type).toBe(ErrorType.NETWORK);
      expect(errorDetails.message).toBe(ERROR_MESSAGES.NETWORK_TIMEOUT);
    });

    it('should display offline error when no internet connection', () => {
      // Mock navigator.onLine
      Object.defineProperty(navigator, 'onLine', {
        writable: true,
        value: false,
      });

      const error: any = new Error('Network Error');
      error.isAxiosError = true;

      const errorDetails = ErrorHandler.parseError(error);
      
      expect(errorDetails.type).toBe(ErrorType.NETWORK);
      expect(errorDetails.message).toBe(ERROR_MESSAGES.NETWORK_OFFLINE);

      // Restore navigator.onLine
      Object.defineProperty(navigator, 'onLine', {
        writable: true,
        value: true,
      });
    });
  });

  describe('Requirement 7.3: 401 Token Expired Error', () => {
    it('should display "Session expired, please log in again" for expired tokens', () => {
      const error = createAxiosError(401, 'Token has expired');

      const errorDetails = ErrorHandler.parseError(error);
      
      expect(errorDetails.type).toBe(ErrorType.AUTHENTICATION);
      expect(errorDetails.message).toBe(ERROR_MESSAGES.AUTH_TOKEN_EXPIRED);
      expect(errorDetails.statusCode).toBe(401);
    });

    it('should display session expired message when API returns 401', () => {
      const error = createAxiosError(401, 'Token expired');

      ErrorHandler.showError(error);
      
      expect(message.error).toHaveBeenCalledWith(
        ERROR_MESSAGES.AUTH_TOKEN_EXPIRED,
        5
      );
    });
  });

  describe('Requirement 7.4: 403 Forbidden Error', () => {
    it('should display "You don\'t have permission to perform this action" for 403 errors', () => {
      const error = createAxiosError(403, 'Insufficient permissions');

      const errorDetails = ErrorHandler.parseError(error);
      
      expect(errorDetails.type).toBe(ErrorType.AUTHORIZATION);
      expect(errorDetails.message).toBe(ERROR_MESSAGES.AUTH_INSUFFICIENT_PERMISSIONS);
      expect(errorDetails.statusCode).toBe(403);
    });

    it('should display permission denied message when API returns 403', () => {
      const error = createAxiosError(403, 'Access denied');

      ErrorHandler.showError(error);
      
      expect(message.error).toHaveBeenCalledWith(
        ERROR_MESSAGES.AUTH_INSUFFICIENT_PERMISSIONS,
        5
      );
    });
  });

  describe('Additional Error Scenarios', () => {
    it('should display server error message for 500 errors', () => {
      const error = createAxiosError(500, 'Internal server error');

      const errorDetails = ErrorHandler.parseError(error);
      
      expect(errorDetails.type).toBe(ErrorType.SERVER);
      expect(errorDetails.message).toBe(ERROR_MESSAGES.SERVER_ERROR);
      expect(errorDetails.statusCode).toBe(500);
    });

    it('should display validation error message for 400 errors', () => {
      const error = createAxiosError(400, 'Invalid input data');

      const errorDetails = ErrorHandler.parseError(error);
      
      expect(errorDetails.type).toBe(ErrorType.VALIDATION);
      expect(errorDetails.message).toBe('Invalid input data');
      expect(errorDetails.statusCode).toBe(400);
    });

    it('should handle unknown errors gracefully', () => {
      const error = new Error('Something went wrong');

      const errorDetails = ErrorHandler.parseError(error);
      
      expect(errorDetails.type).toBe(ErrorType.UNKNOWN);
      expect(errorDetails.message).toBe('Something went wrong');
    });

    it('should handle string errors', () => {
      const error = 'Custom error message';

      const errorDetails = ErrorHandler.parseError(error);
      
      expect(errorDetails.type).toBe(ErrorType.UNKNOWN);
      expect(errorDetails.message).toBe('Custom error message');
    });
  });

  describe('ErrorHandler Methods', () => {
    it('should show success messages', () => {
      ErrorHandler.showSuccess('Operation successful');
      
      expect(message.success).toHaveBeenCalledWith('Operation successful', 3);
    });

    it('should show warning messages', () => {
      ErrorHandler.showWarning('Warning message');
      
      expect(message.warning).toHaveBeenCalledWith('Warning message', 4);
    });

    it('should show info messages', () => {
      ErrorHandler.showInfo('Info message');
      
      expect(message.info).toHaveBeenCalledWith('Info message', 3);
    });

    it('should log errors with context', () => {
      const consoleSpy = jest.spyOn(console, 'error').mockImplementation();
      
      const error = new Error('Test error');
      ErrorHandler.logError(error, { component: 'TestComponent' });
      
      expect(consoleSpy).toHaveBeenCalled();
      
      consoleSpy.mockRestore();
    });

    it('should handle auth errors specifically', () => {
      const error = createAxiosError(401, 'Invalid credentials');

      const errorDetails = ErrorHandler.handleAuthError(error, true);
      
      expect(errorDetails.type).toBe(ErrorType.AUTHENTICATION);
      expect(message.error).toHaveBeenCalled();
    });

    it('should handle API errors specifically', () => {
      const error = createAxiosError(500, 'Server error');

      const errorDetails = ErrorHandler.handleApiError(error, true);
      
      expect(errorDetails.type).toBe(ErrorType.SERVER);
      expect(message.error).toHaveBeenCalled();
    });
  });

  describe('Error Message Constants', () => {
    it('should have all required error messages defined', () => {
      expect(ERROR_MESSAGES.AUTH_INVALID_CREDENTIALS).toBe('Invalid username or password');
      expect(ERROR_MESSAGES.AUTH_TOKEN_EXPIRED).toBe('Session expired, please log in again');
      expect(ERROR_MESSAGES.AUTH_INSUFFICIENT_PERMISSIONS).toBe("You don't have permission to perform this action");
      expect(ERROR_MESSAGES.NETWORK_CONNECTION_FAILED).toBe('Unable to connect to server');
      expect(ERROR_MESSAGES.SERVER_ERROR).toBe('Server error, please try again');
    });
  });
});

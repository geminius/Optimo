/**
 * Error Handler Utility
 * 
 * Provides centralized error handling and user-friendly error message mapping
 * for authentication and API errors throughout the application.
 */

import { message } from 'antd';
import axios from 'axios';
import { logger } from './logger';

/**
 * Error types for categorization
 */
export enum ErrorType {
  AUTHENTICATION = 'authentication',
  AUTHORIZATION = 'authorization',
  NETWORK = 'network',
  SERVER = 'server',
  VALIDATION = 'validation',
  UNKNOWN = 'unknown',
}

/**
 * Error message mapping for common error scenarios
 */
export const ERROR_MESSAGES = {
  // Authentication errors (401)
  AUTH_INVALID_CREDENTIALS: 'Invalid username or password',
  AUTH_TOKEN_EXPIRED: 'Session expired, please log in again',
  AUTH_TOKEN_INVALID: 'Invalid authentication token',
  AUTH_REQUIRED: 'Authentication required',
  
  // Authorization errors (403)
  AUTH_INSUFFICIENT_PERMISSIONS: "You don't have permission to perform this action",
  AUTH_FORBIDDEN: 'Access denied',
  
  // Network errors
  NETWORK_CONNECTION_FAILED: 'Unable to connect to server',
  NETWORK_TIMEOUT: 'Request timed out, please try again',
  NETWORK_OFFLINE: 'No internet connection',
  
  // Server errors (500)
  SERVER_ERROR: 'Server error, please try again',
  SERVER_UNAVAILABLE: 'Service temporarily unavailable',
  
  // Validation errors (400)
  VALIDATION_ERROR: 'Invalid input, please check your data',
  VALIDATION_REQUIRED_FIELD: 'Required field is missing',
  
  // Generic errors
  UNKNOWN_ERROR: 'An unexpected error occurred',
  OPERATION_FAILED: 'Operation failed, please try again',
} as const;

/**
 * Error details interface
 */
export interface ErrorDetails {
  type: ErrorType;
  message: string;
  statusCode?: number;
  originalError?: unknown;
}

/**
 * Parse and categorize errors from various sources
 */
export class ErrorHandler {
  /**
   * Parse an error and return structured error details
   * @param error - Error object from catch block
   * @returns Structured error details
   */
  static parseError(error: unknown): ErrorDetails {
    // Handle Axios errors
    if (axios.isAxiosError(error)) {
      const statusCode = error.response?.status;
      
      // Authentication errors (401)
      if (statusCode === 401) {
        const errorData = error.response?.data as { detail?: string; message?: string } | undefined;
        const errorMessage = errorData?.detail || errorData?.message;
        
        // Check if it's a token expiration
        if (errorMessage?.toLowerCase().includes('expired')) {
          return {
            type: ErrorType.AUTHENTICATION,
            message: ERROR_MESSAGES.AUTH_TOKEN_EXPIRED,
            statusCode,
            originalError: error,
          };
        }
        
        // Check if it's invalid credentials
        if (errorMessage?.toLowerCase().includes('invalid') || 
            errorMessage?.toLowerCase().includes('incorrect')) {
          return {
            type: ErrorType.AUTHENTICATION,
            message: ERROR_MESSAGES.AUTH_INVALID_CREDENTIALS,
            statusCode,
            originalError: error,
          };
        }
        
        return {
          type: ErrorType.AUTHENTICATION,
          message: ERROR_MESSAGES.AUTH_REQUIRED,
          statusCode,
          originalError: error,
        };
      }
      
      // Authorization errors (403)
      if (statusCode === 403) {
        return {
          type: ErrorType.AUTHORIZATION,
          message: ERROR_MESSAGES.AUTH_INSUFFICIENT_PERMISSIONS,
          statusCode,
          originalError: error,
        };
      }
      
      // Validation errors (400)
      if (statusCode === 400) {
        const errorData = error.response?.data as { detail?: string; message?: string } | undefined;
        const errorMessage = errorData?.detail || errorData?.message;
        return {
          type: ErrorType.VALIDATION,
          message: errorMessage || ERROR_MESSAGES.VALIDATION_ERROR,
          statusCode,
          originalError: error,
        };
      }
      
      // Server errors (500+)
      if (statusCode && statusCode >= 500) {
        return {
          type: ErrorType.SERVER,
          message: ERROR_MESSAGES.SERVER_ERROR,
          statusCode,
          originalError: error,
        };
      }
      
      // Network errors (no response)
      if (!error.response) {
        // Check if it's a timeout
        if (error.code === 'ECONNABORTED' || error.message?.includes('timeout')) {
          return {
            type: ErrorType.NETWORK,
            message: ERROR_MESSAGES.NETWORK_TIMEOUT,
            originalError: error,
          };
        }
        
        // Check if offline
        if (!navigator.onLine) {
          return {
            type: ErrorType.NETWORK,
            message: ERROR_MESSAGES.NETWORK_OFFLINE,
            originalError: error,
          };
        }
        
        return {
          type: ErrorType.NETWORK,
          message: ERROR_MESSAGES.NETWORK_CONNECTION_FAILED,
          originalError: error,
        };
      }
    }
    
    // Handle Error objects
    if (error instanceof Error) {
      return {
        type: ErrorType.UNKNOWN,
        message: error.message || ERROR_MESSAGES.UNKNOWN_ERROR,
        originalError: error,
      };
    }
    
    // Handle string errors
    if (typeof error === 'string') {
      return {
        type: ErrorType.UNKNOWN,
        message: error,
        originalError: error,
      };
    }
    
    // Unknown error type
    return {
      type: ErrorType.UNKNOWN,
      message: ERROR_MESSAGES.UNKNOWN_ERROR,
      originalError: error,
    };
  }

  /**
   * Display error message using Ant Design message component
   * @param error - Error object or error details
   * @param duration - Duration to show message (seconds), default 5
   */
  static showError(error: unknown | ErrorDetails, duration: number = 5): void {
    const errorDetails = 'type' in (error as ErrorDetails) 
      ? (error as ErrorDetails)
      : this.parseError(error);
    
    message.error(errorDetails.message, duration);
  }

  /**
   * Display success message using Ant Design message component
   * @param msg - Success message to display
   * @param duration - Duration to show message (seconds), default 3
   */
  static showSuccess(msg: string, duration: number = 3): void {
    message.success(msg, duration);
  }

  /**
   * Display warning message using Ant Design message component
   * @param msg - Warning message to display
   * @param duration - Duration to show message (seconds), default 4
   */
  static showWarning(msg: string, duration: number = 4): void {
    message.warning(msg, duration);
  }

  /**
   * Display info message using Ant Design message component
   * @param msg - Info message to display
   * @param duration - Duration to show message (seconds), default 3
   */
  static showInfo(msg: string, duration: number = 3): void {
    message.info(msg, duration);
  }

  /**
   * Log error to console with structured format
   * @param error - Error object or error details
   * @param context - Additional context information
   */
  static logError(error: unknown | ErrorDetails, context?: Record<string, unknown>): void {
    const errorDetails = 'type' in (error as ErrorDetails)
      ? (error as ErrorDetails)
      : this.parseError(error);
    
    logger.error('[Error Handler]', {
      type: errorDetails.type,
      message: errorDetails.message,
      statusCode: errorDetails.statusCode,
      context,
      originalError: errorDetails.originalError,
      timestamp: new Date().toISOString(),
    });
  }

  /**
   * Handle authentication errors specifically
   * @param error - Error object
   * @param showMessage - Whether to display error message (default true)
   * @returns Error details
   */
  static handleAuthError(error: unknown, showMessage: boolean = true): ErrorDetails {
    const errorDetails = this.parseError(error);
    
    if (showMessage) {
      this.showError(errorDetails);
    }
    
    this.logError(errorDetails, { component: 'Authentication' });
    
    return errorDetails;
  }

  /**
   * Handle API errors specifically
   * @param error - Error object
   * @param showMessage - Whether to display error message (default true)
   * @returns Error details
   */
  static handleApiError(error: unknown, showMessage: boolean = true): ErrorDetails {
    const errorDetails = this.parseError(error);
    
    if (showMessage) {
      this.showError(errorDetails);
    }
    
    this.logError(errorDetails, { component: 'API' });
    
    return errorDetails;
  }
}

export default ErrorHandler;

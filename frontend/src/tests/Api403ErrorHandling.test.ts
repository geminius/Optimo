/**
 * API 403 Forbidden Error Handling Tests
 * 
 * Verifies that 403 responses show appropriate error messages
 * without redirecting or clearing authentication tokens.
 * 
 * Requirements tested:
 * - Requirement 3.3: Display appropriate error message for 403
 * - Requirement 7.4: Show "You don't have permission" message
 */

// Mock ErrorHandler.showError BEFORE imports
jest.mock('../utils/errorHandler', () => ({
  __esModule: true,
  default: {
    showError: jest.fn(),
    showSuccess: jest.fn(),
    showInfo: jest.fn(),
    parseError: jest.fn(() => ({ type: 'AUTHORIZATION' })),
  },
  ERROR_MESSAGES: {
    AUTH_TOKEN_EXPIRED: 'Session expired, please log in again',
    AUTH_INSUFFICIENT_PERMISSIONS: "You don't have permission to perform this action",
    SERVER_ERROR: 'Server error, please try again',
    NETWORK_CONNECTION_FAILED: 'Unable to connect to server',
  },
}));

import axios from 'axios';
import MockAdapter from 'axios-mock-adapter';
import AuthService from '../services/auth';
import { User } from '../types/auth';
import ErrorHandler from '../utils/errorHandler';

// Create a mock location object
const mockLocation = {
  href: '',
  assign: jest.fn(),
  reload: jest.fn(),
  replace: jest.fn(),
  pathname: '/',
  search: '',
  hash: '',
  origin: 'http://localhost',
  protocol: 'http:',
  host: 'localhost',
  hostname: 'localhost',
  port: '',
};

// Mock window.location
delete (window as any).location;
(window as any).location = mockLocation;

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

// Create test axios instance with the same interceptors as the real api service
const testApi = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Add request interceptor
testApi.interceptors.request.use(
  (config) => {
    const token = AuthService.getToken();
    if (token && config.headers) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => Promise.reject(error)
);

// Add response interceptor (same as in api.ts)
testApi.interceptors.response.use(
  (response) => response,
  (error) => {
    if (axios.isAxiosError(error)) {
      if (error.response?.status === 401) {
        AuthService.removeToken();
        ErrorHandler.showError({
          type: ErrorHandler.parseError(error).type,
          message: 'Session expired, please log in again',
          statusCode: 401,
          originalError: error,
        });
        window.location.href = '/login';
      } else if (error.response?.status === 403) {
        ErrorHandler.showError({
          type: ErrorHandler.parseError(error).type,
          message: "You don't have permission to perform this action",
          statusCode: 403,
          originalError: error,
        });
      } else if (error.response?.status && error.response.status >= 500) {
        ErrorHandler.showError({
          type: ErrorHandler.parseError(error).type,
          message: 'Server error, please try again',
          statusCode: error.response.status,
          originalError: error,
        });
      } else if (!error.response) {
        ErrorHandler.showError({
          type: ErrorHandler.parseError(error).type,
          message: 'Unable to connect to server',
          originalError: error,
        });
      }
    }
    return Promise.reject(error);
  }
);

const mock = new MockAdapter(testApi);

const createMockUser = (): User => ({
  id: 'test-user-123',
  username: 'testuser',
  role: 'user',
  email: 'test@example.com',
});

describe('API 403 Forbidden Error Handling', () => {
  beforeEach(() => {
    mock.reset();
    mock.resetHistory();
    localStorage.clear();
    sessionStorage.clear();
    mockLocation.href = '';
    jest.clearAllMocks();
  });

  afterAll(() => {
    mock.restore();
  });

  describe('403 Error Display', () => {
    it('should show error message when API returns 403', async () => {
      // Arrange - Set up authenticated user
      const mockToken = 'valid-token-123';
      const mockUser = createMockUser();
      AuthService.setToken(mockToken, mockUser, 3600, true);

      // Mock 403 response
      mock.onGet('/admin/users').reply(403, { detail: 'Insufficient permissions' });

      // Act - Make request that returns 403
      try {
        await testApi.get('/admin/users');
        fail('Expected request to throw error');
      } catch (error) {
        // Expected to throw
      }

      // Assert - Error message shown
      expect(ErrorHandler.showError).toHaveBeenCalledWith(
        expect.objectContaining({
          statusCode: 403,
          message: "You don't have permission to perform this action",
        })
      );
    });

    it('should show error message for 403 on POST requests', async () => {
      const mockToken = 'valid-token-456';
      const mockUser = createMockUser();
      AuthService.setToken(mockToken, mockUser, 3600, true);

      mock.onPost('/admin/settings').reply(403, { detail: 'Admin access required' });

      try {
        await testApi.post('/admin/settings', { setting: 'value' });
        fail('Expected request to throw error');
      } catch (error) {
        // Expected
      }

      expect(ErrorHandler.showError).toHaveBeenCalledWith(
        expect.objectContaining({
          statusCode: 403,
          message: "You don't have permission to perform this action",
        })
      );
    });

    it('should show error message for 403 on DELETE requests', async () => {
      const mockToken = 'valid-token-789';
      const mockUser = createMockUser();
      AuthService.setToken(mockToken, mockUser, 3600, true);

      mock.onDelete('/models/protected-model').reply(403);

      try {
        await testApi.delete('/models/protected-model');
        fail('Expected request to throw error');
      } catch (error) {
        // Expected
      }

      expect(ErrorHandler.showError).toHaveBeenCalledWith(
        expect.objectContaining({
          statusCode: 403,
          message: "You don't have permission to perform this action",
        })
      );
    });

    it('should show error message for 403 on PUT requests', async () => {
      const mockToken = 'valid-token-abc';
      const mockUser = createMockUser();
      AuthService.setToken(mockToken, mockUser, 3600, true);

      mock.onPut('/config/system').reply(403);

      try {
        await testApi.put('/config/system', {});
        fail('Expected request to throw error');
      } catch (error) {
        // Expected
      }

      expect(ErrorHandler.showError).toHaveBeenCalledWith(
        expect.objectContaining({
          statusCode: 403,
          message: "You don't have permission to perform this action",
        })
      );
    });
  });

  describe('403 vs 401 Behavior', () => {
    it('should NOT redirect to login on 403 (unlike 401)', async () => {
      const mockToken = 'valid-token-123';
      const mockUser = createMockUser();
      AuthService.setToken(mockToken, mockUser, 3600, true);

      mock.onGet('/admin/users').reply(403);

      try {
        await testApi.get('/admin/users');
        fail('Expected request to throw error');
      } catch (error) {
        // Expected
      }

      // Should NOT redirect
      expect(mockLocation.href).toBe('');
    });

    it('should NOT clear token on 403 (unlike 401)', async () => {
      const mockToken = 'valid-token-456';
      const mockUser = createMockUser();
      AuthService.setToken(mockToken, mockUser, 3600, true);

      mock.onGet('/admin/users').reply(403);

      try {
        await testApi.get('/admin/users');
        fail('Expected request to throw error');
      } catch (error) {
        // Expected
      }

      // Token should still be present
      expect(AuthService.getToken()).toBe(mockToken);
    });

    it('should allow user to continue after 403 error', async () => {
      const mockToken = 'valid-token-789';
      const mockUser = createMockUser();
      AuthService.setToken(mockToken, mockUser, 3600, true);

      // First request returns 403
      mock.onGet('/admin/users').reply(403);
      
      // Second request succeeds
      mock.onGet('/models').reply(200, [{ id: 'model-1', name: 'Test Model' }]);

      // First request fails with 403
      try {
        await testApi.get('/admin/users');
        fail('Expected request to throw error');
      } catch (error) {
        // Expected
      }

      // User should still be able to make other requests
      const response = await testApi.get('/models');
      expect(response.status).toBe(200);
      expect(response.data).toHaveLength(1);
    });
  });

  describe('Multiple 403 Errors', () => {
    it('should handle multiple 403 responses correctly', async () => {
      const mockToken = 'valid-token-multi';
      const mockUser = createMockUser();
      AuthService.setToken(mockToken, mockUser, 3600, true);

      mock.onGet('/admin/users').reply(403);
      mock.onGet('/admin/settings').reply(403);

      // First 403
      try {
        await testApi.get('/admin/users');
      } catch (error) {
        // Expected
      }

      expect(ErrorHandler.showError).toHaveBeenCalledTimes(1);
      expect(AuthService.getToken()).toBe(mockToken);

      // Second 403
      try {
        await testApi.get('/admin/settings');
      } catch (error) {
        // Expected
      }

      expect(ErrorHandler.showError).toHaveBeenCalledTimes(2);
      expect(AuthService.getToken()).toBe(mockToken);
      expect(mockLocation.href).toBe('');
    });
  });

  describe('Error Message Content', () => {
    it('should display the correct permission denied message', async () => {
      const mockToken = 'valid-token-123';
      const mockUser = createMockUser();
      AuthService.setToken(mockToken, mockUser, 3600, true);

      mock.onGet('/admin/users').reply(403);

      try {
        await testApi.get('/admin/users');
      } catch (error) {
        // Expected
      }

      const errorCall = (ErrorHandler.showError as jest.Mock).mock.calls[0][0];
      expect(errorCall.message).toBe("You don't have permission to perform this action");
      expect(errorCall.statusCode).toBe(403);
    });
  });
});

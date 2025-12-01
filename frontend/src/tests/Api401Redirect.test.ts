// Mock ErrorHandler.showError BEFORE imports
jest.mock('../utils/errorHandler', () => ({
  __esModule: true,
  default: {
    showError: jest.fn(),
    parseError: jest.fn(() => ({ type: 'AUTH_ERROR' })),
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

describe('API 401 Response Redirect', () => {
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

  it('should redirect to /login when API returns 401', async () => {
    const mockToken = 'expired-token-123';
    const mockUser = createMockUser();
    AuthService.setToken(mockToken, mockUser, 3600, true);

    expect(AuthService.getToken()).toBe(mockToken);

    mock.onGet('/models').reply(401, { detail: 'Token expired' });

    try {
      await testApi.get('/models');
      fail('Expected request to throw error');
    } catch (error) {
      // Expected to throw
    }

    expect(AuthService.getToken()).toBeNull();
    expect(mockLocation.href).toBe('/login');
    expect(ErrorHandler.showError).toHaveBeenCalledWith(
      expect.objectContaining({
        statusCode: 401,
        message: 'Session expired, please log in again',
      })
    );
  });

  it('should redirect to /login on 401 for POST requests', async () => {
    const mockToken = 'expired-token-456';
    const mockUser = createMockUser();
    AuthService.setToken(mockToken, mockUser, 3600, true);

    mock.onPost('/models/test-model/optimize').reply(401, { detail: 'Invalid token' });

    try {
      await testApi.post('/models/test-model/optimize', { criteria: {} });
      fail('Expected request to throw error');
    } catch (error) {
      // Expected to throw
    }

    expect(AuthService.getToken()).toBeNull();
    expect(mockLocation.href).toBe('/login');
  });

  it('should redirect to /login on 401 for DELETE requests', async () => {
    const mockToken = 'expired-token-789';
    const mockUser = createMockUser();
    AuthService.setToken(mockToken, mockUser, 3600, true);

    mock.onDelete('/models/test-model').reply(401);

    try {
      await testApi.delete('/models/test-model');
      fail('Expected request to throw error');
    } catch (error) {
      // Expected to throw
    }

    expect(AuthService.getToken()).toBeNull();
    expect(mockLocation.href).toBe('/login');
  });

  it('should redirect to /login on 401 for PUT requests', async () => {
    const mockToken = 'expired-token-abc';
    const mockUser = createMockUser();
    AuthService.setToken(mockToken, mockUser, 3600, true);

    mock.onPut('/config/optimization-criteria').reply(401);

    try {
      await testApi.put('/config/optimization-criteria', {});
      fail('Expected request to throw error');
    } catch (error) {
      // Expected to throw
    }

    expect(AuthService.getToken()).toBeNull();
    expect(mockLocation.href).toBe('/login');
  });

  it('should NOT redirect to /login on 403 Forbidden', async () => {
    const mockToken = 'valid-token-123';
    const mockUser = createMockUser();
    AuthService.setToken(mockToken, mockUser, 3600, true);

    mock.onGet('/admin/users').reply(403, { detail: 'Insufficient permissions' });

    try {
      await testApi.get('/admin/users');
      fail('Expected request to throw error');
    } catch (error) {
      // Expected to throw
    }

    expect(AuthService.getToken()).toBe(mockToken);
    expect(mockLocation.href).toBe('');
    expect(ErrorHandler.showError).toHaveBeenCalledWith(
      expect.objectContaining({
        statusCode: 403,
        message: "You don't have permission to perform this action",
      })
    );
  });

  it('should NOT redirect to /login on 500 Server Error', async () => {
    const mockToken = 'valid-token-456';
    const mockUser = createMockUser();
    AuthService.setToken(mockToken, mockUser, 3600, true);

    mock.onGet('/models').reply(500, { detail: 'Internal server error' });

    try {
      await testApi.get('/models');
      fail('Expected request to throw error');
    } catch (error) {
      // Expected to throw
    }

    expect(AuthService.getToken()).toBe(mockToken);
    expect(mockLocation.href).toBe('');
    expect(ErrorHandler.showError).toHaveBeenCalledWith(
      expect.objectContaining({
        statusCode: 500,
        message: 'Server error, please try again',
      })
    );
  });

  it('should handle multiple 401 responses correctly', async () => {
    const mockToken = 'expired-token-multi';
    const mockUser = createMockUser();
    AuthService.setToken(mockToken, mockUser, 3600, true);

    mock.onGet('/models').reply(401);
    mock.onGet('/optimization/sessions').reply(401);

    try {
      await testApi.get('/models');
    } catch (error) {
      // Expected
    }

    expect(mockLocation.href).toBe('/login');
    expect(AuthService.getToken()).toBeNull();

    mockLocation.href = '';

    try {
      await testApi.get('/optimization/sessions');
    } catch (error) {
      // Expected
    }

    expect(mockLocation.href).toBe('/login');
  });

  it('should clear token before redirecting on 401', async () => {
    const mockToken = 'token-to-clear';
    const mockUser = createMockUser();
    AuthService.setToken(mockToken, mockUser, 3600, true);

    mock.onGet('/models').reply(401);

    try {
      await testApi.get('/models');
    } catch (error) {
      // Expected
    }

    expect(AuthService.getToken()).toBeNull();
    expect(mockLocation.href).toBe('/login');
  });
});

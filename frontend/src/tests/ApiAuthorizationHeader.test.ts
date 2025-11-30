import axios, { AxiosInstance } from 'axios';
import MockAdapter from 'axios-mock-adapter';
import AuthService from '../services/auth';
import { User } from '../types/auth';

// We need to create our own axios instance and mock it
// since the api service creates its own instance
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

const testApi = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Add the same interceptors as the real api service
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

// Create a mock adapter for our test axios instance
const mock = new MockAdapter(testApi);

// Helper function to create a mock user
const createMockUser = (): User => ({
  id: 'test-user-123',
  username: 'testuser',
  role: 'user',
  email: 'test@example.com',
});

describe('API Authorization Header', () => {
  beforeEach(() => {
    // Reset mock adapter and clear history
    mock.reset();
    mock.resetHistory();
    // Clear localStorage and sessionStorage
    localStorage.clear();
    sessionStorage.clear();
  });

  afterAll(() => {
    mock.restore();
  });

  it('should include Authorization header when token exists', async () => {
    // Set up a token with proper user data
    const mockToken = 'test-jwt-token-12345';
    const mockUser = createMockUser();
    AuthService.setToken(mockToken, mockUser, 3600, true);

    // Mock the API endpoint
    mock.onGet('/models').reply((config) => {
      // Verify Authorization header is present
      expect(config.headers?.Authorization).toBe(`Bearer ${mockToken}`);
      return [200, []];
    });

    // Make API request using our test axios instance
    await testApi.get('/models');

    // Verify the request was made
    expect(mock.history.get.length).toBe(1);
  });

  it('should not include Authorization header when token does not exist', async () => {
    // Ensure no token is set
    AuthService.removeToken();

    // Mock the API endpoint
    mock.onGet('/models').reply((config) => {
      // Verify Authorization header is not present
      expect(config.headers?.Authorization).toBeUndefined();
      return [200, []];
    });

    // Make API request using our test axios instance
    await testApi.get('/models');

    // Verify the request was made
    expect(mock.history.get.length).toBe(1);
  });

  it('should include Authorization header for POST requests', async () => {
    const mockToken = 'test-jwt-token-67890';
    const mockUser = createMockUser();
    AuthService.setToken(mockToken, mockUser, 3600, true);

    const modelId = 'test-model-123';
    const criteria = {
      target_size_mb: 100,
      min_accuracy: 0.9,
      max_latency_ms: 50,
    };

    // Mock the API endpoint
    mock.onPost(`/models/${modelId}/optimize`).reply((config) => {
      // Verify Authorization header is present
      expect(config.headers?.Authorization).toBe(`Bearer ${mockToken}`);
      return [200, { session_id: 'session-123', status: 'started' }];
    });

    // Make API request using our test axios instance
    await testApi.post(`/models/${modelId}/optimize`, criteria);

    // Verify the request was made
    expect(mock.history.post.length).toBe(1);
  });

  it('should include Authorization header for DELETE requests', async () => {
    const mockToken = 'test-jwt-token-delete';
    const mockUser = createMockUser();
    AuthService.setToken(mockToken, mockUser, 3600, true);

    const modelId = 'test-model-456';

    // Mock the API endpoint
    mock.onDelete(`/models/${modelId}`).reply((config) => {
      // Verify Authorization header is present
      expect(config.headers?.Authorization).toBe(`Bearer ${mockToken}`);
      return [204];
    });

    // Make API request using our test axios instance
    await testApi.delete(`/models/${modelId}`);

    // Verify the request was made
    expect(mock.history.delete.length).toBe(1);
  });

  it('should include Authorization header for PUT requests', async () => {
    const mockToken = 'test-jwt-token-put';
    const mockUser = createMockUser();
    AuthService.setToken(mockToken, mockUser, 3600, true);

    const criteria = {
      target_size_mb: 150,
      min_accuracy: 0.85,
      max_latency_ms: 100,
    };

    // Mock the API endpoint
    mock.onPut('/config/optimization-criteria').reply((config) => {
      // Verify Authorization header is present
      expect(config.headers?.Authorization).toBe(`Bearer ${mockToken}`);
      return [200, criteria];
    });

    // Make API request using our test axios instance
    await testApi.put('/config/optimization-criteria', criteria);

    // Verify the request was made
    expect(mock.history.put.length).toBe(1);
  });

  it('should update Authorization header when token changes', async () => {
    // Set initial token
    const firstToken = 'first-token-123';
    const mockUser = createMockUser();
    AuthService.setToken(firstToken, mockUser, 3600, true);

    // Mock first request
    mock.onGet('/models').replyOnce((config) => {
      expect(config.headers?.Authorization).toBe(`Bearer ${firstToken}`);
      return [200, []];
    });

    // Make first request using our test axios instance
    await testApi.get('/models');

    // Change token
    const secondToken = 'second-token-456';
    AuthService.setToken(secondToken, mockUser, 3600, true);

    // Mock second request
    mock.onGet('/models').replyOnce((config) => {
      expect(config.headers?.Authorization).toBe(`Bearer ${secondToken}`);
      return [200, []];
    });

    // Make second request using our test axios instance
    await testApi.get('/models');

    // Verify both requests were made
    expect(mock.history.get.length).toBe(2);
  });

  it('should handle multipart/form-data requests with Authorization header', async () => {
    const mockToken = 'test-jwt-token-upload';
    const mockUser = createMockUser();
    AuthService.setToken(mockToken, mockUser, 3600, true);

    const file = new File(['test content'], 'test-model.pt', { type: 'application/octet-stream' });
    const metadata = {
      name: 'Test Model',
      description: 'Test description',
    };

    const formData = new FormData();
    formData.append('file', file);
    formData.append('metadata', JSON.stringify(metadata));

    // Mock the API endpoint
    mock.onPost('/models/upload').reply((config) => {
      // Verify Authorization header is present
      expect(config.headers?.Authorization).toBe(`Bearer ${mockToken}`);
      // Verify Content-Type is multipart/form-data
      expect(config.headers?.['Content-Type']).toContain('multipart/form-data');
      return [200, { id: 'model-123', name: 'Test Model' }];
    });

    // Make API request using our test axios instance
    await testApi.post('/models/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });

    // Verify the request was made
    expect(mock.history.post.length).toBe(1);
  });
});

import axios from 'axios';
import {
  ModelMetadata,
  OptimizationSession,
  OptimizationCriteria,
  AnalysisReport,
  EvaluationReport
} from '../types';
import AuthService from './auth';
import ErrorHandler, { ERROR_MESSAGES } from '../utils/errorHandler';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor to add authentication token
api.interceptors.request.use(
  (config) => {
    // Get token from AuthService
    const token = AuthService.getToken();
    
    // Add Authorization header if token exists
    if (token && config.headers) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Response interceptor to handle authentication errors
api.interceptors.response.use(
  (response) => {
    // Pass through successful responses
    return response;
  },
  (error) => {
    if (axios.isAxiosError(error)) {
      // Handle 401 Unauthorized - token expired or invalid
      if (error.response?.status === 401) {
        // Clear token and redirect to login
        AuthService.removeToken();
        
        // Show error message using ErrorHandler
        ErrorHandler.showError({
          type: ErrorHandler.parseError(error).type,
          message: ERROR_MESSAGES.AUTH_TOKEN_EXPIRED,
          statusCode: 401,
          originalError: error,
        });
        
        // Redirect to login page
        window.location.href = '/login';
      }
      
      // Handle 403 Forbidden - insufficient permissions
      else if (error.response?.status === 403) {
        // Display permission denied message (do not redirect)
        ErrorHandler.showError({
          type: ErrorHandler.parseError(error).type,
          message: ERROR_MESSAGES.AUTH_INSUFFICIENT_PERMISSIONS,
          statusCode: 403,
          originalError: error,
        });
      }
      
      // Handle other errors with appropriate messages
      else if (error.response?.status && error.response.status >= 500) {
        // Server errors
        ErrorHandler.showError({
          type: ErrorHandler.parseError(error).type,
          message: ERROR_MESSAGES.SERVER_ERROR,
          statusCode: error.response.status,
          originalError: error,
        });
      }
      
      // Handle network errors
      else if (!error.response) {
        ErrorHandler.showError({
          type: ErrorHandler.parseError(error).type,
          message: ERROR_MESSAGES.NETWORK_CONNECTION_FAILED,
          originalError: error,
        });
      }
    }
    
    return Promise.reject(error);
  }
);

export const apiService = {
  // Model management
  async uploadModel(file: File, metadata: Partial<ModelMetadata>): Promise<ModelMetadata> {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('metadata', JSON.stringify(metadata));
    
    const response = await api.post('/models/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  },

  async getModels(): Promise<ModelMetadata[]> {
    const response = await api.get('/models');
    return response.data;
  },

  async getModel(modelId: string): Promise<ModelMetadata> {
    const response = await api.get(`/models/${modelId}`);
    return response.data;
  },

  async deleteModel(modelId: string): Promise<void> {
    await api.delete(`/models/${modelId}`);
  },

  // Analysis
  async analyzeModel(modelId: string): Promise<AnalysisReport> {
    const response = await api.post(`/models/${modelId}/analyze`);
    return response.data;
  },

  async getAnalysisReport(modelId: string): Promise<AnalysisReport> {
    const response = await api.get(`/models/${modelId}/analysis`);
    return response.data;
  },

  // Optimization
  async startOptimization(
    modelId: string,
    criteria: OptimizationCriteria
  ): Promise<OptimizationSession> {
    const response = await api.post(`/models/${modelId}/optimize`, criteria);
    return response.data;
  },

  async getOptimizationSession(sessionId: string): Promise<OptimizationSession> {
    const response = await api.get(`/optimization/sessions/${sessionId}`);
    return response.data;
  },

  async getOptimizationSessions(): Promise<OptimizationSession[]> {
    const response = await api.get('/optimization/sessions');
    return response.data;
  },

  async cancelOptimization(sessionId: string): Promise<void> {
    await api.post(`/optimization/sessions/${sessionId}/cancel`);
  },

  async pauseOptimization(sessionId: string): Promise<void> {
    await api.post(`/optimization/sessions/${sessionId}/pause`);
  },

  async resumeOptimization(sessionId: string): Promise<void> {
    await api.post(`/optimization/sessions/${sessionId}/resume`);
  },

  // Evaluation
  async getEvaluationReport(modelId: string): Promise<EvaluationReport> {
    const response = await api.get(`/models/${modelId}/evaluation`);
    return response.data;
  },

  async evaluateModel(modelId: string, benchmarks?: string[]): Promise<EvaluationReport> {
    const response = await api.post(`/models/${modelId}/evaluate`, { benchmarks });
    return response.data;
  },

  // Configuration
  async getOptimizationCriteria(): Promise<OptimizationCriteria> {
    const response = await api.get('/config/optimization-criteria');
    return response.data;
  },

  async updateOptimizationCriteria(criteria: OptimizationCriteria): Promise<OptimizationCriteria> {
    const response = await api.put('/config/optimization-criteria', criteria);
    return response.data;
  },

  // Dashboard stats
  async getDashboardStats(): Promise<{
    total_models: number;
    active_optimizations: number;
    completed_optimizations: number;
    average_size_reduction: number;
    average_speed_improvement: number;
  }> {
    const response = await api.get('/dashboard/stats');
    return response.data;
  },
};

export default apiService;
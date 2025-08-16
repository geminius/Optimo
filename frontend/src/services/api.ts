import axios from 'axios';
import {
  ModelMetadata,
  OptimizationSession,
  OptimizationCriteria,
  AnalysisReport,
  EvaluationReport
} from '../types';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Add request interceptor for authentication
api.interceptors.request.use((config) => {
  const token = localStorage.getItem('auth_token');
  if (token && config.headers) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

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
import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { BrowserRouter } from 'react-router-dom';
import { message } from 'antd';
import App from '../../App';
import apiService from '../../services/api';

// Mock the API service
jest.mock('../../services/api');
const mockApiService = apiService as jest.Mocked<typeof apiService>;

// Mock antd message
jest.mock('antd', () => ({
  ...jest.requireActual('antd'),
  message: {
    error: jest.fn(),
    success: jest.fn(),
    info: jest.fn(),
  },
}));

// Mock WebSocket
jest.mock('../../contexts/WebSocketContext', () => ({
  WebSocketProvider: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  useWebSocket: () => ({
    socket: null,
    isConnected: true,
    subscribeToProgress: jest.fn(),
    unsubscribeFromProgress: jest.fn(),
  }),
}));

const mockStats = {
  total_models: 0,
  active_optimizations: 0,
  completed_optimizations: 0,
  average_size_reduction: 0,
  average_speed_improvement: 0,
};

const mockCriteria = {
  max_size_reduction: 50,
  min_accuracy_retention: 90,
  max_inference_time: 100,
  techniques: ['quantization', 'pruning'],
  hardware_target: 'gpu',
};

const mockModel = {
  id: 'model-123',
  name: 'OpenVLA Test Model',
  version: '1.0.0',
  model_type: 'openvla',
  framework: 'pytorch',
  size_mb: 500,
  parameters: 7000000,
  created_at: '2024-01-01T10:00:00Z',
  tags: ['robotics', 'vision-language'],
};

const mockAnalysisReport = {
  model_id: 'model-123',
  architecture_summary: {
    total_parameters: 7000000,
    model_size_mb: 500,
    layer_count: 24,
    framework: 'pytorch',
  },
  performance_profile: {
    inference_time_ms: 150,
    memory_usage_mb: 2048,
    throughput_samples_per_sec: 6.67,
  },
  optimization_opportunities: [
    {
      technique: 'quantization',
      estimated_impact: {
        size_reduction: 50,
        speed_improvement: 30,
        accuracy_impact: -2,
      },
      feasibility_score: 0.9,
      requirements: ['GPU with INT8 support'],
    },
    {
      technique: 'pruning',
      estimated_impact: {
        size_reduction: 30,
        speed_improvement: 20,
        accuracy_impact: -5,
      },
      feasibility_score: 0.8,
      requirements: ['Structured pruning support'],
    },
  ],
  compatibility_matrix: {
    quantization: true,
    pruning: true,
    distillation: false,
  },
  recommendations: [
    'Apply 8-bit quantization for optimal size/accuracy trade-off',
    'Consider structured pruning for additional size reduction',
  ],
};

const mockOptimizationSession = {
  id: 'session-456',
  model_id: 'model-123',
  status: 'running' as const,
  progress: 0,
  criteria: mockCriteria,
  plan: {
    techniques: ['quantization', 'pruning'],
    estimated_duration: 1800,
    expected_improvements: {
      size_reduction: 40,
      speed_improvement: 25,
      accuracy_impact: -3,
    },
  },
  steps: [
    {
      id: 'step-1',
      technique: 'quantization',
      status: 'pending' as const,
      progress: 0,
    },
    {
      id: 'step-2',
      technique: 'pruning',
      status: 'pending' as const,
      progress: 0,
    },
  ],
  created_at: '2024-01-01T11:00:00Z',
  updated_at: '2024-01-01T11:00:00Z',
};

const mockCompletedSession = {
  ...mockOptimizationSession,
  id: 'session-789',
  status: 'completed' as const,
  progress: 100,
  steps: [
    {
      id: 'step-1',
      technique: 'quantization',
      status: 'completed' as const,
      progress: 100,
      start_time: '2024-01-01T11:00:00Z',
      end_time: '2024-01-01T11:15:00Z',
    },
    {
      id: 'step-2',
      technique: 'pruning',
      status: 'completed' as const,
      progress: 100,
      start_time: '2024-01-01T11:15:00Z',
      end_time: '2024-01-01T11:30:00Z',
    },
  ],
  results: {
    original_size_mb: 500,
    optimized_size_mb: 300,
    size_reduction_percent: 40,
    original_inference_time_ms: 150,
    optimized_inference_time_ms: 112,
    speed_improvement_percent: 25.3,
    accuracy_retention_percent: 97,
    techniques_applied: ['quantization', 'pruning'],
  },
  updated_at: '2024-01-01T11:30:00Z',
};

const mockEvaluationReport = {
  model_id: 'model-123',
  benchmarks: [
    {
      name: 'Manipulation Accuracy',
      score: 0.97,
      unit: 'accuracy',
      baseline_score: 1.0,
    },
    {
      name: 'Navigation Success Rate',
      score: 0.95,
      unit: 'success_rate',
      baseline_score: 0.98,
    },
  ],
  performance_metrics: {
    accuracy: 0.97,
    inference_time_ms: 112,
    memory_usage_mb: 1536,
    throughput: 8.93,
  },
  comparison_baseline: {
    accuracy_change: -3,
    speed_change: 25.3,
    size_change: -40,
  },
  validation_status: 'passed' as const,
  recommendations: [
    'Optimization successful with acceptable accuracy retention',
    'Consider deploying optimized model to production',
  ],
};

const renderApp = () => {
  return render(
    <BrowserRouter>
      <App />
    </BrowserRouter>
  );
};

describe('Complete Optimization Workflow E2E Tests', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    mockApiService.getDashboardStats.mockResolvedValue(mockStats);
    mockApiService.getOptimizationSessions.mockResolvedValue([]);
    mockApiService.getOptimizationCriteria.mockResolvedValue(mockCriteria);
  });

  test('complete end-to-end optimization workflow', async () => {
    // Setup API mocks for the complete workflow
    mockApiService.uploadModel.mockResolvedValue(mockModel);
    mockApiService.analyzeModel.mockResolvedValue(mockAnalysisReport);
    mockApiService.startOptimization.mockResolvedValue(mockOptimizationSession);
    mockApiService.getOptimizationSession
      .mockResolvedValueOnce(mockOptimizationSession)
      .mockResolvedValueOnce({
        ...mockOptimizationSession,
        progress: 50,
        steps: [
          { ...mockOptimizationSession.steps[0], status: 'running', progress: 100 },
          { ...mockOptimizationSession.steps[1], status: 'pending', progress: 0 },
        ],
      })
      .mockResolvedValue(mockCompletedSession);
    mockApiService.getEvaluationReport.mockResolvedValue(mockEvaluationReport);

    renderApp();

    // Step 1: Upload a model
    console.log('Step 1: Uploading model...');
    
    const uploadLink = screen.getByText('Upload Model');
    fireEvent.click(uploadLink);

    await waitFor(() => {
      expect(screen.getByText('Click or drag model file to this area to upload')).toBeInTheDocument();
    });

    // Fill upload form
    const nameInput = screen.getByPlaceholderText('Enter a descriptive name for your model');
    fireEvent.change(nameInput, { target: { value: 'OpenVLA Test Model' } });

    const modelTypeSelect = screen.getByText('Select the type of robotics model').closest('.ant-select');
    if (modelTypeSelect) {
      fireEvent.mouseDown(modelTypeSelect.querySelector('.ant-select-selector')!);
      await waitFor(() => {
        const option = screen.getByText('OPENVLA');
        fireEvent.click(option);
      });
    }

    const frameworkSelect = screen.getByText('Select the ML framework').closest('.ant-select');
    if (frameworkSelect) {
      fireEvent.mouseDown(frameworkSelect.querySelector('.ant-select-selector')!);
      await waitFor(() => {
        const option = screen.getByText('PyTorch');
        fireEvent.click(option);
      });
    }

    // Add tags
    const tagsInput = screen.getByPlaceholderText('e.g., manipulation, navigation, vision');
    fireEvent.change(tagsInput, { target: { value: 'robotics, vision-language' } });

    // Add file
    const file = new File(['model content'], 'openvla-model.pth', { type: 'application/octet-stream' });
    const uploadArea = screen.getByText('Click or drag model file to this area to upload').closest('.ant-upload-drag');
    
    if (uploadArea) {
      const input = uploadArea.querySelector('input[type="file"]');
      if (input) {
        fireEvent.change(input, { target: { files: [file] } });
      }
    }

    // Submit upload
    const uploadButton = screen.getByText('Upload Model');
    fireEvent.click(uploadButton);

    await waitFor(() => {
      expect(mockApiService.uploadModel).toHaveBeenCalledWith(
        file,
        expect.objectContaining({
          name: 'OpenVLA Test Model',
          model_type: 'openvla',
          framework: 'pytorch',
          tags: ['robotics', 'vision-language'],
        })
      );
      expect(message.success).toHaveBeenCalledWith('Model uploaded successfully!');
    });

    // Should navigate back to dashboard
    await waitFor(() => {
      expect(screen.getByText('Total Models')).toBeInTheDocument();
    }, { timeout: 2000 });

    // Step 2: Configure optimization criteria
    console.log('Step 2: Configuring optimization criteria...');
    
    const configLink = screen.getByText('Configuration');
    fireEvent.click(configLink);

    await waitFor(() => {
      expect(screen.getByText('Performance Thresholds')).toBeInTheDocument();
    });

    // Modify criteria for more aggressive optimization
    const maxSizeReductionSlider = screen.getByText('Maximum Size Reduction (%)').closest('.ant-form-item')?.querySelector('.ant-slider-handle');
    if (maxSizeReductionSlider) {
      fireEvent.mouseDown(maxSizeReductionSlider);
      fireEvent.mouseMove(maxSizeReductionSlider, { clientX: 200 });
      fireEvent.mouseUp(maxSizeReductionSlider);
    }

    const saveButton = screen.getByText('Save Configuration');
    fireEvent.click(saveButton);

    await waitFor(() => {
      expect(mockApiService.updateOptimizationCriteria).toHaveBeenCalled();
      expect(message.success).toHaveBeenCalledWith('Configuration saved successfully');
    });

    // Step 3: Start optimization process
    console.log('Step 3: Starting optimization...');
    
    // Navigate back to dashboard
    const dashboardLink = screen.getByText('Dashboard');
    fireEvent.click(dashboardLink);

    // Mock that we now have the uploaded model and start optimization
    mockApiService.getDashboardStats.mockResolvedValue({
      ...mockStats,
      total_models: 1,
    });

    // Simulate starting optimization (this would typically be done through a model management interface)
    // For this test, we'll simulate the API call directly
    await waitFor(() => {
      expect(mockApiService.startOptimization).toHaveBeenCalledWith(
        'model-123',
        expect.objectContaining({
          techniques: ['quantization', 'pruning'],
        })
      );
    });

    // Step 4: Monitor optimization progress
    console.log('Step 4: Monitoring optimization progress...');
    
    // Update dashboard to show active optimization
    mockApiService.getOptimizationSessions.mockResolvedValue([mockOptimizationSession]);
    
    // Refresh dashboard
    const refreshButton = screen.getByText('Refresh') || screen.getByRole('button', { name: /refresh/i });
    if (refreshButton) {
      fireEvent.click(refreshButton);
    }

    await waitFor(() => {
      expect(screen.getByText('session-456')).toBeInTheDocument();
      expect(screen.getByText('RUNNING')).toBeInTheDocument();
    });

    // Step 5: View optimization history and results
    console.log('Step 5: Viewing optimization results...');
    
    // Navigate to history page
    const historyLink = screen.getByText('Optimization History');
    fireEvent.click(historyLink);

    // Mock completed session in history
    mockApiService.getOptimizationSessions.mockResolvedValue([mockCompletedSession]);

    await waitFor(() => {
      expect(screen.getByText('session-789')).toBeInTheDocument();
      expect(screen.getByText('COMPLETED')).toBeInTheDocument();
    });

    // View session details
    const detailsButton = screen.getAllByText('Details')[0];
    fireEvent.click(detailsButton);

    await waitFor(() => {
      expect(screen.getByText('Optimization Session Details')).toBeInTheDocument();
      expect(screen.getByText('40')).toBeInTheDocument(); // Size reduction
      expect(screen.getByText('25.3')).toBeInTheDocument(); // Speed improvement
      expect(screen.getByText('97')).toBeInTheDocument(); // Accuracy retention
    });

    // Step 6: Verify evaluation results
    console.log('Step 6: Verifying evaluation results...');
    
    await waitFor(() => {
      expect(mockApiService.getEvaluationReport).toHaveBeenCalledWith('model-123');
    });

    // Close modal
    const closeButton = screen.getByRole('button', { name: /close/i });
    fireEvent.click(closeButton);

    // Step 7: Verify final dashboard state
    console.log('Step 7: Verifying final state...');
    
    // Navigate back to dashboard
    fireEvent.click(dashboardLink);

    // Update stats to reflect completed optimization
    mockApiService.getDashboardStats.mockResolvedValue({
      total_models: 1,
      active_optimizations: 0,
      completed_optimizations: 1,
      average_size_reduction: 40,
      average_speed_improvement: 25.3,
    });

    await waitFor(() => {
      expect(screen.getByText('1')).toBeInTheDocument(); // Total models
      expect(screen.getByText('40%')).toBeInTheDocument(); // Avg size reduction
      expect(screen.getByText('25.3%')).toBeInTheDocument(); // Avg speed improvement
    });

    console.log('✅ Complete optimization workflow test passed!');
  }, 30000); // Extended timeout for complex workflow

  test('handles optimization failure gracefully', async () => {
    const failedSession = {
      ...mockOptimizationSession,
      id: 'session-failed',
      status: 'failed' as const,
      progress: 25,
      steps: [
        {
          id: 'step-1',
          technique: 'quantization',
          status: 'failed' as const,
          progress: 25,
          start_time: '2024-01-01T11:00:00Z',
          end_time: '2024-01-01T11:05:00Z',
          error_message: 'Quantization failed: Model architecture not supported',
        },
      ],
    };

    mockApiService.uploadModel.mockResolvedValue(mockModel);
    mockApiService.startOptimization.mockResolvedValue(failedSession);
    mockApiService.getOptimizationSessions.mockResolvedValue([failedSession]);

    renderApp();

    // Navigate to history to see failed optimization
    const historyLink = screen.getByText('Optimization History');
    fireEvent.click(historyLink);

    await waitFor(() => {
      expect(screen.getByText('session-failed')).toBeInTheDocument();
      expect(screen.getByText('FAILED')).toBeInTheDocument();
    });

    // View failed session details
    const detailsButton = screen.getByText('Details');
    fireEvent.click(detailsButton);

    await waitFor(() => {
      expect(screen.getByText('Quantization failed: Model architecture not supported')).toBeInTheDocument();
    });
  });

  test('handles concurrent optimizations', async () => {
    const session1 = { ...mockOptimizationSession, id: 'session-1', model_id: 'model-1' };
    const session2 = { ...mockOptimizationSession, id: 'session-2', model_id: 'model-2' };
    
    mockApiService.getOptimizationSessions.mockResolvedValue([session1, session2]);
    mockApiService.getDashboardStats.mockResolvedValue({
      ...mockStats,
      active_optimizations: 2,
    });

    renderApp();

    await waitFor(() => {
      expect(screen.getByText('2')).toBeInTheDocument(); // Active optimizations
      expect(screen.getByText('session-1')).toBeInTheDocument();
      expect(screen.getByText('session-2')).toBeInTheDocument();
    });

    // Should be able to control both sessions
    const pauseButtons = screen.getAllByText('Pause');
    expect(pauseButtons).toHaveLength(2);
  });
});
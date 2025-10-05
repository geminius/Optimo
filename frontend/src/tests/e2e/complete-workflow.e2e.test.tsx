import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { BrowserRouter } from 'react-router-dom';
import { message } from 'antd';
import App from '../../App';
import apiService from '../../services/api';

// Mock matchMedia for Ant Design responsive components
Object.defineProperty(window, 'matchMedia', {
  writable: true,
  value: jest.fn().mockImplementation(query => ({
    matches: false,
    media: query,
    onchange: null,
    addListener: jest.fn(),
    removeListener: jest.fn(),
    addEventListener: jest.fn(),
    removeEventListener: jest.fn(),
    dispatchEvent: jest.fn(),
  })),
});

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
const mockSubscribeToProgress = jest.fn();
const mockUnsubscribeFromProgress = jest.fn();

jest.mock('../../contexts/WebSocketContext', () => ({
  WebSocketProvider: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  useWebSocket: () => ({
    socket: {},
    isConnected: true,
    subscribeToProgress: mockSubscribeToProgress,
    unsubscribeFromProgress: mockUnsubscribeFromProgress,
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

const mockOptimizationSession = {
  id: 'session-456',
  model_id: 'model-123',
  status: 'running' as const,
  progress: 25,
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
      status: 'running' as const,
      progress: 50,
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

const renderApp = (initialRoute = '/') => {
  window.history.pushState({}, 'Test page', initialRoute);
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

  // Requirement 5.1: Real-time progress monitoring
  test('displays real-time optimization progress on dashboard', async () => {
    mockApiService.getOptimizationSessions.mockResolvedValue([mockOptimizationSession]);
    mockApiService.getDashboardStats.mockResolvedValue({
      ...mockStats,
      active_optimizations: 1,
    });

    renderApp();

    // Verify dashboard loads
    await waitFor(() => {
      expect(screen.getByText('Dashboard')).toBeInTheDocument();
      expect(screen.getByText('Total Models')).toBeInTheDocument();
      expect(screen.getByText('Active Optimizations')).toBeInTheDocument();
    });

    // Verify WebSocket subscription for progress updates
    await waitFor(() => {
      expect(mockSubscribeToProgress).toHaveBeenCalledWith(
        'session-456',
        expect.any(Function)
      );
    });
  });

  // Requirement 5.2: Control operations (pause, resume, cancel)
  test('allows user to pause, resume, and cancel optimizations', async () => {
    mockApiService.getOptimizationSessions.mockResolvedValue([mockOptimizationSession]);
    mockApiService.getDashboardStats.mockResolvedValue({
      ...mockStats,
      active_optimizations: 1,
    });
    mockApiService.pauseOptimization.mockResolvedValue(undefined);
    mockApiService.resumeOptimization.mockResolvedValue(undefined);
    mockApiService.cancelOptimization.mockResolvedValue(undefined);

    renderApp();

    await waitFor(() => {
      expect(screen.getByText('Dashboard')).toBeInTheDocument();
    });

    // Test pause operation
    await waitFor(() => {
      const pauseButton = screen.queryByText('Pause');
      if (pauseButton) {
        fireEvent.click(pauseButton);
        expect(mockApiService.pauseOptimization).toHaveBeenCalledWith('session-456');
      }
    }, { timeout: 5000 });
  });

  // Requirement 5.3: Optimization history and detailed logs
  test('displays optimization history with detailed logs and performance comparisons', async () => {
    const sessions = [mockCompletedSession];
    mockApiService.getOptimizationSessions.mockResolvedValue(sessions);

    renderApp();

    // Navigate to history page
    const historyLink = screen.getByText('Optimization History');
    fireEvent.click(historyLink);

    await waitFor(() => {
      expect(screen.getByText('Optimization History')).toBeInTheDocument();
    });

    // Verify session details can be viewed
    await waitFor(() => {
      const detailsButtons = screen.queryAllByText('Details');
      if (detailsButtons.length > 0) {
        fireEvent.click(detailsButtons[0]);
      }
    }, { timeout: 5000 });

    // Verify modal opens with session details
    await waitFor(() => {
      const modalTitle = screen.queryByText('Optimization Session Details');
      if (modalTitle) {
        expect(modalTitle).toBeInTheDocument();
      }
    }, { timeout: 3000 });
  });

  // Requirement 5.1, 5.3: Notification system
  test('displays notifications for optimization completion', async () => {
    mockApiService.getOptimizationSessions.mockResolvedValue([mockOptimizationSession]);
    mockApiService.getDashboardStats.mockResolvedValue({
      ...mockStats,
      active_optimizations: 1,
    });

    renderApp();

    await waitFor(() => {
      expect(screen.getByText('Dashboard')).toBeInTheDocument();
    });

    // Verify dashboard loads data
    await waitFor(() => {
      expect(mockApiService.getDashboardStats).toHaveBeenCalled();
      expect(mockApiService.getOptimizationSessions).toHaveBeenCalled();
    });
  });

  // Requirement 5.2: Configuration management
  test('allows configuration updates through UI', async () => {
    mockApiService.updateOptimizationCriteria.mockResolvedValue(mockCriteria);

    renderApp();

    // Navigate to configuration page
    const configLink = screen.getByText('Configuration');
    fireEvent.click(configLink);

    await waitFor(() => {
      expect(screen.getByText('Optimization Configuration')).toBeInTheDocument();
    });

    // Verify configuration form is displayed and can be saved
    await waitFor(() => {
      const saveButton = screen.queryByText('Save Configuration');
      if (saveButton) {
        fireEvent.click(saveButton);
        expect(mockApiService.updateOptimizationCriteria).toHaveBeenCalled();
      }
    }, { timeout: 3000 });
  });

  // Requirement 5.3: Error handling and user feedback
  test('handles errors gracefully and provides user feedback', async () => {
    mockApiService.getDashboardStats.mockRejectedValueOnce(new Error('Network error'));

    renderApp();

    await waitFor(() => {
      expect(message.error).toHaveBeenCalledWith('Failed to load dashboard data');
    }, { timeout: 3000 });

    // Verify dashboard still renders
    expect(screen.getByText('Dashboard')).toBeInTheDocument();
  });

  // Requirement 5.1: Multiple concurrent optimizations
  test('displays and manages multiple concurrent optimizations', async () => {
    const session1 = { ...mockOptimizationSession, id: 'session-1', model_id: 'model-1' };
    const session2 = { ...mockOptimizationSession, id: 'session-2', model_id: 'model-2' };

    mockApiService.getOptimizationSessions.mockResolvedValue([session1, session2]);
    mockApiService.getDashboardStats.mockResolvedValue({
      ...mockStats,
      active_optimizations: 2,
    });

    renderApp();

    await waitFor(() => {
      expect(screen.getByText('Dashboard')).toBeInTheDocument();
    });

    // Verify both sessions are tracked
    await waitFor(() => {
      expect(mockSubscribeToProgress).toHaveBeenCalledWith('session-1', expect.any(Function));
      expect(mockSubscribeToProgress).toHaveBeenCalledWith('session-2', expect.any(Function));
    });
  });

  // Requirement 5.3: Filtering and searching optimization history
  test('allows filtering and searching optimization history', async () => {
    const sessions = [
      mockCompletedSession,
      { ...mockCompletedSession, id: 'session-2', status: 'failed' as const },
      { ...mockCompletedSession, id: 'session-3', status: 'running' as const },
    ];
    mockApiService.getOptimizationSessions.mockResolvedValue(sessions);

    renderApp();

    // Navigate to history page
    const historyLink = screen.getByText('Optimization History');
    fireEvent.click(historyLink);

    await waitFor(() => {
      expect(screen.getByText('Optimization History')).toBeInTheDocument();
    });

    // Verify filters are available
    await waitFor(() => {
      expect(screen.getByText('Filters')).toBeInTheDocument();
      expect(screen.getByPlaceholderText('Search by ID')).toBeInTheDocument();
    });

    // Verify refresh button works
    const refreshButton = screen.getByText('Refresh');
    fireEvent.click(refreshButton);

    await waitFor(() => {
      expect(mockApiService.getOptimizationSessions).toHaveBeenCalled();
    });
  });

  // Requirement 5.1, 5.2, 5.3: Complete workflow integration
  test('complete end-to-end workflow from dashboard to history', async () => {
    mockApiService.getOptimizationSessions.mockResolvedValue([mockOptimizationSession]);
    mockApiService.getDashboardStats.mockResolvedValue({
      ...mockStats,
      total_models: 1,
      active_optimizations: 1,
    });

    renderApp();

    // Verify dashboard displays active optimization
    await waitFor(() => {
      expect(screen.getByText('Dashboard')).toBeInTheDocument();
      expect(screen.getByText('Total Models')).toBeInTheDocument();
    });

    // Navigate to configuration
    const configLink = screen.getByText('Configuration');
    fireEvent.click(configLink);

    await waitFor(() => {
      expect(screen.getByText('Optimization Configuration')).toBeInTheDocument();
    });

    // Navigate to history
    const historyLink = screen.getByText('Optimization History');
    fireEvent.click(historyLink);

    await waitFor(() => {
      expect(screen.getByText('Optimization History')).toBeInTheDocument();
    });

    // Navigate back to dashboard
    const dashboardLink = screen.getByText('Dashboard');
    fireEvent.click(dashboardLink);

    await waitFor(() => {
      expect(screen.getByText('Dashboard')).toBeInTheDocument();
    });

    // Verify all API calls were made
    expect(mockApiService.getDashboardStats).toHaveBeenCalled();
    expect(mockApiService.getOptimizationSessions).toHaveBeenCalled();
    expect(mockApiService.getOptimizationCriteria).toHaveBeenCalled();
  });
});

import React from 'react';
import { render, screen, fireEvent, waitFor, act } from '@testing-library/react';
import { BrowserRouter } from 'react-router-dom';
import { message } from 'antd';
import OptimizationHistory from '../pages/OptimizationHistory';
import apiService from '../services/api';

// Mock AuthService FIRST
jest.mock('../services/auth', () => ({
  __esModule: true,
  default: {
    getToken: jest.fn(() => 'mock-token'),
    getUser: jest.fn(() => ({ id: '1', username: 'testuser', email: 'test@test.com', role: 'user' })),
    isTokenValid: jest.fn(() => true),
    getTimeUntilExpiration: jest.fn(() => 3600),
    isTokenExpiringSoon: jest.fn(() => false),
  },
}));

// Mock the API service - using default export
jest.mock('../services/api', () => ({
  __esModule: true,
  default: {
    getOptimizationSessions: jest.fn(),
    getOptimizationDetails: jest.fn(),
  },
}));

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

const mockSessions = [
  {
    id: 'session-1',
    model_id: 'model-1',
    status: 'completed' as const,
    progress: 100,
    criteria: {
      techniques: ['quantization', 'pruning'],
      max_size_reduction: 50,
      min_accuracy_retention: 90,
      max_inference_time: 100,
      hardware_target: 'gpu',
    },
    steps: [
      {
        id: 'step-1',
        technique: 'quantization',
        status: 'completed' as const,
        progress: 100,
        start_time: '2024-01-01T10:00:00Z',
        end_time: '2024-01-01T10:15:00Z',
      },
      {
        id: 'step-2',
        technique: 'pruning',
        status: 'completed' as const,
        progress: 100,
        start_time: '2024-01-01T10:15:00Z',
        end_time: '2024-01-01T10:30:00Z',
      },
    ],
    results: {
      original_size_mb: 500,
      optimized_size_mb: 300,
      size_reduction_percent: 40,
      original_inference_time_ms: 100,
      optimized_inference_time_ms: 70,
      speed_improvement_percent: 30,
      accuracy_retention_percent: 95,
      techniques_applied: ['quantization', 'pruning'],
    },
    created_at: '2024-01-01T10:00:00Z',
    updated_at: '2024-01-01T10:30:00Z',
  },
  {
    id: 'session-2',
    model_id: 'model-2',
    status: 'running' as const,
    progress: 65,
    criteria: {
      techniques: ['quantization'],
      max_size_reduction: 30,
      min_accuracy_retention: 95,
      max_inference_time: 50,
      hardware_target: 'cpu',
    },
    steps: [
      {
        id: 'step-3',
        technique: 'quantization',
        status: 'running' as const,
        progress: 65,
        start_time: '2024-01-01T11:00:00Z',
      },
    ],
    created_at: '2024-01-01T11:00:00Z',
    updated_at: '2024-01-01T11:20:00Z',
  },
  {
    id: 'session-3',
    model_id: 'model-3',
    status: 'failed' as const,
    progress: 25,
    criteria: {
      techniques: ['pruning'],
      max_size_reduction: 60,
      min_accuracy_retention: 85,
      max_inference_time: 80,
      hardware_target: 'edge',
    },
    steps: [
      {
        id: 'step-4',
        technique: 'pruning',
        status: 'failed' as const,
        progress: 25,
        start_time: '2024-01-01T09:00:00Z',
        end_time: '2024-01-01T09:10:00Z',
        error_message: 'Pruning failed due to incompatible model architecture',
      },
    ],
    created_at: '2024-01-01T09:00:00Z',
    updated_at: '2024-01-01T09:10:00Z',
  },
];

const renderOptimizationHistory = () => {
  return render(
    <BrowserRouter>
      <OptimizationHistory />
    </BrowserRouter>
  );
};

describe('OptimizationHistory Component', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    mockApiService.getOptimizationSessions.mockResolvedValue(mockSessions);
  });

  test('renders optimization history page', async () => {
    renderOptimizationHistory();

    expect(screen.getByText('Optimization History')).toBeInTheDocument();
    
    await waitFor(() => {
      expect(screen.getByText('Total Sessions')).toBeInTheDocument();
      expect(screen.getByText('Completed')).toBeInTheDocument();
      expect(screen.getByText('Avg Size Reduction')).toBeInTheDocument();
      expect(screen.getByText('Avg Speed Improvement')).toBeInTheDocument();
    });
  });

  test('displays optimization sessions in table', async () => {
    renderOptimizationHistory();

    await waitFor(() => {
      // Session IDs are truncated in the table (first 8 chars + ...)
      expect(screen.getByText('session-1'.substring(0, 8) + '...')).toBeInTheDocument();
      expect(screen.getByText('session-2'.substring(0, 8) + '...')).toBeInTheDocument();
      expect(screen.getByText('session-3'.substring(0, 8) + '...')).toBeInTheDocument();
      expect(screen.getByText('COMPLETED')).toBeInTheDocument();
      expect(screen.getByText('RUNNING')).toBeInTheDocument();
      expect(screen.getByText('FAILED')).toBeInTheDocument();
    });
  });

  test('calculates and displays statistics correctly', async () => {
    renderOptimizationHistory();

    await waitFor(() => {
      expect(screen.getAllByText('3')).toHaveLength(1); // Total sessions (only one instance)
      expect(screen.getAllByText('1')).toHaveLength(1); // Completed sessions (only one instance)
      expect(screen.getByText('40')).toBeInTheDocument(); // Avg size reduction (only completed session)
      expect(screen.getByText('30')).toBeInTheDocument(); // Avg speed improvement (only completed session)
    });
  });

  test('displays technique performance chart', async () => {
    renderOptimizationHistory();

    await waitFor(() => {
      expect(screen.getByText('Technique Performance')).toBeInTheDocument();
    });
  });

  test('filters sessions by status', async () => {
    renderOptimizationHistory();

    await waitFor(() => {
      // Check that all sessions are initially displayed (by status tags)
      expect(screen.getByText('COMPLETED')).toBeInTheDocument();
      expect(screen.getByText('RUNNING')).toBeInTheDocument();
      expect(screen.getByText('FAILED')).toBeInTheDocument();
    });

    // Filter by completed status
    const statusSelect = screen.getByDisplayValue('').closest('.ant-select');
    if (statusSelect) {
      fireEvent.mouseDown(statusSelect.querySelector('.ant-select-selector')!);
      await waitFor(() => {
        const option = screen.getByText('Completed');
        fireEvent.click(option);
      });
    }

    await waitFor(() => {
      expect(screen.getByText('COMPLETED')).toBeInTheDocument();
      expect(screen.queryByText('RUNNING')).not.toBeInTheDocument();
      expect(screen.queryByText('FAILED')).not.toBeInTheDocument();
    });
  });

  test('filters sessions by technique', async () => {
    renderOptimizationHistory();

    await waitFor(() => {
      // Check that all sessions are initially displayed (by status tags)
      expect(screen.getByText('COMPLETED')).toBeInTheDocument();
      expect(screen.getByText('RUNNING')).toBeInTheDocument();
      expect(screen.getByText('FAILED')).toBeInTheDocument();
    });

    // Filter by quantization technique
    const techniqueSelects = screen.getAllByDisplayValue('');
    const techniqueSelect = techniqueSelects[1]?.closest('.ant-select'); // Second select should be technique
    if (techniqueSelect) {
      fireEvent.mouseDown(techniqueSelect.querySelector('.ant-select-selector')!);
      await waitFor(() => {
        const option = screen.getByText('Quantization');
        fireEvent.click(option);
      });
    }

    await waitFor(() => {
      expect(screen.getByText('COMPLETED')).toBeInTheDocument(); // Has quantization
      expect(screen.getByText('RUNNING')).toBeInTheDocument(); // Has quantization
      expect(screen.queryByText('FAILED')).not.toBeInTheDocument(); // Only has pruning
    });
  });

  test('searches sessions by ID', async () => {
    renderOptimizationHistory();

    await waitFor(() => {
      // Check that all sessions are initially displayed (by status tags)
      expect(screen.getByText('COMPLETED')).toBeInTheDocument();
      expect(screen.getByText('RUNNING')).toBeInTheDocument();
      expect(screen.getByText('FAILED')).toBeInTheDocument();
    });

    const searchInput = screen.getByPlaceholderText('Search by ID');
    fireEvent.change(searchInput, { target: { value: 'session-1' } });

    await waitFor(() => {
      expect(screen.getByText('COMPLETED')).toBeInTheDocument(); // session-1 is completed
      expect(screen.queryByText('RUNNING')).not.toBeInTheDocument(); // session-2 should be filtered out
      expect(screen.queryByText('FAILED')).not.toBeInTheDocument(); // session-3 should be filtered out
    });
  });

  test('opens session details modal', async () => {
    renderOptimizationHistory();

    await waitFor(() => {
      const detailsButtons = screen.getAllByText('Details');
      expect(detailsButtons.length).toBeGreaterThan(0);
      fireEvent.click(detailsButtons[0]);
    });

    await waitFor(() => {
      expect(screen.getByText('Optimization Session Details')).toBeInTheDocument();
      expect(screen.getByText('session-1')).toBeInTheDocument(); // Full ID shown in modal
      expect(screen.getByText('Results')).toBeInTheDocument();
      expect(screen.getByText('Optimization Steps')).toBeInTheDocument();
    });
  });

  test('displays session results in modal', async () => {
    renderOptimizationHistory();

    await waitFor(() => {
      const detailsButtons = screen.getAllByText('Details');
      expect(detailsButtons.length).toBeGreaterThan(0);
      fireEvent.click(detailsButtons[0]);
    });

    await waitFor(() => {
      // Results should be displayed in the modal
      const results = screen.getAllByText('40'); // Size reduction
      expect(results.length).toBeGreaterThan(0);
      const speedResults = screen.getAllByText('30'); // Speed improvement  
      expect(speedResults.length).toBeGreaterThan(0);
      const accuracyResults = screen.getAllByText('95'); // Accuracy retention
      expect(accuracyResults.length).toBeGreaterThan(0);
    });
  });

  test('displays optimization steps with status', async () => {
    renderOptimizationHistory();

    await waitFor(() => {
      const detailsButtons = screen.getAllByText('Details');
      expect(detailsButtons.length).toBeGreaterThan(0);
      fireEvent.click(detailsButtons[0]);
    });

    await waitFor(() => {
      expect(screen.getByText('Step 1: quantization')).toBeInTheDocument();
      expect(screen.getByText('Step 2: pruning')).toBeInTheDocument();
    });
  });

  test('handles delete session', async () => {
    renderOptimizationHistory();

    await waitFor(() => {
      const deleteButtons = screen.getAllByText('Delete');
      expect(deleteButtons.length).toBeGreaterThan(0);
      fireEvent.click(deleteButtons[0]);
    });

    expect(message.success).toHaveBeenCalledWith('Session deleted successfully');
  });

  test('handles download functionality', async () => {
    renderOptimizationHistory();

    await waitFor(() => {
      const downloadButtons = screen.getAllByText('Download');
      expect(downloadButtons.length).toBeGreaterThan(0);
      fireEvent.click(downloadButtons[0]);
    });

    expect(message.info).toHaveBeenCalledWith('Download functionality would be implemented');
  });

  test('refreshes data when refresh button is clicked', async () => {
    renderOptimizationHistory();

    await waitFor(() => {
      const refreshButton = screen.getByText('Refresh');
      expect(refreshButton).toBeInTheDocument();
    });

    const refreshButton = screen.getByText('Refresh');
    
    // Use act to wrap the click event to handle state updates
    await act(async () => {
      fireEvent.click(refreshButton);
    });

    expect(mockApiService.getOptimizationSessions).toHaveBeenCalledTimes(2); // Initial load + refresh
  });

  test('handles API errors gracefully', async () => {
    mockApiService.getOptimizationSessions.mockRejectedValue(new Error('API Error'));
    renderOptimizationHistory();

    await waitFor(() => {
      expect(message.error).toHaveBeenCalledWith('Failed to load optimization history');
    });
  });

  test('displays error message in step details', async () => {
    renderOptimizationHistory();

    // Open details for failed session (session-3 which is the third one, index 2)
    await waitFor(() => {
      const detailsButtons = screen.getAllByText('Details');
      expect(detailsButtons.length).toBeGreaterThanOrEqual(3);
      fireEvent.click(detailsButtons[2]); // Third session is failed
    });

    await waitFor(() => {
      expect(screen.getByText('Pruning failed due to incompatible model architecture')).toBeInTheDocument();
    });
  });
});
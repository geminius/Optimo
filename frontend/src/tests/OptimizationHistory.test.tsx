import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { BrowserRouter } from 'react-router-dom';
import { message } from 'antd';
import OptimizationHistory from '../pages/OptimizationHistory';
import apiService from '../services/api';

// Mock the API service
jest.mock('../services/api');
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
      expect(screen.getByText('session-1')).toBeInTheDocument();
      expect(screen.getByText('session-2')).toBeInTheDocument();
      expect(screen.getByText('session-3')).toBeInTheDocument();
      expect(screen.getByText('COMPLETED')).toBeInTheDocument();
      expect(screen.getByText('RUNNING')).toBeInTheDocument();
      expect(screen.getByText('FAILED')).toBeInTheDocument();
    });
  });

  test('calculates and displays statistics correctly', async () => {
    renderOptimizationHistory();

    await waitFor(() => {
      expect(screen.getByText('3')).toBeInTheDocument(); // Total sessions
      expect(screen.getByText('1')).toBeInTheDocument(); // Completed sessions
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
      expect(screen.getByText('session-1')).toBeInTheDocument();
      expect(screen.getByText('session-2')).toBeInTheDocument();
      expect(screen.getByText('session-3')).toBeInTheDocument();
    });

    // Filter by completed status
    const statusSelect = screen.getByPlaceholderText('Status').closest('.ant-select');
    if (statusSelect) {
      fireEvent.mouseDown(statusSelect.querySelector('.ant-select-selector')!);
      await waitFor(() => {
        const option = screen.getByText('Completed');
        fireEvent.click(option);
      });
    }

    await waitFor(() => {
      expect(screen.getByText('session-1')).toBeInTheDocument();
      expect(screen.queryByText('session-2')).not.toBeInTheDocument();
      expect(screen.queryByText('session-3')).not.toBeInTheDocument();
    });
  });

  test('filters sessions by technique', async () => {
    renderOptimizationHistory();

    await waitFor(() => {
      expect(screen.getByText('session-1')).toBeInTheDocument();
      expect(screen.getByText('session-2')).toBeInTheDocument();
      expect(screen.getByText('session-3')).toBeInTheDocument();
    });

    // Filter by quantization technique
    const techniqueSelect = screen.getByPlaceholderText('Technique').closest('.ant-select');
    if (techniqueSelect) {
      fireEvent.mouseDown(techniqueSelect.querySelector('.ant-select-selector')!);
      await waitFor(() => {
        const option = screen.getByText('Quantization');
        fireEvent.click(option);
      });
    }

    await waitFor(() => {
      expect(screen.getByText('session-1')).toBeInTheDocument(); // Has quantization
      expect(screen.getByText('session-2')).toBeInTheDocument(); // Has quantization
      expect(screen.queryByText('session-3')).not.toBeInTheDocument(); // Only has pruning
    });
  });

  test('searches sessions by ID', async () => {
    renderOptimizationHistory();

    await waitFor(() => {
      expect(screen.getByText('session-1')).toBeInTheDocument();
      expect(screen.getByText('session-2')).toBeInTheDocument();
      expect(screen.getByText('session-3')).toBeInTheDocument();
    });

    const searchInput = screen.getByPlaceholderText('Search by ID');
    fireEvent.change(searchInput, { target: { value: 'session-1' } });

    await waitFor(() => {
      expect(screen.getByText('session-1')).toBeInTheDocument();
      expect(screen.queryByText('session-2')).not.toBeInTheDocument();
      expect(screen.queryByText('session-3')).not.toBeInTheDocument();
    });
  });

  test('opens session details modal', async () => {
    renderOptimizationHistory();

    await waitFor(() => {
      const detailsButton = screen.getAllByText('Details')[0];
      fireEvent.click(detailsButton);
    });

    await waitFor(() => {
      expect(screen.getByText('Optimization Session Details')).toBeInTheDocument();
      expect(screen.getByText('session-1')).toBeInTheDocument();
      expect(screen.getByText('Results')).toBeInTheDocument();
      expect(screen.getByText('Optimization Steps')).toBeInTheDocument();
    });
  });

  test('displays session results in modal', async () => {
    renderOptimizationHistory();

    await waitFor(() => {
      const detailsButton = screen.getAllByText('Details')[0];
      fireEvent.click(detailsButton);
    });

    await waitFor(() => {
      expect(screen.getByText('40')).toBeInTheDocument(); // Size reduction
      expect(screen.getByText('30')).toBeInTheDocument(); // Speed improvement
      expect(screen.getByText('95')).toBeInTheDocument(); // Accuracy retention
    });
  });

  test('displays optimization steps with status', async () => {
    renderOptimizationHistory();

    await waitFor(() => {
      const detailsButton = screen.getAllByText('Details')[0];
      fireEvent.click(detailsButton);
    });

    await waitFor(() => {
      expect(screen.getByText('Step 1: quantization')).toBeInTheDocument();
      expect(screen.getByText('Step 2: pruning')).toBeInTheDocument();
    });
  });

  test('handles delete session', async () => {
    renderOptimizationHistory();

    await waitFor(() => {
      const deleteButton = screen.getAllByText('Delete')[0];
      fireEvent.click(deleteButton);
    });

    expect(message.success).toHaveBeenCalledWith('Session deleted successfully');
  });

  test('handles download functionality', async () => {
    renderOptimizationHistory();

    await waitFor(() => {
      const downloadButton = screen.getByText('Download');
      fireEvent.click(downloadButton);
    });

    expect(message.info).toHaveBeenCalledWith('Download functionality would be implemented');
  });

  test('refreshes data when refresh button is clicked', async () => {
    renderOptimizationHistory();

    const refreshButton = screen.getByText('Refresh');
    fireEvent.click(refreshButton);

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

    // Open details for failed session
    await waitFor(() => {
      const detailsButtons = screen.getAllByText('Details');
      const failedSessionDetailsButton = detailsButtons[2]; // Third session is failed
      fireEvent.click(failedSessionDetailsButton);
    });

    await waitFor(() => {
      expect(screen.getByText('Pruning failed due to incompatible model architecture')).toBeInTheDocument();
    });
  });
});
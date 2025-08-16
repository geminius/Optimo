import React from 'react';
import { render, screen, waitFor, fireEvent } from '@testing-library/react';
import { BrowserRouter } from 'react-router-dom';
import { message } from 'antd';
import Dashboard from '../pages/Dashboard';
import apiService from '../services/api';
import { WebSocketProvider } from '../contexts/WebSocketContext';

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

// Mock WebSocket context
jest.mock('../contexts/WebSocketContext', () => ({
  WebSocketProvider: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  useWebSocket: () => ({
    socket: null,
    isConnected: true,
    subscribeToProgress: jest.fn(),
    unsubscribeFromProgress: jest.fn(),
  }),
}));

const mockStats = {
  total_models: 5,
  active_optimizations: 2,
  completed_optimizations: 15,
  average_size_reduction: 35.5,
  average_speed_improvement: 28.3,
};

const mockSessions = [
  {
    id: 'session-1',
    model_id: 'model-1',
    status: 'running' as const,
    progress: 65,
    criteria: {
      techniques: ['quantization', 'pruning'],
      max_size_reduction: 50,
      min_accuracy_retention: 90,
      max_inference_time: 100,
      hardware_target: 'gpu',
    },
    steps: [],
    created_at: '2024-01-01T10:00:00Z',
    updated_at: '2024-01-01T10:30:00Z',
  },
  {
    id: 'session-2',
    model_id: 'model-2',
    status: 'completed' as const,
    progress: 100,
    criteria: {
      techniques: ['quantization'],
      max_size_reduction: 30,
      min_accuracy_retention: 95,
      max_inference_time: 50,
      hardware_target: 'cpu',
    },
    steps: [],
    created_at: '2024-01-01T09:00:00Z',
    updated_at: '2024-01-01T09:45:00Z',
  },
];

const renderDashboard = () => {
  return render(
    <BrowserRouter>
      <WebSocketProvider>
        <Dashboard />
      </WebSocketProvider>
    </BrowserRouter>
  );
};

describe('Dashboard Component', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    mockApiService.getDashboardStats.mockResolvedValue(mockStats);
    mockApiService.getOptimizationSessions.mockResolvedValue(mockSessions);
  });

  test('renders dashboard with statistics', async () => {
    renderDashboard();

    expect(screen.getByText('Dashboard')).toBeInTheDocument();
    
    await waitFor(() => {
      expect(screen.getByText('5')).toBeInTheDocument(); // Total models
      expect(screen.getByText('2')).toBeInTheDocument(); // Active optimizations
      expect(screen.getByText('35.5%')).toBeInTheDocument(); // Avg size reduction
      expect(screen.getByText('28.3%')).toBeInTheDocument(); // Avg speed improvement
    });
  });

  test('displays active optimization sessions', async () => {
    renderDashboard();

    await waitFor(() => {
      expect(screen.getByText('session-1')).toBeInTheDocument();
      expect(screen.getByText('RUNNING')).toBeInTheDocument();
      expect(screen.getByText('quantization')).toBeInTheDocument();
      expect(screen.getByText('pruning')).toBeInTheDocument();
    });
  });

  test('handles pause optimization action', async () => {
    mockApiService.pauseOptimization.mockResolvedValue();
    renderDashboard();

    await waitFor(() => {
      const pauseButton = screen.getByText('Pause');
      fireEvent.click(pauseButton);
    });

    expect(mockApiService.pauseOptimization).toHaveBeenCalledWith('session-1');
    expect(message.success).toHaveBeenCalledWith('Optimization paused successfully');
  });

  test('handles cancel optimization action', async () => {
    mockApiService.cancelOptimization.mockResolvedValue();
    renderDashboard();

    await waitFor(() => {
      const cancelButton = screen.getByText('Cancel');
      fireEvent.click(cancelButton);
    });

    expect(mockApiService.cancelOptimization).toHaveBeenCalledWith('session-1');
    expect(message.success).toHaveBeenCalledWith('Optimization cancelled successfully');
  });

  test('handles API errors gracefully', async () => {
    mockApiService.getDashboardStats.mockRejectedValue(new Error('API Error'));
    renderDashboard();

    await waitFor(() => {
      expect(message.error).toHaveBeenCalledWith('Failed to load dashboard data');
    });
  });

  test('displays performance chart', async () => {
    renderDashboard();

    await waitFor(() => {
      expect(screen.getByText('Optimization Performance Trends')).toBeInTheDocument();
    });
  });

  test('refreshes data periodically', async () => {
    jest.useFakeTimers();
    renderDashboard();

    // Initial load
    expect(mockApiService.getDashboardStats).toHaveBeenCalledTimes(1);
    expect(mockApiService.getOptimizationSessions).toHaveBeenCalledTimes(1);

    // Fast forward 30 seconds
    jest.advanceTimersByTime(30000);

    await waitFor(() => {
      expect(mockApiService.getDashboardStats).toHaveBeenCalledTimes(2);
      expect(mockApiService.getOptimizationSessions).toHaveBeenCalledTimes(2);
    });

    jest.useRealTimers();
  });
});
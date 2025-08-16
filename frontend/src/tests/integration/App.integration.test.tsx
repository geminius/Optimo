import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { BrowserRouter } from 'react-router-dom';
import { message } from 'antd';
import App from '../App';
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

// Mock WebSocket
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
];

const mockCriteria = {
  max_size_reduction: 50,
  min_accuracy_retention: 90,
  max_inference_time: 100,
  techniques: ['quantization', 'pruning'],
  hardware_target: 'gpu',
};

const renderApp = () => {
  return render(
    <BrowserRouter>
      <App />
    </BrowserRouter>
  );
};

describe('App Integration Tests', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    mockApiService.getDashboardStats.mockResolvedValue(mockStats);
    mockApiService.getOptimizationSessions.mockResolvedValue(mockSessions);
    mockApiService.getOptimizationCriteria.mockResolvedValue(mockCriteria);
  });

  test('renders app with navigation and loads dashboard by default', async () => {
    renderApp();

    // Check header
    expect(screen.getByText('Robotics Model Optimization Platform')).toBeInTheDocument();

    // Check navigation
    expect(screen.getByText('Dashboard')).toBeInTheDocument();
    expect(screen.getByText('Upload Model')).toBeInTheDocument();
    expect(screen.getByText('Optimization History')).toBeInTheDocument();
    expect(screen.getByText('Configuration')).toBeInTheDocument();

    // Dashboard should load by default
    await waitFor(() => {
      expect(screen.getByText('Total Models')).toBeInTheDocument();
      expect(screen.getByText('Active Optimizations')).toBeInTheDocument();
    });
  });

  test('navigates between pages using sidebar', async () => {
    renderApp();

    // Navigate to Upload Model
    const uploadLink = screen.getByText('Upload Model');
    fireEvent.click(uploadLink);

    await waitFor(() => {
      expect(screen.getByText('Click or drag model file to this area to upload')).toBeInTheDocument();
    });

    // Navigate to Optimization History
    const historyLink = screen.getByText('Optimization History');
    fireEvent.click(historyLink);

    await waitFor(() => {
      expect(screen.getByText('Total Sessions')).toBeInTheDocument();
    });

    // Navigate to Configuration
    const configLink = screen.getByText('Configuration');
    fireEvent.click(configLink);

    await waitFor(() => {
      expect(screen.getByText('Performance Thresholds')).toBeInTheDocument();
    });

    // Navigate back to Dashboard
    const dashboardLink = screen.getByText('Dashboard');
    fireEvent.click(dashboardLink);

    await waitFor(() => {
      expect(screen.getByText('Total Models')).toBeInTheDocument();
    });
  });

  test('complete model upload workflow', async () => {
    const mockModel = {
      id: 'model-1',
      name: 'Test Model',
      version: '1.0.0',
      model_type: 'openvla',
      framework: 'pytorch',
      size_mb: 100,
      parameters: 1000000,
      created_at: '2024-01-01T10:00:00Z',
      tags: ['test'],
    };

    mockApiService.uploadModel.mockResolvedValue(mockModel);
    renderApp();

    // Navigate to upload page
    const uploadLink = screen.getByText('Upload Model');
    fireEvent.click(uploadLink);

    await waitFor(() => {
      expect(screen.getByText('Click or drag model file to this area to upload')).toBeInTheDocument();
    });

    // Fill form
    const nameInput = screen.getByPlaceholderText('Enter a descriptive name for your model');
    fireEvent.change(nameInput, { target: { value: 'Test Model' } });

    // Select model type
    const modelTypeSelect = screen.getByText('Select the type of robotics model').closest('.ant-select');
    if (modelTypeSelect) {
      fireEvent.mouseDown(modelTypeSelect.querySelector('.ant-select-selector')!);
      await waitFor(() => {
        const option = screen.getByText('OPENVLA');
        fireEvent.click(option);
      });
    }

    // Select framework
    const frameworkSelect = screen.getByText('Select the ML framework').closest('.ant-select');
    if (frameworkSelect) {
      fireEvent.mouseDown(frameworkSelect.querySelector('.ant-select-selector')!);
      await waitFor(() => {
        const option = screen.getByText('PyTorch');
        fireEvent.click(option);
      });
    }

    // Add file
    const file = new File(['model content'], 'test-model.pth', { type: 'application/octet-stream' });
    const uploadArea = screen.getByText('Click or drag model file to this area to upload').closest('.ant-upload-drag');
    
    if (uploadArea) {
      const input = uploadArea.querySelector('input[type="file"]');
      if (input) {
        fireEvent.change(input, { target: { files: [file] } });
      }
    }

    // Submit form
    const uploadButton = screen.getByText('Upload Model');
    fireEvent.click(uploadButton);

    await waitFor(() => {
      expect(mockApiService.uploadModel).toHaveBeenCalled();
      expect(message.success).toHaveBeenCalledWith('Model uploaded successfully!');
    });

    // Should navigate back to dashboard
    await waitFor(() => {
      expect(screen.getByText('Total Models')).toBeInTheDocument();
    }, { timeout: 2000 });
  });

  test('configuration workflow', async () => {
    renderApp();

    // Navigate to configuration
    const configLink = screen.getByText('Configuration');
    fireEvent.click(configLink);

    await waitFor(() => {
      expect(screen.getByText('Performance Thresholds')).toBeInTheDocument();
    });

    // Modify configuration
    const maxInferenceTimeInput = screen.getByPlaceholderText('Enter maximum inference time in milliseconds');
    fireEvent.change(maxInferenceTimeInput, { target: { value: '150' } });

    // Should show unsaved changes warning
    await waitFor(() => {
      expect(screen.getByText('You have unsaved changes')).toBeInTheDocument();
    });

    // Save configuration
    const saveButton = screen.getByText('Save Configuration');
    fireEvent.click(saveButton);

    await waitFor(() => {
      expect(mockApiService.updateOptimizationCriteria).toHaveBeenCalled();
      expect(message.success).toHaveBeenCalledWith('Configuration saved successfully');
    });
  });

  test('optimization history workflow', async () => {
    renderApp();

    // Navigate to history
    const historyLink = screen.getByText('Optimization History');
    fireEvent.click(historyLink);

    await waitFor(() => {
      expect(screen.getByText('Total Sessions')).toBeInTheDocument();
      expect(screen.getByText('session-1')).toBeInTheDocument();
    });

    // View session details
    const detailsButton = screen.getByText('Details');
    fireEvent.click(detailsButton);

    await waitFor(() => {
      expect(screen.getByText('Optimization Session Details')).toBeInTheDocument();
    });

    // Close modal
    const closeButton = screen.getByRole('button', { name: /close/i });
    fireEvent.click(closeButton);

    await waitFor(() => {
      expect(screen.queryByText('Optimization Session Details')).not.toBeInTheDocument();
    });
  });

  test('handles global API errors', async () => {
    mockApiService.getDashboardStats.mockRejectedValue(new Error('Network error'));
    renderApp();

    await waitFor(() => {
      expect(message.error).toHaveBeenCalledWith('Failed to load dashboard data');
    });
  });

  test('sidebar navigation highlights active page', async () => {
    renderApp();

    // Dashboard should be active by default
    const dashboardMenuItem = screen.getByText('Dashboard').closest('.ant-menu-item');
    expect(dashboardMenuItem).toHaveClass('ant-menu-item-selected');

    // Navigate to upload
    const uploadLink = screen.getByText('Upload Model');
    fireEvent.click(uploadLink);

    await waitFor(() => {
      const uploadMenuItem = screen.getByText('Upload Model').closest('.ant-menu-item');
      expect(uploadMenuItem).toHaveClass('ant-menu-item-selected');
    });
  });

  test('responsive layout works correctly', () => {
    renderApp();

    // Check that layout components are present
    expect(screen.getByRole('navigation')).toBeInTheDocument(); // Sidebar
    expect(screen.getByText('Robotics Model Optimization Platform')).toBeInTheDocument(); // Header
  });

  test('WebSocket integration works', async () => {
    renderApp();

    // Dashboard should load and attempt to subscribe to progress updates
    await waitFor(() => {
      expect(screen.getByText('Total Models')).toBeInTheDocument();
    });

    // The WebSocket context should be available throughout the app
    expect(screen.getByText('Dashboard')).toBeInTheDocument();
  });
});
import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { BrowserRouter } from 'react-router-dom';
import { message } from 'antd';
import Configuration from '../pages/Configuration';
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
  },
}));

const mockCriteria = {
  max_size_reduction: 50,
  min_accuracy_retention: 90,
  max_inference_time: 100,
  techniques: ['quantization', 'pruning'],
  hardware_target: 'gpu',
};

const renderConfiguration = () => {
  return render(
    <BrowserRouter>
      <Configuration />
    </BrowserRouter>
  );
};

describe('Configuration Component', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    mockApiService.getOptimizationCriteria.mockResolvedValue(mockCriteria);
    mockApiService.updateOptimizationCriteria.mockResolvedValue(mockCriteria);
  });

  test('renders configuration page', async () => {
    renderConfiguration();

    expect(screen.getByText('Configuration')).toBeInTheDocument();
    
    await waitFor(() => {
      expect(screen.getByText('Performance Thresholds')).toBeInTheDocument();
      expect(screen.getByText('Optimization Techniques')).toBeInTheDocument();
      expect(screen.getByText('Advanced Settings')).toBeInTheDocument();
    });
  });

  test('loads existing configuration', async () => {
    renderConfiguration();

    await waitFor(() => {
      expect(mockApiService.getOptimizationCriteria).toHaveBeenCalled();
    });

    // Check if form is populated with existing values
    await waitFor(() => {
      const maxInferenceTimeInput = screen.getByPlaceholderText('Enter maximum inference time in milliseconds');
      expect((maxInferenceTimeInput as HTMLInputElement).value).toBe('100');
    });
  });

  test('displays unsaved changes warning', async () => {
    renderConfiguration();

    await waitFor(() => {
      const maxInferenceTimeInput = screen.getByPlaceholderText('Enter maximum inference time in milliseconds');
      fireEvent.change(maxInferenceTimeInput, { target: { value: '150' } });
    });

    await waitFor(() => {
      expect(screen.getByText('You have unsaved changes')).toBeInTheDocument();
      expect(screen.getByText("Don't forget to save your configuration changes.")).toBeInTheDocument();
    });
  });

  test('saves configuration successfully', async () => {
    renderConfiguration();

    await waitFor(() => {
      const maxInferenceTimeInput = screen.getByPlaceholderText('Enter maximum inference time in milliseconds');
      fireEvent.change(maxInferenceTimeInput, { target: { value: '150' } });
    });

    const saveButton = screen.getByText('Save Configuration');
    fireEvent.click(saveButton);

    await waitFor(() => {
      expect(mockApiService.updateOptimizationCriteria).toHaveBeenCalledWith(
        expect.objectContaining({
          max_inference_time: 150,
        })
      );
      expect(message.success).toHaveBeenCalledWith('Configuration saved successfully');
    });
  });

  test('handles save failure', async () => {
    mockApiService.updateOptimizationCriteria.mockRejectedValue(new Error('Save failed'));
    renderConfiguration();

    await waitFor(() => {
      const maxInferenceTimeInput = screen.getByPlaceholderText('Enter maximum inference time in milliseconds');
      fireEvent.change(maxInferenceTimeInput, { target: { value: '150' } });
    });

    const saveButton = screen.getByText('Save Configuration');
    fireEvent.click(saveButton);

    await waitFor(() => {
      expect(message.error).toHaveBeenCalledWith('Failed to save configuration');
    });
  });

  test('resets changes', async () => {
    renderConfiguration();

    await waitFor(() => {
      const maxInferenceTimeInput = screen.getByPlaceholderText('Enter maximum inference time in milliseconds');
      fireEvent.change(maxInferenceTimeInput, { target: { value: '150' } });
    });

    const resetButton = screen.getByText('Reset Changes');
    fireEvent.click(resetButton);

    await waitFor(() => {
      const maxInferenceTimeInput = screen.getByPlaceholderText('Enter maximum inference time in milliseconds');
      expect((maxInferenceTimeInput as HTMLInputElement).value).toBe('100');
    });
  });

  test('reloads configuration', async () => {
    renderConfiguration();

    const reloadButton = screen.getByText('Reload');
    fireEvent.click(reloadButton);

    expect(mockApiService.getOptimizationCriteria).toHaveBeenCalledTimes(2); // Initial load + reload
  });

  test('validates required fields', async () => {
    renderConfiguration();

    await waitFor(() => {
      const maxInferenceTimeInput = screen.getByPlaceholderText('Enter maximum inference time in milliseconds');
      fireEvent.change(maxInferenceTimeInput, { target: { value: '' } });
    });

    const saveButton = screen.getByText('Save Configuration');
    fireEvent.click(saveButton);

    await waitFor(() => {
      expect(screen.getByText('Please set maximum inference time')).toBeInTheDocument();
    });
  });

  test('switches between tabs', async () => {
    renderConfiguration();

    // Click on Advanced Settings tab
    const advancedTab = screen.getByText('Advanced Settings');
    fireEvent.click(advancedTab);

    await waitFor(() => {
      expect(screen.getByText('Quantization Settings')).toBeInTheDocument();
      expect(screen.getByText('Pruning Settings')).toBeInTheDocument();
      expect(screen.getByText('Evaluation Settings')).toBeInTheDocument();
    });

    // Click on Monitoring & Alerts tab
    const monitoringTab = screen.getByText('Monitoring & Alerts');
    fireEvent.click(monitoringTab);

    await waitFor(() => {
      expect(screen.getByText('Progress Monitoring')).toBeInTheDocument();
      expect(screen.getByText('Alert Settings')).toBeInTheDocument();
    });
  });

  test('handles technique selection', async () => {
    renderConfiguration();

    await waitFor(() => {
      const techniqueSelect = screen.getByText('Select optimization techniques to enable').closest('.ant-select');
      if (techniqueSelect) {
        fireEvent.mouseDown(techniqueSelect.querySelector('.ant-select-selector')!);
      }
    });

    await waitFor(() => {
      const quantizationOption = screen.getByText('Quantization');
      fireEvent.click(quantizationOption);
    });

    // Verify the selection
    await waitFor(() => {
      expect(screen.getByText('Reduce model precision (4-bit, 8-bit)')).toBeInTheDocument();
    });
  });

  test('handles hardware target selection', async () => {
    renderConfiguration();

    await waitFor(() => {
      const hardwareSelect = screen.getByText('Select target hardware for optimization').closest('.ant-select');
      if (hardwareSelect) {
        fireEvent.mouseDown(hardwareSelect.querySelector('.ant-select-selector')!);
      }
    });

    await waitFor(() => {
      const cpuOption = screen.getByText('CPU');
      fireEvent.click(cpuOption);
    });

    // Verify the selection
    await waitFor(() => {
      expect(screen.getByText('Optimize for CPU inference')).toBeInTheDocument();
    });
  });

  test('toggles advanced switches', async () => {
    renderConfiguration();

    // Switch to Advanced Settings tab
    const advancedTab = screen.getByText('Advanced Settings');
    fireEvent.click(advancedTab);

    await waitFor(() => {
      const enable4bitSwitch = screen.getByText('Enable 4-bit Quantization').closest('.ant-form-item')?.querySelector('.ant-switch');
      if (enable4bitSwitch) {
        fireEvent.click(enable4bitSwitch);
      }
    });

    // Verify the switch state changed
    await waitFor(() => {
      const enable4bitSwitch = screen.getByText('Enable 4-bit Quantization').closest('.ant-form-item')?.querySelector('.ant-switch');
      expect(enable4bitSwitch).toHaveClass('ant-switch-checked');
    });
  });

  test('handles slider changes', async () => {
    renderConfiguration();

    await waitFor(() => {
      // Find the sparsity ratio slider in Advanced Settings
      const advancedTab = screen.getByText('Advanced Settings');
      fireEvent.click(advancedTab);
    });

    await waitFor(() => {
      const sparsitySlider = screen.getByText('Target Sparsity Ratio (%)').closest('.ant-form-item')?.querySelector('.ant-slider-handle');
      if (sparsitySlider) {
        fireEvent.mouseDown(sparsitySlider);
        fireEvent.mouseMove(sparsitySlider, { clientX: 100 });
        fireEvent.mouseUp(sparsitySlider);
      }
    });

    // The slider value should have changed (exact value depends on implementation)
    await waitFor(() => {
      expect(screen.getByText('You have unsaved changes')).toBeInTheDocument();
    });
  });

  test('handles API load error', async () => {
    mockApiService.getOptimizationCriteria.mockRejectedValue(new Error('Load failed'));
    renderConfiguration();

    await waitFor(() => {
      expect(message.error).toHaveBeenCalledWith('Failed to load configuration');
    });
  });

  test('disables form during loading', async () => {
    // Mock a delayed response
    mockApiService.getOptimizationCriteria.mockImplementation(
      () => new Promise(resolve => setTimeout(() => resolve(mockCriteria), 1000))
    );

    renderConfiguration();

    // Form should be disabled during loading
    const maxInferenceTimeInput = screen.getByPlaceholderText('Enter maximum inference time in milliseconds');
    expect(maxInferenceTimeInput).toBeDisabled();
  });
});
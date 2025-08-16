import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { BrowserRouter } from 'react-router-dom';
import { message } from 'antd';
import ModelUpload from '../pages/ModelUpload';
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

// Mock react-router-dom navigate
const mockNavigate = jest.fn();
jest.mock('react-router-dom', () => ({
  ...jest.requireActual('react-router-dom'),
  useNavigate: () => mockNavigate,
}));

const renderModelUpload = () => {
  return render(
    <BrowserRouter>
      <ModelUpload />
    </BrowserRouter>
  );
};

describe('ModelUpload Component', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  test('renders upload form', () => {
    renderModelUpload();

    expect(screen.getByText('Upload Model')).toBeInTheDocument();
    expect(screen.getByText('Model Name')).toBeInTheDocument();
    expect(screen.getByText('Model Type')).toBeInTheDocument();
    expect(screen.getByText('Framework')).toBeInTheDocument();
    expect(screen.getByText('Click or drag model file to this area to upload')).toBeInTheDocument();
  });

  test('displays supported formats', () => {
    renderModelUpload();

    expect(screen.getByText('Supported Formats')).toBeInTheDocument();
    expect(screen.getByText('.pth - PyTorch Model')).toBeInTheDocument();
    expect(screen.getByText('.onnx - ONNX Model')).toBeInTheDocument();
  });

  test('handles file selection and auto-detection', async () => {
    renderModelUpload();

    const file = new File(['model content'], 'test-model.pth', { type: 'application/octet-stream' });
    const uploadArea = screen.getByText('Click or drag model file to this area to upload').closest('.ant-upload-drag');
    
    if (uploadArea) {
      const input = uploadArea.querySelector('input[type="file"]');
      if (input) {
        fireEvent.change(input, { target: { files: [file] } });
      }
    }

    await waitFor(() => {
      expect(screen.getByText('test-model.pth')).toBeInTheDocument();
    });
  });

  test('validates required fields', async () => {
    renderModelUpload();

    const uploadButton = screen.getByText('Upload Model');
    fireEvent.click(uploadButton);

    await waitFor(() => {
      expect(screen.getByText('Please enter model name')).toBeInTheDocument();
      expect(screen.getByText('Please select model type')).toBeInTheDocument();
      expect(screen.getByText('Please select framework')).toBeInTheDocument();
    });
  });

  test('handles successful upload', async () => {
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
    renderModelUpload();

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
      expect(mockApiService.uploadModel).toHaveBeenCalledWith(
        file,
        expect.objectContaining({
          name: 'Test Model',
          model_type: 'openvla',
          framework: 'pytorch',
        })
      );
      expect(message.success).toHaveBeenCalledWith('Model uploaded successfully!');
    });

    // Should navigate to dashboard after successful upload
    await waitFor(() => {
      expect(mockNavigate).toHaveBeenCalledWith('/');
    }, { timeout: 2000 });
  });

  test('handles upload failure', async () => {
    mockApiService.uploadModel.mockRejectedValue(new Error('Upload failed'));
    renderModelUpload();

    // Fill required fields and add file
    const nameInput = screen.getByPlaceholderText('Enter a descriptive name for your model');
    fireEvent.change(nameInput, { target: { value: 'Test Model' } });

    const file = new File(['model content'], 'test-model.pth', { type: 'application/octet-stream' });
    const uploadArea = screen.getByText('Click or drag model file to this area to upload').closest('.ant-upload-drag');
    
    if (uploadArea) {
      const input = uploadArea.querySelector('input[type="file"]');
      if (input) {
        fireEvent.change(input, { target: { files: [file] } });
      }
    }

    const uploadButton = screen.getByText('Upload Model');
    fireEvent.click(uploadButton);

    await waitFor(() => {
      expect(message.error).toHaveBeenCalledWith('Failed to upload model. Please try again.');
    });
  });

  test('resets form after successful upload', async () => {
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
    renderModelUpload();

    // Fill form and upload
    const nameInput = screen.getByPlaceholderText('Enter a descriptive name for your model');
    fireEvent.change(nameInput, { target: { value: 'Test Model' } });

    const file = new File(['model content'], 'test-model.pth', { type: 'application/octet-stream' });
    const uploadArea = screen.getByText('Click or drag model file to this area to upload').closest('.ant-upload-drag');
    
    if (uploadArea) {
      const input = uploadArea.querySelector('input[type="file"]');
      if (input) {
        fireEvent.change(input, { target: { files: [file] } });
      }
    }

    const uploadButton = screen.getByText('Upload Model');
    fireEvent.click(uploadButton);

    await waitFor(() => {
      expect(message.success).toHaveBeenCalledWith('Model uploaded successfully!');
    });

    // Form should be reset
    await waitFor(() => {
      expect((nameInput as HTMLInputElement).value).toBe('');
    });
  });

  test('handles reset button', () => {
    renderModelUpload();

    const nameInput = screen.getByPlaceholderText('Enter a descriptive name for your model');
    fireEvent.change(nameInput, { target: { value: 'Test Model' } });

    const resetButton = screen.getByText('Reset');
    fireEvent.click(resetButton);

    expect((nameInput as HTMLInputElement).value).toBe('');
  });
});
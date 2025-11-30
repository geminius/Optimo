import React from 'react';
import { render, screen } from '@testing-library/react';
import { BrowserRouter } from 'react-router-dom';
import { message } from 'antd';
import App from '../App';
import apiService from '../services/api';

// Mock the API service
jest.mock('../services/api', () => ({
  getDashboardStats: jest.fn().mockResolvedValue({
    total_models: 0,
    active_optimizations: 0,
    completed_optimizations: 0,
    average_size_reduction: 0,
    average_speed_improvement: 0,
  }),
  getOptimizationSessions: jest.fn().mockResolvedValue([]),
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

// Mock AuthService
jest.mock('../services/auth', () => ({
  __esModule: true,
  default: {
    getToken: jest.fn().mockReturnValue('mock-token'),
    getUser: jest.fn().mockReturnValue({ id: '1', username: 'testuser', role: 'user' }),
    isTokenValid: jest.fn().mockReturnValue(true),
    getTimeUntilExpiration: jest.fn().mockReturnValue(3600000), // 1 hour
    isTokenExpiringSoon: jest.fn().mockReturnValue(false),
    login: jest.fn(),
    logout: jest.fn(),
    setToken: jest.fn(),
    removeToken: jest.fn(),
  },
}));

// Mock SessionTimeoutWarning component
jest.mock('../components/auth/SessionTimeoutWarning', () => {
  return function MockSessionTimeoutWarning() {
    return null;
  };
});

describe('App Component', () => {
  test('renders without crashing', () => {
    render(
      <BrowserRouter>
        <App />
      </BrowserRouter>
    );
    
    expect(screen.getByText('Robotics Model Optimization Platform')).toBeInTheDocument();
  });

  test('renders navigation menu', () => {
    render(
      <BrowserRouter>
        <App />
      </BrowserRouter>
    );
    
    expect(screen.getByText('Dashboard')).toBeInTheDocument();
    expect(screen.getByText('Upload Model')).toBeInTheDocument();
    expect(screen.getByText('Optimization History')).toBeInTheDocument();
    expect(screen.getByText('Configuration')).toBeInTheDocument();
  });
});
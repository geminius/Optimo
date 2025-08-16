import React from 'react';
import { render, screen } from '@testing-library/react';
import { BrowserRouter } from 'react-router-dom';
import App from '../App';

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
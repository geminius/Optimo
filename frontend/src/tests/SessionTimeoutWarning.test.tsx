/**
 * Tests for SessionTimeoutWarning component
 */

import React from 'react';
import { render, screen, waitFor, act } from '@testing-library/react';
import { BrowserRouter } from 'react-router-dom';
import SessionTimeoutWarning from '../components/auth/SessionTimeoutWarning';
import { AuthProvider } from '../contexts/AuthContext';
import AuthService from '../services/auth';

// Mock AuthService
jest.mock('../services/auth', () => ({
  __esModule: true,
  default: {
    getTimeUntilExpiration: jest.fn(),
    isTokenExpiringSoon: jest.fn(),
    isTokenValid: jest.fn(),
  },
}));

// Mock useAuth hook
jest.mock('../hooks/useAuth', () => ({
  useAuth: () => ({
    isAuthenticated: true,
    logout: jest.fn(),
    refreshToken: jest.fn(),
    user: { id: '1', username: 'testuser', role: 'user' },
    token: 'mock-token',
    isLoading: false,
    error: null,
    login: jest.fn(),
  }),
}));

describe('SessionTimeoutWarning', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('should not show warning when token is not expiring soon', () => {
    // Mock token with plenty of time remaining
    (AuthService.getTimeUntilExpiration as jest.Mock).mockReturnValue(600000); // 10 minutes
    (AuthService.isTokenExpiringSoon as jest.Mock).mockReturnValue(false);

    render(
      <BrowserRouter>
        <SessionTimeoutWarning warningMinutes={5} />
      </BrowserRouter>
    );

    // Warning modal should not be visible
    expect(screen.queryByText(/Session Expiring Soon/i)).not.toBeInTheDocument();
  });

  it('should show warning when token is expiring soon', async () => {
    // Mock token expiring in 4 minutes
    (AuthService.getTimeUntilExpiration as jest.Mock).mockReturnValue(240000); // 4 minutes
    (AuthService.isTokenExpiringSoon as jest.Mock).mockReturnValue(true);

    await act(async () => {
      render(
        <BrowserRouter>
          <SessionTimeoutWarning warningMinutes={5} />
        </BrowserRouter>
      );
    });

    // Wait for the warning to appear
    await waitFor(() => {
      expect(screen.getByText(/Session Expiring Soon/i)).toBeInTheDocument();
    });

    // Check for action buttons
    expect(screen.getByText(/Logout Now/i)).toBeInTheDocument();
    expect(screen.getByText(/Extend Session/i)).toBeInTheDocument();
  });

  it('should format time remaining correctly', async () => {
    // Mock token expiring in 2 minutes 30 seconds
    (AuthService.getTimeUntilExpiration as jest.Mock).mockReturnValue(150000); // 2:30
    (AuthService.isTokenExpiringSoon as jest.Mock).mockReturnValue(true);

    await act(async () => {
      render(
        <BrowserRouter>
          <SessionTimeoutWarning warningMinutes={5} />
        </BrowserRouter>
      );
    });

    await waitFor(() => {
      expect(screen.getByText(/2:30/)).toBeInTheDocument();
    });
  });

  it('should not render when user is not authenticated', () => {
    // Override the mock for this test
    jest.mock('../hooks/useAuth', () => ({
      useAuth: () => ({
        isAuthenticated: false,
        logout: jest.fn(),
        refreshToken: jest.fn(),
        user: null,
        token: null,
        isLoading: false,
        error: null,
        login: jest.fn(),
      }),
    }));

    (AuthService.getTimeUntilExpiration as jest.Mock).mockReturnValue(240000);
    (AuthService.isTokenExpiringSoon as jest.Mock).mockReturnValue(true);

    render(
      <BrowserRouter>
        <SessionTimeoutWarning warningMinutes={5} />
      </BrowserRouter>
    );

    // Warning should not be visible for unauthenticated users
    expect(screen.queryByText(/Session Expiring Soon/i)).not.toBeInTheDocument();
  });
});

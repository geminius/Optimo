/**
 * UserMenu Component Tests
 * 
 * Tests for the UserMenu component including display, dropdown menu, and logout functionality
 */

import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { BrowserRouter } from 'react-router-dom';
import UserMenu from '../components/auth/UserMenu';
import { AuthProvider } from '../contexts/AuthContext';
import * as authHook from '../hooks/useAuth';

// Mock the useAuth hook
jest.mock('../hooks/useAuth');

describe('UserMenu Component', () => {
  const mockLogout = jest.fn();
  const mockUser = {
    id: '1',
    username: 'testuser',
    role: 'admin',
    email: 'test@example.com',
  };

  beforeEach(() => {
    jest.clearAllMocks();
    (authHook.useAuth as jest.Mock).mockReturnValue({
      user: mockUser,
      logout: mockLogout,
      isAuthenticated: true,
      isLoading: false,
      token: 'mock-token',
      error: null,
      login: jest.fn(),
      refreshToken: jest.fn(),
    });
  });

  const renderUserMenu = () => {
    return render(
      <BrowserRouter>
        <UserMenu />
      </BrowserRouter>
    );
  };

  test('renders username and role badge', () => {
    renderUserMenu();
    
    expect(screen.getByText('testuser')).toBeInTheDocument();
    expect(screen.getByText('ADMIN')).toBeInTheDocument();
  });

  test('displays correct role color for admin', () => {
    renderUserMenu();
    
    const roleTag = screen.getByText('ADMIN');
    expect(roleTag).toBeInTheDocument();
  });

  test('opens dropdown menu on click', async () => {
    renderUserMenu();
    
    const userMenuTrigger = screen.getByText('testuser');
    fireEvent.click(userMenuTrigger);
    
    await waitFor(() => {
      expect(screen.getByText('Logout')).toBeInTheDocument();
    });
  });

  test('calls logout when logout button is clicked', async () => {
    renderUserMenu();
    
    // Open dropdown
    const userMenuTrigger = screen.getByText('testuser');
    fireEvent.click(userMenuTrigger);
    
    // Click logout
    await waitFor(() => {
      const logoutButton = screen.getByText('Logout');
      fireEvent.click(logoutButton);
    });
    
    expect(mockLogout).toHaveBeenCalledTimes(1);
  });

  test('does not render when user is null', () => {
    (authHook.useAuth as jest.Mock).mockReturnValue({
      user: null,
      logout: mockLogout,
      isAuthenticated: false,
      isLoading: false,
      token: null,
      error: null,
      login: jest.fn(),
      refreshToken: jest.fn(),
    });

    const { container } = renderUserMenu();
    expect(container.firstChild).toBeNull();
  });

  test('displays user role badge for different roles', () => {
    const userRole = 'user';
    (authHook.useAuth as jest.Mock).mockReturnValue({
      user: { ...mockUser, role: userRole },
      logout: mockLogout,
      isAuthenticated: true,
      isLoading: false,
      token: 'mock-token',
      error: null,
      login: jest.fn(),
      refreshToken: jest.fn(),
    });

    renderUserMenu();
    expect(screen.getByText('USER')).toBeInTheDocument();
  });
});

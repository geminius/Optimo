import React from 'react';
import { render, screen, waitFor } from '@testing-library/react';
import { WebSocketProvider, useWebSocket } from '../contexts/WebSocketContext';
import { io } from 'socket.io-client';
import AuthService from '../services/auth';

// Mock socket.io-client
jest.mock('socket.io-client');
const mockIo = io as jest.MockedFunction<typeof io>;

// Mock AuthService
jest.mock('../services/auth', () => ({
  __esModule: true,
  default: {
    getToken: jest.fn(),
    removeToken: jest.fn(),
  },
}));

// Test component that uses the WebSocket context
const TestComponent: React.FC = () => {
  const { socket, isConnected, subscribeToProgress, unsubscribeFromProgress } = useWebSocket();
  
  return (
    <div>
      <div data-testid="connection-status">
        {isConnected ? 'Connected' : 'Disconnected'}
      </div>
      <div data-testid="socket-exists">
        {socket ? 'Socket exists' : 'No socket'}
      </div>
      <button 
        onClick={() => subscribeToProgress('test-session', () => {})}
        data-testid="subscribe-button"
      >
        Subscribe
      </button>
      <button 
        onClick={() => unsubscribeFromProgress('test-session')}
        data-testid="unsubscribe-button"
      >
        Unsubscribe
      </button>
    </div>
  );
};

describe('WebSocketContext', () => {
  let mockSocket: any;
  const mockGetToken = AuthService.getToken as jest.MockedFunction<typeof AuthService.getToken>;
  const mockRemoveToken = AuthService.removeToken as jest.MockedFunction<typeof AuthService.removeToken>;

  beforeEach(() => {
    mockSocket = {
      on: jest.fn(),
      off: jest.fn(),
      emit: jest.fn(),
      close: jest.fn(),
    };
    mockIo.mockReturnValue(mockSocket);
    // Default: return a token so WebSocket connects
    mockGetToken.mockReturnValue('test-token');
  });

  afterEach(() => {
    jest.clearAllMocks();
  });

  test('provides WebSocket context to children', () => {
    render(
      <WebSocketProvider>
        <TestComponent />
      </WebSocketProvider>
    );

    expect(screen.getByTestId('socket-exists')).toHaveTextContent('Socket exists');
    expect(mockIo).toHaveBeenCalledWith('http://localhost:8000', {
      transports: ['websocket'],
      auth: {
        token: 'test-token',
      },
    });
  });

  test('handles connection events', async () => {
    render(
      <WebSocketProvider>
        <TestComponent />
      </WebSocketProvider>
    );

    // Initially disconnected
    expect(screen.getByTestId('connection-status')).toHaveTextContent('Disconnected');

    // Simulate connection
    const connectHandler = mockSocket.on.mock.calls.find((call: any) => call[0] === 'connect')[1];
    connectHandler();

    await waitFor(() => {
      expect(screen.getByTestId('connection-status')).toHaveTextContent('Connected');
    });
  });

  test('handles disconnection events', async () => {
    render(
      <WebSocketProvider>
        <TestComponent />
      </WebSocketProvider>
    );

    // Simulate connection first
    const connectHandler = mockSocket.on.mock.calls.find((call: any) => call[0] === 'connect')[1];
    connectHandler();

    await waitFor(() => {
      expect(screen.getByTestId('connection-status')).toHaveTextContent('Connected');
    });

    // Simulate disconnection
    const disconnectHandler = mockSocket.on.mock.calls.find((call: any) => call[0] === 'disconnect')[1];
    disconnectHandler();

    await waitFor(() => {
      expect(screen.getByTestId('connection-status')).toHaveTextContent('Disconnected');
    });
  });

  test('handles connection errors', async () => {
    const consoleSpy = jest.spyOn(console, 'error').mockImplementation();
    
    render(
      <WebSocketProvider>
        <TestComponent />
      </WebSocketProvider>
    );

    // Simulate connection error
    const errorHandler = mockSocket.on.mock.calls.find((call: any) => call[0] === 'connect_error')[1];
    const error = new Error('Connection failed');
    errorHandler(error);

    await waitFor(() => {
      expect(screen.getByTestId('connection-status')).toHaveTextContent('Disconnected');
      expect(consoleSpy).toHaveBeenCalledWith('WebSocket connection error:', error);
    });

    consoleSpy.mockRestore();
  });

  test('subscribes to progress updates', () => {
    render(
      <WebSocketProvider>
        <TestComponent />
      </WebSocketProvider>
    );

    const subscribeButton = screen.getByTestId('subscribe-button');
    subscribeButton.click();

    expect(mockSocket.emit).toHaveBeenCalledWith('subscribe_progress', { session_id: 'test-session' });
    expect(mockSocket.on).toHaveBeenCalledWith('progress_test-session', expect.any(Function));
  });

  test('unsubscribes from progress updates', () => {
    render(
      <WebSocketProvider>
        <TestComponent />
      </WebSocketProvider>
    );

    const unsubscribeButton = screen.getByTestId('unsubscribe-button');
    unsubscribeButton.click();

    expect(mockSocket.emit).toHaveBeenCalledWith('unsubscribe_progress', { session_id: 'test-session' });
    expect(mockSocket.off).toHaveBeenCalledWith('progress_test-session');
  });

  test('cleans up socket on unmount', () => {
    const { unmount } = render(
      <WebSocketProvider>
        <TestComponent />
      </WebSocketProvider>
    );

    unmount();

    expect(mockSocket.close).toHaveBeenCalled();
  });

  test('uses custom WebSocket URL from environment', () => {
    const originalEnv = process.env.REACT_APP_WS_URL;
    process.env.REACT_APP_WS_URL = 'ws://custom-url:3000';

    render(
      <WebSocketProvider>
        <TestComponent />
      </WebSocketProvider>
    );

    expect(mockIo).toHaveBeenCalledWith('ws://custom-url:3000', expect.any(Object));

    process.env.REACT_APP_WS_URL = originalEnv;
  });

  test('uses auth token from AuthService', () => {
    const originalEnv = process.env.REACT_APP_WS_URL;
    process.env.REACT_APP_WS_URL = 'http://localhost:8000';
    
    mockGetToken.mockReturnValue('custom-test-token');

    render(
      <WebSocketProvider>
        <TestComponent />
      </WebSocketProvider>
    );

    expect(mockGetToken).toHaveBeenCalled();
    expect(mockIo).toHaveBeenCalledWith('http://localhost:8000', {
      transports: ['websocket'],
      auth: {
        token: 'custom-test-token',
      },
    });

    process.env.REACT_APP_WS_URL = originalEnv;
  });

  test('does not connect when no token is available', () => {
    mockGetToken.mockReturnValue(null);

    render(
      <WebSocketProvider>
        <TestComponent />
      </WebSocketProvider>
    );

    expect(mockGetToken).toHaveBeenCalled();
    expect(mockIo).not.toHaveBeenCalled();
    expect(screen.getByTestId('socket-exists')).toHaveTextContent('No socket');
  });

  test('handles authentication errors and redirects to login', async () => {
    const originalLocation = window.location;
    delete (window as any).location;
    window.location = { ...originalLocation, href: '' } as Location;

    render(
      <WebSocketProvider>
        <TestComponent />
      </WebSocketProvider>
    );

    // Simulate authentication error
    const errorHandler = mockSocket.on.mock.calls.find((call: any) => call[0] === 'connect_error')[1];
    const error = new Error('Authentication failed');
    errorHandler(error);

    await waitFor(() => {
      expect(mockRemoveToken).toHaveBeenCalled();
      expect(window.location.href).toBe('/login');
    });

    window.location = originalLocation;
  });

  test('reconnects after login event', async () => {
    const { rerender } = render(
      <WebSocketProvider>
        <TestComponent />
      </WebSocketProvider>
    );

    // Clear initial calls
    mockIo.mockClear();
    mockSocket.close.mockClear();

    // Simulate login event
    window.dispatchEvent(new Event('auth:login'));

    await waitFor(() => {
      expect(mockSocket.close).toHaveBeenCalled();
      expect(mockIo).toHaveBeenCalled();
    });
  });

  test('disconnects after logout event', async () => {
    render(
      <WebSocketProvider>
        <TestComponent />
      </WebSocketProvider>
    );

    // Clear initial calls
    mockSocket.close.mockClear();

    // Simulate logout event
    window.dispatchEvent(new Event('auth:logout'));

    await waitFor(() => {
      expect(mockSocket.close).toHaveBeenCalled();
    });
  });

  test('throws error when useWebSocket is used outside provider', () => {
    const consoleSpy = jest.spyOn(console, 'error').mockImplementation();

    expect(() => {
      render(<TestComponent />);
    }).toThrow('useWebSocket must be used within a WebSocketProvider');

    consoleSpy.mockRestore();
  });
});
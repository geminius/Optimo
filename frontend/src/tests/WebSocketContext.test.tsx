import React from 'react';
import { render, screen, waitFor } from '@testing-library/react';
import { WebSocketProvider, useWebSocket } from '../contexts/WebSocketContext';
import { io } from 'socket.io-client';

// Mock socket.io-client
jest.mock('socket.io-client');
const mockIo = io as jest.MockedFunction<typeof io>;

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

  beforeEach(() => {
    mockSocket = {
      on: jest.fn(),
      off: jest.fn(),
      emit: jest.fn(),
      close: jest.fn(),
    };
    mockIo.mockReturnValue(mockSocket);
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
        token: null, // localStorage.getItem returns null in test environment
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
    const connectHandler = mockSocket.on.mock.calls.find(call => call[0] === 'connect')[1];
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
    const connectHandler = mockSocket.on.mock.calls.find(call => call[0] === 'connect')[1];
    connectHandler();

    await waitFor(() => {
      expect(screen.getByTestId('connection-status')).toHaveTextContent('Connected');
    });

    // Simulate disconnection
    const disconnectHandler = mockSocket.on.mock.calls.find(call => call[0] === 'disconnect')[1];
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
    const errorHandler = mockSocket.on.mock.calls.find(call => call[0] === 'connect_error')[1];
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

  test('uses auth token from localStorage', () => {
    const mockGetItem = jest.spyOn(Storage.prototype, 'getItem');
    mockGetItem.mockReturnValue('test-token');

    render(
      <WebSocketProvider>
        <TestComponent />
      </WebSocketProvider>
    );

    expect(mockIo).toHaveBeenCalledWith('http://localhost:8000', {
      transports: ['websocket'],
      auth: {
        token: 'test-token',
      },
    });

    mockGetItem.mockRestore();
  });

  test('throws error when useWebSocket is used outside provider', () => {
    const consoleSpy = jest.spyOn(console, 'error').mockImplementation();

    expect(() => {
      render(<TestComponent />);
    }).toThrow('useWebSocket must be used within a WebSocketProvider');

    consoleSpy.mockRestore();
  });
});
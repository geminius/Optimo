import React, { createContext, useContext, useEffect, useState, ReactNode, useCallback } from 'react';
import { io, Socket } from 'socket.io-client';
import { ProgressUpdate } from '../types';
import AuthService from '../services/auth';
import { logger } from '../utils/logger';

interface WebSocketContextType {
  socket: Socket | null;
  isConnected: boolean;
  connectionError: string | null;
  subscribeToProgress: (sessionId: string, callback: (update: ProgressUpdate) => void) => void;
  unsubscribeFromProgress: (sessionId: string) => void;
  reconnect: () => void;
  disconnect: () => void;
}

const WebSocketContext = createContext<WebSocketContextType | undefined>(undefined);

interface WebSocketProviderProps {
  children: ReactNode;
}

export const WebSocketProvider: React.FC<WebSocketProviderProps> = ({ children }) => {
  const [socket, setSocket] = useState<Socket | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [connectionError, setConnectionError] = useState<string | null>(null);

  const disconnect = useCallback(() => {
    if (socket) {
      socket.close();
      setSocket(null);
      setIsConnected(false);
      setConnectionError(null);
    }
  }, [socket]);

  const connect = useCallback(() => {
    // Get token from AuthService
    const token = AuthService.getToken();
    
    // Only connect if token exists
    if (!token) {
      logger.log('No token available, skipping WebSocket connection');
      setConnectionError('Not authenticated');
      return;
    }

    const socketUrl = process.env.REACT_APP_WS_URL || 'http://localhost:8000';
    const newSocket = io(socketUrl, {
      transports: ['websocket'],
      auth: {
        token: token,
      },
    });

    newSocket.on('connect', () => {
      logger.log('WebSocket connected');
      setIsConnected(true);
      setConnectionError(null);
    });

    newSocket.on('disconnect', () => {
      logger.log('WebSocket disconnected');
      setIsConnected(false);
    });

    newSocket.on('connect_error', (error) => {
      logger.error('WebSocket connection error:', error);
      setIsConnected(false);
      
      // Check if error is authentication-related
      const errorMessage = error.message || '';
      if (errorMessage.includes('Authentication') || 
          errorMessage.includes('Unauthorized') || 
          errorMessage.includes('401')) {
        logger.log('WebSocket authentication failed, clearing token');
        setConnectionError('Authentication failed');
        
        // Clear token and redirect to login
        AuthService.removeToken();
        window.location.href = '/login';
      } else {
        setConnectionError(errorMessage || 'Connection failed');
      }
    });

    setSocket(newSocket);
  }, []);

  const reconnect = useCallback(() => {
    disconnect();
    setTimeout(() => {
      connect();
    }, 100);
  }, [disconnect, connect]);

  useEffect(() => {
    connect();

    // Listen for auth events to reconnect/disconnect WebSocket
    const handleLogin = () => {
      logger.log('Auth login event received, reconnecting WebSocket');
      reconnect();
    };

    const handleLogout = () => {
      logger.log('Auth logout event received, disconnecting WebSocket');
      disconnect();
    };

    window.addEventListener('auth:login', handleLogin);
    window.addEventListener('auth:logout', handleLogout);

    return () => {
      if (socket) {
        socket.close();
      }
      window.removeEventListener('auth:login', handleLogin);
      window.removeEventListener('auth:logout', handleLogout);
    };
  }, [connect, reconnect, disconnect, socket]);

  const subscribeToProgress = (sessionId: string, callback: (update: ProgressUpdate) => void) => {
    if (socket) {
      socket.emit('subscribe_progress', { session_id: sessionId });
      socket.on(`progress_${sessionId}`, callback);
    }
  };

  const unsubscribeFromProgress = (sessionId: string) => {
    if (socket) {
      socket.emit('unsubscribe_progress', { session_id: sessionId });
      socket.off(`progress_${sessionId}`);
    }
  };

  const value: WebSocketContextType = {
    socket,
    isConnected,
    connectionError,
    subscribeToProgress,
    unsubscribeFromProgress,
    reconnect,
    disconnect,
  };

  return (
    <WebSocketContext.Provider value={value}>
      {children}
    </WebSocketContext.Provider>
  );
};

export const useWebSocket = (): WebSocketContextType => {
  const context = useContext(WebSocketContext);
  if (context === undefined) {
    throw new Error('useWebSocket must be used within a WebSocketProvider');
  }
  return context;
};
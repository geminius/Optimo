import React, { createContext, useContext, useEffect, useState, ReactNode } from 'react';
import { io, Socket } from 'socket.io-client';
import { ProgressUpdate } from '../types';

interface WebSocketContextType {
  socket: Socket | null;
  isConnected: boolean;
  subscribeToProgress: (sessionId: string, callback: (update: ProgressUpdate) => void) => void;
  unsubscribeFromProgress: (sessionId: string) => void;
}

const WebSocketContext = createContext<WebSocketContextType | undefined>(undefined);

interface WebSocketProviderProps {
  children: ReactNode;
}

export const WebSocketProvider: React.FC<WebSocketProviderProps> = ({ children }) => {
  const [socket, setSocket] = useState<Socket | null>(null);
  const [isConnected, setIsConnected] = useState(false);

  useEffect(() => {
    const socketUrl = process.env.REACT_APP_WS_URL || 'http://localhost:8000';
    const newSocket = io(socketUrl, {
      transports: ['websocket'],
      auth: {
        token: localStorage.getItem('auth_token'),
      },
    });

    newSocket.on('connect', () => {
      console.log('WebSocket connected');
      setIsConnected(true);
    });

    newSocket.on('disconnect', () => {
      console.log('WebSocket disconnected');
      setIsConnected(false);
    });

    newSocket.on('connect_error', (error) => {
      console.error('WebSocket connection error:', error);
      setIsConnected(false);
    });

    setSocket(newSocket);

    return () => {
      newSocket.close();
    };
  }, []);

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
    subscribeToProgress,
    unsubscribeFromProgress,
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
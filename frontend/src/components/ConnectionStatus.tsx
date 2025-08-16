import React from 'react';
import { Badge } from 'antd';
import { WifiOutlined, DisconnectOutlined } from '@ant-design/icons';
import { useWebSocket } from '../contexts/WebSocketContext';

const ConnectionStatus: React.FC = () => {
  const { isConnected } = useWebSocket();

  return (
    <div className={`connection-status ${isConnected ? 'connected' : 'disconnected'}`}>
      <Badge 
        status={isConnected ? 'success' : 'error'} 
        text={
          <span style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
            {isConnected ? <WifiOutlined /> : <DisconnectOutlined />}
            {isConnected ? 'Connected' : 'Disconnected'}
          </span>
        }
      />
    </div>
  );
};

export default ConnectionStatus;
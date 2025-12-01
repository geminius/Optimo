import React from 'react';
import { Badge, Tooltip } from 'antd';
import { WifiOutlined, DisconnectOutlined } from '@ant-design/icons';
import { useWebSocket } from '../contexts/WebSocketContext';

const ConnectionStatus: React.FC = () => {
  const { isConnected, connectionError } = useWebSocket();

  const statusText = isConnected ? 'Connected' : 'Disconnected';
  const tooltipText = connectionError 
    ? `Connection Error: ${connectionError}` 
    : statusText;

  return (
    <div className={`connection-status ${isConnected ? 'connected' : 'disconnected'}`}>
      <Tooltip title={tooltipText}>
        <Badge 
          status={isConnected ? 'success' : 'error'} 
          text={
            <span style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
              {isConnected ? <WifiOutlined /> : <DisconnectOutlined />}
              {statusText}
            </span>
          }
        />
      </Tooltip>
    </div>
  );
};

export default ConnectionStatus;
import React from 'react';
import { Spin } from 'antd';
import { LoadingOutlined } from '@ant-design/icons';

interface LoadingSpinnerProps {
  size?: 'small' | 'default' | 'large';
  tip?: string;
  spinning?: boolean;
  children?: React.ReactNode;
}

const LoadingSpinner: React.FC<LoadingSpinnerProps> = ({ 
  size = 'default', 
  tip = 'Loading...', 
  spinning = true,
  children 
}) => {
  const antIcon = <LoadingOutlined style={{ fontSize: size === 'large' ? 24 : 16 }} spin />;

  if (children) {
    return (
      <Spin indicator={antIcon} tip={tip} spinning={spinning} size={size}>
        {children}
      </Spin>
    );
  }

  return (
    <div style={{ 
      display: 'flex', 
      justifyContent: 'center', 
      alignItems: 'center', 
      minHeight: 200,
      flexDirection: 'column',
      gap: 16
    }}>
      <Spin indicator={antIcon} size={size} />
      <div style={{ color: '#666', fontSize: 14 }}>{tip}</div>
    </div>
  );
};

export default LoadingSpinner;
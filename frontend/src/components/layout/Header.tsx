import React from 'react';
import { Layout } from 'antd';
import ConnectionStatus from '../ConnectionStatus';
import UserMenu from '../auth/UserMenu';
import { useAuth } from '../../hooks/useAuth';
import './Header.css';

const { Header: AntHeader } = Layout;

const Header: React.FC = () => {
  const { isAuthenticated } = useAuth();

  return (
    <AntHeader 
      className="site-layout-background app-header" 
      style={{ 
        padding: 0, 
        background: '#fff', 
        position: 'relative', 
        display: 'flex', 
        alignItems: 'center', 
        justifyContent: 'space-between' 
      }}
    >
      <div className="header-title" style={{ padding: '0 24px', fontSize: '18px', fontWeight: 'bold' }}>
        <span className="header-title-full">Robotics Model Optimization Platform</span>
        <span className="header-title-short">RMOP</span>
      </div>
      <div className="header-actions" style={{ display: 'flex', alignItems: 'center', gap: '16px', paddingRight: '24px' }}>
        <ConnectionStatus />
        {isAuthenticated && <UserMenu />}
      </div>
    </AntHeader>
  );
};

export default Header;

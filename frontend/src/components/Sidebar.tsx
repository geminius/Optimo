import React from 'react';
import { Layout, Menu } from 'antd';
import { useNavigate, useLocation } from 'react-router-dom';
import {
  DashboardOutlined,
  UploadOutlined,
  HistoryOutlined,
  SettingOutlined,
} from '@ant-design/icons';

const { Sider } = Layout;

const Sidebar: React.FC = () => {
  const navigate = useNavigate();
  const location = useLocation();

  const menuItems = [
    {
      key: '/',
      icon: <DashboardOutlined />,
      label: 'Dashboard',
    },
    {
      key: '/upload',
      icon: <UploadOutlined />,
      label: 'Upload Model',
    },
    {
      key: '/history',
      icon: <HistoryOutlined />,
      label: 'Optimization History',
    },
    {
      key: '/config',
      icon: <SettingOutlined />,
      label: 'Configuration',
    },
  ];

  const handleMenuClick = (key: string) => {
    navigate(key);
  };

  return (
    <Sider
      collapsible
      theme="dark"
      style={{
        overflow: 'auto',
        height: '100vh',
        position: 'fixed',
        left: 0,
        top: 0,
        bottom: 0,
      }}
    >
      <div className="logo">
        RMOP
      </div>
      <Menu
        theme="dark"
        mode="inline"
        selectedKeys={[location.pathname]}
        items={menuItems}
        onClick={({ key }) => handleMenuClick(key)}
      />
    </Sider>
  );
};

export default Sidebar;
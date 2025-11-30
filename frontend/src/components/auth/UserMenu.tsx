/**
 * UserMenu Component
 * 
 * Displays authenticated user information with a dropdown menu containing
 * user options and logout functionality. Shows username and role badge.
 */

import React from 'react';
import { Dropdown, Avatar, Tag, Space, message } from 'antd';
import { UserOutlined, LogoutOutlined, SettingOutlined } from '@ant-design/icons';
import type { MenuProps } from 'antd';
import { useAuth } from '../../hooks/useAuth';
import { logger } from '../../utils/logger';

/**
 * UserMenu component that displays user info and provides logout functionality
 */
const UserMenu: React.FC = () => {
  const { user, logout } = useAuth();

  /**
   * Handle logout action
   * Sub-task 11.2: Implement logout functionality
   */
  const handleLogout = () => {
    try {
      // Call logout from auth context
      logout();
      
      // Show confirmation message
      message.success('Successfully logged out');
    } catch (error) {
      logger.error('Logout error:', error);
      message.error('Failed to logout. Please try again.');
    }
  };

  /**
   * Get role badge color based on user role
   */
  const getRoleColor = (role: string): string => {
    switch (role.toLowerCase()) {
      case 'admin':
        return 'red';
      case 'user':
        return 'blue';
      case 'viewer':
        return 'green';
      default:
        return 'default';
    }
  };

  /**
   * Dropdown menu items
   */
  const menuItems: MenuProps['items'] = [
    {
      key: 'profile',
      icon: <UserOutlined />,
      label: 'Profile',
      disabled: true, // Placeholder for future implementation
    },
    {
      key: 'settings',
      icon: <SettingOutlined />,
      label: 'Settings',
      disabled: true, // Placeholder for future implementation
    },
    {
      type: 'divider',
    },
    {
      key: 'logout',
      icon: <LogoutOutlined />,
      label: 'Logout',
      danger: true,
      onClick: handleLogout,
    },
  ];

  // Don't render if no user is authenticated
  if (!user) {
    return null;
  }

  return (
    <Dropdown menu={{ items: menuItems }} placement="bottomRight" trigger={['click']}>
      <Space
        style={{
          cursor: 'pointer',
          padding: '0 16px',
          height: '100%',
          display: 'flex',
          alignItems: 'center',
        }}
      >
        <Avatar
          size="small"
          icon={<UserOutlined />}
          style={{ backgroundColor: '#1890ff' }}
        />
        <span style={{ fontWeight: 500 }}>{user.username}</span>
        <Tag color={getRoleColor(user.role)} style={{ margin: 0 }}>
          {user.role.toUpperCase()}
        </Tag>
      </Space>
    </Dropdown>
  );
};

export default UserMenu;

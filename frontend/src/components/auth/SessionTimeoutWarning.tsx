/**
 * Session Timeout Warning Component
 * 
 * Displays a warning notification when the user's session is about to expire.
 * Provides options to extend the session or logout.
 * Automatically logs out when the token expires.
 */

import React, { useEffect, useState, useCallback } from 'react';
import { Modal, Button, Progress } from 'antd';
import { ExclamationCircleOutlined } from '@ant-design/icons';
import AuthService from '../../services/auth';
import { useAuth } from '../../hooks/useAuth';
import { logger } from '../../utils/logger';

interface SessionTimeoutWarningProps {
  warningMinutes?: number; // Minutes before expiration to show warning (default: 5)
}

/**
 * SessionTimeoutWarning component
 * Monitors token expiration and shows warning dialog
 */
const SessionTimeoutWarning: React.FC<SessionTimeoutWarningProps> = ({ 
  warningMinutes = 5 
}) => {
  const { isAuthenticated, logout, refreshToken } = useAuth();
  const [showWarning, setShowWarning] = useState(false);
  const [timeRemaining, setTimeRemaining] = useState(0);
  const [isExtending, setIsExtending] = useState(false);

  /**
   * Format time remaining in MM:SS format
   */
  const formatTimeRemaining = useCallback((milliseconds: number): string => {
    const totalSeconds = Math.floor(milliseconds / 1000);
    const minutes = Math.floor(totalSeconds / 60);
    const seconds = totalSeconds % 60;
    return `${minutes}:${seconds.toString().padStart(2, '0')}`;
  }, []);

  /**
   * Calculate progress percentage for visual indicator
   */
  const calculateProgress = useCallback((remaining: number): number => {
    const warningThresholdMs = warningMinutes * 60 * 1000;
    const percentage = (remaining / warningThresholdMs) * 100;
    return Math.max(0, Math.min(100, percentage));
  }, [warningMinutes]);

  /**
   * Handle session extension
   */
  const handleExtendSession = useCallback(async () => {
    setIsExtending(true);
    try {
      // Attempt to refresh token
      await refreshToken();
      setShowWarning(false);
      setTimeRemaining(0);
    } catch (error) {
      logger.error('Failed to extend session:', error);
      // If refresh fails, user will need to re-login
      logout();
    } finally {
      setIsExtending(false);
    }
  }, [refreshToken, logout]);

  /**
   * Handle logout from warning dialog
   */
  const handleLogout = useCallback(() => {
    setShowWarning(false);
    logout();
  }, [logout]);

  /**
   * Monitor token expiration and show warning
   */
  useEffect(() => {
    // Only monitor if user is authenticated
    if (!isAuthenticated) {
      setShowWarning(false);
      setTimeRemaining(0);
      return;
    }

    // Check token expiration every second
    const checkInterval = setInterval(() => {
      const remaining = AuthService.getTimeUntilExpiration();
      
      // Token has expired - auto logout
      if (remaining === 0) {
        setShowWarning(false);
        logout();
        return;
      }

      // Check if we should show warning
      const shouldShowWarning = AuthService.isTokenExpiringSoon(warningMinutes);
      
      if (shouldShowWarning) {
        setShowWarning(true);
        setTimeRemaining(remaining);
      } else {
        setShowWarning(false);
        setTimeRemaining(0);
      }
    }, 1000); // Check every second for accurate countdown

    // Cleanup interval on unmount
    return () => {
      clearInterval(checkInterval);
    };
  }, [isAuthenticated, warningMinutes, logout]);

  // Don't render anything if not showing warning
  if (!showWarning || !isAuthenticated) {
    return null;
  }

  const progress = calculateProgress(timeRemaining);
  const progressStatus = progress > 50 ? 'normal' : progress > 25 ? 'exception' : 'exception';

  return (
    <Modal
      title={
        <span>
          <ExclamationCircleOutlined style={{ color: '#faad14', marginRight: 8 }} />
          Session Expiring Soon
        </span>
      }
      open={showWarning}
      closable={false}
      maskClosable={false}
      footer={[
        <Button key="logout" onClick={handleLogout}>
          Logout Now
        </Button>,
        <Button
          key="extend"
          type="primary"
          loading={isExtending}
          onClick={handleExtendSession}
        >
          Extend Session
        </Button>,
      ]}
      width={400}
    >
      <div style={{ textAlign: 'center' }}>
        <p style={{ fontSize: '16px', marginBottom: '16px' }}>
          Your session will expire in:
        </p>
        <p style={{ fontSize: '32px', fontWeight: 'bold', margin: '16px 0', color: '#faad14' }}>
          {formatTimeRemaining(timeRemaining)}
        </p>
        <Progress
          percent={progress}
          status={progressStatus}
          showInfo={false}
          strokeColor={{
            '0%': '#52c41a',
            '50%': '#faad14',
            '100%': '#ff4d4f',
          }}
        />
        <p style={{ marginTop: '16px', color: '#8c8c8c' }}>
          Click "Extend Session" to continue working, or "Logout Now" to end your session.
        </p>
      </div>
    </Modal>
  );
};

export default SessionTimeoutWarning;

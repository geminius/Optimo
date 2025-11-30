/**
 * ProtectedRoute Component
 * 
 * Route guard component that ensures only authenticated users can access protected routes.
 * Redirects unauthenticated users to the login page and optionally checks user roles.
 * 
 * Requirements: 5.1, 5.2, 5.4, 6.5
 */

import React from 'react';
import { Navigate } from 'react-router-dom';
import { useAuth } from '../../hooks/useAuth';
import LoadingSpinner from '../LoadingSpinner';
import { Alert } from 'antd';

/**
 * Props for ProtectedRoute component
 */
interface ProtectedRouteProps {
  children: React.ReactNode;
  requiredRole?: string;
}

/**
 * ProtectedRoute component that guards routes requiring authentication
 * 
 * @param children - Child components to render if authenticated
 * @param requiredRole - Optional role required to access the route
 * 
 * @example
 * ```tsx
 * <ProtectedRoute>
 *   <Dashboard />
 * </ProtectedRoute>
 * 
 * <ProtectedRoute requiredRole="admin">
 *   <AdminPanel />
 * </ProtectedRoute>
 * ```
 */
const ProtectedRoute: React.FC<ProtectedRouteProps> = ({ children, requiredRole }) => {
  const { isAuthenticated, isLoading, user } = useAuth();

  // Show loading spinner while checking authentication status
  // Requirement 5.4: Show loading state while checking auth
  if (isLoading) {
    return <LoadingSpinner tip="Checking authentication..." />;
  }

  // Redirect to login if not authenticated
  // Requirement 5.1: Redirect unauthenticated users to login
  // Requirement 5.2: Allow access when authenticated
  if (!isAuthenticated) {
    return <Navigate to="/login" replace />;
  }

  // Check role-based access control if requiredRole is specified
  // Requirement 6.5: Check user role for admin-specific routes
  if (requiredRole && user?.role !== requiredRole) {
    return (
      <div style={{ 
        display: 'flex', 
        justifyContent: 'center', 
        alignItems: 'center', 
        minHeight: '400px',
        padding: '20px'
      }}>
        <Alert
          message="Access Denied"
          description={`You don't have permission to access this page. Required role: ${requiredRole}`}
          type="error"
          showIcon
        />
      </div>
    );
  }

  // Render children if authenticated and authorized
  return <>{children}</>;
};

export default ProtectedRoute;

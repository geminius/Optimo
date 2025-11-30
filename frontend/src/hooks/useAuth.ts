/**
 * Custom useAuth Hook
 * 
 * Provides access to authentication context throughout the application.
 * Must be used within an AuthProvider component.
 */

import { useContext } from 'react';
import AuthContext from '../contexts/AuthContext';
import { AuthContextType } from '../types/auth';

/**
 * Custom hook to access authentication context
 * 
 * @returns AuthContextType - Authentication state and methods
 * @throws Error if used outside of AuthProvider
 * 
 * @example
 * ```tsx
 * function MyComponent() {
 *   const { user, isAuthenticated, login, logout } = useAuth();
 *   
 *   if (!isAuthenticated) {
 *     return <div>Please log in</div>;
 *   }
 *   
 *   return <div>Welcome, {user?.username}!</div>;
 * }
 * ```
 */
export const useAuth = (): AuthContextType => {
  const context = useContext(AuthContext);
  
  if (context === undefined) {
    throw new Error(
      'useAuth must be used within an AuthProvider. ' +
      'Make sure your component is wrapped with <AuthProvider>.'
    );
  }
  
  return context;
};

export default useAuth;

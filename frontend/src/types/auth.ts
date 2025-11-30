/**
 * Authentication Types and Interfaces
 * 
 * Defines all TypeScript types and interfaces for the authentication system,
 * including user data, authentication state, API responses, and context types.
 */

/**
 * User interface representing an authenticated user
 */
export interface User {
  id: string;
  username: string;
  role: string;
  email?: string;
}

/**
 * Authentication state interface for managing auth status
 */
export interface AuthState {
  user: User | null;
  token: string | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  error: string | null;
}

/**
 * Login response from the backend API
 */
export interface LoginResponse {
  access_token: string;
  token_type: string;
  expires_in: number;
  user: User;
}

/**
 * Login request payload
 */
export interface LoginRequest {
  username: string;
  password: string;
}

/**
 * Authentication context type for React Context API
 */
export interface AuthContextType {
  user: User | null;
  token: string | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  error: string | null;
  login: (username: string, password: string, rememberMe?: boolean) => Promise<void>;
  logout: () => void;
  refreshToken: () => Promise<void>;
}

/**
 * Stored authentication data in localStorage
 */
export interface StoredAuth {
  token: string;
  user: User;
  expiresAt: string; // ISO timestamp
  rememberMe: boolean;
}

/**
 * JWT token payload structure (decoded)
 */
export interface TokenPayload {
  sub: string; // Subject (user ID)
  username: string;
  role: string;
  exp: number; // Expiration timestamp
  iat: number; // Issued at timestamp
}

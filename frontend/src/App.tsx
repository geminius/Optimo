import React from 'react';
import { Routes, Route } from 'react-router-dom';
import { Layout } from 'antd';
import Sidebar from './components/Sidebar';
import Dashboard from './pages/Dashboard';
import ModelUpload from './pages/ModelUpload';
import OptimizationHistory from './pages/OptimizationHistory';
import Configuration from './pages/Configuration';
import ErrorBoundary from './components/ErrorBoundary';
import { WebSocketProvider } from './contexts/WebSocketContext';
import { AuthProvider } from './contexts/AuthContext';
import LoginPage from './components/auth/LoginPage';
import ProtectedRoute from './components/auth/ProtectedRoute';
import Header from './components/layout/Header';
import SessionTimeoutWarning from './components/auth/SessionTimeoutWarning';

const { Content } = Layout;

const App: React.FC = () => {
  return (
    <ErrorBoundary>
      <AuthProvider>
        {/* Session timeout warning - shows when token is about to expire */}
        <SessionTimeoutWarning warningMinutes={5} />
        
        <Routes>
          {/* Public route - Login page */}
          <Route path="/login" element={<LoginPage />} />
          
          {/* Protected routes - Main application */}
          <Route path="/*" element={
            <ProtectedRoute>
              <WebSocketProvider>
                <Layout style={{ minHeight: '100vh' }}>
                  <Sidebar />
                  <Layout className="site-layout">
                    <Header />
                    <Content style={{ margin: '24px 16px', padding: 24, background: '#fff', borderRadius: 6 }}>
                      <ErrorBoundary>
                        <Routes>
                          <Route path="/" element={<Dashboard />} />
                          <Route path="/dashboard" element={<Dashboard />} />
                          <Route path="/upload" element={<ModelUpload />} />
                          <Route path="/history" element={<OptimizationHistory />} />
                          <Route path="/config" element={<Configuration />} />
                        </Routes>
                      </ErrorBoundary>
                    </Content>
                  </Layout>
                </Layout>
              </WebSocketProvider>
            </ProtectedRoute>
          } />
        </Routes>
      </AuthProvider>
    </ErrorBoundary>
  );
};

export default App;
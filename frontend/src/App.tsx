import React from 'react';
import { Routes, Route } from 'react-router-dom';
import { Layout } from 'antd';
import Sidebar from './components/Sidebar';
import Dashboard from './pages/Dashboard';
import ModelUpload from './pages/ModelUpload';
import OptimizationHistory from './pages/OptimizationHistory';
import Configuration from './pages/Configuration';
import ErrorBoundary from './components/ErrorBoundary';
import ConnectionStatus from './components/ConnectionStatus';
import { WebSocketProvider } from './contexts/WebSocketContext';

const { Header, Content } = Layout;

const App: React.FC = () => {
  return (
    <ErrorBoundary>
      <WebSocketProvider>
        <Layout style={{ minHeight: '100vh' }}>
          <Sidebar />
          <Layout className="site-layout">
            <Header className="site-layout-background" style={{ padding: 0, background: '#fff', position: 'relative' }}>
              <div style={{ padding: '0 24px', fontSize: '18px', fontWeight: 'bold' }}>
                Robotics Model Optimization Platform
              </div>
              <ConnectionStatus />
            </Header>
            <Content style={{ margin: '24px 16px', padding: 24, background: '#fff', borderRadius: 6 }}>
              <ErrorBoundary>
                <Routes>
                  <Route path="/" element={<Dashboard />} />
                  <Route path="/upload" element={<ModelUpload />} />
                  <Route path="/history" element={<OptimizationHistory />} />
                  <Route path="/config" element={<Configuration />} />
                </Routes>
              </ErrorBoundary>
            </Content>
          </Layout>
        </Layout>
      </WebSocketProvider>
    </ErrorBoundary>
  );
};

export default App;
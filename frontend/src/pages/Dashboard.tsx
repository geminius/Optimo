import React, { useState, useEffect } from 'react';
import { Row, Col, Card, Statistic, Table, Progress, Tag, Button, message } from 'antd';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { PlayCircleOutlined, PauseCircleOutlined, StopOutlined } from '@ant-design/icons';
import apiService from '../services/api';
import { useWebSocket } from '../contexts/WebSocketContext';
import { OptimizationSession, ProgressUpdate } from '../types';
import { logger } from '../utils/logger';

const Dashboard: React.FC = () => {
  const [stats, setStats] = useState({
    total_models: 0,
    active_optimizations: 0,
    completed_optimizations: 0,
    average_size_reduction: 0,
    average_speed_improvement: 0,
  });
  const [activeSessions, setActiveSessions] = useState<OptimizationSession[]>([]);
  const [performanceData, setPerformanceData] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const { subscribeToProgress, unsubscribeFromProgress } = useWebSocket();

  useEffect(() => {
    loadDashboardData();
    const interval = setInterval(loadDashboardData, 30000); // Refresh every 30 seconds
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    // Subscribe to progress updates for active sessions
    activeSessions.forEach(session => {
      if (session.status === 'running') {
        subscribeToProgress(session.id, handleProgressUpdate);
      }
    });

    return () => {
      activeSessions.forEach(session => {
        unsubscribeFromProgress(session.id);
      });
    };
  }, [activeSessions, subscribeToProgress, unsubscribeFromProgress]);

  const loadDashboardData = async () => {
    try {
      const [statsData, sessionsData] = await Promise.all([
        apiService.getDashboardStats(),
        apiService.getOptimizationSessions(),
      ]);

      setStats(statsData);
      setActiveSessions(sessionsData.filter(s => s.status === 'running'));
      
      // Generate mock performance data for the chart
      const mockData = Array.from({ length: 7 }, (_, i) => ({
        day: `Day ${i + 1}`,
        optimizations: Math.floor(Math.random() * 10) + 1,
        avgReduction: Math.floor(Math.random() * 30) + 20,
      }));
      setPerformanceData(mockData);
    } catch (error) {
      logger.error('Failed to load dashboard data:', error);
      message.error('Failed to load dashboard data');
    } finally {
      setLoading(false);
    }
  };

  const handleProgressUpdate = (update: ProgressUpdate) => {
    setActiveSessions(prev => 
      prev.map(session => 
        session.id === update.session_id 
          ? { ...session, progress: update.progress, status: update.status as any }
          : session
      )
    );
  };

  const handleSessionAction = async (sessionId: string, action: 'pause' | 'resume' | 'cancel') => {
    try {
      switch (action) {
        case 'pause':
          await apiService.pauseOptimization(sessionId);
          break;
        case 'resume':
          await apiService.resumeOptimization(sessionId);
          break;
        case 'cancel':
          await apiService.cancelOptimization(sessionId);
          break;
      }
      message.success(`Optimization ${action}d successfully`);
      loadDashboardData();
    } catch (error) {
      logger.error(`Failed to ${action} optimization:`, error);
      message.error(`Failed to ${action} optimization`);
    }
  };

  const columns = [
    {
      title: 'Session ID',
      dataIndex: 'id',
      key: 'id',
      render: (id: string) => id.substring(0, 8) + '...',
      responsive: ['sm'] as any,
    },
    {
      title: 'Model',
      dataIndex: 'model_id',
      key: 'model_id',
      render: (modelId: string) => modelId.substring(0, 8) + '...',
    },
    {
      title: 'Status',
      dataIndex: 'status',
      key: 'status',
      render: (status: string) => {
        const color = status === 'running' ? 'blue' : status === 'completed' ? 'green' : 'red';
        return <Tag color={color}>{status.toUpperCase()}</Tag>;
      },
    },
    {
      title: 'Progress',
      dataIndex: 'progress',
      key: 'progress',
      render: (progress: number) => (
        <Progress percent={progress || 0} size="small" />
      ),
    },
    {
      title: 'Techniques',
      dataIndex: 'criteria',
      key: 'techniques',
      render: (criteria: any) => (
        <div>
          {criteria?.techniques?.map((tech: string) => (
            <Tag key={tech}>{tech}</Tag>
          ))}
        </div>
      ),
      responsive: ['md'] as any,
    },
    {
      title: 'Actions',
      key: 'actions',
      fixed: 'right' as any,
      width: 200,
      render: (_: any, record: OptimizationSession) => (
        <div>
          {record.status === 'running' && (
            <>
              <Button
                size="small"
                icon={<PauseCircleOutlined />}
                onClick={() => handleSessionAction(record.id, 'pause')}
                style={{ marginRight: 8 }}
              >
                Pause
              </Button>
              <Button
                size="small"
                danger
                icon={<StopOutlined />}
                onClick={() => handleSessionAction(record.id, 'cancel')}
              >
                Cancel
              </Button>
            </>
          )}
          {record.status === 'paused' && (
            <Button
              size="small"
              type="primary"
              icon={<PlayCircleOutlined />}
              onClick={() => handleSessionAction(record.id, 'resume')}
            >
              Resume
            </Button>
          )}
        </div>
      ),
    },
  ];

  return (
    <div>
      <h1>Dashboard</h1>
      
      {/* Statistics Cards */}
      <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
        <Col xs={24} sm={12} md={12} lg={6}>
          <Card>
            <Statistic
              title="Total Models"
              value={stats.total_models}
              loading={loading}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} md={12} lg={6}>
          <Card>
            <Statistic
              title="Active Optimizations"
              value={stats.active_optimizations}
              loading={loading}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} md={12} lg={6}>
          <Card>
            <Statistic
              title="Avg Size Reduction"
              value={stats.average_size_reduction}
              suffix="%"
              loading={loading}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} md={12} lg={6}>
          <Card>
            <Statistic
              title="Avg Speed Improvement"
              value={stats.average_speed_improvement}
              suffix="%"
              loading={loading}
            />
          </Card>
        </Col>
      </Row>

      {/* Performance Chart */}
      <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
        <Col xs={24}>
          <Card title="Optimization Performance Trends">
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={performanceData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="day" />
                <YAxis />
                <Tooltip />
                <Line 
                  type="monotone" 
                  dataKey="optimizations" 
                  stroke="#1890ff" 
                  name="Optimizations"
                />
                <Line 
                  type="monotone" 
                  dataKey="avgReduction" 
                  stroke="#52c41a" 
                  name="Avg Reduction %"
                />
              </LineChart>
            </ResponsiveContainer>
          </Card>
        </Col>
      </Row>

      {/* Active Optimizations Table */}
      <Row gutter={[16, 16]}>
        <Col xs={24}>
          <Card title="Active Optimizations">
            <Table
              columns={columns}
              dataSource={activeSessions}
              rowKey="id"
              loading={loading}
              pagination={{ pageSize: 10 }}
              scroll={{ x: 800 }}
            />
          </Card>
        </Col>
      </Row>
    </div>
  );
};

export default Dashboard;
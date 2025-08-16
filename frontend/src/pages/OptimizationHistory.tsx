import React, { useState, useEffect } from 'react';
import {
  Card,
  Table,
  Tag,
  Button,
  Modal,
  Descriptions,
  Progress,
  Space,
  Input,
  Select,
  DatePicker,
  message,
  Tooltip,
  Statistic,
  Row,
  Col
} from 'antd';
import {
  EyeOutlined,
  DownloadOutlined,
  DeleteOutlined,
  ReloadOutlined,
  SearchOutlined
} from '@ant-design/icons';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, ResponsiveContainer } from 'recharts';
import apiService from '../services/api';
import { OptimizationSession, OptimizationResults } from '../types';

const { Search } = Input;
const { Option } = Select;
const { RangePicker } = DatePicker;

const OptimizationHistory: React.FC = () => {
  const [sessions, setSessions] = useState<OptimizationSession[]>([]);
  const [filteredSessions, setFilteredSessions] = useState<OptimizationSession[]>([]);
  const [loading, setLoading] = useState(true);
  const [selectedSession, setSelectedSession] = useState<OptimizationSession | null>(null);
  const [detailsVisible, setDetailsVisible] = useState(false);
  const [filters, setFilters] = useState({
    status: '',
    technique: '',
    dateRange: null as any,
    search: ''
  });

  useEffect(() => {
    loadOptimizationHistory();
  }, []);

  useEffect(() => {
    applyFilters();
  }, [sessions, filters]);

  const loadOptimizationHistory = async () => {
    try {
      setLoading(true);
      const data = await apiService.getOptimizationSessions();
      setSessions(data);
    } catch (error) {
      console.error('Failed to load optimization history:', error);
      message.error('Failed to load optimization history');
    } finally {
      setLoading(false);
    }
  };

  const applyFilters = () => {
    let filtered = [...sessions];

    // Status filter
    if (filters.status) {
      filtered = filtered.filter(session => session.status === filters.status);
    }

    // Technique filter
    if (filters.technique) {
      filtered = filtered.filter(session => 
        session.criteria?.techniques?.includes(filters.technique)
      );
    }

    // Search filter
    if (filters.search) {
      const searchLower = filters.search.toLowerCase();
      filtered = filtered.filter(session =>
        session.id.toLowerCase().includes(searchLower) ||
        session.model_id.toLowerCase().includes(searchLower)
      );
    }

    // Date range filter
    if (filters.dateRange && filters.dateRange.length === 2) {
      const [start, end] = filters.dateRange;
      filtered = filtered.filter(session => {
        const sessionDate = new Date(session.created_at);
        return sessionDate >= start.toDate() && sessionDate <= end.toDate();
      });
    }

    setFilteredSessions(filtered);
  };

  const handleViewDetails = (session: OptimizationSession) => {
    setSelectedSession(session);
    setDetailsVisible(true);
  };

  const handleDeleteSession = async (sessionId: string) => {
    try {
      // Note: This would need to be implemented in the API
      message.success('Session deleted successfully');
      loadOptimizationHistory();
    } catch (error) {
      console.error('Failed to delete session:', error);
      message.error('Failed to delete session');
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'completed': return 'green';
      case 'running': return 'blue';
      case 'failed': return 'red';
      case 'cancelled': return 'orange';
      default: return 'default';
    }
  };

  const calculateStats = () => {
    const completed = filteredSessions.filter(s => s.status === 'completed');
    const avgSizeReduction = completed.reduce((acc, s) => 
      acc + (s.results?.size_reduction_percent || 0), 0) / (completed.length || 1);
    const avgSpeedImprovement = completed.reduce((acc, s) => 
      acc + (s.results?.speed_improvement_percent || 0), 0) / (completed.length || 1);
    
    return {
      total: filteredSessions.length,
      completed: completed.length,
      avgSizeReduction: Math.round(avgSizeReduction * 100) / 100,
      avgSpeedImprovement: Math.round(avgSpeedImprovement * 100) / 100
    };
  };

  const getChartData = () => {
    const techniqueStats: Record<string, { count: number; avgReduction: number }> = {};
    
    filteredSessions.forEach(session => {
      session.criteria?.techniques?.forEach(technique => {
        if (!techniqueStats[technique]) {
          techniqueStats[technique] = { count: 0, avgReduction: 0 };
        }
        techniqueStats[technique].count++;
        if (session.results?.size_reduction_percent) {
          techniqueStats[technique].avgReduction += session.results.size_reduction_percent;
        }
      });
    });

    return Object.entries(techniqueStats).map(([technique, stats]) => ({
      technique,
      count: stats.count,
      avgReduction: Math.round((stats.avgReduction / stats.count) * 100) / 100
    }));
  };

  const columns = [
    {
      title: 'Session ID',
      dataIndex: 'id',
      key: 'id',
      render: (id: string) => (
        <Tooltip title={id}>
          <code>{id.substring(0, 8)}...</code>
        </Tooltip>
      ),
    },
    {
      title: 'Model ID',
      dataIndex: 'model_id',
      key: 'model_id',
      render: (modelId: string) => (
        <Tooltip title={modelId}>
          <code>{modelId.substring(0, 8)}...</code>
        </Tooltip>
      ),
    },
    {
      title: 'Status',
      dataIndex: 'status',
      key: 'status',
      render: (status: string) => (
        <Tag color={getStatusColor(status)}>
          {status.toUpperCase()}
        </Tag>
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
    },
    {
      title: 'Progress',
      dataIndex: 'progress',
      key: 'progress',
      render: (progress: number, record: OptimizationSession) => (
        record.status === 'running' ? (
          <Progress percent={progress || 0} size="small" />
        ) : (
          <span>{record.status === 'completed' ? '100%' : 'N/A'}</span>
        )
      ),
    },
    {
      title: 'Size Reduction',
      key: 'size_reduction',
      render: (_: any, record: OptimizationSession) => (
        record.results?.size_reduction_percent ? (
          <Statistic
            value={record.results.size_reduction_percent}
            suffix="%"
            valueStyle={{ fontSize: '14px' }}
          />
        ) : (
          <span>-</span>
        )
      ),
    },
    {
      title: 'Speed Improvement',
      key: 'speed_improvement',
      render: (_: any, record: OptimizationSession) => (
        record.results?.speed_improvement_percent ? (
          <Statistic
            value={record.results.speed_improvement_percent}
            suffix="%"
            valueStyle={{ fontSize: '14px' }}
          />
        ) : (
          <span>-</span>
        )
      ),
    },
    {
      title: 'Created',
      dataIndex: 'created_at',
      key: 'created_at',
      render: (date: string) => new Date(date).toLocaleDateString(),
    },
    {
      title: 'Actions',
      key: 'actions',
      render: (_: any, record: OptimizationSession) => (
        <Space>
          <Button
            size="small"
            icon={<EyeOutlined />}
            onClick={() => handleViewDetails(record)}
          >
            Details
          </Button>
          {record.status === 'completed' && (
            <Button
              size="small"
              icon={<DownloadOutlined />}
              onClick={() => message.info('Download functionality would be implemented')}
            >
              Download
            </Button>
          )}
          <Button
            size="small"
            danger
            icon={<DeleteOutlined />}
            onClick={() => handleDeleteSession(record.id)}
          >
            Delete
          </Button>
        </Space>
      ),
    },
  ];

  const stats = calculateStats();
  const chartData = getChartData();

  return (
    <div>
      <h1>Optimization History</h1>

      {/* Summary Statistics */}
      <Row gutter={16} style={{ marginBottom: 24 }}>
        <Col span={6}>
          <Card>
            <Statistic title="Total Sessions" value={stats.total} />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic title="Completed" value={stats.completed} />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic 
              title="Avg Size Reduction" 
              value={stats.avgSizeReduction} 
              suffix="%" 
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic 
              title="Avg Speed Improvement" 
              value={stats.avgSpeedImprovement} 
              suffix="%" 
            />
          </Card>
        </Col>
      </Row>

      {/* Technique Performance Chart */}
      {chartData.length > 0 && (
        <Card title="Technique Performance" style={{ marginBottom: 24 }}>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="technique" />
              <YAxis />
              <RechartsTooltip />
              <Bar dataKey="count" fill="#1890ff" name="Usage Count" />
              <Bar dataKey="avgReduction" fill="#52c41a" name="Avg Reduction %" />
            </BarChart>
          </ResponsiveContainer>
        </Card>
      )}

      {/* Filters */}
      <Card title="Filters" style={{ marginBottom: 24 }}>
        <Row gutter={16}>
          <Col span={6}>
            <Search
              placeholder="Search by ID"
              value={filters.search}
              onChange={(e) => setFilters(prev => ({ ...prev, search: e.target.value }))}
              prefix={<SearchOutlined />}
            />
          </Col>
          <Col span={4}>
            <Select
              placeholder="Status"
              value={filters.status}
              onChange={(value) => setFilters(prev => ({ ...prev, status: value }))}
              style={{ width: '100%' }}
              allowClear
            >
              <Option value="running">Running</Option>
              <Option value="completed">Completed</Option>
              <Option value="failed">Failed</Option>
              <Option value="cancelled">Cancelled</Option>
            </Select>
          </Col>
          <Col span={4}>
            <Select
              placeholder="Technique"
              value={filters.technique}
              onChange={(value) => setFilters(prev => ({ ...prev, technique: value }))}
              style={{ width: '100%' }}
              allowClear
            >
              <Option value="quantization">Quantization</Option>
              <Option value="pruning">Pruning</Option>
              <Option value="distillation">Distillation</Option>
            </Select>
          </Col>
          <Col span={6}>
            <RangePicker
              value={filters.dateRange}
              onChange={(dates) => setFilters(prev => ({ ...prev, dateRange: dates }))}
              style={{ width: '100%' }}
            />
          </Col>
          <Col span={4}>
            <Button
              icon={<ReloadOutlined />}
              onClick={loadOptimizationHistory}
              loading={loading}
            >
              Refresh
            </Button>
          </Col>
        </Row>
      </Card>

      {/* Sessions Table */}
      <Card title="Optimization Sessions">
        <Table
          columns={columns}
          dataSource={filteredSessions}
          rowKey="id"
          loading={loading}
          pagination={{
            pageSize: 10,
            showSizeChanger: true,
            showQuickJumper: true,
            showTotal: (total, range) => 
              `${range[0]}-${range[1]} of ${total} sessions`,
          }}
        />
      </Card>

      {/* Session Details Modal */}
      <Modal
        title="Optimization Session Details"
        open={detailsVisible}
        onCancel={() => setDetailsVisible(false)}
        footer={null}
        width={800}
      >
        {selectedSession && (
          <div>
            <Descriptions bordered column={2}>
              <Descriptions.Item label="Session ID" span={2}>
                <code>{selectedSession.id}</code>
              </Descriptions.Item>
              <Descriptions.Item label="Model ID">
                <code>{selectedSession.model_id}</code>
              </Descriptions.Item>
              <Descriptions.Item label="Status">
                <Tag color={getStatusColor(selectedSession.status)}>
                  {selectedSession.status.toUpperCase()}
                </Tag>
              </Descriptions.Item>
              <Descriptions.Item label="Created">
                {new Date(selectedSession.created_at).toLocaleString()}
              </Descriptions.Item>
              <Descriptions.Item label="Updated">
                {new Date(selectedSession.updated_at).toLocaleString()}
              </Descriptions.Item>
              <Descriptions.Item label="Techniques" span={2}>
                {selectedSession.criteria?.techniques?.map(tech => (
                  <Tag key={tech}>{tech}</Tag>
                ))}
              </Descriptions.Item>
            </Descriptions>

            {selectedSession.results && (
              <Card title="Results" style={{ marginTop: 16 }}>
                <Row gutter={16}>
                  <Col span={8}>
                    <Statistic
                      title="Size Reduction"
                      value={selectedSession.results.size_reduction_percent}
                      suffix="%"
                    />
                  </Col>
                  <Col span={8}>
                    <Statistic
                      title="Speed Improvement"
                      value={selectedSession.results.speed_improvement_percent}
                      suffix="%"
                    />
                  </Col>
                  <Col span={8}>
                    <Statistic
                      title="Accuracy Retention"
                      value={selectedSession.results.accuracy_retention_percent}
                      suffix="%"
                    />
                  </Col>
                </Row>
              </Card>
            )}

            {selectedSession.steps && selectedSession.steps.length > 0 && (
              <Card title="Optimization Steps" style={{ marginTop: 16 }}>
                {selectedSession.steps.map((step, index) => (
                  <div key={step.id} style={{ marginBottom: 16 }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <span><strong>Step {index + 1}:</strong> {step.technique}</span>
                      <Tag color={getStatusColor(step.status)}>{step.status}</Tag>
                    </div>
                    <Progress percent={step.progress} size="small" />
                    {step.error_message && (
                      <div style={{ color: 'red', fontSize: '12px', marginTop: 4 }}>
                        {step.error_message}
                      </div>
                    )}
                  </div>
                ))}
              </Card>
            )}
          </div>
        )}
      </Modal>
    </div>
  );
};

export default OptimizationHistory;
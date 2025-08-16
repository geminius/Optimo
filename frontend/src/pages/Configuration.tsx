import React, { useState, useEffect } from 'react';
import {
  Card,
  Form,
  InputNumber,
  Select,
  Switch,
  Button,
  message,
  Divider,
  Space,
  Tooltip,
  Alert,
  Tabs,
  Slider,
  Row,
  Col
} from 'antd';
import { InfoCircleOutlined, SaveOutlined, ReloadOutlined } from '@ant-design/icons';
import apiService from '../services/api';
import { OptimizationCriteria } from '../types';

const { Option } = Select;
// const { TabPane } = Tabs;

const Configuration: React.FC = () => {
  const [form] = Form.useForm();
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [criteria, setCriteria] = useState<OptimizationCriteria | null>(null);
  const [hasChanges, setHasChanges] = useState(false);

  useEffect(() => {
    loadConfiguration();
  }, []);

  const loadConfiguration = async () => {
    try {
      setLoading(true);
      const config = await apiService.getOptimizationCriteria();
      setCriteria(config);
      form.setFieldsValue(config);
      setHasChanges(false);
    } catch (error) {
      console.error('Failed to load configuration:', error);
      message.error('Failed to load configuration');
    } finally {
      setLoading(false);
    }
  };

  const handleSave = async (values: OptimizationCriteria) => {
    try {
      setSaving(true);
      const updatedConfig = await apiService.updateOptimizationCriteria(values);
      setCriteria(updatedConfig);
      setHasChanges(false);
      message.success('Configuration saved successfully');
    } catch (error) {
      console.error('Failed to save configuration:', error);
      message.error('Failed to save configuration');
    } finally {
      setSaving(false);
    }
  };

  const handleFormChange = () => {
    setHasChanges(true);
  };

  const handleReset = () => {
    if (criteria) {
      form.setFieldsValue(criteria);
      setHasChanges(false);
    }
  };

  const optimizationTechniques = [
    { value: 'quantization', label: 'Quantization', description: 'Reduce model precision (4-bit, 8-bit)' },
    { value: 'pruning', label: 'Pruning', description: 'Remove unnecessary model parameters' },
    { value: 'distillation', label: 'Knowledge Distillation', description: 'Transfer knowledge to smaller model' },
    { value: 'compression', label: 'Model Compression', description: 'Tensor decomposition and compression' },
    { value: 'architecture_search', label: 'Architecture Search', description: 'Find optimal model architecture' },
  ];

  const hardwareTargets = [
    { value: 'cpu', label: 'CPU', description: 'Optimize for CPU inference' },
    { value: 'gpu', label: 'GPU', description: 'Optimize for GPU inference' },
    { value: 'edge', label: 'Edge Device', description: 'Optimize for edge/mobile devices' },
    { value: 'cloud', label: 'Cloud', description: 'Optimize for cloud deployment' },
  ];

  return (
    <div>
      <h1>Configuration</h1>

      {hasChanges && (
        <Alert
          message="You have unsaved changes"
          description="Don't forget to save your configuration changes."
          type="warning"
          showIcon
          style={{ marginBottom: 24 }}
        />
      )}

      <Form
        form={form}
        layout="vertical"
        onFinish={handleSave}
        onValuesChange={handleFormChange}
        disabled={loading}
      >
        <Tabs defaultActiveKey="optimization">
          <Tabs.TabPane tab="Optimization Criteria" key="optimization">
            <Card title="Performance Thresholds">
              <Row gutter={24}>
                <Col span={12}>
                  <Form.Item
                    name="max_size_reduction"
                    label={
                      <span>
                        Maximum Size Reduction (%)
                        <Tooltip title="Maximum percentage of model size reduction allowed">
                          <InfoCircleOutlined style={{ marginLeft: 4 }} />
                        </Tooltip>
                      </span>
                    }
                    rules={[{ required: true, message: 'Please set maximum size reduction' }]}
                  >
                    <Slider
                      min={0}
                      max={90}
                      marks={{
                        0: '0%',
                        30: '30%',
                        60: '60%',
                        90: '90%'
                      }}
                    />
                  </Form.Item>
                </Col>
                <Col span={12}>
                  <Form.Item
                    name="min_accuracy_retention"
                    label={
                      <span>
                        Minimum Accuracy Retention (%)
                        <Tooltip title="Minimum percentage of original accuracy that must be retained">
                          <InfoCircleOutlined style={{ marginLeft: 4 }} />
                        </Tooltip>
                      </span>
                    }
                    rules={[{ required: true, message: 'Please set minimum accuracy retention' }]}
                  >
                    <Slider
                      min={70}
                      max={100}
                      marks={{
                        70: '70%',
                        80: '80%',
                        90: '90%',
                        100: '100%'
                      }}
                    />
                  </Form.Item>
                </Col>
              </Row>

              <Form.Item
                name="max_inference_time"
                label={
                  <span>
                    Maximum Inference Time (ms)
                    <Tooltip title="Maximum allowed inference time per sample">
                      <InfoCircleOutlined style={{ marginLeft: 4 }} />
                    </Tooltip>
                  </span>
                }
                rules={[{ required: true, message: 'Please set maximum inference time' }]}
              >
                <InputNumber
                  min={1}
                  max={10000}
                  style={{ width: '100%' }}
                  placeholder="Enter maximum inference time in milliseconds"
                />
              </Form.Item>
            </Card>

            <Card title="Optimization Techniques" style={{ marginTop: 16 }}>
              <Form.Item
                name="techniques"
                label="Enabled Techniques"
                rules={[{ required: true, message: 'Please select at least one technique' }]}
              >
                <Select
                  mode="multiple"
                  placeholder="Select optimization techniques to enable"
                  style={{ width: '100%' }}
                >
                  {optimizationTechniques.map(technique => (
                    <Option key={technique.value} value={technique.value}>
                      <div>
                        <div>{technique.label}</div>
                        <div style={{ fontSize: '12px', color: '#666' }}>
                          {technique.description}
                        </div>
                      </div>
                    </Option>
                  ))}
                </Select>
              </Form.Item>

              <Form.Item
                name="hardware_target"
                label="Hardware Target"
                rules={[{ required: true, message: 'Please select hardware target' }]}
              >
                <Select placeholder="Select target hardware for optimization">
                  {hardwareTargets.map(target => (
                    <Option key={target.value} value={target.value}>
                      <div>
                        <div>{target.label}</div>
                        <div style={{ fontSize: '12px', color: '#666' }}>
                          {target.description}
                        </div>
                      </div>
                    </Option>
                  ))}
                </Select>
              </Form.Item>
            </Card>
          </Tabs.TabPane>

          <Tabs.TabPane tab="Advanced Settings" key="advanced">
            <Card title="Quantization Settings">
              <Form.Item
                name={['advanced', 'quantization', 'enable_4bit']}
                label="Enable 4-bit Quantization"
                valuePropName="checked"
              >
                <Switch />
              </Form.Item>

              <Form.Item
                name={['advanced', 'quantization', 'enable_8bit']}
                label="Enable 8-bit Quantization"
                valuePropName="checked"
              >
                <Switch />
              </Form.Item>

              <Form.Item
                name={['advanced', 'quantization', 'use_awq']}
                label="Use AWQ (Activation-aware Weight Quantization)"
                valuePropName="checked"
              >
                <Switch />
              </Form.Item>
            </Card>

            <Card title="Pruning Settings" style={{ marginTop: 16 }}>
              <Form.Item
                name={['advanced', 'pruning', 'structured_pruning']}
                label="Enable Structured Pruning"
                valuePropName="checked"
              >
                <Switch />
              </Form.Item>

              <Form.Item
                name={['advanced', 'pruning', 'unstructured_pruning']}
                label="Enable Unstructured Pruning"
                valuePropName="checked"
              >
                <Switch />
              </Form.Item>

              <Form.Item
                name={['advanced', 'pruning', 'sparsity_ratio']}
                label="Target Sparsity Ratio (%)"
              >
                <Slider
                  min={0}
                  max={90}
                  marks={{
                    0: '0%',
                    25: '25%',
                    50: '50%',
                    75: '75%',
                    90: '90%'
                  }}
                />
              </Form.Item>
            </Card>

            <Card title="Evaluation Settings" style={{ marginTop: 16 }}>
              <Form.Item
                name={['advanced', 'evaluation', 'run_benchmarks']}
                label="Run Benchmarks After Optimization"
                valuePropName="checked"
              >
                <Switch />
              </Form.Item>

              <Form.Item
                name={['advanced', 'evaluation', 'validate_accuracy']}
                label="Validate Accuracy After Each Step"
                valuePropName="checked"
              >
                <Switch />
              </Form.Item>

              <Form.Item
                name={['advanced', 'evaluation', 'rollback_on_failure']}
                label="Auto-rollback on Validation Failure"
                valuePropName="checked"
              >
                <Switch />
              </Form.Item>
            </Card>
          </Tabs.TabPane>

          <Tabs.TabPane tab="Monitoring & Alerts" key="monitoring">
            <Card title="Progress Monitoring">
              <Form.Item
                name={['monitoring', 'enable_realtime_updates']}
                label="Enable Real-time Progress Updates"
                valuePropName="checked"
              >
                <Switch />
              </Form.Item>

              <Form.Item
                name={['monitoring', 'update_interval']}
                label="Update Interval (seconds)"
              >
                <InputNumber
                  min={1}
                  max={300}
                  style={{ width: '100%' }}
                />
              </Form.Item>
            </Card>

            <Card title="Alert Settings" style={{ marginTop: 16 }}>
              <Form.Item
                name={['alerts', 'notify_on_completion']}
                label="Notify on Optimization Completion"
                valuePropName="checked"
              >
                <Switch />
              </Form.Item>

              <Form.Item
                name={['alerts', 'notify_on_failure']}
                label="Notify on Optimization Failure"
                valuePropName="checked"
              >
                <Switch />
              </Form.Item>

              <Form.Item
                name={['alerts', 'email_notifications']}
                label="Enable Email Notifications"
                valuePropName="checked"
              >
                <Switch />
              </Form.Item>

              <Form.Item
                name={['alerts', 'slack_notifications']}
                label="Enable Slack Notifications"
                valuePropName="checked"
              >
                <Switch />
              </Form.Item>
            </Card>
          </Tabs.TabPane>
        </Tabs>

        <Divider />

        <Form.Item>
          <Space>
            <Button
              type="primary"
              htmlType="submit"
              loading={saving}
              icon={<SaveOutlined />}
              disabled={!hasChanges}
            >
              Save Configuration
            </Button>
            <Button
              onClick={handleReset}
              disabled={!hasChanges}
            >
              Reset Changes
            </Button>
            <Button
              onClick={loadConfiguration}
              loading={loading}
              icon={<ReloadOutlined />}
            >
              Reload
            </Button>
          </Space>
        </Form.Item>
      </Form>
    </div>
  );
};

export default Configuration;
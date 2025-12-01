/**
 * LoginPage Component
 * 
 * Provides user authentication interface with form validation,
 * error handling, and loading states. Integrates with AuthContext
 * for authentication management.
 */

import React, { useState, useEffect } from 'react';
import { Form, Input, Button, Checkbox, Alert, Card, Typography } from 'antd';
import { UserOutlined, LockOutlined, RobotOutlined } from '@ant-design/icons';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../../hooks/useAuth';
import ErrorHandler from '../../utils/errorHandler';
import './LoginPage.css';

const { Title, Text } = Typography;

interface LoginFormValues {
  username: string;
  password: string;
  remember: boolean;
}

/**
 * LoginPage component for user authentication
 */
const LoginPage: React.FC = () => {
  const [form] = Form.useForm();
  const navigate = useNavigate();
  const { login, isAuthenticated, isLoading, error } = useAuth();
  const [submitting, setSubmitting] = useState(false);
  const [loginError, setLoginError] = useState<string | null>(null);

  /**
   * Redirect to dashboard if already authenticated
   * Requirement 1.5: IF the user is already authenticated THEN redirect to dashboard
   */
  useEffect(() => {
    if (isAuthenticated && !isLoading) {
      navigate('/dashboard');
    }
  }, [isAuthenticated, isLoading, navigate]);

  /**
   * Handle form submission
   * Sub-task 5.3: Add login submission logic
   * Task 15: Pass rememberMe value to login function
   */
  const handleSubmit = async (values: LoginFormValues) => {
    try {
      setSubmitting(true);
      setLoginError(null);

      // Call login function from AuthContext with rememberMe preference
      await login(values.username, values.password, values.remember);

      // Navigation is handled by AuthContext after successful login
    } catch (err) {
      // Parse error using ErrorHandler for consistent error messages
      const errorDetails = ErrorHandler.parseError(err);
      setLoginError(errorDetails.message);
      
      // Log error for debugging
      ErrorHandler.logError(errorDetails, { component: 'LoginPage', username: values.username });
    } finally {
      setSubmitting(false);
    }
  };

  /**
   * Handle form validation failure
   */
  const handleValidationFailed = () => {
    setLoginError(null);
  };

  return (
    <div className="login-container">
      <div className="login-content">
        {/* Platform Logo and Branding - Sub-task 5.4 */}
        <div className="login-header">
          <RobotOutlined className="login-logo-icon" />
          <Title level={2} className="login-title">
            Robotics Model Optimization Platform
          </Title>
          <Text type="secondary" className="login-subtitle">
            Sign in to optimize your robotics models
          </Text>
        </div>

        {/* Login Form Card */}
        <Card className="login-card">
          <Form
            form={form}
            name="login"
            onFinish={handleSubmit}
            onFinishFailed={handleValidationFailed}
            autoComplete="off"
            size="large"
          >
            {/* Display error messages - Sub-task 5.3 */}
            {(loginError || error) && (
              <Form.Item>
                <Alert
                  message={loginError || error}
                  type="error"
                  showIcon
                  closable
                  onClose={() => setLoginError(null)}
                />
              </Form.Item>
            )}

            {/* Username Field - Sub-task 5.1 & 5.2 */}
            <Form.Item
              name="username"
              rules={[
                {
                  required: true,
                  message: 'Please enter your username',
                },
              ]}
            >
              <Input
                prefix={<UserOutlined />}
                placeholder="Username"
                autoComplete="username"
              />
            </Form.Item>

            {/* Password Field - Sub-task 5.1 & 5.2 */}
            <Form.Item
              name="password"
              rules={[
                {
                  required: true,
                  message: 'Please enter your password',
                },
                {
                  min: 6,
                  message: 'Password must be at least 6 characters',
                },
              ]}
            >
              <Input.Password
                prefix={<LockOutlined />}
                placeholder="Password"
                autoComplete="current-password"
              />
            </Form.Item>

            {/* Remember Me Checkbox - Sub-task 5.1 */}
            <Form.Item>
              <Form.Item name="remember" valuePropName="checked" noStyle initialValue={true}>
                <Checkbox>Remember me</Checkbox>
              </Form.Item>
            </Form.Item>

            {/* Submit Button with Loading State - Sub-task 5.1 & 5.3 */}
            <Form.Item>
              <Button
                type="primary"
                htmlType="submit"
                loading={submitting || isLoading}
                block
                size="large"
              >
                {submitting || isLoading ? 'Signing in...' : 'Sign In'}
              </Button>
            </Form.Item>
          </Form>

          {/* Additional Information */}
          <div className="login-footer">
            <Text type="secondary" style={{ fontSize: '12px' }}>
              Default credentials: admin / admin
            </Text>
          </div>
        </Card>

        {/* Platform Information */}
        <div className="login-info">
          <Text type="secondary" style={{ fontSize: '13px' }}>
            Optimize robotics models with AI-powered techniques including quantization,
            pruning, distillation, and architecture search.
          </Text>
        </div>
      </div>
    </div>
  );
};

export default LoginPage;

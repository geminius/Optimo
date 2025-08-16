import React, { Component, ErrorInfo, ReactNode } from 'react';
import { Result, Button } from 'antd';
import { BugOutlined } from '@ant-design/icons';

interface Props {
  children: ReactNode;
}

interface State {
  hasError: boolean;
  error?: Error;
  errorInfo?: ErrorInfo;
}

class ErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error('ErrorBoundary caught an error:', error, errorInfo);
    this.setState({ error, errorInfo });
  }

  handleReload = () => {
    window.location.reload();
  };

  handleReset = () => {
    this.setState({ hasError: false, error: undefined, errorInfo: undefined });
  };

  render() {
    if (this.state.hasError) {
      return (
        <div className="error-boundary">
          <Result
            status="error"
            icon={<BugOutlined />}
            title="Something went wrong"
            subTitle="An unexpected error occurred. Please try reloading the page or contact support if the problem persists."
            extra={[
              <Button type="primary" onClick={this.handleReload} key="reload">
                Reload Page
              </Button>,
              <Button onClick={this.handleReset} key="reset">
                Try Again
              </Button>,
            ]}
          >
            {process.env.NODE_ENV === 'development' && this.state.error && (
              <div style={{ textAlign: 'left', marginTop: 16 }}>
                <details>
                  <summary>Error Details (Development Mode)</summary>
                  <pre style={{ 
                    background: '#f5f5f5', 
                    padding: 16, 
                    borderRadius: 4, 
                    overflow: 'auto',
                    fontSize: 12,
                    marginTop: 8
                  }}>
                    {this.state.error.toString()}
                    {this.state.errorInfo?.componentStack}
                  </pre>
                </details>
              </div>
            )}
          </Result>
        </div>
      );
    }

    return this.props.children;
  }
}

export default ErrorBoundary;
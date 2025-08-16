import React, { useState } from 'react';
import { 
  Card, 
  Upload, 
  Form, 
  Input, 
  Select, 
  Button, 
  message, 
  Progress, 
  Tag,
  Space,
  Divider
} from 'antd';
import { InboxOutlined, UploadOutlined } from '@ant-design/icons';
import { useNavigate } from 'react-router-dom';
import apiService from '../services/api';
import { ModelMetadata } from '../types';

const { Dragger } = Upload;
const { Option } = Select;
const { TextArea } = Input;

const ModelUpload: React.FC = () => {
  const [form] = Form.useForm();
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const navigate = useNavigate();

  const supportedFormats = [
    { ext: '.pth', framework: 'pytorch', description: 'PyTorch Model' },
    { ext: '.pt', framework: 'pytorch', description: 'PyTorch Model' },
    { ext: '.onnx', framework: 'onnx', description: 'ONNX Model' },
    { ext: '.pb', framework: 'tensorflow', description: 'TensorFlow SavedModel' },
    { ext: '.h5', framework: 'tensorflow', description: 'Keras Model' },
  ];

  const modelTypes = [
    'openvla',
    'rt1',
    'rt2',
    'palm-e',
    'robotic-transformer',
    'custom'
  ];

  const handleFileSelect = (file: File) => {
    setSelectedFile(file);
    
    // Auto-detect framework based on file extension
    const extension = '.' + file.name.split('.').pop()?.toLowerCase();
    const format = supportedFormats.find(f => f.ext === extension);
    
    if (format) {
      form.setFieldsValue({
        framework: format.framework,
        name: file.name.substring(0, file.name.lastIndexOf(extension)),
      });
    }
    
    return false; // Prevent automatic upload
  };

  const handleUpload = async (values: any) => {
    if (!selectedFile) {
      message.error('Please select a file to upload');
      return;
    }

    setUploading(true);
    setUploadProgress(0);

    try {
      const metadata: Partial<ModelMetadata> = {
        name: values.name,
        model_type: values.model_type,
        framework: values.framework,
        version: values.version || '1.0.0',
        tags: values.tags ? values.tags.split(',').map((tag: string) => tag.trim()) : [],
      };

      // Simulate upload progress
      const progressInterval = setInterval(() => {
        setUploadProgress(prev => {
          if (prev >= 90) {
            clearInterval(progressInterval);
            return prev;
          }
          return prev + Math.random() * 10;
        });
      }, 200);

      const uploadedModel = await apiService.uploadModel(selectedFile, metadata);
      
      clearInterval(progressInterval);
      setUploadProgress(100);
      
      message.success('Model uploaded successfully!');
      
      // Reset form and state
      form.resetFields();
      setSelectedFile(null);
      setUploadProgress(0);
      
      // Navigate to dashboard or model details
      setTimeout(() => {
        navigate('/');
      }, 1500);
      
    } catch (error) {
      console.error('Upload failed:', error);
      message.error('Failed to upload model. Please try again.');
    } finally {
      setUploading(false);
    }
  };

  const uploadProps = {
    name: 'file',
    multiple: false,
    beforeUpload: handleFileSelect,
    showUploadList: false,
    accept: supportedFormats.map(f => f.ext).join(','),
  };

  return (
    <div>
      <h1>Upload Model</h1>
      
      <Card title="Model Upload" style={{ marginBottom: 24 }}>
        <Form
          form={form}
          layout="vertical"
          onFinish={handleUpload}
          disabled={uploading}
        >
          {/* File Upload Area */}
          <Form.Item
            label="Model File"
            required
            help="Supported formats: PyTorch (.pth, .pt), ONNX (.onnx), TensorFlow (.pb, .h5)"
          >
            <Dragger {...uploadProps} style={{ marginBottom: 16 }}>
              <p className="ant-upload-drag-icon">
                <InboxOutlined />
              </p>
              <p className="ant-upload-text">
                Click or drag model file to this area to upload
              </p>
              <p className="ant-upload-hint">
                Support for single file upload. Maximum file size: 5GB
              </p>
            </Dragger>
            
            {selectedFile && (
              <div style={{ marginTop: 8 }}>
                <Tag color="blue">{selectedFile.name}</Tag>
                <span style={{ marginLeft: 8, color: '#666' }}>
                  {(selectedFile.size / (1024 * 1024)).toFixed(2)} MB
                </span>
              </div>
            )}
          </Form.Item>

          {/* Upload Progress */}
          {uploading && (
            <Form.Item>
              <Progress 
                percent={Math.round(uploadProgress)} 
                status={uploadProgress === 100 ? 'success' : 'active'}
              />
            </Form.Item>
          )}

          <Divider />

          {/* Model Metadata */}
          <Form.Item
            name="name"
            label="Model Name"
            rules={[{ required: true, message: 'Please enter model name' }]}
          >
            <Input placeholder="Enter a descriptive name for your model" />
          </Form.Item>

          <Form.Item
            name="model_type"
            label="Model Type"
            rules={[{ required: true, message: 'Please select model type' }]}
          >
            <Select placeholder="Select the type of robotics model">
              {modelTypes.map(type => (
                <Option key={type} value={type}>
                  {type.toUpperCase()}
                </Option>
              ))}
            </Select>
          </Form.Item>

          <Form.Item
            name="framework"
            label="Framework"
            rules={[{ required: true, message: 'Please select framework' }]}
          >
            <Select placeholder="Select the ML framework">
              <Option value="pytorch">PyTorch</Option>
              <Option value="tensorflow">TensorFlow</Option>
              <Option value="onnx">ONNX</Option>
            </Select>
          </Form.Item>

          <Form.Item
            name="version"
            label="Version"
            initialValue="1.0.0"
          >
            <Input placeholder="Model version (e.g., 1.0.0)" />
          </Form.Item>

          <Form.Item
            name="tags"
            label="Tags"
            help="Comma-separated tags for categorization"
          >
            <Input placeholder="e.g., manipulation, navigation, vision" />
          </Form.Item>

          <Form.Item
            name="description"
            label="Description"
          >
            <TextArea 
              rows={3} 
              placeholder="Optional description of the model and its capabilities"
            />
          </Form.Item>

          <Form.Item>
            <Space>
              <Button 
                type="primary" 
                htmlType="submit" 
                loading={uploading}
                disabled={!selectedFile}
                icon={<UploadOutlined />}
              >
                {uploading ? 'Uploading...' : 'Upload Model'}
              </Button>
              <Button onClick={() => form.resetFields()}>
                Reset
              </Button>
            </Space>
          </Form.Item>
        </Form>
      </Card>

      {/* Supported Formats Info */}
      <Card title="Supported Formats" size="small">
        <div>
          {supportedFormats.map(format => (
            <Tag key={format.ext} style={{ margin: '4px' }}>
              {format.ext} - {format.description}
            </Tag>
          ))}
        </div>
      </Card>
    </div>
  );
};

export default ModelUpload;
"""
OpenAPI documentation configuration.
"""

from typing import Dict, Any


def get_openapi_config() -> Dict[str, Any]:
    """Get OpenAPI configuration for the API."""
    return {
        "title": "Robotics Model Optimization Platform API",
        "description": """
        ## AI Agentic Platform for Robotics Model Optimization
        
        This API provides endpoints for automatically optimizing robotics models like OpenVLA using AI agents.
        
        ### Features
        - **Model Upload**: Upload robotics models in various formats (PyTorch, TensorFlow, ONNX)
        - **Automatic Analysis**: AI agents analyze model architecture and performance
        - **Intelligent Optimization**: Automated optimization using techniques like quantization and pruning
        - **Real-time Monitoring**: Track optimization progress and system status
        - **Comprehensive Evaluation**: Detailed performance comparison and validation
        
        ### Authentication
        All endpoints require authentication using Bearer tokens. Use the `/auth/login` endpoint to obtain a token.
        
        ### Workflow
        1. **Upload Model**: Use `/models/upload` to upload your robotics model
        2. **Start Optimization**: Use `/optimize` to begin the optimization process
        3. **Monitor Progress**: Use `/sessions/{session_id}/status` to track progress
        4. **Get Results**: Use `/sessions/{session_id}/results` to retrieve optimization results
        
        ### Supported Model Types
        - OpenVLA models
        - RT-1 and RT-2 models
        - Custom robotics models
        - PyTorch (.pt, .pth)
        - TensorFlow (.pb, .h5)
        - ONNX (.onnx)
        - SafeTensors (.safetensors)
        
        ### Optimization Techniques
        - **Quantization**: 4-bit, 8-bit, and dynamic quantization
        - **Pruning**: Structured and unstructured pruning
        - **Knowledge Distillation**: Model compression through distillation
        - **Architecture Search**: Neural architecture optimization
        """,
        "version": "1.0.0",
        "contact": {
            "name": "Robotics Model Optimization Platform",
            "email": "support@example.com"
        },
        "license": {
            "name": "MIT License",
            "url": "https://opensource.org/licenses/MIT"
        },
        "servers": [
            {
                "url": "http://localhost:8000",
                "description": "Development server"
            },
            {
                "url": "https://api.robotics-optimization.example.com",
                "description": "Production server"
            }
        ],
        "tags": [
            {
                "name": "Health",
                "description": "Health check and system status endpoints"
            },
            {
                "name": "Authentication",
                "description": "User authentication and authorization"
            },
            {
                "name": "Models",
                "description": "Model upload, listing, and management"
            },
            {
                "name": "Optimization",
                "description": "Model optimization operations"
            },
            {
                "name": "Sessions",
                "description": "Optimization session management and monitoring"
            },
            {
                "name": "Results",
                "description": "Optimization results and evaluation reports"
            },
            {
                "name": "Monitoring",
                "description": "System monitoring and metrics"
            }
        ]
    }


def get_openapi_examples() -> Dict[str, Any]:
    """Get OpenAPI examples for request/response models."""
    return {
        "OptimizationRequest": {
            "summary": "Basic optimization request",
            "description": "Start optimization with default settings",
            "value": {
                "model_id": "550e8400-e29b-41d4-a716-446655440000",
                "criteria_name": "balanced_optimization",
                "target_accuracy_threshold": 0.95,
                "max_size_reduction_percent": 50.0,
                "max_latency_increase_percent": 10.0,
                "optimization_techniques": ["quantization", "pruning"],
                "priority": 1,
                "notes": "Optimize for deployment on edge devices"
            }
        },
        "ModelUpload": {
            "summary": "Model upload with metadata",
            "description": "Upload a robotics model with descriptive metadata",
            "value": {
                "name": "OpenVLA-7B-Optimized",
                "description": "OpenVLA model fine-tuned for robotic manipulation tasks",
                "tags": "openvla,manipulation,fine-tuned"
            }
        },
        "LoginRequest": {
            "summary": "User login",
            "description": "Authenticate with username and password",
            "value": {
                "username": "admin",
                "password": "admin"
            }
        }
    }
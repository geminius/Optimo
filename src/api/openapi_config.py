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
        - **Real-time Monitoring**: Track optimization progress and system status via REST and WebSocket
        - **Comprehensive Evaluation**: Detailed performance comparison and validation
        - **Dashboard Statistics**: Aggregate metrics and insights across all optimizations
        - **Configuration Management**: Customize optimization criteria and constraints
        
        ### Authentication
        All endpoints require authentication using Bearer tokens. Use the `/auth/login` endpoint to obtain a token.
        
        **Obtaining a Token:**
        ```bash
        curl -X POST http://localhost:8000/auth/login \\
          -H "Content-Type: application/json" \\
          -d '{"username": "admin", "password": "admin"}'
        ```
        
        **Using the Token:**
        Include the token in the Authorization header:
        ```bash
        curl -H "Authorization: Bearer YOUR_TOKEN_HERE" \\
          http://localhost:8000/dashboard/stats
        ```
        
        **Token Expiration:**
        - Tokens expire after 1 hour (3600 seconds)
        - Expired tokens return 401 Unauthorized
        - Refresh tokens by logging in again
        
        ### Workflow
        1. **Upload Model**: Use `/models/upload` to upload your robotics model
        2. **View Dashboard**: Use `/dashboard/stats` to see system overview
        3. **Configure Optimization**: Use `/config/optimization-criteria` to set preferences
        4. **Start Optimization**: Use `/optimize` to begin the optimization process
        5. **Monitor Progress**: Use `/sessions/{session_id}/status` or WebSocket for real-time updates
        6. **List Sessions**: Use `/optimization/sessions` to view all sessions with filtering
        7. **Get Results**: Use `/sessions/{session_id}/results` to retrieve optimization results
        
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
        - **Compression**: General model size reduction
        
        ### WebSocket Real-Time Updates
        Connect to `/socket.io` for real-time optimization progress updates.
        
        **Connection Example (JavaScript):**
        ```javascript
        import io from 'socket.io-client';
        
        const socket = io('http://localhost:8000', {
          path: '/socket.io',
          transports: ['websocket']
        });
        
        socket.on('connect', () => {
          console.log('Connected:', socket.id);
          socket.emit('subscribe_session', { session_id: 'your-session-id' });
        });
        
        socket.on('progress_update', (data) => {
          console.log('Progress:', data.progress_percentage);
        });
        ```
        
        See the WebSocket section below for detailed event documentation.
        
        ### Error Handling
        All errors follow a standardized format:
        ```json
        {
          "error": "ErrorType",
          "message": "Human-readable error message",
          "details": {},
          "timestamp": "2024-01-01T12:00:00Z",
          "request_id": "uuid"
        }
        ```
        
        **Common Error Codes:**
        - `400 Bad Request`: Invalid input or validation error
        - `401 Unauthorized`: Missing or invalid authentication token
        - `403 Forbidden`: Insufficient permissions
        - `404 Not Found`: Resource not found
        - `409 Conflict`: Conflicting configuration or state
        - `413 Payload Too Large`: File size exceeds limit
        - `500 Internal Server Error`: Server-side error
        
        ### Rate Limiting
        API requests are rate-limited per user to prevent abuse. Limits:
        - 100 requests per minute for standard endpoints
        - 10 optimization starts per hour
        - WebSocket connections: 5 concurrent per user
        
        ### Pagination
        List endpoints support pagination with query parameters:
        - `skip`: Number of items to skip (default: 0)
        - `limit`: Maximum items to return (default: 50, max: 100)
        
        ### Filtering
        Session list endpoint supports filtering:
        - `status`: Filter by session status (running, completed, failed, cancelled)
        - `model_id`: Filter by specific model
        - `start_date`: Filter sessions created after date (ISO 8601)
        - `end_date`: Filter sessions created before date (ISO 8601)
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
                "name": "Dashboard",
                "description": "Dashboard statistics and aggregate metrics"
            },
            {
                "name": "Configuration",
                "description": "Optimization criteria and system configuration"
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
        # Authentication Examples
        "LoginRequest": {
            "summary": "User login",
            "description": "Authenticate with username and password",
            "value": {
                "username": "admin",
                "password": "admin"
            }
        },
        "LoginResponse": {
            "summary": "Successful login",
            "description": "Authentication token and user information",
            "value": {
                "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
                "token_type": "bearer",
                "expires_in": 3600,
                "user": {
                    "id": "admin",
                    "username": "admin",
                    "role": "administrator"
                }
            }
        },
        
        # Model Examples
        "ModelUpload": {
            "summary": "Model upload with metadata",
            "description": "Upload a robotics model with descriptive metadata",
            "value": {
                "name": "OpenVLA-7B-Optimized",
                "description": "OpenVLA model fine-tuned for robotic manipulation tasks",
                "tags": "openvla,manipulation,fine-tuned"
            }
        },
        "ModelUploadResponse": {
            "summary": "Successful model upload",
            "description": "Confirmation of model upload with metadata",
            "value": {
                "model_id": "550e8400-e29b-41d4-a716-446655440000",
                "filename": "robotics_vla_model.pt",
                "size_mb": 245.8,
                "upload_time": "2024-01-01T12:00:00Z",
                "message": "Model uploaded successfully"
            }
        },
        
        # Dashboard Examples
        "DashboardStats": {
            "summary": "Dashboard statistics",
            "description": "Aggregate metrics across all optimizations",
            "value": {
                "total_models": 15,
                "active_optimizations": 3,
                "completed_optimizations": 42,
                "failed_optimizations": 2,
                "average_size_reduction": 32.5,
                "average_speed_improvement": 18.7,
                "total_sessions": 47,
                "last_updated": "2024-01-01T12:00:00Z"
            }
        },
        
        # Configuration Examples
        "OptimizationCriteriaRequest": {
            "summary": "Update optimization criteria",
            "description": "Configure optimization preferences and constraints",
            "value": {
                "name": "edge_deployment",
                "target_accuracy_threshold": 0.95,
                "max_size_reduction_percent": 60.0,
                "max_latency_increase_percent": 5.0,
                "optimization_techniques": ["quantization", "pruning", "distillation"],
                "hardware_constraints": {
                    "max_memory_mb": 512,
                    "target_device": "jetson_nano"
                },
                "custom_parameters": {
                    "quantization_bits": 8,
                    "pruning_ratio": 0.3
                }
            }
        },
        "OptimizationCriteriaResponse": {
            "summary": "Current optimization criteria",
            "description": "Active optimization configuration",
            "value": {
                "name": "edge_deployment",
                "target_accuracy_threshold": 0.95,
                "max_size_reduction_percent": 60.0,
                "max_latency_increase_percent": 5.0,
                "optimization_techniques": ["quantization", "pruning", "distillation"],
                "hardware_constraints": {
                    "max_memory_mb": 512,
                    "target_device": "jetson_nano"
                },
                "custom_parameters": {
                    "quantization_bits": 8,
                    "pruning_ratio": 0.3
                },
                "created_at": "2024-01-01T10:00:00Z",
                "updated_at": "2024-01-01T12:00:00Z"
            }
        },
        
        # Optimization Examples
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
        "OptimizationResponse": {
            "summary": "Optimization started",
            "description": "Confirmation that optimization session has started",
            "value": {
                "session_id": "session_abc123",
                "model_id": "550e8400-e29b-41d4-a716-446655440000",
                "status": "started",
                "message": "Optimization session started successfully"
            }
        },
        
        # Session Examples
        "SessionListResponse": {
            "summary": "List of optimization sessions",
            "description": "Paginated list with filtering applied",
            "value": {
                "sessions": [
                    {
                        "session_id": "session_abc123",
                        "model_id": "550e8400-e29b-41d4-a716-446655440000",
                        "model_name": "robotics_vla_model.pt",
                        "status": "running",
                        "progress_percentage": 45.0,
                        "techniques": ["quantization", "pruning"],
                        "size_reduction_percent": None,
                        "speed_improvement_percent": None,
                        "created_at": "2024-01-01T12:00:00Z",
                        "updated_at": "2024-01-01T12:15:00Z",
                        "completed_at": None
                    },
                    {
                        "session_id": "session_def456",
                        "model_id": "660e8400-e29b-41d4-a716-446655440001",
                        "model_name": "rt2_model.pt",
                        "status": "completed",
                        "progress_percentage": 100.0,
                        "techniques": ["quantization"],
                        "size_reduction_percent": 28.5,
                        "speed_improvement_percent": 15.2,
                        "created_at": "2024-01-01T10:00:00Z",
                        "updated_at": "2024-01-01T11:30:00Z",
                        "completed_at": "2024-01-01T11:30:00Z"
                    }
                ],
                "total": 47,
                "skip": 0,
                "limit": 50
            }
        },
        "SessionStatusResponse": {
            "summary": "Session status details",
            "description": "Current status and progress of optimization session",
            "value": {
                "session_id": "session_abc123",
                "status": "running",
                "progress_percentage": 45.0,
                "current_step": "Applying quantization",
                "start_time": "2024-01-01T12:00:00Z",
                "last_update": "2024-01-01T12:15:00Z",
                "error_message": None,
                "model_id": "550e8400-e29b-41d4-a716-446655440000",
                "steps_completed": 3
            }
        },
        
        # Results Examples
        "EvaluationResponse": {
            "summary": "Optimization results",
            "description": "Complete evaluation and comparison metrics",
            "value": {
                "session_id": "session_abc123",
                "model_id": "550e8400-e29b-41d4-a716-446655440000",
                "status": "completed",
                "optimization_summary": "Successfully optimized model with 32% size reduction and 18% speed improvement",
                "performance_improvements": {
                    "size_reduction_percent": 32.0,
                    "speed_improvement_percent": 18.0,
                    "accuracy_change_percent": -0.5
                },
                "techniques_applied": ["quantization", "pruning"],
                "evaluation_metrics": {
                    "accuracy": 0.945,
                    "inference_time_ms": 12.5,
                    "model_size_mb": 136.0
                },
                "comparison_baseline": {
                    "original_accuracy": 0.950,
                    "original_inference_time_ms": 15.0,
                    "original_model_size_mb": 200.0
                },
                "recommendations": [
                    "Model optimization successful with minimal accuracy loss",
                    "Consider further quantization for additional size reduction",
                    "Model ready for edge deployment"
                ]
            }
        },
        
        # Error Examples
        "ErrorResponse_400": {
            "summary": "Validation error",
            "description": "Invalid request parameters",
            "value": {
                "error": "ValidationError",
                "message": "Invalid query parameter: status must be one of [running, completed, failed, cancelled]",
                "details": {
                    "field": "status",
                    "provided_value": "invalid_status",
                    "allowed_values": ["running", "completed", "failed", "cancelled"]
                },
                "timestamp": "2024-01-01T12:00:00Z",
                "request_id": "req_abc123"
            }
        },
        "ErrorResponse_401": {
            "summary": "Authentication error",
            "description": "Missing or invalid authentication token",
            "value": {
                "error": "Unauthorized",
                "message": "Authentication token is missing or invalid",
                "details": {
                    "reason": "token_expired",
                    "expired_at": "2024-01-01T11:00:00Z"
                },
                "timestamp": "2024-01-01T12:00:00Z",
                "request_id": "req_def456"
            }
        },
        "ErrorResponse_404": {
            "summary": "Resource not found",
            "description": "Requested resource does not exist",
            "value": {
                "error": "NotFound",
                "message": "Session not found",
                "details": {
                    "session_id": "session_nonexistent",
                    "resource_type": "optimization_session"
                },
                "timestamp": "2024-01-01T12:00:00Z",
                "request_id": "req_ghi789"
            }
        },
        "ErrorResponse_500": {
            "summary": "Server error",
            "description": "Internal server error",
            "value": {
                "error": "InternalServerError",
                "message": "An unexpected error occurred while processing your request",
                "details": {
                    "error_id": "err_xyz789",
                    "support_message": "Please contact support with this error ID"
                },
                "timestamp": "2024-01-01T12:00:00Z",
                "request_id": "req_jkl012"
            }
        }
    }



def get_websocket_documentation() -> Dict[str, Any]:
    """
    Get comprehensive WebSocket documentation.
    
    Returns:
        Dictionary containing WebSocket connection and event documentation
    """
    return {
        "overview": """
        # WebSocket Real-Time Updates
        
        The platform provides real-time updates via WebSocket connections using Socket.IO.
        This allows clients to receive immediate notifications about optimization progress,
        session status changes, and system events without polling.
        
        ## Connection Endpoint
        
        **URL**: `ws://localhost:8000/socket.io` (development)
        **Protocol**: Socket.IO v4.x
        **Transport**: WebSocket (with fallback to polling)
        
        ## Authentication
        
        WebSocket connections require authentication. Include your Bearer token when connecting:
        
        ```javascript
        const socket = io('http://localhost:8000', {
          path: '/socket.io',
          transports: ['websocket', 'polling'],
          auth: {
            token: 'YOUR_BEARER_TOKEN'
          }
        });
        ```
        
        ## Connection Lifecycle
        
        1. **Connect**: Client establishes connection to server
        2. **Authenticate**: Server validates authentication token
        3. **Subscribe**: Client subscribes to specific sessions or system events
        4. **Receive Events**: Server broadcasts events to subscribed clients
        5. **Disconnect**: Client or server closes connection
        
        ## Reconnection Handling
        
        Socket.IO automatically handles reconnection with exponential backoff:
        - Initial reconnection delay: 1 second
        - Maximum reconnection delay: 5 seconds
        - Reconnection attempts: Unlimited
        
        ```javascript
        socket.on('connect', () => {
          console.log('Connected to server');
          // Re-subscribe to sessions after reconnection
          socket.emit('subscribe_session', { session_id: 'your-session-id' });
        });
        
        socket.on('disconnect', (reason) => {
          console.log('Disconnected:', reason);
          if (reason === 'io server disconnect') {
            // Server disconnected, manually reconnect
            socket.connect();
          }
          // Otherwise, Socket.IO will automatically reconnect
        });
        ```
        
        ## Room-Based Subscriptions
        
        Clients can subscribe to specific optimization sessions to receive targeted updates:
        
        ```javascript
        // Subscribe to a specific session
        socket.emit('subscribe_session', { session_id: 'session_abc123' });
        
        // Unsubscribe from a session
        socket.emit('unsubscribe_session', { session_id: 'session_abc123' });
        ```
        
        ## Event Types
        
        Events are categorized into several types:
        - **Connection Events**: Connection lifecycle management
        - **Session Events**: Optimization session updates
        - **Notification Events**: General notifications and alerts
        - **System Events**: Platform status and health
        
        ## Error Handling
        
        Handle connection errors gracefully:
        
        ```javascript
        socket.on('connect_error', (error) => {
          console.error('Connection error:', error.message);
          // Check authentication token validity
          // Display error message to user
        });
        
        socket.on('error', (error) => {
          console.error('Socket error:', error);
          // Handle specific error types
        });
        ```
        
        ## Best Practices
        
        1. **Always handle reconnection**: Implement reconnection logic and re-subscribe to sessions
        2. **Validate event data**: Check event payloads before using them
        3. **Limit subscriptions**: Only subscribe to sessions you're actively monitoring
        4. **Clean up on unmount**: Unsubscribe and disconnect when component unmounts
        5. **Handle errors**: Implement error handlers for all event types
        6. **Use heartbeat**: Implement ping/pong for connection health monitoring
        
        ## Performance Considerations
        
        - Maximum concurrent connections per user: 5
        - Event broadcast rate: Up to 10 events/second per session
        - Message size limit: 1MB per event
        - Connection timeout: 60 seconds of inactivity
        
        ## Security
        
        - All connections require valid authentication tokens
        - Tokens are validated on connection and periodically during session
        - Users can only subscribe to their own sessions (unless admin)
        - All events are logged for audit purposes
        """,
        
        "connection_examples": {
            "javascript": """
// JavaScript/TypeScript with Socket.IO client
import io from 'socket.io-client';

const socket = io('http://localhost:8000', {
  path: '/socket.io',
  transports: ['websocket', 'polling'],
  auth: {
    token: localStorage.getItem('auth_token')
  },
  reconnection: true,
  reconnectionDelay: 1000,
  reconnectionDelayMax: 5000,
  reconnectionAttempts: Infinity
});

// Connection events
socket.on('connect', () => {
  console.log('Connected:', socket.id);
});

socket.on('disconnect', (reason) => {
  console.log('Disconnected:', reason);
});

socket.on('connect_error', (error) => {
  console.error('Connection error:', error.message);
});

// Subscribe to session updates
socket.emit('subscribe_session', { 
  session_id: 'session_abc123' 
});

// Listen for progress updates
socket.on('progress_update', (data) => {
  console.log(`Progress: ${data.progress_percentage}%`);
  console.log(`Step: ${data.step_name}`);
  updateProgressBar(data.progress_percentage);
});

// Listen for completion
socket.on('session_completed', (data) => {
  console.log('Optimization completed!');
  console.log('Results:', data.results);
  displayResults(data.results);
});

// Listen for failures
socket.on('session_failed', (data) => {
  console.error('Optimization failed:', data.error_message);
  displayError(data.error_message);
});

// Clean up on unmount
function cleanup() {
  socket.emit('unsubscribe_session', { 
    session_id: 'session_abc123' 
  });
  socket.disconnect();
}
""",
            "python": """
# Python with python-socketio client
import socketio

sio = socketio.Client()

@sio.event
def connect():
    print('Connected to server')
    # Subscribe to session
    sio.emit('subscribe_session', {
        'session_id': 'session_abc123'
    })

@sio.event
def disconnect():
    print('Disconnected from server')

@sio.on('progress_update')
def on_progress(data):
    print(f"Progress: {data['progress_percentage']}%")
    print(f"Step: {data['step_name']}")

@sio.on('session_completed')
def on_completed(data):
    print('Optimization completed!')
    print('Results:', data['results'])

@sio.on('session_failed')
def on_failed(data):
    print(f"Optimization failed: {data['error_message']}")

# Connect with authentication
sio.connect('http://localhost:8000', 
            socketio_path='/socket.io',
            auth={'token': 'YOUR_BEARER_TOKEN'})

# Wait for events
sio.wait()
""",
            "react": """
// React Hook for WebSocket connection
import { useEffect, useState } from 'react';
import io from 'socket.io-client';

function useOptimizationSocket(sessionId: string) {
  const [socket, setSocket] = useState(null);
  const [progress, setProgress] = useState(0);
  const [status, setStatus] = useState('connecting');
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    const token = localStorage.getItem('auth_token');
    
    const newSocket = io('http://localhost:8000', {
      path: '/socket.io',
      auth: { token },
      transports: ['websocket', 'polling']
    });

    newSocket.on('connect', () => {
      setStatus('connected');
      newSocket.emit('subscribe_session', { session_id: sessionId });
    });

    newSocket.on('disconnect', () => {
      setStatus('disconnected');
    });

    newSocket.on('progress_update', (data) => {
      setProgress(data.progress_percentage);
      setStatus('running');
    });

    newSocket.on('session_completed', (data) => {
      setResults(data.results);
      setStatus('completed');
    });

    newSocket.on('session_failed', (data) => {
      setError(data.error_message);
      setStatus('failed');
    });

    setSocket(newSocket);

    return () => {
      newSocket.emit('unsubscribe_session', { session_id: sessionId });
      newSocket.disconnect();
    };
  }, [sessionId]);

  return { socket, progress, status, results, error };
}

// Usage in component
function OptimizationMonitor({ sessionId }) {
  const { progress, status, results, error } = useOptimizationSocket(sessionId);

  return (
    <div>
      <p>Status: {status}</p>
      <p>Progress: {progress}%</p>
      {results && <pre>{JSON.stringify(results, null, 2)}</pre>}
      {error && <p className="error">{error}</p>}
    </div>
  );
}
"""
        },
        
        "event_reference": {
            "session_started": {
                "direction": "server_to_client",
                "description": "Emitted when a new optimization session starts",
                "payload": {
                    "session_id": "string - Unique session identifier",
                    "model_id": "string - Model being optimized",
                    "model_name": "string - Human-readable model name",
                    "techniques": "array - Optimization techniques to be applied",
                    "timestamp": "string - ISO 8601 timestamp"
                },
                "example": {
                    "session_id": "session_abc123",
                    "model_id": "model_456",
                    "model_name": "robotics_vla_model.pt",
                    "techniques": ["quantization", "pruning"],
                    "timestamp": "2024-01-01T12:00:00Z"
                }
            },
            "progress_update": {
                "direction": "server_to_client",
                "description": "Emitted periodically during optimization with progress updates",
                "frequency": "Every 5-10 seconds or on significant milestones",
                "payload": {
                    "session_id": "string - Session identifier",
                    "current_step": "integer - Current step number",
                    "total_steps": "integer - Total number of steps",
                    "step_name": "string - Name of current step",
                    "progress_percentage": "float - Progress percentage (0-100)",
                    "estimated_completion": "string - Estimated completion time (ISO 8601)",
                    "elapsed_time": "string - Elapsed time (formatted)",
                    "remaining_time": "string - Remaining time (formatted)",
                    "metadata": "object - Additional step-specific data"
                },
                "example": {
                    "session_id": "session_abc123",
                    "current_step": 5,
                    "total_steps": 10,
                    "step_name": "Applying quantization",
                    "progress_percentage": 50.0,
                    "estimated_completion": "2024-01-01T12:30:00Z",
                    "elapsed_time": "0:15:00",
                    "remaining_time": "0:15:00",
                    "metadata": {
                        "quantization_bits": 8,
                        "layers_processed": 25
                    }
                }
            },
            "session_completed": {
                "direction": "server_to_client",
                "description": "Emitted when optimization session completes successfully",
                "payload": {
                    "session_id": "string - Session identifier",
                    "results": "object - Optimization results",
                    "timestamp": "string - Completion timestamp (ISO 8601)"
                },
                "example": {
                    "session_id": "session_abc123",
                    "results": {
                        "size_reduction_percent": 32.5,
                        "speed_improvement_percent": 18.7,
                        "accuracy_change_percent": -0.5,
                        "original_size_mb": 200.0,
                        "optimized_size_mb": 135.0,
                        "techniques_applied": ["quantization", "pruning"]
                    },
                    "timestamp": "2024-01-01T12:30:00Z"
                }
            },
            "session_failed": {
                "direction": "server_to_client",
                "description": "Emitted when optimization session fails",
                "payload": {
                    "session_id": "string - Session identifier",
                    "error_message": "string - Error description",
                    "error_type": "string - Error category",
                    "timestamp": "string - Failure timestamp (ISO 8601)"
                },
                "example": {
                    "session_id": "session_abc123",
                    "error_message": "Model validation failed: accuracy below threshold",
                    "error_type": "ValidationError",
                    "timestamp": "2024-01-01T12:15:00Z"
                }
            },
            "session_cancelled": {
                "direction": "server_to_client",
                "description": "Emitted when optimization session is cancelled by user",
                "payload": {
                    "session_id": "string - Session identifier",
                    "timestamp": "string - Cancellation timestamp (ISO 8601)"
                },
                "example": {
                    "session_id": "session_abc123",
                    "timestamp": "2024-01-01T12:10:00Z"
                }
            },
            "notification": {
                "direction": "server_to_client",
                "description": "General notification message",
                "payload": {
                    "id": "string - Notification ID",
                    "type": "string - Notification type (info, warning, error, success)",
                    "title": "string - Notification title",
                    "message": "string - Notification message",
                    "timestamp": "string - Notification timestamp (ISO 8601)",
                    "session_id": "string - Related session ID (optional)",
                    "metadata": "object - Additional data"
                },
                "example": {
                    "id": "notif_1",
                    "type": "info",
                    "title": "Optimization Started",
                    "message": "Your optimization session has begun",
                    "timestamp": "2024-01-01T12:00:00Z",
                    "session_id": "session_abc123",
                    "metadata": {}
                }
            },
            "alert": {
                "direction": "server_to_client",
                "description": "System alert for important events or issues",
                "payload": {
                    "id": "string - Alert ID",
                    "severity": "string - Alert severity (low, medium, high, critical)",
                    "title": "string - Alert title",
                    "description": "string - Alert description",
                    "timestamp": "string - Alert timestamp (ISO 8601)",
                    "session_id": "string - Related session ID (optional)",
                    "resolved": "boolean - Whether alert is resolved",
                    "metadata": "object - Additional data"
                },
                "example": {
                    "id": "alert_1",
                    "severity": "high",
                    "title": "High Memory Usage",
                    "description": "System memory usage exceeds 90%",
                    "timestamp": "2024-01-01T12:00:00Z",
                    "session_id": None,
                    "resolved": False,
                    "metadata": {
                        "memory_usage_percent": 92.5
                    }
                }
            },
            "subscribe_session": {
                "direction": "client_to_server",
                "description": "Subscribe to updates for a specific session",
                "payload": {
                    "session_id": "string - Session ID to subscribe to"
                },
                "example": {
                    "session_id": "session_abc123"
                },
                "response": "subscribed event with confirmation"
            },
            "unsubscribe_session": {
                "direction": "client_to_server",
                "description": "Unsubscribe from updates for a specific session",
                "payload": {
                    "session_id": "string - Session ID to unsubscribe from"
                },
                "example": {
                    "session_id": "session_abc123"
                },
                "response": "unsubscribed event with confirmation"
            },
            "ping": {
                "direction": "client_to_server",
                "description": "Health check ping from client",
                "payload": {
                    "timestamp": "string - Ping timestamp (ISO 8601)"
                },
                "example": {
                    "timestamp": "2024-01-01T12:00:00Z"
                },
                "response": "pong event with server timestamp"
            }
        },
        
        "troubleshooting": {
            "connection_refused": {
                "symptom": "Cannot establish WebSocket connection",
                "possible_causes": [
                    "Server is not running",
                    "Incorrect URL or port",
                    "Firewall blocking WebSocket connections",
                    "CORS configuration issue"
                ],
                "solutions": [
                    "Verify server is running: curl http://localhost:8000/health",
                    "Check WebSocket URL and port are correct",
                    "Ensure firewall allows WebSocket connections",
                    "Verify CORS settings allow your origin"
                ]
            },
            "authentication_failed": {
                "symptom": "Connection established but immediately disconnected",
                "possible_causes": [
                    "Invalid or expired authentication token",
                    "Token not included in connection",
                    "Token format incorrect"
                ],
                "solutions": [
                    "Verify token is valid: check /auth/login response",
                    "Include token in auth parameter when connecting",
                    "Refresh token if expired",
                    "Check token format (should be Bearer token)"
                ]
            },
            "no_events_received": {
                "symptom": "Connected but not receiving events",
                "possible_causes": [
                    "Not subscribed to session",
                    "Session ID incorrect",
                    "User doesn't have permission to view session",
                    "Event listeners not registered"
                ],
                "solutions": [
                    "Emit subscribe_session event after connecting",
                    "Verify session ID is correct",
                    "Check user has permission to view session",
                    "Register event listeners before subscribing"
                ]
            },
            "frequent_disconnections": {
                "symptom": "Connection drops frequently",
                "possible_causes": [
                    "Network instability",
                    "Server overload",
                    "Client-side timeout",
                    "Proxy or load balancer issues"
                ],
                "solutions": [
                    "Check network stability",
                    "Implement reconnection logic with backoff",
                    "Increase client timeout settings",
                    "Configure proxy for WebSocket support"
                ]
            }
        }
    }



def get_authentication_documentation() -> Dict[str, Any]:
    """
    Get comprehensive authentication and authorization documentation.
    
    Returns:
        Dictionary containing authentication documentation
    """
    return {
        "overview": """
        # Authentication & Authorization
        
        The Robotics Model Optimization Platform API uses Bearer token authentication
        to secure all endpoints. This ensures that only authorized users can access
        the platform's features and data.
        
        ## Authentication Flow
        
        1. **Login**: Obtain an access token by providing credentials
        2. **Include Token**: Add token to Authorization header in all requests
        3. **Token Validation**: Server validates token on each request
        4. **Token Expiration**: Tokens expire after 1 hour
        5. **Refresh**: Login again to obtain a new token
        
        ## Security Features
        
        - **Bearer Token Authentication**: Industry-standard token-based auth
        - **Role-Based Access Control (RBAC)**: Different permissions for users and admins
        - **Token Expiration**: Automatic token expiration for security
        - **Audit Logging**: All authentication events are logged
        - **Secure Storage**: Tokens should be stored securely (e.g., httpOnly cookies)
        
        ## User Roles
        
        ### Administrator
        - Full access to all endpoints
        - Can view all users' sessions
        - Can update system configuration
        - Can manage users (future feature)
        
        ### Regular User
        - Can upload models
        - Can start optimizations
        - Can view own sessions only
        - Cannot modify system configuration
        """,
        
        "obtaining_token": {
            "description": "How to obtain an authentication token",
            "endpoint": "POST /auth/login",
            "request_body": {
                "username": "string - User's username",
                "password": "string - User's password"
            },
            "response": {
                "access_token": "string - Bearer token for authentication",
                "token_type": "string - Always 'bearer'",
                "expires_in": "integer - Token lifetime in seconds (3600)",
                "user": {
                    "id": "string - User ID",
                    "username": "string - Username",
                    "role": "string - User role (administrator or user)"
                }
            },
            "examples": {
                "curl": """
# Login request
curl -X POST http://localhost:8000/auth/login \\
  -H "Content-Type: application/json" \\
  -d '{
    "username": "admin",
    "password": "admin"
  }'

# Response
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 3600,
  "user": {
    "id": "admin",
    "username": "admin",
    "role": "administrator"
  }
}
""",
                "javascript": """
// JavaScript/TypeScript
async function login(username, password) {
  const response = await fetch('http://localhost:8000/auth/login', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({ username, password })
  });
  
  if (!response.ok) {
    throw new Error('Login failed');
  }
  
  const data = await response.json();
  
  // Store token securely
  localStorage.setItem('auth_token', data.access_token);
  localStorage.setItem('token_expires_at', 
    Date.now() + (data.expires_in * 1000));
  
  return data;
}
""",
                "python": """
# Python with requests
import requests
import time

def login(username, password):
    response = requests.post(
        'http://localhost:8000/auth/login',
        json={'username': username, 'password': password}
    )
    
    if response.status_code != 200:
        raise Exception('Login failed')
    
    data = response.json()
    
    # Store token and expiration
    token = data['access_token']
    expires_at = time.time() + data['expires_in']
    
    return token, expires_at
"""
            }
        },
        
        "using_token": {
            "description": "How to use the authentication token in requests",
            "header_format": "Authorization: Bearer YOUR_TOKEN_HERE",
            "examples": {
                "curl": """
# Using token in request
curl -X GET http://localhost:8000/dashboard/stats \\
  -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
""",
                "javascript": """
// JavaScript/TypeScript
async function getDashboardStats() {
  const token = localStorage.getItem('auth_token');
  
  const response = await fetch('http://localhost:8000/dashboard/stats', {
    headers: {
      'Authorization': `Bearer ${token}`
    }
  });
  
  if (response.status === 401) {
    // Token expired or invalid, redirect to login
    window.location.href = '/login';
    return;
  }
  
  return await response.json();
}

// Reusable fetch with auth
async function authenticatedFetch(url, options = {}) {
  const token = localStorage.getItem('auth_token');
  
  const response = await fetch(url, {
    ...options,
    headers: {
      ...options.headers,
      'Authorization': `Bearer ${token}`
    }
  });
  
  if (response.status === 401) {
    // Handle token expiration
    localStorage.removeItem('auth_token');
    window.location.href = '/login';
    throw new Error('Authentication required');
  }
  
  return response;
}
""",
                "python": """
# Python with requests
import requests

class APIClient:
    def __init__(self, base_url, token):
        self.base_url = base_url
        self.token = token
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'Bearer {token}'
        })
    
    def get_dashboard_stats(self):
        response = self.session.get(f'{self.base_url}/dashboard/stats')
        
        if response.status_code == 401:
            raise Exception('Token expired or invalid')
        
        response.raise_for_status()
        return response.json()

# Usage
client = APIClient('http://localhost:8000', token)
stats = client.get_dashboard_stats()
"""
            }
        },
        
        "token_expiration": {
            "description": "Handling token expiration and refresh",
            "expiration_time": "3600 seconds (1 hour)",
            "detection": "Server returns 401 Unauthorized when token expires",
            "handling": {
                "check_before_request": """
// Check token expiration before making request
function isTokenExpired() {
  const expiresAt = localStorage.getItem('token_expires_at');
  if (!expiresAt) return true;
  
  return Date.now() >= parseInt(expiresAt);
}

async function makeAuthenticatedRequest(url, options) {
  if (isTokenExpired()) {
    // Redirect to login or refresh token
    await refreshToken();
  }
  
  return authenticatedFetch(url, options);
}
""",
                "handle_401_response": """
// Handle 401 responses
async function handleAuthError(response) {
  if (response.status === 401) {
    // Clear stored token
    localStorage.removeItem('auth_token');
    localStorage.removeItem('token_expires_at');
    
    // Redirect to login
    window.location.href = '/login?redirect=' + 
      encodeURIComponent(window.location.pathname);
    
    throw new Error('Authentication required');
  }
  
  return response;
}
""",
                "automatic_refresh": """
// Automatic token refresh (if refresh tokens are implemented)
async function refreshToken() {
  const refreshToken = localStorage.getItem('refresh_token');
  
  const response = await fetch('http://localhost:8000/auth/refresh', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({ refresh_token: refreshToken })
  });
  
  if (!response.ok) {
    // Refresh failed, redirect to login
    window.location.href = '/login';
    return;
  }
  
  const data = await response.json();
  localStorage.setItem('auth_token', data.access_token);
  localStorage.setItem('token_expires_at', 
    Date.now() + (data.expires_in * 1000));
}
"""
            }
        },
        
        "authorization": {
            "description": "Role-based access control and permissions",
            "roles": {
                "administrator": {
                    "permissions": [
                        "View all sessions (any user)",
                        "Update optimization criteria configuration",
                        "View system metrics and logs",
                        "Manage users (future)",
                        "All regular user permissions"
                    ],
                    "restricted_endpoints": []
                },
                "user": {
                    "permissions": [
                        "Upload models",
                        "Start optimizations",
                        "View own sessions only",
                        "View dashboard statistics",
                        "View optimization criteria (read-only)"
                    ],
                    "restricted_endpoints": [
                        "PUT /config/optimization-criteria (403 Forbidden)"
                    ]
                }
            },
            "checking_permissions": """
// Client-side permission check
function hasPermission(user, action) {
  const permissions = {
    'administrator': ['*'],  // All permissions
    'user': [
      'upload_model',
      'start_optimization',
      'view_own_sessions',
      'view_dashboard'
    ]
  };
  
  const userPermissions = permissions[user.role] || [];
  
  return userPermissions.includes('*') || 
         userPermissions.includes(action);
}

// Usage
if (hasPermission(currentUser, 'update_config')) {
  // Show configuration update UI
} else {
  // Hide or disable configuration update UI
}
""",
            "handling_403": """
// Handle 403 Forbidden responses
async function makeRequest(url, options) {
  const response = await authenticatedFetch(url, options);
  
  if (response.status === 403) {
    // User doesn't have permission
    const error = await response.json();
    
    showNotification({
      type: 'error',
      title: 'Access Denied',
      message: error.message || 'You do not have permission to perform this action'
    });
    
    throw new Error('Insufficient permissions');
  }
  
  return response;
}
"""
        },
        
        "error_responses": {
            "401_unauthorized": {
                "description": "Missing or invalid authentication token",
                "causes": [
                    "No Authorization header provided",
                    "Invalid token format",
                    "Token expired",
                    "Token signature invalid"
                ],
                "example": {
                    "error": "Unauthorized",
                    "message": "Authentication token is missing or invalid",
                    "details": {
                        "reason": "token_expired",
                        "expired_at": "2024-01-01T11:00:00Z"
                    },
                    "timestamp": "2024-01-01T12:00:00Z",
                    "request_id": "req_abc123"
                },
                "resolution": "Login again to obtain a new token"
            },
            "403_forbidden": {
                "description": "Insufficient permissions for requested action",
                "causes": [
                    "User role doesn't have required permission",
                    "Attempting to access another user's resources",
                    "Admin-only endpoint accessed by regular user"
                ],
                "example": {
                    "error": "Forbidden",
                    "message": "Administrator role required to update configuration",
                    "details": {
                        "required_role": "administrator",
                        "user_role": "user"
                    },
                    "timestamp": "2024-01-01T12:00:00Z",
                    "request_id": "req_def456"
                },
                "resolution": "Contact administrator for elevated permissions"
            }
        },
        
        "best_practices": {
            "token_storage": [
                "Store tokens securely (httpOnly cookies preferred)",
                "Never store tokens in localStorage for production (XSS risk)",
                "Use secure, httpOnly cookies when possible",
                "Clear tokens on logout",
                "Don't log or expose tokens in error messages"
            ],
            "token_transmission": [
                "Always use HTTPS in production",
                "Include token in Authorization header, not URL",
                "Don't send tokens in query parameters",
                "Validate token on every request"
            ],
            "error_handling": [
                "Handle 401 errors gracefully (redirect to login)",
                "Handle 403 errors with user-friendly messages",
                "Implement automatic token refresh if supported",
                "Log authentication failures for security monitoring"
            ],
            "security": [
                "Implement rate limiting on login endpoint",
                "Use strong passwords (enforce password policy)",
                "Monitor for suspicious authentication patterns",
                "Implement account lockout after failed attempts",
                "Use HTTPS/TLS for all API communication"
            ]
        },
        
        "troubleshooting": {
            "login_fails": {
                "symptom": "Cannot login with correct credentials",
                "checks": [
                    "Verify username and password are correct",
                    "Check if account is active",
                    "Verify API server is running",
                    "Check network connectivity",
                    "Review server logs for errors"
                ]
            },
            "token_immediately_invalid": {
                "symptom": "Token works initially but fails immediately",
                "checks": [
                    "Verify token is being stored correctly",
                    "Check token format in Authorization header",
                    "Ensure 'Bearer ' prefix is included",
                    "Verify no extra whitespace in token"
                ]
            },
            "frequent_401_errors": {
                "symptom": "Getting 401 errors frequently",
                "checks": [
                    "Check if token has expired (1 hour lifetime)",
                    "Verify system clocks are synchronized",
                    "Implement token refresh logic",
                    "Check if token is being cleared unexpectedly"
                ]
            },
            "403_on_own_resources": {
                "symptom": "Getting 403 when accessing own resources",
                "checks": [
                    "Verify user ID matches resource owner",
                    "Check user role and permissions",
                    "Review authorization logic",
                    "Check server logs for authorization decisions"
                ]
            }
        }
    }

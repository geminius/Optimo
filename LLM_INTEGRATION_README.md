# NVIDIA NIM LLM Integration

This document describes the integration of NVIDIA's NIM (NVIDIA Inference Microservice) with the Robotics Model Optimization Platform for intelligent validation and recommendations.

## Overview

The LLM integration adds AI-powered validation and recommendation capabilities to the optimization platform:

- **Intelligent Validation**: LLM analyzes optimization results and provides reasoning
- **Smart Recommendations**: Context-aware suggestions for improving optimization
- **Quality Assessment**: Automated evaluation of optimization trade-offs
- **Error Analysis**: Intelligent diagnosis of optimization issues

## Configuration

### Environment Variables

Add these variables to your `.env` file:

```bash
# NVIDIA NIM Configuration (OpenAI-compatible)
OPENAI_API_KEY=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
OPENAI_BASE_URL=https://integrate.api.nvidia.com/v1
LLM_MODEL=deepseek-ai/deepseek-v3.2

# LLM Service Settings
LLM_ENABLED=true
LLM_TIMEOUT_SECONDS=30
LLM_MAX_RETRIES=3
LLM_CACHE_ENABLED=true
LLM_CACHE_TTL_SECONDS=3600

# LLM Validation Settings
LLM_VALIDATION_ENABLED=true
LLM_CONFIDENCE_THRESHOLD=0.8
LLM_MAX_TOKENS=2048
```

### Configuration File

The `config/default.json` file includes LLM settings:

```json
{
  "llm_service": {
    "enabled": true,
    "validation_enabled": true,
    "confidence_threshold": 0.8,
    "timeout_seconds": 30,
    "max_retries": 3,
    "cache_enabled": true,
    "cache_ttl_seconds": 3600,
    "max_tokens": 2048
  }
}
```

## Features

### 1. LLM-Enhanced Evaluation Agent

The `EvaluationAgent` now includes LLM-based validation:

```python
# Automatic LLM validation during evaluation
evaluation_agent = EvaluationAgent({
    "llm_validation_enabled": True,
    "llm_confidence_threshold": 0.8
})

# Evaluation includes LLM insights
report = await evaluation_agent.evaluate_model(model, benchmarks)
# report.recommendations now includes AI-generated suggestions
```

### 2. Validation Service

Validate optimization results with AI reasoning:

```python
from src.services.llm_service import llm_service, ValidationRequest

# Create validation request
request = ValidationRequest(
    validation_type="robotics_optimization_evaluation",
    model_metrics={
        "inference_time_ms": 45.2,
        "memory_usage_mb": 512.0,
        "accuracy": 0.94
    },
    optimization_config={
        "techniques_applied": ["quantization", "pruning"],
        "target_accuracy": 0.95
    }
)

# Get LLM validation
result = await llm_service.validate_optimization_result(request)
print(f"Valid: {result.is_valid}")
print(f"Reasoning: {result.reasoning}")
```

### 3. Recommendation Generation

Generate intelligent optimization recommendations:

```python
recommendations = await llm_service.generate_recommendations(
    model_metrics={
        "inference_time_ms": 85.0,
        "memory_usage_mb": 800.0,
        "accuracy": 0.92
    },
    optimization_config={
        "techniques_applied": ["quantization"],
        "target_size_mb": 150.0
    },
    context={
        "deployment_target": "nvidia_jetson",
        "performance_issues": ["high_memory_usage"]
    }
)
```

### 4. API Endpoints

New REST endpoints for LLM functionality:

- `GET /api/v1/llm/health` - LLM service health check
- `POST /api/v1/llm/validate` - Validate optimization results
- `POST /api/v1/llm/recommendations` - Generate recommendations
- `POST /api/v1/llm/cache/clear` - Clear response cache
- `GET /api/v1/llm/config` - Get LLM configuration

## Usage Examples

### Basic Health Check

```bash
curl -X GET "http://localhost:8000/api/v1/llm/health"
```

### Validate Optimization Results

```bash
curl -X POST "http://localhost:8000/api/v1/llm/validate" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d '{
    "validation_type": "robotics_optimization_evaluation",
    "model_metrics": {
      "inference_time_ms": 45.2,
      "memory_usage_mb": 512.0,
      "accuracy": 0.94
    },
    "optimization_config": {
      "techniques_applied": ["quantization", "pruning"]
    }
  }'
```

### Generate Recommendations

```bash
curl -X POST "http://localhost:8000/api/v1/llm/recommendations" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d '{
    "model_metrics": {
      "inference_time_ms": 85.0,
      "memory_usage_mb": 800.0,
      "accuracy": 0.92
    },
    "optimization_config": {
      "techniques_applied": ["quantization"]
    }
  }'
```

## Testing

Run the integration test script:

```bash
python test_llm_integration.py
```

This will test:
- Environment configuration
- LLM service health
- Validation functionality
- Recommendation generation
- Response caching

## Architecture

### Service Layer

- **LLMService**: Singleton service managing NVIDIA NIM integration
- **Caching**: Intelligent response caching with TTL
- **Retry Logic**: Robust error handling with exponential backoff
- **Health Monitoring**: Continuous service health checks

### Integration Points

1. **Evaluation Agent**: Automatic LLM validation during model evaluation
2. **API Layer**: REST endpoints for direct LLM access
3. **Error Handling**: Custom exceptions for LLM-specific errors
4. **Configuration**: Environment-based configuration with defaults

### Data Flow

```
Model Optimization → Evaluation Agent → LLM Service → NVIDIA NIM
                                    ↓
Results + AI Insights ← Response Processing ← LLM Response
```

## Error Handling

The integration includes comprehensive error handling:

- **LLMServiceError**: Network and API errors
- **LLMValidationError**: Validation-specific errors
- **Graceful Degradation**: Fallback to standard validation if LLM unavailable
- **Retry Logic**: Automatic retry with exponential backoff

## Performance Considerations

### Caching Strategy

- **Response Caching**: Identical requests return cached responses
- **TTL Management**: Configurable cache expiration (default: 1 hour)
- **Cache Cleanup**: Automatic cleanup of old entries

### Rate Limiting

- **Timeout Configuration**: Configurable request timeouts
- **Retry Limits**: Maximum retry attempts to prevent infinite loops
- **Concurrent Requests**: Service handles multiple concurrent requests

### Optimization

- **Async Operations**: All LLM calls are asynchronous
- **Batch Processing**: Future support for batch validation requests
- **Selective Usage**: LLM validation can be disabled per request

## Security

### API Key Management

- **Environment Variables**: API keys stored in environment, not code
- **Key Masking**: API keys masked in logs and health checks
- **Secure Transmission**: All requests use HTTPS

### Input Validation

- **Request Validation**: All inputs validated before sending to LLM
- **Response Sanitization**: LLM responses validated before processing
- **Error Sanitization**: Error messages sanitized to prevent information leakage

## Monitoring

### Health Checks

- **Service Health**: Regular health checks of LLM service
- **Response Time**: Monitoring of LLM response times
- **Error Rates**: Tracking of failed requests and error types

### Metrics

- **Cache Hit Rate**: Percentage of requests served from cache
- **Average Response Time**: Mean response time for LLM requests
- **Request Volume**: Number of LLM requests per time period

## Troubleshooting

### Common Issues

1. **API Key Invalid**
   - Check `OPENAI_API_KEY` environment variable
   - Verify key is active in NVIDIA NIM console

2. **Service Unavailable**
   - Check `OPENAI_BASE_URL` configuration
   - Verify network connectivity to NVIDIA NIM

3. **Model Not Found**
   - Verify `LLM_MODEL` is available in your NIM instance
   - Check model name spelling and availability

4. **Timeout Errors**
   - Increase `LLM_TIMEOUT_SECONDS` if needed
   - Check network latency to NVIDIA NIM

### Debug Mode

Enable debug logging:

```bash
export LOG_LEVEL=DEBUG
```

### Health Check

Check service status:

```bash
curl http://localhost:8000/api/v1/llm/health
```

## Future Enhancements

### Planned Features

1. **Batch Processing**: Support for batch validation requests
2. **Custom Prompts**: User-configurable validation prompts
3. **Model Selection**: Dynamic model selection based on task
4. **Advanced Caching**: Semantic caching based on request similarity
5. **Metrics Dashboard**: Real-time LLM service metrics

### Integration Opportunities

1. **Planning Agent**: LLM-assisted optimization strategy selection
2. **Analysis Agent**: AI-powered model architecture analysis
3. **Monitoring**: Intelligent anomaly detection in optimization results
4. **Documentation**: Auto-generated optimization reports

## Support

For issues with the LLM integration:

1. Check the troubleshooting section above
2. Run the test script: `python test_llm_integration.py`
3. Check logs for detailed error messages
4. Verify NVIDIA NIM service status and quotas
#!/usr/bin/env python3
"""
Test script for NVIDIA NIM LLM integration.

This script tests the LLM service integration with the robotics model
optimization platform using your NVIDIA NIM configuration.
"""

import asyncio
import os
import sys
import json
from pathlib import Path

# Add src to path so we can import modules
sys.path.insert(0, str(Path(__file__).parent / "src"))

from services.llm_service import llm_service, ValidationRequest
from utils.exceptions import LLMServiceError, LLMValidationError


async def test_llm_health():
    """Test LLM service health check."""
    print("=" * 60)
    print("Testing LLM Service Health Check")
    print("=" * 60)
    
    try:
        health = await llm_service.health_check()
        print(f"Health Status: {health}")
        
        if health.get("status") == "healthy":
            print("✓ LLM service is healthy")
            return True
        else:
            print("✗ LLM service is not healthy")
            return False
            
    except Exception as e:
        print(f"✗ Health check failed: {e}")
        return False


async def test_llm_validation():
    """Test LLM-based validation."""
    print("\n" + "=" * 60)
    print("Testing LLM Validation")
    print("=" * 60)
    
    # Create sample validation request
    validation_request = ValidationRequest(
        validation_type="robotics_optimization_evaluation",
        model_metrics={
            "inference_time_ms": 45.2,
            "memory_usage_mb": 512.0,
            "model_size_mb": 150.0,
            "accuracy": 0.94,
            "throughput_samples_per_sec": 22.1
        },
        optimization_config={
            "techniques_applied": ["quantization", "pruning"],
            "quantization_bits": 8,
            "pruning_ratio": 0.3,
            "target_accuracy": 0.95
        },
        context={
            "platform": "robotics_model_optimization",
            "deployment_target": "edge_devices",
            "original_model_size_mb": 200.0,
            "original_inference_time_ms": 60.0
        }
    )
    
    try:
        result = await llm_service.validate_optimization_result(validation_request)
        
        print(f"Validation Result:")
        print(f"  Valid: {result.is_valid}")
        print(f"  Confidence: {result.confidence_score:.2f}")
        print(f"  Reasoning: {result.reasoning}")
        print(f"  Recommendations: {len(result.recommendations)}")
        
        for i, rec in enumerate(result.recommendations, 1):
            print(f"    {i}. {rec}")
        
        if result.warnings:
            print(f"  Warnings: {len(result.warnings)}")
            for warning in result.warnings:
                print(f"    - {warning}")
        
        if result.errors:
            print(f"  Errors: {len(result.errors)}")
            for error in result.errors:
                print(f"    - {error}")
        
        print("✓ LLM validation completed successfully")
        return True
        
    except LLMValidationError as e:
        print(f"✗ LLM validation error: {e}")
        return False
    except Exception as e:
        print(f"✗ Validation failed: {e}")
        return False


async def test_llm_recommendations():
    """Test LLM recommendation generation."""
    print("\n" + "=" * 60)
    print("Testing LLM Recommendations")
    print("=" * 60)
    
    model_metrics = {
        "inference_time_ms": 85.0,
        "memory_usage_mb": 800.0,
        "model_size_mb": 300.0,
        "accuracy": 0.92,
        "throughput_samples_per_sec": 12.0
    }
    
    optimization_config = {
        "techniques_applied": ["quantization"],
        "quantization_bits": 8,
        "target_size_mb": 150.0,
        "target_inference_time_ms": 50.0
    }
    
    context = {
        "platform": "robotics_optimization",
        "deployment_target": "nvidia_jetson",
        "performance_issues": ["high_memory_usage", "slow_inference"]
    }
    
    try:
        recommendations = await llm_service.generate_recommendations(
            model_metrics=model_metrics,
            optimization_config=optimization_config,
            context=context
        )
        
        print(f"Generated {len(recommendations)} recommendations:")
        for i, rec in enumerate(recommendations, 1):
            print(f"  {i}. {rec}")
        
        print("✓ LLM recommendations generated successfully")
        return True
        
    except Exception as e:
        print(f"✗ Recommendation generation failed: {e}")
        return False


async def test_llm_caching():
    """Test LLM response caching."""
    print("\n" + "=" * 60)
    print("Testing LLM Caching")
    print("=" * 60)
    
    # Make the same request twice to test caching
    model_metrics = {"accuracy": 0.95, "inference_time_ms": 30.0}
    optimization_config = {"technique": "quantization"}
    
    try:
        # First request (should hit LLM)
        print("Making first request...")
        start_time = asyncio.get_event_loop().time()
        recommendations1 = await llm_service.generate_recommendations(
            model_metrics=model_metrics,
            optimization_config=optimization_config
        )
        first_duration = asyncio.get_event_loop().time() - start_time
        
        # Second request (should use cache)
        print("Making second request...")
        start_time = asyncio.get_event_loop().time()
        recommendations2 = await llm_service.generate_recommendations(
            model_metrics=model_metrics,
            optimization_config=optimization_config
        )
        second_duration = asyncio.get_event_loop().time() - start_time
        
        print(f"First request duration: {first_duration:.3f}s")
        print(f"Second request duration: {second_duration:.3f}s")
        
        if second_duration < first_duration * 0.5:  # Cache should be much faster
            print("✓ Caching appears to be working")
        else:
            print("? Caching may not be working as expected")
        
        # Clear cache
        llm_service.clear_cache()
        print("✓ Cache cleared")
        
        return True
        
    except Exception as e:
        print(f"✗ Caching test failed: {e}")
        return False


def check_environment():
    """Check if environment is properly configured."""
    print("=" * 60)
    print("Checking Environment Configuration")
    print("=" * 60)
    
    required_vars = ["OPENAI_API_KEY", "OPENAI_BASE_URL", "LLM_MODEL"]
    missing_vars = []
    
    for var in required_vars:
        value = os.getenv(var)
        if value:
            if var == "OPENAI_API_KEY":
                # Mask the API key for security
                masked_value = value[:8] + "..." + value[-4:] if len(value) > 12 else "***"
                print(f"✓ {var}: {masked_value}")
            else:
                print(f"✓ {var}: {value}")
        else:
            print(f"✗ {var}: Not set")
            missing_vars.append(var)
    
    # Check optional settings
    optional_vars = [
        "LLM_ENABLED", "LLM_TIMEOUT_SECONDS", "LLM_MAX_RETRIES",
        "LLM_CACHE_ENABLED", "LLM_VALIDATION_ENABLED"
    ]
    
    print("\nOptional settings:")
    for var in optional_vars:
        value = os.getenv(var)
        if value:
            print(f"  {var}: {value}")
        else:
            print(f"  {var}: Using default")
    
    if missing_vars:
        print(f"\n✗ Missing required environment variables: {', '.join(missing_vars)}")
        print("\nPlease set these variables in your .env file or environment:")
        print("  OPENAI_API_KEY=nvapi-your-api-key-here")
        print("  OPENAI_BASE_URL=https://integrate.api.nvidia.com/v1")
        print("  LLM_MODEL=deepseek-ai/deepseek-v3.2")
        return False
    
    print("\n✓ Environment configuration looks good")
    return True


async def main():
    """Run all LLM integration tests."""
    print("NVIDIA NIM LLM Integration Test")
    print("=" * 60)
    
    # Check environment first
    if not check_environment():
        print("\n✗ Environment check failed. Please fix configuration and try again.")
        return False
    
    # Check if LLM service is available
    if not llm_service.is_available():
        print("\n✗ LLM service is not available. Check your configuration.")
        return False
    
    print(f"\nLLM Service Configuration:")
    print(f"  Enabled: {llm_service.enabled}")
    print(f"  Model: {llm_service.model}")
    print(f"  Base URL: {llm_service.base_url}")
    print(f"  Timeout: {llm_service.timeout_seconds}s")
    print(f"  Cache Enabled: {llm_service.cache_enabled}")
    
    # Run tests
    tests = [
        ("Health Check", test_llm_health),
        ("Validation", test_llm_validation),
        ("Recommendations", test_llm_recommendations),
        ("Caching", test_llm_caching),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = await test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n✗ {test_name} test failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    passed = 0
    for test_name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All tests passed! LLM integration is working correctly.")
        return True
    else:
        print("❌ Some tests failed. Please check the configuration and try again.")
        return False


if __name__ == "__main__":
    # Load environment variables from .env file if it exists
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        print("Note: python-dotenv not installed. Using system environment variables only.")
    
    # Run the tests
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
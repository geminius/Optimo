"""
LLM Service for NVIDIA NIM integration.

This service provides LLM-based validation and assessment capabilities
using NVIDIA's NIM service through OpenAI-compatible API.
"""

import os
import json
import asyncio
import logging
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import threading
import time
from enum import Enum

import aiohttp
import openai
from openai import AsyncOpenAI

from ..utils.exceptions import PlatformError, ErrorCategory, ErrorSeverity
from ..utils.retry import RetryConfig, RetryableOperation


logger = logging.getLogger(__name__)


class LLMServiceError(PlatformError):
    """Errors related to LLM service operations."""
    
    def __init__(
        self,
        message: str,
        model: Optional[str] = None,
        request_type: Optional[str] = None,
        **kwargs
    ):
        context = kwargs.get('context', {})
        if model:
            context['model'] = model
        if request_type:
            context['request_type'] = request_type
        
        super().__init__(
            message=message,
            category=ErrorCategory.NETWORK,
            context=context,
            **kwargs
        )


class LLMValidationError(PlatformError):
    """Errors during LLM-based validation."""
    
    def __init__(
        self,
        message: str,
        validation_type: Optional[str] = None,
        confidence_score: Optional[float] = None,
        **kwargs
    ):
        context = kwargs.get('context', {})
        if validation_type:
            context['validation_type'] = validation_type
        if confidence_score is not None:
            context['confidence_score'] = confidence_score
        
        super().__init__(
            message=message,
            category=ErrorCategory.VALIDATION,
            context=context,
            **kwargs
        )


@dataclass
class LLMResponse:
    """Response from LLM service."""
    content: str
    confidence_score: float
    model: str
    tokens_used: int
    response_time_ms: float
    timestamp: datetime
    cached: bool = False


@dataclass
class ValidationRequest:
    """Request for LLM-based validation."""
    validation_type: str
    model_metrics: Dict[str, Any]
    optimization_config: Dict[str, Any]
    context: Optional[Dict[str, Any]] = None


@dataclass
class ValidationResponse:
    """Response from LLM validation."""
    is_valid: bool
    confidence_score: float
    reasoning: str
    recommendations: List[str]
    warnings: List[str]
    errors: List[str]
    metadata: Dict[str, Any]


class LLMService:
    """
    Service for interacting with NVIDIA NIM through OpenAI-compatible API.
    
    Provides LLM-based validation, assessment, and recommendation generation
    for robotics model optimization.
    """
    
    _instance = None
    _lock = threading.RLock()
    
    def __new__(cls):
        """Singleton pattern to ensure single instance."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self):
        """Initialize the LLM service."""
        if self._initialized:
            return
        
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Configuration from environment
        self.enabled = os.getenv("LLM_ENABLED", "true").lower() == "true"
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.base_url = os.getenv("OPENAI_BASE_URL", "https://integrate.api.nvidia.com/v1")
        self.model = os.getenv("LLM_MODEL", "deepseek-ai/deepseek-v3.2")
        
        # Service settings
        self.timeout_seconds = int(os.getenv("LLM_TIMEOUT_SECONDS", "30"))
        self.max_retries = int(os.getenv("LLM_MAX_RETRIES", "3"))
        self.cache_enabled = os.getenv("LLM_CACHE_ENABLED", "true").lower() == "true"
        self.cache_ttl_seconds = int(os.getenv("LLM_CACHE_TTL_SECONDS", "3600"))
        
        # Validation settings
        self.validation_enabled = os.getenv("LLM_VALIDATION_ENABLED", "true").lower() == "true"
        self.confidence_threshold = float(os.getenv("LLM_CONFIDENCE_THRESHOLD", "0.8"))
        self.max_tokens = int(os.getenv("LLM_MAX_TOKENS", "2048"))
        
        # Initialize OpenAI client
        self.client = None
        if self.enabled and self.api_key:
            self.client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.timeout_seconds
            )
        
        # Response cache
        self._cache = {}
        self._cache_lock = threading.RLock()
        
        # Retry configuration
        self.retry_config = RetryConfig(
            max_attempts=self.max_retries,
            base_delay=1.0,
            max_delay=10.0,
            exponential_base=2.0
        )
        
        self._initialized = True
        self.logger.info(f"LLMService initialized - Enabled: {self.enabled}, Model: {self.model}")
    
    def is_available(self) -> bool:
        """Check if LLM service is available."""
        return self.enabled and self.client is not None and self.api_key is not None
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on LLM service."""
        if not self.is_available():
            return {
                "status": "unavailable",
                "enabled": self.enabled,
                "configured": self.client is not None,
                "error": "Service not properly configured"
            }
        
        try:
            # Simple test request
            start_time = time.time()
            response = await self._make_request(
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=10
            )
            response_time = (time.time() - start_time) * 1000
            
            return {
                "status": "healthy",
                "model": self.model,
                "response_time_ms": response_time,
                "cache_size": len(self._cache)
            }
            
        except Exception as e:
            self.logger.error(f"LLM service health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e)
            }
    
    async def validate_optimization_result(
        self,
        request: ValidationRequest
    ) -> ValidationResponse:
        """
        Validate optimization results using LLM analysis.
        
        Args:
            request: Validation request with metrics and configuration
            
        Returns:
            ValidationResponse with LLM assessment
        """
        if not self.validation_enabled or not self.is_available():
            return ValidationResponse(
                is_valid=True,
                confidence_score=0.0,
                reasoning="LLM validation disabled or unavailable",
                recommendations=[],
                warnings=["LLM validation not performed"],
                errors=[],
                metadata={"llm_used": False}
            )
        
        try:
            # Create validation prompt
            prompt = self._create_validation_prompt(request)
            
            # Make LLM request
            llm_response = await self._make_llm_request(prompt)
            
            # Parse response
            validation_result = self._parse_validation_response(llm_response)
            
            self.logger.info(
                f"LLM validation completed - Valid: {validation_result.is_valid}, "
                f"Confidence: {validation_result.confidence_score:.2f}"
            )
            
            return validation_result
            
        except Exception as e:
            self.logger.error(f"LLM validation failed: {e}")
            raise LLMValidationError(
                f"Validation failed: {str(e)}",
                validation_type=request.validation_type
            )
    
    async def generate_recommendations(
        self,
        model_metrics: Dict[str, Any],
        optimization_config: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """
        Generate optimization recommendations using LLM.
        
        Args:
            model_metrics: Current model performance metrics
            optimization_config: Optimization configuration used
            context: Additional context information
            
        Returns:
            List of recommendation strings
        """
        if not self.is_available():
            return ["LLM service unavailable for recommendations"]
        
        try:
            prompt = self._create_recommendation_prompt(
                model_metrics, optimization_config, context
            )
            
            llm_response = await self._make_llm_request(prompt)
            recommendations = self._parse_recommendations_response(llm_response)
            
            self.logger.info(f"Generated {len(recommendations)} LLM recommendations")
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Failed to generate LLM recommendations: {e}")
            return [f"Error generating recommendations: {str(e)}"]
    
    async def _make_llm_request(self, prompt: str) -> LLMResponse:
        """Make request to LLM service with caching and retry logic."""
        # Check cache first
        cache_key = self._get_cache_key(prompt)
        if self.cache_enabled:
            cached_response = self._get_cached_response(cache_key)
            if cached_response:
                return cached_response
        
        # Make request with retry logic
        retry_operation = RetryableOperation(
            operation=self._make_request,
            config=self.retry_config,
            operation_name="llm_request"
        )
        
        start_time = time.time()
        
        try:
            messages = [{"role": "user", "content": prompt}]
            response = await retry_operation.execute(
                messages=messages,
                max_tokens=self.max_tokens
            )
            
            response_time_ms = (time.time() - start_time) * 1000
            
            # Create LLM response object
            llm_response = LLMResponse(
                content=response.choices[0].message.content,
                confidence_score=self._estimate_confidence(response),
                model=self.model,
                tokens_used=response.usage.total_tokens if response.usage else 0,
                response_time_ms=response_time_ms,
                timestamp=datetime.now(),
                cached=False
            )
            
            # Cache the response
            if self.cache_enabled:
                self._cache_response(cache_key, llm_response)
            
            return llm_response
            
        except Exception as e:
            self.logger.error(f"LLM request failed: {e}")
            raise LLMServiceError(
                f"Request failed: {str(e)}",
                model=self.model,
                request_type="completion"
            )
    
    async def _make_request(self, messages: List[Dict[str, str]], max_tokens: int):
        """Make raw request to OpenAI-compatible API."""
        if not self.client:
            raise LLMServiceError("LLM client not initialized")
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.1,  # Low temperature for consistent validation
                timeout=self.timeout_seconds
            )
            return response
            
        except openai.APITimeoutError as e:
            raise LLMServiceError(f"Request timeout: {str(e)}", model=self.model)
        except openai.APIError as e:
            raise LLMServiceError(f"API error: {str(e)}", model=self.model)
        except Exception as e:
            raise LLMServiceError(f"Unexpected error: {str(e)}", model=self.model)
    
    def _create_validation_prompt(self, request: ValidationRequest) -> str:
        """Create validation prompt for LLM."""
        prompt = f"""
You are an expert in robotics model optimization. Analyze the following optimization results and provide validation.

Validation Type: {request.validation_type}

Model Metrics:
{json.dumps(request.model_metrics, indent=2)}

Optimization Configuration:
{json.dumps(request.optimization_config, indent=2)}

Context:
{json.dumps(request.context or {}, indent=2)}

Please provide a JSON response with the following structure:
{{
    "is_valid": boolean,
    "confidence_score": float (0.0 to 1.0),
    "reasoning": "detailed explanation of your assessment",
    "recommendations": ["list", "of", "recommendations"],
    "warnings": ["list", "of", "warnings"],
    "errors": ["list", "of", "errors"]
}}

Focus on:
1. Whether the optimization achieved reasonable improvements
2. If accuracy degradation is within acceptable limits
3. Whether the optimization is suitable for robotics deployment
4. Any potential issues or concerns
5. Actionable recommendations for improvement
"""
        return prompt.strip()
    
    def _create_recommendation_prompt(
        self,
        model_metrics: Dict[str, Any],
        optimization_config: Dict[str, Any],
        context: Optional[Dict[str, Any]]
    ) -> str:
        """Create recommendation prompt for LLM."""
        prompt = f"""
You are an expert in robotics model optimization. Based on the following metrics and configuration, provide optimization recommendations.

Current Model Metrics:
{json.dumps(model_metrics, indent=2)}

Optimization Configuration Used:
{json.dumps(optimization_config, indent=2)}

Additional Context:
{json.dumps(context or {}, indent=2)}

Please provide 3-5 specific, actionable recommendations to improve the optimization results. Focus on:
1. Techniques that could further reduce model size or improve inference speed
2. Parameter adjustments for better accuracy-performance trade-offs
3. Hardware-specific optimizations for robotics deployment
4. Potential issues to watch out for
5. Next steps for production deployment

Provide your response as a JSON array of recommendation strings:
["recommendation 1", "recommendation 2", "recommendation 3"]
"""
        return prompt.strip()
    
    def _parse_validation_response(self, llm_response: LLMResponse) -> ValidationResponse:
        """Parse LLM response into ValidationResponse."""
        try:
            # Try to parse JSON response
            response_data = json.loads(llm_response.content)
            
            return ValidationResponse(
                is_valid=response_data.get("is_valid", True),
                confidence_score=min(
                    response_data.get("confidence_score", 0.5),
                    llm_response.confidence_score
                ),
                reasoning=response_data.get("reasoning", "No reasoning provided"),
                recommendations=response_data.get("recommendations", []),
                warnings=response_data.get("warnings", []),
                errors=response_data.get("errors", []),
                metadata={
                    "llm_used": True,
                    "model": llm_response.model,
                    "tokens_used": llm_response.tokens_used,
                    "response_time_ms": llm_response.response_time_ms
                }
            )
            
        except json.JSONDecodeError:
            # Fallback: treat as plain text
            return ValidationResponse(
                is_valid=True,
                confidence_score=llm_response.confidence_score,
                reasoning=llm_response.content,
                recommendations=[],
                warnings=["Could not parse structured LLM response"],
                errors=[],
                metadata={
                    "llm_used": True,
                    "model": llm_response.model,
                    "parse_error": True
                }
            )
    
    def _parse_recommendations_response(self, llm_response: LLMResponse) -> List[str]:
        """Parse LLM response into recommendations list."""
        try:
            # Try to parse as JSON array
            recommendations = json.loads(llm_response.content)
            if isinstance(recommendations, list):
                return [str(rec) for rec in recommendations]
            else:
                return [str(recommendations)]
                
        except json.JSONDecodeError:
            # Fallback: split by lines
            lines = llm_response.content.strip().split('\n')
            return [line.strip() for line in lines if line.strip()]
    
    def _estimate_confidence(self, response) -> float:
        """Estimate confidence score from OpenAI response."""
        # Simple heuristic based on response characteristics
        if not response.choices or not response.choices[0].message.content:
            return 0.0
        
        content = response.choices[0].message.content
        
        # Longer, more detailed responses tend to be more confident
        length_score = min(len(content) / 1000, 1.0)
        
        # Presence of structured data suggests higher confidence
        structure_score = 0.5
        if '{' in content and '}' in content:
            structure_score = 0.8
        
        # Combine scores
        confidence = (length_score * 0.3 + structure_score * 0.7)
        return min(max(confidence, 0.1), 1.0)
    
    def _get_cache_key(self, prompt: str) -> str:
        """Generate cache key for prompt."""
        import hashlib
        return hashlib.md5(prompt.encode()).hexdigest()
    
    def _get_cached_response(self, cache_key: str) -> Optional[LLMResponse]:
        """Get cached response if available and not expired."""
        with self._cache_lock:
            if cache_key in self._cache:
                cached_response, timestamp = self._cache[cache_key]
                
                # Check if cache is still valid
                if datetime.now() - timestamp < timedelta(seconds=self.cache_ttl_seconds):
                    cached_response.cached = True
                    return cached_response
                else:
                    # Remove expired cache entry
                    del self._cache[cache_key]
        
        return None
    
    def _cache_response(self, cache_key: str, response: LLMResponse) -> None:
        """Cache LLM response."""
        with self._cache_lock:
            self._cache[cache_key] = (response, datetime.now())
            
            # Simple cache cleanup - remove oldest entries if cache is too large
            if len(self._cache) > 1000:
                # Remove 10% of oldest entries
                sorted_items = sorted(
                    self._cache.items(),
                    key=lambda x: x[1][1]  # Sort by timestamp
                )
                
                for i in range(len(sorted_items) // 10):
                    del self._cache[sorted_items[i][0]]
    
    def clear_cache(self) -> None:
        """Clear response cache."""
        with self._cache_lock:
            self._cache.clear()
        self.logger.info("LLM response cache cleared")


# Global service instance
llm_service = LLMService()
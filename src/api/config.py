"""
Configuration API endpoints for optimization criteria management.

This module provides REST API endpoints for retrieving and updating
optimization criteria configuration.
"""

import logging
from typing import Optional
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, status, Request

from .models import OptimizationCriteriaRequest, OptimizationCriteriaResponse, ErrorResponse
from .dependencies import get_admin_user, get_current_user
from .auth import User
from ..services.config_manager import ConfigurationManager
from ..services.cache_service import CacheService
from ..config.optimization_criteria import (
    OptimizationCriteria,
    OptimizationConstraints,
    OptimizationTechnique,
    PerformanceMetric,
    PerformanceThreshold
)


logger = logging.getLogger(__name__)

# Cache TTL for configuration data (5 minutes)
CONFIG_CACHE_TTL = 300.0

router = APIRouter(prefix="/config", tags=["configuration"])


async def get_config_manager(request: Request) -> ConfigurationManager:
    """
    Dependency to get ConfigurationManager instance.
    
    Args:
        request: FastAPI request object
        
    Returns:
        ConfigurationManager instance
        
    Raises:
        HTTPException: If ConfigurationManager is not available
    """
    if not hasattr(request.app.state, 'config_manager'):
        # Create singleton instance if not exists
        config_manager = ConfigurationManager()
        request.app.state.config_manager = config_manager
    
    return request.app.state.config_manager


@router.get(
    "/optimization-criteria",
    response_model=OptimizationCriteriaResponse,
    summary="Get Optimization Criteria Configuration",
    description="""
    Retrieve the current optimization criteria configuration.
    
    This endpoint returns the active optimization criteria that will be used
    for new optimization sessions. The configuration includes:
    
    - **Accuracy thresholds**: Minimum accuracy to maintain during optimization
    - **Size reduction targets**: Maximum allowed model size reduction
    - **Latency constraints**: Maximum allowed inference time increase
    - **Enabled techniques**: Which optimization techniques are allowed
    - **Hardware constraints**: Target device specifications
    - **Custom parameters**: Additional configuration options
    
    **Default Configuration:**
    If no custom configuration has been set, default values are returned:
    - Target accuracy threshold: 0.95 (95%)
    - Max size reduction: 50%
    - Max latency increase: 10%
    - Enabled techniques: All available techniques
    
    **Use Cases:**
    - Display current settings in configuration UI
    - Verify configuration before starting optimization
    - Audit configuration changes
    - Export configuration for backup
    
    **Authentication:**
    Requires valid Bearer token. All authenticated users can view configuration.
    
    **Example Response:**
    ```json
    {
      "name": "edge_deployment",
      "target_accuracy_threshold": 0.95,
      "max_size_reduction_percent": 60.0,
      "max_latency_increase_percent": 5.0,
      "optimization_techniques": ["quantization", "pruning"],
      "hardware_constraints": {
        "max_memory_mb": 512,
        "target_device": "jetson_nano"
      },
      "created_at": "2024-01-01T10:00:00Z",
      "updated_at": "2024-01-01T12:00:00Z"
    }
    ```
    """,
    responses={
        200: {
            "description": "Configuration retrieved successfully",
            "content": {
                "application/json": {
                    "example": {
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
                }
            }
        },
        401: {
            "description": "Unauthorized - Invalid or missing authentication token",
            "model": ErrorResponse
        },
        500: {
            "description": "Internal server error - Failed to load configuration",
            "model": ErrorResponse
        }
    }
)
async def get_optimization_criteria(
    current_user: User = Depends(get_current_user),
    config_manager: ConfigurationManager = Depends(get_config_manager)
) -> OptimizationCriteriaResponse:
    """
    Get current optimization criteria configuration.
    
    This endpoint retrieves the active optimization criteria configuration.
    If no configuration file exists, default values are returned.
    
    Args:
        current_user: Authenticated user (from dependency)
        config_manager: ConfigurationManager instance (from dependency)
        
    Returns:
        OptimizationCriteriaResponse with current configuration
        
    Raises:
        HTTPException: If configuration cannot be loaded
    """
    try:
        logger.info(
            f"Loading optimization criteria configuration",
            extra={
                "component": "ConfigAPI",
                "user_id": current_user.id,
                "username": current_user.username,
                "role": current_user.role
            }
        )
        
        # Get cache service
        cache_service = CacheService()
        cache_key = "config:optimization_criteria"
        
        # Try to get from cache first
        def load_config() -> OptimizationCriteriaResponse:
            """Load configuration (called if cache miss)."""
            criteria = config_manager.get_current_configuration()
            return _criteria_to_response(criteria)
        
        # Get from cache or load
        response = cache_service.get_or_set(
            cache_key,
            load_config,
            ttl_seconds=CONFIG_CACHE_TTL
        )
        
        logger.info(
            f"Successfully loaded configuration: {response.name}",
            extra={"user_id": current_user.id, "config_name": response.name}
        )
        
        return response
        
    except Exception as e:
        logger.error(
            f"Failed to load optimization criteria: {e}",
            extra={"user_id": current_user.id},
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load configuration: {str(e)}"
        )


@router.put(
    "/optimization-criteria",
    response_model=OptimizationCriteriaResponse,
    summary="Update Optimization Criteria Configuration",
    description="""
    Update the optimization criteria configuration.
    
    This endpoint allows administrators to modify the optimization criteria
    that will be applied to new optimization sessions. The configuration
    is validated before being saved to ensure consistency and correctness.
    
    **Authorization:**
    - Requires administrator role
    - Regular users will receive 403 Forbidden
    - All configuration changes are logged for audit
    
    **Validation Rules:**
    - `target_accuracy_threshold`: Must be between 0.0 and 1.0
    - `max_size_reduction_percent`: Must be between 0.0 and 100.0
    - `max_latency_increase_percent`: Must be >= 0.0
    - `optimization_techniques`: Must be valid technique names
    - Hardware constraints must be compatible with techniques
    
    **Validation Errors:**
    If validation fails, a 400 error is returned with detailed error messages:
    ```json
    {
      "message": "Configuration validation failed",
      "errors": [
        "target_accuracy_threshold must be between 0.0 and 1.0",
        "Unknown optimization technique: invalid_technique"
      ],
      "warnings": [
        "High size reduction may impact accuracy"
      ]
    }
    ```
    
    **Available Optimization Techniques:**
    - `quantization`: Reduce precision (INT8, INT4, etc.)
    - `pruning`: Remove unnecessary weights
    - `distillation`: Compress via knowledge distillation
    - `compression`: General compression techniques
    - `architecture_search`: Neural architecture optimization
    
    **Use Cases:**
    - Configure platform for specific deployment targets
    - Adjust optimization aggressiveness
    - Enable/disable specific techniques
    - Set hardware-specific constraints
    
    **Example Request:**
    ```bash
    curl -X PUT http://localhost:8000/config/optimization-criteria \\
      -H "Authorization: Bearer YOUR_ADMIN_TOKEN" \\
      -H "Content-Type: application/json" \\
      -d '{
        "name": "edge_deployment",
        "target_accuracy_threshold": 0.95,
        "max_size_reduction_percent": 60.0,
        "max_latency_increase_percent": 5.0,
        "optimization_techniques": ["quantization", "pruning"],
        "hardware_constraints": {
          "max_memory_mb": 512,
          "target_device": "jetson_nano"
        }
      }'
    ```
    
    **Configuration Persistence:**
    - Configuration is saved to disk immediately
    - Applies to all new optimization sessions
    - Existing sessions continue with their original configuration
    - Configuration survives server restarts
    """,
    responses={
        200: {
            "description": "Configuration updated successfully",
            "content": {
                "application/json": {
                    "example": {
                        "name": "edge_deployment",
                        "target_accuracy_threshold": 0.95,
                        "max_size_reduction_percent": 60.0,
                        "max_latency_increase_percent": 5.0,
                        "optimization_techniques": ["quantization", "pruning"],
                        "hardware_constraints": {
                            "max_memory_mb": 512,
                            "target_device": "jetson_nano"
                        },
                        "custom_parameters": {},
                        "created_at": "2024-01-01T10:00:00Z",
                        "updated_at": "2024-01-01T12:00:00Z"
                    }
                }
            }
        },
        400: {
            "description": "Bad request - Invalid configuration data",
            "model": ErrorResponse,
            "content": {
                "application/json": {
                    "example": {
                        "error": "ValidationError",
                        "message": "Configuration validation failed",
                        "details": {
                            "errors": [
                                "target_accuracy_threshold must be between 0.0 and 1.0",
                                "Unknown optimization technique: invalid_technique"
                            ],
                            "warnings": [
                                "High size reduction target may impact accuracy"
                            ]
                        },
                        "timestamp": "2024-01-01T12:00:00Z",
                        "request_id": "req_abc123"
                    }
                }
            }
        },
        401: {
            "description": "Unauthorized - Invalid or missing authentication token",
            "model": ErrorResponse
        },
        403: {
            "description": "Forbidden - Admin access required",
            "model": ErrorResponse,
            "content": {
                "application/json": {
                    "example": {
                        "error": "Forbidden",
                        "message": "Administrator role required to update configuration",
                        "details": {
                            "required_role": "administrator",
                            "user_role": "user"
                        },
                        "timestamp": "2024-01-01T12:00:00Z",
                        "request_id": "req_def456"
                    }
                }
            }
        },
        500: {
            "description": "Internal server error - Failed to save configuration",
            "model": ErrorResponse
        }
    }
)
async def update_optimization_criteria(
    request_data: OptimizationCriteriaRequest,
    admin_user: User = Depends(get_admin_user),
    config_manager: ConfigurationManager = Depends(get_config_manager)
) -> OptimizationCriteriaResponse:
    """
    Update optimization criteria configuration.
    
    This endpoint updates the optimization criteria configuration.
    The configuration is validated before being saved. Only users
    with administrator role can update the configuration.
    
    Args:
        request_data: New optimization criteria configuration
        admin_user: Authenticated admin user (from dependency)
        config_manager: ConfigurationManager instance (from dependency)
        
    Returns:
        OptimizationCriteriaResponse with updated configuration
        
    Raises:
        HTTPException: If validation fails or configuration cannot be saved
    """
    try:
        logger.info(
            f"Admin user updating optimization criteria configuration",
            extra={
                "component": "ConfigAPI",
                "user_id": admin_user.id,
                "username": admin_user.username,
                "role": admin_user.role,
                "config_name": request_data.name,
                "action": "update_configuration"
            }
        )
        
        # Convert request to OptimizationCriteria
        criteria = _request_to_criteria(request_data)
        
        # Validate configuration
        validation_result = config_manager.validate_configuration(criteria)
        
        if not validation_result.is_valid:
            error_details = {
                "errors": validation_result.errors,
                "warnings": validation_result.warnings
            }
            logger.warning(
                f"Configuration validation failed: {validation_result.errors}",
                extra={
                    "user_id": admin_user.id,
                    "config_name": request_data.name,
                    "validation_errors": validation_result.errors
                }
            )
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "message": "Configuration validation failed",
                    "errors": validation_result.errors,
                    "warnings": validation_result.warnings
                }
            )
        
        # Log warnings if any
        if validation_result.warnings:
            for warning in validation_result.warnings:
                logger.warning(
                    f"Configuration warning: {warning}",
                    extra={"user_id": admin_user.id, "config_name": request_data.name}
                )
        
        # Update configuration
        success = config_manager.update_configuration(criteria)
        
        if not success:
            logger.error(
                f"Failed to update configuration",
                extra={"user_id": admin_user.id, "config_name": request_data.name}
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to save configuration"
            )
        
        # Invalidate cache after update
        cache_service = CacheService()
        cache_service.invalidate("config:optimization_criteria")
        logger.debug("Configuration cache invalidated after update")
        
        # Get updated configuration
        updated_criteria = config_manager.get_current_configuration()
        response = _criteria_to_response(updated_criteria)
        
        logger.info(
            f"Successfully updated configuration: {updated_criteria.name}",
            extra={
                "component": "ConfigAPI",
                "user_id": admin_user.id,
                "username": admin_user.username,
                "role": admin_user.role,
                "config_name": updated_criteria.name,
                "action": "update_configuration_success"
            }
        )
        
        return response
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(
            f"Failed to update optimization criteria: {e}",
            extra={"user_id": admin_user.id},
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update configuration: {str(e)}"
        )


def _criteria_to_response(criteria: OptimizationCriteria) -> OptimizationCriteriaResponse:
    """
    Convert OptimizationCriteria to API response model.
    
    Args:
        criteria: OptimizationCriteria object
        
    Returns:
        OptimizationCriteriaResponse
    """
    # Extract optimization techniques
    techniques = [t.value for t in criteria.constraints.allowed_techniques]
    
    # Build hardware constraints
    hardware_constraints = criteria.constraints.hardware_constraints.copy() if criteria.constraints.hardware_constraints else {}
    
    # Build custom parameters from performance thresholds and priority weights
    custom_parameters = {
        "performance_thresholds": [
            {
                "metric": threshold.metric.value,
                "min_value": threshold.min_value,
                "max_value": threshold.max_value,
                "target_value": threshold.target_value,
                "tolerance": threshold.tolerance
            }
            for threshold in criteria.performance_thresholds
        ],
        "priority_weights": {
            metric.value: weight 
            for metric, weight in criteria.priority_weights.items()
        },
        "target_deployment": criteria.target_deployment,
        "description": criteria.description
    }
    
    # Get timestamps (use current time as placeholder since OptimizationCriteria doesn't have timestamps)
    now = datetime.now()
    
    return OptimizationCriteriaResponse(
        name=criteria.name,
        target_accuracy_threshold=criteria.constraints.preserve_accuracy_threshold,
        max_size_reduction_percent=100.0,  # Not directly in OptimizationCriteria, use default
        max_latency_increase_percent=10.0,  # Not directly in OptimizationCriteria, use default
        optimization_techniques=techniques,
        hardware_constraints=hardware_constraints if hardware_constraints else None,
        custom_parameters=custom_parameters,
        created_at=now,
        updated_at=now
    )


def _request_to_criteria(request: OptimizationCriteriaRequest) -> OptimizationCriteria:
    """
    Convert API request model to OptimizationCriteria.
    
    Args:
        request: OptimizationCriteriaRequest
        
    Returns:
        OptimizationCriteria object
    """
    # Parse optimization techniques
    allowed_techniques = []
    for technique_str in request.optimization_techniques:
        try:
            technique = OptimizationTechnique(technique_str)
            allowed_techniques.append(technique)
        except ValueError:
            logger.warning(f"Unknown optimization technique: {technique_str}")
    
    # If no techniques specified, allow all
    if not allowed_techniques:
        allowed_techniques = list(OptimizationTechnique)
    
    # Build constraints
    constraints = OptimizationConstraints(
        max_optimization_time_minutes=60,  # Default value
        max_memory_usage_gb=16.0,  # Default value
        preserve_accuracy_threshold=request.target_accuracy_threshold,
        allowed_techniques=allowed_techniques,
        forbidden_techniques=[],
        hardware_constraints=request.hardware_constraints or {}
    )
    
    # Parse custom parameters for performance thresholds and priority weights
    performance_thresholds = []
    priority_weights = {}
    description = f"Optimization criteria: {request.name}"
    target_deployment = "general"
    
    if request.custom_parameters:
        # Extract performance thresholds
        if "performance_thresholds" in request.custom_parameters:
            for threshold_data in request.custom_parameters["performance_thresholds"]:
                try:
                    threshold = PerformanceThreshold(
                        metric=PerformanceMetric(threshold_data["metric"]),
                        min_value=threshold_data.get("min_value"),
                        max_value=threshold_data.get("max_value"),
                        target_value=threshold_data.get("target_value"),
                        tolerance=threshold_data.get("tolerance", 0.05)
                    )
                    performance_thresholds.append(threshold)
                except (KeyError, ValueError) as e:
                    logger.warning(f"Invalid performance threshold: {e}")
        
        # Extract priority weights
        if "priority_weights" in request.custom_parameters:
            for metric_str, weight in request.custom_parameters["priority_weights"].items():
                try:
                    metric = PerformanceMetric(metric_str)
                    priority_weights[metric] = float(weight)
                except (ValueError, TypeError) as e:
                    logger.warning(f"Invalid priority weight for {metric_str}: {e}")
        
        # Extract other parameters
        if "target_deployment" in request.custom_parameters:
            target_deployment = request.custom_parameters["target_deployment"]
        
        if "description" in request.custom_parameters:
            description = request.custom_parameters["description"]
    
    # Create default performance thresholds if none provided
    if not performance_thresholds:
        performance_thresholds = [
            PerformanceThreshold(
                metric=PerformanceMetric.ACCURACY,
                min_value=request.target_accuracy_threshold,
                tolerance=0.05
            ),
            PerformanceThreshold(
                metric=PerformanceMetric.MODEL_SIZE,
                max_value=request.max_size_reduction_percent,
                tolerance=0.1
            )
        ]
    
    # Create default priority weights if none provided
    if not priority_weights:
        priority_weights = {
            PerformanceMetric.ACCURACY: 0.5,
            PerformanceMetric.MODEL_SIZE: 0.3,
            PerformanceMetric.INFERENCE_TIME: 0.2
        }
    
    return OptimizationCriteria(
        name=request.name,
        description=description,
        performance_thresholds=performance_thresholds,
        constraints=constraints,
        priority_weights=priority_weights,
        target_deployment=target_deployment
    )

"""
Model storage and management functionality.
"""

import os
import json
import shutil
import hashlib
import logging
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any
from datetime import datetime
from functools import reduce

from .core import ModelMetadata, ModelType, ModelFramework, PerformanceProfile
from .validation import ModelFormatValidator


logger = logging.getLogger(__name__)


class ModelStoreError(Exception):
    """Exception raised for model store operations."""
    pass


class ModelStore:
    """
    Manages model storage, loading, and metadata operations.
    
    Supports multiple model formats (PyTorch, TensorFlow, ONNX) with
    versioning, validation, and profiling capabilities.
    """
    
    def __init__(self, storage_root: str = "models", enable_versioning: bool = False):
        """
        Initialize ModelStore.
        
        Args:
            storage_root: Root directory for model storage
            enable_versioning: Whether to enable model versioning
        """
        self.storage_root = Path(storage_root)
        self.enable_versioning = enable_versioning
        
        # Create storage structure
        self._create_storage_structure()
        
        # Framework-specific loaders
        self._framework_loaders = {
            ModelFramework.PYTORCH: self._load_pytorch_model,
            ModelFramework.TENSORFLOW: self._load_tensorflow_model,
            ModelFramework.ONNX: self._load_onnx_model,
        }
    
    def _create_storage_structure(self):
        """Create necessary storage directories."""
        directories = ["metadata", "versions", "temp"]
        for directory in directories:
            (self.storage_root / directory).mkdir(parents=True, exist_ok=True)
    
    def _detect_framework(self, file_path: str) -> ModelFramework:
        """
        Detect model framework from file extension.
        
        Args:
            file_path: Path to model file
            
        Returns:
            Detected framework
        """
        extension = Path(file_path).suffix.lower()
        
        if extension in ['.pt', '.pth', '.pkl']:
            return ModelFramework.PYTORCH
        elif extension in ['.pb', '.h5', '.keras']:
            return ModelFramework.TENSORFLOW
        elif extension == '.onnx':
            return ModelFramework.ONNX
        else:
            # Default to PyTorch for unknown extensions
            return ModelFramework.PYTORCH
    
    def calculate_model_size(self, path: str) -> float:
        """
        Calculate model size in MB.
        
        Args:
            path: Path to model file or directory
            
        Returns:
            Size in MB
        """
        try:
            path_obj = Path(path)
            if not path_obj.exists():
                return 0.0
            
            if path_obj.is_file():
                return path_obj.stat().st_size / (1024 * 1024)
            elif path_obj.is_dir():
                total_size = sum(
                    f.stat().st_size 
                    for f in path_obj.rglob('*') 
                    if f.is_file()
                )
                return total_size / (1024 * 1024)
            else:
                return 0.0
        except Exception as e:
            logger.warning(f"Error calculating model size for {path}: {e}")
            return 0.0
    
    def _calculate_checksum(self, file_path: str) -> str:
        """Calculate SHA256 checksum of file."""
        sha256_hash = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(chunk)
            return sha256_hash.hexdigest()
        except Exception as e:
            logger.warning(f"Error calculating checksum for {file_path}: {e}")
            return ""
    
    def store_model(
        self, 
        model_path: str, 
        metadata: Optional[ModelMetadata] = None,
        overwrite: bool = False
    ) -> ModelMetadata:
        """
        Store a model with metadata.
        
        Args:
            model_path: Path to model file
            metadata: Model metadata (auto-generated if None)
            overwrite: Whether to overwrite existing model
            
        Returns:
            Stored model metadata
            
        Raises:
            ModelStoreError: If storage fails
        """
        if not os.path.exists(model_path):
            raise ModelStoreError(f"Model file {model_path} does not exist")
        
        # Validate model file
        validator = ModelFormatValidator()
        is_valid, errors = validator.validate_model_file(model_path)
        if not is_valid:
            raise ModelStoreError(f"Model validation failed: {', '.join(errors)}")
        
        # Generate metadata if not provided
        if metadata is None:
            metadata = ModelMetadata(
                name=Path(model_path).stem,
                framework=self._detect_framework(model_path)
            )
        
        # Check if model already exists
        if self.model_exists(metadata.id) and not overwrite:
            raise ModelStoreError(f"Model {metadata.id} already exists")
        
        # Calculate additional metadata
        metadata.size_mb = self.calculate_model_size(model_path)
        metadata.checksum = self._calculate_checksum(model_path)
        metadata.parameters = self._calculate_parameters(model_path, metadata.framework)
        
        # Copy model file to storage
        model_filename = f"{metadata.id}{Path(model_path).suffix}"
        stored_path = self.storage_root / "models" / model_filename
        stored_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(model_path, stored_path)
        
        metadata.file_path = str(stored_path)
        metadata.updated_at = datetime.now()
        
        # Store metadata
        self._store_metadata(metadata)
        
        # Create version if versioning enabled
        if self.enable_versioning:
            self._create_version(metadata)
        
        logger.info(f"Stored model {metadata.id} ({metadata.size_mb:.2f} MB)")
        return metadata
    
    def load_model(self, model_id: str) -> Tuple[Any, ModelMetadata]:
        """
        Load a model by ID.
        
        Args:
            model_id: Model identifier
            
        Returns:
            Tuple of (model_object, metadata)
            
        Raises:
            ModelStoreError: If loading fails
        """
        metadata = self.get_metadata(model_id)
        if metadata is None:
            raise ModelStoreError(f"Model {model_id} not found")
        
        if metadata.framework not in self._framework_loaders:
            raise ModelStoreError(f"Unsupported framework: {metadata.framework}")
        
        loader = self._framework_loaders[metadata.framework]
        model = loader(metadata.file_path)
        
        logger.info(f"Loaded model {model_id}")
        return model, metadata
    
    def get_metadata(self, model_id: str) -> Optional[ModelMetadata]:
        """
        Get model metadata by ID.
        
        Args:
            model_id: Model identifier
            
        Returns:
            Model metadata or None if not found
        """
        metadata_path = self.storage_root / "metadata" / f"{model_id}.json"
        if not metadata_path.exists():
            return None
        
        try:
            with open(metadata_path, 'r') as f:
                data = json.load(f)
            return ModelMetadata.from_dict(data)
        except Exception as e:
            logger.error(f"Error loading metadata for {model_id}: {e}")
            return None
    
    def list_models(self, model_type: Optional[ModelType] = None) -> List[ModelMetadata]:
        """
        List all stored models.
        
        Args:
            model_type: Filter by model type (optional)
            
        Returns:
            List of model metadata
        """
        models = []
        metadata_dir = self.storage_root / "metadata"
        
        if not metadata_dir.exists():
            return models
        
        for metadata_file in metadata_dir.glob("*.json"):
            try:
                with open(metadata_file, 'r') as f:
                    data = json.load(f)
                metadata = ModelMetadata.from_dict(data)
                
                if model_type is None or metadata.model_type == model_type:
                    models.append(metadata)
            except Exception as e:
                logger.warning(f"Error loading metadata from {metadata_file}: {e}")
        
        return sorted(models, key=lambda m: m.created_at, reverse=True)
    
    def delete_model(self, model_id: str) -> bool:
        """
        Delete a model and its metadata.
        
        Args:
            model_id: Model identifier
            
        Returns:
            True if deleted successfully, False otherwise
        """
        try:
            metadata = self.get_metadata(model_id)
            if metadata is None:
                return False
            
            # Delete model file
            if metadata.file_path and os.path.exists(metadata.file_path):
                os.remove(metadata.file_path)
            
            # Delete metadata
            metadata_path = self.storage_root / "metadata" / f"{model_id}.json"
            if metadata_path.exists():
                metadata_path.unlink()
            
            # Delete versions if versioning enabled
            if self.enable_versioning:
                versions_dir = self.storage_root / "versions" / model_id
                if versions_dir.exists():
                    shutil.rmtree(versions_dir)
            
            logger.info(f"Deleted model {model_id}")
            return True
        except Exception as e:
            logger.error(f"Error deleting model {model_id}: {e}")
            return False
    
    def model_exists(self, model_id: str) -> bool:
        """
        Check if a model exists.
        
        Args:
            model_id: Model identifier
            
        Returns:
            True if model exists, False otherwise
        """
        metadata_path = self.storage_root / "metadata" / f"{model_id}.json"
        return metadata_path.exists()
    
    def profile_model(self, model_id: str) -> PerformanceProfile:
        """
        Profile a model's performance characteristics.
        
        Args:
            model_id: Model identifier
            
        Returns:
            Performance profile
            
        Raises:
            ModelStoreError: If profiling fails
        """
        metadata = self.get_metadata(model_id)
        if metadata is None:
            raise ModelStoreError(f"Model {model_id} not found")
        
        # Basic profiling - in a real implementation this would be more sophisticated
        profile = PerformanceProfile(
            memory_usage_mb=metadata.size_mb,
            inference_time_ms=0.0,  # Would be measured during actual inference
            throughput_samples_per_sec=0.0,  # Would be measured during actual inference
            profiling_timestamp=datetime.now()
        )
        
        return profile
    
    def get_model_versions(self, model_id: str) -> List[str]:
        """
        Get all versions of a model.
        
        Args:
            model_id: Model identifier
            
        Returns:
            List of version strings, sorted in reverse order (latest first)
        """
        versions_dir = self.storage_root / "versions" / model_id
        if not versions_dir.exists():
            return []
        
        versions = []
        for version_dir in versions_dir.iterdir():
            if version_dir.is_dir():
                versions.append(version_dir.name)
        
        # Sort versions (simple string sort, could be improved with semantic versioning)
        return sorted(versions, reverse=True)
    
    def _store_metadata(self, metadata: ModelMetadata):
        """Store model metadata to disk."""
        metadata_path = self.storage_root / "metadata" / f"{metadata.id}.json"
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata.to_dict(), f, indent=2, default=str)
    
    def _create_version(self, metadata: ModelMetadata):
        """Create a version entry for the model."""
        version_dir = self.storage_root / "versions" / metadata.id / metadata.version
        version_dir.mkdir(parents=True, exist_ok=True)
        
        # Store version metadata
        with open(version_dir / "metadata.json", 'w') as f:
            json.dump(metadata.to_dict(), f, indent=2, default=str)
    
    def _calculate_parameters(self, model_path: str, framework: ModelFramework) -> int:
        """Calculate number of parameters in model."""
        if framework == ModelFramework.PYTORCH:
            return self._calculate_pytorch_parameters(model_path)
        elif framework == ModelFramework.TENSORFLOW:
            return self._calculate_tensorflow_parameters(model_path)
        elif framework == ModelFramework.ONNX:
            return self._calculate_onnx_parameters(model_path)
        else:
            return 0
    
    def _calculate_pytorch_parameters(self, model_path: str) -> int:
        """Calculate parameters for PyTorch model."""
        try:
            import torch
            state_dict = torch.load(model_path, map_location='cpu')
            if isinstance(state_dict, dict):
                return sum(param.numel() for param in state_dict.values() if hasattr(param, 'numel'))
            return 0
        except ImportError:
            logger.warning("PyTorch not available for parameter calculation")
            return 0
        except Exception as e:
            logger.warning(f"Error calculating PyTorch parameters: {e}")
            return 0
    
    def _calculate_tensorflow_parameters(self, model_path: str) -> int:
        """Calculate parameters for TensorFlow model."""
        try:
            import tensorflow as tf
            model = tf.keras.models.load_model(model_path)
            return model.count_params()
        except ImportError:
            logger.warning("TensorFlow not available for parameter calculation")
            return 0
        except Exception as e:
            logger.warning(f"Error calculating TensorFlow parameters: {e}")
            return 0
    
    def _calculate_onnx_parameters(self, model_path: str) -> int:
        """Calculate parameters for ONNX model."""
        try:
            import onnx
            model = onnx.load(model_path)
            total_params = 0
            for initializer in model.graph.initializer:
                param_count = reduce(lambda x, y: x * y, initializer.dims, 1)
                total_params += param_count
            return total_params
        except ImportError:
            logger.warning("ONNX not available for parameter calculation")
            return 0
        except Exception as e:
            logger.warning(f"Error calculating ONNX parameters: {e}")
            return 0
    
    def _load_pytorch_model(self, model_path: str):
        """Load PyTorch model."""
        import torch
        return torch.load(model_path, map_location='cpu')
    
    def _load_tensorflow_model(self, model_path: str):
        """Load TensorFlow model."""
        import tensorflow as tf
        return tf.keras.models.load_model(model_path)
    
    def _load_onnx_model(self, model_path: str):
        """Load ONNX model."""
        import onnx
        return onnx.load(model_path)


class ModelVersionManager:
    """
    Manages model versioning operations.
    """
    
    def __init__(self, model_store: ModelStore):
        """
        Initialize version manager.
        
        Args:
            model_store: ModelStore instance
        """
        self.model_store = model_store
    
    def create_version(
        self, 
        model_id: str, 
        version: str, 
        description: str = ""
    ) -> bool:
        """
        Create a new version of a model.
        
        Args:
            model_id: Model identifier
            version: Version string
            description: Version description
            
        Returns:
            True if version created successfully, False otherwise
        """
        try:
            metadata = self.model_store.get_metadata(model_id)
            if metadata is None:
                return False
            
            # Create version directory
            version_dir = self.model_store.storage_root / "versions" / model_id / version
            version_dir.mkdir(parents=True, exist_ok=True)
            
            # Update metadata with version info
            metadata.version = version
            metadata.description = description
            metadata.updated_at = datetime.now()
            
            # Store version metadata
            with open(version_dir / "metadata.json", 'w') as f:
                json.dump(metadata.to_dict(), f, indent=2, default=str)
            
            logger.info(f"Created version {version} for model {model_id}")
            return True
        except Exception as e:
            logger.error(f"Error creating version {version} for model {model_id}: {e}")
            return False
    
    def compare_versions(
        self, 
        model_id: str, 
        version1: str, 
        version2: str
    ) -> Dict[str, Any]:
        """
        Compare two versions of a model.
        
        Args:
            model_id: Model identifier
            version1: First version to compare
            version2: Second version to compare
            
        Returns:
            Comparison results
            
        Raises:
            ModelStoreError: If versions not found
        """
        # Load version metadata
        v1_path = self.model_store.storage_root / "versions" / model_id / version1 / "metadata.json"
        v2_path = self.model_store.storage_root / "versions" / model_id / version2 / "metadata.json"
        
        if not v1_path.exists() or not v2_path.exists():
            raise ModelStoreError(f"Version {version1} or {version2} not found for model {model_id}")
        
        with open(v1_path, 'r') as f:
            v1_data = json.load(f)
        with open(v2_path, 'r') as f:
            v2_data = json.load(f)
        
        return {
            'model_id': model_id,
            'version1': version1,
            'version2': version2,
            'size_difference_mb': v2_data.get('size_mb', 0) - v1_data.get('size_mb', 0),
            'parameter_difference': v2_data.get('parameters', 0) - v1_data.get('parameters', 0),
            'version1_metadata': v1_data,
            'version2_metadata': v2_data
        }
    
    def get_version_history(self, model_id: str) -> List[Dict[str, Any]]:
        """
        Get version history for a model.
        
        Args:
            model_id: Model identifier
            
        Returns:
            List of version history entries
        """
        versions_dir = self.model_store.storage_root / "versions" / model_id
        if not versions_dir.exists():
            return []
        
        history = []
        for version_dir in versions_dir.iterdir():
            if version_dir.is_dir():
                metadata_path = version_dir / "metadata.json"
                if metadata_path.exists():
                    try:
                        with open(metadata_path, 'r') as f:
                            data = json.load(f)
                        
                        history.append({
                            'version': data.get('version', version_dir.name),
                            'created_at': data.get('created_at'),
                            'updated_at': data.get('updated_at'),
                            'size_mb': data.get('size_mb', 0),
                            'parameters': data.get('parameters', 0),
                            'description': data.get('description', ''),
                            'checksum': data.get('checksum', '')
                        })
                    except Exception as e:
                        logger.warning(f"Error loading version metadata from {metadata_path}: {e}")
        
        # Sort by version (simple string sort)
        return sorted(history, key=lambda x: x['version'], reverse=True)


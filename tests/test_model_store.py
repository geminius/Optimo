"""
Unit tests for ModelStore and related functionality.
"""

import pytest
import tempfile
import os
import json
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open
from datetime import datetime

from src.models import (
    ModelStore, ModelStoreError, ModelVersionManager,
    ModelMetadata, ModelType, ModelFramework, PerformanceProfile
)


class TestModelStore:
    """Test ModelStore functionality."""
    
    @pytest.fixture
    def temp_storage(self):
        """Create temporary storage directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def model_store(self, temp_storage):
        """Create ModelStore instance with temporary storage."""
        return ModelStore(storage_root=temp_storage)
    
    @pytest.fixture
    def sample_model_file(self):
        """Create a sample model file."""
        temp_file = tempfile.NamedTemporaryFile(suffix=".pt", delete=False)
        temp_file.write(b"fake pytorch model content")
        temp_file.flush()
        yield temp_file.name
        os.unlink(temp_file.name)
    
    @pytest.fixture
    def sample_metadata(self):
        """Create sample model metadata."""
        return ModelMetadata(
            name="test_model",
            version="1.0.0",
            model_type=ModelType.OPENVLA,
            framework=ModelFramework.PYTORCH,
            description="Test model for unit tests",
            author="test_user",
            tags=["test", "robotics"]
        )
    
    def test_init_creates_storage_structure(self, temp_storage):
        """Test that ModelStore creates necessary directories."""
        store = ModelStore(storage_root=temp_storage)
        
        assert (Path(temp_storage) / "metadata").exists()
        assert (Path(temp_storage) / "versions").exists()
        assert (Path(temp_storage) / "temp").exists()
    
    def test_detect_framework_pytorch(self, model_store):
        """Test framework detection for PyTorch files."""
        assert model_store._detect_framework("model.pt") == ModelFramework.PYTORCH
        assert model_store._detect_framework("model.pth") == ModelFramework.PYTORCH
        assert model_store._detect_framework("model.pkl") == ModelFramework.PYTORCH
    
    def test_detect_framework_tensorflow(self, model_store):
        """Test framework detection for TensorFlow files."""
        assert model_store._detect_framework("model.pb") == ModelFramework.TENSORFLOW
        assert model_store._detect_framework("model.h5") == ModelFramework.TENSORFLOW
        assert model_store._detect_framework("model.keras") == ModelFramework.TENSORFLOW
    
    def test_detect_framework_onnx(self, model_store):
        """Test framework detection for ONNX files."""
        assert model_store._detect_framework("model.onnx") == ModelFramework.ONNX
    
    def test_detect_framework_unknown_defaults_pytorch(self, model_store):
        """Test that unknown extensions default to PyTorch."""
        assert model_store._detect_framework("model.unknown") == ModelFramework.PYTORCH
    
    def test_calculate_model_size_file(self, sample_model_file, model_store):
        """Test model size calculation for files."""
        size_mb = model_store.calculate_model_size(sample_model_file)
        
        expected_size = os.path.getsize(sample_model_file) / (1024 * 1024)
        assert abs(size_mb - expected_size) < 0.001
    
    def test_calculate_model_size_directory(self, temp_storage, model_store):
        """Test model size calculation for directories."""
        # Create a directory with some files
        test_dir = Path(temp_storage) / "test_model"
        test_dir.mkdir()
        
        (test_dir / "file1.txt").write_text("content1")
        (test_dir / "file2.txt").write_text("content2")
        
        size_mb = model_store.calculate_model_size(str(test_dir))
        
        # Should be sum of all files in directory
        assert size_mb > 0
    
    def test_calculate_model_size_nonexistent(self, model_store):
        """Test model size calculation for nonexistent path."""
        size_mb = model_store.calculate_model_size("nonexistent.pt")
        assert size_mb == 0.0
    
    def test_store_model_success(self, model_store, sample_model_file, sample_metadata):
        """Test successful model storage."""
        with patch('src.models.validation.ModelFormatValidator.validate_model_file') as mock_validate:
            mock_validate.return_value = (True, [])
            
            stored_metadata = model_store.store_model(sample_model_file, sample_metadata)
            
            assert stored_metadata.id == sample_metadata.id
            assert stored_metadata.name == sample_metadata.name
            assert stored_metadata.size_mb > 0
            assert stored_metadata.checksum is not None
            assert stored_metadata.file_path is not None
            
            # Check that model file was copied
            assert os.path.exists(stored_metadata.file_path)
            
            # Check that metadata was stored
            assert model_store.model_exists(stored_metadata.id)
    
    def test_store_model_nonexistent_file(self, model_store, sample_metadata):
        """Test storing nonexistent model file."""
        with pytest.raises(ModelStoreError, match="does not exist"):
            model_store.store_model("nonexistent.pt", sample_metadata)
    
    def test_store_model_validation_failure(self, model_store, sample_model_file, sample_metadata):
        """Test model storage with validation failure."""
        with patch('src.models.validation.ModelFormatValidator.validate_model_file') as mock_validate:
            mock_validate.return_value = (False, ["Validation error"])
            
            with pytest.raises(ModelStoreError, match="Model validation failed"):
                model_store.store_model(sample_model_file, sample_metadata)
    
    def test_store_model_duplicate_without_overwrite(self, model_store, sample_model_file, sample_metadata):
        """Test storing duplicate model without overwrite flag."""
        with patch('src.models.validation.ModelFormatValidator.validate_model_file') as mock_validate:
            mock_validate.return_value = (True, [])
            
            # Store model first time
            model_store.store_model(sample_model_file, sample_metadata)
            
            # Try to store again without overwrite
            with pytest.raises(ModelStoreError, match="already exists"):
                model_store.store_model(sample_model_file, sample_metadata)
    
    def test_store_model_duplicate_with_overwrite(self, model_store, sample_model_file, sample_metadata):
        """Test storing duplicate model with overwrite flag."""
        with patch('src.models.validation.ModelFormatValidator.validate_model_file') as mock_validate:
            mock_validate.return_value = (True, [])
            
            # Store model first time
            first_metadata = model_store.store_model(sample_model_file, sample_metadata)
            
            # Store again with overwrite
            second_metadata = model_store.store_model(sample_model_file, sample_metadata, overwrite=True)
            
            assert first_metadata.id == second_metadata.id
            assert second_metadata.updated_at >= first_metadata.updated_at
    
    def test_store_model_auto_metadata_creation(self, model_store, sample_model_file):
        """Test automatic metadata creation when none provided."""
        with patch('src.models.validation.ModelFormatValidator.validate_model_file') as mock_validate:
            mock_validate.return_value = (True, [])
            
            stored_metadata = model_store.store_model(sample_model_file)
            
            assert stored_metadata.name == Path(sample_model_file).stem
            assert stored_metadata.framework == ModelFramework.PYTORCH
            assert stored_metadata.size_mb > 0
    
    def test_load_model_success(self, model_store, sample_model_file, sample_metadata):
        """Test successful model loading."""
        with patch('src.models.validation.ModelFormatValidator.validate_model_file') as mock_validate:
            mock_validate.return_value = (True, [])
            
            # Store model first
            stored_metadata = model_store.store_model(sample_model_file, sample_metadata)
            
            # Mock PyTorch loading
            with patch('builtins.__import__') as mock_import:
                mock_torch = MagicMock()
                mock_model = MagicMock()
                mock_torch.load.return_value = mock_model
                
                def import_side_effect(name, *args, **kwargs):
                    if name == 'torch':
                        return mock_torch
                    return __import__(name, *args, **kwargs)
                
                mock_import.side_effect = import_side_effect
                
                loaded_model, loaded_metadata = model_store.load_model(stored_metadata.id)
                
                assert loaded_model == mock_model
                assert loaded_metadata.id == stored_metadata.id
    
    def test_load_model_nonexistent(self, model_store):
        """Test loading nonexistent model."""
        with pytest.raises(ModelStoreError, match="not found"):
            model_store.load_model("nonexistent_id")
    
    def test_load_model_unsupported_framework(self, model_store, sample_model_file):
        """Test loading model with unsupported framework."""
        # Create metadata with unsupported framework
        metadata = ModelMetadata(
            name="test_model",
            framework=ModelFramework.JAX  # Not supported in _framework_loaders
        )
        
        with patch('src.models.validation.ModelFormatValidator.validate_model_file') as mock_validate:
            mock_validate.return_value = (True, [])
            
            stored_metadata = model_store.store_model(sample_model_file, metadata)
            
            with pytest.raises(ModelStoreError, match="Unsupported framework"):
                model_store.load_model(stored_metadata.id)
    
    def test_get_metadata_success(self, model_store, sample_model_file, sample_metadata):
        """Test successful metadata retrieval."""
        with patch('src.models.validation.ModelFormatValidator.validate_model_file') as mock_validate:
            mock_validate.return_value = (True, [])
            
            stored_metadata = model_store.store_model(sample_model_file, sample_metadata)
            retrieved_metadata = model_store.get_metadata(stored_metadata.id)
            
            assert retrieved_metadata is not None
            assert retrieved_metadata.id == stored_metadata.id
            assert retrieved_metadata.name == stored_metadata.name
            assert retrieved_metadata.framework == stored_metadata.framework
    
    def test_get_metadata_nonexistent(self, model_store):
        """Test metadata retrieval for nonexistent model."""
        metadata = model_store.get_metadata("nonexistent_id")
        assert metadata is None
    
    def test_list_models_empty(self, model_store):
        """Test listing models when store is empty."""
        models = model_store.list_models()
        assert len(models) == 0
    
    def test_list_models_with_models(self, model_store, sample_model_file):
        """Test listing models with stored models."""
        with patch('src.models.validation.ModelFormatValidator.validate_model_file') as mock_validate:
            mock_validate.return_value = (True, [])
            
            # Store multiple models
            metadata1 = ModelMetadata(name="model1", model_type=ModelType.OPENVLA)
            metadata2 = ModelMetadata(name="model2", model_type=ModelType.RT1)
            
            model_store.store_model(sample_model_file, metadata1)
            model_store.store_model(sample_model_file, metadata2)
            
            # List all models
            all_models = model_store.list_models()
            assert len(all_models) == 2
            
            # List filtered by type
            openvla_models = model_store.list_models(ModelType.OPENVLA)
            assert len(openvla_models) == 1
            assert openvla_models[0].model_type == ModelType.OPENVLA
    
    def test_delete_model_success(self, model_store, sample_model_file, sample_metadata):
        """Test successful model deletion."""
        with patch('src.models.validation.ModelFormatValidator.validate_model_file') as mock_validate:
            mock_validate.return_value = (True, [])
            
            stored_metadata = model_store.store_model(sample_model_file, sample_metadata)
            
            # Verify model exists
            assert model_store.model_exists(stored_metadata.id)
            
            # Delete model
            success = model_store.delete_model(stored_metadata.id)
            assert success
            
            # Verify model no longer exists
            assert not model_store.model_exists(stored_metadata.id)
    
    def test_delete_model_nonexistent(self, model_store):
        """Test deleting nonexistent model."""
        success = model_store.delete_model("nonexistent_id")
        assert not success
    
    def test_model_exists(self, model_store, sample_model_file, sample_metadata):
        """Test model existence check."""
        with patch('src.models.validation.ModelFormatValidator.validate_model_file') as mock_validate:
            mock_validate.return_value = (True, [])
            
            # Model doesn't exist initially
            assert not model_store.model_exists(sample_metadata.id)
            
            # Store model
            stored_metadata = model_store.store_model(sample_model_file, sample_metadata)
            
            # Model exists now
            assert model_store.model_exists(stored_metadata.id)
    
    def test_profile_model_success(self, model_store, sample_model_file, sample_metadata):
        """Test successful model profiling."""
        with patch('src.models.validation.ModelFormatValidator.validate_model_file') as mock_validate:
            mock_validate.return_value = (True, [])
            
            stored_metadata = model_store.store_model(sample_model_file, sample_metadata)
            profile = model_store.profile_model(stored_metadata.id)
            
            assert isinstance(profile, PerformanceProfile)
            assert profile.memory_usage_mb > 0
            assert isinstance(profile.profiling_timestamp, datetime)
    
    def test_profile_model_nonexistent(self, model_store):
        """Test profiling nonexistent model."""
        with pytest.raises(ModelStoreError, match="not found"):
            model_store.profile_model("nonexistent_id")
    
    def test_calculate_pytorch_parameters(self, model_store):
        """Test PyTorch parameter calculation."""
        with patch('builtins.__import__') as mock_import:
            mock_torch = MagicMock()
            
            # Mock state dict with known parameters
            mock_state_dict = {
                'layer1.weight': MagicMock(),
                'layer1.bias': MagicMock(),
                'layer2.weight': MagicMock()
            }
            mock_state_dict['layer1.weight'].numel.return_value = 1000
            mock_state_dict['layer1.bias'].numel.return_value = 10
            mock_state_dict['layer2.weight'].numel.return_value = 500
            
            mock_torch.load.return_value = mock_state_dict
            
            def import_side_effect(name, *args, **kwargs):
                if name == 'torch':
                    return mock_torch
                return __import__(name, *args, **kwargs)
            
            mock_import.side_effect = import_side_effect
            
            params = model_store._calculate_pytorch_parameters("dummy_path")
            assert params == 1510  # 1000 + 10 + 500
    
    def test_calculate_pytorch_parameters_no_torch(self, model_store):
        """Test PyTorch parameter calculation when torch not available."""
        with patch('builtins.__import__') as mock_import:
            def import_side_effect(name, *args, **kwargs):
                if name == 'torch':
                    raise ImportError("No module named 'torch'")
                return __import__(name, *args, **kwargs)
            
            mock_import.side_effect = import_side_effect
            
            params = model_store._calculate_pytorch_parameters("dummy_path")
            assert params == 0
    
    def test_calculate_tensorflow_parameters(self, model_store):
        """Test TensorFlow parameter calculation."""
        with patch('builtins.__import__') as mock_import:
            mock_tf = MagicMock()
            mock_model = MagicMock()
            mock_model.count_params.return_value = 2000
            mock_tf.keras.models.load_model.return_value = mock_model
            
            def import_side_effect(name, *args, **kwargs):
                if name == 'tensorflow':
                    return mock_tf
                return __import__(name, *args, **kwargs)
            
            mock_import.side_effect = import_side_effect
            
            params = model_store._calculate_tensorflow_parameters("model.h5")
            assert params == 2000
    
    def test_calculate_tensorflow_parameters_no_tf(self, model_store):
        """Test TensorFlow parameter calculation when TF not available."""
        with patch('builtins.__import__') as mock_import:
            def import_side_effect(name, *args, **kwargs):
                if name == 'tensorflow':
                    raise ImportError("No module named 'tensorflow'")
                return __import__(name, *args, **kwargs)
            
            mock_import.side_effect = import_side_effect
            
            params = model_store._calculate_tensorflow_parameters("model.h5")
            assert params == 0
    
    def test_calculate_onnx_parameters(self, model_store):
        """Test ONNX parameter calculation."""
        with patch('builtins.__import__') as mock_import:
            mock_onnx = MagicMock()
            mock_model = MagicMock()
            
            # Mock initializers with known dimensions
            mock_init1 = MagicMock()
            mock_init1.dims = [10, 20]  # 200 parameters
            mock_init2 = MagicMock()
            mock_init2.dims = [5, 5, 3]  # 75 parameters
            
            mock_model.graph.initializer = [mock_init1, mock_init2]
            mock_onnx.load.return_value = mock_model
            
            def import_side_effect(name, *args, **kwargs):
                if name == 'onnx':
                    return mock_onnx
                return __import__(name, *args, **kwargs)
            
            mock_import.side_effect = import_side_effect
            
            params = model_store._calculate_onnx_parameters("model.onnx")
            assert params == 275  # 200 + 75
    
    def test_calculate_onnx_parameters_no_onnx(self, model_store):
        """Test ONNX parameter calculation when ONNX not available."""
        with patch('builtins.__import__') as mock_import:
            def import_side_effect(name, *args, **kwargs):
                if name == 'onnx':
                    raise ImportError("No module named 'onnx'")
                return __import__(name, *args, **kwargs)
            
            mock_import.side_effect = import_side_effect
            
            params = model_store._calculate_onnx_parameters("model.onnx")
            assert params == 0
    
    def test_versioning_enabled(self, temp_storage):
        """Test model store with versioning enabled."""
        store = ModelStore(storage_root=temp_storage, enable_versioning=True)
        assert store.enable_versioning
    
    def test_versioning_disabled(self, temp_storage):
        """Test model store with versioning disabled."""
        store = ModelStore(storage_root=temp_storage, enable_versioning=False)
        assert not store.enable_versioning
    
    def test_get_model_versions_empty(self, model_store):
        """Test getting versions for model with no versions."""
        versions = model_store.get_model_versions("nonexistent_id")
        assert len(versions) == 0
    
    def test_get_model_versions_with_versions(self, model_store, sample_model_file, sample_metadata, temp_storage):
        """Test getting versions for model with versions."""
        with patch('src.models.validation.ModelFormatValidator.validate_model_file') as mock_validate:
            mock_validate.return_value = (True, [])
            
            # Store model (creates version if versioning enabled)
            stored_metadata = model_store.store_model(sample_model_file, sample_metadata)
            
            # Manually create additional version directories for testing
            versions_dir = Path(temp_storage) / "versions" / stored_metadata.id
            versions_dir.mkdir(parents=True, exist_ok=True)
            (versions_dir / "1.0.0").mkdir(exist_ok=True)
            (versions_dir / "1.1.0").mkdir(exist_ok=True)
            (versions_dir / "2.0.0").mkdir(exist_ok=True)
            
            versions = model_store.get_model_versions(stored_metadata.id)
            
            # Should be sorted in reverse order (latest first)
            assert "2.0.0" in versions
            assert "1.1.0" in versions
            assert "1.0.0" in versions
            assert versions.index("2.0.0") < versions.index("1.1.0")


class TestModelVersionManager:
    """Test ModelVersionManager functionality."""
    
    @pytest.fixture
    def temp_storage(self):
        """Create temporary storage directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def model_store(self, temp_storage):
        """Create ModelStore instance with temporary storage."""
        return ModelStore(storage_root=temp_storage)
    
    @pytest.fixture
    def version_manager(self, model_store):
        """Create ModelVersionManager instance."""
        return ModelVersionManager(model_store)
    
    @pytest.fixture
    def sample_model_file(self):
        """Create a sample model file."""
        temp_file = tempfile.NamedTemporaryFile(suffix=".pt", delete=False)
        temp_file.write(b"fake pytorch model content")
        temp_file.flush()
        yield temp_file.name
        os.unlink(temp_file.name)
    
    @pytest.fixture
    def stored_model(self, model_store, sample_model_file):
        """Store a sample model for testing."""
        with patch('src.models.validation.ModelFormatValidator.validate_model_file') as mock_validate:
            mock_validate.return_value = (True, [])
            
            metadata = ModelMetadata(
                name="test_model",
                version="1.0.0",
                framework=ModelFramework.PYTORCH
            )
            
            return model_store.store_model(sample_model_file, metadata)
    
    def test_create_version_success(self, version_manager, stored_model):
        """Test successful version creation."""
        success = version_manager.create_version(
            stored_model.id, 
            "1.1.0", 
            "Updated version with improvements"
        )
        
        assert success
        
        # Check that version was created
        versions = version_manager.model_store.get_model_versions(stored_model.id)
        assert "1.1.0" in versions
    
    def test_create_version_nonexistent_model(self, version_manager):
        """Test creating version for nonexistent model."""
        success = version_manager.create_version("nonexistent_id", "1.0.0")
        assert not success
    
    def test_compare_versions_success(self, version_manager, stored_model, temp_storage):
        """Test successful version comparison."""
        # Create additional version manually for testing
        versions_dir = Path(temp_storage) / "versions" / stored_model.id
        versions_dir.mkdir(parents=True, exist_ok=True)
        
        # Create version 1.0.0
        v1_dir = versions_dir / "1.0.0"
        v1_dir.mkdir(exist_ok=True)
        metadata1 = stored_model
        metadata1.version = "1.0.0"
        metadata1.size_mb = 100.0
        metadata1.parameters = 1000
        
        with open(v1_dir / "metadata.json", 'w') as f:
            json.dump({
                'id': metadata1.id,
                'name': metadata1.name,
                'version': '1.0.0',
                'model_type': metadata1.model_type.value,
                'framework': metadata1.framework.value,
                'size_mb': 100.0,
                'parameters': 1000,
                'created_at': metadata1.created_at.isoformat(),
                'updated_at': metadata1.updated_at.isoformat(),
                'tags': metadata1.tags,
                'description': metadata1.description,
                'author': metadata1.author,
                'file_path': metadata1.file_path,
                'checksum': metadata1.checksum
            }, f)
        
        # Create version 1.1.0
        v2_dir = versions_dir / "1.1.0"
        v2_dir.mkdir(exist_ok=True)
        metadata2 = stored_model
        metadata2.version = "1.1.0"
        metadata2.size_mb = 95.0
        metadata2.parameters = 950
        
        with open(v2_dir / "metadata.json", 'w') as f:
            json.dump({
                'id': metadata2.id,
                'name': metadata2.name,
                'version': '1.1.0',
                'model_type': metadata2.model_type.value,
                'framework': metadata2.framework.value,
                'size_mb': 95.0,
                'parameters': 950,
                'created_at': metadata2.created_at.isoformat(),
                'updated_at': metadata2.updated_at.isoformat(),
                'tags': metadata2.tags,
                'description': metadata2.description,
                'author': metadata2.author,
                'file_path': metadata2.file_path,
                'checksum': metadata2.checksum
            }, f)
        
        comparison = version_manager.compare_versions(stored_model.id, "1.0.0", "1.1.0")
        
        assert comparison['model_id'] == stored_model.id
        assert comparison['version1'] == "1.0.0"
        assert comparison['version2'] == "1.1.0"
        assert comparison['size_difference_mb'] == -5.0  # 95 - 100
        assert comparison['parameter_difference'] == -50  # 950 - 1000
    
    def test_compare_versions_nonexistent(self, version_manager, stored_model):
        """Test comparing nonexistent versions."""
        with pytest.raises(ModelStoreError, match="not found"):
            version_manager.compare_versions(stored_model.id, "1.0.0", "2.0.0")
    
    def test_get_version_history_empty(self, version_manager):
        """Test getting version history for model with no versions."""
        history = version_manager.get_version_history("nonexistent_id")
        assert len(history) == 0
    
    def test_get_version_history_with_versions(self, version_manager, stored_model, temp_storage):
        """Test getting version history with versions."""
        # Create version directories manually for testing
        versions_dir = Path(temp_storage) / "versions" / stored_model.id
        versions_dir.mkdir(parents=True, exist_ok=True)
        
        # Create multiple versions
        for version in ["1.0.0", "1.1.0", "2.0.0"]:
            v_dir = versions_dir / version
            v_dir.mkdir(exist_ok=True)
            
            metadata = stored_model
            metadata.version = version
            
            with open(v_dir / "metadata.json", 'w') as f:
                json.dump({
                    'id': metadata.id,
                    'name': metadata.name,
                    'version': version,
                    'model_type': metadata.model_type.value,
                    'framework': metadata.framework.value,
                    'size_mb': metadata.size_mb,
                    'parameters': metadata.parameters,
                    'created_at': metadata.created_at.isoformat(),
                    'updated_at': metadata.updated_at.isoformat(),
                    'tags': metadata.tags,
                    'description': f"Version {version}",
                    'author': metadata.author,
                    'file_path': metadata.file_path,
                    'checksum': metadata.checksum
                }, f)
        
        history = version_manager.get_version_history(stored_model.id)
        
        assert len(history) == 3
        
        # Check that history contains expected versions
        versions_in_history = [item['version'] for item in history]
        assert "1.0.0" in versions_in_history
        assert "1.1.0" in versions_in_history
        assert "2.0.0" in versions_in_history
        
        # Check that each history item has expected fields
        for item in history:
            assert 'version' in item
            assert 'created_at' in item
            assert 'updated_at' in item
            assert 'size_mb' in item
            assert 'parameters' in item
            assert 'description' in item
            assert 'checksum' in item


if __name__ == "__main__":
    pytest.main([__file__])
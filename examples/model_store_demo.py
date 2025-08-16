#!/usr/bin/env python3
"""
Demo script showing ModelStore functionality.
"""

import os
import tempfile
import shutil
from pathlib import Path

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import (
    ModelStore, ModelMetadata, ModelType, ModelFramework, 
    ModelVersionManager
)


def create_dummy_model_file(filename: str, content: str = "dummy model content") -> str:
    """Create a dummy model file for demonstration."""
    temp_file = tempfile.NamedTemporaryFile(suffix=filename, delete=False)
    temp_file.write(content.encode())
    temp_file.flush()
    return temp_file.name


def main():
    """Demonstrate ModelStore functionality."""
    print("🤖 ModelStore Demo")
    print("=" * 50)
    
    # Create temporary storage directory
    storage_dir = tempfile.mkdtemp(prefix="model_store_demo_")
    print(f"📁 Using storage directory: {storage_dir}")
    
    try:
        # Initialize ModelStore
        store = ModelStore(storage_root=storage_dir, enable_versioning=True)
        print("✅ ModelStore initialized")
        
        # Create some dummy model files
        pytorch_model = create_dummy_model_file(".pt", "PyTorch model weights")
        tensorflow_model = create_dummy_model_file(".h5", "TensorFlow model data")
        onnx_model = create_dummy_model_file(".onnx", "ONNX model graph")
        
        print("\n📦 Storing models...")
        
        # Store PyTorch model (without validation for demo)
        pytorch_metadata = ModelMetadata(
            name="openvla_base",
            version="1.0.0",
            model_type=ModelType.OPENVLA,
            framework=ModelFramework.PYTORCH,
            description="Base OpenVLA model for robotics tasks",
            author="robotics_team",
            tags=["robotics", "vision", "language", "actions"]
        )
        
        # Mock validation to bypass framework requirements for demo
        from unittest.mock import patch
        with patch('src.models.validation.ModelFormatValidator.validate_model_file') as mock_validate:
            mock_validate.return_value = (True, [])
            stored_pytorch = store.store_model(pytorch_model, pytorch_metadata)
            print(f"✅ Stored PyTorch model: {stored_pytorch.name} ({stored_pytorch.size_mb:.2f} MB)")
        
        # Store TensorFlow model
        tf_metadata = ModelMetadata(
            name="rt1_policy",
            version="2.1.0",
            model_type=ModelType.RT1,
            framework=ModelFramework.TENSORFLOW,
            description="RT-1 robotics transformer policy",
            author="robotics_team",
            tags=["robotics", "transformer", "policy"]
        )
        
        with patch('src.models.validation.ModelFormatValidator.validate_model_file') as mock_validate:
            mock_validate.return_value = (True, [])
            stored_tf = store.store_model(tensorflow_model, tf_metadata)
            print(f"✅ Stored TensorFlow model: {stored_tf.name} ({stored_tf.size_mb:.2f} MB)")
        
        # Store ONNX model
        onnx_metadata = ModelMetadata(
            name="mobile_manipulator",
            version="1.5.0",
            model_type=ModelType.CUSTOM,
            framework=ModelFramework.ONNX,
            description="Optimized mobile manipulator model",
            author="optimization_team",
            tags=["robotics", "mobile", "manipulation", "optimized"]
        )
        
        with patch('src.models.validation.ModelFormatValidator.validate_model_file') as mock_validate:
            mock_validate.return_value = (True, [])
            stored_onnx = store.store_model(onnx_model, onnx_metadata)
            print(f"✅ Stored ONNX model: {stored_onnx.name} ({stored_onnx.size_mb:.2f} MB)")
        
        print("\n📋 Listing all models...")
        all_models = store.list_models()
        for model in all_models:
            print(f"  • {model.name} v{model.version} ({model.framework.value}) - {model.description}")
        
        print(f"\n📊 Total models stored: {len(all_models)}")
        
        # Demonstrate model loading
        print(f"\n🔄 Loading PyTorch model: {stored_pytorch.name}")
        try:
            loaded_model, loaded_metadata = store.load_model(stored_pytorch.id)
            print(f"✅ Successfully loaded model: {loaded_metadata.name}")
        except Exception as e:
            print(f"⚠️  Model loading failed (expected without PyTorch): {e}")
        
        # Demonstrate versioning
        print(f"\n🔄 Creating new version of {stored_pytorch.name}...")
        version_manager = ModelVersionManager(store)
        
        # Update model and create new version
        updated_metadata = stored_pytorch
        updated_metadata.version = "1.1.0"
        updated_metadata.description = "Updated OpenVLA model with performance improvements"
        
        success = version_manager.create_version(
            stored_pytorch.id, 
            "1.1.0", 
            "Performance improvements and bug fixes"
        )
        
        if success:
            print("✅ Created new version 1.1.0")
            
            # Show version history
            history = version_manager.get_version_history(stored_pytorch.id)
            print(f"📚 Version history for {stored_pytorch.name}:")
            for version_info in history:
                print(f"  • v{version_info['version']} - {version_info['description']}")
        
        # Demonstrate model profiling
        print(f"\n📈 Profiling model: {stored_pytorch.name}")
        try:
            profile = store.profile_model(stored_pytorch.id)
            print(f"  Memory usage: {profile.memory_usage_mb:.2f} MB")
            print(f"  Profiled at: {profile.profiling_timestamp}")
        except Exception as e:
            print(f"⚠️  Profiling failed: {e}")
        
        # Demonstrate filtering
        print(f"\n🔍 Filtering models by type...")
        openvla_models = store.list_models(ModelType.OPENVLA)
        rt1_models = store.list_models(ModelType.RT1)
        custom_models = store.list_models(ModelType.CUSTOM)
        
        print(f"  OpenVLA models: {len(openvla_models)}")
        print(f"  RT-1 models: {len(rt1_models)}")
        print(f"  Custom models: {len(custom_models)}")
        
        # Demonstrate model existence check
        print(f"\n🔍 Checking model existence...")
        print(f"  {stored_pytorch.name} exists: {store.model_exists(stored_pytorch.id)}")
        print(f"  Non-existent model exists: {store.model_exists('fake-id-123')}")
        
        # Demonstrate metadata retrieval
        print(f"\n📄 Retrieving metadata for {stored_tf.name}...")
        metadata = store.get_metadata(stored_tf.id)
        if metadata:
            print(f"  Name: {metadata.name}")
            print(f"  Framework: {metadata.framework.value}")
            print(f"  Size: {metadata.size_mb:.2f} MB")
            print(f"  Tags: {', '.join(metadata.tags)}")
            print(f"  Created: {metadata.created_at.strftime('%Y-%m-%d %H:%M:%S')}")
        
        print(f"\n🗑️  Demonstrating model deletion...")
        # Delete one model
        success = store.delete_model(stored_onnx.id)
        if success:
            print(f"✅ Deleted model: {stored_onnx.name}")
            remaining_models = store.list_models()
            print(f"📊 Remaining models: {len(remaining_models)}")
        
        print(f"\n✨ Demo completed successfully!")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        print(f"\n🧹 Cleaning up temporary files...")
        try:
            os.unlink(pytorch_model)
            os.unlink(tensorflow_model)
            os.unlink(onnx_model)
            shutil.rmtree(storage_dir)
            print("✅ Cleanup completed")
        except Exception as e:
            print(f"⚠️  Cleanup warning: {e}")


if __name__ == "__main__":
    main()
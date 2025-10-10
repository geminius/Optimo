"""
Quick test script for the Analysis Agent to debug shape mismatch issues.
"""

import torch
import torch.nn as nn
import logging
from pathlib import Path


class RoboticsVLAModel(nn.Module):
    """Vision-Language-Action model for robotics tasks."""
    
    def __init__(self, input_dim=1280, hidden_dim=256, action_dim=7):
        super().__init__()
        
        # Input projection (simulates vision+language fusion)
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        # Feature encoder
        self.encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        # Cross-attention fusion
        self.cross_attention = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)
        
        # Action decoder
        self.action_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
    
    def forward(self, x):
        """Forward pass with single concatenated input."""
        # Project input
        features = self.input_projection(x)
        
        # Encode features
        encoded = self.encoder(features)
        
        # Apply self-attention
        attended, _ = self.cross_attention(
            encoded.unsqueeze(1),
            encoded.unsqueeze(1),
            encoded.unsqueeze(1)
        )
        
        # Decode to actions
        actions = self.action_decoder(attended.squeeze(1))
        
        return actions

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def inspect_model(model_path: str):
    """Inspect the model to understand its structure and input requirements."""
    print(f"\n{'='*60}")
    print(f"Inspecting model: {model_path}")
    print(f"{'='*60}\n")
    
    try:
        # Load the model
        model = torch.load(model_path, map_location='cpu', weights_only=False)
        
        # Check what was loaded
        print(f"Type of loaded object: {type(model)}")
        
        if isinstance(model, dict):
            print(f"Dictionary keys: {model.keys()}")
            if 'model' in model:
                model = model['model']
                print(f"Extracted 'model' key, type: {type(model)}")
        
        if not isinstance(model, torch.nn.Module):
            print(f"❌ Not a torch.nn.Module, cannot analyze")
            return None
        
        # Print model structure
        print(f"\n📊 Model Structure:")
        print(model)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        print(f"\n📈 Total parameters: {total_params:,}")
        
        # Analyze first layer to infer input shape
        print(f"\n🔍 Analyzing first layer:")
        first_layer = None
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # Leaf module
                first_layer = module
                print(f"  First leaf layer: {name} ({type(module).__name__})")
                
                if hasattr(module, 'in_features'):
                    print(f"  ✓ in_features: {module.in_features}")
                if hasattr(module, 'out_features'):
                    print(f"  ✓ out_features: {module.out_features}")
                if hasattr(module, 'in_channels'):
                    print(f"  ✓ in_channels: {module.in_channels}")
                if hasattr(module, 'out_channels'):
                    print(f"  ✓ out_channels: {module.out_channels}")
                if hasattr(module, 'weight'):
                    print(f"  ✓ weight shape: {module.weight.shape}")
                break
        
        # Try to find input shape by testing
        print(f"\n🧪 Testing input shapes:")
        test_shapes = [
            (1, 1280),         # Based on error message
            (1, 256),          # Based on error message
            (1, 3, 224, 224),  # Standard image
            (1, 512),
            (1, 1024),
            (1, 2048),
        ]
        
        model.eval()
        working_shape = None
        
        for shape in test_shapes:
            try:
                dummy_input = torch.randn(shape)
                with torch.no_grad():
                    output = model(dummy_input)
                print(f"  ✅ Shape {shape} works! Output shape: {output.shape}")
                working_shape = shape
                break
            except Exception as e:
                print(f"  ❌ Shape {shape} failed: {str(e)[:80]}")
        
        if working_shape:
            print(f"\n✅ Found working input shape: {working_shape}")
        else:
            print(f"\n❌ No working input shape found")
        
        return model, working_shape
        
    except Exception as e:
        print(f"❌ Error inspecting model: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_analysis_agent(model_path: str):
    """Test the analysis agent with the model."""
    print(f"\n{'='*60}")
    print(f"Testing Analysis Agent")
    print(f"{'='*60}\n")
    
    try:
        from src.agents.analysis.agent import AnalysisAgent
        
        config = {
            "profiling_samples": 10,  # Reduced for faster testing
            "warmup_samples": 2
        }
        
        agent = AnalysisAgent(config)
        
        if not agent.initialize():
            print("❌ Failed to initialize agent")
            return
        
        print("✅ Agent initialized")
        print(f"Running analysis on: {model_path}")
        
        report = agent.analyze_model(model_path)
        
        print(f"\n✅ Analysis completed!")
        print(f"\n📊 Analysis Report:")
        print(f"  Model ID: {report.model_id}")
        print(f"  Total layers: {report.architecture_summary.total_layers}")
        print(f"  Total parameters: {report.architecture_summary.total_parameters:,}")
        print(f"  Inference time: {report.performance_profile.inference_time_ms:.2f} ms")
        print(f"  Memory usage: {report.performance_profile.memory_usage_mb:.2f} MB")
        print(f"  Analysis duration: {report.analysis_duration_seconds:.2f} seconds")
        print(f"\n🎯 Optimization Opportunities: {len(report.optimization_opportunities)}")
        for opp in report.optimization_opportunities:
            print(f"  - {opp.technique}: {opp.description}")
        
        agent.cleanup()
        
    except Exception as e:
        print(f"❌ Analysis agent test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    model_path = "test_models/robotics_vla_demo.pt"
    
    # First inspect the model
    result = inspect_model(model_path)
    
    # Then test the analysis agent
    if result:
        test_analysis_agent(model_path)

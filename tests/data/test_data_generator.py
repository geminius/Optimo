"""
Test data generator for various model types and optimization scenarios.
Creates synthetic models, datasets, and test cases for comprehensive testing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import json
import tempfile
from pathlib import Path
from dataclasses import dataclass, asdict
import random
from enum import Enum

from src.models.core import ModelMetadata, OptimizationSession
from src.config.optimization_criteria import OptimizationCriteria


class ModelType(Enum):
    """Supported model types for test generation."""
    CNN = "cnn"
    TRANSFORMER = "transformer"
    RESNET = "resnet"
    LSTM = "lstm"
    MLP = "mlp"
    ROBOTICS_VLA = "robotics_vla"


class OptimizationTechnique(Enum):
    """Optimization techniques for test scenarios."""
    QUANTIZATION = "quantization"
    PRUNING = "pruning"
    DISTILLATION = "distillation"
    ARCHITECTURE_SEARCH = "architecture_search"


@dataclass
class TestScenario:
    """Test scenario configuration."""
    name: str
    model_type: ModelType
    model_size: str  # "small", "medium", "large"
    optimization_techniques: List[OptimizationTechnique]
    expected_speedup: float
    max_accuracy_loss: float
    complexity_level: str  # "simple", "moderate", "complex"
    should_succeed: bool
    description: str


class SyntheticModelGenerator:
    """Generator for synthetic models of various types and sizes."""
    
    @staticmethod
    def create_cnn_model(size: str = "medium") -> nn.Module:
        """Create CNN model of specified size."""
        size_configs = {
            "small": {"channels": [16, 32], "fc_size": 64},
            "medium": {"channels": [32, 64, 128], "fc_size": 256},
            "large": {"channels": [64, 128, 256, 512], "fc_size": 1024}
        }
        
        config = size_configs.get(size, size_configs["medium"])
        
        class SyntheticCNN(nn.Module):
            def __init__(self, channels, fc_size):
                super().__init__()
                
                # Convolutional layers
                layers = []
                in_channels = 3
                for out_channels in channels:
                    layers.extend([
                        nn.Conv2d(in_channels, out_channels, 3, padding=1),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU(),
                        nn.MaxPool2d(2)
                    ])
                    in_channels = out_channels
                
                self.features = nn.Sequential(*layers)
                
                # Calculate feature size after convolutions
                # Assuming input size 224x224, after each maxpool it's halved
                feature_size = channels[-1] * (224 // (2 ** len(channels))) ** 2
                
                # Classifier
                self.classifier = nn.Sequential(
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(),
                    nn.Linear(channels[-1], fc_size),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    nn.Linear(fc_size, 1000)
                )
            
            def forward(self, x):
                x = self.features(x)
                x = self.classifier(x)
                return x
        
        return SyntheticCNN(config["channels"], config["fc_size"])
    
    @staticmethod
    def create_transformer_model(size: str = "medium") -> nn.Module:
        """Create transformer model of specified size."""
        size_configs = {
            "small": {"d_model": 128, "nhead": 4, "num_layers": 2, "vocab_size": 1000},
            "medium": {"d_model": 256, "nhead": 8, "num_layers": 6, "vocab_size": 5000},
            "large": {"d_model": 512, "nhead": 16, "num_layers": 12, "vocab_size": 10000}
        }
        
        config = size_configs.get(size, size_configs["medium"])
        
        class SyntheticTransformer(nn.Module):
            def __init__(self, vocab_size, d_model, nhead, num_layers):
                super().__init__()
                self.d_model = d_model
                self.embedding = nn.Embedding(vocab_size, d_model)
                self.pos_encoding = nn.Parameter(torch.randn(1000, d_model))
                
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model, nhead, dim_feedforward=d_model * 4, batch_first=True
                )
                self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
                self.output_projection = nn.Linear(d_model, vocab_size)
            
            def forward(self, x):
                seq_len = x.size(1)
                x = self.embedding(x) * np.sqrt(self.d_model)
                x = x + self.pos_encoding[:seq_len]
                x = self.transformer(x)
                x = self.output_projection(x)
                return x
        
        return SyntheticTransformer(**config)
    
    @staticmethod
    def create_resnet_model(size: str = "medium") -> nn.Module:
        """Create ResNet-style model of specified size."""
        size_configs = {
            "small": {"layers": [2, 2, 2, 2], "channels": [64, 128, 256, 512]},
            "medium": {"layers": [3, 4, 6, 3], "channels": [64, 128, 256, 512]},
            "large": {"layers": [3, 4, 23, 3], "channels": [64, 128, 256, 512]}
        }
        
        config = size_configs.get(size, size_configs["medium"])
        
        class BasicBlock(nn.Module):
            def __init__(self, in_channels, out_channels, stride=1):
                super().__init__()
                self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
                self.bn1 = nn.BatchNorm2d(out_channels)
                self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
                self.bn2 = nn.BatchNorm2d(out_channels)
                
                self.shortcut = nn.Sequential()
                if stride != 1 or in_channels != out_channels:
                    self.shortcut = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                        nn.BatchNorm2d(out_channels)
                    )
            
            def forward(self, x):
                out = F.relu(self.bn1(self.conv1(x)))
                out = self.bn2(self.conv2(out))
                out += self.shortcut(x)
                out = F.relu(out)
                return out
        
        class SyntheticResNet(nn.Module):
            def __init__(self, layers, channels):
                super().__init__()
                self.in_channels = channels[0]
                
                self.conv1 = nn.Conv2d(3, channels[0], 7, 2, 3, bias=False)
                self.bn1 = nn.BatchNorm2d(channels[0])
                self.maxpool = nn.MaxPool2d(3, 2, 1)
                
                self.layer1 = self._make_layer(BasicBlock, channels[0], layers[0], 1)
                self.layer2 = self._make_layer(BasicBlock, channels[1], layers[1], 2)
                self.layer3 = self._make_layer(BasicBlock, channels[2], layers[2], 2)
                self.layer4 = self._make_layer(BasicBlock, channels[3], layers[3], 2)
                
                self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
                self.fc = nn.Linear(channels[3], 1000)
            
            def _make_layer(self, block, out_channels, num_blocks, stride):
                strides = [stride] + [1] * (num_blocks - 1)
                layers = []
                for stride in strides:
                    layers.append(block(self.in_channels, out_channels, stride))
                    self.in_channels = out_channels
                return nn.Sequential(*layers)
            
            def forward(self, x):
                x = self.maxpool(F.relu(self.bn1(self.conv1(x))))
                x = self.layer1(x)
                x = self.layer2(x)
                x = self.layer3(x)
                x = self.layer4(x)
                x = self.avgpool(x)
                x = torch.flatten(x, 1)
                x = self.fc(x)
                return x
        
        return SyntheticResNet(config["layers"], config["channels"])
    
    @staticmethod
    def create_lstm_model(size: str = "medium") -> nn.Module:
        """Create LSTM model of specified size."""
        size_configs = {
            "small": {"vocab_size": 1000, "embed_size": 128, "hidden_size": 256, "num_layers": 2},
            "medium": {"vocab_size": 5000, "embed_size": 256, "hidden_size": 512, "num_layers": 3},
            "large": {"vocab_size": 10000, "embed_size": 512, "hidden_size": 1024, "num_layers": 4}
        }
        
        config = size_configs.get(size, size_configs["medium"])
        
        class SyntheticLSTM(nn.Module):
            def __init__(self, vocab_size, embed_size, hidden_size, num_layers):
                super().__init__()
                self.embedding = nn.Embedding(vocab_size, embed_size)
                self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
                self.fc = nn.Linear(hidden_size, vocab_size)
                self.dropout = nn.Dropout(0.3)
            
            def forward(self, x):
                x = self.embedding(x)
                x, _ = self.lstm(x)
                x = self.dropout(x)
                x = self.fc(x)
                return x
        
        return SyntheticLSTM(**config)
    
    @staticmethod
    def create_mlp_model(size: str = "medium") -> nn.Module:
        """Create MLP model of specified size."""
        size_configs = {
            "small": {"layers": [784, 256, 128, 10]},
            "medium": {"layers": [784, 512, 256, 128, 10]},
            "large": {"layers": [784, 1024, 512, 256, 128, 10]}
        }
        
        config = size_configs.get(size, size_configs["medium"])
        
        class SyntheticMLP(nn.Module):
            def __init__(self, layers):
                super().__init__()
                
                network_layers = []
                for i in range(len(layers) - 1):
                    network_layers.append(nn.Linear(layers[i], layers[i + 1]))
                    if i < len(layers) - 2:  # No activation after last layer
                        network_layers.append(nn.ReLU())
                        network_layers.append(nn.Dropout(0.2))
                
                self.network = nn.Sequential(*network_layers)
            
            def forward(self, x):
                return self.network(x)
        
        return SyntheticMLP(config["layers"])
    
    @staticmethod
    def create_robotics_vla_model(size: str = "medium") -> nn.Module:
        """Create robotics VLA-style model of specified size."""
        size_configs = {
            "small": {"vision_dim": 256, "text_dim": 128, "action_dim": 64, "num_layers": 4},
            "medium": {"vision_dim": 512, "text_dim": 256, "action_dim": 128, "num_layers": 6},
            "large": {"vision_dim": 1024, "text_dim": 512, "action_dim": 256, "num_layers": 8}
        }
        
        config = size_configs.get(size, size_configs["medium"])
        
        class SyntheticRoboticsVLA(nn.Module):
            def __init__(self, vision_dim, text_dim, action_dim, num_layers):
                super().__init__()
                
                # Vision encoder (simplified)
                self.vision_encoder = nn.Sequential(
                    nn.Conv2d(3, 64, 7, 2, 3),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d((8, 8)),
                    nn.Flatten(),
                    nn.Linear(64 * 64, vision_dim)
                )
                
                # Text encoder
                self.text_encoder = nn.Sequential(
                    nn.Embedding(1000, text_dim // 2),
                    nn.LSTM(text_dim // 2, text_dim, batch_first=True)
                )
                
                # Fusion and action prediction
                fusion_dim = vision_dim + text_dim
                layers = []
                current_dim = fusion_dim
                
                for i in range(num_layers):
                    next_dim = max(action_dim, current_dim // 2)
                    layers.extend([
                        nn.Linear(current_dim, next_dim),
                        nn.ReLU(),
                        nn.Dropout(0.1)
                    ])
                    current_dim = next_dim
                
                layers.append(nn.Linear(current_dim, action_dim))
                self.action_head = nn.Sequential(*layers)
            
            def forward(self, vision_input, text_input):
                # Encode vision
                vision_features = self.vision_encoder(vision_input)
                
                # Encode text
                text_embedded = self.text_encoder[0](text_input)
                text_features, _ = self.text_encoder[1](text_embedded)
                text_features = text_features[:, -1, :]  # Use last hidden state
                
                # Fuse and predict actions
                fused_features = torch.cat([vision_features, text_features], dim=1)
                actions = self.action_head(fused_features)
                
                return actions
        
        return SyntheticRoboticsVLA(**config)


class TestDataGenerator:
    """Generator for comprehensive test data and scenarios."""
    
    def __init__(self):
        self.model_generator = SyntheticModelGenerator()
        self.test_scenarios = self._create_test_scenarios()
    
    def _create_test_scenarios(self) -> List[TestScenario]:
        """Create comprehensive test scenarios."""
        scenarios = []
        
        # Basic optimization scenarios
        scenarios.extend([
            TestScenario(
                name="cnn_quantization_simple",
                model_type=ModelType.CNN,
                model_size="small",
                optimization_techniques=[OptimizationTechnique.QUANTIZATION],
                expected_speedup=1.5,
                max_accuracy_loss=0.05,
                complexity_level="simple",
                should_succeed=True,
                description="Simple CNN quantization with conservative settings"
            ),
            TestScenario(
                name="transformer_pruning_moderate",
                model_type=ModelType.TRANSFORMER,
                model_size="medium",
                optimization_techniques=[OptimizationTechnique.PRUNING],
                expected_speedup=2.0,
                max_accuracy_loss=0.03,
                complexity_level="moderate",
                should_succeed=True,
                description="Transformer pruning with moderate sparsity"
            ),
            TestScenario(
                name="resnet_multi_optimization",
                model_type=ModelType.RESNET,
                model_size="large",
                optimization_techniques=[
                    OptimizationTechnique.QUANTIZATION,
                    OptimizationTechnique.PRUNING
                ],
                expected_speedup=3.0,
                max_accuracy_loss=0.08,
                complexity_level="complex",
                should_succeed=True,
                description="Large ResNet with multiple optimization techniques"
            )
        ])
        
        # Edge case scenarios
        scenarios.extend([
            TestScenario(
                name="tiny_model_aggressive_optimization",
                model_type=ModelType.MLP,
                model_size="small",
                optimization_techniques=[
                    OptimizationTechnique.QUANTIZATION,
                    OptimizationTechnique.PRUNING
                ],
                expected_speedup=5.0,
                max_accuracy_loss=0.01,
                complexity_level="complex",
                should_succeed=False,
                description="Overly aggressive optimization on tiny model (should fail)"
            ),
            TestScenario(
                name="lstm_impossible_constraints",
                model_type=ModelType.LSTM,
                model_size="large",
                optimization_techniques=[OptimizationTechnique.QUANTIZATION],
                expected_speedup=10.0,
                max_accuracy_loss=0.001,
                complexity_level="complex",
                should_succeed=False,
                description="LSTM with impossible performance constraints"
            )
        ])
        
        # Robotics-specific scenarios
        scenarios.extend([
            TestScenario(
                name="robotics_vla_conservative",
                model_type=ModelType.ROBOTICS_VLA,
                model_size="medium",
                optimization_techniques=[OptimizationTechnique.QUANTIZATION],
                expected_speedup=1.3,
                max_accuracy_loss=0.02,
                complexity_level="moderate",
                should_succeed=True,
                description="Conservative optimization for robotics VLA model"
            ),
            TestScenario(
                name="robotics_vla_aggressive",
                model_type=ModelType.ROBOTICS_VLA,
                model_size="large",
                optimization_techniques=[
                    OptimizationTechnique.QUANTIZATION,
                    OptimizationTechnique.PRUNING
                ],
                expected_speedup=2.5,
                max_accuracy_loss=0.05,
                complexity_level="complex",
                should_succeed=True,
                description="Aggressive optimization for large robotics model"
            )
        ])
        
        return scenarios
    
    def generate_model(self, model_type: ModelType, size: str = "medium") -> nn.Module:
        """Generate a model of specified type and size."""
        generators = {
            ModelType.CNN: self.model_generator.create_cnn_model,
            ModelType.TRANSFORMER: self.model_generator.create_transformer_model,
            ModelType.RESNET: self.model_generator.create_resnet_model,
            ModelType.LSTM: self.model_generator.create_lstm_model,
            ModelType.MLP: self.model_generator.create_mlp_model,
            ModelType.ROBOTICS_VLA: self.model_generator.create_robotics_vla_model
        }
        
        generator = generators.get(model_type)
        if not generator:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        return generator(size)
    
    def generate_test_input(self, model_type: ModelType, batch_size: int = 1) -> torch.Tensor:
        """Generate appropriate test input for model type."""
        input_configs = {
            ModelType.CNN: lambda: torch.randn(batch_size, 3, 224, 224),
            ModelType.TRANSFORMER: lambda: torch.randint(0, 1000, (batch_size, 50)),
            ModelType.RESNET: lambda: torch.randn(batch_size, 3, 224, 224),
            ModelType.LSTM: lambda: torch.randint(0, 1000, (batch_size, 20)),
            ModelType.MLP: lambda: torch.randn(batch_size, 784),
            ModelType.ROBOTICS_VLA: lambda: (
                torch.randn(batch_size, 3, 224, 224),  # Vision input
                torch.randint(0, 1000, (batch_size, 10))  # Text input
            )
        }
        
        generator = input_configs.get(model_type)
        if not generator:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        return generator()
    
    def create_optimization_criteria(self, scenario: TestScenario) -> OptimizationCriteria:
        """Create optimization criteria from test scenario."""
        from src.config.optimization_criteria import OptimizationConstraints, PerformanceThreshold, PerformanceMetric, OptimizationTechnique as ConfigOptimizationTechnique
        
        # Create performance thresholds
        thresholds = [
            PerformanceThreshold(
                metric=PerformanceMetric.ACCURACY,
                min_value=1.0 - scenario.max_accuracy_loss,
                tolerance=0.01
            )
        ]
        
        # Create constraints
        constraints = OptimizationConstraints(
            max_optimization_time_minutes=30,
            preserve_accuracy_threshold=1.0 - scenario.max_accuracy_loss,
            allowed_techniques=[ConfigOptimizationTechnique(tech.value) for tech in scenario.optimization_techniques]
        )
        
        return OptimizationCriteria(
            name=f"criteria_{scenario.name}",
            description=f"Optimization criteria for {scenario.description}",
            performance_thresholds=thresholds,
            constraints=constraints,
            target_deployment="general"
        )
    
    def generate_model_metadata(self, model: nn.Module, model_type: ModelType, 
                              size: str) -> ModelMetadata:
        """Generate metadata for a model."""
        # Calculate model size
        param_size = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
        size_mb = (param_size + buffer_size) / (1024 * 1024)
        
        # Count parameters
        num_params = sum(p.numel() for p in model.parameters())
        
        return ModelMetadata(
            id=f"test_{model_type.value}_{size}_{random.randint(1000, 9999)}",
            name=f"Test {model_type.value.upper()} ({size})",
            version="1.0.0",
            model_type=model_type.value,
            framework="pytorch",
            size_mb=size_mb,
            parameters=num_params,
            tags=[model_type.value, size, "synthetic", "test"]
        )
    
    def save_test_model(self, model: nn.Module, filepath: Path) -> Path:
        """Save model to file and return path."""
        torch.save(model.state_dict(), filepath)
        return filepath
    
    def generate_test_dataset(self, scenario: TestScenario, 
                            num_samples: int = 100) -> Dict[str, Any]:
        """Generate test dataset for a scenario."""
        model = self.generate_model(scenario.model_type, scenario.model_size)
        
        # Generate input samples
        inputs = []
        for _ in range(num_samples):
            input_data = self.generate_test_input(scenario.model_type)
            inputs.append(input_data)
        
        # Generate expected outputs (using the model itself)
        model.eval()
        outputs = []
        with torch.no_grad():
            for input_data in inputs:
                if scenario.model_type == ModelType.ROBOTICS_VLA:
                    output = model(*input_data)
                else:
                    output = model(input_data)
                outputs.append(output)
        
        return {
            'model': model,
            'inputs': inputs,
            'outputs': outputs,
            'metadata': self.generate_model_metadata(model, scenario.model_type, scenario.model_size),
            'criteria': self.create_optimization_criteria(scenario)
        }
    
    def create_test_suite(self, output_dir: Path) -> Dict[str, Any]:
        """Create complete test suite with all scenarios."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        test_suite = {
            'scenarios': [],
            'models': {},
            'metadata': {
                'total_scenarios': len(self.test_scenarios),
                'model_types': list(set(s.model_type.value for s in self.test_scenarios)),
                'optimization_techniques': list(set(
                    tech.value for s in self.test_scenarios for tech in s.optimization_techniques
                ))
            }
        }
        
        for scenario in self.test_scenarios:
            print(f"Generating test data for scenario: {scenario.name}")
            
            # Generate test dataset
            dataset = self.generate_test_dataset(scenario)
            
            # Save model
            model_path = output_dir / f"{scenario.name}_model.pth"
            self.save_test_model(dataset['model'], model_path)
            
            # Save test data
            # Convert scenario to dict with enum values as strings
            scenario_dict = asdict(scenario)
            scenario_dict['model_type'] = scenario.model_type.value
            scenario_dict['optimization_techniques'] = [tech.value for tech in scenario.optimization_techniques]
            
            # Convert metadata to dict with datetime as string
            metadata_dict = asdict(dataset['metadata'])
            if 'created_at' in metadata_dict:
                metadata_dict['created_at'] = metadata_dict['created_at'].isoformat()
            if 'updated_at' in metadata_dict:
                metadata_dict['updated_at'] = metadata_dict['updated_at'].isoformat()
            
            test_data = {
                'scenario': scenario_dict,
                'model_path': str(model_path),
                'metadata': metadata_dict,
                'criteria': {
                    'name': dataset['criteria'].name,
                    'description': dataset['criteria'].description,
                    'target_deployment': dataset['criteria'].target_deployment
                },
                'num_test_samples': len(dataset['inputs'])
            }
            
            test_suite['scenarios'].append(test_data)
            test_suite['models'][scenario.name] = str(model_path)
        
        # Save test suite configuration
        suite_config_path = output_dir / "test_suite.json"
        with open(suite_config_path, 'w') as f:
            json.dump(test_suite, f, indent=2)
        
        return test_suite
    
    def get_scenario_by_name(self, name: str) -> Optional[TestScenario]:
        """Get test scenario by name."""
        for scenario in self.test_scenarios:
            if scenario.name == name:
                return scenario
        return None
    
    def get_scenarios_by_complexity(self, complexity: str) -> List[TestScenario]:
        """Get scenarios by complexity level."""
        return [s for s in self.test_scenarios if s.complexity_level == complexity]
    
    def get_scenarios_by_model_type(self, model_type: ModelType) -> List[TestScenario]:
        """Get scenarios by model type."""
        return [s for s in self.test_scenarios if s.model_type == model_type]
#!/usr/bin/env python3
"""
Demo script for the Analysis Agent.

This script demonstrates how to use the AnalysisAgent to analyze a model
and identify optimization opportunities.
"""

import torch
import torch.nn as nn
import tempfile
import os
from pathlib import Path

# Add the src directory to the path so we can import our modules
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.agents.analysis.agent import AnalysisAgent


class DemoModel(nn.Module):
    """A demo model for analysis."""
    
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((7, 7))
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 7 * 7, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def main():
    """Run the analysis agent demo."""
    print("🤖 Analysis Agent Demo")
    print("=" * 50)
    
    # Create a demo model
    print("📦 Creating demo model...")
    model = DemoModel()
    
    # Save the model to a temporary file
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        torch.save(model, f.name)
        model_path = f.name
    
    try:
        # Initialize the Analysis Agent
        print("🔧 Initializing Analysis Agent...")
        config = {
            "profiling_samples": 50,
            "warmup_samples": 10
        }
        agent = AnalysisAgent(config)
        agent.initialize()
        
        # Analyze the model
        print("🔍 Analyzing model...")
        report = agent.analyze_model(model_path)
        
        # Display results
        print("\n📊 Analysis Results")
        print("-" * 30)
        
        print(f"Model ID: {report.model_id}")
        print(f"Analysis Duration: {report.analysis_duration_seconds:.2f} seconds")
        
        # Architecture Summary
        arch = report.architecture_summary
        print(f"\n🏗️  Architecture Summary:")
        print(f"  Total Layers: {arch.total_layers}")
        print(f"  Total Parameters: {arch.total_parameters:,}")
        print(f"  Trainable Parameters: {arch.trainable_parameters:,}")
        print(f"  Model Depth: {arch.model_depth}")
        print(f"  Memory Footprint: {arch.memory_footprint_mb:.2f} MB")
        print(f"  Layer Types: {dict(arch.layer_types)}")
        
        # Performance Profile
        perf = report.performance_profile
        print(f"\n⚡ Performance Profile:")
        print(f"  Inference Time: {perf.inference_time_ms:.2f} ms")
        print(f"  Memory Usage: {perf.memory_usage_mb:.2f} MB")
        print(f"  Throughput: {perf.throughput_samples_per_sec:.2f} samples/sec")
        print(f"  CPU Utilization: {perf.cpu_utilization_percent:.1f}%")
        print(f"  GPU Utilization: {perf.gpu_utilization_percent:.1f}%")
        
        # Optimization Opportunities
        print(f"\n🎯 Optimization Opportunities ({len(report.optimization_opportunities)}):")
        for i, opp in enumerate(report.optimization_opportunities, 1):
            print(f"  {i}. {opp.technique.title()}")
            print(f"     Size Reduction: {opp.estimated_size_reduction_percent:.1f}%")
            print(f"     Speed Improvement: {opp.estimated_speed_improvement_percent:.1f}%")
            print(f"     Accuracy Impact: {opp.estimated_accuracy_impact_percent:.1f}%")
            print(f"     Confidence: {opp.confidence_score:.2f}")
            print(f"     Complexity: {opp.complexity}")
            print(f"     Description: {opp.description}")
            print()
        
        # Compatibility Matrix
        print(f"🔧 Technique Compatibility:")
        for technique, compatible in report.compatibility_matrix.items():
            status = "✅" if compatible else "❌"
            print(f"  {status} {technique.replace('_', ' ').title()}")
        
        # Recommendations
        print(f"\n💡 Recommendations ({len(report.recommendations)}):")
        for i, rec in enumerate(report.recommendations, 1):
            print(f"  {i}. {rec.technique.title()} (Priority: {rec.priority})")
            print(f"     Rationale: {rec.rationale}")
            print(f"     Expected Benefits: {', '.join(rec.expected_benefits)}")
            print(f"     Potential Risks: {', '.join(rec.potential_risks)}")
            print(f"     Estimated Effort: {rec.estimated_effort}")
            print()
        
        # Test bottleneck identification
        print("🔍 Identifying Bottlenecks...")
        bottlenecks = agent.identify_bottlenecks(model)
        
        if bottlenecks:
            print(f"\n⚠️  Identified Bottlenecks ({len(bottlenecks)}):")
            for i, bottleneck in enumerate(bottlenecks, 1):
                print(f"  {i}. {bottleneck['type'].replace('_', ' ').title()}")
                print(f"     Description: {bottleneck['description']}")
                print(f"     Severity: {bottleneck['severity']}")
                if 'suggestions' in bottleneck:
                    print(f"     Suggestions: {', '.join(bottleneck['suggestions'])}")
                print()
        else:
            print("✅ No significant bottlenecks identified")
        
        # Cleanup
        agent.cleanup()
        print("\n✅ Analysis completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        raise
    
    finally:
        # Clean up temporary file
        if os.path.exists(model_path):
            os.unlink(model_path)


if __name__ == "__main__":
    main()
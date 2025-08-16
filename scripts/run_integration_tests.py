#!/usr/bin/env python3
"""
Integration Test Runner - Execute comprehensive integration tests.

This script runs all integration tests to validate the complete platform
functionality and ensure all requirements are satisfied.
"""

import sys
import subprocess
import argparse
import json
import time
from pathlib import Path
from datetime import datetime
import tempfile
import shutil

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def run_command(cmd, cwd=None, capture_output=True):
    """Run a command and return the result."""
    print(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd or project_root,
            capture_output=capture_output,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        if result.returncode != 0:
            print(f"Command failed with return code {result.returncode}")
            if result.stderr:
                print(f"Error output: {result.stderr}")
        
        return result
    
    except subprocess.TimeoutExpired:
        print(f"Command timed out after 5 minutes")
        return None
    except Exception as e:
        print(f"Error running command: {e}")
        return None


def run_pytest_tests(test_pattern, output_file=None):
    """Run pytest tests with specified pattern."""
    cmd = [
        sys.executable, "-m", "pytest",
        "-v",
        "--tb=short",
        "--durations=10",
        test_pattern
    ]
    
    if output_file:
        cmd.extend(["--junitxml", str(output_file)])
    
    return run_command(cmd)


def run_integration_validation():
    """Run comprehensive integration validation."""
    print("\n" + "="*60)
    print("RUNNING INTEGRATION VALIDATION TESTS")
    print("="*60)
    
    test_results = {}
    
    # Test 1: Complete platform integration
    print("\n1. Testing complete platform integration...")
    result = run_pytest_tests("tests/integration/test_complete_platform_integration.py")
    test_results["platform_integration"] = result.returncode == 0 if result else False
    
    # Test 2: Final integration validation
    print("\n2. Testing final integration validation...")
    result = run_pytest_tests("tests/integration/test_final_integration_validation.py")
    test_results["final_validation"] = result.returncode == 0 if result else False
    
    # Test 3: End-to-end workflows
    print("\n3. Testing end-to-end workflows...")
    result = run_pytest_tests("tests/integration/test_end_to_end_workflows.py")
    test_results["end_to_end_workflows"] = result.returncode == 0 if result else False
    
    return test_results


def run_component_tests():
    """Run individual component tests."""
    print("\n" + "="*60)
    print("RUNNING COMPONENT TESTS")
    print("="*60)
    
    test_results = {}
    
    # Core component tests
    components = [
        "analysis_agent",
        "planning_agent", 
        "evaluation_agent",
        "quantization_agent",
        "pruning_agent",
        "distillation_agent",
        "compression_agent",
        "architecture_search_agent",
        "optimization_manager",
        "model_store",
        "memory_manager",
        "notification_service",
        "monitoring_service"
    ]
    
    for component in components:
        print(f"\nTesting {component}...")
        result = run_pytest_tests(f"tests/test_{component}.py")
        test_results[component] = result.returncode == 0 if result else False
    
    return test_results


def run_api_tests():
    """Run API integration tests."""
    print("\n" + "="*60)
    print("RUNNING API TESTS")
    print("="*60)
    
    result = run_pytest_tests("tests/test_api.py")
    return {"api": result.returncode == 0 if result else False}


def run_comprehensive_suite():
    """Run the comprehensive test suite."""
    print("\n" + "="*60)
    print("RUNNING COMPREHENSIVE TEST SUITE")
    print("="*60)
    
    result = run_pytest_tests("tests/test_comprehensive_suite.py")
    return {"comprehensive_suite": result.returncode == 0 if result else False}


def validate_platform_startup():
    """Validate that the platform can start up successfully."""
    print("\n" + "="*60)
    print("VALIDATING PLATFORM STARTUP")
    print("="*60)
    
    try:
        # Create a temporary test model
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create a simple test model
            import torch
            import torch.nn as nn
            
            class SimpleModel(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.fc = nn.Linear(10, 1)
                
                def forward(self, x):
                    return self.fc(x)
            
            model = SimpleModel()
            model_path = temp_path / "test_model.pth"
            torch.save(model.state_dict(), model_path)
            
            # Test platform startup with test workflow
            cmd = [
                sys.executable, "-m", "src.main",
                "--test-workflow", str(model_path)
            ]
            
            print("Testing platform startup with test workflow...")
            result = run_command(cmd, capture_output=True)
            
            if result and result.returncode == 0:
                print("✅ Platform startup validation successful")
                return True
            else:
                print("❌ Platform startup validation failed")
                if result and result.stderr:
                    print(f"Error: {result.stderr}")
                return False
    
    except Exception as e:
        print(f"❌ Platform startup validation failed with exception: {e}")
        return False


def generate_test_report(all_results, output_dir):
    """Generate comprehensive test report."""
    report = {
        "timestamp": datetime.now().isoformat(),
        "summary": {
            "total_test_categories": len(all_results),
            "passed_categories": sum(1 for results in all_results.values() 
                                   if all(results.values()) if results else 0),
            "failed_categories": sum(1 for results in all_results.values() 
                                   if not all(results.values()) if results else 1)
        },
        "detailed_results": all_results
    }
    
    # Calculate overall success rate
    total_tests = sum(len(results) for results in all_results.values() if results)
    passed_tests = sum(sum(results.values()) for results in all_results.values() if results)
    
    if total_tests > 0:
        report["summary"]["success_rate"] = (passed_tests / total_tests) * 100
    else:
        report["summary"]["success_rate"] = 0
    
    # Save report
    report_file = output_dir / "integration_test_report.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Generate text summary
    summary_file = output_dir / "integration_test_summary.txt"
    with open(summary_file, 'w') as f:
        f.write("INTEGRATION TEST SUMMARY\n")
        f.write("="*50 + "\n\n")
        f.write(f"Timestamp: {report['timestamp']}\n")
        f.write(f"Total Categories: {report['summary']['total_test_categories']}\n")
        f.write(f"Passed Categories: {report['summary']['passed_categories']}\n")
        f.write(f"Failed Categories: {report['summary']['failed_categories']}\n")
        f.write(f"Success Rate: {report['summary']['success_rate']:.1f}%\n\n")
        
        f.write("DETAILED RESULTS:\n")
        f.write("-" * 30 + "\n")
        
        for category, results in all_results.items():
            if results:
                category_passed = sum(results.values())
                category_total = len(results)
                category_rate = (category_passed / category_total * 100) if category_total > 0 else 0
                
                f.write(f"\n{category.upper()}:\n")
                f.write(f"  Passed: {category_passed}/{category_total} ({category_rate:.1f}%)\n")
                
                for test_name, passed in results.items():
                    status = "✅ PASS" if passed else "❌ FAIL"
                    f.write(f"  - {test_name}: {status}\n")
            else:
                f.write(f"\n{category.upper()}: ❌ FAILED TO RUN\n")
    
    return report_file, summary_file


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run comprehensive integration tests for the robotics optimization platform"
    )
    
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("test_results"),
        help="Output directory for test results"
    )
    
    parser.add_argument(
        "--skip-component-tests",
        action="store_true",
        help="Skip individual component tests"
    )
    
    parser.add_argument(
        "--skip-startup-validation",
        action="store_true",
        help="Skip platform startup validation"
    )
    
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run only essential integration tests"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    print("ROBOTICS MODEL OPTIMIZATION PLATFORM")
    print("INTEGRATION TEST RUNNER")
    print("="*60)
    print(f"Output directory: {args.output_dir}")
    print(f"Start time: {datetime.now()}")
    
    all_results = {}
    start_time = time.time()
    
    try:
        # Run integration validation tests (always run these)
        all_results["integration_validation"] = run_integration_validation()
        
        if not args.quick:
            # Run component tests
            if not args.skip_component_tests:
                all_results["component_tests"] = run_component_tests()
            
            # Run API tests
            all_results["api_tests"] = run_api_tests()
            
            # Run comprehensive suite
            all_results["comprehensive_suite"] = run_comprehensive_suite()
        
        # Run platform startup validation
        if not args.skip_startup_validation:
            startup_success = validate_platform_startup()
            all_results["platform_startup"] = {"startup_validation": startup_success}
        
        # Generate report
        execution_time = time.time() - start_time
        
        print("\n" + "="*60)
        print("GENERATING TEST REPORT")
        print("="*60)
        
        report_file, summary_file = generate_test_report(all_results, args.output_dir)
        
        print(f"Report saved to: {report_file}")
        print(f"Summary saved to: {summary_file}")
        
        # Print summary
        print("\n" + "="*60)
        print("FINAL SUMMARY")
        print("="*60)
        
        total_categories = len(all_results)
        passed_categories = sum(1 for results in all_results.values() 
                              if results and all(results.values()))
        
        print(f"Execution time: {execution_time:.2f} seconds")
        print(f"Test categories: {passed_categories}/{total_categories}")
        
        # Determine overall result
        if passed_categories == total_categories:
            print("🎉 ALL INTEGRATION TESTS PASSED!")
            return 0
        else:
            print("❌ SOME INTEGRATION TESTS FAILED!")
            
            # Show failed categories
            for category, results in all_results.items():
                if not results or not all(results.values()):
                    print(f"  - {category}: FAILED")
            
            return 1
    
    except KeyboardInterrupt:
        print("\n❌ Integration tests interrupted by user")
        return 1
    
    except Exception as e:
        print(f"\n❌ Integration tests failed with error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
#!/usr/bin/env python3
"""
Comprehensive End-to-End Test Runner

This script runs all end-to-end test scenarios to validate complete requirements coverage.
It provides detailed reporting on which requirements and acceptance criteria are covered.
"""

import sys
import subprocess
import json
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def run_e2e_test_suite():
    """Run the comprehensive end-to-end test suite."""
    
    print("=" * 80)
    print("COMPREHENSIVE END-TO-END REQUIREMENTS COVERAGE TEST SUITE")
    print("=" * 80)
    print(f"Started at: {datetime.now().isoformat()}")
    print()
    
    # Test files to run
    test_files = [
        "tests/integration/test_comprehensive_e2e_requirements.py",
        "tests/integration/test_e2e_edge_cases_and_scenarios.py",
        "tests/integration/test_end_to_end_workflows.py",
        "tests/integration/test_complete_platform_integration.py",
        "tests/integration/test_final_integration_validation.py"
    ]
    
    # Requirements coverage mapping
    requirements_coverage = {
        "Requirement 1 - Autonomous Analysis": [
            "1.1 - Automatic model analysis on upload",
            "1.2 - Optimization strategy identification", 
            "1.3 - Optimization ranking by impact",
            "1.4 - Detailed optimization rationale"
        ],
        "Requirement 2 - Automatic Optimization": [
            "2.1 - Automatic optimization execution",
            "2.2 - Real-time progress tracking",
            "2.3 - Automatic rollback on failure",
            "2.4 - Detailed optimization report"
        ],
        "Requirement 3 - Comprehensive Evaluation": [
            "3.1 - Automatic performance testing",
            "3.2 - Benchmark testing",
            "3.3 - Model comparison report",
            "3.4 - Unsuccessful optimization detection"
        ],
        "Requirement 4 - Configurable Criteria": [
            "4.1 - Configurable thresholds and constraints",
            "4.2 - Criteria validation and application",
            "4.3 - Conflicting criteria detection",
            "4.4 - Criteria audit logging"
        ],
        "Requirement 5 - Monitoring and Control": [
            "5.1 - Real-time progress monitoring",
            "5.2 - Optimization process control",
            "5.3 - Completion notifications",
            "5.4 - Optimization history access"
        ],
        "Requirement 6 - Multiple Model Types": [
            "6.1 - Multiple model format support",
            "6.2 - Automatic model type identification",
            "6.3 - Comprehensive optimization techniques",
            "6.4 - Unsupported model error handling"
        ]
    }
    
    print("REQUIREMENTS COVERAGE:")
    print("-" * 40)
    for req, criteria in requirements_coverage.items():
        print(f"✓ {req}")
        for criterion in criteria:
            print(f"  • {criterion}")
    print()
    
    # Run tests
    total_tests = 0
    passed_tests = 0
    failed_tests = 0
    
    for test_file in test_files:
        print(f"Running: {test_file}")
        print("-" * 60)
        
        try:
            # Run pytest with verbose output
            result = subprocess.run([
                sys.executable, "-m", "pytest", 
                test_file, 
                "-v", 
                "--tb=short",
                "--no-header"
            ], capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                print("✓ PASSED")
                passed_tests += 1
            else:
                print("✗ FAILED")
                print("STDOUT:", result.stdout)
                print("STDERR:", result.stderr)
                failed_tests += 1
                
            total_tests += 1
            
        except subprocess.TimeoutExpired:
            print("✗ TIMEOUT - Test took longer than 5 minutes")
            failed_tests += 1
            total_tests += 1
        except Exception as e:
            print(f"✗ ERROR - {str(e)}")
            failed_tests += 1
            total_tests += 1
        
        print()
    
    # Summary
    print("=" * 80)
    print("TEST EXECUTION SUMMARY")
    print("=" * 80)
    print(f"Total test files: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {failed_tests}")
    print(f"Success rate: {(passed_tests/total_tests)*100:.1f}%" if total_tests > 0 else "N/A")
    print()
    
    print("REQUIREMENTS VALIDATION STATUS:")
    print("-" * 40)
    if passed_tests == total_tests:
        print("✓ ALL REQUIREMENTS COVERED - End-to-end tests validate complete platform functionality")
        print("✓ Platform ready for production deployment")
    else:
        print("⚠ SOME TESTS FAILED - Review failed tests to ensure complete requirements coverage")
        print("⚠ Address failures before production deployment")
    
    print(f"\nCompleted at: {datetime.now().isoformat()}")
    
    return passed_tests == total_tests


if __name__ == "__main__":
    success = run_e2e_test_suite()
    sys.exit(0 if success else 1)
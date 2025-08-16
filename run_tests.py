#!/usr/bin/env python3
"""
Simple test execution script for the robotics model optimization platform.
Provides easy access to the comprehensive testing suite.
"""

import sys
import argparse
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from tests.automation.test_runner import TestRunner, TestSuite


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run tests for the robotics model optimization platform",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_tests.py                    # Run all tests
  python run_tests.py --suite unit       # Run only unit tests
  python run_tests.py --suite performance # Run performance benchmarks
  python run_tests.py --suite stress     # Run stress tests
  python run_tests.py --quick            # Run quick test subset
  python run_tests.py --generate-data    # Only generate test data
        """
    )
    
    parser.add_argument(
        "--suite",
        choices=[s.value for s in TestSuite],
        default="all",
        help="Test suite to run (default: all)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("test_results"),
        help="Output directory for test results (default: test_results)"
    )
    
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick test subset (unit + integration only)"
    )
    
    parser.add_argument(
        "--generate-data",
        action="store_true",
        help="Only generate test data, don't run tests"
    )
    
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run tests in parallel where possible"
    )
    
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output"
    )
    
    args = parser.parse_args()
    
    # Create test runner
    runner = TestRunner(args.output_dir)
    
    if args.verbose:
        print(f"Test execution ID: {runner.execution_id}")
        print(f"Output directory: {args.output_dir}")
        print(f"Environment: {runner.get_environment_info()}")
    
    # Handle special modes
    if args.generate_data:
        print("Generating test data...")
        success = runner.generate_test_data()
        if success:
            print("Test data generated successfully!")
            return 0
        else:
            print("Failed to generate test data!")
            return 1
    
    # Determine which suite to run
    if args.quick:
        # Run only unit and integration tests for quick feedback
        suites_to_run = [TestSuite.UNIT, TestSuite.INTEGRATION]
        print("Running quick test subset (unit + integration)...")
    else:
        suite = TestSuite(args.suite)
        if suite == TestSuite.ALL:
            suites_to_run = [TestSuite.UNIT, TestSuite.INTEGRATION, TestSuite.PERFORMANCE, TestSuite.STRESS]
        else:
            suites_to_run = [suite]
    
    # Run tests
    all_results = []
    for suite in suites_to_run:
        print(f"\n{'='*60}")
        print(f"Running {suite.value.upper()} tests...")
        print(f"{'='*60}")
        
        results = runner.run_test_suite(suite, args.parallel)
        all_results.extend(results)
        
        # Print immediate results
        for result in results:
            success_rate = (result.passed_tests / result.total_tests * 100) if result.total_tests > 0 else 0
            print(f"\n{result.suite_name.upper()} Results:")
            print(f"  Tests: {result.total_tests}")
            print(f"  Passed: {result.passed_tests}")
            print(f"  Failed: {result.failed_tests}")
            print(f"  Success Rate: {success_rate:.1f}%")
            print(f"  Duration: {result.total_duration_seconds:.2f}s")
    
    # Generate final report
    print(f"\n{'='*60}")
    print("Generating final report...")
    print(f"{'='*60}")
    
    report = runner.generate_report()
    saved_files = runner.save_report(report, ["json", "html", "txt"])
    
    # Print summary
    print(f"\nFINAL RESULTS:")
    print(f"Total Tests: {report.summary['total_tests']}")
    print(f"Passed: {report.summary['passed_tests']}")
    print(f"Failed: {report.summary['failed_tests']}")
    print(f"Skipped: {report.summary['skipped_tests']}")
    print(f"Errors: {report.summary['error_tests']}")
    print(f"Success Rate: {report.summary['success_rate']:.1f}%")
    print(f"Total Duration: {report.total_duration_seconds:.2f}s")
    
    print(f"\nReports saved:")
    for file_path in saved_files:
        print(f"  - {file_path}")
    
    # Return appropriate exit code
    if report.summary['failed_tests'] > 0 or report.summary['error_tests'] > 0:
        print(f"\n❌ Tests failed!")
        return 1
    else:
        print(f"\n✅ All tests passed!")
        return 0


if __name__ == "__main__":
    sys.exit(main())
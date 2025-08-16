"""
Automated test execution and reporting system.
Orchestrates comprehensive testing across all test suites and generates reports.
"""

import subprocess
import sys
import time
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import tempfile
import shutil
import concurrent.futures
from enum import Enum

# Import test modules
sys.path.append(str(Path(__file__).parent.parent.parent))

from tests.data.test_data_generator import TestDataGenerator
from tests.performance.test_optimization_benchmarks import OptimizationBenchmarks
from tests.stress.test_concurrent_optimizations import ConcurrentOptimizationStressTester


class TestSuite(Enum):
    """Available test suites."""
    UNIT = "unit"
    INTEGRATION = "integration"
    PERFORMANCE = "performance"
    STRESS = "stress"
    END_TO_END = "end_to_end"
    ALL = "all"


@dataclass
class TestResult:
    """Individual test result."""
    test_name: str
    suite: str
    status: str  # "passed", "failed", "skipped", "error"
    duration_seconds: float
    error_message: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


@dataclass
class TestSuiteResult:
    """Test suite execution result."""
    suite_name: str
    total_tests: int
    passed_tests: int
    failed_tests: int
    skipped_tests: int
    error_tests: int
    total_duration_seconds: float
    test_results: List[TestResult]


@dataclass
class TestExecutionReport:
    """Complete test execution report."""
    execution_id: str
    start_time: datetime
    end_time: datetime
    total_duration_seconds: float
    suite_results: List[TestSuiteResult]
    summary: Dict[str, Any]
    environment_info: Dict[str, Any]


class TestRunner:
    """Automated test runner and orchestrator."""
    
    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or Path("test_results")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.test_data_generator = TestDataGenerator()
        self.benchmark_suite = OptimizationBenchmarks()
        self.stress_tester = ConcurrentOptimizationStressTester()
        
        self.execution_id = f"test_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.results: List[TestSuiteResult] = []
    
    def get_environment_info(self) -> Dict[str, Any]:
        """Collect environment information."""
        import torch
        import platform
        import psutil
        
        return {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "pytorch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
            "cpu_count": psutil.cpu_count(),
            "memory_gb": psutil.virtual_memory().total / (1024**3),
            "timestamp": datetime.now().isoformat()
        }
    
    def run_pytest_suite(self, test_path: str, suite_name: str, 
                        extra_args: List[str] = None) -> TestSuiteResult:
        """Run pytest suite and parse results."""
        print(f"Running {suite_name} tests...")
        
        # Prepare pytest command
        cmd = [
            sys.executable, "-m", "pytest",
            test_path,
            "--json-report",
            f"--json-report-file={self.output_dir}/{suite_name}_results.json",
            "-v"
        ]
        
        if extra_args:
            cmd.extend(extra_args)
        
        start_time = time.perf_counter()
        
        try:
            # Run pytest
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout
            )
            
            end_time = time.perf_counter()
            duration = end_time - start_time
            
            # Parse JSON report
            json_report_path = self.output_dir / f"{suite_name}_results.json"
            if json_report_path.exists():
                with open(json_report_path, 'r') as f:
                    pytest_report = json.load(f)
                
                return self._parse_pytest_report(pytest_report, suite_name, duration)
            else:
                # Fallback if JSON report not available
                return TestSuiteResult(
                    suite_name=suite_name,
                    total_tests=0,
                    passed_tests=0,
                    failed_tests=1,
                    skipped_tests=0,
                    error_tests=0,
                    total_duration_seconds=duration,
                    test_results=[
                        TestResult(
                            test_name=f"{suite_name}_execution",
                            suite=suite_name,
                            status="failed",
                            duration_seconds=duration,
                            error_message=result.stderr or "Unknown error"
                        )
                    ]
                )
        
        except subprocess.TimeoutExpired:
            return TestSuiteResult(
                suite_name=suite_name,
                total_tests=0,
                passed_tests=0,
                failed_tests=1,
                skipped_tests=0,
                error_tests=0,
                total_duration_seconds=3600,
                test_results=[
                    TestResult(
                        test_name=f"{suite_name}_timeout",
                        suite=suite_name,
                        status="error",
                        duration_seconds=3600,
                        error_message="Test suite timed out after 1 hour"
                    )
                ]
            )
        
        except Exception as e:
            return TestSuiteResult(
                suite_name=suite_name,
                total_tests=0,
                passed_tests=0,
                failed_tests=0,
                skipped_tests=0,
                error_tests=1,
                total_duration_seconds=0,
                test_results=[
                    TestResult(
                        test_name=f"{suite_name}_error",
                        suite=suite_name,
                        status="error",
                        duration_seconds=0,
                        error_message=str(e)
                    )
                ]
            )
    
    def _parse_pytest_report(self, pytest_report: Dict[str, Any], 
                           suite_name: str, duration: float) -> TestSuiteResult:
        """Parse pytest JSON report into TestSuiteResult."""
        summary = pytest_report.get("summary", {})
        tests = pytest_report.get("tests", [])
        
        test_results = []
        for test in tests:
            test_result = TestResult(
                test_name=test.get("nodeid", "unknown"),
                suite=suite_name,
                status=test.get("outcome", "unknown"),
                duration_seconds=test.get("duration", 0),
                error_message=test.get("call", {}).get("longrepr") if test.get("outcome") == "failed" else None,
                details={
                    "setup_duration": test.get("setup", {}).get("duration", 0),
                    "call_duration": test.get("call", {}).get("duration", 0),
                    "teardown_duration": test.get("teardown", {}).get("duration", 0)
                }
            )
            test_results.append(test_result)
        
        return TestSuiteResult(
            suite_name=suite_name,
            total_tests=summary.get("total", 0),
            passed_tests=summary.get("passed", 0),
            failed_tests=summary.get("failed", 0),
            skipped_tests=summary.get("skipped", 0),
            error_tests=summary.get("error", 0),
            total_duration_seconds=duration,
            test_results=test_results
        )
    
    def run_performance_benchmarks(self) -> TestSuiteResult:
        """Run performance benchmarks."""
        print("Running performance benchmarks...")
        
        start_time = time.perf_counter()
        
        try:
            # Run comprehensive benchmarks
            benchmark_results = self.benchmark_suite.run_comprehensive_benchmarks()
            
            end_time = time.perf_counter()
            duration = end_time - start_time
            
            # Convert benchmark results to test results
            test_results = []
            passed_tests = 0
            failed_tests = 0
            
            for result in benchmark_results:
                status = "passed" if result.success else "failed"
                if result.success:
                    passed_tests += 1
                else:
                    failed_tests += 1
                
                test_result = TestResult(
                    test_name=f"benchmark_{result.technique}_{result.model_type}",
                    suite="performance",
                    status=status,
                    duration_seconds=result.optimization_time_seconds,
                    error_message=result.error_message if not result.success else None,
                    details={
                        "speedup_factor": result.speedup_factor,
                        "memory_reduction": result.memory_reduction,
                        "accuracy_retention": result.accuracy_retention,
                        "model_size_mb": result.model_size_mb
                    }
                )
                test_results.append(test_result)
            
            # Save benchmark report
            benchmark_report = self.benchmark_suite.generate_report()
            with open(self.output_dir / "performance_benchmark_report.txt", 'w') as f:
                f.write(benchmark_report)
            
            return TestSuiteResult(
                suite_name="performance",
                total_tests=len(benchmark_results),
                passed_tests=passed_tests,
                failed_tests=failed_tests,
                skipped_tests=0,
                error_tests=0,
                total_duration_seconds=duration,
                test_results=test_results
            )
        
        except Exception as e:
            return TestSuiteResult(
                suite_name="performance",
                total_tests=0,
                passed_tests=0,
                failed_tests=0,
                skipped_tests=0,
                error_tests=1,
                total_duration_seconds=0,
                test_results=[
                    TestResult(
                        test_name="performance_benchmark_error",
                        suite="performance",
                        status="error",
                        duration_seconds=0,
                        error_message=str(e)
                    )
                ]
            )
    
    def run_stress_tests(self) -> TestSuiteResult:
        """Run stress tests."""
        print("Running stress tests...")
        
        start_time = time.perf_counter()
        
        try:
            # Run different stress test scenarios
            stress_results = []
            
            # Concurrent sessions test
            import asyncio
            concurrent_result = asyncio.run(
                self.stress_tester.test_concurrent_sessions(10, "automated_concurrent")
            )
            stress_results.append(concurrent_result)
            
            # Memory pressure test
            memory_result = self.stress_tester.test_memory_pressure(5)
            stress_results.append(memory_result)
            
            # Rapid session creation test
            rapid_result = self.stress_tester.test_rapid_session_creation(3, 10)
            stress_results.append(rapid_result)
            
            end_time = time.perf_counter()
            duration = end_time - start_time
            
            # Convert stress results to test results
            test_results = []
            passed_tests = 0
            failed_tests = 0
            
            for result in stress_results:
                # Consider test passed if success rate > 70%
                success_rate = result.successful_sessions / result.concurrent_sessions if result.concurrent_sessions > 0 else 0
                status = "passed" if success_rate > 0.7 else "failed"
                
                if status == "passed":
                    passed_tests += 1
                else:
                    failed_tests += 1
                
                test_result = TestResult(
                    test_name=f"stress_{result.test_name}",
                    suite="stress",
                    status=status,
                    duration_seconds=result.total_time_seconds,
                    error_message="; ".join(result.errors[:3]) if result.errors else None,
                    details={
                        "concurrent_sessions": result.concurrent_sessions,
                        "successful_sessions": result.successful_sessions,
                        "success_rate": success_rate,
                        "peak_memory_mb": result.peak_memory_mb,
                        "peak_cpu_percent": result.peak_cpu_percent
                    }
                )
                test_results.append(test_result)
            
            # Save stress test report
            stress_report = self.stress_tester.generate_stress_test_report()
            with open(self.output_dir / "stress_test_report.txt", 'w') as f:
                f.write(stress_report)
            
            return TestSuiteResult(
                suite_name="stress",
                total_tests=len(stress_results),
                passed_tests=passed_tests,
                failed_tests=failed_tests,
                skipped_tests=0,
                error_tests=0,
                total_duration_seconds=duration,
                test_results=test_results
            )
        
        except Exception as e:
            return TestSuiteResult(
                suite_name="stress",
                total_tests=0,
                passed_tests=0,
                failed_tests=0,
                skipped_tests=0,
                error_tests=1,
                total_duration_seconds=0,
                test_results=[
                    TestResult(
                        test_name="stress_test_error",
                        suite="stress",
                        status="error",
                        duration_seconds=0,
                        error_message=str(e)
                    )
                ]
            )
    
    def generate_test_data(self) -> bool:
        """Generate test data for all scenarios."""
        print("Generating test data...")
        
        try:
            test_data_dir = self.output_dir / "test_data"
            test_suite = self.test_data_generator.create_test_suite(test_data_dir)
            
            print(f"Generated test data for {len(test_suite['scenarios'])} scenarios")
            return True
        
        except Exception as e:
            print(f"Failed to generate test data: {e}")
            return False
    
    def run_test_suite(self, suite: TestSuite, 
                      parallel: bool = False) -> List[TestSuiteResult]:
        """Run specified test suite(s)."""
        results = []
        
        if suite == TestSuite.ALL:
            suites_to_run = [
                TestSuite.UNIT,
                TestSuite.INTEGRATION,
                TestSuite.PERFORMANCE,
                TestSuite.STRESS
            ]
        else:
            suites_to_run = [suite]
        
        # Generate test data first
        if not self.generate_test_data():
            print("Warning: Test data generation failed, some tests may not work properly")
        
        for test_suite in suites_to_run:
            if test_suite == TestSuite.UNIT:
                result = self.run_pytest_suite("tests/test_*.py", "unit")
                results.append(result)
            
            elif test_suite == TestSuite.INTEGRATION:
                result = self.run_pytest_suite("tests/integration/", "integration")
                results.append(result)
            
            elif test_suite == TestSuite.PERFORMANCE:
                result = self.run_performance_benchmarks()
                results.append(result)
            
            elif test_suite == TestSuite.STRESS:
                result = self.run_stress_tests()
                results.append(result)
        
        self.results.extend(results)
        return results
    
    def generate_report(self) -> TestExecutionReport:
        """Generate comprehensive test execution report."""
        end_time = datetime.now()
        start_time = end_time  # Will be updated based on first test
        
        total_duration = 0
        total_tests = 0
        total_passed = 0
        total_failed = 0
        total_skipped = 0
        total_errors = 0
        
        for suite_result in self.results:
            total_duration += suite_result.total_duration_seconds
            total_tests += suite_result.total_tests
            total_passed += suite_result.passed_tests
            total_failed += suite_result.failed_tests
            total_skipped += suite_result.skipped_tests
            total_errors += suite_result.error_tests
        
        summary = {
            "total_tests": total_tests,
            "passed_tests": total_passed,
            "failed_tests": total_failed,
            "skipped_tests": total_skipped,
            "error_tests": total_errors,
            "success_rate": (total_passed / total_tests * 100) if total_tests > 0 else 0,
            "total_suites": len(self.results)
        }
        
        report = TestExecutionReport(
            execution_id=self.execution_id,
            start_time=start_time,
            end_time=end_time,
            total_duration_seconds=total_duration,
            suite_results=self.results,
            summary=summary,
            environment_info=self.get_environment_info()
        )
        
        return report
    
    def save_report(self, report: TestExecutionReport, 
                   formats: List[str] = None) -> List[Path]:
        """Save report in specified formats."""
        if formats is None:
            formats = ["json", "html", "txt"]
        
        saved_files = []
        
        # JSON format
        if "json" in formats:
            json_path = self.output_dir / f"{self.execution_id}_report.json"
            with open(json_path, 'w') as f:
                json.dump(asdict(report), f, indent=2, default=str)
            saved_files.append(json_path)
        
        # Text format
        if "txt" in formats:
            txt_path = self.output_dir / f"{self.execution_id}_report.txt"
            with open(txt_path, 'w') as f:
                f.write(self._generate_text_report(report))
            saved_files.append(txt_path)
        
        # HTML format
        if "html" in formats:
            html_path = self.output_dir / f"{self.execution_id}_report.html"
            with open(html_path, 'w') as f:
                f.write(self._generate_html_report(report))
            saved_files.append(html_path)
        
        return saved_files
    
    def _generate_text_report(self, report: TestExecutionReport) -> str:
        """Generate text format report."""
        lines = []
        lines.append("=" * 80)
        lines.append("ROBOTICS MODEL OPTIMIZATION PLATFORM - TEST EXECUTION REPORT")
        lines.append("=" * 80)
        lines.append(f"Execution ID: {report.execution_id}")
        lines.append(f"Start Time: {report.start_time}")
        lines.append(f"End Time: {report.end_time}")
        lines.append(f"Total Duration: {report.total_duration_seconds:.2f} seconds")
        lines.append("")
        
        # Summary
        lines.append("SUMMARY")
        lines.append("-" * 40)
        lines.append(f"Total Tests: {report.summary['total_tests']}")
        lines.append(f"Passed: {report.summary['passed_tests']}")
        lines.append(f"Failed: {report.summary['failed_tests']}")
        lines.append(f"Skipped: {report.summary['skipped_tests']}")
        lines.append(f"Errors: {report.summary['error_tests']}")
        lines.append(f"Success Rate: {report.summary['success_rate']:.1f}%")
        lines.append("")
        
        # Environment
        lines.append("ENVIRONMENT")
        lines.append("-" * 40)
        for key, value in report.environment_info.items():
            lines.append(f"{key}: {value}")
        lines.append("")
        
        # Suite Results
        lines.append("SUITE RESULTS")
        lines.append("-" * 40)
        for suite_result in report.suite_results:
            lines.append(f"\n{suite_result.suite_name.upper()} SUITE:")
            lines.append(f"  Total Tests: {suite_result.total_tests}")
            lines.append(f"  Passed: {suite_result.passed_tests}")
            lines.append(f"  Failed: {suite_result.failed_tests}")
            lines.append(f"  Duration: {suite_result.total_duration_seconds:.2f}s")
            
            # Show failed tests
            failed_tests = [t for t in suite_result.test_results if t.status == "failed"]
            if failed_tests:
                lines.append("  Failed Tests:")
                for test in failed_tests[:5]:  # Show first 5
                    lines.append(f"    - {test.test_name}: {test.error_message}")
                if len(failed_tests) > 5:
                    lines.append(f"    ... and {len(failed_tests) - 5} more")
        
        return "\n".join(lines)
    
    def _generate_html_report(self, report: TestExecutionReport) -> str:
        """Generate HTML format report."""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Test Execution Report - {report.execution_id}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .summary {{ margin: 20px 0; }}
                .suite {{ margin: 20px 0; border: 1px solid #ddd; padding: 15px; border-radius: 5px; }}
                .passed {{ color: green; }}
                .failed {{ color: red; }}
                .error {{ color: orange; }}
                .skipped {{ color: gray; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Test Execution Report</h1>
                <p><strong>Execution ID:</strong> {report.execution_id}</p>
                <p><strong>Duration:</strong> {report.total_duration_seconds:.2f} seconds</p>
                <p><strong>Success Rate:</strong> {report.summary['success_rate']:.1f}%</p>
            </div>
            
            <div class="summary">
                <h2>Summary</h2>
                <table>
                    <tr><th>Metric</th><th>Count</th></tr>
                    <tr><td>Total Tests</td><td>{report.summary['total_tests']}</td></tr>
                    <tr><td class="passed">Passed</td><td>{report.summary['passed_tests']}</td></tr>
                    <tr><td class="failed">Failed</td><td>{report.summary['failed_tests']}</td></tr>
                    <tr><td class="skipped">Skipped</td><td>{report.summary['skipped_tests']}</td></tr>
                    <tr><td class="error">Errors</td><td>{report.summary['error_tests']}</td></tr>
                </table>
            </div>
        """
        
        # Add suite results
        for suite_result in report.suite_results:
            html += f"""
            <div class="suite">
                <h3>{suite_result.suite_name.upper()} Suite</h3>
                <p>Duration: {suite_result.total_duration_seconds:.2f}s</p>
                <p>
                    <span class="passed">Passed: {suite_result.passed_tests}</span> | 
                    <span class="failed">Failed: {suite_result.failed_tests}</span> | 
                    <span class="skipped">Skipped: {suite_result.skipped_tests}</span> | 
                    <span class="error">Errors: {suite_result.error_tests}</span>
                </p>
            </div>
            """
        
        html += """
        </body>
        </html>
        """
        
        return html


def main():
    """Main entry point for automated test runner."""
    parser = argparse.ArgumentParser(description="Automated test runner for robotics optimization platform")
    parser.add_argument(
        "--suite", 
        choices=[s.value for s in TestSuite], 
        default="all",
        help="Test suite to run"
    )
    parser.add_argument(
        "--output-dir", 
        type=Path, 
        default=Path("test_results"),
        help="Output directory for test results"
    )
    parser.add_argument(
        "--parallel", 
        action="store_true",
        help="Run tests in parallel where possible"
    )
    parser.add_argument(
        "--formats", 
        nargs="+", 
        choices=["json", "html", "txt"], 
        default=["json", "html", "txt"],
        help="Report output formats"
    )
    
    args = parser.parse_args()
    
    # Create test runner
    runner = TestRunner(args.output_dir)
    
    print(f"Starting automated test execution: {runner.execution_id}")
    print(f"Output directory: {args.output_dir}")
    
    # Run tests
    suite = TestSuite(args.suite)
    results = runner.run_test_suite(suite, args.parallel)
    
    # Generate and save report
    report = runner.generate_report()
    saved_files = runner.save_report(report, args.formats)
    
    print(f"\nTest execution completed!")
    print(f"Total tests: {report.summary['total_tests']}")
    print(f"Success rate: {report.summary['success_rate']:.1f}%")
    print(f"Reports saved to:")
    for file_path in saved_files:
        print(f"  - {file_path}")
    
    # Exit with appropriate code
    if report.summary['failed_tests'] > 0 or report.summary['error_tests'] > 0:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
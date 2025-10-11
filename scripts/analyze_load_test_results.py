#!/usr/bin/env python3
"""
Analyze load test results and generate performance reports.

This script processes load test output and generates detailed performance
analysis reports with recommendations.
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List


class LoadTestAnalyzer:
    """Analyze load test results and generate reports."""
    
    # Performance thresholds
    THRESHOLDS = {
        "dashboard": {
            "success_rate": 95.0,
            "avg_response_time": 1.0,
            "p95_response_time": 2.0,
            "requests_per_second": 50
        },
        "sessions": {
            "success_rate": 95.0,
            "avg_response_time": 1.5,
            "p95_response_time": 3.0,
            "requests_per_second": 40
        },
        "config": {
            "success_rate": 95.0,
            "avg_response_time": 0.5,
            "p95_response_time": 1.0,
            "requests_per_second": 100
        },
        "websocket": {
            "success_rate": 90.0,
            "avg_connection_time": 1.0
        }
    }
    
    def __init__(self, results_file: str = None):
        """Initialize analyzer with optional results file."""
        self.results_file = results_file
        self.results: Dict[str, Any] = {}
        self.issues: List[str] = []
        self.recommendations: List[str] = []
    
    def load_results(self) -> bool:
        """Load test results from file."""
        if not self.results_file:
            print("No results file specified")
            return False
        
        try:
            with open(self.results_file, 'r') as f:
                self.results = json.load(f)
            return True
        except FileNotFoundError:
            print(f"Results file not found: {self.results_file}")
            return False
        except json.JSONDecodeError:
            print(f"Invalid JSON in results file: {self.results_file}")
            return False
    
    def analyze_endpoint(self, endpoint_name: str, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze metrics for a specific endpoint."""
        thresholds = self.THRESHOLDS.get(endpoint_name, {})
        analysis = {
            "endpoint": endpoint_name,
            "metrics": metrics,
            "status": "PASS",
            "issues": [],
            "recommendations": []
        }
        
        # Check success rate
        if "success_rate" in metrics and "success_rate" in thresholds:
            if metrics["success_rate"] < thresholds["success_rate"]:
                analysis["status"] = "FAIL"
                analysis["issues"].append(
                    f"Success rate {metrics['success_rate']:.2f}% below threshold {thresholds['success_rate']}%"
                )
                analysis["recommendations"].append(
                    "Investigate error logs and increase error handling robustness"
                )
        
        # Check average response time
        if "avg_response_time" in metrics and "avg_response_time" in thresholds:
            if metrics["avg_response_time"] > thresholds["avg_response_time"]:
                analysis["status"] = "FAIL"
                analysis["issues"].append(
                    f"Average response time {metrics['avg_response_time']:.3f}s exceeds threshold {thresholds['avg_response_time']}s"
                )
                analysis["recommendations"].append(
                    "Consider adding caching, optimizing database queries, or scaling horizontally"
                )
        
        # Check P95 response time
        if "p95_response_time" in metrics and "p95_response_time" in thresholds:
            if metrics["p95_response_time"] > thresholds["p95_response_time"]:
                if analysis["status"] == "PASS":
                    analysis["status"] = "WARN"
                analysis["issues"].append(
                    f"P95 response time {metrics['p95_response_time']:.3f}s exceeds threshold {thresholds['p95_response_time']}s"
                )
                analysis["recommendations"].append(
                    "Investigate slow queries and optimize tail latency"
                )
        
        # Check throughput
        if "requests_per_second" in metrics and "requests_per_second" in thresholds:
            if metrics["requests_per_second"] < thresholds["requests_per_second"]:
                if analysis["status"] == "PASS":
                    analysis["status"] = "WARN"
                analysis["issues"].append(
                    f"Throughput {metrics['requests_per_second']:.2f} req/s below target {thresholds['requests_per_second']} req/s"
                )
                analysis["recommendations"].append(
                    "Consider increasing worker count or optimizing request handling"
                )
        
        return analysis
    
    def generate_report(self) -> str:
        """Generate a comprehensive performance report."""
        report_lines = [
            "=" * 80,
            "LOAD TEST PERFORMANCE REPORT",
            "=" * 80,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            ""
        ]
        
        if not self.results:
            report_lines.append("No results to analyze")
            return "\n".join(report_lines)
        
        # Overall summary
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results.values() if r.get("status") == "PASS")
        failed_tests = sum(1 for r in self.results.values() if r.get("status") == "FAIL")
        warned_tests = sum(1 for r in self.results.values() if r.get("status") == "WARN")
        
        report_lines.extend([
            "OVERALL SUMMARY",
            "-" * 80,
            f"Total Tests: {total_tests}",
            f"Passed: {passed_tests}",
            f"Failed: {failed_tests}",
            f"Warnings: {warned_tests}",
            ""
        ])
        
        # Detailed results per endpoint
        report_lines.extend([
            "DETAILED RESULTS",
            "-" * 80,
            ""
        ])
        
        for endpoint_name, result in self.results.items():
            analysis = self.analyze_endpoint(endpoint_name, result)
            
            report_lines.extend([
                f"Endpoint: {endpoint_name.upper()}",
                f"Status: {analysis['status']}",
                ""
            ])
            
            # Metrics
            if "metrics" in analysis:
                report_lines.append("Metrics:")
                for key, value in analysis["metrics"].items():
                    if isinstance(value, float):
                        report_lines.append(f"  {key}: {value:.3f}")
                    else:
                        report_lines.append(f"  {key}: {value}")
                report_lines.append("")
            
            # Issues
            if analysis["issues"]:
                report_lines.append("Issues:")
                for issue in analysis["issues"]:
                    report_lines.append(f"  - {issue}")
                report_lines.append("")
            
            # Recommendations
            if analysis["recommendations"]:
                report_lines.append("Recommendations:")
                for rec in analysis["recommendations"]:
                    report_lines.append(f"  - {rec}")
                report_lines.append("")
            
            report_lines.append("-" * 80)
            report_lines.append("")
        
        # Overall recommendations
        if failed_tests > 0 or warned_tests > 0:
            report_lines.extend([
                "OVERALL RECOMMENDATIONS",
                "-" * 80,
                ""
            ])
            
            if failed_tests > 0:
                report_lines.extend([
                    "Critical Actions Required:",
                    "  1. Review error logs for failed tests",
                    "  2. Optimize slow endpoints before production deployment",
                    "  3. Consider infrastructure scaling",
                    ""
                ])
            
            if warned_tests > 0:
                report_lines.extend([
                    "Performance Improvements Suggested:",
                    "  1. Implement caching for frequently accessed data",
                    "  2. Add database indexes for common queries",
                    "  3. Monitor performance metrics in production",
                    ""
                ])
        else:
            report_lines.extend([
                "CONCLUSION",
                "-" * 80,
                "All load tests passed! System is ready for production deployment.",
                ""
            ])
        
        report_lines.append("=" * 80)
        
        return "\n".join(report_lines)
    
    def save_report(self, output_file: str):
        """Save report to file."""
        report = self.generate_report()
        with open(output_file, 'w') as f:
            f.write(report)
        print(f"Report saved to: {output_file}")
    
    def compare_with_baseline(self, baseline_file: str) -> Dict[str, Any]:
        """Compare current results with baseline."""
        try:
            with open(baseline_file, 'r') as f:
                baseline = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            print(f"Could not load baseline file: {baseline_file}")
            return {}
        
        comparison = {}
        
        for endpoint_name in self.results:
            if endpoint_name not in baseline:
                continue
            
            current = self.results[endpoint_name]
            base = baseline[endpoint_name]
            
            comparison[endpoint_name] = {
                "regression": False,
                "improvements": [],
                "degradations": []
            }
            
            # Compare metrics
            for metric in ["avg_response_time", "p95_response_time", "success_rate"]:
                if metric in current and metric in base:
                    current_val = current[metric]
                    base_val = base[metric]
                    
                    if metric == "success_rate":
                        # Higher is better
                        change_percent = ((current_val - base_val) / base_val) * 100
                        if change_percent < -5:  # 5% degradation
                            comparison[endpoint_name]["regression"] = True
                            comparison[endpoint_name]["degradations"].append(
                                f"{metric}: {change_percent:.2f}% decrease"
                            )
                        elif change_percent > 5:
                            comparison[endpoint_name]["improvements"].append(
                                f"{metric}: {change_percent:.2f}% increase"
                            )
                    else:
                        # Lower is better
                        change_percent = ((current_val - base_val) / base_val) * 100
                        if change_percent > 10:  # 10% degradation
                            comparison[endpoint_name]["regression"] = True
                            comparison[endpoint_name]["degradations"].append(
                                f"{metric}: {change_percent:.2f}% increase"
                            )
                        elif change_percent < -10:
                            comparison[endpoint_name]["improvements"].append(
                                f"{metric}: {change_percent:.2f}% decrease"
                            )
        
        return comparison


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python analyze_load_test_results.py <results_file> [baseline_file]")
        print("\nExample:")
        print("  python analyze_load_test_results.py test_results/load_test_results.json")
        print("  python analyze_load_test_results.py test_results/load_test_results.json test_results/baseline.json")
        sys.exit(1)
    
    results_file = sys.argv[1]
    baseline_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    analyzer = LoadTestAnalyzer(results_file)
    
    if not analyzer.load_results():
        sys.exit(1)
    
    # Generate and print report
    report = analyzer.generate_report()
    print(report)
    
    # Save report
    output_file = results_file.replace('.json', '_report.txt')
    analyzer.save_report(output_file)
    
    # Compare with baseline if provided
    if baseline_file:
        print("\n" + "=" * 80)
        print("BASELINE COMPARISON")
        print("=" * 80)
        
        comparison = analyzer.compare_with_baseline(baseline_file)
        
        for endpoint_name, comp in comparison.items():
            print(f"\n{endpoint_name.upper()}:")
            
            if comp["improvements"]:
                print("  Improvements:")
                for imp in comp["improvements"]:
                    print(f"    + {imp}")
            
            if comp["degradations"]:
                print("  Degradations:")
                for deg in comp["degradations"]:
                    print(f"    - {deg}")
            
            if comp["regression"]:
                print("  ⚠️  PERFORMANCE REGRESSION DETECTED")
            elif not comp["improvements"] and not comp["degradations"]:
                print("  No significant changes")


if __name__ == "__main__":
    main()

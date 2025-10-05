#!/usr/bin/env python3
"""
Deployment validation test runner.
Runs comprehensive deployment validation tests with proper reporting.
"""

import subprocess
import sys
import os
import time
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any


class DeploymentValidator:
    """Deployment validation test runner."""
    
    def __init__(self, environment: str = "development", verbose: bool = False):
        self.environment = environment
        self.verbose = verbose
        self.test_results = {}
        
    def run_pre_validation_checks(self) -> bool:
        """Run pre-validation checks before starting tests."""
        print("🔍 Running pre-validation checks...")
        
        # Check Docker and Docker Compose
        if not self._check_command("docker"):
            print("❌ Docker is not available")
            return False
            
        if not self._check_command("docker-compose"):
            print("❌ Docker Compose is not available")
            return False
        
        # Check if services are running
        result = subprocess.run(
            ["docker-compose", "ps"],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            print("❌ Docker Compose services not accessible")
            return False
        
        # Check if key services are up
        services_output = result.stdout
        required_services = ["api", "db", "redis"]
        
        for service in required_services:
            if service not in services_output or "Up" not in services_output:
                print(f"⚠️  Service {service} may not be running")
        
        print("✅ Pre-validation checks passed")
        return True
    
    def run_deployment_validation_tests(self) -> Dict[str, Any]:
        """Run deployment validation tests."""
        print("🧪 Running deployment validation tests...")
        
        test_command = [
            "python", "-m", "pytest",
            "tests/deployment/test_deployment_validation.py",
            "-v",
            "--tb=short",
            "--json-report",
            "--json-report-file=test_results/deployment_validation_report.json"
        ]
        
        if self.verbose:
            test_command.append("-s")
        
        # Set environment variables
        env = os.environ.copy()
        env["ENVIRONMENT"] = self.environment
        
        result = subprocess.run(test_command, env=env, capture_output=True, text=True)
        
        # Parse results
        test_results = {
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "passed": result.returncode == 0
        }
        
        # Try to load JSON report
        report_file = "test_results/deployment_validation_report.json"
        if os.path.exists(report_file):
            try:
                with open(report_file, 'r') as f:
                    json_report = json.load(f)
                    test_results["detailed_results"] = json_report
            except json.JSONDecodeError:
                pass
        
        self.test_results["deployment_validation"] = test_results
        return test_results
    
    def run_production_deployment_tests(self) -> Dict[str, Any]:
        """Run production-specific deployment tests."""
        print("🏭 Running production deployment tests...")
        
        test_command = [
            "python", "-m", "pytest",
            "tests/deployment/test_production_deployment.py",
            "-v",
            "--tb=short",
            "--json-report",
            "--json-report-file=test_results/production_deployment_report.json"
        ]
        
        if self.verbose:
            test_command.append("-s")
        
        # Set environment variables
        env = os.environ.copy()
        env["ENVIRONMENT"] = self.environment
        
        result = subprocess.run(test_command, env=env, capture_output=True, text=True)
        
        test_results = {
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "passed": result.returncode == 0
        }
        
        self.test_results["production_deployment"] = test_results
        return test_results
    
    def run_integration_tests(self) -> Dict[str, Any]:
        """Run deployment integration tests."""
        print("🔗 Running deployment integration tests...")
        
        test_command = [
            "python", "-m", "pytest",
            "tests/deployment/test_deployment_integration.py",
            "-v",
            "--tb=short",
            "--json-report",
            "--json-report-file=test_results/deployment_integration_report.json"
        ]
        
        if self.verbose:
            test_command.append("-s")
        
        # Set environment variables
        env = os.environ.copy()
        env["ENVIRONMENT"] = self.environment
        
        result = subprocess.run(test_command, env=env, capture_output=True, text=True)
        
        test_results = {
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "passed": result.returncode == 0
        }
        
        self.test_results["deployment_integration"] = test_results
        return test_results
    
    def run_all_tests(self) -> bool:
        """Run all deployment validation tests."""
        print(f"🚀 Starting deployment validation for environment: {self.environment}")
        
        # Ensure test results directory exists
        os.makedirs("test_results", exist_ok=True)
        
        # Run pre-validation checks
        if not self.run_pre_validation_checks():
            return False
        
        # Run all test suites
        all_passed = True
        
        # Basic deployment validation
        result = self.run_deployment_validation_tests()
        if not result["passed"]:
            all_passed = False
            print("❌ Deployment validation tests failed")
        else:
            print("✅ Deployment validation tests passed")
        
        # Production-specific tests (if in production environment)
        if self.environment == "production":
            result = self.run_production_deployment_tests()
            if not result["passed"]:
                all_passed = False
                print("❌ Production deployment tests failed")
            else:
                print("✅ Production deployment tests passed")
        
        # Integration tests
        result = self.run_integration_tests()
        if not result["passed"]:
            all_passed = False
            print("❌ Deployment integration tests failed")
        else:
            print("✅ Deployment integration tests passed")
        
        # Generate summary report
        self._generate_summary_report()
        
        if all_passed:
            print("🎉 All deployment validation tests passed!")
        else:
            print("❌ Some deployment validation tests failed")
        
        return all_passed
    
    def _check_command(self, command: str) -> bool:
        """Check if a command is available."""
        try:
            subprocess.run([command, "--version"], capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False
    
    def _generate_summary_report(self):
        """Generate a summary report of all test results."""
        summary = {
            "environment": self.environment,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "overall_status": "PASSED" if all(
                result.get("passed", False) for result in self.test_results.values()
            ) else "FAILED",
            "test_suites": {}
        }
        
        for suite_name, results in self.test_results.items():
            summary["test_suites"][suite_name] = {
                "status": "PASSED" if results.get("passed", False) else "FAILED",
                "return_code": results.get("returncode", -1)
            }
        
        # Save summary report
        summary_file = "test_results/deployment_validation_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"📊 Summary report saved to {summary_file}")
        
        # Print summary to console
        print("\n📋 Deployment Validation Summary:")
        print(f"Environment: {summary['environment']}")
        print(f"Overall Status: {summary['overall_status']}")
        print("Test Suites:")
        for suite_name, suite_results in summary["test_suites"].items():
            status_icon = "✅" if suite_results["status"] == "PASSED" else "❌"
            print(f"  {status_icon} {suite_name}: {suite_results['status']}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run deployment validation tests")
    parser.add_argument(
        "--environment", "-e",
        choices=["development", "production", "testing"],
        default="development",
        help="Deployment environment to validate"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    parser.add_argument(
        "--suite", "-s",
        choices=["validation", "production", "integration", "all"],
        default="all",
        help="Test suite to run"
    )
    
    args = parser.parse_args()
    
    validator = DeploymentValidator(
        environment=args.environment,
        verbose=args.verbose
    )
    
    if args.suite == "all":
        success = validator.run_all_tests()
    elif args.suite == "validation":
        validator.run_pre_validation_checks()
        result = validator.run_deployment_validation_tests()
        success = result["passed"]
    elif args.suite == "production":
        validator.run_pre_validation_checks()
        result = validator.run_production_deployment_tests()
        success = result["passed"]
    elif args.suite == "integration":
        validator.run_pre_validation_checks()
        result = validator.run_integration_tests()
        success = result["passed"]
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
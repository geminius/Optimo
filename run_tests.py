#!/usr/bin/env python3
"""
Main test runner script for the Robotics Model Optimization Platform.
Provides a convenient interface to run all test suites.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from tests.automation.test_runner import main

if __name__ == "__main__":
    main()

# Documentation Structure

This document provides an overview of the documentation organization for the Robotics Model Optimization Platform.

## Directory Structure

```
docs/
├── README.md                          # Main documentation index
├── STRUCTURE.md                       # This file - documentation organization
└── testing/                           # Testing documentation
    ├── README.md                      # Testing guide and quick start
    ├── TEST_SUITE_SUMMARY.md         # Complete test suite overview
    ├── TASK_17_VERIFICATION.md       # Implementation verification report
    └── TASK_STATUS_CONFIRMATION.md   # Status confirmation report
```

## Root Level Documentation

### Project Documentation (at project root)
- **README.md** - Main project documentation with quick start guide
- **DEPLOYMENT.md** - Deployment instructions and configuration
- **INTEGRATION_SUMMARY.md** - Integration details and architecture
- **GEMINI.md** - Gemini-specific documentation
- **run_tests.py** - Main test execution script

### Configuration Files (at project root)
- **pytest.ini** - Pytest configuration
- **requirements.txt** - Python dependencies
- **setup.py** - Package setup configuration
- **docker-compose.yml** - Docker orchestration
- **Makefile** - Build and development commands

## Documentation Categories

### 1. Getting Started
- [Main README](../README.md) - Project overview, features, quick start
- [Deployment Guide](../DEPLOYMENT.md) - How to deploy the platform

### 2. Testing
- [Testing Guide](./testing/README.md) - How to run and write tests
- [Test Suite Summary](./testing/TEST_SUITE_SUMMARY.md) - Complete test overview
- [Verification Report](./testing/TASK_17_VERIFICATION.md) - Implementation details

### 3. API Documentation
- OpenAPI docs available at `/docs` endpoint when running the API server
- Interactive API testing at `/redoc` endpoint

### 4. Architecture
- [Integration Summary](../INTEGRATION_SUMMARY.md) - System architecture and integration

## Quick Navigation

### For Users
1. Start with [README.md](../README.md) for project overview
2. Follow installation instructions
3. Check [DEPLOYMENT.md](../DEPLOYMENT.md) for deployment options
4. Use API documentation at `/docs` endpoint

### For Developers
1. Read [README.md](../README.md) for architecture overview
2. Review [Testing Guide](./testing/README.md) for testing practices
3. Check [Integration Summary](../INTEGRATION_SUMMARY.md) for system design
4. Explore code in `src/` directory

### For Testers
1. Start with [Testing Guide](./testing/README.md)
2. Review [Test Suite Summary](./testing/TEST_SUITE_SUMMARY.md)
3. Run tests using `python3 run_tests.py`
4. Check test reports in `test_results/` directory

### For DevOps
1. Review [DEPLOYMENT.md](../DEPLOYMENT.md)
2. Check Docker configuration files
3. Review CI/CD integration in testing docs
4. Monitor logs in `logs/` directory

## File Naming Conventions

### Documentation Files
- **README.md** - Main documentation for a directory
- **UPPERCASE.md** - Standalone documentation files
- **lowercase.md** - Supporting documentation files

### Test Files
- **test_*.py** - Unit test files
- **test_*_integration.py** - Integration test files
- **test_*_benchmarks.py** - Performance benchmark files

### Configuration Files
- **lowercase.yml** - YAML configuration files
- **lowercase.ini** - INI configuration files
- **.lowercase** - Hidden configuration files

## Documentation Standards

### Markdown Files
- Use clear headings (H1 for title, H2 for sections)
- Include code examples with syntax highlighting
- Add links to related documentation
- Keep line length reasonable (80-120 characters)
- Use bullet points for lists
- Include emojis for visual clarity (optional)

### Code Examples
- Always include language identifier for syntax highlighting
- Show complete, runnable examples
- Include comments for complex operations
- Show expected output when relevant

### Links
- Use relative links for internal documentation
- Use absolute links for external resources
- Check links are valid and not broken
- Provide context for links

## Maintenance

### Adding New Documentation
1. Determine appropriate category (testing, deployment, etc.)
2. Create file in appropriate directory
3. Update relevant README.md files
4. Add links from main documentation
5. Follow naming conventions

### Updating Documentation
1. Keep documentation in sync with code changes
2. Update version numbers and dates
3. Test all code examples
4. Verify all links still work
5. Update table of contents if needed

### Deprecating Documentation
1. Mark as deprecated with clear notice
2. Provide link to replacement documentation
3. Keep for at least one version cycle
4. Remove after sufficient notice period

## Contributing to Documentation

### Guidelines
- Write clear, concise documentation
- Use examples to illustrate concepts
- Keep documentation up-to-date with code
- Follow existing style and structure
- Test all code examples before committing

### Review Process
1. Create documentation in feature branch
2. Ensure all examples work
3. Check spelling and grammar
4. Verify links and references
5. Submit pull request for review

## Support

For documentation issues:
1. Check this structure guide
2. Review existing documentation
3. Search for similar issues
4. Create issue with specific details

## Version History

- **v1.0** (2025-10-05) - Initial documentation structure
  - Organized testing documentation
  - Created main docs index
  - Established structure guidelines

# Quick Reference Guide

## 🎯 Most Common Tasks

### Running Tests

```bash
# Run all tests
python3 run_tests.py --suite all

# Run specific suite
python3 run_tests.py --suite integration

# Generate reports
python3 run_tests.py --suite all --formats json html txt
```

### Reading Documentation

| What You Need | Where to Look |
|---------------|---------------|
| Project overview | [README.md](../README.md) |
| Testing guide | [testing/README.md](testing/README.md) |
| Test details | [testing/TEST_SUITE_SUMMARY.md](testing/TEST_SUITE_SUMMARY.md) |
| Deployment | [DEPLOYMENT.md](../DEPLOYMENT.md) |
| Documentation structure | [STRUCTURE.md](STRUCTURE.md) |

### Finding Information

| Question | Answer |
|----------|--------|
| How do I run tests? | `python3 run_tests.py --suite all` |
| Where are test docs? | `docs/testing/` directory |
| How do I add tests? | See [testing/README.md](testing/README.md) |
| What tests exist? | See [testing/TEST_SUITE_SUMMARY.md](testing/TEST_SUITE_SUMMARY.md) |
| How is this organized? | See [STRUCTURE.md](STRUCTURE.md) |

## 📂 File Locations

### At Project Root
- `README.md` - Main project documentation
- `run_tests.py` - Test execution script
- `DEPLOYMENT.md` - Deployment guide
- `ORGANIZATION_SUMMARY.md` - Organization changes

### In docs/
- `docs/README.md` - Documentation index
- `docs/STRUCTURE.md` - Organization guide
- `docs/QUICK_REFERENCE.md` - This file

### In docs/testing/
- `docs/testing/README.md` - Testing guide
- `docs/testing/TEST_SUITE_SUMMARY.md` - Test overview
- `docs/testing/TASK_17_VERIFICATION.md` - Verification report
- `docs/testing/TASK_STATUS_CONFIRMATION.md` - Status report

## 🚀 Quick Commands

### Testing
```bash
# All tests
python3 run_tests.py --suite all

# Unit tests only
python3 run_tests.py --suite unit

# Integration tests
python3 run_tests.py --suite integration

# Performance benchmarks
python3 run_tests.py --suite performance

# Stress tests
python3 run_tests.py --suite stress

# With pytest
pytest
pytest -m integration
pytest --quick
pytest --cov=src
```

### Development
```bash
# Format code
black src/ tests/

# Run linting
flake8 src/ tests/

# Type checking
mypy src/

# Start API server
uvicorn src.api.main:app --reload

# Start frontend
cd frontend && npm start
```

### Docker
```bash
# Development
docker-compose -f docker-compose.dev.yml up

# Production
docker-compose -f docker-compose.prod.yml up

# Build
docker-compose build
```

## 📖 Documentation Paths

### For New Users
1. [README.md](../README.md) - Start here
2. Installation section
3. Quick start examples
4. [DEPLOYMENT.md](../DEPLOYMENT.md) - Deployment options

### For Developers
1. [README.md](../README.md) - Architecture overview
2. [testing/README.md](testing/README.md) - Testing practices
3. Source code in `src/`
4. Examples in `examples/`

### For Testers
1. [testing/README.md](testing/README.md) - Testing guide
2. [testing/TEST_SUITE_SUMMARY.md](testing/TEST_SUITE_SUMMARY.md) - Test details
3. Run `python3 run_tests.py --suite all`
4. Check `test_results/` for reports

### For DevOps
1. [DEPLOYMENT.md](../DEPLOYMENT.md) - Deployment guide
2. Docker files at root
3. Configuration in `config/`
4. Monitoring at `/monitoring` endpoint

## 🔍 Search Tips

### Finding Files
```bash
# Find test files
find tests/ -name "*.py"

# Find documentation
find docs/ -name "*.md"

# Find configuration
find . -name "*.yml" -o -name "*.yaml"
```

### Searching Content
```bash
# Search in tests
grep -r "test_name" tests/

# Search in docs
grep -r "keyword" docs/

# Search in source
grep -r "function_name" src/
```

## 💡 Tips

### Testing
- Use `--quick` to skip slow tests during development
- Check `test_results/` for detailed reports
- Use markers: `-m integration`, `-m performance`, `-m stress`
- Run specific files: `pytest tests/test_specific.py`

### Documentation
- All test docs are in `docs/testing/`
- Main docs are at project root
- Use relative links in documentation
- Keep docs in sync with code

### Development
- Follow existing code structure
- Add tests for new features
- Update documentation
- Run tests before committing

## 🆘 Troubleshooting

### Tests Not Running
```bash
# Check dependencies
pip install -r requirements.txt

# Check pytest
pytest --version

# Check test script
python3 run_tests.py --help
```

### Documentation Not Found
```bash
# Check docs directory
ls -la docs/

# Check testing docs
ls -la docs/testing/

# Verify organization
cat docs/STRUCTURE.md
```

### Import Errors
```bash
# Install package
pip install -e .

# Check Python path
python3 -c "import sys; print(sys.path)"

# Verify installation
pip list | grep robotics
```

## 📞 Getting Help

1. **Check documentation** - Start with relevant README
2. **Search issues** - Look for similar problems
3. **Review examples** - Check `examples/` directory
4. **Read error messages** - They often contain solutions
5. **Create issue** - Provide detailed information

## 🔗 Important Links

- [Main README](../README.md)
- [Testing Guide](testing/README.md)
- [Test Suite Summary](testing/TEST_SUITE_SUMMARY.md)
- [Deployment Guide](../DEPLOYMENT.md)
- [Documentation Structure](STRUCTURE.md)
- [Organization Summary](../ORGANIZATION_SUMMARY.md)

---

**Last Updated:** October 5, 2025

For more detailed information, see the full documentation in each section.

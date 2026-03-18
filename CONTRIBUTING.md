# Contributing to xModelFactory

Thank you for your interest in contributing to xModelFactory! This document provides guidelines and instructions for contributing.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [How to Contribute](#how-to-contribute)
- [Pull Request Process](#pull-request-process)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Documentation](#documentation)

## Code of Conduct

This project and everyone participating in it is governed by our commitment to providing a welcoming and inclusive environment for all contributors.

## Getting Started

1. Fork the repository
2. Clone your fork locally
3. Set up your development environment
4. Make your changes
5. Submit a pull request

## Development Setup

### Prerequisites

- Python 3.8 or higher
- pip or conda
- Git

### Installation

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/xModelFactory.git
cd xModelFactory

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"

# Install pre-commit hooks (optional)
pre-commit install
```

### Optional Dependencies

```bash
# For DeepSpeed support
pip install deepspeed

# For Lion optimizer
pip install lion-pytorch

# For all dependencies
pip install -e ".[all]"
```

## How to Contribute

### Reporting Bugs

1. Check if the bug has already been reported in [Issues](https://github.com/yourusername/xModelFactory/issues)
2. If not, create a new issue with:
   - Clear description of the bug
   - Steps to reproduce
   - Expected vs actual behavior
   - Your environment (Python version, PyTorch version, OS)

### Suggesting Features

1. Open an issue with the label `enhancement`
2. Describe the feature and its use case
3. Discuss with maintainers before implementation

### Contributing Code

1. Pick an issue to work on (or propose a new one)
2. Create a feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. Make your changes
4. Write/update tests
5. Update documentation if needed
6. Submit a pull request

## Pull Request Process

1. Ensure your code passes all tests
2. Update the README.md if needed
3. Update documentation for any new features
4. Follow the PR template
5. Request review from maintainers

### PR Checklist

- [ ] Code follows the project's coding standards
- [ ] All tests pass
- [ ] New features have corresponding tests
- [ ] Documentation is updated
- [ ] Commit messages are clear and descriptive

## Coding Standards

### Python Style

- Follow PEP 8 guidelines
- Use Black for code formatting:
  ```bash
  black xmodel_factory/
  ```
- Use type hints for function signatures
- Keep line length to 100 characters

### Docstrings

Use Google-style docstrings:

```python
def function_name(param1: str, param2: int) -> bool:
    """Short description of function.

    Longer description if needed.

    Args:
        param1: Description of first parameter.
        param2: Description of second parameter.

    Returns:
        Description of return value.

    Raises:
        ValueError: When invalid input is provided.
    """
    pass
```

### Code Organization

- One class/module per file for major components
- Keep related functions together
- Avoid circular imports

## Testing

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_basic.py

# Run with coverage
pytest tests/ --cov=xmodel_factory
```

### Writing Tests

- Place tests in the `tests/` directory
- Name test files as `test_*.py`
- Use descriptive test function names
- Test both success and failure cases

Example:

```python
def test_model_config_creation():
    """Test that ModelConfig can be created with valid parameters."""
    config = ModelConfig(
        vocab_size=32000,
        hidden_size=512,
    )
    assert config.vocab_size == 32000
    assert config.hidden_size == 512
```

## Documentation

### Building Documentation

```bash
cd docs/
make html
```

### Documentation Guidelines

- Keep README.md up to date
- Document all public APIs
- Include usage examples
- Update CHANGELOG.md for significant changes

## Release Process

1. Update version in `xmodel_factory/__init__.py`
2. Update CHANGELOG.md
3. Create a GitHub release
4. Build and publish to PyPI:
   ```bash
   python -m build
   twine upload dist/*
   ```

## Questions?

Feel free to open an issue for questions or reach out to the maintainers.

Thank you for contributing to xModelFactory!
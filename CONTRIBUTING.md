# Contributing to Fashion Search Engine

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## 🚀 Getting Started

1. **Fork the repository**
   ```bash
   git clone https://github.com/yourusername/fashion-search-engine.git
   cd fashion-search-engine
   ```

2. **Set up development environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   pip install -e .[dev]
   ```

3. **Create a branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

## 📝 Code Style

We follow PEP 8 guidelines with the following tools:

### Formatting
```bash
# Format code with black
black models/ data_loading/ training/ inference/

# Check with flake8
flake8 models/ data_loading/ training/ inference/

# Type checking with mypy
mypy models/ data_loading/ training/ inference/
```

### Style Guidelines

- **Imports**: Use absolute imports, organize in groups (standard library, third-party, local)
- **Docstrings**: Use Google-style docstrings for all public functions and classes
- **Type hints**: Add type hints to all function signatures
- **Line length**: Maximum 100 characters (black default)
- **Naming**:
  - Classes: `PascalCase`
  - Functions/variables: `snake_case`
  - Constants: `UPPER_CASE`
  - Private: `_leading_underscore`

### Example

```python
from typing import Tuple
import torch
import torch.nn as nn


class MyModule(nn.Module):
    """
    Brief description of the module.
    
    Longer description with details about the implementation,
    architecture, or important notes.
    
    Args:
        input_dim: Dimension of input features.
        output_dim: Dimension of output features.
        
    Example:
        >>> model = MyModule(input_dim=64, output_dim=10)
        >>> x = torch.randn(8, 64)
        >>> output = model(x)
    """
    
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self._input_dim = input_dim
        self.linear = nn.Linear(input_dim, output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the module."""
        return self.linear(x)
```

## 🧪 Testing

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_models.py

# Run with coverage
pytest --cov=models --cov=training --cov=inference tests/

# Run with verbose output
pytest -v tests/
```

### Writing Tests

- Place tests in the `tests/` directory
- Name test files `test_*.py`
- Name test functions `test_*`
- Use descriptive test names
- Include docstrings explaining what is being tested

Example:
```python
import pytest
import torch
from models import MobileNetBackbone


def test_backbone_output_shape():
    """Test that backbone produces correct output shape."""
    backbone = MobileNetBackbone()
    x = torch.randn(4, 3, 64, 64)
    output = backbone(x)
    assert output.shape == (4, 64, 4, 4)


def test_backbone_with_different_input_sizes():
    """Test backbone with various input sizes."""
    backbone = MobileNetBackbone()
    test_cases = [
        (1, 3, 64, 64),
        (8, 3, 128, 128),
        (16, 3, 224, 224),
    ]
    for input_shape in test_cases:
        x = torch.randn(*input_shape)
        output = backbone(x)
        assert output.shape[0] == input_shape[0]
```

## 📚 Documentation

### Code Documentation

- Add docstrings to all public classes and functions
- Use type hints for all function parameters and return values
- Include usage examples in docstrings where helpful

### README Updates

If your contribution affects user-facing features:
- Update the main README.md
- Add examples if introducing new functionality
- Update the table of contents if needed

## 🔄 Pull Request Process

1. **Ensure your code passes all checks**
   ```bash
   black .
   flake8 .
   mypy .
   pytest tests/
   ```

2. **Update documentation**
   - Add/update docstrings
   - Update README if needed
   - Add examples if relevant

3. **Write a clear PR description**
   - What does this PR do?
   - Why is this change needed?
   - How was it tested?
   - Any breaking changes?

4. **Link related issues**
   - Reference issue numbers: `Fixes #123`
   - Explain how the PR addresses the issue

5. **Request review**
   - Tag relevant reviewers
   - Respond to feedback promptly
   - Make requested changes

### PR Template

```markdown
## Description
Brief description of changes

## Motivation
Why is this change needed?

## Changes
- Change 1
- Change 2

## Testing
How was this tested?

## Checklist
- [ ] Code follows style guidelines
- [ ] Tests pass locally
- [ ] Documentation updated
- [ ] No breaking changes (or documented)
```

## 🐛 Bug Reports

### Before Submitting

1. Check existing issues
2. Try the latest version
3. Isolate the problem

### Bug Report Template

```markdown
## Bug Description
Clear description of the bug

## To Reproduce
Steps to reproduce:
1. ...
2. ...

## Expected Behavior
What should happen

## Actual Behavior
What actually happens

## Environment
- OS: 
- Python version:
- PyTorch version:
- CUDA version (if applicable):

## Additional Context
Screenshots, error messages, etc.
```

## 💡 Feature Requests

### Feature Request Template

```markdown
## Feature Description
Clear description of the feature

## Motivation
Why is this feature needed?
What problem does it solve?

## Proposed Solution
How should it work?

## Alternatives Considered
Other approaches you've thought about

## Additional Context
Examples, mockups, etc.
```

## 🎯 Areas for Contribution

### High Priority
- [ ] Add more data augmentation strategies
- [ ] Implement hard negative mining
- [ ] Add ONNX export functionality
- [ ] Create web API with FastAPI
- [ ] Add more visualization utilities

### Documentation
- [ ] Tutorial notebooks
- [ ] Architecture deep dives
- [ ] Training best practices guide
- [ ] Deployment guide

### Testing
- [ ] Increase test coverage
- [ ] Add integration tests
- [ ] Benchmark scripts

### Models
- [ ] Additional backbone architectures
- [ ] Alternative loss functions
- [ ] Multi-task learning extensions

## 📞 Communication

- **Issues**: For bugs and feature requests
- **Discussions**: For questions and ideas
- **Email**: For private matters

## 🙏 Recognition

Contributors will be:
- Listed in the README
- Credited in release notes
- Acknowledged in the project

Thank you for contributing to Fashion Search Engine! 🎉

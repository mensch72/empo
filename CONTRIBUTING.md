# Contributing to EMPO

Thank you for your interest in contributing to EMPO! This document provides guidelines for contributing to the project.

## Development Setup

### 1. Fork and Clone

```bash
git clone https://github.com/yourusername/empo.git
cd empo
```

### 2. Set Up Development Environment

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your settings
vim .env

# Start development environment
make up

# Enter container
make shell
```

### 3. Verify Setup

```bash
# Run verification script
bash scripts/verify_setup.sh

# Run tests
python tests/test_structure.py
```

## Development Workflow

### Making Changes

1. Create a feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes in the container:
   ```bash
   make shell
   # Edit files, test changes
   ```

3. Test your changes:
   ```bash
   # Run tests
   pytest tests/ -v
   
   # Run linters
   ruff check .
   black .
   
   # Test training script
   python train.py --num-episodes 10
   ```

4. Commit your changes:
   ```bash
   git add .
   git commit -m "feat: Add your feature description"
   ```

5. Push and create a pull request:
   ```bash
   git push origin feature/your-feature-name
   ```

## Code Style

### Python

- Follow PEP 8 style guide
- Use type hints where appropriate
- Write docstrings for public functions/classes
- Format code with `black`
- Lint with `ruff`

Example:
```python
def train_agent(
    env_name: str,
    num_episodes: int,
    learning_rate: float = 0.001
) -> Dict[str, Any]:
    """
    Train a reinforcement learning agent.
    
    Args:
        env_name: Name of the environment
        num_episodes: Number of training episodes
        learning_rate: Learning rate for optimizer
        
    Returns:
        Dictionary containing training metrics
    """
    # Implementation
    pass
```

### Shell Scripts

- Use `#!/bin/bash` shebang
- Add error handling with `set -e`
- Include comments for complex logic
- Test scripts with `bash -n script.sh`

### YAML/Configuration

- Use 2 spaces for indentation
- Include comments for non-obvious settings
- Validate with `yamllint` if available

## Testing

### Writing Tests

Place tests in the `tests/` directory:

```python
# tests/test_feature.py
import pytest
from empo import feature


def test_feature_functionality():
    """Test that feature works correctly."""
    result = feature.do_something()
    assert result == expected_value
```

### Running Tests

```bash
# In container
pytest tests/ -v

# With coverage
pytest tests/ --cov=src/empo --cov-report=html
```

## Documentation

### Docstrings

Use Google-style docstrings:

```python
def function_name(arg1: str, arg2: int) -> bool:
    """
    Short description.
    
    Longer description if needed.
    
    Args:
        arg1: Description of arg1
        arg2: Description of arg2
        
    Returns:
        Description of return value
        
    Raises:
        ValueError: When invalid input is provided
        
    Example:
        >>> result = function_name("test", 42)
        >>> print(result)
        True
    """
```

### README Updates

If your changes affect setup or usage:
- Update README.md
- Update QUICKSTART.md if needed
- Add examples for new features

## Docker and Deployment

### Testing Docker Changes

```bash
# Test build
docker build -t empo:test .

# Test with compose
docker compose up --build

# Verify GPU support
docker compose exec empo-dev nvidia-smi
```

### Testing Singularity Changes

If you have Singularity/Apptainer installed:

```bash
# Build SIF
apptainer build empo.sif Dockerfile

# Test execution
apptainer exec empo.sif python --version

# Test with GPU
apptainer exec --nv empo.sif nvidia-smi
```

## Pull Request Guidelines

### Before Submitting

- [ ] Code follows style guidelines
- [ ] Tests pass
- [ ] Documentation is updated
- [ ] Commits are meaningful and well-formatted
- [ ] Changes work in both Docker and Singularity (if applicable)

### PR Description

Include:
1. **What**: Brief description of changes
2. **Why**: Motivation and context
3. **How**: Implementation approach
4. **Testing**: How you tested the changes
5. **Screenshots**: If UI/output changes

### Commit Messages

Follow conventional commits:

- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `style:` Code style changes (formatting)
- `refactor:` Code refactoring
- `test:` Adding or updating tests
- `chore:` Maintenance tasks

Examples:
```
feat: Add support for PPO algorithm
fix: Resolve GPU memory leak in training loop
docs: Update cluster deployment instructions
```

## Getting Help

- **Issues**: Open an issue on GitHub
- **Discussions**: Use GitHub Discussions for questions
- **Pull Requests**: Tag maintainers for review

## Code Review Process

1. Automated checks run on PR
2. Maintainer reviews code
3. Address feedback
4. Approval and merge

## License

By contributing, you agree that your contributions will be licensed under the project's license.

## Questions?

Feel free to open an issue or discussion if you have any questions about contributing!

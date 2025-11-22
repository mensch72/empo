#!/usr/bin/env python3
"""
Simple test script that can run without pytest.
Verifies basic repository structure and imports.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def test_import_empo():
    """Test that the empo package can be imported."""
    try:
        import empo
        assert empo.__version__ == "0.1.0"
        print("✓ empo package imports correctly")
        return True
    except Exception as e:
        print(f"✗ Failed to import empo: {e}")
        return False


def test_train_script_exists():
    """Test that train.py exists."""
    train_script = Path(__file__).parent.parent / "train.py"
    if train_script.exists():
        print("✓ train.py exists")
        return True
    else:
        print("✗ train.py not found")
        return False


def test_config_exists():
    """Test that default config exists."""
    config_file = Path(__file__).parent.parent / "configs" / "default.yaml"
    if config_file.exists():
        print("✓ configs/default.yaml exists")
        return True
    else:
        print("✗ configs/default.yaml not found")
        return False


def test_requirements_exist():
    """Test that requirement files exist."""
    req_file = Path(__file__).parent.parent / "requirements.txt"
    req_dev_file = Path(__file__).parent.parent / "requirements-dev.txt"
    if req_file.exists() and req_dev_file.exists():
        print("✓ requirements files exist")
        return True
    else:
        print("✗ requirements files not found")
        return False


def test_dockerfile_exists():
    """Test that Dockerfile exists."""
    dockerfile = Path(__file__).parent.parent / "Dockerfile"
    if dockerfile.exists():
        print("✓ Dockerfile exists")
        return True
    else:
        print("✗ Dockerfile not found")
        return False


def test_docker_compose_exists():
    """Test that docker-compose.yml exists."""
    compose_file = Path(__file__).parent.parent / "docker-compose.yml"
    if compose_file.exists():
        print("✓ docker-compose.yml exists")
        return True
    else:
        print("✗ docker-compose.yml not found")
        return False


def main():
    """Run all tests."""
    print("=" * 50)
    print("Running EMPO Repository Tests")
    print("=" * 50)
    print()
    
    tests = [
        test_import_empo,
        test_train_script_exists,
        test_config_exists,
        test_requirements_exist,
        test_dockerfile_exists,
        test_docker_compose_exists,
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print()
    print("=" * 50)
    passed = sum(results)
    total = len(results)
    print(f"Results: {passed}/{total} tests passed")
    print("=" * 50)
    
    return all(results)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

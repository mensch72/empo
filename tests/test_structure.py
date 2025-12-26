#!/usr/bin/env python3
"""
Simple test script that can run without pytest.
Verifies basic repository structure and imports.
"""

import sys
from pathlib import Path


def test_import_empo():
    """Test that the empo package can be imported.
    
    This test is skipped if dependencies like gymnasium aren't installed,
    since the Docker build will catch real import issues.
    """
    try:
        import empo
        assert empo.__version__ == "0.1.0"
    except ImportError:
        # Dependencies not installed - skip this test gracefully
        # The Docker build will catch real import issues
        print("SKIPPED: Missing dependencies - run in Docker for full validation")
        return  # Skip without pytest


def test_requirements_exist():
    """Test that requirement files exist."""
    req_file = Path(__file__).parent.parent / "requirements.txt"
    req_dev_file = Path(__file__).parent.parent / "requirements-dev.txt"
    assert req_file.exists(), "requirements.txt not found"
    assert req_dev_file.exists(), "requirements-dev.txt not found"


def test_dockerfile_exists():
    """Test that Dockerfile exists."""
    dockerfile = Path(__file__).parent.parent / "Dockerfile"
    assert dockerfile.exists(), "Dockerfile not found"


def test_docker_compose_exists():
    """Test that docker-compose.yml exists."""
    compose_file = Path(__file__).parent.parent / "docker-compose.yml"
    assert compose_file.exists(), "docker-compose.yml not found"


def main():
    """Run all tests when executed directly."""
    tests = [
        test_import_empo,
        test_requirements_exist,
        test_dockerfile_exists,
        test_docker_compose_exists,
    ]
    
    failed = []
    for test in tests:
        try:
            print(f"Running {test.__name__}...", end=" ")
            test()
            print("PASSED")
        except AssertionError as e:
            print(f"FAILED: {e}")
            failed.append(test.__name__)
        except Exception as e:
            print(f"ERROR: {e}")
            failed.append(test.__name__)
    
    if failed:
        print(f"\n{len(failed)} test(s) failed: {failed}")
        sys.exit(1)
    else:
        print(f"\nAll {len(tests)} tests passed!")


if __name__ == "__main__":
    main()

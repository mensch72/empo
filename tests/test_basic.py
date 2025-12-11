"""
Basic tests for the EMPO framework.

Run with: pytest tests/
"""

import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def test_import_empo():
    """Test that the empo package can be imported."""
    import empo
    assert empo.__version__ == "0.1.0"


def test_requirements_exist():
    """Test that requirement files exist."""
    req_file = Path(__file__).parent.parent / "requirements.txt"
    req_dev_file = Path(__file__).parent.parent / "requirements-dev.txt"
    assert req_file.exists()
    assert req_dev_file.exists()


def test_dockerfile_exists():
    """Test that Dockerfile exists."""
    dockerfile = Path(__file__).parent.parent / "Dockerfile"
    assert dockerfile.exists()


def test_docker_compose_exists():
    """Test that docker-compose.yml exists."""
    compose_file = Path(__file__).parent.parent / "docker-compose.yml"
    assert compose_file.exists()


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__, "-v"])

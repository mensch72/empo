#!/usr/bin/env python3
"""
Test script to verify MineRL installation.

This script checks that MineRL can be imported and that the basic
environment infrastructure is available.

Run with: python tests/test_minerl_installation.py
Or with pytest: pytest tests/test_minerl_installation.py -v

Note: This test requires the hierarchical dependencies to be installed:
    pip install -r requirements-hierarchical.txt
"""

import sys


def test_minerl_import():
    """Test that minerl can be imported."""
    try:
        import minerl
        print(f"✓ MineRL version: {minerl.__version__}")
        return True
    except ImportError as e:
        print(f"✗ Failed to import minerl: {e}")
        print("  Install with: pip install -r requirements-hierarchical.txt")
        return False


def test_minerl_envs_available():
    """Test that MineRL environments are registered."""
    try:
        import minerl
        import gymnasium as gym
        
        # Check if any MineRL environments are registered
        env_specs = [spec for spec in gym.envs.registry.keys() if 'MineRL' in spec]
        if env_specs:
            print(f"✓ Found {len(env_specs)} MineRL environments registered")
            for spec in env_specs[:5]:  # Show first 5
                print(f"  - {spec}")
            if len(env_specs) > 5:
                print(f"  ... and {len(env_specs) - 5} more")
            return True
        else:
            print("⚠ No MineRL environments found in gym registry")
            print("  This might be expected if environments need explicit loading")
            return True  # Not a failure, just a warning
    except Exception as e:
        print(f"✗ Error checking MineRL environments: {e}")
        return False


def test_ollama_import():
    """Test that ollama can be imported."""
    try:
        import ollama
        print(f"✓ Ollama Python client imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Failed to import ollama: {e}")
        print("  Install with: pip install -r requirements-hierarchical.txt")
        return False


def main():
    """Run all MineRL installation tests."""
    print("=" * 50)
    print("MineRL Installation Test")
    print("=" * 50)
    print()
    
    results = []
    
    print("1. Testing MineRL import...")
    results.append(test_minerl_import())
    print()
    
    print("2. Testing MineRL environments...")
    results.append(test_minerl_envs_available())
    print()
    
    print("3. Testing Ollama import...")
    results.append(test_ollama_import())
    print()
    
    print("=" * 50)
    if all(results):
        print("✓ All tests passed!")
        return 0
    else:
        print("✗ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())

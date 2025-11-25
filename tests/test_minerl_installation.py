#!/usr/bin/env python3
"""
Test script to verify MineRL and Ollama installation and integration.

This script checks that:
1. MineRL can be imported and environments are available
2. Ollama client can connect to the Ollama server
3. A screenshot can be captured from Minecraft and sent to a vision LLM

Run with: python tests/test_minerl_installation.py
Or with pytest: pytest tests/test_minerl_installation.py -v

For the full integration test (--integration flag):
    1. Start Ollama: make up-hierarchical
    2. Pull the vision model: docker exec ollama ollama pull qwen2.5-vl:3b
    3. Run: python tests/test_minerl_installation.py --integration

Note: This test requires the hierarchical dependencies to be installed:
    pip install -r requirements-hierarchical.txt
"""

import argparse
import io
import os
import sys


# =============================================================================
# Configuration Constants
# =============================================================================

# Default Ollama server host
DEFAULT_OLLAMA_HOST = "http://localhost:11434"

# Default vision model for Ollama
DEFAULT_VISION_MODEL = "qwen2.5-vl:3b"

# Default MineRL environment to use for testing
DEFAULT_MINERL_ENV = "MineRLNavigateDense-v0"

# Number of random actions to take to get an interesting Minecraft scene
# This allows the agent to move around a bit and see something other than spawn point
NUM_WARMUP_STEPS = 10


# =============================================================================
# Basic Import Tests
# =============================================================================

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
        print("✓ Ollama Python client imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Failed to import ollama: {e}")
        print("  Install with: pip install -r requirements-hierarchical.txt")
        return False


def test_ollama_connection():
    """Test that we can connect to the Ollama server."""
    try:
        import ollama
        
        # Get Ollama host from environment or use default
        ollama_host = os.environ.get("OLLAMA_HOST", DEFAULT_OLLAMA_HOST)
        print(f"  Connecting to Ollama at {ollama_host}...")
        
        client = ollama.Client(host=ollama_host)
        models = client.list()
        
        model_names = [m.get("name", m.get("model", "unknown")) for m in models.get("models", [])]
        if model_names:
            print(f"✓ Connected to Ollama. Available models: {', '.join(model_names)}")
        else:
            print("✓ Connected to Ollama. No models installed yet.")
            print(f"  Pull a model with: docker exec ollama ollama pull {DEFAULT_VISION_MODEL}")
        return True
    except Exception as e:
        print(f"✗ Failed to connect to Ollama: {e}")
        print("  Make sure Ollama is running: make up-hierarchical")
        return False


# =============================================================================
# Integration Tests
# =============================================================================

def test_minerl_screenshot():
    """Test capturing a screenshot from MineRL environment."""
    try:
        import minerl
        import gymnasium as gym
        import numpy as np
        
        print("  Creating MineRL environment (this may take a moment)...")
        
        # Try the default environment, fall back to any available MineRL env
        env_name = os.environ.get("MINERL_ENV", DEFAULT_MINERL_ENV)
        try:
            env = gym.make(env_name)
        except gym.error.Error:
            # Fall back to finding any available MineRL environment
            minerl_envs = [spec for spec in gym.envs.registry.keys() if 'MineRL' in spec]
            if not minerl_envs:
                print(f"✗ No MineRL environments available")
                return None
            env_name = minerl_envs[0]
            print(f"  Falling back to {env_name}")
            env = gym.make(env_name)
        
        print(f"  Using environment: {env_name}")
        print("  Resetting environment...")
        obs, info = env.reset()
        
        # Take random actions to get an interesting frame (move around from spawn)
        for _ in range(NUM_WARMUP_STEPS):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                obs, info = env.reset()
        
        # Extract the POV (point of view) image
        if isinstance(obs, dict) and "pov" in obs:
            screenshot = obs["pov"]
        elif isinstance(obs, np.ndarray):
            screenshot = obs
        else:
            print(f"✗ Unexpected observation format: {type(obs)}")
            env.close()
            return None
        
        print(f"✓ Captured screenshot: shape={screenshot.shape}, dtype={screenshot.dtype}")
        
        env.close()
        return screenshot
        
    except Exception as e:
        print(f"✗ Failed to capture MineRL screenshot: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_vision_llm_description(screenshot):
    """Send screenshot to Ollama vision LLM and get description."""
    try:
        import ollama
        from PIL import Image
        import numpy as np
        
        ollama_host = os.environ.get("OLLAMA_HOST", DEFAULT_OLLAMA_HOST)
        model_name = os.environ.get("OLLAMA_VISION_MODEL", DEFAULT_VISION_MODEL)
        
        # Check if the model is available
        client = ollama.Client(host=ollama_host)
        models = client.list()
        model_names = [m.get("name", m.get("model", "")) for m in models.get("models", [])]
        
        if not any(model_name in name for name in model_names):
            print(f"✗ Vision model '{model_name}' not found in Ollama")
            print(f"  Available models: {', '.join(model_names) if model_names else 'none'}")
            print(f"  Pull it with: docker exec ollama ollama pull {model_name}")
            return False
        
        print(f"  Sending screenshot to {model_name} via {ollama_host}...")
        
        # Convert numpy array to PIL Image, then to bytes
        if isinstance(screenshot, np.ndarray):
            # MineRL images are typically RGB
            img = Image.fromarray(screenshot.astype("uint8"), "RGB")
        else:
            img = screenshot
        
        # Save to bytes buffer
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        image_bytes = buffer.getvalue()
        
        response = client.chat(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": "Describe what you see in this Minecraft screenshot in 2-3 sentences.",
                    "images": [image_bytes],
                }
            ],
        )
        
        description = response["message"]["content"]
        print(f"✓ LLM description received:")
        print(f"  \"{description}\"")
        return True
        
    except Exception as e:
        print(f"✗ Failed to get LLM description: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_basic_tests():
    """Run basic import tests only."""
    print("=" * 60)
    print("MineRL & Ollama Installation Test (Basic)")
    print("=" * 60)
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
    
    print("=" * 60)
    if all(results):
        print("✓ All basic tests passed!")
        return 0
    else:
        print("✗ Some tests failed")
        return 1


def run_integration_tests():
    """Run full integration tests including MineRL environment and Ollama vision."""
    print("=" * 60)
    print("MineRL & Ollama Integration Test")
    print("=" * 60)
    print()
    
    results = []
    
    print("1. Testing MineRL import...")
    results.append(test_minerl_import())
    print()
    
    print("2. Testing Ollama import...")
    results.append(test_ollama_import())
    print()
    
    print("3. Testing Ollama connection...")
    ollama_ok = test_ollama_connection()
    results.append(ollama_ok)
    print()
    
    print("4. Testing MineRL screenshot capture...")
    screenshot = test_minerl_screenshot()
    results.append(screenshot is not None)
    print()
    
    if screenshot is not None and ollama_ok:
        print("5. Testing vision LLM description...")
        results.append(test_vision_llm_description(screenshot))
        print()
    else:
        print("5. Skipping vision LLM test (prerequisites failed)")
        results.append(False)
        print()
    
    print("=" * 60)
    if all(results):
        print("✓ All integration tests passed!")
        return 0
    else:
        print("✗ Some tests failed")
        return 1


def main():
    """Run MineRL and Ollama tests."""
    parser = argparse.ArgumentParser(
        description="Test MineRL and Ollama installation and integration"
    )
    parser.add_argument(
        "--integration",
        action="store_true",
        help="Run full integration test (requires Ollama server and MineRL environment)",
    )
    args = parser.parse_args()
    
    if args.integration:
        return run_integration_tests()
    else:
        return run_basic_tests()


if __name__ == "__main__":
    sys.exit(main())

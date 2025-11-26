#!/usr/bin/env python3
"""
Test script to verify MineLand and Ollama installation and integration.

This script checks that:
1. MineLand can be imported and environments are available
2. Ollama client can connect to the Ollama server
3. A screenshot can be captured from Minecraft and sent to a vision LLM

Run with: python tests/test_mineland_installation.py
Or with pytest: pytest tests/test_mineland_installation.py -v

For the full integration test (--integration flag):
    1. Start Ollama: make up-hierarchical
    2. Pull the vision model: docker exec ollama ollama pull qwen2.5-vl:3b
    3. Run: python tests/test_mineland_installation.py --integration

Note: MineLand is automatically installed when using `make up-hierarchical`.
The Docker image includes all dependencies (Java JDK 17, Node.js 18.x, MineLand).
See https://github.com/cocacola-lab/MineLand for more information.
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

# Number of random actions to take to get an interesting Minecraft scene
# This allows the agents to move around a bit and see something other than spawn point
NUM_WARMUP_STEPS = 10


# =============================================================================
# Basic Import Tests
# =============================================================================

def test_mineland_import():
    """Test that mineland can be imported."""
    try:
        import mineland
        print("✓ MineLand imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Failed to import mineland: {e}")
        print("  Make sure you started the environment with: make up-hierarchical")
        print("  MineLand is automatically installed in the hierarchical Docker image.")
        return False


def test_mineland_dependencies():
    """Test that MineLand's key dependencies are available."""
    results = []
    
    # Check gymnasium
    try:
        import gymnasium as gym
        print("✓ gymnasium is available")
        results.append(True)
    except ImportError as e:
        print(f"✗ gymnasium not available: {e}")
        results.append(False)
    
    # Check numpy
    try:
        import numpy as np
        print("✓ numpy is available")
        results.append(True)
    except ImportError as e:
        print(f"✗ numpy not available: {e}")
        results.append(False)
    
    return all(results)


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

def test_mineland_screenshot():
    """Test capturing a screenshot from MineLand environment.
    
    Note: MineLand requires a Minecraft server to be running externally.
    The integration test will skip screenshot capture if the server is not available.
    """
    try:
        import mineland
        import numpy as np
        
        print("  Creating MineLand environment (this may take a moment)...")
        print("  Note: MineLand requires a Minecraft server to be running externally.")
        print("  See: https://github.com/cocacola-lab/MineLand#setup-minecraft-server")
        
        # MineLand uses a different API than MineRL
        # It creates multi-agent environments with the make() function
        # Use 'survival' which is the default task type in MineLand
        # The server_host and server_port should point to a running Minecraft server
        server_host = os.environ.get("MINELAND_SERVER_HOST", "localhost")
        server_port = int(os.environ.get("MINELAND_SERVER_PORT", "25565"))
        
        print(f"  Connecting to Minecraft server at {server_host}:{server_port}...")
        
        env = mineland.make(
            task_id="playground",  # Use playground task - a simple sandbox without goals
            agents_count=1,
            server_host=server_host,
            server_port=server_port,
        )
        
        print("  Resetting environment...")
        obs = env.reset()
        
        # Take random actions to get an interesting frame (move around from spawn)
        for _ in range(NUM_WARMUP_STEPS):
            # MineLand uses a different action format - action is per agent
            # For simplicity, use no-op action (agent stays still)
            action = [mineland.Action()] * 1  # No-op action for 1 agent
            obs, code_info, event, done, task_info = env.step(action)
            if done:
                obs = env.reset()
        
        # Extract the RGB observation image
        # MineLand observations are a list of agent observations
        # Each agent obs contains 'rgb' key for the visual observation
        if isinstance(obs, list) and len(obs) > 0:
            # Multi-agent case: get first agent's observation
            agent_obs = obs[0]
            if hasattr(agent_obs, 'rgb'):
                screenshot = agent_obs.rgb
            elif isinstance(agent_obs, dict) and "rgb" in agent_obs:
                screenshot = agent_obs["rgb"]
            else:
                print(f"✗ Unexpected agent observation format: {type(agent_obs)}")
                print(f"  Available attributes: {dir(agent_obs)}")
                env.close()
                return None
        elif isinstance(obs, np.ndarray):
            screenshot = obs
        else:
            print(f"✗ Unexpected observation format: {type(obs)}")
            env.close()
            return None
        
        print(f"✓ Captured screenshot: shape={screenshot.shape}, dtype={screenshot.dtype}")
        
        env.close()
        return screenshot
        
    except ConnectionRefusedError:
        print("✗ Could not connect to Minecraft server.")
        print("  MineLand requires an external Minecraft server.")
        print("  See: https://github.com/cocacola-lab/MineLand#setup-minecraft-server")
        return None
    except Exception as e:
        error_msg = str(e)
        if "connection" in error_msg.lower() or "refused" in error_msg.lower():
            print("✗ Could not connect to Minecraft server.")
            print("  MineLand requires an external Minecraft server.")
            print("  See: https://github.com/cocacola-lab/MineLand#setup-minecraft-server")
        else:
            print(f"✗ Failed to capture MineLand screenshot: {e}")
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
            # MineLand images are typically RGB
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
    print("MineLand & Ollama Installation Test (Basic)")
    print("=" * 60)
    print()
    
    results = []
    
    print("1. Testing MineLand import...")
    results.append(test_mineland_import())
    print()
    
    print("2. Testing MineLand dependencies...")
    results.append(test_mineland_dependencies())
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
    """Run full integration tests including MineLand environment and Ollama vision."""
    print("=" * 60)
    print("MineLand & Ollama Integration Test")
    print("=" * 60)
    print()
    
    results = []
    
    print("1. Testing MineLand import...")
    results.append(test_mineland_import())
    print()
    
    print("2. Testing Ollama import...")
    results.append(test_ollama_import())
    print()
    
    print("3. Testing Ollama connection...")
    ollama_ok = test_ollama_connection()
    results.append(ollama_ok)
    print()
    
    print("4. Testing MineLand screenshot capture...")
    screenshot = test_mineland_screenshot()
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
    """Run MineLand and Ollama tests."""
    parser = argparse.ArgumentParser(
        description="Test MineLand and Ollama installation and integration"
    )
    parser.add_argument(
        "--integration",
        action="store_true",
        help="Run full integration test (requires Ollama server and MineLand environment)",
    )
    args = parser.parse_args()
    
    if args.integration:
        return run_integration_tests()
    else:
        return run_basic_tests()


if __name__ == "__main__":
    sys.exit(main())

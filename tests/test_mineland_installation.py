#!/usr/bin/env python3
"""
Test script to verify MineLand and Ollama installation and integration.

This script checks that:
1. MineLand can be imported and environments are available
2. Ollama client can connect to the Ollama server
3. A screenshot can be captured from Minecraft and sent to a vision LLM

Architecture (when running with `make up-hierarchical`):
- empo-dev container: Your RL/planning code + MineLand (spawns Minecraft internally)
- ollama container: Runs the LLM server (accessible at ollama:11434)

IMPORTANT: MineLand spawns Minecraft internally using headless mode.
No separate Minecraft server container is needed.

Run basic tests: python tests/test_mineland_installation.py
Run integration tests: python tests/test_mineland_installation.py --integration

For the full integration test (--integration flag):
    1. Start containers: make up-hierarchical
    2. Pull the vision model: docker exec ollama ollama pull qwen2.5vl:7b
    3. Run: make test-mineland-integration

Note: First run may take 1-2 minutes to download Minecraft.

See https://github.com/cocacola-lab/MineLand for more information.
"""

import argparse
import io
import os
import signal
import sys


# =============================================================================
# Configuration Constants
# =============================================================================

# Default Ollama server host
# When running in Docker, use container name 'ollama' for inter-container communication
DEFAULT_OLLAMA_HOST = "http://ollama:11434"

# Default vision model for Ollama (qwen2.5vl is a vision-language model)
DEFAULT_VISION_MODEL = "qwen2.5vl:7b"

# Number of random actions to take to get an interesting Minecraft scene
NUM_WARMUP_STEPS = 10

# Timeout for MineLand operations (in seconds)
MINELAND_TIMEOUT = 120  # 2 minutes

# Xvfb (virtual display) configuration for MineLand RGB capture
XVFB_DISPLAY = ':99'
XVFB_RESOLUTION = '1024x768x24'
XVFB_STARTUP_DELAY = 1  # seconds to wait for Xvfb to start


# =============================================================================
# Helper Functions
# =============================================================================

class TimeoutError(Exception):
    """Raised when an operation times out."""
    pass


def timeout_handler(signum, frame):
    """Signal handler for timeout."""
    raise TimeoutError("Operation timed out")


# =============================================================================
# Basic Import Tests
# =============================================================================

def test_mineland_import():
    """Test that mineland can be imported."""
    try:
        import mineland
        print("✓ MineLand imported successfully")
        
        # Check key exports are available
        if hasattr(mineland, 'MineLand'):
            print("  ✓ mineland.MineLand class available")
        else:
            print("  ⚠ mineland.MineLand not found - may need different API")
            print(f"    Available exports: {[x for x in dir(mineland) if not x.startswith('_')]}")
        
        if hasattr(mineland, 'make'):
            print("  ✓ mineland.make() function available")
        
        if hasattr(mineland, 'Action'):
            print("  ✓ mineland.Action class available")
            
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

def start_xvfb():
    """Start Xvfb virtual display for MineLand rendering.
    
    MineLand requires Xvfb even in headless mode to capture RGB frames.
    Returns the subprocess if started, None if already running or failed.
    """
    import subprocess
    import time
    
    # Check if DISPLAY is already set (Xvfb might already be running)
    if os.environ.get('DISPLAY'):
        print(f"  DISPLAY already set to {os.environ['DISPLAY']}")
        return None
    
    # Start Xvfb on display :99 (common convention for headless)
    print("  Starting Xvfb virtual display...")
    try:
        xvfb_proc = subprocess.Popen(
            ['Xvfb', ':99', '-screen', '0', '1024x768x24'],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        time.sleep(1)  # Give Xvfb time to start
        os.environ['DISPLAY'] = ':99'
        print(f"  ✓ Xvfb started on DISPLAY={os.environ['DISPLAY']}")
        return xvfb_proc
    except FileNotFoundError:
        print("  ⚠ Xvfb not found - install with: apt-get install xvfb")
        print("    MineLand may not capture RGB frames without a display")
        return None
    except Exception as e:
        print(f"  ⚠ Failed to start Xvfb: {e}")
        return None


def test_mineland_screenshot():
    """Test capturing a screenshot from MineLand environment.
    
    MineLand spawns Minecraft internally, so this test runs MineLand
    directly in the current container (empo-dev) using headless mode.
    
    Note: This requires significant resources and can take 1-2 minutes
    to start Minecraft on first run.
    
    IMPORTANT: MineLand requires Xvfb for RGB capture even in headless mode!
    """
    env = None
    xvfb_proc = None
    try:
        import mineland
        import numpy as np
        
        # Start Xvfb if not already running (required for RGB capture)
        xvfb_proc = start_xvfb()
        
        print("  Starting MineLand in headless mode...")
        print("  Note: First run downloads Minecraft (~1-2 minutes)")
        print(f"  Timeout: {MINELAND_TIMEOUT}s")
        
        # Set up a timeout for the MineLand initialization
        if hasattr(signal, 'SIGALRM'):
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(MINELAND_TIMEOUT)
        
        try:
            # Create MineLand environment
            # This will spawn Minecraft internally
            # Note: Must specify image_size to get RGB observations
            # See: https://github.com/cocacola-lab/MineLand/blob/main/scripts/rgb_frame.py
            # 
            # IMPORTANT: headless=False is required for RGB capture!
            # headless=True disables the prismarine-viewer which captures RGB frames.
            # The Xvfb virtual display provides a "fake" screen for rendering.
            env = mineland.make(
                task_id="playground",
                agents_count=1,
                headless=False,  # Must be False to capture RGB frames (Xvfb provides display)
                image_size=(180, 320),  # (height, width) - required for RGB capture
            )
            
            print("  ✓ MineLand environment created")
            print("  Resetting environment...")
            obs = env.reset()
            
        finally:
            if hasattr(signal, 'SIGALRM'):
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
        
        # Take actions to get an interesting frame
        print(f"  Taking {NUM_WARMUP_STEPS} warmup steps...")
        for step in range(NUM_WARMUP_STEPS):
            # Use no-op action: Action(type=RESUME, code="")
            action = mineland.Action.no_op(1)  # 1 agent
            obs, code_info, event, done, task_info = env.step(action)
        
        # Extract the RGB observation image
        # MineLand returns observations in CHW format (channels, height, width)
        if isinstance(obs, list) and len(obs) > 0:
            agent_obs = obs[0]
            if hasattr(agent_obs, 'rgb'):
                screenshot = agent_obs.rgb
            elif isinstance(agent_obs, dict) and "rgb" in agent_obs:
                screenshot = agent_obs["rgb"]
            else:
                print(f"✗ Unexpected agent observation format: {type(agent_obs)}")
                print(f"  Available attributes: {dir(agent_obs) if hasattr(agent_obs, '__dir__') else 'N/A'}")
                env.close()
                return None
        elif isinstance(obs, np.ndarray):
            screenshot = obs
        else:
            print(f"✗ Unexpected observation format: {type(obs)}")
            env.close()
            return None
        
        # Check if screenshot is valid (not empty)
        if screenshot is None or screenshot.size == 0 or (len(screenshot.shape) >= 2 and screenshot.shape[1] == 0):
            print(f"✗ Screenshot is empty or invalid: shape={screenshot.shape if screenshot is not None else 'None'}")
            print("  Make sure image_size is specified in mineland.make()")
            env.close()
            return None
        
        print(f"✓ Captured screenshot: shape={screenshot.shape}, dtype={screenshot.dtype}")
        
        # Convert from CHW to HWC format (MineLand uses CHW, PIL uses HWC)
        if len(screenshot.shape) == 3 and screenshot.shape[0] == 3:
            screenshot = np.transpose(screenshot, (1, 2, 0))
            print(f"  Transposed to HWC format: shape={screenshot.shape}")
        
        # Save screenshot to outputs directory
        try:
            from PIL import Image
            outputs_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs")
            os.makedirs(outputs_dir, exist_ok=True)
            screenshot_path = os.path.join(outputs_dir, "mineland_screenshot.png")
            
            # Convert numpy array to PIL Image and save
            if screenshot.dtype != np.uint8:
                screenshot = (screenshot * 255).astype(np.uint8)
            img = Image.fromarray(screenshot)
            img.save(screenshot_path)
            print(f"✓ Screenshot saved to: {screenshot_path}")
        except Exception as e:
            print(f"⚠ Could not save screenshot: {e}")
        
        env.close()
        # Stop Xvfb if we started it
        if xvfb_proc is not None:
            xvfb_proc.terminate()
        return screenshot
    
    except TimeoutError:
        print(f"✗ MineLand initialization timed out after {MINELAND_TIMEOUT} seconds")
        print("  Minecraft may still be downloading or starting.")
        print("  Try again or increase timeout.")
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
        if xvfb_proc is not None:
            xvfb_proc.terminate()
        return None
    except Exception as e:
        error_msg = str(e)
        print(f"✗ Failed to capture MineLand screenshot: {e}")
        import traceback
        traceback.print_exc()
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
        if xvfb_proc is not None:
            xvfb_proc.terminate()
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
        
        # Save LLM response to outputs directory
        try:
            outputs_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs")
            os.makedirs(outputs_dir, exist_ok=True)
            response_path = os.path.join(outputs_dir, "mineland_llm_response.txt")
            with open(response_path, "w") as f:
                f.write(f"Model: {model_name}\n")
                f.write(f"Prompt: Describe what you see in this Minecraft screenshot in 2-3 sentences.\n")
                f.write(f"\nResponse:\n{description}\n")
            print(f"✓ LLM response saved to: {response_path}")
        except Exception as e:
            print(f"⚠ Could not save LLM response: {e}")
        
        return True
        
    except ImportError as e:
        print(f"✗ Missing required package: {e}")
        print("  Install with: pip install ollama Pillow")
        return False
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
        # Force exit because MineLand spawns background threads that don't terminate cleanly
        os._exit(0)
    else:
        print("✗ Some tests failed")
        # Force exit because MineLand spawns background threads that don't terminate cleanly
        os._exit(1)


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

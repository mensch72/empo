#!/usr/bin/env python3
"""
Test script to verify MineLand and Ollama installation and integration.

This script checks that:
1. MineLand can be imported and environments are available
2. Ollama client can connect to the Ollama server
3. A three-player Minecraft world can be created with custom terrain
4. Screenshots can be captured from all three player perspectives
5. Vision LLM can describe the screenshots with spatial information

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
NUM_WARMUP_STEPS = 5

# Timeout for MineLand operations (in seconds)
MINELAND_TIMEOUT = 180  # 3 minutes for 3 agents

# Maximum number of world build commands to execute during tests
# (Full world build has thousands of commands, we limit for test speed)
MAX_TEST_WORLD_COMMANDS = 100

# Number of agents in the three-player world
AGENTS_COUNT = 3

# Xvfb (virtual display) configuration for MineLand RGB capture
XVFB_DISPLAY = ":99"
XVFB_RESOLUTION = "1024x768x24"
XVFB_STARTUP_DELAY = 1  # seconds to wait for Xvfb to start

# Prompt template for multi-player LLM descriptions with all 6 images
SIX_IMAGE_DESCRIPTION_PROMPT = """You are analyzing 6 Minecraft screenshots from a three-player world.
The world has:
- A central river dividing east and west
- Western forest (abundant wood, some stone)
- Eastern rocky area (abundant stone, some wood)
- Northern mountain with cave system
- Southern plains with farmland

The three players are:
1. **Robot** at coordinates (0, 70, 0) - center of valley near river
2. **Human A** at coordinates (-60, 70, 0) - west side, forest area
3. **Human B** at coordinates (60, 70, 0) - east side, rocky area

The 6 images are arranged as follows:
- Row 1: Robot (step 1), Human A (step 1), Human B (step 1)
- Row 2: Robot (step 2), Human A (step 2), Human B (step 2)

Please describe each of the 6 views in detail, including:
1. What terrain and environment features are visible
2. Spatial information: what is to the left, right, ahead relative to each player's view
3. Any visible resources (trees, stone, water, ores)
4. How the views differ between the two timesteps for each player
5. Notable landmarks that would help orient each player

Be specific about positions and distances. Format your response with clear headers for each player and timestep."""


# =============================================================================
# Helper Functions
# =============================================================================


class TimeoutError(Exception):
    """Raised when an operation times out."""



def timeout_handler(signum, frame):
    """Signal handler for timeout."""
    raise TimeoutError("Operation timed out")


def start_xvfb():
    """Start Xvfb virtual display for MineLand rendering.

    MineLand requires Xvfb even in headless mode to capture RGB frames.
    Returns the subprocess if started, None if already running or failed.
    """
    import subprocess
    import time

    # Check if DISPLAY is already set (Xvfb might already be running)
    if os.environ.get("DISPLAY"):
        print(f"  DISPLAY already set to {os.environ['DISPLAY']}")
        return None

    # Start Xvfb on display :99 (common convention for headless)
    print("  Starting Xvfb virtual display...")
    try:
        xvfb_proc = subprocess.Popen(
            ["Xvfb", ":99", "-screen", "0", "1024x768x24"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        time.sleep(1)  # Give Xvfb time to start
        os.environ["DISPLAY"] = ":99"
        print(f"  ✓ Xvfb started on DISPLAY={os.environ['DISPLAY']}")
        return xvfb_proc
    except FileNotFoundError:
        print("  ⚠ Xvfb not found - install with: apt-get install xvfb")
        print("    MineLand may not capture RGB frames without a display")
        return None
    except Exception as e:
        print(f"  ⚠ Failed to start Xvfb: {e}")
        return None


# =============================================================================
# Basic Import Tests
# =============================================================================


def test_mineland_import():
    """Test that mineland can be imported."""
    import mineland

    print("✓ MineLand imported successfully")

    # Check key exports are available
    if hasattr(mineland, "MineLand"):
        print("  ✓ mineland.MineLand class available")
    else:
        print("  ⚠ mineland.MineLand not found - may need different API")
        print(
            f"    Available exports: {[x for x in dir(mineland) if not x.startswith('_')]}"
        )

    if hasattr(mineland, "make"):
        print("  ✓ mineland.make() function available")

    if hasattr(mineland, "Action"):
        print("  ✓ mineland.Action class available")


def test_mineland_dependencies():
    """Test that MineLand's key dependencies are available."""
    # Check gymnasium
    print("✓ gymnasium is available")

    # Check numpy
    print("✓ numpy is available")


def test_ollama_import():
    """Test that ollama can be imported."""
    print("✓ Ollama Python client imported successfully")


def test_ollama_connection():
    """Test that we can connect to the Ollama server."""
    import ollama

    # Get Ollama host from environment or use default
    ollama_host = os.environ.get("OLLAMA_HOST", DEFAULT_OLLAMA_HOST)
    print(f"  Connecting to Ollama at {ollama_host}...")

    client = ollama.Client(host=ollama_host)
    models = client.list()

    model_names = [
        m.get("name", m.get("model", "unknown")) for m in models.get("models", [])
    ]
    if model_names:
        print(f"✓ Connected to Ollama. Available models: {', '.join(model_names)}")
    else:
        print("✓ Connected to Ollama. No models installed yet.")
        print(
            f"  Pull a model with: docker exec ollama ollama pull {DEFAULT_VISION_MODEL}"
        )


# =============================================================================
# Integration Tests
# =============================================================================


def extract_screenshot(agent_obs, agent_name="agent"):
    """Extract RGB screenshot from agent observation.

    Args:
        agent_obs: Agent observation object from MineLand
        agent_name: Name for logging purposes

    Returns:
        numpy array in HWC format (height, width, channels), or None if failed
    """
    import numpy as np

    screenshot = None
    if hasattr(agent_obs, "rgb"):
        screenshot = agent_obs.rgb
    elif isinstance(agent_obs, dict) and "rgb" in agent_obs:
        screenshot = agent_obs["rgb"]
    else:
        print(f"  ⚠ Could not get RGB for {agent_name}")
        return None

    # Check if screenshot is valid
    if screenshot is None or screenshot.size == 0:
        print(f"  ⚠ Empty screenshot for {agent_name}")
        return None

    # Convert from CHW to HWC format if needed
    if len(screenshot.shape) == 3 and screenshot.shape[0] == 3:
        screenshot = np.transpose(screenshot, (1, 2, 0))

    # Convert to uint8 if needed
    if screenshot.dtype != np.uint8:
        screenshot = (screenshot * 255).astype(np.uint8)

    return screenshot


def test_three_player_world():
    """Create a 3-player MineLand world and capture screenshots from all perspectives.

    This test:
    1. Creates a MineLand environment with 3 agents (robot, human_a, human_b)
    2. Sets up the custom world terrain using Minecraft commands
    3. Takes 2 steps with random actions to get 6 screenshots (3 players × 2 timesteps)
    4. Returns the 6 screenshots organized by player and timestep

    Returns:
        Dictionary with keys 'step1' and 'step2', each containing a dict mapping
        player names to screenshots. Returns None if failed.
    """
    env = None
    xvfb_proc = None
    try:
        import mineland
        import time

        # Import our world configuration and setup functions
        from src.llm_hierarchical_modeler import (
            get_spawn_points,
            get_player_spawn_info,
            generate_world_commands,
            generate_teleport_commands,
        )

        # Get the spawn info
        spawn_points = get_spawn_points()
        spawn_info = get_player_spawn_info()
        player_names = [sp["name"] for sp in spawn_points]

        print(f"  Creating 3-player world with players: {player_names}")
        for info in spawn_info:
            print(
                f"    - {info['name']}: {info['coordinates']} ({info['description']})"
            )

        # Start Xvfb if not already running (required for RGB capture)
        xvfb_proc = start_xvfb()

        print("  Starting MineLand with 3 agents...")
        print("  Note: First run downloads Minecraft (~1-2 minutes)")
        print(f"  Timeout: {MINELAND_TIMEOUT}s")

        # Set up a timeout for the MineLand initialization
        if hasattr(signal, "SIGALRM"):
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(MINELAND_TIMEOUT)

        try:
            # Create agent configs for all 3 players
            agents_config = [{"name": name} for name in player_names]

            env = mineland.make(
                task_id="playground",
                agents_count=AGENTS_COUNT,
                agents_config=agents_config,
                headless=False,  # Required for RGB capture
                image_size=(180, 320),  # (height, width)
            )

            print(f"  ✓ MineLand {AGENTS_COUNT}-agent environment created")
            print("  Resetting environment...")
            obs = env.reset()

        finally:
            if hasattr(signal, "SIGALRM"):
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)

        # Access server manager to execute commands
        server_manager = None
        if hasattr(env, "env") and hasattr(env.env, "server_manager"):
            server_manager = env.env.server_manager
        elif hasattr(env, "server_manager"):
            server_manager = env.server_manager

        if server_manager is None:
            print("  ⚠ Cannot access server manager, skipping world setup")
        else:
            # Build the custom world terrain
            print("  Building custom world terrain...")
            world_commands = generate_world_commands()
            # Execute a subset of commands to avoid timeout
            for i, cmd in enumerate(world_commands[:MAX_TEST_WORLD_COMMANDS]):
                server_manager.execute(cmd)
                if i % 10 == 0:
                    time.sleep(0.1)  # Allow server to process commands
            print(
                f"  ✓ Executed {min(MAX_TEST_WORLD_COMMANDS, len(world_commands))} "
                "world build commands"
            )

            # Teleport players to their spawn positions
            print("  Teleporting players to spawn positions...")
            for cmd in generate_teleport_commands():
                server_manager.execute(cmd)
            time.sleep(2)

        # Take initial warmup steps
        print(f"  Taking {NUM_WARMUP_STEPS} warmup steps...")
        for step in range(NUM_WARMUP_STEPS):
            action = mineland.Action.no_op(AGENTS_COUNT)
            obs, code_info, event, done, task_info = env.step(action)

        # Collect screenshots from 2 timesteps
        all_screenshots = {"step1": {}, "step2": {}}

        for step_num in range(1, 3):
            print(f"  Capturing screenshots for step {step_num}...")

            # Take a step with no-op actions for stability
            action = mineland.Action.no_op(AGENTS_COUNT)
            obs, code_info, event, done, task_info = env.step(action)

            # Extract screenshots for all agents
            if isinstance(obs, list) and len(obs) >= AGENTS_COUNT:
                for i, (name, agent_obs) in enumerate(zip(player_names, obs)):
                    screenshot = extract_screenshot(agent_obs, name)
                    if screenshot is not None:
                        all_screenshots[f"step{step_num}"][name] = screenshot
                        print(f"    ✓ {name} step {step_num}: shape={screenshot.shape}")
                    else:
                        print(f"    ⚠ Failed to capture {name} step {step_num}")
            else:
                print(
                    f"  ⚠ Expected {AGENTS_COUNT} observations, got: {len(obs) if isinstance(obs, list) else type(obs)}"
                )

        # Verify we got all 6 screenshots
        total_screenshots = sum(len(v) for v in all_screenshots.values())
        if total_screenshots < 6:
            print(f"✗ Only captured {total_screenshots}/6 screenshots")
        else:
            print("✓ Captured all 6 screenshots successfully")

        # Save individual screenshots to outputs directory
        try:
            from PIL import Image

            outputs_dir = os.path.join(
                os.path.dirname(os.path.dirname(__file__)), "outputs"
            )
            os.makedirs(outputs_dir, exist_ok=True)

            for step_key, step_screenshots in all_screenshots.items():
                for name, screenshot in step_screenshots.items():
                    screenshot_path = os.path.join(
                        outputs_dir, f"mineland_{name}_{step_key}.png"
                    )
                    img = Image.fromarray(screenshot)
                    img.save(screenshot_path)
                    print(f"  ✓ Saved {name} {step_key} to: {screenshot_path}")
        except Exception as e:
            print(f"  ⚠ Could not save individual screenshots: {e}")

        env.close()
        if xvfb_proc is not None:
            xvfb_proc.terminate()

        return all_screenshots

    except TimeoutError:
        print(f"✗ MineLand initialization timed out after {MINELAND_TIMEOUT} seconds")
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
        if xvfb_proc is not None:
            xvfb_proc.terminate()
        return None
    except Exception as e:
        print(f"✗ Failed to create three-player world: {e}")
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


def create_six_image_grid_pdf(all_screenshots, llm_prompt, llm_response, output_path):
    """Create a PDF with 6 screenshots in a 2x3 grid with LLM caption.

    Args:
        all_screenshots: Dict with 'step1' and 'step2' keys, each containing
                        player name -> screenshot mappings
        llm_prompt: The prompt sent to the LLM
        llm_response: The LLM's response
        output_path: Path to save the PDF

    Returns:
        True if successful, False otherwise
    """
    try:
        from PIL import Image
        from reportlab.lib.pagesizes import letter, landscape
        from reportlab.platypus import (
            SimpleDocTemplate,
            Image as RLImage,
            Paragraph,
            Spacer,
            Table,
            TableStyle,
        )
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.lib import colors
        import tempfile

        print("  Creating PDF with 2×3 grid of screenshots...")

        # Get player names in order
        player_names = ["robot", "human_a", "human_b"]

        # Create temporary image files for the grid
        temp_files = []
        grid_images = []

        # Row 1: Step 1 images (robot, human_a, human_b)
        # Row 2: Step 2 images (robot, human_a, human_b)
        for step_key in ["step1", "step2"]:
            row_images = []
            for name in player_names:
                if name in all_screenshots.get(step_key, {}):
                    screenshot = all_screenshots[step_key][name]
                    # Save to temp file
                    temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                    temp_files.append(temp_file.name)
                    img = Image.fromarray(screenshot)
                    img.save(temp_file.name)
                    row_images.append(temp_file.name)
                else:
                    row_images.append(None)
            grid_images.append(row_images)

        # Create PDF
        doc = SimpleDocTemplate(
            output_path,
            pagesize=landscape(letter),
            leftMargin=0.5 * inch,
            rightMargin=0.5 * inch,
            topMargin=0.5 * inch,
            bottomMargin=0.5 * inch,
        )

        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            "Title",
            parent=styles["Heading1"],
            fontSize=14,
            alignment=1,  # Center
        )
        caption_style = ParagraphStyle(
            "Caption",
            parent=styles["Normal"],
            fontSize=8,
            leading=10,
        )

        story = []

        # Title
        story.append(
            Paragraph(
                "Three-Player Minecraft World - 6 Views (2 Timesteps × 3 Players)",
                title_style,
            )
        )
        story.append(Spacer(1, 0.2 * inch))

        # Create table with images
        # Each image should be about 3 inches wide to fit 3 across in landscape
        # MineLand image_size=(180, 320) means 320 width x 180 height
        # Aspect ratio = 320/180 ≈ 1.78
        img_width = 2.8 * inch
        img_height = img_width / (320 / 180)  # Correct aspect ratio

        table_data = []

        # Header row
        table_data.append(
            [
                Paragraph("<b>Robot (0, 70, 0)</b>", caption_style),
                Paragraph("<b>Human A (-60, 70, 0)</b>", caption_style),
                Paragraph("<b>Human B (60, 70, 0)</b>", caption_style),
            ]
        )

        # Image rows
        for step_idx, row_images in enumerate(grid_images):
            row = []
            for img_path in row_images:
                if img_path:
                    row.append(RLImage(img_path, width=img_width, height=img_height))
                else:
                    row.append(Paragraph("(Missing)", caption_style))
            table_data.append(row)
            # Add step label
            table_data.append(
                [
                    Paragraph(f"<i>Step {step_idx + 1}</i>", caption_style),
                    Paragraph(f"<i>Step {step_idx + 1}</i>", caption_style),
                    Paragraph(f"<i>Step {step_idx + 1}</i>", caption_style),
                ]
            )

        table = Table(table_data, colWidths=[img_width + 0.2 * inch] * 3)
        table.setStyle(
            TableStyle(
                [
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                    ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ]
            )
        )

        story.append(table)
        story.append(Spacer(1, 0.3 * inch))

        # LLM Prompt section
        story.append(Paragraph("<b>LLM Prompt:</b>", styles["Heading3"]))
        # Wrap prompt text
        prompt_text = llm_prompt.replace("\n", "<br/>")
        story.append(Paragraph(prompt_text, caption_style))
        story.append(Spacer(1, 0.2 * inch))

        # LLM Response section
        story.append(Paragraph("<b>LLM Response:</b>", styles["Heading3"]))
        # Wrap response text
        response_text = llm_response.replace("\n", "<br/>")
        story.append(Paragraph(response_text, caption_style))

        # Build PDF
        doc.build(story)

        # Cleanup temp files
        for temp_file in temp_files:
            try:
                os.unlink(temp_file)
            except Exception:
                pass

        print(f"  ✓ PDF saved to: {output_path}")
        return True

    except ImportError as e:
        print(f"  ⚠ Could not create PDF (missing reportlab): {e}")
        print("    Install with: pip install reportlab")
        return False
    except Exception as e:
        print(f"  ⚠ Failed to create PDF: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_six_image_llm_description(all_screenshots):
    """Send all 6 screenshots to the LLM in a single call for detailed descriptions.

    Args:
        all_screenshots: Dict with 'step1' and 'step2' keys, each containing
                        player name -> screenshot mappings

    Returns:
        Tuple of (prompt, response, pdf_created) or None if failed
    """
    try:
        import ollama
        from PIL import Image

        ollama_host = os.environ.get("OLLAMA_HOST", DEFAULT_OLLAMA_HOST)
        model_name = os.environ.get("OLLAMA_VISION_MODEL", DEFAULT_VISION_MODEL)

        # Check if the model is available
        client = ollama.Client(host=ollama_host)
        models = client.list()
        model_names = [
            m.get("name", m.get("model", "")) for m in models.get("models", [])
        ]

        if not any(model_name in name for name in model_names):
            print(f"✗ Vision model '{model_name}' not found in Ollama")
            print(
                f"  Available models: {', '.join(model_names) if model_names else 'none'}"
            )
            print(f"  Pull it with: docker exec ollama ollama pull {model_name}")
            return None

        print(f"  Sending all 6 screenshots to {model_name}...")

        # Collect all 6 images in order: robot_s1, human_a_s1, human_b_s1, robot_s2, human_a_s2, human_b_s2
        player_names = ["robot", "human_a", "human_b"]
        image_bytes_list = []

        for step_key in ["step1", "step2"]:
            for name in player_names:
                if name in all_screenshots.get(step_key, {}):
                    screenshot = all_screenshots[step_key][name]
                    img = Image.fromarray(screenshot.astype("uint8"), "RGB")
                    buffer = io.BytesIO()
                    img.save(buffer, format="PNG")
                    image_bytes_list.append(buffer.getvalue())
                else:
                    print(f"  ⚠ Missing {name} {step_key}")

        if len(image_bytes_list) < 6:
            print(f"  ⚠ Only have {len(image_bytes_list)}/6 images")

        prompt = SIX_IMAGE_DESCRIPTION_PROMPT

        response = client.chat(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                    "images": image_bytes_list,
                }
            ],
        )

        llm_response = response["message"]["content"]
        print(f"✓ LLM description received ({len(llm_response)} chars)")

        # Save text response to outputs directory
        try:
            outputs_dir = os.path.join(
                os.path.dirname(os.path.dirname(__file__)), "outputs"
            )
            os.makedirs(outputs_dir, exist_ok=True)
            response_path = os.path.join(
                outputs_dir, "mineland_six_view_description.txt"
            )

            with open(response_path, "w") as f:
                f.write("Six-View Three-Player World LLM Description\n")
                f.write(f"Model: {model_name}\n")
                f.write("=" * 60 + "\n\n")
                f.write("PROMPT:\n")
                f.write("-" * 40 + "\n")
                f.write(prompt + "\n\n")
                f.write("RESPONSE:\n")
                f.write("-" * 40 + "\n")
                f.write(llm_response + "\n")

            print(f"  ✓ Text response saved to: {response_path}")
        except Exception as e:
            print(f"  ⚠ Could not save text response: {e}")

        # Create PDF with 2×3 grid and caption
        pdf_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "outputs",
            "mineland_six_views.pdf",
        )
        pdf_created = create_six_image_grid_pdf(
            all_screenshots, prompt, llm_response, pdf_path
        )

        # Print response
        print("\n" + "=" * 60)
        print("LLM Description of All 6 Views:")
        print("=" * 60)
        print(llm_response)
        print()

        return (prompt, llm_response, pdf_created)

    except ImportError as e:
        print(f"✗ Missing required package: {e}")
        print("  Install with: pip install ollama Pillow reportlab")
        return None
    except Exception as e:
        print(f"✗ Failed to get LLM description: {e}")
        import traceback

        traceback.print_exc()
        return None


def run_basic_tests():
    """Run basic import tests only."""
    print("=" * 60)
    print("MineLand & Ollama Installation Test (Basic)")
    print("=" * 60)
    print()

    print("1. Testing MineLand import...")
    test_mineland_import()
    print()

    print("2. Testing MineLand dependencies...")
    test_mineland_dependencies()
    print()

    print("3. Testing Ollama import...")
    test_ollama_import()
    print()

    print("=" * 60)
    print("✓ All basic tests passed!")
    return 0


def run_integration_tests():
    """Run full integration tests with 3-player MineLand and Ollama vision."""
    print("=" * 60)
    print("MineLand & Ollama Integration Test (3-Player)")
    print("=" * 60)
    print()

    print("1. Testing MineLand import...")
    test_mineland_import()
    print()

    print("2. Testing Ollama import...")
    test_ollama_import()
    print()

    print("3. Testing Ollama connection...")
    test_ollama_connection()
    print()

    # Main test: 3-player world with 6 screenshots
    print("4. Creating 3-player MineLand world and capturing 6 screenshots...")
    all_screenshots = test_three_player_world()

    # Check we got 6 screenshots
    total_screenshots = 0
    if all_screenshots:
        total_screenshots = sum(len(v) for v in all_screenshots.values())
    assert total_screenshots == 6, f"Expected 6 screenshots, got {total_screenshots}"
    print()

    print("5. Sending 6 screenshots to vision LLM for detailed description...")
    llm_result = test_six_image_llm_description(all_screenshots)
    assert llm_result is not None, "LLM description failed"
    print()

    print("=" * 60)
    print("✓ All integration tests passed!")
    # Force exit because MineLand spawns background threads that don't terminate cleanly
    os._exit(0)


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

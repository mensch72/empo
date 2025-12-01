#!/usr/bin/env python3
"""
Control Button Demo with Neural Policy Learning.

This script demonstrates the ControlButton mechanism where:
- A robot (grey agent) can program control buttons with specific actions
- A human (yellow agent) can then use those buttons to control the robot

The scenario:
- 3 control buttons programmed with 'left', 'forward', 'right' actions
- 3 rocks that the robot (grey) can push
- The human learns to use the control buttons to guide the robot to push rocks

The human learns policies for 3 goals:
- Goal 1: Get the robot to where Rock 1 started
- Goal 2: Get the robot to where Rock 2 started  
- Goal 3: Get the robot to where Rock 3 started

This demonstrates human-robot control via programmable control interfaces.
"""

import sys
import os
import time
import random

# Add paths for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'vendor', 'multigrid'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np

# Patch gym import for compatibility
import gymnasium as gym
sys.modules['gym'] = gym

from gym_multigrid.multigrid import MultiGridEnv, Grid, Agent, Wall, World, Actions, ControlButton, Rock

# Output directory for movies and images
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
OUTPUT_DIR = os.path.join(PROJECT_DIR, 'outputs')

# ============================================================================
# Environment Definition
# ============================================================================

# Initial layout for prequel (robot and human need to program buttons):
# - Human (yellow) starts at (1, 1) out of the way
# - Robot (grey) starts at (2, 3) in position to program buttons
# - Buttons at (2, 2), (3, 3), (2, 4) need to be programmed
# - Rocks at (5, 2), (5, 3), (5, 4)
PREQUEL_MAP = """
We We We We We We We We
We Ay .. .. .. .. .. We
We .. CB .. .. Ro .. We
We .. Ae CB .. Ro We We
We .. CB .. .. Ro .. We
We .. .. .. .. .. .. We
We We We We We We We We
"""

# Ready state layout: after robot programs buttons and steps aside
# - Human (yellow) at (2, 3) - between the buttons, ready to use them
# - Robot (grey) at (1, 5) - stepped out of the way
# - Buttons already programmed
READY_MAP = """
We We We We We We We We
We .. .. .. .. .. .. We
We .. CB .. .. Ro .. We
We .. Ay CB .. Ro We We
We .. CB .. .. Ro .. We
We Ae .. .. .. .. .. We
We We We We We We We We
"""


class ControlButtonEnv(MultiGridEnv):
    """
    Environment with control buttons that let humans control robot actions.
    
    Can be initialized in two modes:
    1. pre_programmed=False: Robot starts in position to program buttons (for prequel)
    2. pre_programmed=True: Buttons are already programmed, agents in ready positions
       - Human at (2,3) between buttons
       - Robot at (1,5) stepped aside
    
    Button programming:
    - Upper (2, 2) -> left action
    - Right (3, 3) -> forward action
    - Lower (2, 4) -> right action
    
    The robot can push rocks (has can_push_rocks=True).
    """
    
    def __init__(self, max_steps: int = 100, pre_programmed: bool = True):
        """
        Args:
            max_steps: Maximum steps per episode
            pre_programmed: If True, buttons start pre-programmed with agents in ready positions
        """
        self.pre_programmed = pre_programmed
        # Use different maps based on mode
        map_str = READY_MAP if pre_programmed else PREQUEL_MAP
        super().__init__(
            map=map_str,
            max_steps=max_steps,
            partial_obs=False,
            objects_set=World,
            actions_set=Actions
        )
        
        # Store initial rock positions for goal checking
        self.rock_positions = []
        for j in range(self.grid.height):
            for i in range(self.grid.width):
                cell = self.grid.get(i, j)
                if cell is not None and cell.type == 'rock':
                    self.rock_positions.append((i, j))
    
    def _gen_grid(self, width, height):
        """Generate the grid and optionally pre-program control buttons."""
        # Call parent to generate from map
        super()._gen_grid(width, height)
        
        # Set robot to be able to push rocks
        for agent in self.agents:
            if agent.color == 'grey':
                agent.can_push_rocks = True
        
        if self.pre_programmed:
            # Pre-program the control buttons with actions as specified:
            # Upper button (2, 2) -> left
            # Right button (3, 3) -> forward
            # Lower button (2, 4) -> right
            button_actions = {
                (2, 2): Actions.left,     # Upper button -> left
                (3, 3): Actions.forward,  # Right button -> forward
                (2, 4): Actions.right,    # Lower button -> right
            }
            
            # Find the robot agent index
            robot_idx = None
            for i, agent in enumerate(self.agents):
                if agent.color == 'grey':
                    robot_idx = i
                    break
            
            if robot_idx is not None:
                for j in range(self.grid.height):
                    for i in range(self.grid.width):
                        cell = self.grid.get(i, j)
                        if cell is not None and cell.type == 'controlbutton':
                            pos = (i, j)
                            if pos in button_actions:
                                cell.controlled_agent = robot_idx
                                cell.triggered_action = button_actions[pos]


def create_movie(frames, output_path, fps=2):
    """
    Create a movie (GIF) from a list of frames.
    
    Args:
        frames: List of (title, image) tuples or just images
        output_path: Path to save the movie
        fps: Frames per second
    """
    try:
        from PIL import Image
        import matplotlib.pyplot as plt
        from io import BytesIO
        
        # Convert frames to PIL images with titles
        pil_frames = []
        for frame in frames:
            if isinstance(frame, tuple):
                title, img = frame
            else:
                title, img = None, frame
            
            # Create figure with title
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.imshow(img)
            if title:
                ax.set_title(title, fontsize=14)
            ax.axis('off')
            
            # Convert to image using buffer
            buf = BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
            buf.seek(0)
            pil_frames.append(Image.open(buf).copy())
            buf.close()
            plt.close(fig)
        
        # Save as GIF
        if pil_frames:
            duration = int(1000 / fps)  # milliseconds per frame
            pil_frames[0].save(
                output_path,
                save_all=True,
                append_images=pil_frames[1:],
                duration=duration,
                loop=0
            )
            print(f"Saved movie to {output_path}")
            return True
    except ImportError as e:
        print(f"Could not create movie: {e}")
        return False
    return False


def demonstrate_prequel_with_movie():
    """
    Demonstrate the prequel phase where robot programs buttons, saving as a movie.
    """
    print("=" * 70)
    print("Prequel Demonstration: Robot Programs Buttons")
    print("=" * 70)
    print()
    
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Create environment WITHOUT pre-programming
    env = ControlButtonEnv(max_steps=100, pre_programmed=False)
    env.reset()
    
    # Find agents
    human_idx = None
    robot_idx = None
    for i, agent in enumerate(env.agents):
        if agent.color == 'yellow':
            human_idx = i
        elif agent.color == 'grey':
            robot_idx = i
    
    print(f"Initial state:")
    print(f"  Robot at: {tuple(env.agents[robot_idx].pos)}, dir={env.agents[robot_idx].dir}")
    print(f"  Human at: {tuple(env.agents[human_idx].pos)}")
    print()
    
    frames = []
    
    try:
        import matplotlib.pyplot as plt
        
        # Capture initial frame
        img = env.render(mode='rgb_array')
        frames.append(('Initial: Robot ready to program', img))
        
        actions = [Actions.still] * len(env.agents)
        
        # === Program upper button (2, 2) with 'left' ===
        print("Programming upper button (2,2) with 'left'...")
        
        # Turn to face north
        actions[robot_idx] = Actions.left
        env.step(actions)
        actions[robot_idx] = Actions.left
        env.step(actions)
        print(f"  Robot turned to face north, dir={env.agents[robot_idx].dir}")
        
        img = env.render(mode='rgb_array')
        frames.append(('Robot faces upper button', img))
        
        # Toggle to enter programming mode
        actions[robot_idx] = Actions.toggle
        env.step(actions)
        print(f"  Robot toggled button to enter programming mode")
        
        # Program 'left' action
        actions[robot_idx] = Actions.left
        env.step(actions)
        print(f"  Robot performed 'left' - button now programmed")
        
        img = env.render(mode='rgb_array')
        frames.append(('Upper button: "left" programmed', img))
        
        # === Program lower button (2, 4) with 'right' ===
        print("\nProgramming lower button (2,4) with 'right'...")
        
        # Turn to face south
        actions[robot_idx] = Actions.left
        env.step(actions)
        print(f"  Robot turned to face south, dir={env.agents[robot_idx].dir}")
        
        img = env.render(mode='rgb_array')
        frames.append(('Robot faces lower button', img))
        
        # Toggle to enter programming mode
        actions[robot_idx] = Actions.toggle
        env.step(actions)
        print(f"  Robot toggled button to enter programming mode")
        
        # Program 'right' action
        actions[robot_idx] = Actions.right
        env.step(actions)
        print(f"  Robot performed 'right' - button now programmed")
        
        img = env.render(mode='rgb_array')
        frames.append(('Lower button: "rght" programmed', img))
        
        # === Program right button (3, 3) with 'forward' ===
        print("\nProgramming right button (3,3) with 'forward'...")
        
        # Turn to face east
        actions[robot_idx] = Actions.left
        env.step(actions)
        actions[robot_idx] = Actions.left
        env.step(actions)
        print(f"  Robot turned to face east, dir={env.agents[robot_idx].dir}")
        
        img = env.render(mode='rgb_array')
        frames.append(('Robot faces right button', img))
        
        # Toggle to enter programming mode
        actions[robot_idx] = Actions.toggle
        env.step(actions)
        print(f"  Robot toggled button to enter programming mode")
        
        # Program 'forward' action
        actions[robot_idx] = Actions.forward
        env.step(actions)
        print(f"  Robot performed 'forward' - button now programmed")
        
        img = env.render(mode='rgb_array')
        frames.append(('All buttons programmed!', img))
        
        # Check final button states
        print("\nFinal button states (programmed):")
        action_names = {1: 'left', 2: 'right', 3: 'forward', 6: 'toggle'}
        for j in range(env.grid.height):
            for i in range(env.grid.width):
                cell = env.grid.get(i, j)
                if cell is not None and cell.type == 'controlbutton':
                    action_name = action_names.get(cell.triggered_action, str(cell.triggered_action)) if cell.triggered_action else "None"
                    print(f"  Button at ({i}, {j}): action={action_name}, controlled_agent={cell.controlled_agent}")
        
        # Save frames as individual images
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        for idx, (title, img) in enumerate(frames):
            ax = axes[idx // 4, idx % 4]
            ax.imshow(img)
            ax.set_title(title, fontsize=10)
            ax.axis('off')
        # Hide unused axes
        for idx in range(len(frames), 8):
            axes[idx // 4, idx % 4].axis('off')
        plt.tight_layout()
        prequel_img_path = os.path.join(OUTPUT_DIR, 'control_button_prequel.png')
        plt.savefig(prequel_img_path, dpi=150, bbox_inches='tight')
        print(f"\nSaved prequel frames to {prequel_img_path}")
        plt.close()
        
        # Save as movie/GIF
        movie_path = os.path.join(OUTPUT_DIR, 'control_button_prequel.gif')
        create_movie(frames, movie_path, fps=1)
        
    except ImportError as e:
        print(f"(matplotlib not available for rendering: {e})")
    
    print()
    return env, frames


def demonstrate_human_control_with_movie():
    """
    Demonstrate human using control buttons to control robot, saving as a movie.
    """
    print("=" * 70)
    print("Human Control Demonstration: Using Programmed Buttons")
    print("=" * 70)
    print()
    
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Create environment with pre-programmed buttons
    env = ControlButtonEnv(max_steps=100, pre_programmed=True)
    env.reset()
    
    # Find agents
    human_idx = None
    robot_idx = None
    for i, agent in enumerate(env.agents):
        if agent.color == 'yellow':
            human_idx = i
        elif agent.color == 'grey':
            robot_idx = i
    
    print(f"Initial state:")
    print(f"  Human at: {tuple(env.agents[human_idx].pos)}, dir={env.agents[human_idx].dir}")
    print(f"  Robot at: {tuple(env.agents[robot_idx].pos)}, dir={env.agents[robot_idx].dir}")
    print()
    
    frames = []
    
    try:
        import matplotlib.pyplot as plt
        
        actions = [Actions.still] * len(env.agents)
        
        # Capture initial frame
        img = env.render(mode='rgb_array')
        frames.append(('Initial: Buttons programmed', img))
        
        # Human needs to move to face a button
        # Human at (1,1), needs to get to position to toggle buttons
        
        # Move human to (2,2) area - but can't go through button
        # Let's have human turn and move
        print("Human moving into position...")
        
        # Turn right to face south
        actions[human_idx] = Actions.right
        env.step(actions)
        img = env.render(mode='rgb_array')
        frames.append(('Human turns south', img))
        
        # Move forward
        actions[human_idx] = Actions.forward
        env.step(actions)
        img = env.render(mode='rgb_array')
        frames.append(('Human moves to (1,2)', img))
        
        # Move forward again
        actions[human_idx] = Actions.forward
        env.step(actions)
        img = env.render(mode='rgb_array')
        frames.append(('Human at (1,3)', img))
        
        # Turn right to face west... no, east to face buttons
        actions[human_idx] = Actions.left
        env.step(actions)
        img = env.render(mode='rgb_array')
        frames.append(('Human faces east', img))
        
        # Move forward to get closer
        actions[human_idx] = Actions.forward
        env.step(actions)
        img = env.render(mode='rgb_array')
        frames.append(('Human at (2,3)', img))
        
        # Now human can toggle the right button at (3,3)
        print("\nHuman toggles right button (programmed for 'forward')...")
        actions[human_idx] = Actions.toggle
        actions[robot_idx] = Actions.still
        env.step(actions)
        print(f"  Robot forced_next_action: {env.agents[robot_idx].forced_next_action}")
        img = env.render(mode='rgb_array')
        frames.append(('Human triggers "fore" button', img))
        
        # Robot's next action is forced
        print("Robot executes forced 'forward' action...")
        robot_pos_before = tuple(env.agents[robot_idx].pos)
        actions[human_idx] = Actions.still
        actions[robot_idx] = Actions.still  # Robot "chooses" still, but forced to forward
        env.step(actions)
        robot_pos_after = tuple(env.agents[robot_idx].pos)
        print(f"  Robot moved: {robot_pos_before} -> {robot_pos_after}")
        img = env.render(mode='rgb_array')
        frames.append(('Robot moves forward!', img))
        
        # Human turns to face upper button
        print("\nHuman toggles upper button (programmed for 'left')...")
        actions[human_idx] = Actions.left  # Turn north
        env.step(actions)
        actions[human_idx] = Actions.toggle
        env.step(actions)
        print(f"  Robot forced_next_action: {env.agents[robot_idx].forced_next_action}")
        img = env.render(mode='rgb_array')
        frames.append(('Human triggers "left" button', img))
        
        # Robot turns left
        robot_dir_before = env.agents[robot_idx].dir
        actions[human_idx] = Actions.still
        actions[robot_idx] = Actions.still
        env.step(actions)
        robot_dir_after = env.agents[robot_idx].dir
        print(f"  Robot turned: dir {robot_dir_before} -> {robot_dir_after}")
        img = env.render(mode='rgb_array')
        frames.append(('Robot turns left!', img))
        
        # Save frames as individual images
        fig, axes = plt.subplots(2, 5, figsize=(25, 10))
        for idx, (title, img) in enumerate(frames):
            ax = axes[idx // 5, idx % 5]
            ax.imshow(img)
            ax.set_title(title, fontsize=10)
            ax.axis('off')
        # Hide unused axes
        for idx in range(len(frames), 10):
            axes[idx // 5, idx % 5].axis('off')
        plt.tight_layout()
        control_img_path = os.path.join(OUTPUT_DIR, 'control_button_human_control.png')
        plt.savefig(control_img_path, dpi=150, bbox_inches='tight')
        print(f"\nSaved control frames to {control_img_path}")
        plt.close()
        
        # Save as movie/GIF
        movie_path = os.path.join(OUTPUT_DIR, 'control_button_human_control.gif')
        create_movie(frames, movie_path, fps=1)
        
    except ImportError as e:
        print(f"(matplotlib not available for rendering: {e})")
    
    print()
    return env, frames


def create_full_rollout_movie():
    """
    Create a complete rollout movie showing:
    1. Prequel: Robot programs all buttons (handcrafted policy)
    2. Human control phase: Human uses buttons to control robot
    """
    print("=" * 70)
    print("Full Rollout Movie: Prequel + Human Control")
    print("=" * 70)
    print()
    
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    all_frames = []
    
    # Create environment WITHOUT pre-programming for full demo
    env = ControlButtonEnv(max_steps=100, pre_programmed=False)
    env.reset()
    
    # Find agents
    human_idx = None
    robot_idx = None
    for i, agent in enumerate(env.agents):
        if agent.color == 'yellow':
            human_idx = i
        elif agent.color == 'grey':
            robot_idx = i
    
    try:
        import matplotlib.pyplot as plt
        
        actions = [Actions.still] * len(env.agents)
        
        # ========== PREQUEL PHASE ==========
        print("=== PREQUEL: Robot programs buttons ===")
        
        img = env.render(mode='rgb_array')
        all_frames.append(('PREQUEL: Initial state', img))
        
        # Program upper button with 'left'
        actions[robot_idx] = Actions.left
        env.step(actions)
        actions[robot_idx] = Actions.left
        env.step(actions)
        actions[robot_idx] = Actions.toggle
        env.step(actions)
        actions[robot_idx] = Actions.left
        env.step(actions)
        img = env.render(mode='rgb_array')
        all_frames.append(('PREQUEL: Upper = "left"', img))
        
        # Program lower button with 'right'
        actions[robot_idx] = Actions.left
        env.step(actions)
        actions[robot_idx] = Actions.toggle
        env.step(actions)
        actions[robot_idx] = Actions.right
        env.step(actions)
        img = env.render(mode='rgb_array')
        all_frames.append(('PREQUEL: Lower = "rght"', img))
        
        # Program right button with 'forward'
        actions[robot_idx] = Actions.left
        env.step(actions)
        actions[robot_idx] = Actions.left
        env.step(actions)
        actions[robot_idx] = Actions.toggle
        env.step(actions)
        actions[robot_idx] = Actions.forward
        env.step(actions)
        img = env.render(mode='rgb_array')
        all_frames.append(('PREQUEL: Right = "fore"', img))
        
        print("  Buttons programmed!")
        
        # ========== HUMAN CONTROL PHASE ==========
        print("\n=== HUMAN CONTROL: Using buttons ===")
        
        img = env.render(mode='rgb_array')
        all_frames.append(('CONTROL: Ready', img))
        
        # Human moves into position
        actions[human_idx] = Actions.right
        actions[robot_idx] = Actions.still
        env.step(actions)
        actions[human_idx] = Actions.forward
        env.step(actions)
        actions[human_idx] = Actions.forward
        env.step(actions)
        actions[human_idx] = Actions.left
        env.step(actions)
        actions[human_idx] = Actions.forward
        env.step(actions)
        img = env.render(mode='rgb_array')
        all_frames.append(('CONTROL: Human in position', img))
        
        # Human triggers right button (forward)
        actions[human_idx] = Actions.toggle
        env.step(actions)
        img = env.render(mode='rgb_array')
        all_frames.append(('CONTROL: Trigger "fore"', img))
        
        # Robot executes forced forward
        actions[human_idx] = Actions.still
        env.step(actions)
        img = env.render(mode='rgb_array')
        all_frames.append(('CONTROL: Robot forward!', img))
        
        # Human triggers upper button (left)
        actions[human_idx] = Actions.left
        env.step(actions)
        actions[human_idx] = Actions.toggle
        env.step(actions)
        img = env.render(mode='rgb_array')
        all_frames.append(('CONTROL: Trigger "left"', img))
        
        # Robot executes forced left
        actions[human_idx] = Actions.still
        env.step(actions)
        img = env.render(mode='rgb_array')
        all_frames.append(('CONTROL: Robot turns left!', img))
        
        print("  Control demonstration complete!")
        
        # Save as movie
        movie_path = os.path.join(OUTPUT_DIR, 'control_button_full_rollout.gif')
        create_movie(all_frames, movie_path, fps=1)
        
        # Save summary image
        n_frames = len(all_frames)
        cols = min(5, n_frames)
        rows = (n_frames + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 5*rows))
        axes = axes.flatten() if n_frames > 1 else [axes]
        for idx, (title, img) in enumerate(all_frames):
            axes[idx].imshow(img)
            axes[idx].set_title(title, fontsize=9)
            axes[idx].axis('off')
        for idx in range(len(all_frames), len(axes)):
            axes[idx].axis('off')
        plt.tight_layout()
        summary_path = os.path.join(OUTPUT_DIR, 'control_button_full_rollout.png')
        plt.savefig(summary_path, dpi=150, bbox_inches='tight')
        print(f"\nSaved rollout summary to {summary_path}")
        plt.close()
        
    except ImportError as e:
        print(f"(matplotlib not available: {e})")
    
    print()
    return all_frames


def main():
    """Main function to run the control button demo."""
    
    print("=" * 70)
    print("Control Button Demo")
    print("=" * 70)
    print()
    print("This demo shows how ControlButton objects work:")
    print("1. Robot (grey) programs buttons with specific actions")
    print("2. Human (yellow) can then trigger these buttons")
    print("3. When triggered, the robot performs the programmed action")
    print("   on the NEXT step (forced_next_action mechanism)")
    print()
    print("Button layout after programming:")
    print("  - Upper button → 'left' action")
    print("  - Right button → 'fore' action")
    print("  - Lower button → 'rght' action")
    print()
    print(f"Output directory: {OUTPUT_DIR}")
    print()
    
    # Create full rollout movie
    create_full_rollout_movie()
    
    print("=" * 70)
    print("Demo Complete")
    print("=" * 70)
    print()
    print(f"Check {OUTPUT_DIR} for generated movies and images:")
    print("  - control_button_full_rollout.gif - Complete demo movie")
    print("  - control_button_full_rollout.png - Summary image")


if __name__ == "__main__":
    main()

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

The human learns policies for 2 goals:
- Goal 1: Get the robot to position (4, 1) - top position
- Goal 2: Get the robot to position (4, 5) - bottom position

Workflow:
1. Prequel phase: Robot programs buttons and steps aside to (4,3) facing right, human moves to (2,3)
2. Learning phase: Train neural network for human to reach goal positions via button control
3. Rollout phase: Show prequel + human following learned policy

This demonstrates human-robot control via programmable control interfaces.
"""

import sys
import os
import time
import random

# Add paths for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'vendor', 'multigrid'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


# Patch gym import for compatibility
import gymnasium as gym
sys.modules['gym'] = gym

from gym_multigrid.multigrid import MultiGridEnv, World, Actions
from empo.possible_goal import PossibleGoal, PossibleGoalSampler
from empo.nn_based import train_neural_policy_prior

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
# - Robot (grey) at (4, 3) facing east - stepped aside
# - Buttons already programmed
READY_MAP = """
We We We We We We We We
We .. .. .. .. .. .. We
We .. CB .. .. Ro .. We
We .. Ay CB Ae Ro We We
We .. CB .. .. Ro .. We
We .. .. .. .. .. .. We
We We We We We We We We
"""


class ControlButtonEnv(MultiGridEnv):
    """
    Environment with control buttons that let humans control robot actions.
    
    Can be initialized in two modes:
    1. pre_programmed=False: Robot starts in position to program buttons (for prequel)
    2. pre_programmed=True: Buttons are already programmed, agents in ready positions
       - Human at (2,3) between buttons
       - Robot at (4,3) stepped aside, facing east
    
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


# ============================================================================
# Goal Definitions for Neural Network Learning
# ============================================================================

class RobotAtRockGoal(PossibleGoal):
    """A goal where the robot should reach a specific rock position."""
    
    def __init__(self, world_model, robot_agent_index: int, target_pos: tuple):
        super().__init__(world_model)
        self.robot_agent_index = robot_agent_index
        self.target_pos = tuple(target_pos)
    
    def is_achieved(self, state) -> int:
        """Returns 1 if the robot is at the target position."""
        step_count, agent_states, mobile_objects, mutable_objects = state
        if self.robot_agent_index < len(agent_states):
            agent_state = agent_states[self.robot_agent_index]
            if len(agent_state) >= 2:
                pos_x, pos_y = int(agent_state[0]), int(agent_state[1])
                if pos_x == self.target_pos[0] and pos_y == self.target_pos[1]:
                    return 1
        return 0
    
    def __str__(self):
        return f"RobotAt({self.target_pos[0]},{self.target_pos[1]})"
    
    def __repr__(self):
        return self.__str__()
    
    def __hash__(self):
        return hash((self.robot_agent_index, self.target_pos[0], self.target_pos[1]))
    
    def __eq__(self, other):
        if not isinstance(other, RobotAtRockGoal):
            return False
        return (self.robot_agent_index == other.robot_agent_index and 
                self.target_pos == other.target_pos)


class RockPositionGoalSampler(PossibleGoalSampler):
    """
    A goal sampler that samples from the 3 rock positions.
    """
    
    def __init__(self, world_model, robot_idx: int, rock_positions: list):
        super().__init__(world_model)
        self.robot_idx = robot_idx
        self.rock_positions = rock_positions
    
    def sample(self, state, human_agent_index: int) -> tuple:
        """Sample a random rock position as goal."""
        target_pos = random.choice(self.rock_positions)
        goal = RobotAtRockGoal(self.world_model, self.robot_idx, target_pos)
        return goal, 1.0


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
        frames.append(('Lower button: "right" programmed', img))
        
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
        frames.append(('Human triggers "forward" button', img))
        
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
    1. Prequel: Robot programs all buttons, steps aside to (1,5)
    2. Prequel: Human moves to position (2,3) between buttons
    3. Human control phase: Human uses buttons to control robot (following learned/demo policy)
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
    
    print(f"Initial positions:")
    print(f"  Human (yellow): {tuple(env.agents[human_idx].pos)}, dir={env.agents[human_idx].dir}")
    print(f"  Robot (grey): {tuple(env.agents[robot_idx].pos)}, dir={env.agents[robot_idx].dir}")
    
    try:
        import matplotlib.pyplot as plt
        
        actions = [Actions.still] * len(env.agents)
        
        # ========== PREQUEL PHASE 1: ROBOT PROGRAMS BUTTONS ==========
        print("\n=== PREQUEL: Robot programs buttons ===")
        
        img = env.render(mode='rgb_array')
        all_frames.append(('PREQUEL: Initial state', img))
        
        # Robot at (2, 3) facing south (dir=1)
        # Upper button at (2, 2) is north of robot
        
        # Turn left twice to face north (dir=3)
        actions[robot_idx] = Actions.left
        env.step(actions)
        img = env.render(mode='rgb_array')
        all_frames.append(('', img))
        actions[robot_idx] = Actions.left
        env.step(actions)
        img = env.render(mode='rgb_array')
        all_frames.append(('', img))
        # Toggle upper button (2, 2)
        actions[robot_idx] = Actions.toggle
        env.step(actions)
        # Program 'left' action (robot now faces west)
        actions[robot_idx] = Actions.left
        env.step(actions)
        img = env.render(mode='rgb_array')
        all_frames.append(('PREQUEL: Upper = "L"', img))
        print(f"  Upper button (2,2) programmed with 'left'")
        
        # Now facing west (dir=2). Turn left to face south (dir=1)
        actions[robot_idx] = Actions.left
        env.step(actions)
        img = env.render(mode='rgb_array')
        all_frames.append(('', img))
        # Toggle lower button (2, 4)
        actions[robot_idx] = Actions.toggle
        env.step(actions)
        img = env.render(mode='rgb_array')
        all_frames.append(('', img))
        # Program 'right' action (robot now faces west again)
        actions[robot_idx] = Actions.right
        env.step(actions)
        img = env.render(mode='rgb_array')
        all_frames.append(('PREQUEL: Lower = "R"', img))
        print(f"  Lower button (2,4) programmed with 'right'")
        
        # Now facing west (dir=2). Need to face east to toggle right button (3,3)
        # Turn left twice: west -> south -> east
        actions[robot_idx] = Actions.left
        env.step(actions)
        img = env.render(mode='rgb_array')
        all_frames.append(('', img))
        actions[robot_idx] = Actions.left
        env.step(actions)
        img = env.render(mode='rgb_array')
        all_frames.append(('', img))
        # Toggle right button (3, 3)
        actions[robot_idx] = Actions.toggle
        env.step(actions)
        img = env.render(mode='rgb_array')
        all_frames.append(('', img))
        # Program 'forward' action - robot stays in place (button blocks movement)
        actions[robot_idx] = Actions.forward
        env.step(actions)
        img = env.render(mode='rgb_array')
        all_frames.append(('PREQUEL: Right = "F"', img))
        print(f"  Right button (3,3) programmed with 'forward'")
        print(f"  Robot at: {tuple(env.agents[robot_idx].pos)}, dir={env.agents[robot_idx].dir}")
        
        print("  All buttons programmed!")
        
        # ========== PREQUEL PHASE 2: ROBOT STEPS ASIDE TO (4, 3) FACING EAST ==========
        print("\n=== PREQUEL: Robot steps aside to (4,3) facing right ===")
        
        # Robot now at (2, 3) facing east (dir=0). Need to go to (4, 3) facing east.
        # Move forward twice to (4, 3), keep facing east
        actions[robot_idx] = Actions.forward  # (2,3) -> blocked by button at (3,3)
        env.step(actions)
        img = env.render(mode='rgb_array')
        all_frames.append(('', img))
        # Need to go around: turn right (face south), go to (2,4), turn left (face east), go to (4,4), turn left (face north), go to (4,3), turn right (face east)
        # Or simpler: stay at (2,3), turn to face east, let human control phase proceed
        # Actually the robot needs to move. Let's do: go around the button
        
        # Turn right to face south
        actions[robot_idx] = Actions.right  # east -> south
        env.step(actions)
        img = env.render(mode='rgb_array')
        all_frames.append(('', img))
        # Move forward to (2, 4)
        actions[robot_idx] = Actions.forward
        env.step(actions)
        img = env.render(mode='rgb_array')
        all_frames.append(('', img))
        # Turn left to face east
        actions[robot_idx] = Actions.left  # south -> east
        env.step(actions)
        img = env.render(mode='rgb_array')
        all_frames.append(('', img))
        # Move forward to (3, 4)
        actions[robot_idx] = Actions.forward
        env.step(actions)
        img = env.render(mode='rgb_array')
        all_frames.append(('', img))
        # Move forward to (4, 4)
        actions[robot_idx] = Actions.forward
        env.step(actions)
        img = env.render(mode='rgb_array')
        all_frames.append(('', img))
        # Turn left to face north
        actions[robot_idx] = Actions.left  # east -> north
        env.step(actions)
        img = env.render(mode='rgb_array')
        all_frames.append(('', img))
        # Move forward to (4, 3)
        actions[robot_idx] = Actions.forward
        env.step(actions)
        img = env.render(mode='rgb_array')
        all_frames.append(('', img))
        # Turn right to face east
        actions[robot_idx] = Actions.right  # north -> east
        env.step(actions)
        
        img = env.render(mode='rgb_array')
        all_frames.append(('PREQUEL: Robot at (4,3) facing right', img))
        print(f"  Robot now at: {tuple(env.agents[robot_idx].pos)}, dir={env.agents[robot_idx].dir}")
        
        # ========== PREQUEL PHASE 3: HUMAN MOVES TO (2, 3) ==========
        print("\n=== PREQUEL: Human moves to (2,3) ===")
        
        # Human is at (1, 1) facing up (dir=3). Need to get to (2, 3).
        # Path: (1,1) -> (1,2) -> (1,3) -> (2,3)
        
        actions[robot_idx] = Actions.still  # Robot stays still now
        
        # Turn right to face east (dir=0)
        actions[human_idx] = Actions.right
        env.step(actions)
        img = env.render(mode='rgb_array')
        all_frames.append(('', img))
        # Turn right to face south (dir=1)
        actions[human_idx] = Actions.right
        env.step(actions)
        img = env.render(mode='rgb_array')
        all_frames.append(('', img))
        # Move forward to (1, 2)
        actions[human_idx] = Actions.forward
        env.step(actions)
        img = env.render(mode='rgb_array')
        all_frames.append(('', img))
        # Move forward to (1, 3)
        actions[human_idx] = Actions.forward
        env.step(actions)
        img = env.render(mode='rgb_array')
        all_frames.append(('', img))
        # Turn left to face east (dir=0)
        actions[human_idx] = Actions.left
        env.step(actions)
        img = env.render(mode='rgb_array')
        all_frames.append(('', img))
        # Move forward to (2, 3)
        actions[human_idx] = Actions.forward
        env.step(actions)
        
        img = env.render(mode='rgb_array')
        all_frames.append(('PREQUEL: Human at (2,3) - READY!', img))
        print(f"  Human now at: {tuple(env.agents[human_idx].pos)}")
        print(f"  Robot at: {tuple(env.agents[robot_idx].pos)}, dir={env.agents[robot_idx].dir}")
        print(f"\n  === READY STATE ACHIEVED ===")
        print(f"  Human at (2,3) between buttons, Robot at (4,3) facing right")
        
        # ========== HUMAN CONTROL PHASE ==========
        print("\n=== CONTROL: Human uses buttons ===")
        
        # Human at (2, 3) facing east (dir=0) can toggle:
        # - Right button (3, 3) -> forward (face east and toggle)
        # - Upper button (2, 2) -> left (face north and toggle)
        # - Lower button (2, 4) -> right (face south and toggle)
        
        # Demo: trigger forward button to move robot forward
        # Human facing east, toggle right button
        actions[human_idx] = Actions.toggle
        env.step(actions)
        img = env.render(mode='rgb_array')
        all_frames.append(('CONTROL: Human triggers "F"', img))
        print(f"  Human triggers forward button")
        
        # Robot executes forced forward
        actions[human_idx] = Actions.still
        env.step(actions)  # Robot forced forward
        img = env.render(mode='rgb_array')
        all_frames.append(('CONTROL: Robot moves forward!', img))
        print(f"  Robot moved to: {tuple(env.agents[robot_idx].pos)}")
        
        # Human triggers upper button (left)
        actions[human_idx] = Actions.left  # Face north
        env.step(actions)
        actions[human_idx] = Actions.toggle  # Toggle upper button
        env.step(actions)
        img = env.render(mode='rgb_array')
        all_frames.append(('CONTROL: Human triggers "L"', img))
        print(f"  Human triggers left button")
        
        # Robot turns left
        actions[human_idx] = Actions.still
        env.step(actions)
        img = env.render(mode='rgb_array')
        all_frames.append(('CONTROL: Robot turns left!', img))
        print(f"  Robot turned left, dir={env.agents[robot_idx].dir}")
        
        # Human triggers lower button (right)
        # Human facing north, turn to face south
        actions[human_idx] = Actions.left  # Face west
        env.step(actions)
        img = env.render(mode='rgb_array')
        all_frames.append(('', img))
        actions[human_idx] = Actions.left  # Face south
        env.step(actions)
        img = env.render(mode='rgb_array')
        all_frames.append(('', img))
        actions[human_idx] = Actions.toggle  # Toggle lower button
        env.step(actions)
        img = env.render(mode='rgb_array')
        all_frames.append(('CONTROL: Human triggers "R"', img))
        print(f"  Human triggers right button")
        
        # Robot turns right
        actions[human_idx] = Actions.still
        env.step(actions)
        img = env.render(mode='rgb_array')
        all_frames.append(('CONTROL: Robot turns right!', img))
        print(f"  Robot turned right, dir={env.agents[robot_idx].dir}")
        
        print("\n  Control demonstration complete!")
        
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


# Configuration for quick mode vs full mode
N_EPISODES_FULL = 2000  # Training episodes for full mode
N_EPISODES_QUICK = 50   # Training episodes for quick test


def train_and_rollout_with_learned_policy(quick_mode=False):
    """
    Train a neural network for the human to learn to reach goal positions,
    then demonstrate rollouts with prequel + learned policy.
    
    Uses train_neural_policy_prior() to train the policy, then neural_prior.sample()
    to get actions during rollouts.
    """
    n_episodes = N_EPISODES_QUICK if quick_mode else N_EPISODES_FULL
    
    print("=" * 70)
    print("Neural Network Learning: Human learns to guide robot to goals")
    print("=" * 70)
    print()
    
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Create environment with pre-programmed buttons (ready state)
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
    
    print(f"Environment ready state:")
    print(f"  Human (yellow) at: {tuple(env.agents[human_idx].pos)}")
    print(f"  Robot (grey) at: {tuple(env.agents[robot_idx].pos)}")
    print(f"  Action space: {env.action_space.n} actions")
    print()
    
    # Two goals: robot at (4, 1) and (4, 5) - easier to learn
    goal_cells = [(4, 1), (4, 5)]
    
    # Create goal sampler
    goal_sampler = RockPositionGoalSampler(env, robot_idx, goal_cells)
    
    print(f"Training neural network for {len(goal_cells)} goal positions...")
    print(f"  Goals: {goal_cells}")
    print(f"  Training episodes: {n_episodes}")
    print("  Human learns to guide robot to goal positions via control buttons")
    print()
    
    device = 'cpu'
    beta = 100.0  # High beta for deterministic behavior (safe with bounded Q in [0,1])
    
    t0 = time.time()
    try:
        # Train the neural policy prior
        neural_prior = train_neural_policy_prior(
            world_model=env,
            human_agent_indices=[human_idx],
            goal_sampler=goal_sampler,
            num_episodes=n_episodes,
            steps_per_episode=50,
            beta=beta,
            gamma=0.99,  # Standard discount factor
            learning_rate=1e-3,  # Can use higher LR now with bounded Q
            batch_size=128,  # Larger batch size
            replay_buffer_size=10000,
            updates_per_episode=4,
            train_phi_network=False,
            epsilon=0.3,
            device=device,
            verbose=True
        )
        elapsed = time.time() - t0
        print(f"  Training completed in {elapsed:.2f} seconds")
        
        # Run rollouts with learned policy using neural_prior.sample()
        print("\nRunning rollouts with learned policy...")
        
        import matplotlib.pyplot as plt
        
        all_rollout_frames = []
        
        for goal_idx, goal_pos in enumerate(goal_cells):
            print(f"\n  Rollout for goal: robot at {goal_pos}")
            
            env.reset()
            rollout_frames = []
            
            # Create goal object for this rollout
            goal = RobotAtRockGoal(env, robot_idx, goal_pos)
            
            for step in range(min(30, env.max_steps)):  # Limit steps for demo
                state = env.get_state()
                img = env.render(mode='rgb_array')
                rollout_frames.append((f'Goal: Robot at {goal_pos} | Step {step}', img))
                
                # Get human action from learned policy using sample()
                human_action = neural_prior.sample(state, human_idx, goal)
                
                # Robot stays still (human controls via buttons)
                actions = [Actions.still] * len(env.agents)
                actions[human_idx] = human_action
                
                _, _, done, _ = env.step(actions)
                
                # Check if goal achieved
                robot_state = state[1][robot_idx]
                if int(robot_state[0]) == goal_pos[0] and int(robot_state[1]) == goal_pos[1]:
                    print(f"    Goal achieved at step {step}!")
                    break
                
                if done:
                    break
            
            all_rollout_frames.extend(rollout_frames)
        
        # Save rollouts movie
        movie_path = os.path.join(OUTPUT_DIR, 'control_button_learned_rollouts.gif')
        create_movie(all_rollout_frames, movie_path, fps=2)
        
        return neural_prior
        
    except Exception as e:
        print(f"  Neural network training failed: {e}")
        print("  (This may be due to action space mismatch or missing dependencies)")
        import traceback
        traceback.print_exc()
        return None


def main(quick_mode=False):
    """Main function to run the control button demo."""
    mode_str = "QUICK TEST MODE" if quick_mode else "FULL MODE"
    
    print("=" * 70)
    print("Control Button Demo with Neural Policy Learning")
    print(f"  [{mode_str}]")
    print("=" * 70)
    print()
    print("This demo shows how ControlButton objects work:")
    print("1. Robot (grey) programs buttons with specific actions")
    print("2. Robot steps aside to (1,5)")
    print("3. Human (yellow) moves to position (2,3) between buttons")
    print("4. Human triggers buttons to control robot movement")
    print()
    print("Button layout after programming:")
    print("  - Upper button (2,2) → 'L' (left) action")
    print("  - Right button (3,3) → 'F' (forward) action")
    print("  - Lower button (2,4) → 'R' (right) action")
    print()
    print(f"Output directory: {OUTPUT_DIR}")
    print()
    
    # Part 1: Create full rollout movie with prequel
    create_full_rollout_movie()
    
    # Part 2: Try to train neural network and run learned policy
    print()
    try:
        train_and_rollout_with_learned_policy(quick_mode=quick_mode)
    except Exception as e:
        print(f"Neural network learning skipped: {e}")
    
    print()
    print("=" * 70)
    print("Demo Complete")
    print("=" * 70)
    print()
    print(f"Check {OUTPUT_DIR} for generated movies and images:")
    print("  - control_button_full_rollout.gif - Complete prequel + control demo")
    print("  - control_button_full_rollout.png - Summary image")
    print("  - control_button_learned_rollouts.gif - Neural network learned policy (if trained)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Control Button Demo')
    parser.add_argument('--quick', '-q', action='store_true',
                        help='Run in quick test mode with fewer training episodes')
    args = parser.parse_args()
    main(quick_mode=args.quick)

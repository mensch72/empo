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

# ============================================================================
# Environment Definition
# ============================================================================

# Layout after robot has programmed buttons and stepped out of the way:
# - Human (yellow) in between buttons at (2, 3)
# - Robot (grey) out of the way at (4, 3)
# - Upper button at (2, 2) -> left
# - Right button at (3, 3) -> forward  
# - Lower button at (2, 4) -> right
# - Rocks at (5, 2), (5, 3), (5, 4)
CONTROL_BUTTON_MAP = """
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
    
    After the prequel programming phase:
    - Human (yellow) is between the buttons at (2, 3)
    - Robot (grey) is out of the way at (4, 3)
    - 3 Control Buttons:
      - Upper (2, 2) -> left action
      - Right (3, 3) -> forward action
      - Lower (2, 4) -> right action
    - 3 Rocks at (5, 2), (5, 3), (5, 4)
    
    The robot can push rocks (has can_push_rocks=True).
    """
    
    def __init__(self, max_steps: int = 100, pre_programmed: bool = True):
        """
        Args:
            max_steps: Maximum steps per episode
            pre_programmed: If True, buttons start pre-programmed (simulating robot already programmed them)
        """
        self.pre_programmed = pre_programmed
        super().__init__(
            map=CONTROL_BUTTON_MAP,
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


def get_prequel_actions():
    """
    Get the sequence of actions for the prequel where robot programs the buttons.
    
    The robot starts at the same position as in the map and needs to:
    1. Move to face each button
    2. Toggle to enter programming mode
    3. Perform the action to program
    4. Move to the next button
    5. Finally step out of the way
    
    Returns:
        List of (human_action, robot_action) tuples for each step
    """
    # This is a simplified prequel - in a full implementation, the robot would
    # navigate to each button and program it. For now, we assume pre-programming.
    # The actual prequel steps would depend on the initial positions.
    return []


def demonstrate_control_button():
    """
    Simple demonstration of how control buttons work.
    """
    print("=" * 70)
    print("Control Button Demonstration")
    print("=" * 70)
    print()
    
    # Create environment with pre-programmed buttons
    env = ControlButtonEnv(max_steps=100, pre_programmed=True)
    env.reset()
    
    print("Environment layout (after robot programming phase):")
    print("  Yellow (Human): In between control buttons, can trigger them")
    print("  Grey (Robot): Stepped aside, controlled by buttons, can push rocks")
    print("  Green squares: Control buttons with action indicators:")
    print("    - Upper button (2,2): LEFT action (arrow pointing left)")
    print("    - Right button (3,3): FORWARD action (arrow pointing up)")
    print("    - Lower button (2,4): RIGHT action (arrow pointing right)")
    print("  Grey circles: Rocks (can be pushed by robot)")
    print("  Green dashed lines: Connection from buttons to controlled robot")
    print()
    
    # Find agents
    human_idx = None
    robot_idx = None
    for i, agent in enumerate(env.agents):
        if agent.color == 'yellow':
            human_idx = i
        elif agent.color == 'grey':
            robot_idx = i
    
    print(f"Human (yellow) at: {tuple(env.agents[human_idx].pos)}, dir={env.agents[human_idx].dir}")
    print(f"Robot (grey) at: {tuple(env.agents[robot_idx].pos)}, dir={env.agents[robot_idx].dir}")
    print()
    
    # Find control buttons and their programming
    print("Control buttons (pre-programmed):")
    action_names = {1: 'left', 2: 'right', 3: 'forward', 6: 'toggle'}
    for j in range(env.grid.height):
        for i in range(env.grid.width):
            cell = env.grid.get(i, j)
            if cell is not None and cell.type == 'controlbutton':
                action_name = action_names.get(cell.triggered_action, str(cell.triggered_action)) if cell.triggered_action is not None else "None"
                print(f"  Button at ({i}, {j}): action={action_name}, controlled_agent={cell.controlled_agent}")
    print()
    
    # Render the environment
    try:
        import matplotlib.pyplot as plt
        img = env.render(mode='rgb_array')
        plt.figure(figsize=(8, 8))
        plt.imshow(img)
        plt.title("Control Button Environment\n(Dashed lines show button-to-robot connections)")
        plt.axis('off')
        plt.savefig('/tmp/control_button_env.png', dpi=150, bbox_inches='tight')
        print("Saved environment rendering to /tmp/control_button_env.png")
        plt.close()
    except ImportError:
        print("(matplotlib not available for rendering)")
    
    # Demonstrate: Human triggers upper button (left)
    print("\nDemonstration: Human triggers buttons to control robot")
    print("-" * 50)
    
    # Human already faces north (dir=3), directly facing upper button at (2,2)
    print("Step 1: Human is already facing upper button (north)")
    print(f"  Human direction: {env.agents[human_idx].dir} (3=north)")
    print(f"  Human front_pos: {tuple(env.agents[human_idx].front_pos)}")
    
    # Human toggles upper button
    print("\nStep 2: Human toggles upper button (programmed for 'left')")
    print(f"  Robot direction before: {env.agents[robot_idx].dir}")
    actions = [Actions.still] * len(env.agents)
    actions[human_idx] = Actions.toggle
    env.step(actions)
    print(f"  Robot forced_next_action set: {env.agents[robot_idx].forced_next_action}")
    
    # Robot's action is forced on next step
    print("\nStep 3: On next step, robot's action is FORCED to 'left'")
    robot_dir_before = env.agents[robot_idx].dir
    actions = [Actions.still] * len(env.agents)
    actions[robot_idx] = Actions.forward  # Robot "wants" to go forward
    env.step(actions)
    print(f"  Robot direction changed: {robot_dir_before} -> {env.agents[robot_idx].dir} (turned left!)")
    print()
    
    print("Demonstration complete!")
    print()
    return env


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
    print("  - Upper button → LEFT action")
    print("  - Right button → FORWARD action")
    print("  - Lower button → RIGHT action")
    print()
    print("The dashed lines in rendering show connections from buttons to robot.")
    print()
    
    # Run the demonstration
    demonstrate_control_button()
    
    # Show rock positions as potential goals
    env = ControlButtonEnv(max_steps=100, pre_programmed=True)
    env.reset()
    
    print("=" * 70)
    print("Potential Goals (Robot reaching rock positions)")
    print("=" * 70)
    print()
    
    rocks = []
    for j in range(env.grid.height):
        for i in range(env.grid.width):
            cell = env.grid.get(i, j)
            if cell is not None and cell.type == 'rock':
                rocks.append((i, j))
    
    for i, (x, y) in enumerate(rocks):
        print(f"  Goal {i+1}: Robot reaches position ({x}, {y})")
    
    print()
    print("The human can use control buttons to guide the robot to these goals.")
    print()
    print("=" * 70)
    print("Demo Complete")
    print("=" * 70)


if __name__ == "__main__":
    main()

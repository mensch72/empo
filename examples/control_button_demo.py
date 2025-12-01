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

CONTROL_BUTTON_MAP = """
We We We We We We We We
We Ay .. .. .. .. .. We
We .. CB .. .. Ro .. We
We Ae .. CB .. Ro We We
We .. CB .. .. Ro .. We
We .. .. .. .. .. .. We
We We We We We We We We
"""


class ControlButtonEnv(MultiGridEnv):
    """
    Environment with control buttons that let humans control robot actions.
    
    Layout:
    - Human (yellow) at (1, 1)
    - Robot (grey) at (1, 3)
    - 3 Control Buttons at (2, 2), (3, 3), (2, 4)
    - 3 Rocks at (5, 2), (5, 3), (5, 4)
    
    The robot can push rocks (has can_push_rocks=True).
    The robot programs the control buttons, then the human uses them.
    """
    
    def __init__(self, max_steps: int = 100, pre_programmed: bool = True):
        """
        Args:
            max_steps: Maximum steps per episode
            pre_programmed: If True, buttons start pre-programmed with left/forward/right
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
            # Pre-program the control buttons with actions
            # Button 1 (top): left
            # Button 2 (middle): forward
            # Button 3 (bottom): right
            button_actions = {
                (2, 2): Actions.left,     # Top button -> left
                (3, 3): Actions.forward,  # Middle button -> forward
                (2, 4): Actions.right,    # Bottom button -> right
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
    
    print("Environment layout:")
    print("  Yellow (Human): Can trigger control buttons")
    print("  Grey (Robot): Controlled by buttons, can push rocks")
    print("  Green squares: Control buttons (programmed with actions)")
    print("  Grey circles: Rocks (can be pushed by robot)")
    print()
    
    # Find agents
    human_idx = None
    robot_idx = None
    for i, agent in enumerate(env.agents):
        if agent.color == 'yellow':
            human_idx = i
        elif agent.color == 'grey':
            robot_idx = i
    
    print(f"Human (yellow) at: {tuple(env.agents[human_idx].pos)}")
    print(f"Robot (grey) at: {tuple(env.agents[robot_idx].pos)}")
    print(f"Robot direction: {env.agents[robot_idx].dir} (0=right, 1=down, 2=left, 3=up)")
    print()
    
    # Find control buttons and their programming
    print("Control buttons (pre-programmed):")
    for j in range(env.grid.height):
        for i in range(env.grid.width):
            cell = env.grid.get(i, j)
            if cell is not None and cell.type == 'controlbutton':
                action_name = Actions.available[cell.triggered_action] if cell.triggered_action is not None else "None"
                print(f"  Button at ({i}, {j}): action={action_name}")
    print()
    
    # Simulate the human using control buttons
    print("Demonstration: Human triggers buttons to control robot")
    print("-" * 50)
    
    # First, position the human to face a button
    # Human starts at (1,1) facing up (dir=3)
    # Need to move to face a button
    
    # Step 1: Human turns right to face east
    print("Step 1: Human turns right (to face east)")
    actions = [Actions.still] * len(env.agents)
    actions[human_idx] = Actions.right
    env.step(actions)
    print(f"  Human direction: {env.agents[human_idx].dir}")
    
    # Step 2: Human moves forward to (2, 1)
    print("Step 2: Human moves forward")
    actions = [Actions.still] * len(env.agents)
    actions[human_idx] = Actions.forward
    env.step(actions)
    print(f"  Human position: {tuple(env.agents[human_idx].pos)}")
    
    # Step 3: Human turns down to face the top button at (2, 2)
    print("Step 3: Human turns right (to face south)")
    actions = [Actions.still] * len(env.agents)
    actions[human_idx] = Actions.right
    env.step(actions)
    print(f"  Human direction: {env.agents[human_idx].dir}")
    
    # Step 4: Human toggles the button - this should make robot turn left
    print("Step 4: Human toggles top button (programmed for 'left')")
    print(f"  Robot direction before: {env.agents[robot_idx].dir}")
    actions = [Actions.still] * len(env.agents)
    actions[human_idx] = Actions.toggle
    env.step(actions)
    print(f"  Robot direction after: {env.agents[robot_idx].dir}")
    print(f"  (Robot should have turned left!)")
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
    print()
    print("Use case: Teaching humans to control robot behavior through")
    print("programmable interfaces.")
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

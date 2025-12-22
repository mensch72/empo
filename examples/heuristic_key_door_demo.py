#!/usr/bin/env python3
"""
Heuristic Key-Door Demo.

This script demonstrates the heuristic potential-based policy's ability to:
1. Pick up a key that can open locked doors
2. Navigate to and unlock doors using the key
3. Drop the key after it's no longer useful
4. Open closed (but unlocked) doors
5. NOT pick up keys that can't open any locked door
6. Reach the goal area

The demo uses a specific map:
```
We We We We We We We
We Kr Lr .. We .. We
We Ay .. .. Lr .. We
We Kg Cg .. We .. We
We We We We We We We 
```

Where:
- Ay = Yellow agent (human)
- Kr = Red key
- Kg = Green key (should NOT be picked up - no locked green doors)
- Lr = Locked red door
- Cg = Closed (unlocked) green door (should be opened)
- We = Wall

The human's goal is to reach the rightmost column ((5,1) to (5,3)).

Expected behavior:
1. Pick up the red key (Kr) - needed for locked doors
2. Open the first locked red door (Lr at (2,1))
3. Open the second locked red door (Lr at (4,2))
4. Drop the red key (no longer needed)
5. Move to the goal area
6. Do NOT pick up the green key (Kg) - no locked green doors
7. DO open the closed green door (Cg at (2,3)) - it's closed but unlocked

Note: Can't use 'Kb' for blue key because it conflicts with KillButton code.

Usage:
    python heuristic_key_door_demo.py

Requirements:
    - matplotlib
    - ffmpeg (optional, for MP4 output; falls back to GIF)
"""

import sys
import os
from typing import List, Dict, Optional

import numpy as np

from gym_multigrid.multigrid import (
    MultiGridEnv, World, Actions
)
from empo.multigrid import (
    ReachRectangleGoal,
    render_goals_on_frame,
)
from empo.human_policy_prior import HeuristicPotentialPolicy
from empo.nn_based.multigrid.path_distance import PathDistanceCalculator


# ============================================================================
# Configuration
# ============================================================================

# The specific map for this demo
# Note: Map coordinates are (x, y) where x is column, y is row
# Row 0 is top, column 0 is left
# 
# Map codes:
#   We = Wall (grey)
#   Ay = Yellow agent
#   Kr = Red key, Kg = Green key
#   Lr = Locked red door
#   Cg = Closed (unlocked) green door
#   .. = Empty floor
#
# Note: Can't use Kb for blue key because 'Kb' is reserved for KillButton
#
# Test scenario:
# - Agent should pick up red key (Kr) to open locked red doors (Lr)
# - Agent should NOT pick up green key (Kg) since green door (Cg) is unlocked
# - Agent should open the closed green door (Cg) on the way to goal
MAP_STR = """
We We We We We We We
We Kr Lr .. We .. We
We Ay .. .. Lr .. We
We Kg Cg .. We .. We
We We We We We We We
"""

MAX_STEPS = 50
ROLLOUT_STEPS = 40
DEFAULT_SOFTNESS = 1000.0  # High = more deterministic


# ============================================================================
# Environment Setup
# ============================================================================

def create_demo_env() -> MultiGridEnv:
    """Create the demo environment with the specific map."""
    env = MultiGridEnv(
        map=MAP_STR,
        max_steps=MAX_STEPS,
        partial_obs=False,
        objects_set=World,
        actions_set=Actions
    )
    env.reset()
    return env


def get_human_agent_index(env: MultiGridEnv) -> int:
    """Get the index of the human (yellow) agent."""
    for i, agent in enumerate(env.agents):
        if agent.color == 'yellow':
            return i
    raise ValueError("No yellow agent found")


# ============================================================================
# Policy Creation
# ============================================================================

def create_heuristic_policy(
    env: MultiGridEnv,
    human_agent_indices: List[int],
    softness: float = DEFAULT_SOFTNESS
) -> HeuristicPotentialPolicy:
    """Create a heuristic potential-based policy."""
    path_calculator = PathDistanceCalculator(
        grid_height=env.height,
        grid_width=env.width,
        world_model=env
    )
    
    policy = HeuristicPotentialPolicy(
        world_model=env,
        human_agent_indices=human_agent_indices,
        path_calculator=path_calculator,
        softness=softness,
        num_actions=8  # Full Actions: still, left, right, forward, pickup, drop, toggle, done
    )
    
    return policy


# ============================================================================
# Rollout and Verification
# ============================================================================

def run_rollout(
    env: MultiGridEnv,
    policy: HeuristicPotentialPolicy,
    human_agent_index: int,
    goal: ReachRectangleGoal,
    max_steps: int = ROLLOUT_STEPS,
    verbose: bool = True
) -> tuple:
    """
    Run a single rollout and track key events.
    
    Returns:
        Tuple of (frames, events_log, success)
    """
    frames = []
    events_log = []
    
    # Track state
    picked_up_red_key = False
    picked_up_green_key = False  # Should NOT happen
    opened_doors = []
    dropped_key = False
    reached_goal = False
    
    action_names = ['still', 'left', 'right', 'forward', 'pickup', 'drop', 'toggle', 'done']
    
    for step in range(max_steps):
        state = env.get_state()
        step_count, agent_states, mobile_objects, mutable_objects = state
        
        # Get agent state
        agent_state = agent_states[human_agent_index]
        agent_x, agent_y = int(agent_state[0]), int(agent_state[1])
        agent_dir = int(agent_state[2])
        carrying_type = agent_state[6] if len(agent_state) > 6 else None
        carrying_color = agent_state[7] if len(agent_state) > 7 else None
        
        # Render frame with goal overlay
        frame = render_goals_on_frame(env, {human_agent_index: goal})
        frames.append(frame)
        
        # Check for red key pickup
        if carrying_type == 'key' and carrying_color == 'red' and not picked_up_red_key:
            picked_up_red_key = True
            events_log.append(f"Step {step}: Picked up red key")
            if verbose:
                print(f"  Step {step}: ✓ Picked up red key")
        
        # Check for green key pickup (should NOT happen)
        if carrying_type == 'key' and carrying_color == 'green' and not picked_up_green_key:
            picked_up_green_key = True
            events_log.append(f"Step {step}: UNEXPECTED - Picked up green key!")
            if verbose:
                print(f"  Step {step}: ✗ UNEXPECTED - Picked up green key!")
        
        # Check for key drop
        if picked_up_red_key and carrying_type is None and not dropped_key:
            dropped_key = True
            events_log.append(f"Step {step}: Dropped key")
            if verbose:
                print(f"  Step {step}: ✓ Dropped key")
        
        # Check for door unlocks (by checking mutable_objects)
        for obj in mutable_objects:
            if obj[0] == 'door':
                door_x, door_y, is_open, is_locked = obj[1], obj[2], obj[3], obj[4]
                door_key = (door_x, door_y)
                if is_open and door_key not in opened_doors:
                    opened_doors.append(door_key)
                    events_log.append(f"Step {step}: Opened door at ({door_x}, {door_y})")
                    if verbose:
                        print(f"  Step {step}: ✓ Opened door at ({door_x}, {door_y})")
        
        # Check if reached goal
        goal_rect = goal.target_rect
        if goal_rect[0] <= agent_x <= goal_rect[2] and goal_rect[1] <= agent_y <= goal_rect[3]:
            reached_goal = True
            events_log.append(f"Step {step}: Reached goal at ({agent_x}, {agent_y})")
            if verbose:
                print(f"  Step {step}: ✓ Reached goal at ({agent_x}, {agent_y})!")
            # Capture final frame
            frame = render_goals_on_frame(env, {human_agent_index: goal})
            frames.append(frame)
            break
        
        # Get action from policy
        action_probs = policy(state, human_agent_index, goal)
        action = np.argmax(action_probs)
        
        if verbose and step < 30:  # Limit verbose output
            print(f"  Step {step}: Agent at ({agent_x},{agent_y}) dir={agent_dir}, "
                  f"carrying={carrying_type}, action={action_names[action]}")
        
        # Execute action
        actions = [action]
        _, _, done, _ = env.step(actions)
        
        if done:
            break
    
    # Capture final frame if not already done
    if not reached_goal:
        frame = render_goals_on_frame(env, {human_agent_index: goal})
        frames.append(frame)
    
    # Check if green door at (2,3) was opened
    opened_green_door = (2, 3) in opened_doors
    
    # Determine success:
    # - Must pick up red key
    # - Must NOT pick up green key (no locked green doors)
    # - Must open at least 3 doors (2 locked red + 1 closed green)
    # - Must reach goal
    success = (picked_up_red_key and 
               not picked_up_green_key and 
               len(opened_doors) >= 3 and 
               opened_green_door and
               reached_goal)
    
    return frames, events_log, success, {
        'picked_up_red_key': picked_up_red_key,
        'picked_up_green_key': picked_up_green_key,
        'opened_doors': opened_doors,
        'opened_green_door': opened_green_door,
        'dropped_key': dropped_key,
        'reached_goal': reached_goal
    }


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 70)
    print("Heuristic Key-Door Demo")
    print("=" * 70)
    print()
    print("Map:")
    print(MAP_STR)
    print()
    print("Goal: Reach the rightmost cells (column 5, rows 1-3)")
    print("Expected behavior:")
    print("  1. Pick up red key (Kr) - needed for locked doors")
    print("  2. Open first locked red door (Lr at column 2)")
    print("  3. Open second locked red door (Lr at column 4)")
    print("  4. Drop the red key (no longer needed)")
    print("  5. Open closed green door (Cg at (2,3)) - unlocked but closed")
    print("  6. Do NOT pick up green key (Kg) - no locked green doors")
    print("  7. Reach goal area")
    print()
    
    # Create environment
    print("Creating environment...")
    env = create_demo_env()
    
    # Get human agent index
    human_idx = get_human_agent_index(env)
    print(f"Human agent index: {human_idx}")
    
    # Create policy
    print("Creating heuristic policy...")
    policy = create_heuristic_policy(env, [human_idx])
    
    # Define goal: reach rightmost column (x=5, y=1 to y=3)
    goal = ReachRectangleGoal(env, human_idx, (5, 1, 5, 3))
    print(f"Goal rectangle: {goal.target_rect}")
    print()
    
    # Run rollout
    print("Running rollout...")
    print("-" * 50)
    frames, events, success, details = run_rollout(
        env, policy, human_idx, goal, 
        max_steps=ROLLOUT_STEPS, verbose=True
    )
    print("-" * 50)
    print()
    
    # Report results
    print("Results:")
    print(f"  Picked up red key: {'✓' if details['picked_up_red_key'] else '✗'}")
    print(f"  Did NOT pick up green key: {'✓' if not details['picked_up_green_key'] else '✗ UNEXPECTED'}")
    print(f"  Doors opened: {len(details['opened_doors'])} ({details['opened_doors']})")
    print(f"  Opened green door (2,3): {'✓' if details['opened_green_door'] else '✗'}")
    print(f"  Dropped key: {'✓' if details['dropped_key'] else '✗'}")
    print(f"  Reached goal: {'✓' if details['reached_goal'] else '✗'}")
    print()
    
    if success:
        print("✓ SUCCESS: Agent completed all expected behaviors!")
    else:
        print("✗ INCOMPLETE: Agent did not complete all expected behaviors")
        if not details['picked_up_red_key']:
            print("  - Did not pick up red key")
        if details['picked_up_green_key']:
            print("  - UNEXPECTED: Picked up green key (should not - no locked green doors)")
        if len(details['opened_doors']) < 3:
            print(f"  - Only opened {len(details['opened_doors'])} doors (expected 3: 2 red + 1 green)")
        if not details['opened_green_door']:
            print("  - Did not open green door at (2,3)")
        if not details['reached_goal']:
            print("  - Did not reach goal")
    print()
    
    # Save video
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'outputs')
    os.makedirs(output_dir, exist_ok=True)
    movie_path = os.path.join(output_dir, 'heuristic_key_door_demo.mp4')
    
    # Remove existing file
    if os.path.exists(movie_path):
        os.remove(movie_path)
    
    print(f"Saving {len(frames)} frames to video...")
    
    # Use environment's save_video method by temporarily setting frames
    env._video_frames = frames
    env.save_video(movie_path, fps=5)  # Slower fps to see behavior clearly
    
    print()
    print("=" * 70)
    print("Demo completed!")
    print(f"Video output: {os.path.abspath(movie_path)}")
    print("=" * 70)
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

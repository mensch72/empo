#!/usr/bin/env python3
"""
Cooperative Puzzle Demo - All Object Types

This example creates a puzzle that requires two agents to cooperate:
- Agent 0 (RED): Can move rocks, can enter magic walls, can push blocks
- Agent 1 (GREEN): Cannot move rocks, cannot enter magic walls, can push blocks

The puzzle requires:
1. RED agent to push a rock to clear a path
2. RED agent to enter through a magic wall to get a key
3. Both agents to push blocks together
4. GREEN agent to unlock a door and reach the goal
5. Navigate around lava, unsteady ground, and other obstacles

Object types included:
- Wall, Floor, Door (locked), Key, Ball, Box, Goal, ObjectGoal
- Lava, Switch, Block, Rock, UnsteadyGround, MagicWall, Agents
"""

import sys
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from gym_multigrid.multigrid import (
    MultiGridEnv, Grid, Agent, Wall, Floor, Door, Key, Ball, Box,
    Goal, Lava, Switch, Block, Rock, UnsteadyGround, MagicWall, World
)


class CooperativePuzzleEnv(MultiGridEnv):
    """
    A carefully designed puzzle environment that showcases all object types
    and requires two agents to cooperate to solve.
    
    Layout (10x12 grid):
    - RED agent starts top-left, can push rocks and enter magic walls
    - GREEN agent starts bottom-left, cannot push rocks or enter magic walls
    - Goal is in the bottom-right, behind a locked door
    - Key is behind a magic wall that only RED can enter
    - A rock blocks the path to the magic wall (only RED can push)
    - Blocks require cooperative pushing
    - Lava and unsteady ground add hazards
    - Various decorative elements (floor, switch, ball, box, objgoal)
    """
    
    def __init__(self):
        # Agent 0 (RED): Can push rocks, can enter magic walls
        # Agent 1 (GREEN): Cannot push rocks, cannot enter magic walls
        self.agents = [
            Agent(World, 0, can_enter_magic_walls=True, can_push_rocks=True),   # RED
            Agent(World, 1, can_enter_magic_walls=False, can_push_rocks=False),  # GREEN
        ]
        
        super().__init__(
            width=12,
            height=10,
            max_steps=500,  # Long episode for complex puzzle
            agents=self.agents,
            partial_obs=False,
            objects_set=World
        )
    
    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        
        # Add walls around the perimeter
        for i in range(width):
            self.grid.set(i, 0, Wall(World))
            self.grid.set(i, height-1, Wall(World))
        for j in range(height):
            self.grid.set(0, j, Wall(World))
            self.grid.set(width-1, j, Wall(World))
        
        # === AGENT STARTING POSITIONS ===
        # RED agent (can push rocks, enter magic walls)
        self.agents[0].pos = np.array([1, 1])
        self.agents[0].dir = 0  # facing right
        self.grid.set(1, 1, self.agents[0])
        
        # GREEN agent (cannot push rocks or enter magic walls)
        self.agents[1].pos = np.array([1, 8])
        self.agents[1].dir = 0  # facing right
        self.grid.set(1, 8, self.agents[1])
        
        # === GOAL AREA (bottom-right) ===
        # Goal behind a locked door (index 1 = green)
        self.grid.set(10, 8, Goal(World, 1))
        
        # Locked door blocking the goal
        door = Door(World, 'blue', is_open=False, is_locked=True)
        self.grid.set(9, 8, door)
        
        # === KEY AREA (behind magic wall, top area) ===
        # Magic wall - only RED can enter from the left side
        magic_wall = MagicWall(World, magic_side=2, entry_probability=0.8, 
                               solidify_probability=0.1, color='grey')
        self.grid.set(6, 2, magic_wall)
        
        # Blue key behind magic wall
        key = Key(World, 'blue')
        self.grid.set(8, 2, key)
        
        # Floor marking the key area
        self.grid.set(7, 2, Floor(World, 'yellow'))
        
        # === ROCK BLOCKING PATH TO MAGIC WALL ===
        # Only RED agent can push this rock (RED has can_push_rocks=True)
        rock = Rock(World)
        self.grid.set(5, 2, rock)
        
        # === COOPERATIVE BLOCK PUZZLE ===
        # Blocks that need to be pushed to clear the path
        self.grid.set(7, 5, Block(World))
        self.grid.set(8, 5, Block(World))
        
        # === HAZARDS ===
        # Lava pit - dangerous!
        self.grid.set(4, 4, Lava(World))
        self.grid.set(5, 4, Lava(World))
        self.grid.set(4, 5, Lava(World))
        
        # Unsteady ground - agents may stumble
        unsteady1 = UnsteadyGround(World, stumble_probability=0.4, color='brown')
        unsteady2 = UnsteadyGround(World, stumble_probability=0.4, color='brown')
        self.grid.set(3, 6, unsteady1)
        self.terrain_grid.set(3, 6, unsteady1)
        self.grid.set(3, 7, unsteady2)
        self.terrain_grid.set(3, 7, unsteady2)
        
        # === DECORATIVE/INTERACTIVE ELEMENTS ===
        # Switch
        switch = Switch(World)
        self.grid.set(2, 4, switch)
        
        # Ball (collectible)
        ball = Ball(World, 1, 'purple')
        self.grid.set(9, 3, ball)
        
        # Box (can be picked up)
        box = Box(World, 'red')
        self.grid.set(2, 6, box)
        
        # Object goal (delivery target)
        # objgoal = ObjectGoal(World, 0, target_type='ball', reward=1)
        # self.grid.set(10, 3, objgoal)
        
        # === INTERNAL WALLS FOR MAZE STRUCTURE ===
        # Wall creating a corridor
        self.grid.set(3, 3, Wall(World))
        self.grid.set(4, 3, Wall(World))
        
        # Wall separating upper and lower areas
        self.grid.set(6, 4, Wall(World))
        self.grid.set(6, 5, Wall(World))
        self.grid.set(6, 6, Wall(World))
        
        # Wall near goal area
        self.grid.set(8, 7, Wall(World))
        self.grid.set(9, 7, Wall(World))
        
        # === FLOOR MARKERS ===
        # Path markers
        self.grid.set(2, 1, Floor(World, 'blue'))
        self.grid.set(3, 1, Floor(World, 'blue'))
        self.grid.set(4, 1, Floor(World, 'blue'))
        
        # Goal approach area
        self.grid.set(10, 7, Floor(World, 'green'))
        self.grid.set(10, 6, Floor(World, 'green'))


def render_grid_to_array(env):
    """Render the environment grid to a numpy array for animation."""
    img = env.render(mode='rgb_array', highlight=False)
    return img


def solve_puzzle_actions():
    """
    Returns a sequence of actions for both agents to solve the puzzle.
    Actions: 0=left, 1=right, 2=toggle, 3=forward, 4=pickup, 5=drop, 6=done
    
    The solution involves:
    1. RED navigates to push the rock
    2. RED enters magic wall area to get key
    3. RED brings key to GREEN
    4. GREEN takes key and unlocks door
    5. GREEN reaches goal
    """
    actions = []
    
    # Phase 1: RED moves toward the rock (agent 0 moves, agent 1 waits)
    # RED: right, forward, forward, forward, right, forward (to face rock)
    actions.append([3, 6])  # RED forward, GREEN done
    actions.append([3, 6])  # RED forward
    actions.append([3, 6])  # RED forward
    actions.append([1, 6])  # RED turn right
    actions.append([3, 6])  # RED forward
    
    # Phase 2: RED pushes the rock and enters magic wall
    actions.append([3, 6])  # RED pushes rock forward
    actions.append([3, 6])  # RED forward (tries to enter magic wall - may fail)
    actions.append([3, 6])  # Try again if needed
    actions.append([3, 6])  # Keep trying
    actions.append([3, 6])  # Keep trying
    
    # Phase 3: RED picks up key
    actions.append([3, 6])  # RED forward
    actions.append([4, 6])  # RED pickup key
    
    # Phase 4: RED returns with key
    actions.append([1, 6])  # RED turn right
    actions.append([1, 6])  # RED turn right (face left)
    actions.append([3, 6])  # RED forward
    actions.append([3, 6])  # RED forward
    actions.append([3, 6])  # RED forward out of magic wall area
    
    # Phase 5: RED navigates to drop key for GREEN
    actions.append([0, 6])  # RED turn left
    actions.append([3, 6])  # RED forward
    actions.append([3, 6])  # RED forward
    actions.append([3, 6])  # RED forward
    actions.append([3, 6])  # RED forward
    actions.append([1, 6])  # RED turn right
    actions.append([3, 6])  # RED forward
    actions.append([3, 6])  # RED forward
    actions.append([5, 6])  # RED drop key
    
    # Phase 6: GREEN picks up key and moves toward door
    actions.append([6, 3])  # RED done, GREEN forward
    actions.append([6, 3])  # GREEN forward
    actions.append([6, 3])  # GREEN forward
    actions.append([6, 3])  # GREEN forward
    actions.append([6, 3])  # GREEN forward
    actions.append([6, 3])  # GREEN forward
    actions.append([6, 4])  # GREEN pickup key
    actions.append([6, 3])  # GREEN forward
    actions.append([6, 3])  # GREEN forward
    actions.append([6, 0])  # GREEN turn left
    actions.append([6, 3])  # GREEN forward
    
    # Phase 7: GREEN unlocks door and enters goal
    actions.append([6, 2])  # GREEN toggle (unlock door)
    actions.append([6, 3])  # GREEN forward into goal
    
    # Add some random exploration to make it longer and more interesting
    for _ in range(100):
        # Random movements for both agents
        actions.append([np.random.choice([0, 1, 3, 6]), np.random.choice([0, 1, 3, 6])])
    
    return actions


def create_animation(output_path='cooperative_puzzle_animation.mp4', num_steps=200):
    """Create and save an animation showing the cooperative puzzle."""
    
    print("=" * 70)
    print("Cooperative Puzzle Demo - All Object Types")
    print("=" * 70)
    print()
    print("Object Types Included:")
    print("  - Wall, Floor, Door (locked), Key, Ball, Box")
    print("  - Goal, Lava, Switch, Block, Rock")
    print("  - UnsteadyGround, MagicWall, Agents")
    print()
    print("Puzzle Setup:")
    print("  - RED agent (top-left): Can push rocks, enter magic walls")
    print("  - GREEN agent (bottom-left): Cannot push rocks/enter magic walls")
    print("  - Goal is behind a LOCKED DOOR (bottom-right)")
    print("  - KEY is behind a MAGIC WALL (only RED can enter)")
    print("  - A ROCK blocks the path (only RED can push)")
    print("  - LAVA and UNSTEADY GROUND add hazards")
    print()
    print("Solution requires cooperation:")
    print("  1. RED pushes the rock to clear path")
    print("  2. RED enters magic wall to get the key")
    print("  3. RED brings key to GREEN")
    print("  4. GREEN unlocks door and reaches goal")
    print()
    
    # Create environment
    env = CooperativePuzzleEnv()
    env.reset()
    
    # Collect frames
    frames = []
    
    # Initial state
    frames.append(render_grid_to_array(env))
    
    # Get solution actions
    puzzle_actions = solve_puzzle_actions()
    
    print(f"Simulating {num_steps} steps...")
    for step in range(min(num_steps, len(puzzle_actions))):
        actions = puzzle_actions[step]
        obs, rewards, done, info = env.step(actions)
        frames.append(render_grid_to_array(env))
        
        if (step + 1) % 50 == 0:
            print(f"  Step {step + 1}/{num_steps} complete")
        
        if done:
            print(f"  Puzzle solved at step {step + 1}!")
            break
    
    # Continue with random actions if we haven't reached num_steps
    while len(frames) - 1 < num_steps:
        actions = [np.random.choice([0, 1, 3, 6]), np.random.choice([0, 1, 3, 6])]
        obs, rewards, done, info = env.step(actions)
        frames.append(render_grid_to_array(env))
        
        if (len(frames) - 1) % 50 == 0:
            print(f"  Step {len(frames) - 1}/{num_steps} complete")
    
    print(f"\nCollected {len(frames)} frames")
    
    # Create animation
    print(f"Saving animation to {output_path}...")
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Initialize the image
    im = ax.imshow(frames[0])
    
    def update(frame_idx):
        im.set_array(frames[frame_idx])
        ax.set_title(f'Cooperative Puzzle - Step {frame_idx}/{len(frames)-1}\n'
                     'RED: Can push rocks & enter magic walls | '
                     'GREEN: Cannot', 
                     fontsize=10, fontweight='bold')
        return [im]
    
    # Create animation
    anim = animation.FuncAnimation(
        fig, 
        update, 
        frames=len(frames),
        interval=150,  # 150ms between frames
        blit=True,
        repeat=True
    )
    
    # Save as MP4
    try:
        writer = animation.FFMpegWriter(fps=7, bitrate=1800)
        anim.save(output_path, writer=writer)
        print(f"✓ Animation saved successfully to {output_path}")
        print(f"  Total frames: {len(frames)}")
        print(f"  Duration: ~{len(frames)/7:.1f} seconds")
    except Exception as e:
        print(f"✗ Error saving animation: {e}")
        print("  Note: FFmpeg is required to save MP4 files.")
        
        # Try saving as GIF as fallback
        print("\nTrying to save as GIF instead...")
        gif_path = output_path.replace('.mp4', '.gif')
        try:
            anim.save(gif_path, writer='pillow', fps=7)
            print(f"✓ Animation saved as GIF to {gif_path}")
        except Exception as e2:
            print(f"✗ Error saving GIF: {e2}")
    
    plt.close()


# Configuration for quick mode vs full mode
NUM_STEPS_FULL = 200   # Full mode: 200 steps
NUM_STEPS_QUICK = 30   # Quick mode: 30 steps


def main(quick_mode=False):
    """Main function to run the cooperative puzzle demo."""
    num_steps = NUM_STEPS_QUICK if quick_mode else NUM_STEPS_FULL
    mode_str = "QUICK TEST MODE" if quick_mode else "FULL MODE"
    
    print(f"[{mode_str}]")
    print()
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'outputs')
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, 'cooperative_puzzle_animation.mp4')
    
    # Create animation with specified number of steps
    create_animation(output_path, num_steps=num_steps)
    
    print()
    print("=" * 70)
    print("Done! You can view the animation at:")
    print(f"  {os.path.abspath(output_path)}")
    print("=" * 70)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Cooperative Puzzle Demo')
    parser.add_argument('--quick', '-q', action='store_true',
                        help='Run in quick test mode with fewer steps')
    args = parser.parse_args()
    main(quick_mode=args.quick)

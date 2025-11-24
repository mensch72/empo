"""
Example demonstrating magic walls in action.

This creates a simple environment with magic walls that agents can attempt to enter
with certain probabilities from specific directions.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'vendor', 'multigrid'))

import numpy as np
from gym_multigrid.multigrid import MultiGridEnv, Grid, Agent, MagicWall, Wall, World, Floor


class MagicWallDemo(MultiGridEnv):
    """Demo environment showcasing magic walls."""
    
    def __init__(self):
        # Create 2 agents: one can enter magic walls, one cannot
        self.agents = [
            Agent(World, 0, can_enter_magic_walls=True),   # Red agent CAN enter
            Agent(World, 1, can_enter_magic_walls=False),  # Green agent CANNOT enter
        ]
        super().__init__(
            width=12,
            height=8,
            max_steps=100,
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
        
        # Place agents on the left side
        self.agents[0].pos = np.array([2, 2])
        self.agents[0].dir = 0  # facing right
        self.grid.set(2, 2, self.agents[0])
        
        self.agents[1].pos = np.array([2, 5])
        self.agents[1].dir = 0  # facing right
        self.grid.set(2, 5, self.agents[1])
        
        # Create magic walls with different properties
        # Magic wall 1: Can be entered from left (magic_side=2), 100% probability
        mw1 = MagicWall(World, magic_side=2, entry_probability=1.0, color='grey')
        self.grid.set(5, 2, mw1)
        
        # Magic wall 2: Can be entered from left (magic_side=2), 50% probability
        mw2 = MagicWall(World, magic_side=2, entry_probability=0.5, color='grey')
        self.grid.set(5, 5, mw2)
        
        # Add some floor markers to make the demo clearer
        for i in range(3, 5):
            self.grid.set(i, 2, Floor(World, color='blue'))
            self.grid.set(i, 5, Floor(World, color='green'))
        
        # Add goal areas past the magic walls
        for i in range(6, 9):
            self.grid.set(i, 2, Floor(World, color='yellow'))
            self.grid.set(i, 5, Floor(World, color='yellow'))


def main():
    print("Magic Wall Demo")
    print("=" * 50)
    print()
    print("Setup:")
    print("- RED agent (top): CAN enter magic walls")
    print("- GREEN agent (bottom): CANNOT enter magic walls")
    print()
    print("Magic Walls:")
    print("- Top wall (row 2): 100% entry probability")
    print("- Bottom wall (row 5): 50% entry probability")
    print("- Both can be entered from the LEFT side (marked with blue dashed line)")
    print()
    print("Watch as agents try to pass through!")
    print()
    
    env = MagicWallDemo()
    env.reset()
    
    # Try to render
    try:
        import matplotlib.pyplot as plt
        img = env.render(mode='rgb_array')
        plt.figure(figsize=(10, 6))
        plt.imshow(img)
        plt.axis('off')
        plt.title('Magic Wall Demo - Initial State')
        plt.tight_layout()
        plt.savefig('/tmp/magic_wall_demo_initial.png', dpi=150, bbox_inches='tight')
        print("Saved initial state to /tmp/magic_wall_demo_initial.png")
        print()
    except Exception as e:
        print(f"Could not render: {e}")
    
    # Simulate some steps
    print("Simulation:")
    for step_num in range(1, 11):
        # Both agents try to move forward
        actions = [3, 3]  # forward for both
        obs, rewards, done, info = env.step(actions)
        
        print(f"Step {step_num}:")
        print(f"  RED agent position: {tuple(env.agents[0].pos)}")
        print(f"  GREEN agent position: {tuple(env.agents[1].pos)}")
        
        # Check if agents reached magic walls
        if step_num == 3:
            red_pos = tuple(env.agents[0].pos)
            green_pos = tuple(env.agents[1].pos)
            if red_pos[0] == 5:
                print(f"  -> RED agent ENTERED the magic wall at {red_pos}!")
            else:
                print(f"  -> RED agent still approaching...")
            
            if green_pos[0] == 5:
                print(f"  -> GREEN agent tried but CANNOT enter (no permission)")
            else:
                print(f"  -> GREEN agent blocked by magic wall")
        
        if done:
            print("  Episode done!")
            break
    
    print()
    print("Demo complete!")
    print()
    print("Key observations:")
    print("- RED agent can pass through the magic wall (has permission)")
    print("- GREEN agent cannot pass through (no permission)")
    print("- The 50% probability wall may require multiple attempts")


if __name__ == '__main__':
    main()

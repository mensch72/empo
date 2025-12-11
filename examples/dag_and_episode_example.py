#!/usr/bin/env python3
"""
Example demonstrating DAG computation and episode visualization.

This script creates a small 4x4 multigrid environment with:
- 2 agents placed in opposite corners
- One agent that can push rocks, one that cannot
- 2 rocks and 2 blocks in the center
- 10-step timeout

It then:
1. Computes and plots the DAG structure of all reachable states
2. Saves one sample episode as a GIF animation
"""

import sys
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from gym_multigrid.multigrid import MultiGridEnv, Grid, Agent, Block, Rock, Wall, World


class SmallDAGEnv(MultiGridEnv):
    """Small 4x4 environment for DAG visualization with no walls."""
    
    def __init__(self):
        # Create 2 agents - agent 1 can push rocks, agent 0 cannot
        self.agents = [Agent(World, 0, can_push_rocks=False), Agent(World, 1, can_push_rocks=True)]
        
        super().__init__(
            width=4,
            height=4,
            max_steps=10,  # 10-step timeout
            agents=self.agents,
            partial_obs=False,
            objects_set=World
        )
    
    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        
        # No walls - empty grid
        
        # Place agents in opposite corners of the 4x4 grid
        # Agent 0 at (0, 0) - bottom-left corner (cannot push rocks)
        agent0 = self.agents[0]
        agent0.pos = np.array([0, 0])
        agent0.dir = 0  # facing right
        self.grid.set(0, 0, agent0)
        
        # Agent 1 at (3, 3) - top-right corner (can push rocks)
        agent1 = self.agents[1]
        agent1.pos = np.array([3, 3])
        agent1.dir = 2  # facing left
        self.grid.set(3, 3, agent1)
        
        # Place 2 blocks at (1,1) and (1,2)
        block1 = Block(World)
        self.grid.set(1, 1, block1)
        
        block2 = Block(World)
        self.grid.set(1, 2, block2)
        
        # Place 2 rocks at (2,1) and (2,2)
        # Only agent 1 can push rocks (has can_push_rocks=True)
        rock1 = Rock(World)
        self.grid.set(2, 1, rock1)
        
        rock2 = Rock(World)
        self.grid.set(2, 2, rock2)
    
    def _is_valid_pos(self, pos):
        """Check if a position is within grid bounds."""
        x, y = pos
        return 0 <= x < self.width and 0 <= y < self.height
    
    def _compute_successor_state(self, state, actions, ordering):
        """
        Override to handle out-of-bounds positions gracefully.
        Treat grid boundaries as implicit walls.
        """
        # Start from the given state
        self.set_state(state)
        
        # Increment step counter
        self.step_count += 1
        
        # Execute each agent's action in the specified order
        for i in ordering:
            # Skip if agent shouldn't act
            if (self.agents[i].terminated or 
                self.agents[i].paused or 
                not self.agents[i].started or 
                actions[i] == self.actions.still):
                continue
            
            # Get the position in front of the agent
            fwd_pos = self.agents[i].front_pos
            
            # Check bounds before accessing grid
            if not self._is_valid_pos(fwd_pos):
                # Treat out-of-bounds as wall - skip this action
                if actions[i] == self.actions.left:
                    self.agents[i].dir = (self.agents[i].dir - 1) % 4
                elif actions[i] == self.actions.right:
                    self.agents[i].dir = (self.agents[i].dir + 1) % 4
                continue
            
            # Get the contents of the cell in front of the agent
            fwd_cell = self.grid.get(*fwd_pos)
            
            # Execute the action
            if actions[i] == self.actions.left:
                self.agents[i].dir = (self.agents[i].dir - 1) % 4
            
            elif actions[i] == self.actions.right:
                self.agents[i].dir = (self.agents[i].dir + 1) % 4
            
            elif actions[i] == self.actions.forward:
                if fwd_cell is None or fwd_cell.can_overlap():
                    # Move forward
                    self.grid.set(*self.agents[i].pos, None)
                    self.grid.set(*fwd_pos, self.agents[i])
                    self.agents[i].pos = fwd_pos
                elif fwd_cell.type in ['block', 'rock']:
                    # Try to push
                    if fwd_cell.type == 'rock' and not fwd_cell.can_be_pushed_by(self.agents[i]):
                        continue
                    
                    # Calculate push position
                    push_pos = fwd_pos + self.agents[i].dir_vec
                    
                    if not self._is_valid_pos(push_pos):
                        continue  # Can't push out of bounds
                    
                    push_cell = self.grid.get(*push_pos)
                    
                    if push_cell is None or push_cell.can_overlap():
                        # Push object
                        self.grid.set(*self.agents[i].pos, None)
                        self.grid.set(*fwd_pos, self.agents[i])
                        self.grid.set(*push_pos, fwd_cell)
                        self.agents[i].pos = fwd_pos
        
        # Return new state
        return self.get_state()
    
    def _identify_conflict_blocks(self, actions, active_agents):
        """
        Override to handle out-of-bounds positions gracefully.
        Note: This method assumes the environment is already set to the relevant state
        (i.e., the caller has already called set_state() before calling this method).
        """
        # Constants for resource types
        RESOURCE_INDEPENDENT = 'independent'
        RESOURCE_CELL = 'cell'
        RESOURCE_PICKUP = 'pickup'
        RESOURCE_DROP_AGENT = 'drop_agent'
        
        # Track which resource each agent targets
        agent_targets = {}  # agent_idx -> resource identifier
        
        for agent_idx in active_agents:
            action = actions[agent_idx]
            agent = self.agents[agent_idx]
            
            # Determine what resource this agent is targeting
            if action == self.actions.forward:
                # Check what the agent is trying to move into
                fwd_pos = agent.front_pos
                
                # Check bounds
                if not self._is_valid_pos(fwd_pos):
                    # Out of bounds - treat as independent (can't move)
                    agent_targets[agent_idx] = (RESOURCE_INDEPENDENT, agent_idx)
                    continue
                
                fwd_cell = self.grid.get(*fwd_pos)
                
                # If pushing blocks/rocks, check if can push
                if fwd_cell and fwd_cell.type in ['block', 'rock']:
                    # Simplified: just check if can push one step
                    push_pos = fwd_pos + agent.dir_vec
                    if not self._is_valid_pos(push_pos):
                        agent_targets[agent_idx] = (RESOURCE_INDEPENDENT, agent_idx)
                    else:
                        push_cell = self.grid.get(*push_pos)
                        if push_cell is None or push_cell.can_overlap():
                            agent_targets[agent_idx] = (RESOURCE_CELL, tuple(push_pos))
                        else:
                            agent_targets[agent_idx] = (RESOURCE_INDEPENDENT, agent_idx)
                else:
                    # Normal movement
                    agent_targets[agent_idx] = (RESOURCE_CELL, tuple(fwd_pos))
            
            elif action == self.actions.pickup:
                fwd_pos = agent.front_pos
                if not self._is_valid_pos(fwd_pos):
                    agent_targets[agent_idx] = (RESOURCE_INDEPENDENT, agent_idx)
                else:
                    fwd_cell = self.grid.get(*fwd_pos)
                    if fwd_cell and fwd_cell.can_pickup():
                        agent_targets[agent_idx] = (RESOURCE_PICKUP, tuple(fwd_pos))
                    else:
                        agent_targets[agent_idx] = (RESOURCE_INDEPENDENT, agent_idx)
            
            elif action == self.actions.drop:
                fwd_pos = agent.front_pos
                if not self._is_valid_pos(fwd_pos):
                    agent_targets[agent_idx] = (RESOURCE_INDEPENDENT, agent_idx)
                else:
                    fwd_cell = self.grid.get(*fwd_pos)
                    if fwd_cell and fwd_cell.type == 'agent':
                        agent_targets[agent_idx] = (RESOURCE_DROP_AGENT, tuple(fwd_pos))
                    else:
                        agent_targets[agent_idx] = (RESOURCE_INDEPENDENT, agent_idx)
            
            else:
                agent_targets[agent_idx] = (RESOURCE_INDEPENDENT, agent_idx)
        
        # Group agents by their target resource
        resource_to_agents = {}
        for agent_idx, resource in agent_targets.items():
            if resource not in resource_to_agents:
                resource_to_agents[resource] = []
            resource_to_agents[resource].append(agent_idx)
        
        # Extract conflict blocks (resources targeted by > 1 agent)
        conflict_blocks = []
        for resource, agents in resource_to_agents.items():
            resource_type, _ = resource
            if resource_type == RESOURCE_INDEPENDENT:
                # Independent agents each form their own "block" of size 1
                for agent_idx in agents:
                    conflict_blocks.append([agent_idx])
            elif len(agents) > 1:
                # Multiple agents target same resource - conflict!
                conflict_blocks.append(agents)
            else:
                # Single agent targeting a resource - no conflict
                conflict_blocks.append(agents)
        
        return conflict_blocks
    
    def step(self, actions):
        """Override step to handle out-of-bounds positions gracefully."""
        self.step_count += 1

        order = np.random.permutation(len(actions))
        rewards = np.zeros(len(actions))
        done = False

        for i in order:
            agent = self.agents[i]
            
            if agent.terminated or agent.paused or not agent.started or actions[i] == self.actions.still:
                continue

            # Get the position in front of the agent
            fwd_pos = agent.front_pos
            
            # Check if forward position is valid (treat out-of-bounds as wall)
            if not self._is_valid_pos(fwd_pos):
                # Can't move out of bounds - treat as wall
                continue
            
            # Get the forward cell
            fwd_cell = self.grid.get(*fwd_pos)

            # Handle different actions
            if actions[i] == self.actions.forward:
                if fwd_cell is None or fwd_cell.can_overlap():
                    self.grid.set(*agent.pos, None)
                    self.grid.set(*fwd_pos, agent)
                    agent.pos = fwd_pos
                elif fwd_cell.type in ['block', 'rock']:
                    # Try to push the object
                    if fwd_cell.type == 'rock' and not fwd_cell.can_be_pushed_by(agent):
                        continue  # Can't push this rock
                    
                    # Calculate where object would be pushed to
                    push_pos = fwd_pos + agent.dir_vec
                    
                    if not self._is_valid_pos(push_pos):
                        continue  # Can't push out of bounds
                    
                    push_cell = self.grid.get(*push_pos)
                    
                    if push_cell is None or push_cell.can_overlap():
                        # Can push - move object and agent
                        self.grid.set(*agent.pos, None)
                        self.grid.set(*fwd_pos, agent)
                        self.grid.set(*push_pos, fwd_cell)
                        agent.pos = fwd_pos
                        
            elif actions[i] == self.actions.left:
                agent.dir = (agent.dir - 1) % 4
            elif actions[i] == self.actions.right:
                agent.dir = (agent.dir + 1) % 4

        # Check termination
        if self.step_count >= self.max_steps:
            done = True

        obs = self.gen_obs()
        return obs, rewards, done, {}



def render_grid_to_array(env):
    """Render the environment grid to a numpy array for animation."""
    img = env.render(mode='rgb_array', highlight=False)
    return img


def create_sample_episode_gif(env, output_path='sample_episode.gif'):
    """Create and save a GIF of one sample episode."""
    print(f"\nCreating sample episode GIF...")
    
    # Reset environment
    env.reset()
    
    # Collect frames
    frames = []
    frames.append(render_grid_to_array(env))
    
    # Run a sample episode with random actions
    done = False
    step_count = 0
    max_steps = 10
    
    print(f"Running sample episode (max {max_steps} steps)...")
    while not done and step_count < max_steps:
        # Random actions for both agents
        actions = [env.action_space.sample() for _ in env.agents]
        obs, rewards, done, info = env.step(actions)
        frames.append(render_grid_to_array(env))
        step_count += 1
        print(f"  Step {step_count}: actions={actions}, done={done}")
    
    # Create animation
    print(f"Saving GIF to {output_path}...")
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Initialize the image
    im = ax.imshow(frames[0])
    
    def update(frame_idx):
        im.set_array(frames[frame_idx])
        ax.set_title(f'Sample Episode - Step {frame_idx}/{len(frames)-1}', 
                     fontsize=12, fontweight='bold')
        return [im]
    
    # Create animation
    anim = animation.FuncAnimation(
        fig, 
        update, 
        frames=len(frames),
        interval=500,  # 500ms between frames
        blit=True,
        repeat=True
    )
    
    # Save as GIF
    try:
        anim.save(output_path, writer='pillow', fps=2)
        print(f"✓ Sample episode GIF saved to {output_path}")
        print(f"  Total frames: {len(frames)}")
    except Exception as e:
        print(f"✗ Error saving GIF: {e}")
    
    plt.close()


def compute_and_plot_dag(env, output_path='dag_plot.pdf'):
    """Compute and plot the DAG structure using WorldModel methods."""
    print("\nComputing DAG structure...")
    print("  This may take a moment for complex environments...")
    
    try:
        # env already inherits from WorldModel, so we can call get_dag directly
        states, state_to_idx, successors = env.get_dag()
        
        print(f"✓ DAG computed successfully!")
        print(f"  Total reachable states: {len(states)}")
        print(f"  Root state index: 0")
        
        # Count terminal states
        terminal_count = sum(1 for succ_list in successors if len(succ_list) == 0)
        print(f"  Terminal states: {terminal_count}")
        
        # Count edges
        total_edges = sum(len(succ_list) for succ_list in successors)
        print(f"  Total transitions: {total_edges}")
        
        # Plot DAG (skip if too large for visualization)
        MAX_STATES_FOR_PLOT = 500  # Graphviz becomes very slow beyond this
        
        if len(states) > MAX_STATES_FOR_PLOT:
            print(f"\n⚠ Skipping DAG plot: {len(states)} states exceeds visualization limit ({MAX_STATES_FOR_PLOT})")
            print(f"  The state space is too large to visualize effectively.")
            print(f"  Consider adding walls or reducing the grid size for visualization.")
        else:
            print(f"\nPlotting DAG to {output_path}...")
            
            # Create simple labels showing state index
            state_labels = {state: f"S{idx}" for idx, state in enumerate(states)}
            
            # Plot with PDF format using env's plot_dag method
            env.plot_dag(
                states=states,
                state_to_idx=state_to_idx,
                successors=successors,
                output_file=output_path.replace('.pdf', ''),
                format='pdf',
                state_labels=state_labels,
                rankdir='TB'  # Top to bottom layout
            )
            
            print(f"✓ DAG plot saved to {output_path}")
        
        return states, state_to_idx, successors
        
    except Exception as e:
        print(f"✗ Error computing/plotting DAG: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


def main():
    """Main function to run the DAG and episode example."""
    print("=" * 70)
    print("DAG Computation and Episode Visualization Example")
    print("=" * 70)
    print()
    print("Environment Setup:")
    print("  - Grid size: 4x4 (no walls)")
    print("  - Agents: 2 (in opposite corners)")
    print("  - Agent 0: Position (0,0) - bottom-left (cannot push rocks)")
    print("  - Agent 1: Position (3,3) - top-right (can push rocks)")
    print("  - Blocks: At positions (1,1) and (1,2)")
    print("  - Rocks: At positions (2,1) and (2,2)")
    print("  - Timeout: 10 steps")
    print()
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'outputs')
    os.makedirs(output_dir, exist_ok=True)
    
    # Create environment
    print("Creating environment...")
    env = SmallDAGEnv()
    env.reset()
    print("✓ Environment created")
    
    # Compute and plot DAG
    dag_output = os.path.join(output_dir, 'dag_plot.pdf')
    states, state_to_idx, successors = compute_and_plot_dag(env, dag_output)
    
    # Create sample episode GIF
    gif_output = os.path.join(output_dir, 'sample_episode.gif')
    create_sample_episode_gif(env, gif_output)
    
    # Summary
    print()
    print("=" * 70)
    print("Done! Generated files:")
    print(f"  DAG plot (PDF): {os.path.abspath(dag_output)}")
    print(f"  Sample episode (GIF): {os.path.abspath(gif_output)}")
    print("=" * 70)


if __name__ == "__main__":
    main()

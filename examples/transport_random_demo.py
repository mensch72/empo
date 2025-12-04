#!/usr/bin/env python3
"""
Transport Environment Random Play Demo.

This script demonstrates the ai_transport PettingZoo environment with:
1. A randomly generated 2D road network (10-15 nodes)
2. Random human (passenger) policies 
3. Random vehicle policies
4. No neural network learning - just basic random agent behavior

The script showcases two modes:
- Using the raw ai_transport parallel_env with policy classes
- Using the TransportEnvWrapper with action masking (gym-like interface)

Usage:
    python transport_random_demo.py           # Full run (100 steps)
    python transport_random_demo.py --quick   # Quick test run (20 steps)
    python transport_random_demo.py --render  # Enable graphical rendering
    python transport_random_demo.py --wrapper # Use TransportEnvWrapper with action masking

Requirements:
    - ai_transport (vendored in vendor/ai_transport)
    - networkx
    - numpy
    - matplotlib (optional, for rendering)
"""

import sys
import os
import argparse
import random

# Add paths for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'vendor', 'ai_transport'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from ai_transport import parallel_env
from ai_transport.policies import (
    RandomHumanPolicy,
    TargetDestinationHumanPolicy,
    RandomVehiclePolicy,
    ShortestPathVehiclePolicy
)

# Import the EMPO wrapper for gym-style interface
from empo.transport import (
    TransportEnvWrapper,
    TransportActions,
    StepType,
    create_transport_env,
)


# ============================================================================
# Configuration
# ============================================================================

NUM_NODES = 12              # Number of nodes in the road network
NUM_HUMANS = 4              # Number of human (passenger) agents
NUM_VEHICLES = 2            # Number of vehicle agents
BIDIRECTIONAL_PROB = 0.5    # Probability of bidirectional edges
HUMAN_SPEED = 1.5           # Walking speed of humans
VEHICLE_SPEED = 3.0         # Speed of vehicles
VEHICLE_CAPACITY = 4        # Capacity of each vehicle

# Full run configuration (default)
NUM_STEPS_FULL = 100

# Quick test configuration (for --quick flag)
NUM_STEPS_QUICK = 20


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Transport Environment Random Play Demo"
    )
    parser.add_argument(
        '--quick', '-q',
        action='store_true',
        help='Run in quick test mode with reduced simulation steps'
    )
    parser.add_argument(
        '--steps', '-s',
        type=int,
        default=None,
        help='Number of simulation steps (overrides default)'
    )
    parser.add_argument(
        '--render', '-r',
        action='store_true',
        help='Enable graphical rendering (requires matplotlib)'
    )
    parser.add_argument(
        '--save-video', '-v',
        type=str,
        default=None,
        help='Save simulation to video file (MP4 or GIF)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--humans', '-H',
        type=int,
        default=NUM_HUMANS,
        help=f'Number of human agents (default: {NUM_HUMANS})'
    )
    parser.add_argument(
        '--vehicles', '-V',
        type=int,
        default=NUM_VEHICLES,
        help=f'Number of vehicle agents (default: {NUM_VEHICLES})'
    )
    parser.add_argument(
        '--nodes', '-n',
        type=int,
        default=NUM_NODES,
        help=f'Number of network nodes (default: {NUM_NODES})'
    )
    parser.add_argument(
        '--wrapper', '-w',
        action='store_true',
        help='Use TransportEnvWrapper with action masking (gym-like interface)'
    )
    return parser.parse_args()


def save_video_matplotlib(frames, output_path: str, fps: int = 5):
    """
    Save frames as video using matplotlib animation.
    
    This approach works without imageio[ffmpeg] by using matplotlib's
    animation module, which can save to MP4 via FFMpegWriter or fall back to GIF.
    
    Args:
        frames: List of numpy arrays (RGB images)
        output_path: Path to save the video
        fps: Frames per second
    """
    if not frames:
        print("No frames to save!")
        return
    
    print(f"Creating video with {len(frames)} frames...")
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.axis('off')
    
    im = ax.imshow(frames[0])
    title = ax.set_title('Transport Simulation', fontsize=12)
    
    def update(frame_idx):
        im.set_array(frames[frame_idx])
        title.set_text(f'Transport Simulation - Frame {frame_idx + 1}/{len(frames)}')
        return [im, title]
    
    anim = animation.FuncAnimation(
        fig, update, frames=len(frames),
        interval=1000 // fps, blit=True, repeat=True
    )
    
    # Try to save as MP4 first, fall back to GIF
    try:
        writer = animation.FFMpegWriter(fps=fps, bitrate=2000)
        anim.save(output_path, writer=writer)
        print(f"✓ Video saved to {output_path}")
    except Exception as e:
        print(f"Could not save MP4 ({e}), trying GIF...")
        gif_path = output_path.replace('.mp4', '.gif')
        try:
            anim.save(gif_path, writer='pillow', fps=fps)
            print(f"✓ Video saved as GIF to {gif_path}")
        except Exception as e2:
            print(f"Error saving video: {e2}")
    
    plt.close(fig)


def create_random_network(env, num_nodes: int, seed: int = 42):
    """Create a random 2D road network using the environment's built-in method."""
    return env.create_random_2d_network(
        num_nodes=num_nodes,
        bidirectional_prob=BIDIRECTIONAL_PROB,
        speed_mean=VEHICLE_SPEED,
        capacity_mean=10.0,
        coord_std=10.0,
        seed=seed
    )


def create_policies(env, network, seed: int = 42):
    """
    Create random and semi-intelligent policies for all agents.
    
    Humans:
        - Half use RandomHumanPolicy (random actions)
        - Half use TargetDestinationHumanPolicy (goal-directed, but not learned)
    
    Vehicles:
        - Half use RandomVehiclePolicy (random actions)
        - Half use ShortestPathVehiclePolicy (pathfinding, but not learned)
    """
    policies = {}
    
    # Create human policies
    human_agents = [a for a in env.possible_agents if a.startswith('human_')]
    for i, agent_id in enumerate(human_agents):
        if i % 2 == 0:
            # Random policy
            policies[agent_id] = RandomHumanPolicy(
                agent_id, 
                pass_prob_boarding=0.3,
                seed=seed + i
            )
        else:
            # Target-directed policy (no learning, just heuristic)
            policies[agent_id] = TargetDestinationHumanPolicy(
                agent_id,
                network,
                target_change_rate=0.1,
                seed=seed + i
            )
    
    # Create vehicle policies
    vehicle_agents = [a for a in env.possible_agents if a.startswith('vehicle_')]
    for i, agent_id in enumerate(vehicle_agents):
        if i % 2 == 0:
            # Random policy
            policies[agent_id] = RandomVehiclePolicy(
                agent_id,
                pass_prob_routing=0.2,
                seed=seed + 100 + i
            )
        else:
            # Shortest path policy (no learning, just pathfinding)
            policies[agent_id] = ShortestPathVehiclePolicy(
                agent_id,
                network,
                seed=seed + 100 + i
            )
    
    return policies


def print_state_summary(env, obs, step: int):
    """Print a summary of the current environment state."""
    print(f"\n--- Step {step} ---")
    print(f"Step type: {env.step_type}")
    print(f"Real time: {env.real_time:.2f}s")
    
    # Count agents at nodes vs on edges
    agents_at_nodes = 0
    agents_on_edges = 0
    humans_aboard = 0
    
    for agent_id in env.agents:
        pos = env.agent_positions[agent_id]
        if isinstance(pos, tuple):
            agents_on_edges += 1
        else:
            agents_at_nodes += 1
        
        if agent_id.startswith('human_'):
            aboard = env.human_aboard.get(agent_id)
            if aboard is not None:
                humans_aboard += 1
    
    print(f"Agents at nodes: {agents_at_nodes}, on edges: {agents_on_edges}")
    print(f"Humans aboard vehicles: {humans_aboard}")


def run_simulation(args):
    """Run the transport environment simulation."""
    # Determine number of steps
    if args.steps is not None:
        num_steps = args.steps
    elif args.quick:
        num_steps = NUM_STEPS_QUICK
    else:
        num_steps = NUM_STEPS_FULL
    
    mode_str = "QUICK TEST MODE" if args.quick else "FULL MODE"
    
    print("=" * 70)
    print("AI Transport Environment - Random Play Demo")
    print(f"  [{mode_str}]")
    print("=" * 70)
    print()
    
    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Create initial environment to generate network
    print("1. Creating random 2D road network...")
    temp_env = parallel_env(num_humans=args.humans, num_vehicles=args.vehicles)
    network = create_random_network(temp_env, args.nodes, args.seed)
    print(f"   Network: {network.number_of_nodes()} nodes, {network.number_of_edges()} edges")
    
    # Create environment with the network
    render_mode = 'human' if args.render else None
    env = parallel_env(
        num_humans=args.humans,
        num_vehicles=args.vehicles,
        network=network,
        human_speeds=[HUMAN_SPEED] * args.humans,
        vehicle_speeds=[VEHICLE_SPEED] * args.vehicles,
        vehicle_capacities=[VEHICLE_CAPACITY] * args.vehicles,
        vehicle_fuel_uses=[1.0] * args.vehicles,
        observation_scenario='full',  # Use full observations for demo
        render_mode=render_mode
    )
    
    print(f"   Humans: {args.humans}, Vehicles: {args.vehicles}")
    print(f"   Agents: {env.possible_agents}")
    print()
    
    # Create policies
    print("2. Creating agent policies...")
    policies = create_policies(env, network, args.seed)
    for agent_id, policy in policies.items():
        policy_type = type(policy).__name__
        print(f"   {agent_id}: {policy_type}")
    print()
    
    # Reset environment
    print("3. Initializing environment...")
    obs, info = env.reset(seed=args.seed)
    
    # Initialize frame collection for video
    frames = []
    
    # Enable graphical rendering if needed for video or display
    if args.save_video or args.render:
        print("   Enabling graphical rendering...")
        env.enable_rendering('graphical')
        if args.save_video:
            print(f"   Video will be saved to: {args.save_video}")
    
    # Capture initial frame
    if args.save_video or args.render:
        fig = env.render()
        if args.save_video and fig is not None:
            # Convert matplotlib figure to numpy array
            fig.canvas.draw()
            buf = np.asarray(fig.canvas.buffer_rgba())
            frame = buf[:, :, :3]  # Convert RGBA to RGB
            frames.append(frame.copy())
    
    # Run simulation
    print(f"\n4. Running simulation for {num_steps} steps...")
    
    step_type_counts = {}
    boarding_events = 0
    unboarding_events = 0
    
    for step in range(num_steps):
        # Get actions from policies
        actions = {}
        for agent in env.agents:
            policy = policies.get(agent)
            if policy:
                action_space_size = env.action_space(agent).n
                action, justification = policy.get_action(obs[agent], action_space_size)
                actions[agent] = action
            else:
                actions[agent] = 0  # Pass action
        
        # Count step types
        step_type = env.step_type
        step_type_counts[step_type] = step_type_counts.get(step_type, 0) + 1
        
        # Track boarding/unboarding events
        if step_type == 'boarding':
            for agent, action in actions.items():
                if agent.startswith('human_') and action > 0:
                    boarding_events += 1
        elif step_type == 'unboarding':
            for agent, action in actions.items():
                if agent.startswith('human_') and action > 0:
                    unboarding_events += 1
        
        # Print summary every N steps
        print_interval = max(1, num_steps // 10)
        if step % print_interval == 0:
            print_state_summary(env, obs, step)
        
        # Take step
        obs, rewards, terminations, truncations, infos = env.step(actions)
        
        # Render and capture frame if needed
        if args.render or args.save_video:
            fig = env.render()
            if args.save_video and fig is not None:
                # Convert matplotlib figure to numpy array
                fig.canvas.draw()
                buf = np.asarray(fig.canvas.buffer_rgba())
                frame = buf[:, :, :3]  # Convert RGBA to RGB
                frames.append(frame.copy())
        
        # Check for termination
        if any(terminations.values()) or any(truncations.values()):
            print(f"\nSimulation terminated at step {step}")
            break
    
    # Final summary
    print("\n" + "=" * 70)
    print("5. Simulation Summary")
    print("=" * 70)
    print(f"\nTotal steps: {step + 1}")
    print(f"Final real time: {env.real_time:.2f}s")
    print(f"\nStep type distribution:")
    for step_type, count in sorted(step_type_counts.items()):
        print(f"   {step_type}: {count}")
    print(f"\nBoarding events: {boarding_events}")
    print(f"Unboarding events: {unboarding_events}")
    
    # Final agent positions
    print(f"\nFinal agent positions:")
    for agent_id in env.possible_agents:
        if agent_id in env.agents:
            pos = env.agent_positions[agent_id]
            pos_str = f"edge {pos}" if isinstance(pos, tuple) else f"node {pos}"
            
            extra_info = ""
            if agent_id.startswith('human_'):
                aboard = env.human_aboard.get(agent_id)
                if aboard:
                    extra_info = f" (aboard {aboard})"
            elif agent_id.startswith('vehicle_'):
                dest = env.vehicle_destinations.get(agent_id)
                if dest is not None:
                    extra_info = f" (destination: node {dest})"
            
            print(f"   {agent_id}: {pos_str}{extra_info}")
    
    # Save video if recording
    if args.save_video:
        if frames:
            save_video_matplotlib(frames, args.save_video, fps=5)
        else:
            print("\nWarning: No frames were captured for video")
    
    # Clean up
    env.close()
    
    print("\n" + "=" * 70)
    print("Demo completed successfully!")
    print("=" * 70)


def main():
    """Main entry point."""
    args = parse_args()
    
    try:
        run_simulation(args)
    except ImportError as e:
        print(f"Error: {e}")
        print("\nMake sure ai_transport is properly vendored.")
        print("See VENDOR.md for setup instructions.")
        sys.exit(1)
    except Exception as e:
        print(f"Error during simulation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

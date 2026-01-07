"""
Example demonstrating visualization and video recording.

This example shows:
1. Graphical rendering of the transport network
2. Recording frames during simulation
3. Saving video as MP4
"""

import os
import numpy as np
from ai_transport import parallel_env


def main():
    print("=" * 70)
    print("AI Transport Environment - Visualization Demo")
    print("=" * 70)
    
    # Create a random 2D network
    print("\n1. Creating random 2D network...")
    env = parallel_env(num_humans=4, num_vehicles=2)
    network = env.create_random_2d_network(
        num_nodes=10,
        bidirectional_prob=0.5,
        speed_mean=3.0,  # Slower speeds for more visible motion
        capacity_mean=10.0,
        coord_std=15.0,  # Larger spread for longer edges
        seed=42
    )
    
    # Create environment with the network
    env = parallel_env(
        num_humans=4,
        num_vehicles=2,
        network=network,
        render_mode="human"
    )
    
    env.reset(seed=42)
    env.initialize_random_positions(seed=42)
    
    print(f"   Network: {len(network.nodes())} nodes, {len(network.edges())} edges")
    print(f"   Agents: {len(env.human_agents)} humans, {len(env.vehicle_agents)} vehicles")
    
    # Enable graphical rendering
    print("\n2. Enabling graphical rendering...")
    env.enable_rendering('graphical')
    
    # Render initial state
    print("\n3. Rendering initial state...")
    env.render()
    env.save_frame('transport_initial.png')
    print("   Saved initial frame to transport_initial.png")
    
    # Start video recording
    print("\n4. Starting video recording...")
    env.start_video_recording()
    
    # Run simulation - environment automatically cycles through step types
    print("\n5. Running simulation...")
    num_cycles = 40  # Number of complete step type cycles
    
    # Action probabilities for visible motion
    VEHICLE_ROUTE_PROB = 0.8  # Probability vehicle sets a destination  
    UNBOARD_PROB = 0.2  # Probability human unboards
    BOARD_PROB = 0.5  # Probability human boards
    VEHICLE_DEPART_PROB = 0.6  # Lower probability for more staggered departures
    HUMAN_WALK_PROB = 0.5  # Lower probability for more staggered departures
    
    # Create directory for debug frames
    os.makedirs('debug_frames', exist_ok=True)
    frame_counter = 0
    
    for cycle in range(num_cycles):
        # Take actions for current step type (environment cycles automatically)
        current_step = env.step_type
        actions = {}
        
        if current_step == 'routing':
            # Vehicles set destinations
            for agent in env.agents:
                if agent in env.vehicle_agents:
                    pos = env.agent_positions[agent]
                    if not isinstance(pos, tuple) and np.random.random() < VEHICLE_ROUTE_PROB:
                        nodes = list(env.network.nodes())
                        dest_idx = np.random.randint(1, len(nodes) + 1)  # 1..N (not 0/None)
                        actions[agent] = dest_idx
                    else:
                        actions[agent] = 0
                else:
                    actions[agent] = 0
        
        elif current_step == 'unboarding':
            # Humans unboard
            for agent in env.agents:
                if agent in env.human_agents:
                    aboard = env.human_aboard.get(agent)
                    if aboard is not None:
                        vehicle_pos = env.agent_positions[aboard]
                        if not isinstance(vehicle_pos, tuple) and np.random.random() < UNBOARD_PROB:
                            actions[agent] = 1
                        else:
                            actions[agent] = 0
                    else:
                        actions[agent] = 0
                else:
                    actions[agent] = 0
        
        elif current_step == 'boarding':
            # Humans board
            for agent in env.agents:
                if agent in env.human_agents:
                    pos = env.agent_positions[agent]
                    aboard = env.human_aboard.get(agent)
                    if not isinstance(pos, tuple) and aboard is None:
                        # Find vehicles at same node
                        vehicles_here = [v for v in env.vehicle_agents
                                       if not isinstance(env.agent_positions[v], tuple) 
                                       and env.agent_positions[v] == pos]
                        if vehicles_here and np.random.random() < BOARD_PROB:
                            actions[agent] = 1
                        else:
                            actions[agent] = 0
                    else:
                        actions[agent] = 0
                else:
                    actions[agent] = 0
        
        elif current_step == 'departing':
            # Agents depart on edges
            for agent in env.agents:
                pos = env.agent_positions[agent]
                if not isinstance(pos, tuple):  # At node
                    outgoing = list(env.network.out_edges(pos))
                    if outgoing:
                        if agent in env.vehicle_agents:
                            actions[agent] = 1 if np.random.random() < VEHICLE_DEPART_PROB else 0
                        elif agent in env.human_agents and env.human_aboard.get(agent) is None:
                            actions[agent] = 1 if np.random.random() < HUMAN_WALK_PROB else 0
                        else:
                            actions[agent] = 0
                    else:
                        actions[agent] = 0
                else:
                    actions[agent] = 0
        
        # Take step (environment auto-cycles to next step type)
        obs, rewards, terms, truncs, infos = env.step(actions)
        
        # Only render and record after departing step (when visual state changes)
        # step() doesn't auto-render when recording, so we control it here
        if env.step_type == 'routing':  # Just finished departing step
            env.render()  # This captures the frame to video
            # Also save individual debug frame
            debug_filename = os.path.join('debug_frames', f'frame_{frame_counter:04d}_cycle{cycle}_time{env.real_time:.2f}.png')
            env.save_frame(debug_filename)
            frame_counter += 1
        
        if (cycle + 1) % 10 == 0:
            print(f"   Cycle {cycle + 1}/{num_cycles} completed (time: {env.real_time:.2f}, step: {env.step_type})")
    
    # Save video
    print("\n6. Saving video...")
    env.save_video('transport_simulation.mp4', fps=5)
    
    # Save final frame
    print("\n7. Saving final frame...")
    env.save_frame('transport_final.png')
    print("   Saved final frame to transport_final.png")
    
    # Close environment
    env.close()
    
    print("\n" + "=" * 70)
    print("Demo complete!")
    print("=" * 70)
    print("\nGenerated files:")
    print("  - transport_initial.png (initial state)")
    print("  - transport_final.png (final state)")
    print("  - transport_simulation.mp4 (full simulation video)")
    print(f"  - debug_frames/ directory with {frame_counter} individual PNG frames")
    print(f"\nVideo contains {frame_counter} frames showing progression through departing steps.")
    print("Each frame shows the state after a departing step (when agents move).")
    print("\nNote: To enable video recording, install imageio with:")
    print("  pip install imageio[ffmpeg]")


if __name__ == "__main__":
    main()

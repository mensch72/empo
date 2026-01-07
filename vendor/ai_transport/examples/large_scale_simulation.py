"""
Large-Scale Transport Simulation
=================================

This script demonstrates a large-scale simulation of the AI transport system with:
- 20 vehicles and 120 humans
- 60-node network
- Comprehensive statistics tracking
- Video output
- Multiple metrics visualization

Statistics Tracked:
- Number of humans aboard vehicles / walking / waiting
- Average distance between humans and their targets
- Number of successful target destinations reached
- Average time to reach targets
- Vehicle occupancy rates
- Network utilization

Run this script to generate:
- large_scale_statistics.png: Multi-panel plot of metrics over time
- large_scale_simulation.mp4: Video of simulation
- Console output with detailed statistics
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

from ai_transport import parallel_env
from ai_transport.policies import TargetDestinationHumanPolicy, ShortestPathVehiclePolicy


def main():
    print("=" * 70)
    print("AI Transport Environment - Large-Scale Simulation")
    print("=" * 70)
    print()
    
    # Configuration
    NUM_NODES = 60
    NUM_VEHICLES = 20
    NUM_HUMANS = 120
    NUM_STEPS = 5000
    VIDEO_FRAMES = 50  # Record every 10 steps
    SEED = 42
    
    print("Configuration:")
    print(f"  Network: {NUM_NODES} nodes")
    print(f"  Vehicles: {NUM_VEHICLES}")
    print(f"  Humans: {NUM_HUMANS}")
    print(f"  Simulation steps: {NUM_STEPS}")
    print()
    
    # Create environment
    print("1. Creating random network...")
    env_temp = parallel_env(num_humans=NUM_HUMANS, num_vehicles=NUM_VEHICLES)
    network = env_temp.create_random_2d_network(
        num_nodes=NUM_NODES,
        coord_std=30.0,  # Use coord_std instead of spread for network size
        bidirectional_prob=0.5,
        seed=SEED
    )
    print(f"   Created network with {network.number_of_nodes()} nodes, {network.number_of_edges()} edges")
    
    # Create environment with varied agent attributes
    np.random.seed(SEED)
    vehicle_capacities = np.random.randint(6, 9, NUM_VEHICLES).tolist()
    vehicle_speeds = np.random.uniform(2.5, 3.5, NUM_VEHICLES).tolist()
    human_speeds = np.random.uniform(1.0, 2.0, NUM_HUMANS).tolist()
    
    env = parallel_env(
        num_humans=NUM_HUMANS,
        num_vehicles=NUM_VEHICLES,
        network=network,
        vehicle_capacities=vehicle_capacities,
        vehicle_speeds=vehicle_speeds,
        human_speeds=human_speeds,
        observation_scenario='full',
        render_mode='human'
    )
    
    # Reset environment
    print("\n2. Initializing environment...")
    obs, info = env.reset(seed=SEED)
    env.enable_rendering('graphical')
    print("   Environment initialized")
    
    # Create policies
    print("\n3. Creating policies...")
    policies = {}
    
    # Human policies with varied target change rates
    for i in range(NUM_HUMANS):
        human_id = f'human_{i}'
        target_change_rate = 1e-10 #np.random.uniform(0.05, 0.15)  # Change target every 7-20 seconds
        policies[human_id] = TargetDestinationHumanPolicy(
            human_id, network, target_change_rate=target_change_rate, seed=SEED + i
        )
    
    # Vehicle policies
    for i in range(NUM_VEHICLES):
        vehicle_id = f'vehicle_{i}'
        policies[vehicle_id] = ShortestPathVehiclePolicy(vehicle_id, network, seed=SEED + 1000 + i)
    
    print(f"   Created {len(policies)} policies")
    
    # Initialize statistics tracking
    stats = {
        'time': [],
        'humans_aboard': [],
        'humans_walking': [],
        'humans_waiting': [],
        'avg_distance_to_target': [],
        'targets_reached': [],
        'avg_time_to_target': [],
        'vehicle_occupancy': [],
        'agents_on_edges': [],
        'agents_at_nodes': []
    }
    
    # Track individual human target reaching
    human_target_history = defaultdict(list)  # {human_id: [(set_time, reach_time, success)]}
    human_current_target_set_time = {}  # {human_id: time when current target was set}
    
    # Start video recording
    print("\n4. Starting simulation...")
    env.start_video_recording()
    frame_count = 0
    
    # Run simulation
    for step in range(NUM_STEPS):
        # Get actions from policies
        actions = {}
        for agent in env.agents:
            action_space_size = env.action_space(agent).n
            action, justification = policies[agent].get_action(obs[agent], action_space_size)
            actions[agent] = action
        
        # Step environment
        obs, rewards, terms, truncs, infos = env.step(actions)
        
        # Render frame for video only after departing steps (when visual state changes)
        # This prevents duplicate frames from routing/unboarding/boarding steps
        if env.step_type == 'departing' and frame_count < VIDEO_FRAMES:
            env.render()
            frame_count += 1
        
        # Collect statistics
        real_time = env.real_time
        stats['time'].append(real_time)
        
        # Count human status
        humans_aboard = 0
        humans_walking = 0
        humans_waiting = 0
        total_distance = 0
        distance_count = 0
        
        for i in range(NUM_HUMANS):
            human_id = f'human_{i}'
            position = env.agent_positions[human_id]
            aboard = env.human_aboard[human_id]
            
            if aboard is not None:
                humans_aboard += 1
            elif isinstance(position, tuple):
                humans_walking += 1
            else:
                humans_waiting += 1
            
            # Track distance to target
            policy = policies[human_id]
            if hasattr(policy, 'target') and policy.target is not None:
                # Get human position coordinates
                if isinstance(position, tuple):
                    edge, coord = position
                    u, v = edge
                    u_pos = np.array([network.nodes[u]['x'], network.nodes[u]['y']])
                    v_pos = np.array([network.nodes[v]['x'], network.nodes[v]['y']])
                    edge_vec = v_pos - u_pos
                    edge_length = network.edges[u, v]['length']
                    human_pos = u_pos + edge_vec * (coord / edge_length)
                else:
                    node = position
                    human_pos = np.array([network.nodes[node]['x'], network.nodes[node]['y']])
                
                # Get target position
                target = policy.target
                target_pos = np.array([network.nodes[target]['x'], network.nodes[target]['y']])

                # Compute distance
                distance = np.linalg.norm(human_pos - target_pos)
                total_distance += distance
                distance_count += 1
                
                # Check if target was just set (track when it changes)
                if human_id not in human_current_target_set_time:
                    human_current_target_set_time[human_id] = (real_time, target)
                elif human_current_target_set_time[human_id][1] != target:
                    # Target changed without being reached
                    human_current_target_set_time[human_id] = (real_time, target)
                
                # Check if at target (within small threshold)
                if isinstance(position, int) and position == target:
                    # Reached target!
                    if human_id in human_current_target_set_time:
                        set_time, old_target = human_current_target_set_time[human_id]
                        if old_target == target:
                            travel_time = real_time - set_time
                            human_target_history[human_id].append((set_time, real_time, travel_time))
                            # Reset so we don't count multiple times
                            del human_current_target_set_time[human_id]
        
        stats['humans_aboard'].append(humans_aboard)
        stats['humans_walking'].append(humans_walking)
        stats['humans_waiting'].append(humans_waiting)
        stats['avg_distance_to_target'].append(total_distance / distance_count if distance_count > 0 else 0)
        
        # Count targets reached
        total_reached = sum(len(history) for history in human_target_history.values())
        stats['targets_reached'].append(total_reached)
        
        # Average time to reach target
        all_travel_times = []
        for history in human_target_history.values():
            all_travel_times.extend([travel_time for _, _, travel_time in history])
        avg_time = np.mean(all_travel_times) if all_travel_times else 0
        stats['avg_time_to_target'].append(avg_time)
        
        # Vehicle occupancy
        occupancies = []
        for i in range(NUM_VEHICLES):
            vehicle_id = f'vehicle_{i}'
            occupancy = sum(1 for h in range(NUM_HUMANS) if env.human_aboard[f'human_{h}'] == vehicle_id)
            occupancies.append(occupancy)
        stats['vehicle_occupancy'].append(np.mean(occupancies))
        
        # Network utilization
        agents_on_edges = sum(1 for pos in env.agent_positions.values() if isinstance(pos, tuple))
        agents_at_nodes = sum(1 for pos in env.agent_positions.values() if not isinstance(pos, tuple))
        stats['agents_on_edges'].append(agents_on_edges)
        stats['agents_at_nodes'].append(agents_at_nodes)
        
        # Progress indicator
        if (step + 1) % 50 == 0:
            print(f"   Step {step + 1}/{NUM_STEPS} (time: {real_time:.1f}s, targets reached: {total_reached})")
    
    # Save video
    print("\n5. Saving video...")
    env.save_video('large_scale_simulation.mp4', fps=10)
    print(f"   Saved video with {frame_count} frames")
    
    # Close environment
    env.close()
    
    # Plot statistics
    print("\n6. Generating statistics plots...")
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle('Large-Scale Transport Simulation Statistics', fontsize=16, fontweight='bold')
    
    time_array = np.array(stats['time'])
    
    # Plot 1: Human status distribution
    ax = axes[0, 0]
    ax.plot(time_array, stats['humans_aboard'], label='Aboard vehicles', color='blue', alpha=0.7)
    ax.plot(time_array, stats['humans_walking'], label='Walking on edges', color='orange', alpha=0.7)
    ax.plot(time_array, stats['humans_waiting'], label='Waiting at nodes', color='green', alpha=0.7)
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Number of humans')
    ax.set_title('Human Status Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Average distance to target
    ax = axes[0, 1]
    # Smooth the data
    window = 20
    smoothed = np.convolve(stats['avg_distance_to_target'], np.ones(window)/window, mode='valid')
    ax.plot(time_array[:len(smoothed)], smoothed, color='red', linewidth=2)
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Distance (units)')
    ax.set_title('Average Distance to Target (Smoothed)')
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Targets reached (cumulative)
    ax = axes[1, 0]
    ax.plot(time_array, stats['targets_reached'], color='green', linewidth=2)
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Count')
    ax.set_title('Cumulative Targets Reached')
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Average time to reach target
    ax = axes[1, 1]
    # Smooth the data
    smoothed = np.convolve(stats['avg_time_to_target'], np.ones(window)/window, mode='valid')
    ax.plot(time_array[:len(smoothed)], smoothed, color='purple', linewidth=2)
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Time (seconds)')
    ax.set_title('Average Time to Reach Target (Smoothed)')
    ax.grid(True, alpha=0.3)
    
    # Plot 5: Vehicle occupancy
    ax = axes[2, 0]
    smoothed = np.convolve(stats['vehicle_occupancy'], np.ones(window)/window, mode='valid')
    ax.plot(time_array[:len(smoothed)], smoothed, color='blue', linewidth=2)
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Average passengers per vehicle')
    ax.set_title('Vehicle Occupancy (Smoothed)')
    ax.grid(True, alpha=0.3)
    
    # Plot 6: Network utilization
    ax = axes[2, 1]
    ax.plot(time_array, stats['agents_on_edges'], label='On edges', color='orange', alpha=0.7)
    ax.plot(time_array, stats['agents_at_nodes'], label='At nodes', color='green', alpha=0.7)
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Number of agents')
    ax.set_title('Network Utilization')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('large_scale_statistics.png', dpi=150, bbox_inches='tight')
    print("   Saved statistics plot to large_scale_statistics.png")
    
    # Print summary statistics
    print("\n" + "=" * 70)
    print("SIMULATION SUMMARY")
    print("=" * 70)
    print(f"Total simulation time: {time_array[-1]:.1f} seconds")
    print(f"Total steps: {NUM_STEPS}")
    print()
    print("Human Status (Final):")
    print(f"  Aboard vehicles: {stats['humans_aboard'][-1]} ({stats['humans_aboard'][-1]/NUM_HUMANS*100:.1f}%)")
    print(f"  Walking on edges: {stats['humans_walking'][-1]} ({stats['humans_walking'][-1]/NUM_HUMANS*100:.1f}%)")
    print(f"  Waiting at nodes: {stats['humans_waiting'][-1]} ({stats['humans_waiting'][-1]/NUM_HUMANS*100:.1f}%)")
    print()
    print("Target Destinations:")
    print(f"  Total targets reached: {stats['targets_reached'][-1]}")
    print(f"  Humans who reached at least one target: {len(human_target_history)}")
    print(f"  Average targets per human: {stats['targets_reached'][-1]/NUM_HUMANS:.2f}")
    if all_travel_times:
        print(f"  Average time to reach target: {np.mean(all_travel_times):.1f} seconds")
        print(f"  Median time to reach target: {np.median(all_travel_times):.1f} seconds")
        print(f"  Min/Max time to reach target: {np.min(all_travel_times):.1f} / {np.max(all_travel_times):.1f} seconds")
    print()
    print("Vehicle Metrics:")
    print(f"  Average occupancy: {np.mean(stats['vehicle_occupancy']):.2f} passengers")
    print(f"  Peak occupancy: {np.max(stats['vehicle_occupancy']):.2f} passengers")
    print()
    print("Network Utilization:")
    print(f"  Average agents on edges: {np.mean(stats['agents_on_edges']):.1f}")
    print(f"  Average agents at nodes: {np.mean(stats['agents_at_nodes']):.1f}")
    print()
    print("Outputs generated:")
    print("  - large_scale_simulation.mp4 (video)")
    print("  - large_scale_statistics.png (statistics plot)")
    print("=" * 70)


if __name__ == '__main__':
    main()

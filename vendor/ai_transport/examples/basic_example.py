"""
Basic example demonstrating the ai_transport environment.

This example shows:
1. Creating a parallel environment
2. Resetting the environment
3. Taking a few steps with random actions
4. Rendering the environment state
"""

import networkx as nx
from ai_transport import parallel_env


def main():
    # Create a custom network
    G = nx.DiGraph()
    G.add_node(0, name="Station_A")
    G.add_node(1, name="Station_B")
    G.add_node(2, name="Station_C")
    G.add_node(3, name="Station_D")
    
    G.add_edge(0, 1, length=10.0, speed=5.0, capacity=10)
    G.add_edge(1, 2, length=15.0, speed=5.0, capacity=8)
    G.add_edge(2, 3, length=12.0, speed=6.0, capacity=12)
    G.add_edge(3, 0, length=20.0, speed=5.0, capacity=10)
    G.add_edge(1, 3, length=8.0, speed=4.0, capacity=6)
    
    # Create environment with custom parameters
    env = parallel_env(
        render_mode="human",
        num_humans=3,
        num_vehicles=2,
        network=G,
        human_speeds=[1.5, 1.5, 1.5],
        vehicle_speeds=[3.0, 3.0],
        vehicle_capacities=[5, 5],
        vehicle_fuel_uses=[1.2, 1.2]
    )
    
    print("=" * 60)
    print("AI Transport Environment Demo")
    print("=" * 60)
    print(f"\nEnvironment created with:")
    print(f"  - {env.num_humans} human agents")
    print(f"  - {env.num_vehicles} vehicle agents")
    print(f"  - Network with {len(env.network.nodes())} nodes and {len(env.network.edges())} edges")
    print(f"\nAgent list: {env.possible_agents}")
    
    # Display agent attributes
    print("\nAgent attributes:")
    for agent in env.possible_agents:
        attrs = env.agent_attributes[agent]
        print(f"  {agent}: {attrs}")
    
    # Reset environment
    print("\n" + "=" * 60)
    print("Resetting environment...")
    print("=" * 60)
    observations, infos = env.reset(seed=42)
    env.render()
    
    # Take a few steps
    print("\n" + "=" * 60)
    print("Taking 3 steps with random actions...")
    print("=" * 60)
    
    for step in range(3):
        print(f"\n--- Step {step + 1} ---")
        
        # Random actions for all agents
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        print(f"Actions: {actions}")
        
        observations, rewards, terminations, truncations, infos = env.step(actions)
        
        print(f"Rewards: {rewards}")
        env.render()
        
        # Check if any agents terminated
        if any(terminations.values()) or any(truncations.values()):
            print("Some agents terminated or truncated")
            break
    
    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)
    
    env.close()


if __name__ == "__main__":
    main()

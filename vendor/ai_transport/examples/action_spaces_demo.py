"""
Example demonstrating the ai_transport environment with different step types.

This example shows:
1. Creating a parallel environment
2. Different step types and their action spaces
3. How action spaces change based on agent state and step type
"""

import networkx as nx
from ai_transport import parallel_env


def main():
    # Create a custom network
    G = nx.DiGraph()
    G.add_node(0, name="Station_A")
    G.add_node(1, name="Station_B")
    G.add_node(2, name="Station_C")
    
    G.add_edge(0, 1, length=10.0, speed=5.0, capacity=10)
    G.add_edge(1, 2, length=15.0, speed=5.0, capacity=8)
    G.add_edge(2, 0, length=12.0, speed=6.0, capacity=12)
    
    # Create environment
    env = parallel_env(
        render_mode="human",
        num_humans=2,
        num_vehicles=1,
        network=G,
        human_speeds=[1.5, 1.5],
        vehicle_speeds=[3.0],
        vehicle_capacities=[5],
        vehicle_fuel_uses=[1.2]
    )
    
    print("=" * 70)
    print("AI Transport Environment Demo - Action Spaces by Step Type")
    print("=" * 70)
    print(f"\nEnvironment created with:")
    print(f"  - {env.num_humans} human agents")
    print(f"  - {env.num_vehicles} vehicle agents")
    print(f"  - Network with {len(env.network.nodes())} nodes and {len(env.network.edges())} edges")
    
    # Reset environment
    print("\n" + "=" * 70)
    print("Resetting environment...")
    print("=" * 70)
    observations, infos = env.reset(seed=42)
    env.render()
    
    # Demonstrate different step types
    print("\n" + "=" * 70)
    print("Step Type: ROUTING")
    print("=" * 70)
    env.step_type = 'routing'
    print("\nAction spaces in routing step:")
    for agent in env.agents:
        space = env.action_space(agent)
        print(f"  {agent}: {space.n} actions (", end="")
        if agent in env.vehicle_agents:
            print(f"set destination to None or any of {len(env.network.nodes())} nodes)")
        else:
            print("can only pass)")
    
    print("\n" + "=" * 70)
    print("Step Type: BOARDING")
    print("=" * 70)
    env.step_type = 'boarding'
    # Make sure humans are at a node and not aboard
    env.agent_positions['human_0'] = 0
    env.agent_positions['human_1'] = 0
    env.agent_positions['vehicle_0'] = 0
    env.human_aboard['human_0'] = None
    env.human_aboard['human_1'] = None
    
    print("\nAction spaces in boarding step:")
    print("(Humans at node 0, vehicle at node 0)")
    for agent in env.agents:
        space = env.action_space(agent)
        print(f"  {agent}: {space.n} actions (", end="")
        if agent in env.human_agents:
            vehicles_at_node = [v for v in env.vehicle_agents 
                               if env.agent_positions.get(v) == env.agent_positions[agent]]
            print(f"pass or board one of {len(vehicles_at_node)} vehicles)")
        else:
            print("can only pass)")
    
    print("\n" + "=" * 70)
    print("Step Type: UNBOARDING")
    print("=" * 70)
    env.step_type = 'unboarding'
    # Put one human aboard the vehicle
    env.human_aboard['human_0'] = 'vehicle_0'
    
    print("\nAction spaces in unboarding step:")
    print("(human_0 aboard vehicle_0, vehicle at node 0)")
    for agent in env.agents:
        space = env.action_space(agent)
        print(f"  {agent}: {space.n} actions (", end="")
        if agent in env.human_agents:
            if env.human_aboard[agent] is not None:
                print("pass or unboard)")
            else:
                print("can only pass - not aboard)")
        else:
            print("can only pass)")
    
    print("\n" + "=" * 70)
    print("Step Type: DEPARTING")
    print("=" * 70)
    env.step_type = 'departing'
    # Reset positions to node 0
    env.agent_positions['human_0'] = 0
    env.agent_positions['human_1'] = 0
    env.agent_positions['vehicle_0'] = 0
    env.human_aboard['human_0'] = None
    env.human_aboard['human_1'] = 'vehicle_0'  # human_1 aboard
    
    outgoing = list(env.network.out_edges(0))
    print(f"\nAction spaces in departing step:")
    print(f"(Node 0 has {len(outgoing)} outgoing edges)")
    print("(human_0 not aboard, human_1 aboard vehicle_0)")
    for agent in env.agents:
        space = env.action_space(agent)
        print(f"  {agent}: {space.n} actions (", end="")
        if agent in env.vehicle_agents:
            print(f"pass or depart into one of {len(outgoing)} edges)")
        elif agent in env.human_agents:
            if env.human_aboard[agent] is None:
                print(f"pass or walk into one of {len(outgoing)} edges)")
            else:
                print("can only pass - aboard vehicle)")
    
    print("\n" + "=" * 70)
    print("Demo complete!")
    print("=" * 70)
    
    env.close()


if __name__ == "__main__":
    main()

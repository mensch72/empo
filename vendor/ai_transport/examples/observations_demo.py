"""
Example demonstrating different observation scenarios.

This example shows the three observation modes:
1. Full: Agents observe complete state
2. Local: Agents observe only agents at same location
3. Statistical: As local, plus counts at all locations
"""

import networkx as nx
from ai_transport import parallel_env


def print_observation_summary(obs, agent_name, scenario):
    """Print a summary of an observation"""
    print(f"\n{agent_name} observation ({scenario}):")
    print(f"  Keys: {list(obs.keys())}")
    
    if 'agents_here' in obs:
        print(f"  Agents at same location: {list(obs['agents_here'].keys())}")
    
    if 'agent_positions' in obs:
        print(f"  Total agents visible: {len(obs['agent_positions'])}")
    
    if 'node_counts' in obs:
        total_at_nodes = sum(counts['humans'] + counts['vehicles'] 
                            for counts in obs['node_counts'].values())
        print(f"  Total agents at nodes (statistical): {total_at_nodes}")


def main():
    # Create a network
    G = nx.DiGraph()
    G.add_node(0, name="Station_A")
    G.add_node(1, name="Station_B")
    G.add_node(2, name="Station_C")
    
    G.add_edge(0, 1, length=10.0, speed=5.0, capacity=10)
    G.add_edge(1, 2, length=15.0, speed=5.0, capacity=8)
    G.add_edge(2, 0, length=12.0, speed=6.0, capacity=12)
    
    print("=" * 70)
    print("AI Transport Environment - Observation Scenarios Demo")
    print("=" * 70)
    
    # Test each observation scenario
    for scenario in ['full', 'local', 'statistical']:
        print(f"\n{'=' * 70}")
        print(f"Scenario: {scenario.upper()}")
        print("=" * 70)
        
        env = parallel_env(
            num_humans=2,
            num_vehicles=1,
            network=G,
            observation_scenario=scenario
        )
        
        env.reset(seed=42)
        
        # Initially all agents at node 0
        print("\nInitial state: All agents at node 0")
        obs = env._generate_observation_for_agent('human_0')
        print_observation_summary(obs, 'human_0', scenario)
        
        # Move agents to different locations
        env.agent_positions['human_0'] = 0
        env.agent_positions['human_1'] = 1
        env.agent_positions['vehicle_0'] = 1
        
        print("\nAfter moving: human_0 at node 0, human_1 and vehicle_0 at node 1")
        
        # Check human_0's observation
        obs_h0 = env._generate_observation_for_agent('human_0')
        print_observation_summary(obs_h0, 'human_0', scenario)
        
        # Check human_1's observation
        obs_h1 = env._generate_observation_for_agent('human_1')
        print_observation_summary(obs_h1, 'human_1', scenario)
        
        # Move human_1 onto edge
        edge = (1, 2)
        env.agent_positions['human_1'] = (edge, 5.0)
        
        print(f"\nAfter moving: human_1 on edge {edge}")
        obs_h1 = env._generate_observation_for_agent('human_1')
        print_observation_summary(obs_h1, 'human_1', scenario)
    
    # Detailed comparison
    print("\n" + "=" * 70)
    print("DETAILED COMPARISON")
    print("=" * 70)
    
    env_full = parallel_env(num_humans=1, num_vehicles=1, network=G, 
                           observation_scenario='full')
    env_full.reset()
    env_full.agent_positions['human_0'] = 0
    env_full.agent_positions['vehicle_0'] = 1
    
    print("\nSetup: human_0 at node 0, vehicle_0 at node 1")
    print("\n--- FULL Observation (human_0) ---")
    obs_full = env_full._generate_observation_for_agent('human_0')
    print(f"Can see all {len(obs_full['agent_positions'])} agents:")
    for agent, pos in obs_full['agent_positions'].items():
        print(f"  {agent}: {pos}")
    
    env_local = parallel_env(num_humans=1, num_vehicles=1, network=G,
                            observation_scenario='local')
    env_local.reset()
    env_local.agent_positions['human_0'] = 0
    env_local.agent_positions['vehicle_0'] = 1
    
    print("\n--- LOCAL Observation (human_0) ---")
    obs_local = env_local._generate_observation_for_agent('human_0')
    print(f"Can see {len(obs_local['agents_here'])} agent(s) at same location:")
    for agent in obs_local['agents_here'].keys():
        print(f"  {agent}")
    
    env_stat = parallel_env(num_humans=1, num_vehicles=1, network=G,
                           observation_scenario='statistical')
    env_stat.reset()
    env_stat.agent_positions['human_0'] = 0
    env_stat.agent_positions['vehicle_0'] = 1
    
    print("\n--- STATISTICAL Observation (human_0) ---")
    obs_stat = env_stat._generate_observation_for_agent('human_0')
    print(f"Can see {len(obs_stat['agents_here'])} agent(s) at same location")
    print("Plus statistical counts:")
    print("  Node counts:")
    for node, counts in obs_stat['node_counts'].items():
        if counts['humans'] > 0 or counts['vehicles'] > 0:
            print(f"    Node {node}: {counts['humans']} humans, {counts['vehicles']} vehicles")
    
    # Test rewards
    print("\n" + "=" * 70)
    print("REWARDS TEST")
    print("=" * 70)
    
    env = parallel_env(num_humans=1, num_vehicles=1, network=G)
    env.reset()
    env.step_type = 'routing'
    
    actions = {agent: 0 for agent in env.agents}
    obs, rewards, terms, truncs, infos = env.step(actions)
    
    print("\nAll rewards are constantly zero:")
    for agent, reward in rewards.items():
        print(f"  {agent}: {reward}")
    
    print("\n" + "=" * 70)
    print("Demo complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()

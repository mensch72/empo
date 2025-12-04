#!/usr/bin/env python
"""Test script for new requirements"""

from ai_transport import parallel_env
import networkx as nx

print("Testing new requirements...")
print("="*70)

# Test 1: Lists of speeds, capacities, fuel_uses
print("\n1. Testing init with lists of attributes...")
env = parallel_env(
    num_humans=3,
    num_vehicles=2,
    human_speeds=[1.0, 1.5, 2.0],
    vehicle_speeds=[3.0, 4.0],
    vehicle_capacities=[4, 6],
    vehicle_fuel_uses=[1.0, 1.5]
)
print("   ✓ Created environment with custom attribute lists")
print(f"   Human 0 speed: {env.agent_attributes['human_0']['speed']}")
print(f"   Human 1 speed: {env.agent_attributes['human_1']['speed']}")
print(f"   Human 2 speed: {env.agent_attributes['human_2']['speed']}")
print(f"   Vehicle 0 capacity: {env.agent_attributes['vehicle_0']['capacity']}")
print(f"   Vehicle 1 capacity: {env.agent_attributes['vehicle_1']['capacity']}")

# Test 2: Reset initializes positions randomly
print("\n2. Testing reset() initializes positions randomly...")
obs, info = env.reset(seed=42)
print(f"   ✓ Reset called, positions initialized")
print(f"   Agent positions after reset: {env.agent_positions}")
print(f"   Human aboard status: {env.human_aboard}")

# Check if positions are different across multiple resets
obs1, _ = env.reset(seed=1)
pos1 = dict(env.agent_positions)
obs2, _ = env.reset(seed=2)
pos2 = dict(env.agent_positions)
if pos1 != pos2:
    print("   ✓ Positions vary across different seeds")
else:
    print("   ⚠ Positions same across different seeds (might be coincidence)")

# Test 3: Network data is cached
print("\n3. Testing cached network data...")
print(f"   ✓ Network nodes cached: {env._cached_network_nodes is not None}")
print(f"   ✓ Network edges cached: {env._cached_network_edges is not None}")
print(f"   Number of cached nodes: {len(env._cached_network_nodes)}")
print(f"   Number of cached edges: {len(env._cached_network_edges)}")

# Test 4: Local observation ignores coordinate on edges
print("\n4. Testing local observation on same edge...")
# Create a simple network
G = nx.DiGraph()
G.add_node(0, name="A", x=0.0, y=0.0)
G.add_node(1, name="B", x=10.0, y=0.0)
G.add_edge(0, 1, length=10.0, speed=5.0, capacity=10)

env2 = parallel_env(num_humans=2, num_vehicles=0, network=G, observation_scenario='local')
obs, _ = env2.reset()

# Manually place both humans on same edge at different coordinates
env2.agent_positions['human_0'] = ((0, 1), 3.0)
env2.agent_positions['human_1'] = ((0, 1), 7.0)

# Generate observations
obs = env2._generate_observations()
agents_here_0 = obs['human_0']['agents_here']
agents_here_1 = obs['human_1']['agents_here']

print(f"   Human 0 at coordinate 3.0 sees: {list(agents_here_0.keys())}")
print(f"   Human 1 at coordinate 7.0 sees: {list(agents_here_1.keys())}")

if 'human_1' in agents_here_0 and 'human_0' in agents_here_1:
    print("   ✓ Agents on same edge see each other regardless of coordinate")
else:
    print("   ✗ ERROR: Agents on same edge don't see each other!")

# Test 5: What happens when all agents pass in departing step
print("\n5. Testing all agents pass in departing step...")
env3 = parallel_env(num_humans=2, num_vehicles=1)
obs, _ = env3.reset(seed=100)

# Check if any agents are on edges initially
agents_on_edges = [a for a, pos in env3.agent_positions.items() if isinstance(pos, tuple)]
print(f"   Agents initially on edges: {agents_on_edges}")

# Move to departing step
env3.step_type = 'departing'
initial_time = env3.real_time
print(f"   Initial real_time: {initial_time}")
print(f"   Initial step_type: {env3.step_type}")

# All agents pass (action 0)
actions = {agent: 0 for agent in env3.agents}
obs, rewards, terms, truncs, infos = env3.step(actions)

print(f"   After all pass:")
print(f"   New real_time: {env3.real_time}")
print(f"   New step_type: {env3.step_type}")

if agents_on_edges:
    if env3.real_time > initial_time:
        print(f"   ✓ Real time advanced because agents were already on edges")
    else:
        print(f"   ✗ ERROR: Real time should have advanced (agents on edges)")
else:
    if env3.real_time == initial_time:
        print("   ✓ Real time did NOT advance when all agents passed (none on edges)")
    else:
        print(f"   ✗ ERROR: Real time advanced by {env3.real_time - initial_time}")

if env3.step_type == 'routing':
    print("   ✓ Step type advanced to 'routing' (next in cycle)")
else:
    print(f"   ✗ ERROR: Step type is '{env3.step_type}', expected 'routing'")

# Test case 2: All agents at nodes
print("\n6. Testing all agents at nodes, all pass in departing step...")
env4 = parallel_env(num_humans=2, num_vehicles=1)
obs, _ = env4.reset(seed=1)
# Manually place all at nodes
env4.agent_positions = {agent: 0 for agent in env4.agents}
env4.step_type = 'departing'
initial_time = env4.real_time

actions = {agent: 0 for agent in env4.agents}
obs, rewards, terms, truncs, infos = env4.step(actions)

if env4.real_time == initial_time:
    print("   ✓ Real time did NOT advance when all at nodes and all passed")
else:
    print(f"   ✗ ERROR: Real time advanced by {env4.real_time - initial_time}")

print("\n" + "="*70)
print("All tests completed!")

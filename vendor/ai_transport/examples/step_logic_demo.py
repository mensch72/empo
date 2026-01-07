"""
Example demonstrating the step logic for different step types.

This example shows:
1. Routing step: vehicles set destinations
2. Unboarding step: humans unboard from vehicles
3. Boarding step: humans board vehicles with capacity constraints
4. Departing step: agents move along edges with time advancing
"""

import networkx as nx
from ai_transport import parallel_env


def main():
    # Create a network
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
        human_speeds=[2.0, 2.0],
        vehicle_speeds=[3.0],
        vehicle_capacities=[2],
        vehicle_fuel_uses=[1.2]
    )
    
    print("=" * 70)
    print("AI Transport Environment - Step Logic Demo")
    print("=" * 70)
    
    # Reset environment
    print("\n--- Initial State ---")
    env.reset(seed=42)
    env.render()
    
    # Step 1: Routing
    print("\n" + "=" * 70)
    print("Step 1: ROUTING - Vehicle sets destination to node 2")
    print("=" * 70)
    env.step_type = 'routing'
    actions = {
        'human_0': 0,  # Pass
        'human_1': 0,  # Pass
        'vehicle_0': 3  # Set destination to node 2 (action 3 = node index 2)
    }
    env.step(actions)
    env.render()
    
    # Step 2: Boarding
    print("\n" + "=" * 70)
    print("Step 2: BOARDING - Both humans try to board vehicle")
    print("=" * 70)
    env.step_type = 'boarding'
    actions = {
        'human_0': 1,  # Board vehicle_0
        'human_1': 1,  # Board vehicle_0
        'vehicle_0': 0  # Pass
    }
    env.step(actions)
    env.render()
    print(f"Vehicle capacity: {env.agent_attributes['vehicle_0']['capacity']}")
    
    # Step 3: Departing
    print("\n" + "=" * 70)
    print("Step 3: DEPARTING - Vehicle departs with humans aboard")
    print("=" * 70)
    env.step_type = 'departing'
    actions = {
        'human_0': 0,  # Pass (aboard, can't walk)
        'human_1': 0,  # Pass (aboard, can't walk)
        'vehicle_0': 1  # Depart on first outgoing edge
    }
    print(f"Before step - Real time: {env.real_time:.2f}")
    env.step(actions)
    print(f"After step - Real time: {env.real_time:.2f}")
    env.render()
    
    # Step 4: Continue departing (time advances)
    print("\n" + "=" * 70)
    print("Step 4: DEPARTING - Time advances, agents move along edge")
    print("=" * 70)
    actions = {
        'human_0': 0,
        'human_1': 0,
        'vehicle_0': 0
    }
    print(f"Before step - Real time: {env.real_time:.2f}")
    env.step(actions)
    print(f"After step - Real time: {env.real_time:.2f}")
    env.render()
    
    # Step 5: Unboarding
    print("\n" + "=" * 70)
    print("Step 5: UNBOARDING - One human unboards")
    print("=" * 70)
    env.step_type = 'unboarding'
    actions = {
        'human_0': 1,  # Unboard
        'human_1': 0,  # Stay aboard
        'vehicle_0': 0  # Pass
    }
    env.step(actions)
    env.render()
    
    # Step 6: Departing again
    print("\n" + "=" * 70)
    print("Step 6: DEPARTING - Human walks, vehicle departs")
    print("=" * 70)
    env.step_type = 'departing'
    outgoing = list(env.network.out_edges(1))
    print(f"Available edges from node 1: {outgoing}")
    
    actions = {
        'human_0': 1,  # Walk on first edge (not aboard)
        'human_1': 0,  # Pass (aboard)
        'vehicle_0': 1  # Depart on first edge
    }
    print(f"Before step - Real time: {env.real_time:.2f}")
    env.step(actions)
    print(f"After step - Real time: {env.real_time:.2f}")
    env.render()
    
    print("\n" + "=" * 70)
    print("Demo complete!")
    print("=" * 70)
    
    env.close()


if __name__ == "__main__":
    main()

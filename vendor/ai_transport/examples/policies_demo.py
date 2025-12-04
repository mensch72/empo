"""
Demonstration of policy classes for AI Transport environment.

Shows how to use the RandomHumanPolicy, TargetDestinationHumanPolicy,
RandomVehiclePolicy, and ShortestPathVehiclePolicy classes.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_transport import parallel_env
from ai_transport.policies import (
    RandomHumanPolicy,
    TargetDestinationHumanPolicy,
    RandomVehiclePolicy,
    ShortestPathVehiclePolicy
)

def main():
    print("="*70)
    print("AI Transport Environment - Policy Demonstration")
    print("="*70)
    print()
    
    # Create environment with random 2D network
    print("1. Creating environment with random network...")
    env = parallel_env(num_humans=4, num_vehicles=2, observation_scenario='full')
    network = env.create_random_2d_network(num_nodes=10, bidirectional_prob=0.5, seed=42)
    env = parallel_env(
        num_humans=4,
        num_vehicles=2,
        network=network,
        observation_scenario='full',
        render_mode='human'
    )
    
    obs, info = env.reset(seed=42)
    print(f"   Network: {network.number_of_nodes()} nodes, {network.number_of_edges()} edges")
    print(f"   Agents: 4 humans, 2 vehicles")
    print()
    
    # Create policies for agents
    print("2. Creating policies...")
    policies = {}
    
    # Human policies
    policies['human_0'] = RandomHumanPolicy('human_0', pass_prob_boarding=0.5, seed=1)
    policies['human_1'] = RandomHumanPolicy('human_1', pass_prob_boarding=0.3, seed=2)
    policies['human_2'] = TargetDestinationHumanPolicy('human_2', network, target_change_rate=0.1, seed=3)
    policies['human_3'] = TargetDestinationHumanPolicy('human_3', network, target_change_rate=0.05, seed=4)
    
    # Vehicle policies
    policies['vehicle_0'] = RandomVehiclePolicy('vehicle_0', pass_prob_routing=0.2, seed=5)
    policies['vehicle_1'] = ShortestPathVehiclePolicy('vehicle_1', network, seed=6)
    
    print("   Policies:")
    print("   - human_0: RandomHumanPolicy (pass_prob_boarding=0.5)")
    print("   - human_1: RandomHumanPolicy (pass_prob_boarding=0.3)")
    print("   - human_2: TargetDestinationHumanPolicy (change_rate=0.1)")
    print("   - human_3: TargetDestinationHumanPolicy (change_rate=0.05)")
    print("   - vehicle_0: RandomVehiclePolicy (pass_prob_routing=0.2)")
    print("   - vehicle_1: ShortestPathVehiclePolicy")
    print()
    
    # Run simulation
    print("3. Running simulation with policies...")
    print()
    
    for step in range(20):
        # Print summary every 4 steps (before taking action)
        if step % 4 == 0:
            print(f"   Step {step}:")
            print(f"   - Step type: {env.step_type}")
            print(f"   - Real time: {env.real_time:.2f}s")
            print(f"   - Active agents: {len(env.agents)}")
            
            # Show target destinations for TargetDestinationHumanPolicy
            for agent_id, policy in policies.items():
                if isinstance(policy, TargetDestinationHumanPolicy):
                    target = policy.target_destination
                    print(f"   - {agent_id} target: node {target}")
                elif isinstance(policy, ShortestPathVehiclePolicy):
                    dest = policy.current_destination
                    print(f"   - {agent_id} destination: node {dest}")
            
            print()
            print("   Actions taken:")
        
        # Get actions from policies
        actions = {}
        action_justifications = {}
        for agent in env.agents:
            policy = policies.get(agent)
            if policy:
                action_space_size = env.action_space(agent).n
                action, justification = policy.get_action(obs[agent], action_space_size)
                actions[agent] = action
                action_justifications[agent] = justification
            else:
                actions[agent] = 0  # Pass
                action_justifications[agent] = "Passing (no policy)"
        
        # Print actions with justifications
        for agent in env.agents:
            action_idx = actions[agent]
            justification = action_justifications[agent]
            print(f"   - {agent}: action {action_idx} - {justification}")
        
        print()
        
        # Step environment
        obs, rewards, terminations, truncations, infos = env.step(actions)
    
    print("="*70)
    print("Policy demonstration complete!")
    print("="*70)
    print()
    print("Policy Features Demonstrated:")
    print()
    print("Human Policies:")
    print("- RandomHumanPolicy: Random actions with configurable pass probabilities")
    print("- TargetDestinationHumanPolicy: Boards vehicles heading toward target,")
    print("  walks toward target, target changes over time")
    print()
    print("Vehicle Policies:")
    print("- RandomVehiclePolicy: Random actions with configurable pass probabilities")
    print("- ShortestPathVehiclePolicy: Takes fastest path to destination,")
    print("  chooses new destination proportional to distance")
    print()

if __name__ == "__main__":
    main()

"""
Test for ai_transport environment import and basic functionality.

This test validates that:
1. ai_transport is properly vendored and importable
2. Basic environment creation works
3. Environment step cycle works
4. Policy classes are available

Run with: pytest tests/test_transport_env.py -v
"""

import pytest
import sys
import os
from pathlib import Path

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "vendor" / "ai_transport"))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def test_import_ai_transport():
    """Test that ai_transport package can be imported."""
    import ai_transport
    assert hasattr(ai_transport, 'parallel_env')


def test_import_parallel_env():
    """Test that parallel_env can be imported directly."""
    from ai_transport import parallel_env
    assert callable(parallel_env)


def test_import_policies():
    """Test that policy classes can be imported."""
    from ai_transport.policies import (
        RandomHumanPolicy,
        TargetDestinationHumanPolicy,
        RandomVehiclePolicy,
        ShortestPathVehiclePolicy
    )
    assert callable(RandomHumanPolicy)
    assert callable(TargetDestinationHumanPolicy)
    assert callable(RandomVehiclePolicy)
    assert callable(ShortestPathVehiclePolicy)


def test_create_environment():
    """Test basic environment creation."""
    from ai_transport import parallel_env
    
    env = parallel_env(
        num_humans=2,
        num_vehicles=1
    )
    
    assert env is not None
    assert env.num_humans == 2
    assert env.num_vehicles == 1
    assert len(env.possible_agents) == 3


def test_environment_reset():
    """Test environment reset."""
    from ai_transport import parallel_env
    
    env = parallel_env(
        num_humans=2,
        num_vehicles=1
    )
    
    obs, info = env.reset(seed=42)
    
    assert obs is not None
    assert isinstance(obs, dict)
    assert len(obs) == 3  # 2 humans + 1 vehicle


def test_environment_step():
    """Test environment step."""
    from ai_transport import parallel_env
    
    env = parallel_env(
        num_humans=2,
        num_vehicles=1
    )
    
    env.reset(seed=42)
    
    # Get random actions for all agents
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    
    obs, rewards, terminations, truncations, infos = env.step(actions)
    
    assert obs is not None
    assert isinstance(rewards, dict)
    assert isinstance(terminations, dict)
    assert isinstance(truncations, dict)


def test_create_random_network():
    """Test random network generation."""
    from ai_transport import parallel_env
    
    env = parallel_env(num_humans=2, num_vehicles=1)
    
    network = env.create_random_2d_network(
        num_nodes=10,
        bidirectional_prob=0.5,
        seed=42
    )
    
    assert network is not None
    assert network.number_of_nodes() == 10
    assert network.number_of_edges() > 0


def test_environment_with_custom_network():
    """Test environment with custom network."""
    import networkx as nx
    from ai_transport import parallel_env
    
    # Create a simple custom network
    G = nx.DiGraph()
    G.add_node(0, name="A")
    G.add_node(1, name="B")
    G.add_node(2, name="C")
    G.add_edge(0, 1, length=10.0, speed=5.0, capacity=10)
    G.add_edge(1, 2, length=15.0, speed=5.0, capacity=10)
    G.add_edge(2, 0, length=12.0, speed=5.0, capacity=10)
    
    env = parallel_env(
        num_humans=2,
        num_vehicles=1,
        network=G
    )
    
    assert env.network.number_of_nodes() == 3
    assert env.network.number_of_edges() == 3


def test_step_types():
    """Test that environment cycles through step types correctly."""
    from ai_transport import parallel_env
    
    env = parallel_env(num_humans=2, num_vehicles=1)
    env.reset(seed=42)
    
    step_types_seen = set()
    
    for _ in range(20):
        step_types_seen.add(env.step_type)
        actions = {agent: 0 for agent in env.agents}  # All pass
        env.step(actions)
    
    # Should see multiple step types
    assert len(step_types_seen) >= 2


def test_policy_creation():
    """Test that policies can be created and used."""
    from ai_transport import parallel_env
    from ai_transport.policies import RandomHumanPolicy, RandomVehiclePolicy
    
    env = parallel_env(num_humans=2, num_vehicles=1)
    obs, _ = env.reset(seed=42)
    
    # Create policies
    human_policy = RandomHumanPolicy('human_0', seed=42)
    vehicle_policy = RandomVehiclePolicy('vehicle_0', seed=42)
    
    # Get action from human policy
    action_space_size = env.action_space('human_0').n
    action, justification = human_policy.get_action(obs['human_0'], action_space_size)
    assert isinstance(action, int)
    assert 0 <= action < action_space_size
    
    # Get action from vehicle policy
    action_space_size = env.action_space('vehicle_0').n
    action, justification = vehicle_policy.get_action(obs['vehicle_0'], action_space_size)
    assert isinstance(action, int)
    assert 0 <= action < action_space_size


def test_observation_scenarios():
    """Test different observation scenarios."""
    from ai_transport import parallel_env
    
    for scenario in ['full', 'local', 'statistical']:
        env = parallel_env(
            num_humans=2,
            num_vehicles=1,
            observation_scenario=scenario
        )
        obs, _ = env.reset(seed=42)
        
        assert obs is not None
        for agent_id, agent_obs in obs.items():
            assert 'step_type' in agent_obs
            assert 'real_time' in agent_obs


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

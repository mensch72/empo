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


# =============================================================================
# Tests for TransportEnvWrapper and Goal Classes
# =============================================================================

def test_wrapper_action_mask_in_observation():
    """Test that action mask is included in observations."""
    from empo.transport import create_transport_env, TransportActions
    import numpy as np
    
    env = create_transport_env(num_humans=2, num_vehicles=1, num_nodes=5, seed=42)
    obs = env.reset(seed=42)
    
    # Each observation should have an action_mask
    for i, agent_obs in enumerate(obs):
        assert 'action_mask' in agent_obs
        mask = agent_obs['action_mask']
        assert isinstance(mask, np.ndarray)
        assert mask.shape == (TransportActions.NUM_ACTIONS,)
        assert mask.dtype == bool
        # PASS action should always be valid
        assert mask[TransportActions.PASS] == True


def test_wrapper_action_mask_matches_method():
    """Test that action mask in observation matches action_masks() method."""
    from empo.transport import create_transport_env
    import numpy as np
    
    env = create_transport_env(num_humans=2, num_vehicles=1, num_nodes=5, seed=42)
    obs = env.reset(seed=42)
    
    # Get masks from method
    masks_from_method = env.action_masks()
    
    # Compare with masks in observations
    for i, agent_obs in enumerate(obs):
        assert np.array_equal(agent_obs['action_mask'], masks_from_method[i])


def test_transport_goal():
    """Test TransportGoal class."""
    from empo.transport import create_transport_env, TransportGoal
    
    env = create_transport_env(num_humans=2, num_vehicles=1, num_nodes=5, seed=42)
    env.reset(seed=42)
    
    # Create a goal
    goal = TransportGoal(env, agent_idx=0, target_node=2)
    
    assert goal.agent_idx == 0
    assert goal.target_node == 2
    assert goal.target_pos == 2  # For compatibility
    
    # Test hash and equality
    goal2 = TransportGoal(env, agent_idx=0, target_node=2)
    goal3 = TransportGoal(env, agent_idx=0, target_node=3)
    
    assert hash(goal) == hash(goal2)
    assert goal == goal2
    assert goal != goal3


def test_transport_goal_generator():
    """Test TransportGoalGenerator class."""
    from empo.transport import create_transport_env, TransportGoalGenerator, TransportGoal
    
    env = create_transport_env(num_humans=2, num_vehicles=1, num_nodes=5, seed=42)
    env.reset(seed=42)
    
    generator = TransportGoalGenerator(env)
    goals = list(generator.generate(state=None, human_agent_index=0))
    
    # Should generate one goal per node
    assert len(goals) == 5
    
    # Each goal should be a TransportGoal with weight 1.0
    for goal, weight in goals:
        assert isinstance(goal, TransportGoal)
        assert goal.agent_idx == 0
        assert weight == 1.0


def test_transport_goal_sampler():
    """Test TransportGoalSampler class."""
    from empo.transport import create_transport_env, TransportGoalSampler, TransportGoal
    
    env = create_transport_env(num_humans=2, num_vehicles=1, num_nodes=5, seed=42)
    env.reset(seed=42)
    
    sampler = TransportGoalSampler(env, seed=42)
    
    # Sample multiple goals
    for _ in range(10):
        goal, weight = sampler.sample(state=None, human_agent_index=0)
        assert isinstance(goal, TransportGoal)
        assert goal.agent_idx == 0
        assert weight == 1.0
        # Target node should be in the network
        assert goal.target_node in env.network.nodes()


# =============================================================================
# Tests for Network Clustering
# =============================================================================

def test_import_clustering():
    """Test that clustering module can be imported."""
    from ai_transport import cluster_network
    assert callable(cluster_network)


def test_cluster_network_basic():
    """Test basic network clustering."""
    from ai_transport import cluster_network
    import networkx as nx
    
    # Create a simple network with 2D positions
    G = nx.DiGraph()
    for i in range(12):
        G.add_node(i, x=float(i % 4), y=float(i // 4))
    
    # Add some edges
    for i in range(11):
        G.add_edge(i, i + 1)
    
    # Cluster into 3 regions
    cluster_info = cluster_network(G, k=3)
    
    assert 'node_to_cluster' in cluster_info
    assert 'cluster_to_nodes' in cluster_info
    assert 'centroids' in cluster_info
    assert 'cluster_centers' in cluster_info
    assert 'num_clusters' in cluster_info
    
    # All nodes should be assigned to a cluster
    assert len(cluster_info['node_to_cluster']) == 12
    
    # Number of clusters should match
    assert cluster_info['num_clusters'] == 3


def test_cluster_network_with_transport_env():
    """Test clustering on a transport environment network."""
    from ai_transport import parallel_env, cluster_network
    
    # Create environment with network
    env = parallel_env(num_humans=2, num_vehicles=1)
    network = env.create_random_2d_network(
        num_nodes=20,
        bidirectional_prob=0.5,
        speed_mean=3.0,
        capacity_mean=10.0,
        coord_std=10.0,
        seed=42
    )
    
    # Cluster into 4 regions
    cluster_info = cluster_network(network, k=4)
    
    # All nodes should be assigned
    assert len(cluster_info['node_to_cluster']) == 20
    
    # Should have 4 clusters
    assert cluster_info['num_clusters'] == 4
    
    # Each cluster should have at least one node
    for cluster_id in range(4):
        nodes = cluster_info['cluster_to_nodes'].get(cluster_id, [])
        assert len(nodes) > 0, f"Cluster {cluster_id} is empty"


def test_cluster_helper_functions():
    """Test cluster helper functions."""
    from ai_transport import (
        cluster_network,
        get_cluster_for_node,
        get_nodes_in_cluster,
        get_cluster_centroid
    )
    import networkx as nx
    
    # Create a simple network
    G = nx.DiGraph()
    for i in range(9):
        G.add_node(i, x=float(i % 3), y=float(i // 3))
    
    cluster_info = cluster_network(G, k=3)
    
    # Test get_cluster_for_node
    for node in G.nodes():
        cluster_id = get_cluster_for_node(node, cluster_info)
        assert cluster_id is not None
        assert 0 <= cluster_id < 3
    
    # Test get_nodes_in_cluster
    all_nodes = set()
    for cluster_id in range(3):
        nodes = get_nodes_in_cluster(cluster_id, cluster_info)
        assert len(nodes) > 0
        all_nodes.update(nodes)
    assert all_nodes == set(G.nodes())
    
    # Test get_cluster_centroid
    for cluster_id in range(3):
        centroid = get_cluster_centroid(cluster_id, cluster_info)
        assert centroid is not None
        assert centroid in G.nodes()


def test_cluster_empty_network():
    """Test clustering handles empty network."""
    from ai_transport import cluster_network
    import networkx as nx
    
    G = nx.DiGraph()
    cluster_info = cluster_network(G, k=5)
    
    assert cluster_info['num_clusters'] == 0
    assert len(cluster_info['node_to_cluster']) == 0
    assert len(cluster_info['cluster_to_nodes']) == 0


def test_cluster_k_larger_than_nodes():
    """Test clustering when k > number of nodes."""
    from ai_transport import cluster_network
    import networkx as nx
    
    # Create network with only 3 nodes
    G = nx.DiGraph()
    for i in range(3):
        G.add_node(i, x=float(i), y=0.0)
    
    # Request 10 clusters
    cluster_info = cluster_network(G, k=10)
    
    # Should only have 3 clusters (one per node)
    assert cluster_info['num_clusters'] == 3


# =============================================================================
# Tests for Cluster-Based Routing
# =============================================================================

def test_wrapper_cluster_routing_enabled():
    """Test wrapper with cluster-based routing enabled."""
    from empo.transport import create_transport_env
    
    env = create_transport_env(
        num_humans=2, 
        num_vehicles=1, 
        num_nodes=12, 
        num_clusters=4,
        seed=42
    )
    obs = env.reset(seed=42)
    
    # Should have clustering enabled
    assert env.use_clusters == True
    assert env.num_clusters == 4
    assert env.cluster_info is not None
    
    # Observations should include cluster info
    for agent_obs in obs:
        assert 'use_clusters' in agent_obs
        assert agent_obs['use_clusters'] == True
        assert 'num_clusters' in agent_obs
        assert agent_obs['num_clusters'] == 4


def test_wrapper_cluster_action_mask():
    """Test action mask with cluster-based routing."""
    from empo.transport import create_transport_env, TransportActions
    import numpy as np
    
    env = create_transport_env(
        num_humans=2, 
        num_vehicles=1, 
        num_nodes=12, 
        num_clusters=4,
        seed=42
    )
    env.reset(seed=42)
    
    # Find vehicle agent index
    vehicle_idx = env.vehicle_agent_indices[0]
    
    # Wait for routing step
    max_steps = 10
    for _ in range(max_steps):
        # Check current step type BEFORE taking action
        step_type = env.step_type
        masks = env.action_masks()
        
        if step_type == 'routing':
            # Check vehicle can set cluster destinations
            vehicle_mask = masks[vehicle_idx]
            
            # Vehicle at a node should be able to set destinations
            # Check if vehicle is at a node
            vehicle_pos = env.get_agent_position(vehicle_idx)
            if vehicle_pos is not None and not isinstance(vehicle_pos, tuple):
                # DEST_START should be valid (None destination)
                assert vehicle_mask[TransportActions.DEST_START] == True
                
                # Cluster destinations should be valid (4 clusters)
                for cluster_idx in range(4):
                    assert vehicle_mask[TransportActions.DEST_START + 1 + cluster_idx] == True
            break
        
        actions = [TransportActions.PASS] * env.num_agents
        obs, rewards, done, info = env.step(actions)


def test_cluster_goal():
    """Test TransportClusterGoal class."""
    from empo.transport import create_transport_env, TransportClusterGoal
    
    env = create_transport_env(
        num_humans=2, 
        num_vehicles=1, 
        num_nodes=12, 
        num_clusters=4,
        seed=42
    )
    env.reset(seed=42)
    
    # Create a cluster goal
    goal = TransportClusterGoal(env, agent_idx=0, target_cluster=2)
    
    assert goal.agent_idx == 0
    assert goal.target_cluster == 2
    
    # Test hash and equality
    goal2 = TransportClusterGoal(env, agent_idx=0, target_cluster=2)
    goal3 = TransportClusterGoal(env, agent_idx=0, target_cluster=3)
    
    assert hash(goal) == hash(goal2)
    assert goal == goal2
    assert goal != goal3


def test_cluster_goal_generator():
    """Test TransportClusterGoalGenerator class."""
    from empo.transport import (
        create_transport_env, 
        TransportClusterGoalGenerator, 
        TransportClusterGoal
    )
    
    env = create_transport_env(
        num_humans=2, 
        num_vehicles=1, 
        num_nodes=12, 
        num_clusters=4,
        seed=42
    )
    env.reset(seed=42)
    
    generator = TransportClusterGoalGenerator(env)
    goals = list(generator.generate(state=None, human_agent_index=0))
    
    # Should generate one goal per cluster
    assert len(goals) == 4
    
    # Each goal should be a TransportClusterGoal with weight 1.0
    for goal, weight in goals:
        assert isinstance(goal, TransportClusterGoal)
        assert goal.agent_idx == 0
        assert weight == 1.0


def test_cluster_goal_sampler():
    """Test TransportClusterGoalSampler class."""
    from empo.transport import (
        create_transport_env, 
        TransportClusterGoalSampler, 
        TransportClusterGoal
    )
    
    env = create_transport_env(
        num_humans=2, 
        num_vehicles=1, 
        num_nodes=12, 
        num_clusters=4,
        seed=42
    )
    env.reset(seed=42)
    
    sampler = TransportClusterGoalSampler(env, seed=42)
    
    # Sample multiple goals
    for _ in range(10):
        goal, weight = sampler.sample(state=None, human_agent_index=0)
        assert isinstance(goal, TransportClusterGoal)
        assert goal.agent_idx == 0
        assert weight == 1.0
        # Target cluster should be valid
        assert 0 <= goal.target_cluster < 4


def test_wrapper_cluster_vs_node_routing():
    """Test that cluster and node routing work differently."""
    from empo.transport import create_transport_env
    
    # Node-based routing
    env_nodes = create_transport_env(
        num_humans=2, 
        num_vehicles=1, 
        num_nodes=12, 
        seed=42
    )
    env_nodes.reset(seed=42)
    assert env_nodes.use_clusters == False
    
    # Cluster-based routing
    env_clusters = create_transport_env(
        num_humans=2, 
        num_vehicles=1, 
        num_nodes=12, 
        num_clusters=4,
        seed=42
    )
    env_clusters.reset(seed=42)
    assert env_clusters.use_clusters == True
    
    # Both should work but have different num_clusters
    obs_nodes = env_nodes.reset(seed=42)
    obs_clusters = env_clusters.reset(seed=42)
    
    assert obs_nodes[0]['num_clusters'] == 0
    assert obs_clusters[0]['num_clusters'] == 4


# =============================================================================
# Neural Network Encoder Tests
# =============================================================================

def test_import_transport_encoders():
    """Test that transport neural network encoders can be imported."""
    from empo.nn_based.transport import (
        TransportStateEncoder,
        TransportGoalEncoder,
        TransportQNetwork,
        observation_to_graph_data,
        STEP_TYPE_TO_IDX,
        NODE_FEATURE_DIM,
        EDGE_FEATURE_DIM,
    )
    
    assert callable(TransportStateEncoder)
    assert callable(TransportGoalEncoder)
    assert callable(TransportQNetwork)
    assert callable(observation_to_graph_data)
    assert len(STEP_TYPE_TO_IDX) == 4
    assert NODE_FEATURE_DIM > 0
    assert EDGE_FEATURE_DIM > 0


def test_transport_state_encoder_creation():
    """Test TransportStateEncoder can be created."""
    from empo.nn_based.transport import TransportStateEncoder
    
    encoder = TransportStateEncoder(
        num_clusters=10,
        max_nodes=100,
        feature_dim=128,
        hidden_dim=64,
        num_gnn_layers=2,
    )
    
    assert encoder.num_clusters == 10
    assert encoder.max_nodes == 100
    assert encoder.feature_dim == 128
    assert encoder.hidden_dim == 64


def test_transport_goal_encoder_creation():
    """Test TransportGoalEncoder can be created."""
    from empo.nn_based.transport import TransportGoalEncoder
    
    encoder = TransportGoalEncoder(
        max_nodes=100,
        num_clusters=10,
        feature_dim=32,
    )
    
    assert encoder.max_nodes == 100
    assert encoder.num_clusters == 10
    assert encoder.feature_dim == 32


def test_observation_to_graph_data():
    """Test observation_to_graph_data extracts features correctly."""
    from empo.transport import create_transport_env
    from empo.nn_based.transport import (
        observation_to_graph_data,
        NODE_FEATURE_DIM,
        EDGE_FEATURE_DIM,
        GLOBAL_FEATURE_DIM,
        AGENT_FEATURE_DIM,
    )
    
    env = create_transport_env(
        num_humans=2, 
        num_vehicles=1, 
        num_nodes=10, 
        num_clusters=3,
        seed=42
    )
    env.reset(seed=42)
    
    graph_data = observation_to_graph_data(env, query_agent_idx=0)
    
    # Check shapes
    assert graph_data['node_features'].shape == (10, NODE_FEATURE_DIM)
    assert graph_data['edge_index'].shape[0] == 2  # source/target
    assert graph_data['edge_features'].shape[1] == EDGE_FEATURE_DIM
    assert graph_data['global_features'].shape == (GLOBAL_FEATURE_DIM,)
    assert graph_data['agent_features'].shape == (AGENT_FEATURE_DIM,)
    assert graph_data['num_nodes'] == 10
    assert graph_data['num_edges'] > 0


def test_transport_q_network_forward():
    """Test TransportQNetwork forward pass."""
    import torch
    from empo.transport import create_transport_env
    from empo.nn_based.transport import (
        TransportStateEncoder,
        TransportGoalEncoder,
        TransportQNetwork,
        observation_to_graph_data,
        NUM_TRANSPORT_ACTIONS,
    )
    
    # Create encoders with small dimensions for testing
    state_encoder = TransportStateEncoder(
        num_clusters=3,
        max_nodes=20,
        feature_dim=32,
        hidden_dim=32,
        num_gnn_layers=1,
    )
    goal_encoder = TransportGoalEncoder(
        max_nodes=20,
        num_clusters=3,
        feature_dim=16,
    )
    q_network = TransportQNetwork(
        state_encoder,
        goal_encoder,
        num_actions=NUM_TRANSPORT_ACTIONS,
        hidden_dim=32,
    )
    
    # Create environment and get graph data
    env = create_transport_env(
        num_humans=2, 
        num_vehicles=1, 
        num_nodes=10, 
        num_clusters=3,
        seed=42
    )
    env.reset(seed=42)
    
    graph_data = observation_to_graph_data(env, query_agent_idx=0)
    
    # Create a simple goal tensor
    goal_tensor = goal_encoder.encode_goal(0, device='cpu', env=env)
    
    # Forward pass
    with torch.no_grad():
        q_values = q_network(graph_data, goal_tensor)
    
    assert q_values.shape == (1, NUM_TRANSPORT_ACTIONS)


def test_render_clusters():
    """Test render_clusters method."""
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    
    from empo.transport import create_transport_env
    
    env = create_transport_env(
        num_humans=2, 
        num_vehicles=1, 
        num_nodes=12, 
        num_clusters=4,
        seed=42
    )
    env.reset(seed=42)
    
    # Should not raise any errors
    ax = env.render_clusters()
    assert ax is not None


def test_render_clusters_without_clustering_raises():
    """Test that render_clusters raises error when clustering not enabled."""
    from empo.transport import create_transport_env
    import pytest
    
    env = create_transport_env(
        num_humans=2, 
        num_vehicles=1, 
        num_nodes=12,  # No clusters
        seed=42
    )
    env.reset(seed=42)
    
    with pytest.raises(ValueError):
        env.render_clusters()


def test_transport_policy_prior_network():
    """Test TransportPolicyPriorNetwork computes marginal correctly."""
    from empo.transport import create_transport_env, TransportGoal
    from empo.nn_based.transport import (
        TransportQNetwork,
        TransportPolicyPriorNetwork,
    )
    
    env = create_transport_env(
        num_humans=2,
        num_vehicles=1,
        num_nodes=10,
        num_clusters=3,
        seed=42
    )
    env.reset(seed=42)
    
    q_network = TransportQNetwork(
        max_nodes=20,
        num_clusters=3,
        num_actions=42,
        hidden_dim=32,
        state_feature_dim=32,
        goal_feature_dim=16,
        num_gnn_layers=1,
    )
    
    policy_network = TransportPolicyPriorNetwork(q_network)
    
    # Create some goals
    goals = [
        TransportGoal(env, agent_idx=0, target_node=i)
        for i in range(3)
    ]
    
    # Compute marginal
    marginal = policy_network.compute_marginal(
        state=None,
        world_model=env,
        query_agent_idx=0,
        goals=goals,
        device='cpu'
    )
    
    assert marginal.shape == (42,)
    assert abs(marginal.sum().item() - 1.0) < 1e-5


def test_transport_neural_policy_prior():
    """Test TransportNeuralHumanPolicyPrior basic functionality."""
    from empo.transport import create_transport_env, TransportGoal, TransportGoalSampler
    from empo.nn_based.transport import TransportNeuralHumanPolicyPrior
    
    env = create_transport_env(
        num_humans=2,
        num_vehicles=1,
        num_nodes=10,
        num_clusters=3,
        seed=42
    )
    env.reset(seed=42)
    
    goal_sampler = TransportGoalSampler(env, seed=42)
    
    # Create via convenience method
    prior = TransportNeuralHumanPolicyPrior.create(
        world_model=env,
        human_agent_indices=[0, 1],
        max_nodes=20,
        num_clusters=3,
        hidden_dim=32,
        state_feature_dim=32,
        goal_feature_dim=16,
        num_gnn_layers=1,
        goal_sampler=goal_sampler,
    )
    
    # Test goal-specific policy
    goal = TransportGoal(env, agent_idx=0, target_node=3)
    probs = prior(state=None, agent_idx=0, goal=goal)
    
    assert len(probs) == 42
    assert abs(sum(probs.values()) - 1.0) < 1e-5
    
    # Test marginal policy
    probs_marginal = prior(state=None, agent_idx=0, goal=None)
    
    assert len(probs_marginal) == 42
    assert abs(sum(probs_marginal.values()) - 1.0) < 1e-5


def test_transport_neural_policy_prior_save_load():
    """Test TransportNeuralHumanPolicyPrior save and load."""
    import tempfile
    import os
    from empo.transport import create_transport_env, TransportGoal, TransportGoalSampler
    from empo.nn_based.transport import TransportNeuralHumanPolicyPrior
    
    env = create_transport_env(
        num_humans=2,
        num_vehicles=1,
        num_nodes=10,
        num_clusters=3,
        seed=42
    )
    env.reset(seed=42)
    
    goal_sampler = TransportGoalSampler(env, seed=42)
    
    prior = TransportNeuralHumanPolicyPrior.create(
        world_model=env,
        human_agent_indices=[0, 1],
        max_nodes=20,
        num_clusters=3,
        hidden_dim=32,
        state_feature_dim=32,
        goal_feature_dim=16,
        num_gnn_layers=1,
    )
    
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, 'test_model.pt')
        prior.save(filepath)
        
        # Load and verify
        prior_loaded = TransportNeuralHumanPolicyPrior.load(
            filepath=filepath,
            world_model=env,
            human_agent_indices=[0, 1],
            goal_sampler=goal_sampler,
            device='cpu'
        )
        
        # Check loaded model works
        goal = TransportGoal(env, agent_idx=0, target_node=3)
        probs = prior_loaded(state=None, agent_idx=0, goal=goal)
        
        assert len(probs) == 42
        assert abs(sum(probs.values()) - 1.0) < 1e-5


def test_train_transport_neural_policy_prior():
    """Test train_transport_neural_policy_prior function (minimal training)."""
    from empo.transport import create_transport_env, TransportGoalSampler, TransportGoal
    from empo.nn_based.transport import train_transport_neural_policy_prior
    
    env = create_transport_env(
        num_humans=2,
        num_vehicles=1,
        num_nodes=8,
        num_clusters=0,  # Node-based routing for simplicity
        seed=42
    )
    env.reset(seed=42)
    
    goal_sampler = TransportGoalSampler(env, seed=42)
    
    # Train for just a few episodes (minimal test)
    prior = train_transport_neural_policy_prior(
        env=env,
        human_agent_indices=[0, 1],
        goal_sampler=goal_sampler,
        num_episodes=3,
        steps_per_episode=5,
        batch_size=2,
        hidden_dim=32,
        state_feature_dim=32,
        goal_feature_dim=16,
        num_gnn_layers=1,
        verbose=False,
        reward_shaping=False,  # Disable for faster test
    )
    
    # Check that the prior works
    goal = TransportGoal(env, agent_idx=0, target_node=2)
    probs = prior(state=None, agent_idx=0, goal=goal)
    
    assert len(probs) == 42
    assert abs(sum(probs.values()) - 1.0) < 1e-5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

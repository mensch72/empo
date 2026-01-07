import pytest
import numpy as np
import networkx as nx
from ai_transport import env, parallel_env, raw_env


def test_import():
    """Test that the module can be imported"""
    assert env is not None
    assert parallel_env is not None
    assert raw_env is not None


def test_parallel_env_creation():
    """Test creating a parallel environment with default parameters"""
    test_env = parallel_env()
    assert test_env is not None
    assert len(test_env.possible_agents) == 3  # 2 humans + 1 vehicle by default
    assert test_env.num_humans == 2
    assert test_env.num_vehicles == 1


def test_parallel_env_custom_agents():
    """Test creating environment with custom number of agents"""
    test_env = parallel_env(num_humans=3, num_vehicles=2)
    assert len(test_env.possible_agents) == 5
    assert test_env.num_humans == 3
    assert test_env.num_vehicles == 2


def test_agent_attributes():
    """Test that agents have correct attributes"""
    test_env = parallel_env(num_humans=2, num_vehicles=1)
    
    # Check human attributes
    for i in range(2):
        agent = f"human_{i}"
        assert agent in test_env.agent_attributes
        assert 'speed' in test_env.agent_attributes[agent]
        assert test_env.agent_attributes[agent]['speed'] == 1.0
    
    # Check vehicle attributes
    agent = "vehicle_0"
    assert agent in test_env.agent_attributes
    assert 'speed' in test_env.agent_attributes[agent]
    assert 'capacity' in test_env.agent_attributes[agent]
    assert 'fuel_use' in test_env.agent_attributes[agent]
    assert test_env.agent_attributes[agent]['speed'] == 2.0
    assert test_env.agent_attributes[agent]['capacity'] == 4
    assert test_env.agent_attributes[agent]['fuel_use'] == 1.0


def test_custom_agent_attributes():
    """Test creating environment with custom agent attributes"""
    test_env = parallel_env(
        num_humans=1,
        num_vehicles=1,
        human_speeds=[2.5],
        vehicle_speeds=[5.0],
        vehicle_capacities=[6],
        vehicle_fuel_uses=[2.0]
    )
    
    assert test_env.agent_attributes["human_0"]['speed'] == 2.5
    assert test_env.agent_attributes["vehicle_0"]['speed'] == 5.0
    assert test_env.agent_attributes["vehicle_0"]['capacity'] == 6
    assert test_env.agent_attributes["vehicle_0"]['fuel_use'] == 2.0


def test_default_network():
    """Test that default network is created correctly"""
    test_env = parallel_env()
    assert test_env.network is not None
    assert isinstance(test_env.network, nx.DiGraph)
    assert len(test_env.network.nodes()) > 0
    assert len(test_env.network.edges()) > 0
    
    # Check node attributes
    for node in test_env.network.nodes():
        assert 'name' in test_env.network.nodes[node]
    
    # Check edge attributes
    for u, v in test_env.network.edges():
        edge_data = test_env.network[u][v]
        assert 'length' in edge_data
        assert 'speed' in edge_data
        assert 'capacity' in edge_data


def test_custom_network():
    """Test creating environment with custom network"""
    G = nx.DiGraph()
    G.add_node(0, name="Start")
    G.add_node(1, name="End")
    G.add_edge(0, 1, length=20.0, speed=10.0, capacity=5)
    
    test_env = parallel_env(network=G)
    assert test_env.network == G


def test_network_validation():
    """Test that network validation works"""
    # Network without node name attribute should raise error
    G = nx.DiGraph()
    G.add_node(0)
    with pytest.raises(ValueError, match="missing required 'name' attribute"):
        parallel_env(network=G)
    
    # Network without edge attributes should raise error
    G = nx.DiGraph()
    G.add_node(0, name="A")
    G.add_node(1, name="B")
    G.add_edge(0, 1)
    with pytest.raises(ValueError, match="missing required"):
        parallel_env(network=G)


def test_reset():
    """Test environment reset"""
    test_env = parallel_env()
    observations, infos = test_env.reset()
    
    # Check that all agents are present
    assert len(observations) == len(test_env.possible_agents)
    assert len(infos) == len(test_env.possible_agents)
    
    # Check state components initialized
    assert test_env.real_time == 0.0
    assert test_env.agent_positions is not None
    assert test_env.vehicle_destinations is not None
    
    # Check all agents have positions
    for agent in test_env.agents:
        assert agent in test_env.agent_positions
    
    # Check vehicles have destination (should be None initially)
    for agent in test_env.vehicle_agents:
        assert agent in test_env.vehicle_destinations
        assert test_env.vehicle_destinations[agent] is None


def test_position_types():
    """Test that positions can be either nodes or (edge, coordinate) tuples"""
    test_env = parallel_env()
    test_env.reset()
    
    # Check positions are either nodes or (edge, coordinate) tuples
    for agent in test_env.agents:
        pos = test_env.agent_positions[agent]
        if isinstance(pos, tuple):
            # Should be (edge, coordinate) format
            assert len(pos) == 2
            edge, coord = pos
            assert edge in test_env.network.edges()
            edge_length = test_env.network[edge[0]][edge[1]]['length']
            assert 0 <= coord <= edge_length
        else:
            # Should be a node
            assert pos in test_env.network.nodes()
    
    # Test setting position as (edge, coordinate)
    edges = list(test_env.network.edges())
    if edges:
        edge = edges[0]
        edge_length = test_env.network[edge[0]][edge[1]]['length']
        test_env.agent_positions[test_env.agents[0]] = (edge, edge_length / 2)
        pos = test_env.agent_positions[test_env.agents[0]]
        assert isinstance(pos, tuple)
        assert len(pos) == 2
        assert pos[0] == edge


def test_step_basic():
    """Test basic step functionality"""
    test_env = parallel_env()
    test_env.reset()
    
    # Create dummy actions
    actions = {agent: 0 for agent in test_env.agents}
    
    observations, rewards, terminations, truncations, infos = test_env.step(actions)
    
    # Check return values structure
    assert len(observations) == len(test_env.agents)
    assert len(rewards) == len(test_env.agents)
    assert len(terminations) == len(test_env.agents)
    assert len(truncations) == len(test_env.agents)
    assert len(infos) == len(test_env.agents)


def test_step_empty_actions():
    """Test step with empty actions"""
    test_env = parallel_env()
    test_env.reset()
    
    observations, rewards, terminations, truncations, infos = test_env.step({})
    
    # Should return empty dicts and clear agents
    assert observations == {}
    assert rewards == {}
    assert terminations == {}
    assert truncations == {}
    assert infos == {}
    assert len(test_env.agents) == 0


def test_wrapped_env():
    """Test that wrapped env function works"""
    test_env = env()
    assert test_env is not None


def test_raw_env():
    """Test that raw_env (AEC) function works"""
    test_env = raw_env()
    assert test_env is not None


def test_human_aboard_state():
    """Test that human_aboard state is initialized correctly"""
    test_env = parallel_env(num_humans=2, num_vehicles=1)
    test_env.reset()
    
    # Check that all humans have aboard status (can be None or a vehicle)
    for i in range(2):
        agent = f"human_{i}"
        assert agent in test_env.human_aboard
        # Can be None or 'vehicle_0'
        assert test_env.human_aboard[agent] in [None, 'vehicle_0']


def test_step_type_state():
    """Test that step_type state is initialized correctly"""
    test_env = parallel_env()
    test_env.reset()
    
    assert test_env.step_type is not None
    assert test_env.step_type in ['routing', 'unboarding', 'boarding', 'departing']


def test_action_space_routing():
    """Test action spaces during routing step type"""
    test_env = parallel_env(num_humans=2, num_vehicles=1)
    test_env.reset()
    
    # Manually place vehicle at node so it can route
    test_env.agent_positions['vehicle_0'] = 0
    test_env.step_type = 'routing'
    
    # Vehicles at nodes should have actions for each node + None
    num_nodes = len(test_env.network.nodes())
    vehicle_space = test_env.action_space('vehicle_0')
    assert vehicle_space.n == num_nodes + 1
    
    # Humans should only have pass action
    human_space = test_env.action_space('human_0')
    assert human_space.n == 1


def test_action_space_unboarding():
    """Test action spaces during unboarding step type"""
    test_env = parallel_env(num_humans=2, num_vehicles=1)
    test_env.reset()
    test_env.step_type = 'unboarding'
    
    # Human not aboard should only have pass
    test_env.human_aboard['human_0'] = None
    space = test_env.action_space('human_0')
    assert space.n == 1
    
    # Human aboard vehicle at node should have pass and unboard
    test_env.human_aboard['human_1'] = 'vehicle_0'
    test_env.agent_positions['vehicle_0'] = 0  # at node
    space = test_env.action_space('human_1')
    assert space.n == 2
    
    # Vehicles should only have pass
    vehicle_space = test_env.action_space('vehicle_0')
    assert vehicle_space.n == 1


def test_action_space_boarding():
    """Test action spaces during boarding step type"""
    test_env = parallel_env(num_humans=2, num_vehicles=2)
    test_env.reset()
    test_env.step_type = 'boarding'
    
    # Place all agents at node 0
    test_env.agent_positions['human_0'] = 0
    test_env.agent_positions['human_1'] = 0
    test_env.agent_positions['vehicle_0'] = 0
    test_env.agent_positions['vehicle_1'] = 0
    test_env.human_aboard['human_0'] = None
    test_env.human_aboard['human_1'] = None
    
    # Humans at node with 2 vehicles should have pass + 2 boarding options
    space = test_env.action_space('human_0')
    assert space.n == 3  # pass + 2 vehicles
    
    # Vehicles should only have pass
    vehicle_space = test_env.action_space('vehicle_0')
    assert vehicle_space.n == 1


def test_action_space_departing():
    """Test action spaces during departing step type"""
    test_env = parallel_env(num_humans=2, num_vehicles=1)
    test_env.reset()
    test_env.step_type = 'departing'
    
    # Place agents at node 0
    test_env.agent_positions['human_0'] = 0
    test_env.agent_positions['human_1'] = 0
    test_env.agent_positions['vehicle_0'] = 0
    test_env.human_aboard['human_0'] = None  # Not aboard
    test_env.human_aboard['human_1'] = 'vehicle_0'  # Aboard
    
    # Count outgoing edges from node 0
    outgoing_edges = list(test_env.network.out_edges(0))
    
    # Vehicle at node should have pass + outgoing edges
    vehicle_space = test_env.action_space('vehicle_0')
    assert vehicle_space.n == len(outgoing_edges) + 1
    
    # Human at node not aboard should have pass + outgoing edges
    human_space = test_env.action_space('human_0')
    assert human_space.n == len(outgoing_edges) + 1
    
    # Human aboard should only have pass
    human_aboard_space = test_env.action_space('human_1')
    assert human_aboard_space.n == 1


def test_action_space_agents_on_edges():
    """Test that agents on edges can only pass in all step types"""
    test_env = parallel_env(num_humans=1, num_vehicles=1)
    test_env.reset()
    
    # Place agents on an edge
    edges = list(test_env.network.edges())
    if edges:
        edge = edges[0]
        test_env.agent_positions['human_0'] = (edge, 5.0)
        test_env.agent_positions['vehicle_0'] = (edge, 3.0)
        test_env.human_aboard['human_0'] = None
        
        for step_type in ['routing', 'unboarding', 'boarding', 'departing']:
            test_env.step_type = step_type
            
            # Both should only have pass action when on edge
            human_space = test_env.action_space('human_0')
            assert human_space.n == 1
            
            vehicle_space = test_env.action_space('vehicle_0')
            assert vehicle_space.n == 1



def test_routing_step_logic():
    """Test routing step changes destinations but not time"""
    test_env = parallel_env(num_humans=1, num_vehicles=2)
    test_env.reset()
    
    # Manually place all agents at nodes so routing actions work
    test_env.agent_positions = {'human_0': 0, 'vehicle_0': 0, 'vehicle_1': 1}
    test_env.step_type = 'routing'
    
    initial_time = test_env.real_time
    nodes = list(test_env.network.nodes())
    
    # Action 0 = set destination to None
    # Action 1, 2, 3 = set destination to node 0, 1, 2
    actions = {
        'human_0': 0,  # Should do nothing (human can only pass)
        'vehicle_0': 0,  # Set destination to None
        'vehicle_1': 2  # Set destination to node 1
    }
    
    test_env.step(actions)
    
    # Check time didn't advance
    assert test_env.real_time == initial_time
    
    # Check destinations changed
    assert test_env.vehicle_destinations['vehicle_0'] is None
    assert test_env.vehicle_destinations['vehicle_1'] == nodes[1]


def test_unboarding_step_logic():
    """Test unboarding step changes aboard status but not time"""
    test_env = parallel_env(num_humans=2, num_vehicles=1)
    test_env.reset()
    
    # Manually place all agents at nodes
    test_env.agent_positions = {'human_0': 0, 'human_1': 0, 'vehicle_0': 0}
    test_env.step_type = 'unboarding'
    
    # Put humans aboard vehicle
    test_env.human_aboard['human_0'] = 'vehicle_0'
    test_env.human_aboard['human_1'] = 'vehicle_0'
    
    initial_time = test_env.real_time
    
    # Action 0 = pass, Action 1 = unboard
    actions = {
        'human_0': 1,  # Unboard
        'human_1': 0,  # Pass (stay aboard)
        'vehicle_0': 0  # Can only pass
    }
    
    test_env.step(actions)
    
    # Check time didn't advance
    assert test_env.real_time == initial_time
    
    # Check human_0 unboarded, human_1 still aboard
    assert test_env.human_aboard['human_0'] is None
    assert test_env.human_aboard['human_1'] == 'vehicle_0'


def test_boarding_step_logic():
    """Test boarding step with capacity constraints"""
    test_env = parallel_env(num_humans=3, num_vehicles=1)
    test_env.reset(seed=42)
    test_env.step_type = 'boarding'
    
    # Set vehicle capacity to 2
    test_env.agent_attributes['vehicle_0']['capacity'] = 2
    
    # All at node 0
    test_env.agent_positions['human_0'] = 0
    test_env.agent_positions['human_1'] = 0
    test_env.agent_positions['human_2'] = 0
    test_env.agent_positions['vehicle_0'] = 0
    test_env.human_aboard['human_0'] = None
    test_env.human_aboard['human_1'] = None
    test_env.human_aboard['human_2'] = None
    
    initial_time = test_env.real_time
    
    # All humans try to board (action 1 = board vehicle_0)
    actions = {
        'human_0': 1,
        'human_1': 1,
        'human_2': 1,
        'vehicle_0': 0
    }
    
    test_env.step(actions)
    
    # Check time didn't advance
    assert test_env.real_time == initial_time
    
    # Check that exactly 2 humans boarded (capacity = 2)
    humans_aboard = sum(1 for h in test_env.human_agents 
                       if test_env.human_aboard[h] == 'vehicle_0')
    assert humans_aboard == 2


def test_departing_step_logic_basic():
    """Test departing step moves agents onto edges"""
    test_env = parallel_env(num_humans=1, num_vehicles=1)
    test_env.reset()
    test_env.step_type = 'departing'
    
    # Both at node 0
    test_env.agent_positions['human_0'] = 0
    test_env.agent_positions['vehicle_0'] = 0
    test_env.human_aboard['human_0'] = None
    
    initial_time = test_env.real_time
    
    # Get first outgoing edge from node 0
    outgoing = list(test_env.network.out_edges(0))
    assert len(outgoing) > 0
    
    # Action 1 = depart into first outgoing edge
    actions = {
        'human_0': 1,
        'vehicle_0': 1
    }
    
    test_env.step(actions)
    
    # After departing, agents are placed on edge at coord 0, then time advances
    # and they move. One or both may reach the end.
    human_pos = test_env.agent_positions['human_0']
    vehicle_pos = test_env.agent_positions['vehicle_0']
    
    # At least one should have moved (time should have advanced)
    assert test_env.real_time > initial_time
    
    # Human might be on edge or at destination node
    # Vehicle might be on edge or at destination node
    # At least one should have departed (not both at node 0)
    assert not (human_pos == 0 and vehicle_pos == 0)


def test_departing_step_time_advance():
    """Test that departing step advances time correctly"""
    test_env = parallel_env(num_humans=1, num_vehicles=1)
    test_env.reset()
    test_env.step_type = 'departing'
    
    # Place agents on an edge
    edges = list(test_env.network.edges())
    edge = edges[0]
    edge_data = test_env.network[edge[0]][edge[1]]
    edge_length = edge_data['length']
    edge_speed = edge_data['speed']
    human_speed = test_env.agent_attributes['human_0']['speed']
    
    # Place human at coordinate 5, vehicle at coordinate 3
    test_env.agent_positions['human_0'] = (edge, 5.0)
    test_env.agent_positions['vehicle_0'] = (edge, 3.0)
    test_env.human_aboard['human_0'] = None
    
    initial_time = test_env.real_time
    
    # Both pass (don't depart, just move along edge)
    actions = {
        'human_0': 0,
        'vehicle_0': 0
    }
    
    test_env.step(actions)
    
    # Calculate expected time advance
    human_remaining = (edge_length - 5.0) / human_speed
    vehicle_remaining = (edge_length - 3.0) / edge_speed
    expected_delta_t = min(human_remaining, vehicle_remaining)
    
    # Check time advanced correctly
    assert abs(test_env.real_time - (initial_time + expected_delta_t)) < 1e-9


def test_departing_step_arrival_at_node():
    """Test that agents arrive at nodes when reaching edge end"""
    test_env = parallel_env(num_humans=1, num_vehicles=1)
    test_env.reset()
    test_env.step_type = 'departing'
    
    # Place agents very close to end of edge
    edges = list(test_env.network.edges())
    edge = edges[0]
    edge_data = test_env.network[edge[0]][edge[1]]
    edge_length = edge_data['length']
    
    # Place both agents same distance from end so they arrive together
    # Use a distance that works for both speeds
    edge_data['speed']
    test_env.agent_attributes['human_0']['speed']
    
    # Place them close enough that both will arrive
    # For edge_speed=5.0, human_speed=1.0, if we place at edge_length-0.05,
    # vehicle time = 0.05/5.0 = 0.01, human time = 0.05/1.0 = 0.05
    # min is 0.01, so vehicle arrives but human doesn't
    # Let's place human separately closer to end
    test_env.agent_positions['human_0'] = (edge, edge_length - 0.01)  # Very close
    test_env.agent_positions['vehicle_0'] = (edge, edge_length - 0.01)  # Very close
    test_env.human_aboard['human_0'] = None
    
    actions = {
        'human_0': 0,
        'vehicle_0': 0
    }
    
    test_env.step(actions)
    
    # Both should now be at the target node (or at least vehicle should be)
    target_node = edge[1]
    # With coord=9.99, human needs 0.01/1.0=0.01, vehicle needs 0.01/5.0=0.002
    # min is 0.002, so vehicle arrives, human goes to 9.99+1.0*0.002=9.992
    # Let's just check vehicle arrives
    assert test_env.agent_positions['vehicle_0'] == target_node


def test_humans_aboard_move_with_vehicle():
    """Test that humans aboard vehicles move with the vehicle"""
    test_env = parallel_env(num_humans=1, num_vehicles=1)
    test_env.reset()
    test_env.step_type = 'departing'
    
    # Place human aboard vehicle at node
    test_env.agent_positions['human_0'] = 0
    test_env.agent_positions['vehicle_0'] = 0
    test_env.human_aboard['human_0'] = 'vehicle_0'
    
    list(test_env.network.out_edges(0))
    
    # Human tries to walk (but is aboard so shouldn't work)
    # Vehicle departs
    actions = {
        'human_0': 1,  # Try to walk (should be ignored)
        'vehicle_0': 1  # Depart
    }
    
    test_env.step(actions)
    
    # Human should be at same position as vehicle (moved with it)
    assert test_env.agent_positions['human_0'] == test_env.agent_positions['vehicle_0']
    
    # Vehicle should have moved (not at node 0 anymore)
    assert test_env.agent_positions['vehicle_0'] != 0
    
    # Human still aboard
    assert test_env.human_aboard['human_0'] == 'vehicle_0'


def test_observation_scenario_full():
    """Test full observation scenario"""
    test_env = parallel_env(num_humans=2, num_vehicles=1, observation_scenario='full')
    test_env.reset()
    
    obs = test_env._generate_observation_for_agent('human_0')
    
    # Full observation should have all state components
    assert 'real_time' in obs
    assert 'step_type' in obs
    assert 'agent_positions' in obs
    assert 'vehicle_destinations' in obs
    assert 'human_aboard' in obs
    assert 'agent_attributes' in obs
    assert 'network_nodes' in obs
    assert 'network_edges' in obs
    
    # Check all agents are in positions
    assert len(obs['agent_positions']) == 3
    assert 'human_0' in obs['agent_positions']
    assert 'human_1' in obs['agent_positions']
    assert 'vehicle_0' in obs['agent_positions']


def test_observation_scenario_local():
    """Test local observation scenario"""
    test_env = parallel_env(num_humans=2, num_vehicles=1, observation_scenario='local')
    test_env.reset()
    
    # Manually place all agents at same node
    test_env.agent_positions = {'human_0': 0, 'human_1': 0, 'vehicle_0': 0}
    
    # All agents at same node, so should see each other
    obs = test_env._generate_observation_for_agent('human_0')
    
    assert 'real_time' in obs
    assert 'step_type' in obs
    assert 'my_position' in obs
    assert 'agents_here' in obs
    
    # Should see all 3 agents (including self) at same node
    assert len(obs['agents_here']) == 3
    assert 'human_0' in obs['agents_here']
    assert 'human_1' in obs['agents_here']
    assert 'vehicle_0' in obs['agents_here']
    
    # Check agent info includes attributes
    assert 'attributes' in obs['agents_here']['human_0']
    assert 'attributes' in obs['agents_here']['vehicle_0']


def test_observation_scenario_local_separate_locations():
    """Test local observation when agents are at different locations"""
    test_env = parallel_env(num_humans=2, num_vehicles=1, observation_scenario='local')
    test_env.reset()
    
    # Move agents to different locations
    test_env.agent_positions['human_0'] = 0
    test_env.agent_positions['human_1'] = 1
    test_env.agent_positions['vehicle_0'] = 2
    
    obs = test_env._generate_observation_for_agent('human_0')
    
    # Should only see itself at node 0
    assert len(obs['agents_here']) == 1
    assert 'human_0' in obs['agents_here']
    assert 'human_1' not in obs['agents_here']
    assert 'vehicle_0' not in obs['agents_here']


def test_observation_scenario_statistical():
    """Test statistical observation scenario"""
    test_env = parallel_env(num_humans=2, num_vehicles=1, observation_scenario='statistical')
    test_env.reset()
    
    # Manually place all agents at node 0
    test_env.agent_positions = {'human_0': 0, 'human_1': 0, 'vehicle_0': 0}
    
    obs = test_env._generate_observation_for_agent('human_0')
    
    # Should have local observation components
    assert 'real_time' in obs
    assert 'step_type' in obs
    assert 'my_position' in obs
    assert 'agents_here' in obs
    
    # Plus statistical information
    assert 'node_counts' in obs
    assert 'edge_counts' in obs
    
    # Check counts at node 0 (all agents there)
    assert 0 in obs['node_counts']
    assert obs['node_counts'][0]['humans'] == 2
    assert obs['node_counts'][0]['vehicles'] == 1


def test_observation_scenario_statistical_on_edges():
    """Test statistical counts for agents on edges"""
    test_env = parallel_env(num_humans=1, num_vehicles=1, observation_scenario='statistical')
    test_env.reset()
    
    # Place agents on edge
    edges = list(test_env.network.edges())
    edge = edges[0]
    test_env.agent_positions['human_0'] = (edge, 5.0)
    test_env.agent_positions['vehicle_0'] = (edge, 3.0)
    
    obs = test_env._generate_observation_for_agent('human_0')
    
    # Check edge counts
    assert edge in obs['edge_counts']
    assert obs['edge_counts'][edge]['humans'] == 1
    assert obs['edge_counts'][edge]['vehicles'] == 1


def test_rewards_always_zero():
    """Test that rewards are always zero"""
    test_env = parallel_env(num_humans=2, num_vehicles=1)
    test_env.reset()
    test_env.step_type = 'routing'
    
    actions = {agent: 0 for agent in test_env.agents}
    obs, rewards, terms, truncs, infos = test_env.step(actions)
    
    # All rewards should be zero
    for agent in test_env.agents:
        assert rewards[agent] == 0.0


def test_observation_scenario_invalid():
    """Test that invalid observation scenario raises error"""
    with pytest.raises(ValueError, match="observation_scenario must be"):
        parallel_env(observation_scenario='invalid')


def test_observations_in_step():
    """Test that observations are generated correctly in step"""
    test_env = parallel_env(num_humans=1, num_vehicles=1, observation_scenario='full')
    test_env.reset()
    test_env.step_type = 'routing'
    
    actions = {agent: 0 for agent in test_env.agents}
    obs, rewards, terms, truncs, infos = test_env.step(actions)
    
    # Check observations are dicts
    for agent in test_env.agents:
        assert isinstance(obs[agent], dict)
        assert 'real_time' in obs[agent]
        assert 'step_type' in obs[agent]


def test_create_random_2d_network():
    """Test creating a random 2D network"""
    test_env = parallel_env(num_humans=1, num_vehicles=1)
    
    network = test_env.create_random_2d_network(num_nodes=5, seed=42)
    
    # Check network has nodes
    assert len(network.nodes()) == 5
    
    # Check all nodes have required attributes
    for node in network.nodes():
        assert 'name' in network.nodes[node]
        assert 'x' in network.nodes[node]
        assert 'y' in network.nodes[node]
    
    # Check network has edges
    assert len(network.edges()) > 0
    
    # Check all edges have required attributes
    for u, v in network.edges():
        edge_data = network[u][v]
        assert 'length' in edge_data
        assert 'speed' in edge_data
        assert 'capacity' in edge_data
        assert edge_data['length'] > 0
        assert edge_data['speed'] > 0
        assert edge_data['capacity'] > 0


def test_create_random_2d_network_bidirectional():
    """Test bidirectional probability in random network"""
    test_env = parallel_env(num_humans=1, num_vehicles=1)
    
    # With high bidirectional probability
    network_high = test_env.create_random_2d_network(
        num_nodes=5, bidirectional_prob=0.9, seed=42
    )
    
    # With low bidirectional probability
    network_low = test_env.create_random_2d_network(
        num_nodes=5, bidirectional_prob=0.1, seed=42
    )
    
    # Count bidirectional edges (where both (u,v) and (v,u) exist)
    def count_bidirectional(G):
        count = 0
        for u, v in G.edges():
            if G.has_edge(v, u):
                count += 1
        return count // 2  # Divide by 2 since we count each pair twice
    
    bidir_high = count_bidirectional(network_high)
    bidir_low = count_bidirectional(network_low)
    
    # High prob should have more bidirectional edges than low prob
    # (This is probabilistic but with seed should be deterministic)
    assert bidir_high >= bidir_low


def test_create_random_2d_network_length_calculation():
    """Test that edge lengths are computed from coordinates"""
    test_env = parallel_env(num_humans=1, num_vehicles=1)
    
    network = test_env.create_random_2d_network(num_nodes=5, seed=42)
    
    # Check that lengths match Euclidean distance
    for u, v in network.edges():
        x1, y1 = network.nodes[u]['x'], network.nodes[u]['y']
        x2, y2 = network.nodes[v]['x'], network.nodes[v]['y']
        expected_length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        actual_length = network[u][v]['length']
        assert abs(actual_length - expected_length) < 1e-6


def test_initialize_random_positions():
    """Test initializing random agent positions"""
    test_env = parallel_env(num_humans=2, num_vehicles=1)
    test_env.reset()
    
    # Initialize random positions
    test_env.initialize_random_positions(seed=42)
    
    # Check all agents have positions
    for agent in test_env.agents:
        assert agent in test_env.agent_positions
        pos = test_env.agent_positions[agent]
        
        # Position should be either a node or (edge, coordinate) tuple
        if isinstance(pos, tuple):
            assert len(pos) == 2
            edge, coord = pos
            assert edge in test_env.network.edges()
            edge_length = test_env.network[edge[0]][edge[1]]['length']
            assert 0 <= coord <= edge_length
        else:
            assert pos in test_env.network.nodes()


def test_initialize_random_positions_distribution():
    """Test that random positions distribute between nodes and edges"""
    test_env = parallel_env(num_humans=10, num_vehicles=10)
    test_env.reset()
    
    # Initialize random positions
    test_env.initialize_random_positions(seed=42)
    
    # Count agents at nodes vs on edges
    at_nodes = sum(1 for pos in test_env.agent_positions.values() if not isinstance(pos, tuple))
    on_edges = sum(1 for pos in test_env.agent_positions.values() if isinstance(pos, tuple))
    
    # Both should be non-zero (with high probability given 20 agents)
    assert at_nodes > 0
    assert on_edges > 0


def test_random_network_integration():
    """Test using random network in environment"""
    test_env = parallel_env(num_humans=2, num_vehicles=1)
    
    # Create random network
    network = test_env.create_random_2d_network(num_nodes=6, seed=42)
    
    # Create new env with this network
    test_env2 = parallel_env(num_humans=2, num_vehicles=1, network=network)
    test_env2.reset(seed=42)
    
    # Initialize random positions
    test_env2.initialize_random_positions(seed=42)
    
    # Check environment works with random positions
    test_env2.step_type = 'routing'
    actions = {agent: 0 for agent in test_env2.agents}
    obs, rewards, terms, truncs, infos = test_env2.step(actions)
    
    # Should complete without errors
    assert len(obs) == len(test_env2.agents)


def test_graphical_rendering():
    """Test graphical rendering"""
    test_env = parallel_env(num_humans=2, num_vehicles=1, render_mode="human")
    test_env.reset()
    
    # Enable graphical rendering
    test_env.enable_rendering('graphical')
    
    # Should not raise an error
    fig = test_env.render()
    assert fig is not None
    
    test_env.close()


def test_save_frame():
    """Test saving a frame"""
    import tempfile
    import os
    
    test_env = parallel_env(num_humans=1, num_vehicles=1, render_mode="human")
    test_env.reset()
    test_env.enable_rendering('graphical')
    test_env.render()
    
    # Save to temporary file
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, 'test_frame.png')
        test_env.save_frame(filepath)
        
        # Check file was created
        assert os.path.exists(filepath)
        assert os.path.getsize(filepath) > 0
    
    test_env.close()


def test_video_recording():
    """Test video recording functionality"""
    test_env = parallel_env(num_humans=1, num_vehicles=1, render_mode="human")
    test_env.reset()
    
    # Start recording
    test_env.start_video_recording()
    
    # Render a few frames
    for _ in range(3):
        test_env.render()
        test_env.step_type = 'routing'
        actions = {agent: 0 for agent in test_env.agents}
        test_env.step(actions)
    
    # Check frames were recorded
    assert len(test_env.frames) > 0
    
    test_env.close()


def test_text_rendering():
    """Test text rendering still works"""
    test_env = parallel_env(num_humans=1, num_vehicles=1, render_mode="human")
    test_env.reset()
    
    # Text rendering should not raise error
    test_env._render_text()
    
    test_env.close()


def test_render_with_random_network():
    """Test rendering with random 2D network"""
    test_env = parallel_env(num_humans=2, num_vehicles=1, render_mode="human")
    network = test_env.create_random_2d_network(num_nodes=5, seed=42)
    
    test_env = parallel_env(num_humans=2, num_vehicles=1, network=network, render_mode="human")
    test_env.reset()
    test_env.initialize_random_positions(seed=42)
    
    test_env.enable_rendering('graphical')
    fig = test_env.render()
    
    assert fig is not None
    
    test_env.close()

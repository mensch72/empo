#!/usr/bin/env python3
"""
Test script for the neural network-based policy prior.

Tests the modular architecture with:
- Base classes in nn_based/
- Multigrid-specific implementations in nn_based/multigrid/
"""

import os

import numpy as np
import torch
import tempfile

from empo.learning_based import (
    ReplayBuffer,
    Trainer,
)
from empo.learning_based.multigrid import (
    MultiGridStateEncoder,
    MultiGridGoalEncoder,
    MultiGridQNetwork,
    MultiGridNeuralHumanPolicyPrior,
    AGENT_FEATURE_SIZE,
)
from empo.possible_goal import PossibleGoal, PossibleGoalSampler
from typing import Tuple


class SimpleReachGoal(PossibleGoal):
    """A simple goal to reach a target position."""
    
    def __init__(self, world_model, target_pos):
        super().__init__(world_model)
        self.target_pos = target_pos
    
    def is_achieved(self, state) -> int:
        step_count, agent_states, mobile_objects, mutable_objects = state
        if len(agent_states) > 0:
            x, y = agent_states[0][0], agent_states[0][1]
            if x == self.target_pos[0] and y == self.target_pos[1]:
                return 1
        return 0
    
    def __hash__(self):
        return hash(self.target_pos)
    
    def __eq__(self, other):
        return isinstance(other, SimpleReachGoal) and self.target_pos == other.target_pos


class SimpleGoalSampler(PossibleGoalSampler):
    """A simple goal sampler that samples random target positions."""
    
    def __init__(self, world_model):
        super().__init__(world_model)
    
    def sample(self, state, human_agent_index: int) -> Tuple[PossibleGoal, float]:
        x = np.random.randint(0, self.world_model.width)
        y = np.random.randint(0, self.world_model.height)
        return SimpleReachGoal(self.world_model, (x, y)), 1.0


class MockWorldModel:
    """Mock world model for testing."""
    
    def __init__(self, width=10, height=10, num_agents=2):
        self.width = width
        self.height = height
        self.max_steps = 100
        self.grid = MockGrid(width, height)
        self.agents = [MockAgent(f'agent_{i}', 'grey') for i in range(num_agents)]
        self.stumble_prob = 0.0
        self.magic_entry_prob = 1.0
        self.magic_solidify_prob = 0.0


class MockGrid:
    """Mock grid for testing."""
    
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self._cells = {}
    
    def get(self, x, y):
        return self._cells.get((x, y), None)


class MockAgent:
    """Mock agent for testing."""
    
    def __init__(self, name, color='grey'):
        self.name = name
        self.color = color
        self.can_enter_magic_walls = False
        self.can_push_rocks = False
        self.carrying = None
        self.paused = False
        self.terminated = False
        self.forced_next_action = None


def create_mock_state(num_agents=2):
    """Create a mock state tuple."""
    step_count = 0
    agent_states = [(i, i, 0, False, False) for i in range(num_agents)]  # x, y, dir, term, started
    mobile_objects = []
    mutable_objects = []
    return (step_count, agent_states, mobile_objects, mutable_objects)


def test_multigrid_state_encoder():
    """Test the unified MultiGridStateEncoder."""
    print("Testing MultiGridStateEncoder...")
    
    num_agents_per_color = {'grey': 2}
    
    encoder = MultiGridStateEncoder(
        grid_height=10,
        grid_width=10,
        num_agents_per_color=num_agents_per_color,
        num_agent_colors=7,
        feature_dim=256
    )
    
    # Create dummy inputs
    batch_size = 4
    grid_tensor = torch.randn(batch_size, encoder.num_grid_channels, 10, 10)
    global_features = torch.randn(batch_size, 4)
    
    # Calculate agent feature size (agent-agnostic: no query agent, just per-color lists)
    total_agents = sum(num_agents_per_color.values())
    agent_feature_size = AGENT_FEATURE_SIZE * total_agents  # per-color only
    agent_features = torch.randn(batch_size, agent_feature_size)
    
    interactive_features = torch.randn(batch_size, encoder._interactive_input_size)
    
    # Forward pass
    features = encoder(grid_tensor, global_features, agent_features, interactive_features)
    
    assert features.shape == (batch_size, 256), f"Expected (4, 256), got {features.shape}"
    print(f"  ✓ Output shape: {features.shape}")
    print(f"  ✓ Num grid channels: {encoder.num_grid_channels}")
    print("  ✓ MultiGridStateEncoder test passed!")


def test_multigrid_state_encoder_tensorize_state():
    """Test state encoding from actual state tuple."""
    print("Testing MultiGridStateEncoder.tensorize_state...")
    
    world_model = MockWorldModel()
    state = create_mock_state(num_agents=2)
    
    encoder = MultiGridStateEncoder(
        grid_height=10,
        grid_width=10,
        num_agents_per_color={'grey': 2},
        num_agent_colors=7,
        feature_dim=256
    )
    
    grid_tensor, global_features, agent_features, interactive_features = \
        encoder.tensorize_state(state, world_model)
    
    assert grid_tensor.shape[0] == 1
    assert global_features.shape == (1, 4)
    print(f"  ✓ Grid tensor shape: {grid_tensor.shape}")
    print(f"  ✓ Global features shape: {global_features.shape}")
    print(f"  ✓ Agent features shape: {agent_features.shape}")
    print(f"  ✓ Interactive features shape: {interactive_features.shape}")
    print("  ✓ MultiGridStateEncoder.tensorize_state test passed!")


def test_multigrid_goal_encoder():
    """Test the MultiGridGoalEncoder."""
    print("Testing MultiGridGoalEncoder...")
    
    # All goals are rectangles (x1, y1, x2, y2). Point goals are (x, y, x, y).
    encoder = MultiGridGoalEncoder(
        grid_height=10,
        grid_width=10,
        feature_dim=32
    )
    
    # Create dummy input with 4 coordinates (bounding box: x1, y1, x2, y2)
    batch_size = 4
    goal_coords = torch.randn(batch_size, 4)
    
    # Forward pass
    features = encoder(goal_coords)
    
    assert features.shape == (batch_size, 32), f"Expected (4, 32), got {features.shape}"
    print(f"  ✓ Output shape: {features.shape}")
    
    # Test encode_goal with rectangle
    class RectGoal:
        def __init__(self):
            self.target_rect = (2, 3, 5, 7)
    
    rect_goal = RectGoal()
    encoded = encoder.tensorize_goal(rect_goal)
    assert encoded.shape == (1, 4), f"Expected (1, 4), got {encoded.shape}"
    # Verify bounding box encoding (x1, y1, x2, y2)
    assert encoded[0, 0] == 2 and encoded[0, 1] == 3  # x1, y1
    assert encoded[0, 2] == 5 and encoded[0, 3] == 7  # x2, y2
    print(f"  ✓ Rectangle goal encoding: {encoded.tolist()}")
    
    # Test encode_goal with point goal (encoded as (x, y, x, y))
    class PointGoal:
        def __init__(self):
            self.target_pos = (5, 5)
    
    point_goal = PointGoal()
    encoded = encoder.tensorize_goal(point_goal)
    assert encoded.shape == (1, 4), f"Expected (1, 4), got {encoded.shape}"
    # For point goals, bounding box is (x, y, x, y)
    assert encoded[0, 0] == 5 and encoded[0, 1] == 5  # x1, y1
    assert encoded[0, 2] == 5 and encoded[0, 3] == 5  # x2, y2
    print(f"  ✓ Point goal encoding as bbox: {encoded.tolist()}")
    
    # Test compute_goal_weight for rectangle (area)
    weight = MultiGridGoalEncoder.compute_goal_weight(rect_goal)
    expected_area = (1 + 5 - 2) * (1 + 7 - 3)  # (1+x2-x1)*(1+y2-y1) = 4 * 5 = 20
    assert weight == expected_area, f"Expected {expected_area}, got {weight}"
    print(f"  ✓ Rectangle goal weight (area): {weight}")
    
    # Test compute_goal_weight for point goal (area = 1)
    weight = MultiGridGoalEncoder.compute_goal_weight(point_goal)
    assert weight == 1.0, f"Expected 1.0, got {weight}"
    print(f"  ✓ Point goal weight (area): {weight}")
    
    print("  ✓ MultiGridGoalEncoder test passed!")


def test_multigrid_q_network():
    """Test the MultiGridQNetwork."""
    print("Testing MultiGridQNetwork...")
    
    q_network = MultiGridQNetwork(
        grid_height=10,
        grid_width=10,
        num_actions=4,
        num_agents_per_color={'grey': 2},
        num_agent_colors=7,
        state_feature_dim=256,
        goal_feature_dim=32,
        hidden_dim=256,
        beta=1.0
    )
    
    # Test with mock state
    world_model = MockWorldModel()
    state = create_mock_state()
    goal = SimpleReachGoal(world_model, (5, 5))
    
    q_values = q_network.forward(state, world_model, 0, goal)
    
    assert q_values.shape == (1, 4), f"Expected (1, 4), got {q_values.shape}"
    print(f"  ✓ Q-values shape: {q_values.shape}")
    
    # Test policy computation
    policy = q_network.get_policy(q_values)
    assert policy.shape == (1, 4)
    assert torch.allclose(policy.sum(), torch.tensor(1.0), atol=1e-5)
    print(f"  ✓ Policy shape: {policy.shape}")
    print(f"  ✓ Policy sums to 1: {policy.sum().item():.4f}")
    print("  ✓ MultiGridQNetwork test passed!")


def test_replay_buffer():
    """Test the ReplayBuffer."""
    print("Testing ReplayBuffer...")
    
    buffer = ReplayBuffer(capacity=100)
    
    world_model = MockWorldModel()
    state = create_mock_state()
    goal = SimpleReachGoal(world_model, (5, 5))
    
    # Add transitions
    for i in range(50):
        buffer.push(state, i % 4, state, 0, goal)
    
    assert len(buffer) == 50
    
    # Sample batch
    batch = buffer.sample(16)
    assert len(batch) == 16
    
    print(f"  ✓ Buffer length: {len(buffer)}")
    print(f"  ✓ Batch size: {len(batch)}")
    print("  ✓ ReplayBuffer test passed!")


def test_trainer():
    """Test the Trainer class."""
    print("Testing Trainer...")
    
    q_network = MultiGridQNetwork(
        grid_height=10,
        grid_width=10,
        num_actions=4,
        num_agents_per_color={'grey': 2},
        num_agent_colors=7,
    )
    
    target_network = MultiGridQNetwork(
        grid_height=10,
        grid_width=10,
        num_actions=4,
        num_agents_per_color={'grey': 2},
        num_agent_colors=7,
    )
    target_network.load_state_dict(q_network.state_dict())
    
    optimizer = torch.optim.Adam(q_network.parameters())
    buffer = ReplayBuffer(capacity=100)
    
    trainer = Trainer(
        q_network=q_network,
        target_network=target_network,
        optimizer=optimizer,
        replay_buffer=buffer,
        gamma=0.99,
        target_update_freq=10,
        device='cpu'
    )
    
    world_model = MockWorldModel()
    state = create_mock_state()
    goal = SimpleReachGoal(world_model, (5, 5))
    
    # Store some transitions
    for i in range(32):
        trainer.store_transition(state, i % 4, state, 0, goal)
    
    # Train step
    loss = trainer.train_step(batch_size=16)
    assert loss is not None
    
    print(f"  ✓ Training loss: {loss:.4f}")
    print(f"  ✓ Total steps: {trainer.total_steps}")
    print("  ✓ Trainer test passed!")


def test_save_load():
    """Test save and load functionality."""
    print("Testing save/load...")
    
    world_model = MockWorldModel()
    goal_sampler = SimpleGoalSampler(world_model)
    
    q_network = MultiGridQNetwork(
        grid_height=10,
        grid_width=10,
        num_actions=4,
        num_agents_per_color={'grey': 2},
        num_agent_colors=7,
    )
    
    prior = MultiGridNeuralHumanPolicyPrior(
        q_network=q_network,
        world_model=world_model,
        human_agent_indices=[0],
        goal_sampler=goal_sampler,
        action_encoding={0: 'still', 1: 'forward', 2: 'left', 3: 'right'},
        device='cpu'
    )
    
    # Save
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        filepath = f.name
    
    prior.save(filepath)
    print(f"  ✓ Saved to {filepath}")
    
    # Load
    loaded = MultiGridNeuralHumanPolicyPrior.load(
        filepath,
        world_model=world_model,
        human_agent_indices=[0],
        goal_sampler=goal_sampler,
        device='cpu'
    )
    
    # Test that loaded model produces same outputs
    state = create_mock_state()
    goal = SimpleReachGoal(world_model, (5, 5))
    
    q_network.eval()
    loaded.q_network.eval()
    
    with torch.no_grad():
        original_q = q_network.forward(state, world_model, 0, goal)
        loaded_q = loaded.q_network.forward(state, world_model, 0, goal)
    
    assert torch.allclose(original_q, loaded_q, atol=1e-5)
    print("  ✓ Loaded model produces same Q-values")
    
    # Cleanup
    os.unlink(filepath)
    print("  ✓ save/load test passed!")


def test_load_dimension_mismatch():
    """Test that load fails when trying to load from smaller grid to larger grid."""
    print("Testing load with dimension mismatch...")
    
    world_model = MockWorldModel(width=10, height=10)
    
    q_network = MultiGridQNetwork(
        grid_height=10,
        grid_width=10,
        num_actions=4,
        num_agents_per_color={'grey': 2},
        num_agent_colors=7,
    )
    
    prior = MultiGridNeuralHumanPolicyPrior(
        q_network=q_network,
        world_model=world_model,
        human_agent_indices=[0],
        device='cpu'
    )
    
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        filepath = f.name
    
    prior.save(filepath)
    
    # Try to load on larger grid - should fail
    different_world = MockWorldModel(width=15, height=15)
    
    try:
        MultiGridNeuralHumanPolicyPrior.load(
            filepath,
            world_model=different_world,
            human_agent_indices=[0],
            device='cpu'
        )
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Cannot load policy trained on smaller grid" in str(e)
        print(f"  ✓ Correctly raised: {e}")
    
    os.unlink(filepath)
    print("  ✓ load dimension mismatch test passed!")


def test_load_action_conflict():
    """Test that load fails with conflicting action encodings."""
    print("Testing load with action conflict...")
    
    world_model = MockWorldModel()
    
    q_network = MultiGridQNetwork(
        grid_height=10,
        grid_width=10,
        num_actions=4,
        num_agents_per_color={'grey': 2},
        num_agent_colors=7,
    )
    
    # Save with one action encoding
    prior = MultiGridNeuralHumanPolicyPrior(
        q_network=q_network,
        world_model=world_model,
        human_agent_indices=[0],
        action_encoding={0: 'still', 1: 'forward', 2: 'left', 3: 'right'},
        device='cpu'
    )
    
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        filepath = f.name
    
    prior.save(filepath)
    
    # Create world model with conflicting actions
    class MockAction:
        def __init__(self, name):
            self.name = name
    
    world_model_conflict = MockWorldModel()
    world_model_conflict.actions = [
        MockAction('BACKWARD'),  # Conflicts with 'still' at index 0
        MockAction('FORWARD'),
        MockAction('LEFT'),
        MockAction('RIGHT'),
    ]
    
    try:
        MultiGridNeuralHumanPolicyPrior.load(
            filepath,
            world_model=world_model_conflict,
            human_agent_indices=[0],
            device='cpu'
        )
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Action encoding conflict" in str(e)
        print(f"  ✓ Correctly raised: {e}")
    
    os.unlink(filepath)
    print("  ✓ load action conflict test passed!")


def test_soft_clamp():
    """Test the SoftClamp module."""
    from empo.learning_based.util.soft_clamp import SoftClamp

    print("Testing SoftClamp...")    # Test with default range [0.5, 1.5]
    soft_clamp = SoftClamp(a=0.5, b=1.5)
    
    # Values in range should be unchanged
    x_in_range = torch.tensor([0.5, 1.0, 1.5])
    y = soft_clamp(x_in_range)
    assert torch.allclose(y, x_in_range, atol=1e-5), f"In-range values should be unchanged"
    print("  ✓ In-range values are linear")
    
    # Values outside range should be soft-clamped
    x_above = torch.tensor([2.0, 3.0, 10.0])
    y_above = soft_clamp(x_above)
    
    # Should be less than actual values but greater than upper bound
    R = 1.5 - 0.5  # = 1.0
    for i, (orig, clamped) in enumerate(zip(x_above.tolist(), y_above.tolist())):
        assert clamped < orig, f"Clamped should be < original for above-range"
        assert clamped < 1.5 + R, f"Clamped should approach b + R asymptotically"
    print("  ✓ Above-range values are soft-clamped")
    
    x_below = torch.tensor([0.0, -1.0, -10.0])
    y_below = soft_clamp(x_below)
    
    for i, (orig, clamped) in enumerate(zip(x_below.tolist(), y_below.tolist())):
        assert clamped > orig, f"Clamped should be > original for below-range"
        assert clamped > 0.5 - R, f"Clamped should approach a - R asymptotically"
    print("  ✓ Below-range values are soft-clamped")
    
    # Test gradient flow
    x = torch.tensor([0.0, 1.0, 2.0], requires_grad=True)
    y = soft_clamp(x)
    loss = y.sum()
    loss.backward()
    assert x.grad is not None and x.grad.abs().sum() > 0, "Gradients should flow"
    print("  ✓ Gradients flow correctly")
    
    # Test with feasible_range in Q-network
    q_network = MultiGridQNetwork(
        grid_height=10,
        grid_width=10,
        num_actions=4,
        num_agents_per_color={'grey': 2},
        num_agent_colors=7,
        feasible_range=(-1.0, 2.0)
    )
    assert q_network.soft_clamp is not None
    assert q_network.feasible_range == (-1.0, 2.0)
    print("  ✓ Q-network correctly initializes SoftClamp with feasible_range")
    
    print("  ✓ SoftClamp test passed!")


def test_weight_proportional_sampling():
    """Test that weight-proportional rectangle sampling works correctly."""
    print("Testing weight-proportional sampling...")
    
    # Test sample_coordinate_pair_weighted
    n = 5  # Coordinates 0-4
    counts = {}
    num_samples = 100000
    rng = np.random.default_rng(42)
    
    for _ in range(num_samples):
        c1, c2 = MultiGridGoalEncoder.sample_coordinate_pair_weighted(n, rng)
        key = (c1, c2)
        counts[key] = counts.get(key, 0) + 1
    
    # Verify that weights are proportional to (1 + c2 - c1)
    # Expected weight for (c1, c2) is (1 + c2 - c1)
    total_weight = 0
    for c1 in range(n):
        for c2 in range(c1, n):
            total_weight += (1 + c2 - c1)
    
    for c1 in range(n):
        for c2 in range(c1, n):
            key = (c1, c2)
            expected_weight = (1 + c2 - c1)
            expected_prob = expected_weight / total_weight
            observed_prob = counts.get(key, 0) / num_samples
            
            # Allow 20% relative error due to sampling variance
            relative_error = abs(observed_prob - expected_prob) / expected_prob
            assert relative_error < 0.2, f"Pair {key}: expected prob {expected_prob:.4f}, got {observed_prob:.4f}"
    
    print("  ✓ sample_coordinate_pair_weighted produces correct distribution")
    
    # Test sample_rectangle_weighted
    x_range = (1, 4)  # x in [1, 4]
    y_range = (2, 5)  # y in [2, 5]
    
    rect_counts = {}
    for _ in range(num_samples):
        rect = MultiGridGoalEncoder.sample_rectangle_weighted(x_range, y_range, rng)
        rect_counts[rect] = rect_counts.get(rect, 0) + 1
    
    # Verify a few samples have correct relative frequencies
    # Weight = (1+x2-x1)*(1+y2-y1)
    # Point goal (2, 3, 2, 3): weight = 1*1 = 1
    # Small rect (2, 3, 3, 4): weight = 2*2 = 4
    # The ratio should be approximately 1:4
    
    point_count = rect_counts.get((2, 3, 2, 3), 0)
    small_rect_count = rect_counts.get((2, 3, 3, 4), 0)
    
    if point_count > 0 and small_rect_count > 0:
        ratio = small_rect_count / point_count
        # Expected ratio is 4.0, allow some variance
        assert 2.0 < ratio < 8.0, f"Ratio should be ~4.0, got {ratio:.2f}"
        print(f"  ✓ sample_rectangle_weighted: ratio of (2,3,3,4) to (2,3,2,3) is {ratio:.2f} (expected ~4.0)")
    
    # Verify edges
    for _ in range(1000):
        x1, y1, x2, y2 = MultiGridGoalEncoder.sample_rectangle_weighted(x_range, y_range, rng)
        assert x_range[0] <= x1 <= x2 <= x_range[1], f"x coords out of range: {x1}, {x2}"
        assert y_range[0] <= y1 <= y2 <= y_range[1], f"y coords out of range: {y1}, {y2}"
    
    print("  ✓ sample_rectangle_weighted respects coordinate ranges")
    
    print("  ✓ weight-proportional sampling test passed!")


def test_goal_rendering():
    """Test the goal rendering methods."""
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend for tests
    import matplotlib.pyplot as plt
    
    print("Testing goal rendering...")
    
    # Test get_goal_bounding_box
    # 1. Rectangle goal with target_rect
    class RectGoal:
        def __init__(self, rect):
            self.target_rect = rect
    
    goal = RectGoal((2, 3, 5, 6))
    bb = MultiGridGoalEncoder.get_goal_bounding_box(goal)
    assert bb == (2, 3, 5, 6), f"Expected (2, 3, 5, 6), got {bb}"
    print("  ✓ get_goal_bounding_box works with target_rect")
    
    # 2. Point goal with target_pos
    class PointGoal:
        def __init__(self, pos):
            self.target_pos = pos
    
    goal = PointGoal((4, 5))
    bb = MultiGridGoalEncoder.get_goal_bounding_box(goal)
    assert bb == (4, 5, 4, 5), f"Expected (4, 5, 4, 5), got {bb}"
    print("  ✓ get_goal_bounding_box works with target_pos")
    
    # 3. Tuple goal
    bb = MultiGridGoalEncoder.get_goal_bounding_box((1, 2, 3, 4))
    assert bb == (1, 2, 3, 4), f"Expected (1, 2, 3, 4), got {bb}"
    print("  ✓ get_goal_bounding_box works with tuple")
    
    # 4. Reversed coordinates should be normalized
    bb = MultiGridGoalEncoder.get_goal_bounding_box((5, 6, 2, 3))
    assert bb == (2, 3, 5, 6), f"Expected (2, 3, 5, 6), got {bb}"
    print("  ✓ get_goal_bounding_box normalizes reversed coordinates")
    
    # Test closest_point_on_rectangle
    rect = (2, 2, 4, 4)
    tile_size = 32
    inset = 0.08
    
    # Point outside rectangle (agent at 1, 1)
    agent_px = 1 * tile_size + tile_size / 2
    agent_py = 1 * tile_size + tile_size / 2
    closest = MultiGridGoalEncoder.closest_point_on_rectangle(rect, agent_px, agent_py, tile_size, inset)
    
    # Closest point should be top-left corner of rectangle (with inset)
    expected_x = 2 * tile_size + tile_size * inset
    expected_y = 2 * tile_size + tile_size * inset
    assert abs(closest[0] - expected_x) < 0.01 and abs(closest[1] - expected_y) < 0.01, \
        f"Expected ({expected_x}, {expected_y}), got {closest}"
    print("  ✓ closest_point_on_rectangle works for outside point")
    
    # Point inside rectangle (agent at 3, 3)
    agent_px = 3 * tile_size + tile_size / 2
    agent_py = 3 * tile_size + tile_size / 2
    closest = MultiGridGoalEncoder.closest_point_on_rectangle(rect, agent_px, agent_py, tile_size, inset)
    
    # Should return a point on the rectangle boundary
    left = 2 * tile_size + tile_size * inset
    right = (4 + 1) * tile_size - tile_size * inset
    top = 2 * tile_size + tile_size * inset
    bottom = (4 + 1) * tile_size - tile_size * inset
    
    # Point should be on one of the edges
    on_edge = (abs(closest[0] - left) < 0.01 or abs(closest[0] - right) < 0.01 or
               abs(closest[1] - top) < 0.01 or abs(closest[1] - bottom) < 0.01)
    assert on_edge, f"Point inside rectangle should return edge point, got {closest}"
    print("  ✓ closest_point_on_rectangle works for inside point")
    
    # Test render_goal_overlay
    fig, ax = plt.subplots()
    ax.set_xlim(0, 200)
    ax.set_ylim(200, 0)
    
    goal = (2, 2, 4, 4)
    agent_pos = (1, 1)
    
    MultiGridGoalEncoder.render_goal_overlay(
        ax=ax,
        goal=goal,
        agent_pos=agent_pos,
        agent_idx=0,
        tile_size=32,
        goal_color=(0.0, 0.4, 1.0, 0.7),
        line_width=2.5
    )
    
    # Check that patches and lines were added
    assert len(ax.patches) == 1, f"Expected 1 patch (rectangle), got {len(ax.patches)}"
    assert len(ax.lines) == 1, f"Expected 1 line (connection), got {len(ax.lines)}"
    print("  ✓ render_goal_overlay adds rectangle and line")
    
    plt.close(fig)
    
    print("  ✓ goal rendering test passed!")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("Running Neural Policy Prior Tests")
    print("=" * 60)
    
    test_multigrid_state_encoder()
    print()
    
    test_multigrid_state_encoder_tensorize_state()
    print()
    
    test_multigrid_goal_encoder()
    print()
    
    test_multigrid_q_network()
    print()
    
    test_replay_buffer()
    print()
    
    test_trainer()
    print()
    
    test_save_load()
    print()
    
    test_load_dimension_mismatch()
    print()
    
    test_load_action_conflict()
    print()
    
    test_soft_clamp()
    print()
    
    test_weight_proportional_sampling()
    print()
    
    test_goal_rendering()
    print()
    
    print("=" * 60)
    print("All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()

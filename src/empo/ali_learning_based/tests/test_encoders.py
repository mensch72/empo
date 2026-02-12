"""
Tests for state and goal encoders.

Run with:
    PYTHONPATH=src:vendor/multigrid pytest src/ali_learning_based/tests/test_encoders.py -v
"""

import pytest
import torch
import numpy as np

from empo.ali_learning_based.encoders import (
    StateEncoder,
    GoalEncoder,
    NUM_GRID_CHANNELS,
    AGENT_FEATURE_SIZE,
    CH_WALL,
    CH_DOOR,
    CH_KEY,
    CH_GOAL,
    CH_ROBOT,
    CH_HUMAN,
)
from gym_multigrid.multigrid import MultiGridEnv, World, Actions


# -----------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------

# A small 7×5 grid with:
#   Red agent (robot)  at (1, 2)
#   Yellow agent (human) at (5, 3)
#   Red key at (1, 1)
#   Red locked door at (3, 1)
#   Goal cell at (5, 1)
KEY_DOOR_MAP = """
We We We We We We We
We Kr .. Lr .. Gr We
We Ar .. .. .. .. We
We .. .. .. .. Ay We
We We We We We We We
"""


@pytest.fixture
def key_door_env():
    """Create a small key-door environment and return it after reset."""
    env = MultiGridEnv(
        map=KEY_DOOR_MAP,
        max_steps=50,
        partial_obs=False,
        objects_set=World,
        actions_set=Actions,
    )
    env.reset()
    return env


# A minimal 5×5 empty grid with two agents, no keys or doors.
EMPTY_MAP = """
We We We We We
We Ar .. .. We
We .. .. .. We
We .. .. Ay We
We We We We We
"""


@pytest.fixture
def empty_env():
    env = MultiGridEnv(
        map=EMPTY_MAP,
        max_steps=20,
        partial_obs=False,
        objects_set=World,
        actions_set=Actions,
    )
    env.reset()
    return env


# -----------------------------------------------------------------------
# StateEncoder — initialisation
# -----------------------------------------------------------------------

class TestStateEncoderInit:

    def test_dimensions(self, key_door_env):
        enc = StateEncoder(key_door_env)

        # Grid: 8 channels × 5 rows × 7 cols = 280
        # Agent features: 2 agents × 5 = 10
        # Global features: 1 (time_remaining)
        expected = NUM_GRID_CHANNELS * 5 * 7 + AGENT_FEATURE_SIZE * 2 + 1
        assert enc.dim == expected

    def test_robot_and_human_indices(self, key_door_env):
        enc = StateEncoder(key_door_env)
        assert enc.robot_agent_index == 0
        assert enc.human_agent_indices == [1]

    def test_static_walls_cached(self, key_door_env):
        enc = StateEncoder(key_door_env)
        # Border cells should be walls
        assert enc._wall_grid[0, 0] == 1.0  # top-left corner
        assert enc._wall_grid[4, 6] == 1.0  # bottom-right corner
        # Interior cells should not be walls
        assert enc._wall_grid[2, 2] == 0.0

    def test_static_goal_cached(self, key_door_env):
        enc = StateEncoder(key_door_env)
        # Goal at (5, 1) in the map
        assert enc._goal_grid[1, 5] == 1.0
        # Non-goal cell
        assert enc._goal_grid[2, 2] == 0.0

    def test_key_positions_cached(self, key_door_env):
        enc = StateEncoder(key_door_env)
        assert "red" in enc._key_positions
        assert enc._key_positions["red"] == (1, 1)

    def test_door_positions_cached(self, key_door_env):
        enc = StateEncoder(key_door_env)
        assert (3, 1) in enc._door_positions


# -----------------------------------------------------------------------
# StateEncoder — encoding
# -----------------------------------------------------------------------

class TestStateEncoderEncode:

    def test_output_shape(self, key_door_env):
        enc = StateEncoder(key_door_env)
        state = key_door_env.get_state()
        tensor = enc.encode(state)

        assert tensor.shape == (enc.dim,)
        assert tensor.dtype == torch.float32

    def test_deterministic(self, key_door_env):
        """Same state always produces the same tensor."""
        enc = StateEncoder(key_door_env)
        state = key_door_env.get_state()
        t1 = enc.encode(state)
        t2 = enc.encode(state)
        assert torch.equal(t1, t2)

    def test_different_states_differ(self, key_door_env):
        """After a step the encoding should change."""
        enc = StateEncoder(key_door_env)
        state_before = key_door_env.get_state()

        # Take a step (both agents move forward)
        key_door_env.step([Actions.forward, Actions.forward])
        state_after = key_door_env.get_state()

        t_before = enc.encode(state_before)
        t_after = enc.encode(state_after)
        assert not torch.equal(t_before, t_after)

    def test_wall_channel_matches_static(self, key_door_env):
        """Channel 0 (walls) should match the cached wall grid."""
        enc = StateEncoder(key_door_env)
        state = key_door_env.get_state()
        tensor = enc.encode(state)

        # Extract the wall channel from the flat tensor
        H, W = enc.height, enc.width
        grid_flat = tensor[: NUM_GRID_CHANNELS * H * W]
        grid = grid_flat.view(NUM_GRID_CHANNELS, H, W)

        assert torch.equal(grid[CH_WALL], enc._wall_grid)

    def test_door_channel_locked(self, key_door_env):
        """The door at (3,1) should start as locked (value 1.0)."""
        enc = StateEncoder(key_door_env)
        state = key_door_env.get_state()
        tensor = enc.encode(state)

        H, W = enc.height, enc.width
        grid = tensor[: NUM_GRID_CHANNELS * H * W].view(NUM_GRID_CHANNELS, H, W)

        assert grid[CH_DOOR, 1, 3] == 1.0  # locked

    def test_key_channel_present(self, key_door_env):
        """The key at (1,1) should be on the ground initially."""
        enc = StateEncoder(key_door_env)
        state = key_door_env.get_state()
        tensor = enc.encode(state)

        H, W = enc.height, enc.width
        grid = tensor[: NUM_GRID_CHANNELS * H * W].view(NUM_GRID_CHANNELS, H, W)

        assert grid[CH_KEY, 1, 1] == 1.0

    def test_robot_channel(self, key_door_env):
        """The robot (agent 0) should appear at its initial position."""
        enc = StateEncoder(key_door_env)
        state = key_door_env.get_state()
        step_count, agent_states, _, _ = state

        rx = int(agent_states[0][0])
        ry = int(agent_states[0][1])

        tensor = enc.encode(state)
        H, W = enc.height, enc.width
        grid = tensor[: NUM_GRID_CHANNELS * H * W].view(NUM_GRID_CHANNELS, H, W)

        assert grid[CH_ROBOT, ry, rx] == 1.0
        # Only one robot cell should be set
        assert grid[CH_ROBOT].sum() == 1.0

    def test_human_channel(self, key_door_env):
        """The human (agent 1) should appear at its initial position."""
        enc = StateEncoder(key_door_env)
        state = key_door_env.get_state()
        step_count, agent_states, _, _ = state

        hx = int(agent_states[1][0])
        hy = int(agent_states[1][1])

        tensor = enc.encode(state)
        H, W = enc.height, enc.width
        grid = tensor[: NUM_GRID_CHANNELS * H * W].view(NUM_GRID_CHANNELS, H, W)

        assert grid[CH_HUMAN, hy, hx] == 1.0
        assert grid[CH_HUMAN].sum() == 1.0

    def test_time_remaining(self, key_door_env):
        """Time remaining should be (max_steps - step) / max_steps."""
        enc = StateEncoder(key_door_env)
        state = key_door_env.get_state()
        tensor = enc.encode(state)

        # Last element is time_remaining
        step_count = state[0]
        expected_time = (enc.max_steps - step_count) / enc.max_steps
        assert abs(tensor[-1].item() - expected_time) < 1e-6

    def test_direction_onehot(self, key_door_env):
        """Agent direction features should be valid one-hot vectors."""
        enc = StateEncoder(key_door_env)
        state = key_door_env.get_state()
        tensor = enc.encode(state)

        H, W = enc.height, enc.width
        grid_size = NUM_GRID_CHANNELS * H * W
        agent_features = tensor[grid_size : grid_size + AGENT_FEATURE_SIZE * enc.num_agents]

        for i in range(enc.num_agents):
            start = i * AGENT_FEATURE_SIZE
            dir_onehot = agent_features[start : start + 4]
            # Exactly one direction should be 1.0, rest 0.0
            assert dir_onehot.sum().item() == 1.0
            assert (dir_onehot >= 0).all()
            assert (dir_onehot <= 1).all()

    def test_batch_encoding(self, key_door_env):
        """encode_batch should produce a (batch, dim) tensor."""
        enc = StateEncoder(key_door_env)
        s1 = key_door_env.get_state()
        key_door_env.step([Actions.forward, Actions.left])
        s2 = key_door_env.get_state()

        batch = enc.encode_batch([s1, s2])
        assert batch.shape == (2, enc.dim)

    def test_empty_env_no_crash(self, empty_env):
        """Encoder should work on an env with no keys/doors."""
        enc = StateEncoder(empty_env)
        state = empty_env.get_state()
        tensor = enc.encode(state)

        assert tensor.shape == (enc.dim,)
        # No keys or doors → those channels should be all zeros
        H, W = enc.height, enc.width
        grid = tensor[: NUM_GRID_CHANNELS * H * W].view(NUM_GRID_CHANNELS, H, W)
        assert grid[CH_DOOR].sum() == 0.0
        assert grid[CH_KEY].sum() == 0.0


# -----------------------------------------------------------------------
# GoalEncoder
# -----------------------------------------------------------------------

class TestGoalEncoder:

    def test_dim(self, key_door_env):
        enc = GoalEncoder(key_door_env)
        assert enc.dim == 4

    def test_point_goal(self, key_door_env):
        """A point goal at (3, 2) on a 7×5 grid → [3/7, 2/5, 3/7, 2/5]."""
        from empo.world_specific_helpers.multigrid import ReachCellGoal

        enc = GoalEncoder(key_door_env)
        goal = ReachCellGoal(key_door_env, human_agent_index=1, target_pos=(3, 2))
        tensor = enc.encode(goal)

        assert tensor.shape == (4,)
        expected = torch.tensor([3.0 / 7, 2.0 / 5, 3.0 / 7, 2.0 / 5])
        assert torch.allclose(tensor, expected)

    def test_rectangle_goal(self, key_door_env):
        """A rectangle goal (1,1)-(3,3) on a 7×5 grid."""
        from empo.world_specific_helpers.multigrid import ReachRectangleGoal

        enc = GoalEncoder(key_door_env)
        goal = ReachRectangleGoal(
            key_door_env, human_agent_index=1, target_rect=(1, 1, 3, 3)
        )
        tensor = enc.encode(goal)

        expected = torch.tensor([1.0 / 7, 1.0 / 5, 3.0 / 7, 3.0 / 5])
        assert torch.allclose(tensor, expected)

    def test_normalized_range(self, key_door_env):
        """All coordinates should be in [0, 1]."""
        from empo.world_specific_helpers.multigrid import ReachCellGoal

        enc = GoalEncoder(key_door_env)
        # Test corner positions
        for x in [0, key_door_env.width - 1]:
            for y in [0, key_door_env.height - 1]:
                goal = ReachCellGoal(key_door_env, 1, (x, y))
                tensor = enc.encode(goal)
                assert (tensor >= 0).all()
                assert (tensor <= 1).all()

    def test_batch_encoding(self, key_door_env):
        from empo.world_specific_helpers.multigrid import ReachCellGoal

        enc = GoalEncoder(key_door_env)
        goals = [
            ReachCellGoal(key_door_env, 1, (1, 1)),
            ReachCellGoal(key_door_env, 1, (3, 2)),
            ReachCellGoal(key_door_env, 1, (5, 3)),
        ]
        batch = enc.encode_batch(goals)
        assert batch.shape == (3, 4)

    def test_different_goals_differ(self, key_door_env):
        from empo.world_specific_helpers.multigrid import ReachCellGoal

        enc = GoalEncoder(key_door_env)
        g1 = ReachCellGoal(key_door_env, 1, (1, 1))
        g2 = ReachCellGoal(key_door_env, 1, (5, 3))
        assert not torch.equal(enc.encode(g1), enc.encode(g2))

    def test_invalid_goal_raises(self, key_door_env):
        enc = GoalEncoder(key_door_env)
        with pytest.raises(ValueError):
            enc.encode("not a goal")


# -----------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------

if __name__ == "__main__":
    pytest.main([__file__, "-v"])

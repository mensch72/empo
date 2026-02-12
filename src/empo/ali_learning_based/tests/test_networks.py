"""
Tests for neural networks.

Run with:
    PYTHONPATH=src:vendor/multigrid pytest src/empo/ali_learning_based/tests/test_networks.py -v
"""

import pytest
import torch
import copy

from empo.ali_learning_based.encoders import (
    StateEncoder,
    GoalEncoder,
    NUM_GRID_CHANNELS,
)
from empo.ali_learning_based.networks import (
    GridFeatureExtractor,
    QhNet,
    VheNet,
    QrNet,
)
from gym_multigrid.multigrid import MultiGridEnv, World, Actions


# -----------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------

KEY_DOOR_MAP = """
We We We We We We We
We Kr .. Lr .. Gr We
We Ar .. .. .. .. We
We .. .. .. .. Ay We
We We We We We We We
"""

SMALL_MAP = """
We We We We We
We Ar .. .. We
We .. .. .. We
We .. .. Ay We
We We We We We
"""

NUM_ACTIONS = 6  # multigrid: still, forward, left, right, pickup, toggle


@pytest.fixture
def key_door_env():
    env = MultiGridEnv(
        map=KEY_DOOR_MAP, max_steps=50, partial_obs=False,
        objects_set=World, actions_set=Actions,
    )
    env.reset()
    return env


@pytest.fixture
def small_env():
    env = MultiGridEnv(
        map=SMALL_MAP, max_steps=20, partial_obs=False,
        objects_set=World, actions_set=Actions,
    )
    env.reset()
    return env


@pytest.fixture
def encoders(key_door_env):
    return StateEncoder(key_door_env), GoalEncoder(key_door_env)


@pytest.fixture
def state_and_goal(key_door_env, encoders):
    """Return a batch of 1 encoded state and 1 encoded goal."""
    from empo.world_specific_helpers.multigrid import ReachCellGoal

    s_enc, g_enc = encoders
    state = key_door_env.get_state()
    goal = ReachCellGoal(key_door_env, human_agent_index=1, target_pos=(5, 1))
    return s_enc.encode(state).unsqueeze(0), g_enc.encode(goal).unsqueeze(0)


# -----------------------------------------------------------------------
# GridFeatureExtractor
# -----------------------------------------------------------------------

class TestGridFeatureExtractor:

    def test_output_shape(self, encoders):
        s_enc, _ = encoders
        grid_size = NUM_GRID_CHANNELS * s_enc.height * s_enc.width
        extra_dim = s_enc.dim - grid_size

        extractor = GridFeatureExtractor(
            NUM_GRID_CHANNELS, s_enc.height, s_enc.width, extra_dim, feature_dim=128
        )
        dummy = torch.randn(4, s_enc.dim)
        out = extractor(dummy)
        assert out.shape == (4, 128)

    def test_deterministic(self, encoders):
        s_enc, _ = encoders
        grid_size = NUM_GRID_CHANNELS * s_enc.height * s_enc.width
        extra_dim = s_enc.dim - grid_size

        extractor = GridFeatureExtractor(
            NUM_GRID_CHANNELS, s_enc.height, s_enc.width, extra_dim
        )
        extractor.eval()
        x = torch.randn(2, s_enc.dim)
        with torch.no_grad():
            assert torch.equal(extractor(x), extractor(x))

    def test_different_grid_sizes(self, key_door_env, small_env):
        """Feature extractors for different grid sizes both output feature_dim."""
        for env in [key_door_env, small_env]:
            s_enc = StateEncoder(env)
            grid_size = NUM_GRID_CHANNELS * s_enc.height * s_enc.width
            extra_dim = s_enc.dim - grid_size
            extractor = GridFeatureExtractor(
                NUM_GRID_CHANNELS, s_enc.height, s_enc.width, extra_dim, feature_dim=64
            )
            state = env.get_state()
            x = s_enc.encode(state).unsqueeze(0)
            out = extractor(x)
            assert out.shape == (1, 64)


# -----------------------------------------------------------------------
# QhNet — Phase 1
# -----------------------------------------------------------------------

class TestQhNet:

    def test_output_shape(self, encoders, state_and_goal):
        s_enc, g_enc = encoders
        state, goal = state_and_goal

        net = QhNet.from_encoders(s_enc, g_enc, NUM_ACTIONS)
        q = net(state, goal)
        assert q.shape == (1, NUM_ACTIONS)

    def test_batch(self, encoders, key_door_env):
        """Works with batch_size > 1."""
        s_enc, g_enc = encoders
        net = QhNet.from_encoders(s_enc, g_enc, NUM_ACTIONS)

        batch = 8
        states = torch.randn(batch, s_enc.dim)
        goals = torch.randn(batch, g_enc.dim)
        q = net(states, goals)
        assert q.shape == (batch, NUM_ACTIONS)

    def test_gradients_flow(self, encoders, state_and_goal):
        """Loss.backward() should produce gradients on all parameters."""
        s_enc, g_enc = encoders
        state, goal = state_and_goal

        net = QhNet.from_encoders(s_enc, g_enc, NUM_ACTIONS)
        q = net(state, goal)
        loss = q.sum()
        loss.backward()

        for name, param in net.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert param.grad.abs().sum() > 0, f"Zero gradient for {name}"

    def test_from_encoders_matches_manual(self, encoders):
        """from_encoders produces the same architecture as manual construction."""
        s_enc, g_enc = encoders
        grid_size = NUM_GRID_CHANNELS * s_enc.height * s_enc.width
        extra_dim = s_enc.dim - grid_size

        net_auto = QhNet.from_encoders(s_enc, g_enc, NUM_ACTIONS)
        net_manual = QhNet(
            NUM_GRID_CHANNELS, s_enc.height, s_enc.width,
            extra_dim, g_enc.dim, NUM_ACTIONS,
        )
        # Same number of parameters
        auto_params = sum(p.numel() for p in net_auto.parameters())
        manual_params = sum(p.numel() for p in net_manual.parameters())
        assert auto_params == manual_params


# -----------------------------------------------------------------------
# VheNet — Phase 2 goal achievement
# -----------------------------------------------------------------------

class TestVheNet:

    def test_output_shape(self, encoders, state_and_goal):
        s_enc, g_enc = encoders
        state, goal = state_and_goal

        net = VheNet.from_encoders(s_enc, g_enc)
        v = net(state, goal)
        assert v.shape == (1,)

    def test_output_range(self, encoders):
        """Output should always be in [0, 1] (sigmoid)."""
        s_enc, g_enc = encoders
        net = VheNet.from_encoders(s_enc, g_enc)

        # Test with many random inputs
        states = torch.randn(100, s_enc.dim)
        goals = torch.randn(100, g_enc.dim)
        with torch.no_grad():
            v = net(states, goals)
        assert (v >= 0).all()
        assert (v <= 1).all()

    def test_batch(self, encoders):
        s_enc, g_enc = encoders
        net = VheNet.from_encoders(s_enc, g_enc)

        batch = 16
        v = net(torch.randn(batch, s_enc.dim), torch.randn(batch, g_enc.dim))
        assert v.shape == (batch,)

    def test_gradients_flow(self, encoders, state_and_goal):
        s_enc, g_enc = encoders
        state, goal = state_and_goal

        net = VheNet.from_encoders(s_enc, g_enc)
        v = net(state, goal)
        v.sum().backward()

        for name, param in net.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"


# -----------------------------------------------------------------------
# QrNet — Phase 2 robot Q-values
# -----------------------------------------------------------------------

class TestQrNet:

    def test_output_shape(self, encoders, state_and_goal):
        s_enc, _ = encoders
        state, _ = state_and_goal

        net = QrNet.from_encoders(s_enc, NUM_ACTIONS)
        q = net(state)
        assert q.shape == (1, NUM_ACTIONS)

    def test_no_goal_input(self, encoders):
        """QrNet takes only state, no goal."""
        s_enc, _ = encoders
        net = QrNet.from_encoders(s_enc, NUM_ACTIONS)
        q = net(torch.randn(4, s_enc.dim))
        assert q.shape == (4, NUM_ACTIONS)

    def test_gradients_flow(self, encoders, state_and_goal):
        s_enc, _ = encoders
        state, _ = state_and_goal

        net = QrNet.from_encoders(s_enc, NUM_ACTIONS)
        q = net(state)
        q.sum().backward()

        for name, param in net.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"

    def test_target_network_copy(self, encoders):
        """Can create a target network by copying state_dict."""
        s_enc, _ = encoders
        online = QrNet.from_encoders(s_enc, NUM_ACTIONS)
        target = QrNet.from_encoders(s_enc, NUM_ACTIONS)

        # Copy weights
        target.load_state_dict(online.state_dict())

        # Verify outputs match
        x = torch.randn(2, s_enc.dim)
        online.eval()
        target.eval()
        with torch.no_grad():
            assert torch.equal(online(x), target(x))

    def test_target_network_diverges_after_update(self, encoders):
        """After training online, target and online should differ."""
        s_enc, _ = encoders
        online = QrNet.from_encoders(s_enc, NUM_ACTIONS)
        target = QrNet.from_encoders(s_enc, NUM_ACTIONS)
        target.load_state_dict(online.state_dict())

        # One gradient step on online only
        optimizer = torch.optim.SGD(online.parameters(), lr=0.1)
        x = torch.randn(2, s_enc.dim)
        loss = online(x).sum()
        loss.backward()
        optimizer.step()

        # Now they should differ
        online.eval()
        target.eval()
        with torch.no_grad():
            assert not torch.equal(online(x), target(x))


# -----------------------------------------------------------------------
# Parameter counts (sanity check)
# -----------------------------------------------------------------------

class TestParameterCounts:

    def test_reasonable_size(self, encoders):
        """Networks should have a reasonable number of parameters."""
        s_enc, g_enc = encoders
        qh = QhNet.from_encoders(s_enc, g_enc, NUM_ACTIONS)
        vhe = VheNet.from_encoders(s_enc, g_enc)
        qr = QrNet.from_encoders(s_enc, NUM_ACTIONS)

        for name, net in [("QhNet", qh), ("VheNet", vhe), ("QrNet", qr)]:
            n_params = sum(p.numel() for p in net.parameters())
            # Should be between 10K and 500K for our grid sizes
            assert 10_000 < n_params < 500_000, (
                f"{name} has {n_params} params, expected 10K-500K"
            )


# -----------------------------------------------------------------------
# Integration with real encoded data
# -----------------------------------------------------------------------

class TestIntegration:

    def test_end_to_end_with_real_state(self, key_door_env, encoders):
        """Full pipeline: env → encode → network → Q-values."""
        from empo.world_specific_helpers.multigrid import ReachCellGoal

        s_enc, g_enc = encoders
        state = key_door_env.get_state()
        goal = ReachCellGoal(key_door_env, human_agent_index=1, target_pos=(3, 2))

        state_t = s_enc.encode(state).unsqueeze(0)
        goal_t = g_enc.encode(goal).unsqueeze(0)

        qh = QhNet.from_encoders(s_enc, g_enc, NUM_ACTIONS)
        vhe = VheNet.from_encoders(s_enc, g_enc)
        qr = QrNet.from_encoders(s_enc, NUM_ACTIONS)

        with torch.no_grad():
            q_human = qh(state_t, goal_t)
            v_achieve = vhe(state_t, goal_t)
            q_robot = qr(state_t)

        assert q_human.shape == (1, NUM_ACTIONS)
        assert v_achieve.shape == (1,)
        assert q_robot.shape == (1, NUM_ACTIONS)
        assert 0 <= v_achieve.item() <= 1


# -----------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------

if __name__ == "__main__":
    pytest.main([__file__, "-v"])

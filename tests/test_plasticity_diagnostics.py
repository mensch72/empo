"""Tests for the shared plasticity-loss diagnostics and its Phase 2 wrappers.

These tests use lightweight synthetic modules that mimic the Phase 2 network
interfaces, so they run without pufferlib or a full world-model setup.
"""

import torch
import torch.nn as nn

from empo.learning_based.plasticity_diagnostics import (
    measure_plasticity,
    effective_ranks,
    weight_norms,
)
from empo.learning_based.phase2.diagnostics import (
    compute_robot_network_plasticity,
)


# --------------------------------------------------------------------------
# Synthetic networks mirroring the DQN-path robot-network interface
# --------------------------------------------------------------------------


class _Encoder(nn.Module):
    def __init__(self, in_dim=8, feature_dim=24, hidden=32):
        super().__init__()
        self.feature_dim = feature_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, feature_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)

    def tensorize_state(self, state, world_model, device="cpu"):
        return torch.tensor(state, dtype=torch.float32).reshape(1, -1)


class _RobotNet(nn.Module):
    """Mimics BushWorldRobotQNetwork: state_encoder + head + forward_batch."""

    def __init__(self, in_dim=8, feature_dim=24, hidden=32, num_actions=4):
        super().__init__()
        self.state_encoder = _Encoder(in_dim, feature_dim, hidden)
        self.q_head = nn.Sequential(
            nn.Linear(feature_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_actions),
        )

    def forward_batch(self, states, world_model, device="cpu"):
        x = torch.cat(
            [self.state_encoder.tensorize_state(s, world_model, device) for s in states],
            dim=0,
        )
        return self.q_head(self.state_encoder(x))


class _LookupLike(nn.Module):
    """A ReLU-free network (like a lookup table) — should be skipped."""

    def __init__(self, num_actions=4):
        super().__init__()
        self.table = {}
        self.num_actions = num_actions

    def forward_batch(self, states, world_model, device="cpu"):
        return torch.zeros(len(states), self.num_actions)


def _states(n=50, dim=8):
    return [tuple(torch.randn(dim).tolist()) for _ in range(n)]


# --------------------------------------------------------------------------
# Shared low-level helpers
# --------------------------------------------------------------------------


class TestSharedHelpers:
    def test_effective_ranks_full_rank(self):
        feats = torch.randn(64, 16)
        er = effective_ranks(feats)
        assert 0.0 < er["srank"] <= 16.0
        assert 0.0 < er["erank"] <= 16.0

    def test_effective_ranks_rank_one(self):
        base = torch.randn(1, 16)
        feats = base.repeat(64, 1) + torch.arange(64).float().reshape(64, 1) * base
        er = effective_ranks(feats)
        # A near rank-1 matrix should have srank == 1.
        assert er["srank"] == 1

    def test_effective_ranks_degenerate_input(self):
        assert effective_ranks(torch.randn(1, 16)) == {}
        assert effective_ranks(torch.randn(5)) == {}

    def test_weight_norms(self):
        net = _RobotNet()
        wn = weight_norms(net)
        assert "weight_norm/total" in wn
        assert "weight_norm/state_encoder" in wn
        assert "weight_norm/q_head" in wn
        assert all(v >= 0 for v in wn.values())


# --------------------------------------------------------------------------
# measure_plasticity (generic)
# --------------------------------------------------------------------------


class TestMeasurePlasticity:
    def test_keys_and_mode_preserved(self):
        net = _RobotNet()
        states = _states()
        net.train()
        metrics = measure_plasticity(
            net,
            lambda: net.forward_batch(states, None, "cpu"),
            feature_module=net.state_encoder,
        )
        assert "dormant_frac/overall" in metrics
        assert "dead_frac/overall" in metrics
        assert "effective_rank/srank" in metrics
        assert "weight_norm/total" in metrics
        assert net.training is True  # mode restored

        net.eval()
        measure_plasticity(net, lambda: net.forward_batch(states, None, "cpu"))
        assert net.training is False

    def test_metric_ranges(self):
        net = _RobotNet()
        states = _states()
        metrics = measure_plasticity(
            net,
            lambda: net.forward_batch(states, None, "cpu"),
            feature_module=net.state_encoder,
        )
        for key, val in metrics.items():
            if key.startswith(("dormant_frac", "dead_frac")):
                assert 0.0 <= val <= 1.0, f"{key}={val}"
            if key.startswith("weight_norm"):
                assert val >= 0.0

    def test_dead_layer_detection(self):
        net = _RobotNet()
        with torch.no_grad():
            net.q_head[0].weight.zero_()
            net.q_head[0].bias.fill_(-100.0)
        states = _states()
        metrics = measure_plasticity(
            net,
            lambda: net.forward_batch(states, None, "cpu"),
            feature_module=net.state_encoder,
        )
        assert metrics["dead_frac/q_head_1"] == 1.0
        assert metrics["dormant_frac/q_head_1"] == 1.0


# --------------------------------------------------------------------------
# DQN-path wrapper
# --------------------------------------------------------------------------


class TestComputeRobotNetworkPlasticity:
    def test_neural_net(self):
        net = _RobotNet()
        metrics = compute_robot_network_plasticity(net, _states(), None, "cpu")
        assert "dormant_frac/overall" in metrics
        assert "effective_rank/srank" in metrics
        assert "weight_norm/total" in metrics

    def test_lookup_table_skipped(self):
        assert compute_robot_network_plasticity(_LookupLike(), _states(), None, "cpu") == {}

    def test_empty_states(self):
        assert compute_robot_network_plasticity(_RobotNet(), [], None, "cpu") == {}

#!/usr/bin/env python3
"""
Tests for gradient-based convergence metrics (issue #122).

Tests the EMA of gradient norm and cosine similarity between successive
gradients, as implemented in BasePhase2Trainer.
"""

import torch
import torch.nn as nn

from empo.learning_based.phase2 import Phase2Config
from empo.learning_based.phase2.trainer import BasePhase2Trainer


class SimpleNet(nn.Module):
    """Minimal network for testing gradient metrics."""
    def __init__(self, in_dim=4, out_dim=2):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.linear(x)


class FakeNetworks:
    """Fake Phase2Networks-like object holding just what gradient metrics need."""
    def __init__(self):
        self.q_r = SimpleNet()
        self.v_h_e = SimpleNet()
        self.x_h = None
        self.u_r = None
        self.v_r = None
        # Target nets (unused but required by some trainer paths)
        self.q_r_target = SimpleNet()
        self.v_h_e_target = SimpleNet()
        self.x_h_target = None
        self.u_r_target = None
        self.v_r_target = None


class _ConcreteTrainer(BasePhase2Trainer):
    """Minimal concrete subclass to test BasePhase2Trainer methods."""
    def get_state_features_for_rnd(self, *args, **kwargs):
        return None


def _make_stub(**config_kwargs):
    """Create a bare _ConcreteTrainer instance with only the fields
    required by the gradient metric methods (bypasses full __init__)."""
    config = Phase2Config(**config_kwargs)
    obj = _ConcreteTrainer.__new__(_ConcreteTrainer)
    obj.config = config
    obj.networks = FakeNetworks()
    obj._grad_norm_ema = {}
    obj._grad_cosine_sim = {}
    obj._prev_flat_grads = {}
    return obj


# ---------------------------------------------------------------------------
# Unit tests for _update_grad_metrics and helpers
# ---------------------------------------------------------------------------

def test_config_grad_metrics_ema_decay_default():
    """Test that grad_metrics_ema_decay has the expected default."""
    config = Phase2Config()
    assert config.grad_metrics_ema_decay == 0.99
    print("  ✓ Default grad_metrics_ema_decay == 0.99")


def test_config_grad_metrics_ema_decay_custom():
    """Test that grad_metrics_ema_decay can be customized."""
    config = Phase2Config(grad_metrics_ema_decay=0.9)
    assert config.grad_metrics_ema_decay == 0.9
    print("  ✓ Custom grad_metrics_ema_decay == 0.9")


def test_update_grad_metrics_ema_initialization():
    """First call should initialise EMA to the current grad norm value."""
    obj = _make_stub(grad_metrics_ema_decay=0.9)

    # Simulate a gradient by doing a forward/backward on q_r
    x = torch.randn(1, 4)
    out = obj.networks.q_r(x)
    out.sum().backward()

    # Compute grad norm manually
    norm = 0.0
    for p in obj.networks.q_r.parameters():
        if p.grad is not None:
            norm += p.grad.data.norm(2).item() ** 2
    norm = norm ** 0.5

    grad_norms = {'q_r': norm}
    obj._update_grad_metrics(grad_norms)

    # First call: EMA should equal the norm itself
    assert abs(obj._grad_norm_ema['q_r'] - norm) < 1e-8
    # First call: no previous gradient, so cosine_sim should not yet exist
    # (it only appears after the second call)
    assert 'q_r' not in obj._grad_cosine_sim
    print("  ✓ EMA initialized to first grad norm")
    print("  ✓ Cosine similarity not yet available on first step")


def test_update_grad_metrics_ema_decay():
    """EMA should converge toward a constant stream of values."""
    decay = 0.9
    obj = _make_stub(grad_metrics_ema_decay=decay)

    # Feed constant norm for many steps
    constant_norm = 5.0
    for step in range(200):
        # Give q_r a synthetic gradient so flatten works
        x = torch.randn(1, 4)
        out = obj.networks.q_r(x)
        out.sum().backward()
        obj._update_grad_metrics({'q_r': constant_norm})

    # After many steps with constant input, EMA should converge to that constant
    assert abs(obj._grad_norm_ema['q_r'] - constant_norm) < 0.01, (
        f"EMA should converge to {constant_norm}, got {obj._grad_norm_ema['q_r']}"
    )
    print(f"  ✓ EMA converged to ~{constant_norm} after 200 steps")


def test_update_grad_metrics_cosine_similarity():
    """Cosine similarity should be 1 for identical gradients and -1 for opposing."""
    obj = _make_stub(grad_metrics_ema_decay=0.99)

    net = obj.networks.q_r

    # Step 1: set a deterministic gradient
    net.zero_grad()
    for p in net.parameters():
        p.grad = torch.ones_like(p.data)
    norm1 = sum(p.grad.data.norm(2).item() ** 2 for p in net.parameters()) ** 0.5
    obj._update_grad_metrics({'q_r': norm1})
    assert 'q_r' not in obj._grad_cosine_sim  # first step, no previous

    # Step 2: identical gradient → cosine should be 1
    net.zero_grad()
    for p in net.parameters():
        p.grad = torch.ones_like(p.data)
    norm2 = norm1
    obj._update_grad_metrics({'q_r': norm2})
    assert abs(obj._grad_cosine_sim['q_r'] - 1.0) < 1e-6, (
        f"Expected cosine_sim ~1.0, got {obj._grad_cosine_sim['q_r']}"
    )
    print("  ✓ Identical gradients → cosine_sim ≈ 1.0")

    # Step 3: opposing gradient → cosine should be -1
    net.zero_grad()
    for p in net.parameters():
        p.grad = -torch.ones_like(p.data)
    norm3 = norm1  # same magnitude
    obj._update_grad_metrics({'q_r': norm3})
    assert abs(obj._grad_cosine_sim['q_r'] - (-1.0)) < 1e-6, (
        f"Expected cosine_sim ~-1.0, got {obj._grad_cosine_sim['q_r']}"
    )
    print("  ✓ Opposing gradients → cosine_sim ≈ -1.0")


def test_update_grad_metrics_orthogonal():
    """Cosine similarity should be ~0 for orthogonal gradients."""
    obj = _make_stub(grad_metrics_ema_decay=0.99)

    # Use a minimal single-parameter network so we control the gradient vector exactly
    single_net = nn.Linear(2, 1, bias=False)  # 2 params
    obj.networks.q_r = single_net

    # Step 1: gradient = [1, 0]
    single_net.zero_grad()
    single_net.weight.grad = torch.tensor([[1.0, 0.0]])
    obj._update_grad_metrics({'q_r': 1.0})

    # Step 2: gradient = [0, 1] → orthogonal
    single_net.zero_grad()
    single_net.weight.grad = torch.tensor([[0.0, 1.0]])
    obj._update_grad_metrics({'q_r': 1.0})

    assert abs(obj._grad_cosine_sim['q_r']) < 1e-6, (
        f"Expected cosine_sim ~0.0, got {obj._grad_cosine_sim['q_r']}"
    )
    print("  ✓ Orthogonal gradients → cosine_sim ≈ 0.0")


def test_flatten_network_grads():
    """_flatten_network_grads should return a 1-D CPU tensor."""
    obj = _make_stub()

    net = obj.networks.q_r
    # No grad yet → should return None
    net.zero_grad()
    assert obj._flatten_network_grads('q_r') is None
    print("  ✓ Returns None when no gradients present")

    # Give it a gradient
    x = torch.randn(1, 4)
    out = net(x)
    out.sum().backward()

    flat = obj._flatten_network_grads('q_r')
    assert flat is not None
    assert flat.dim() == 1
    assert flat.device.type == 'cpu'

    # Check size equals total parameters with grads
    expected_size = sum(p.numel() for p in net.parameters() if p.grad is not None)
    assert flat.shape[0] == expected_size
    print(f"  ✓ Flattened gradient has {flat.shape[0]} elements (expected {expected_size})")

    # Unknown network → None
    assert obj._flatten_network_grads('nonexistent') is None
    print("  ✓ Returns None for unknown network name")


def test_get_configured_network_map():
    """_get_configured_network_map returns only configured/present networks."""
    # Default config: x_h_use_network=True but x_h=None, u_r/v_r_use_network=False
    obj = _make_stub()

    configured = obj._get_configured_network_map()
    assert 'q_r' in configured
    assert 'v_h_e' in configured
    # x_h_use_network=True by default but networks.x_h is None
    assert 'x_h' not in configured
    assert 'u_r' not in configured
    assert 'v_r' not in configured
    print("  ✓ Default config returns q_r and v_h_e only")


def test_multiple_networks_tracked():
    """EMA and cosine should be tracked independently for each network."""
    obj = _make_stub(grad_metrics_ema_decay=0.5)

    # Give both networks gradients
    for net in [obj.networks.q_r, obj.networks.v_h_e]:
        net.zero_grad()
        x = torch.randn(1, 4)
        out = net(x)
        out.sum().backward()

    grad_norms = {'q_r': 1.0, 'v_h_e': 2.0}
    obj._update_grad_metrics(grad_norms)

    assert abs(obj._grad_norm_ema['q_r'] - 1.0) < 1e-8
    assert abs(obj._grad_norm_ema['v_h_e'] - 2.0) < 1e-8
    print("  ✓ q_r EMA initialized to 1.0, v_h_e EMA initialized to 2.0")

    # Second step
    for net in [obj.networks.q_r, obj.networks.v_h_e]:
        net.zero_grad()
        x = torch.randn(1, 4)
        out = net(x)
        out.sum().backward()

    grad_norms2 = {'q_r': 3.0, 'v_h_e': 4.0}
    obj._update_grad_metrics(grad_norms2)

    # EMA: decay * prev + (1 - decay) * new
    expected_q_r = 0.5 * 1.0 + 0.5 * 3.0  # 2.0
    expected_v_h_e = 0.5 * 2.0 + 0.5 * 4.0  # 3.0
    assert abs(obj._grad_norm_ema['q_r'] - expected_q_r) < 1e-8
    assert abs(obj._grad_norm_ema['v_h_e'] - expected_v_h_e) < 1e-8
    print(f"  ✓ After step 2: q_r EMA={expected_q_r}, v_h_e EMA={expected_v_h_e}")


def test_config_save_yaml_includes_grad_metrics(tmp_path):
    """save_yaml should include the grad_metrics_ema_decay field."""
    import yaml

    config = Phase2Config(grad_metrics_ema_decay=0.95)
    path = str(tmp_path / "config.yaml")
    config.save_yaml(path)

    with open(path) as f:
        saved = yaml.safe_load(f)

    assert saved['regularization']['gradient_metrics']['grad_metrics_ema_decay'] == 0.95
    print("  ✓ save_yaml includes grad_metrics_ema_decay")


def test_config_grad_metrics_ema_decay_validation():
    """grad_metrics_ema_decay must be in [0, 1)."""
    import pytest

    # Valid boundary values
    Phase2Config(grad_metrics_ema_decay=0.0)
    Phase2Config(grad_metrics_ema_decay=0.5)
    print("  ✓ Accepts valid values 0.0, 0.5")

    # Invalid: >= 1
    with pytest.raises(ValueError, match="grad_metrics_ema_decay"):
        Phase2Config(grad_metrics_ema_decay=1.0)
    print("  ✓ Rejects 1.0")

    with pytest.raises(ValueError, match="grad_metrics_ema_decay"):
        Phase2Config(grad_metrics_ema_decay=1.5)
    print("  ✓ Rejects 1.5")

    # Invalid: < 0
    with pytest.raises(ValueError, match="grad_metrics_ema_decay"):
        Phase2Config(grad_metrics_ema_decay=-0.1)
    print("  ✓ Rejects -0.1")


def test_stale_cosine_cleared_on_no_grad():
    """Cosine similarity should be cleared when gradients disappear."""
    obj = _make_stub(grad_metrics_ema_decay=0.99)
    net = obj.networks.q_r

    # Step 1: set gradient
    net.zero_grad()
    for p in net.parameters():
        p.grad = torch.ones_like(p.data)
    obj._update_grad_metrics({'q_r': 1.0})

    # Step 2: same gradient → cosine = 1
    net.zero_grad()
    for p in net.parameters():
        p.grad = torch.ones_like(p.data)
    obj._update_grad_metrics({'q_r': 1.0})
    assert 'q_r' in obj._grad_cosine_sim
    assert abs(obj._grad_cosine_sim['q_r'] - 1.0) < 1e-6

    # Step 3: remove gradients → cosine should be cleared
    net.zero_grad()
    obj._update_grad_metrics({'q_r': 0.0})
    assert 'q_r' not in obj._grad_cosine_sim
    assert 'q_r' not in obj._prev_flat_grads
    print("  ✓ Stale cosine similarity cleared when gradients disappear")


def run_all_tests():
    """Run all gradient metrics tests."""
    import tempfile
    from pathlib import Path

    print("=" * 60)
    print("Running Gradient Metrics Tests (issue #122)")
    print("=" * 60)

    test_config_grad_metrics_ema_decay_default()
    test_config_grad_metrics_ema_decay_custom()
    test_config_grad_metrics_ema_decay_validation()
    print()

    test_flatten_network_grads()
    print()

    test_get_configured_network_map()
    print()

    test_update_grad_metrics_ema_initialization()
    print()

    test_update_grad_metrics_ema_decay()
    print()

    test_update_grad_metrics_cosine_similarity()
    print()

    test_update_grad_metrics_orthogonal()
    print()

    test_multiple_networks_tracked()
    print()

    test_stale_cosine_cleared_on_no_grad()
    print()

    with tempfile.TemporaryDirectory() as td:
        test_config_save_yaml_includes_grad_metrics(Path(td))
    print()

    print("=" * 60)
    print("All gradient metrics tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()

"""
Tests for the BushWorld-specific Phase 1 (human policy prior) DQN path.

These tests verify the BushWorld analogue of the multigrid Phase 1 package:
1. ``BushWorldQNetwork`` — forward shapes and config round-trip.
2. ``train_bushworld_neural_policy_prior`` — short training run produces a
   valid (normalised) goal-conditioned and marginal policy.
3. ``BushWorldNeuralHumanPolicyPrior`` — save/load round-trip.

All tests use a real ``BushWorld`` instance and exercise the encoder pipeline
end-to-end. They reuse the shared, generic Phase 1 ``Trainer`` (no new training
algorithm).
"""

import numpy as np
import pytest
import torch

from empo.bushworld import BushWorld
from empo.learning_based.bushworld.phase1 import (
    BushWorldDirectPhiNetwork,
    BushWorldNeuralHumanPolicyPrior,
    BushWorldQNetwork,
    train_bushworld_neural_policy_prior,
)


# --------------------------------------------------------------------------- #
# Fixtures / helpers
# --------------------------------------------------------------------------- #
def make_corridor(max_steps=4, B=1):
    """1x5 corridor: robot at (2,0), humans at (0,0) and (4,0), all density 1."""
    return BushWorld(
        width=5,
        height=1,
        num_robots=1,
        num_humans=2,
        max_steps=max_steps,
        B=B,
        robot_positions=[(2, 0)],
        human_positions=[(0, 0), (4, 0)],
        initial_densities=[[1, 1, 1, 1, 1]],
    )


def _make_q_network(env, feature_dim=16, goal_feature_dim=8, hidden_dim=16):
    return BushWorldQNetwork(
        grid_height=env.height,
        grid_width=env.width,
        B=env.B,
        num_robots=len(env.robot_agent_indices),
        max_steps=env.max_steps,
        num_actions=env.action_space.n,
        state_feature_dim=feature_dim,
        goal_feature_dim=goal_feature_dim,
        hidden_dim=hidden_dim,
    )


def _probs_to_array(probs_dict, num_actions):
    """Convert the ``{action: prob}`` dict returned by the prior to an array."""
    return np.array([probs_dict[i] for i in range(num_actions)], dtype=np.float64)


# --------------------------------------------------------------------------- #
# Q-network
# --------------------------------------------------------------------------- #
def test_q_network_forward_shape():
    env = make_corridor()
    q_net = _make_q_network(env)
    state = env.initial_state()
    human_idx = env.human_agent_indices[0]
    goal, _ = env.possible_goal_generator.get_sampler().sample(state, human_idx)

    q_values = q_net(state, env, human_idx, goal, device="cpu")
    assert q_values.shape[-1] == env.action_space.n
    assert torch.isfinite(q_values).all()


def test_q_network_config_roundtrip():
    env = make_corridor()
    q_net = _make_q_network(env)
    config = q_net.get_config()
    # Config must be sufficient to reconstruct an equivalent network.
    q_net2 = BushWorldQNetwork(**config)
    assert q_net2.get_config() == config


# --------------------------------------------------------------------------- #
# Direct-phi network
# --------------------------------------------------------------------------- #
def test_direct_phi_network_produces_distribution():
    env = make_corridor()
    phi = BushWorldDirectPhiNetwork(
        grid_height=env.height,
        grid_width=env.width,
        B=env.B,
        num_robots=len(env.robot_agent_indices),
        max_steps=env.max_steps,
        num_actions=env.action_space.n,
        state_feature_dim=16,
        hidden_dim=16,
    )
    state = env.initial_state()
    human_idx = env.human_agent_indices[0]
    probs = phi(state, env, human_idx, device="cpu")
    probs_np = probs.detach().cpu().numpy().reshape(-1)
    assert probs_np.shape[0] == env.action_space.n
    np.testing.assert_allclose(probs_np.sum(), 1.0, atol=1e-5)
    assert (probs_np >= 0).all()


# --------------------------------------------------------------------------- #
# Training + neural human policy prior
# --------------------------------------------------------------------------- #
def test_train_produces_valid_policies():
    env = make_corridor()
    human_idx = env.human_agent_indices[0]
    sampler = env.possible_goal_generator.get_sampler()
    num_actions = env.action_space.n

    prior = train_bushworld_neural_policy_prior(
        env,
        human_agent_indices=[human_idx],
        goal_sampler=sampler,
        num_episodes=5,
        steps_per_episode=4,
        batch_size=8,
        state_feature_dim=16,
        goal_feature_dim=8,
        hidden_dim=16,
        verbose=False,
    )

    state = env.initial_state()
    goal, _ = sampler.sample(state, human_idx)

    # Goal-conditioned policy is a valid distribution.
    gc = _probs_to_array(prior(state, human_idx, goal), num_actions)
    assert gc.shape[0] == num_actions
    np.testing.assert_allclose(gc.sum(), 1.0, atol=1e-4)
    assert (gc >= 0).all()

    # Marginal policy is a valid distribution.
    marg = _probs_to_array(prior(state, human_idx), num_actions)
    assert marg.shape[0] == num_actions
    np.testing.assert_allclose(marg.sum(), 1.0, atol=1e-4)
    assert (marg >= 0).all()


def test_neural_prior_save_load(tmp_path):
    env = make_corridor()
    human_idx = env.human_agent_indices[0]
    sampler = env.possible_goal_generator.get_sampler()
    num_actions = env.action_space.n

    prior = train_bushworld_neural_policy_prior(
        env,
        human_agent_indices=[human_idx],
        goal_sampler=sampler,
        num_episodes=3,
        steps_per_episode=4,
        batch_size=8,
        state_feature_dim=16,
        goal_feature_dim=8,
        hidden_dim=16,
        verbose=False,
    )

    save_path = tmp_path / "bushworld_phase1_prior.pt"
    prior.save(str(save_path))
    assert save_path.exists()

    loaded = BushWorldNeuralHumanPolicyPrior.load(
        str(save_path),
        env,
        human_agent_indices=[human_idx],
        goal_sampler=sampler,
        device="cpu",
    )

    state = env.initial_state()
    goal, _ = sampler.sample(state, human_idx)
    a = _probs_to_array(prior(state, human_idx, goal), num_actions)
    b = _probs_to_array(loaded(state, human_idx, goal), num_actions)
    np.testing.assert_allclose(a, b, atol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

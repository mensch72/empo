"""
Tests for the PPO-based Phase 2 implementation (PufferLib-backed).

These tests verify the **new** PPO code path without touching or importing
anything from the existing DQN-based Phase 2 trainer.  The only shared
imports are the stable public data structures (Phase2Transition,
Phase2ReplayBuffer, base network classes).

Test categories:
    1. PPOPhase2Config — creation, validation, helpers, PufferLib config
    2. EMPOActorCritic — forward (PufferLib convention), encode/decode,
       get_action_and_value, action mapping
    3. EMPOMultiGridEnv — reset, step, observation / reward / info contract
    4. PPOPhase2Trainer — auxiliary training, push_transition_to_aux_buffer,
       PufferLib training loop
    5. Isolation check — no existing Phase 2 files modified
"""

import os
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
import gymnasium

import pufferlib
import pufferlib.emulation
import pufferlib.vector

# ── PPO-path imports (new code under test) ──────────────────────────────
from empo.learning_based.phase2_ppo.config import PPOPhase2Config
from empo.learning_based.phase2_ppo.actor_critic import EMPOActorCritic
from empo.learning_based.phase2_ppo.env_wrapper import EMPOMultiGridEnv
from empo.learning_based.phase2_ppo.trainer import (
    PPOPhase2Trainer,
    PPOAuxiliaryNetworks,
)

# ── Shared read-only imports (from DQN path, not modified) ──────────────
from empo.learning_based.phase2.replay_buffer import (
    Phase2ReplayBuffer,
)
from empo.learning_based.phase2.human_goal_ability import (
    BaseHumanGoalAchievementNetwork,
)


# ======================================================================
# Fixtures & mocks
# ======================================================================


class MockWorldModel:
    """Minimal WorldModel mock for testing the PPO env wrapper."""

    def __init__(self, n_agents: int = 3, n_actions: int = 5):
        self.n_agents = n_agents
        self.n_actions = n_actions
        self._state = (0,) * n_agents  # positions
        self._step_count = 0
        self.human_agent_indices = list(range(n_agents - 1))

    def reset(self, *, seed=None, options=None):
        self._state = (0,) * self.n_agents
        self._step_count = 0
        return 0, {}

    def get_state(self):
        return self._state

    def set_state(self, state):
        self._state = state

    def step(self, joint_action):
        self._step_count += 1
        # Deterministic successor: increment each agent's pos by its action
        new_state = tuple(
            (s + a) % 10 for s, a in zip(self._state, joint_action)
        )
        self._state = new_state
        terminated = self._step_count >= 50
        return 0, 0.0, terminated, False, {}

    def transition_probabilities(self, state, joint_action):
        # Deterministic: single successor with prob 1.0
        new_state = tuple(
            (s + a) % 10 for s, a in zip(state, joint_action)
        )
        return [(1.0, new_state)]


def mock_human_policy_prior(state, human_idx, goal, world_model):
    """Uniform random policy prior (adapts to world_model.n_actions)."""
    n = getattr(world_model, "n_actions", 5)
    return [1.0 / n] * n


def mock_goal_sampler(state, human_idx):
    """Returns a dummy goal and weight."""
    return f"goal_{human_idx}", 1.0


class _PufferLibCompatEnv(EMPOMultiGridEnv):
    """Thin wrapper that strips non-numeric info for PufferLib compatibility.

    PufferLib's Serial vectorisation backend calls ``np.mean()`` on every
    info value.  EMPOMultiGridEnv returns complex auxiliary data (tuples,
    dicts) which cannot be averaged.  This wrapper keeps only scalar numeric
    values so the PufferLib training loop runs without error.
    """

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        numeric_info = {
            k: v
            for k, v in info.items()
            if isinstance(v, (int, float, np.integer, np.floating))
        }
        return obs, reward, terminated, truncated, numeric_info


# ======================================================================
# 1. PPOPhase2Config tests
# ======================================================================


class _MockVhE(BaseHumanGoalAchievementNetwork):
    """Minimal V_h^e implementation for testing auxiliary training."""

    def __init__(self):
        super().__init__(gamma_h=0.99)
        self._linear = torch.nn.Linear(1, 1)

    def forward(self, state, world_model, human_agent_idx, goal, device="cpu"):
        # Return a constant prediction ∈ [0, 1]
        return self.apply_clamp(self._linear(torch.zeros(1, device=device)))

    def get_config(self):
        return {"type": "mock"}


class TestPPOPhase2Config:
    """Tests for the standalone PPO config class."""

    def test_default_creation(self):
        cfg = PPOPhase2Config()
        assert cfg.gamma_r == 0.99
        assert cfg.gamma_h == 0.99
        assert cfg.zeta == 2.0
        assert cfg.ppo_clip_coef == 0.2
        assert cfg.num_joint_actions == 7  # 7^1

    def test_custom_values(self):
        cfg = PPOPhase2Config(
            gamma_r=0.95, num_actions=3, num_robots=2
        )
        assert cfg.gamma_r == 0.95
        assert cfg.num_joint_actions == 9  # 3^2

    def test_zeta_validation(self):
        with pytest.raises(ValueError, match="zeta"):
            PPOPhase2Config(zeta=0.5)

    def test_xi_validation(self):
        with pytest.raises(ValueError, match="xi"):
            PPOPhase2Config(xi=0.0)

    def test_eta_validation(self):
        with pytest.raises(ValueError, match="eta"):
            PPOPhase2Config(eta=0.9)

    def test_gamma_r_validation(self):
        with pytest.raises(ValueError, match="gamma_r"):
            PPOPhase2Config(gamma_r=1.5)

    def test_gamma_h_validation(self):
        with pytest.raises(ValueError, match="gamma_h"):
            PPOPhase2Config(gamma_h=-0.1)

    def test_entropy_schedule(self):
        cfg = PPOPhase2Config(
            ppo_ent_coef_start=0.1,
            ppo_ent_coef_end=0.01,
            ppo_ent_anneal_steps=100,
        )
        # Start
        assert cfg.get_entropy_coef(0) == pytest.approx(0.1)
        # Midpoint
        assert cfg.get_entropy_coef(50) == pytest.approx(0.055)
        # End
        assert cfg.get_entropy_coef(100) == pytest.approx(0.01)
        # Beyond end
        assert cfg.get_entropy_coef(200) == pytest.approx(0.01)

    def test_total_warmup_steps(self):
        cfg = PPOPhase2Config(warmup_u_r_steps=12345)
        assert cfg.get_total_warmup_steps() == 12345

    def test_does_not_inherit_from_phase2config(self):
        """PPOPhase2Config must NOT be a subclass of Phase2Config."""
        from empo.learning_based.phase2.config import Phase2Config

        assert not issubclass(PPOPhase2Config, Phase2Config)

    def test_to_pufferlib_config(self):
        """to_pufferlib_config() produces a valid PufferLib config dict."""
        cfg = PPOPhase2Config(
            num_envs=4,
            ppo_rollout_length=32,
            ppo_num_minibatches=2,
            ppo_update_epochs=3,
            gamma_r=0.97,
            ppo_gae_lambda=0.9,
            ppo_clip_coef=0.1,
            ppo_vf_coef=0.25,
            ppo_max_grad_norm=1.0,
            lr_ppo=1e-3,
            ppo_ent_coef_start=0.05,
        )
        d = cfg.to_pufferlib_config()

        expected_batch = 4 * 32  # num_envs * rollout_length
        assert d["batch_size"] == expected_batch
        assert d["bptt_horizon"] == 32
        assert d["minibatch_size"] == expected_batch // 2
        assert d["update_epochs"] == 3
        assert d["gamma"] == 0.97
        assert d["gae_lambda"] == 0.9
        assert d["clip_coef"] == 0.1
        assert d["vf_coef"] == 0.25
        assert d["max_grad_norm"] == 1.0
        assert d["learning_rate"] == 1e-3
        assert d["ent_coef"] == 0.05
        # Should contain all keys expected by PuffeRL
        for key in [
            "seed", "total_timesteps", "compile", "use_rnn",
            "device", "anneal_lr",
        ]:
            assert key in d, f"Missing PufferLib key: {key}"


# ======================================================================
# 2. EMPOActorCritic tests
# ======================================================================


class TestEMPOActorCritic:
    """Tests for the actor-critic network."""

    def test_forward_no_encoder(self):
        ac = EMPOActorCritic(
            state_encoder=None,
            hidden_dim=32,
            num_actions=4,
            num_robots=1,
            obs_dim=8,
        )
        obs = torch.randn(3, 8)
        logits, value = ac(obs)
        assert logits.shape == (3, 4)
        assert value.shape == (3, 1)

    def test_forward_with_encoder(self):
        enc = torch.nn.Linear(16, 32)
        enc.output_dim = 32  # type: ignore[attr-defined]
        ac = EMPOActorCritic(
            state_encoder=enc,
            hidden_dim=64,
            num_actions=5,
            num_robots=1,
        )
        obs = torch.randn(2, 16)
        logits, value = ac(obs)
        assert logits.shape == (2, 5)
        assert value.shape == (2, 1)

    def test_multi_robot_joint_actions(self):
        ac = EMPOActorCritic(
            state_encoder=None,
            hidden_dim=16,
            num_actions=3,
            num_robots=2,
            obs_dim=4,
        )
        obs = torch.randn(1, 4)
        logits, value = ac(obs)
        assert logits.shape == (1, 9)  # 3^2
        assert value.shape == (1, 1)

    def test_forward_returns_pufferlib_convention(self):
        """forward() returns (logits, value) with PufferLib-expected shapes."""
        ac = EMPOActorCritic(
            state_encoder=None,
            hidden_dim=32,
            num_actions=6,
            num_robots=1,
            obs_dim=10,
        )
        B = 7
        obs = torch.randn(B, 10)
        logits, value = ac.forward(obs)

        assert logits.shape == (B, 6), "logits should be (B, num_actions)"
        assert value.shape == (B, 1), "value should be (B, 1) for PufferLib"
        assert logits.dtype == torch.float32
        assert value.dtype == torch.float32

    def test_forward_eval_exists(self):
        """forward_eval is present and returns same results as forward."""
        ac = EMPOActorCritic(
            state_encoder=None,
            hidden_dim=16,
            num_actions=4,
            obs_dim=8,
        )
        obs = torch.randn(2, 8)
        assert hasattr(ac, "forward_eval"), "forward_eval method missing"

        logits_fwd, value_fwd = ac.forward(obs)
        logits_eval, value_eval = ac.forward_eval(obs)

        assert torch.equal(logits_fwd, logits_eval)
        assert torch.equal(value_fwd, value_eval)

    def test_encode_decode_roundtrip(self):
        """encode_observations + decode_actions matches forward."""
        ac = EMPOActorCritic(
            state_encoder=None,
            hidden_dim=32,
            num_actions=5,
            obs_dim=8,
        )
        obs = torch.randn(4, 8)
        logits_fwd, value_fwd = ac.forward(obs)

        hidden = ac.encode_observations(obs)
        logits_dec, value_dec = ac.decode_actions(hidden)

        assert torch.equal(logits_fwd, logits_dec)
        assert torch.equal(value_fwd, value_dec)

    def test_get_action_and_value(self):
        ac = EMPOActorCritic(
            state_encoder=None,
            hidden_dim=16,
            num_actions=4,
            obs_dim=8,
        )
        obs = torch.randn(5, 8)
        action, lp, ent, val = ac.get_action_and_value(obs)
        assert action.shape == (5,)
        assert lp.shape == (5,)
        assert ent.shape == (5,)
        assert val.shape == (5,)
        # Actions should be valid indices
        assert (action >= 0).all() and (action < 4).all()

    def test_get_action_and_value_with_action(self):
        ac = EMPOActorCritic(
            state_encoder=None,
            hidden_dim=16,
            num_actions=4,
            obs_dim=8,
        )
        obs = torch.randn(3, 8)
        fixed_action = torch.tensor([0, 1, 2])
        action, lp, ent, val = ac.get_action_and_value(obs, fixed_action)
        assert torch.equal(action, fixed_action)

    def test_get_value(self):
        ac = EMPOActorCritic(
            state_encoder=None,
            hidden_dim=16,
            num_actions=4,
            obs_dim=8,
        )
        obs = torch.randn(2, 8)
        val = ac.get_value(obs)
        assert val.shape == (2,)

    def test_action_index_mapping_single_robot(self):
        ac = EMPOActorCritic(
            state_encoder=None,
            hidden_dim=16,
            num_actions=5,
            num_robots=1,
            obs_dim=4,
        )
        for i in range(5):
            assert ac.action_index_to_tuple(i) == (i,)
            assert ac.action_tuple_to_index((i,)) == i

    def test_action_index_mapping_multi_robot(self):
        ac = EMPOActorCritic(
            state_encoder=None,
            hidden_dim=16,
            num_actions=3,
            num_robots=2,
            obs_dim=4,
        )
        # (0,0) → 0, (1,0) → 1, (2,0) → 2, (0,1) → 3, etc.
        for idx in range(9):
            t = ac.action_index_to_tuple(idx)
            assert ac.action_tuple_to_index(t) == idx

    def test_requires_encoder_or_obs_dim(self):
        with pytest.raises(ValueError, match="Either state_encoder"):
            EMPOActorCritic(
                state_encoder=None,
                hidden_dim=16,
                num_actions=4,
            )


# ======================================================================
# 3. EMPOMultiGridEnv tests
# ======================================================================


class TestEMPOMultiGridEnv:
    """Tests for the Gymnasium-compatible environment wrapper."""

    def _make_env(self, **kwargs):
        wm = MockWorldModel(n_agents=3, n_actions=5)
        cfg = PPOPhase2Config(num_actions=5, num_robots=1, steps_per_episode=50)
        return EMPOMultiGridEnv(
            world_model=wm,
            human_policy_prior=mock_human_policy_prior,
            goal_sampler=mock_goal_sampler,
            human_agent_indices=[0, 1],
            robot_agent_indices=[2],
            config=cfg,
            obs_dim=3,
            **kwargs,
        )

    def test_reset_returns_obs_and_info(self):
        env = self._make_env()
        obs, info = env.reset()
        assert isinstance(obs, np.ndarray)
        assert obs.shape == (3,)
        assert "state" in info

    def test_step_returns_five_tuple(self):
        env = self._make_env()
        env.reset()
        obs, reward, terminated, truncated, info = env.step(0)
        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_reward_is_zero_without_aux_nets(self):
        env = self._make_env()
        env.reset()
        _, reward, _, _, _ = env.step(0)
        assert reward == 0.0  # No auxiliary networks → zero reward

    def test_info_contains_auxiliary_data(self):
        env = self._make_env()
        env.reset()
        _, _, _, _, info = env.step(2)
        assert "state" in info
        assert "next_state" in info
        assert "goals" in info
        assert "goal_weights" in info
        assert "human_actions" in info
        assert "transition_probs" in info
        assert "u_r" in info
        assert "env_reward" in info

    def test_transition_probs_per_robot_action(self):
        env = self._make_env()
        env.reset()
        _, _, _, _, info = env.step(0)
        tp = info["transition_probs"]
        # Should have entries for robot actions 0-4
        assert len(tp) == 5

    def test_action_space_discrete(self):
        env = self._make_env()
        assert isinstance(env.action_space, gymnasium.spaces.Discrete)
        assert env.action_space.n == 5

    def test_episode_terminates(self):
        env = self._make_env()
        env.reset()
        terminated = False
        for _ in range(100):
            _, _, terminated, truncated, _ = env.step(
                env.action_space.sample()
            )
            if terminated or truncated:
                break
        assert terminated  # MockWorldModel terminates after 50 steps

    def test_episode_truncates_at_steps_per_episode(self):
        """Episode is truncated when step count reaches steps_per_episode."""
        wm = MockWorldModel(n_agents=3, n_actions=5)
        cfg = PPOPhase2Config(
            num_actions=5, num_robots=1, steps_per_episode=10
        )
        env = EMPOMultiGridEnv(
            world_model=wm,
            human_policy_prior=mock_human_policy_prior,
            goal_sampler=mock_goal_sampler,
            human_agent_indices=[0, 1],
            robot_agent_indices=[2],
            config=cfg,
            obs_dim=3,
        )
        env.reset()
        for i in range(9):
            _, _, terminated, truncated, _ = env.step(0)
            assert not truncated, f"Unexpected truncation at step {i+1}"
        # Step 10 should trigger truncation
        _, _, _, truncated, _ = env.step(0)
        assert truncated

    def test_seeded_rng_reproducibility(self):
        """Two envs with the same seed produce identical trajectories."""
        results = []
        for _ in range(2):
            env = self._make_env()
            env.reset(seed=42)
            trajectory = []
            for _ in range(5):
                _, reward, _, _, info = env.step(0)
                trajectory.append(
                    (info["next_state"], info["human_actions"])
                )
            results.append(trajectory)
        assert results[0] == results[1]

    def test_non_contiguous_agent_indices(self):
        """Handles non-contiguous agent indices (e.g. [1, 3] human, [5] robot)."""
        wm = MockWorldModel(n_agents=6, n_actions=5)
        cfg = PPOPhase2Config(
            num_actions=5, num_robots=1, steps_per_episode=50
        )
        env = EMPOMultiGridEnv(
            world_model=wm,
            human_policy_prior=mock_human_policy_prior,
            goal_sampler=mock_goal_sampler,
            human_agent_indices=[1, 3],
            robot_agent_indices=[5],
            config=cfg,
            obs_dim=3,
        )
        env.reset(seed=0)
        # Should not raise IndexError
        _, _, _, _, info = env.step(0)
        ha = info["human_actions"]
        assert len(ha) == 6  # max(1,3,5) + 1 = 6
        # Human agents should have been assigned actions from the policy prior
        # (uniform over 5 actions, so 0-4 are all valid)
        assert 0 <= ha[1] < 5, f"Human at index 1 got invalid action {ha[1]}"
        assert 0 <= ha[3] < 5, f"Human at index 3 got invalid action {ha[3]}"

    def test_multi_robot_transition_probs(self):
        """Transition probs correctly iterate over joint actions for 2 robots."""
        wm = MockWorldModel(n_agents=4, n_actions=3)
        cfg = PPOPhase2Config(
            num_actions=3, num_robots=2, steps_per_episode=50
        )
        env = EMPOMultiGridEnv(
            world_model=wm,
            human_policy_prior=mock_human_policy_prior,
            goal_sampler=mock_goal_sampler,
            human_agent_indices=[0, 1],
            robot_agent_indices=[2, 3],
            config=cfg,
            obs_dim=3,
        )
        env.reset(seed=0)
        # Multi-robot: action is a tuple (per-robot action)
        _, _, _, _, info = env.step((0, 1))
        tp = info["transition_probs"]
        # Should have 3^2 = 9 joint actions
        assert len(tp) == 9, f"Expected 9 joint actions, got {len(tp)}"
        # Each entry should be a valid transition probability list
        for idx, transitions in tp.items():
            assert isinstance(transitions, list)
            assert len(transitions) > 0
            probs_sum = sum(p for p, _ in transitions)
            assert abs(probs_sum - 1.0) < 1e-6

    def test_multi_robot_action_space_is_multidiscrete(self):
        """Multi-robot env has MultiDiscrete action space."""
        wm = MockWorldModel(n_agents=4, n_actions=3)
        cfg = PPOPhase2Config(
            num_actions=3, num_robots=2, steps_per_episode=50
        )
        env = EMPOMultiGridEnv(
            world_model=wm,
            human_policy_prior=mock_human_policy_prior,
            goal_sampler=mock_goal_sampler,
            human_agent_indices=[0, 1],
            robot_agent_indices=[2, 3],
            config=cfg,
            obs_dim=3,
        )
        assert isinstance(env.action_space, gymnasium.spaces.MultiDiscrete)


# ======================================================================
# 4. PPOPhase2Trainer tests
# ======================================================================


class TestPPOPhase2Trainer:
    """Tests for the PPO trainer (PufferLib-backed)."""

    @staticmethod
    def _make_trainer(**config_overrides):
        defaults = dict(
            num_actions=5,
            num_robots=1,
            hidden_dim=32,
            ppo_rollout_length=16,
            ppo_num_minibatches=2,
            ppo_update_epochs=2,
            aux_buffer_size=1000,
        )
        defaults.update(config_overrides)
        cfg = PPOPhase2Config(**defaults)
        ac = EMPOActorCritic(
            state_encoder=None,
            hidden_dim=defaults["hidden_dim"],
            num_actions=defaults["num_actions"],
            obs_dim=3,
        )
        # Mock auxiliary networks
        mock_v_h_e = MagicMock(spec=["parameters", "eval", "train"])
        mock_v_h_e.parameters.return_value = iter(
            [torch.zeros(1, requires_grad=True)]
        )
        aux = PPOAuxiliaryNetworks(v_h_e=mock_v_h_e)
        return PPOPhase2Trainer(ac, aux, cfg, device="cpu")

    def test_creation(self):
        trainer = self._make_trainer()
        assert trainer.training_step_count == 0
        assert trainer.ppo_iteration == 0

    def test_push_transition_to_aux_buffer(self):
        """push_transition_to_aux_buffer stores a single transition."""
        trainer = self._make_trainer()
        trainer.push_transition_to_aux_buffer(
            state=(1, 2, 3),
            next_state=(2, 3, 4),
            robot_action=(0,),
            goals={0: "g0", 1: "g1"},
            goal_weights={0: 1.0, 1: 1.0},
            human_actions=[0, 1, 2],
            transition_probs={0: [(1.0, (2, 3, 4))]},
            terminal=False,
        )
        assert len(trainer.aux_replay_buffer) == 1
        entry = trainer.aux_replay_buffer.buffer[0]
        assert entry.state == (1, 2, 3)
        assert entry.next_state == (2, 3, 4)
        assert entry.robot_action == (0,)
        assert entry.goals == {0: "g0", 1: "g1"}
        assert entry.terminal is False

    def test_push_transition_to_aux_buffer_multi_robot(self):
        """push_transition_to_aux_buffer stores per-robot action tuple."""
        trainer = self._make_trainer(
            num_actions=3, num_robots=2, hidden_dim=32,
        )
        trainer.push_transition_to_aux_buffer(
            state=(0, 0, 0, 0),
            next_state=(1, 1, 1, 1),
            robot_action=(2, 1),
            goals={},
            goal_weights={},
            human_actions=[0, 0, 0, 0],
            transition_probs={0: [(1.0, (0, 0, 0, 0))]},
            terminal=False,
        )
        assert len(trainer.aux_replay_buffer) == 1
        entry = trainer.aux_replay_buffer.buffer[0]
        assert entry.robot_action == (2, 1)

    def test_auxiliary_training_computes_losses(self):
        """Auxiliary training step produces V_h^e loss values."""
        cfg = PPOPhase2Config(
            num_actions=5,
            num_robots=1,
            hidden_dim=16,
            batch_size=2,
            aux_buffer_size=100,
        )
        ac = EMPOActorCritic(
            state_encoder=None,
            hidden_dim=16,
            num_actions=5,
            obs_dim=3,
        )
        v_h_e = _MockVhE()
        aux = PPOAuxiliaryNetworks(v_h_e=v_h_e)
        trainer = PPOPhase2Trainer(ac, aux, cfg, device="cpu")

        # Fill buffer with enough transitions
        for i in range(5):
            trainer.aux_replay_buffer.push(
                state=(i, 0, 0),
                robot_action=(0,),
                goals={0: "goal_0", 1: "goal_1"},
                goal_weights={0: 1.0, 1: 1.0},
                human_actions=[0, 0, 0],
                next_state=(i + 1, 0, 0),
                terminal=False,
            )

        wm = MockWorldModel(n_agents=3, n_actions=5)
        losses = trainer.train_auxiliary_step(world_model=wm)
        assert "v_h_e_loss" in losses
        assert losses["v_h_e_loss"] >= 0.0

    def test_auxiliary_training_no_world_model_skips(self):
        """When world_model is None, auxiliary step returns empty."""
        cfg = PPOPhase2Config(
            num_actions=5,
            num_robots=1,
            hidden_dim=16,
            batch_size=2,
            aux_buffer_size=100,
        )
        ac = EMPOActorCritic(
            state_encoder=None,
            hidden_dim=16,
            num_actions=5,
            obs_dim=3,
        )
        v_h_e = _MockVhE()
        aux = PPOAuxiliaryNetworks(v_h_e=v_h_e)
        trainer = PPOPhase2Trainer(ac, aux, cfg, device="cpu")

        # Fill buffer
        for i in range(5):
            trainer.aux_replay_buffer.push(
                state=(i, 0, 0),
                robot_action=(0,),
                goals={0: "goal_0"},
                goal_weights={0: 1.0},
                human_actions=[0, 0, 0],
                next_state=(i + 1, 0, 0),
                terminal=False,
            )

        # No world_model → should return empty
        losses = trainer.train_auxiliary_step(world_model=None)
        assert losses == {}

    def test_freeze_auxiliary_networks(self):
        """freeze_auxiliary_networks creates frozen target copies."""
        cfg = PPOPhase2Config(
            num_actions=5, num_robots=1, hidden_dim=16,
        )
        ac = EMPOActorCritic(
            state_encoder=None, hidden_dim=16, num_actions=5, obs_dim=3,
        )
        v_h_e = _MockVhE()
        aux = PPOAuxiliaryNetworks(v_h_e=v_h_e)
        trainer = PPOPhase2Trainer(ac, aux, cfg, device="cpu")

        assert trainer.auxiliary_networks.v_h_e_target is None
        trainer.freeze_auxiliary_networks()
        assert trainer.auxiliary_networks.v_h_e_target is not None
        # Target should be a separate object (deep copy)
        assert (
            trainer.auxiliary_networks.v_h_e_target
            is not trainer.auxiliary_networks.v_h_e
        )
        # Target parameters should be frozen
        for p in trainer.auxiliary_networks.v_h_e_target.parameters():
            assert not p.requires_grad

    def test_pufferlib_training_loop(self):
        """End-to-end: PufferLib training loop completes 2 iterations."""
        cfg = PPOPhase2Config(
            num_actions=5,
            num_robots=1,
            hidden_dim=16,
            ppo_rollout_length=8,
            ppo_num_minibatches=2,
            ppo_update_epochs=1,
            num_envs=4,
            num_ppo_iterations=2,
            aux_training_steps_per_iteration=1,
            aux_buffer_size=500,
            batch_size=4,
            steps_per_episode=50,
        )
        ac = EMPOActorCritic(
            state_encoder=None,
            hidden_dim=16,
            num_actions=5,
            obs_dim=3,
        )
        v_h_e = _MockVhE()
        aux = PPOAuxiliaryNetworks(v_h_e=v_h_e)
        trainer = PPOPhase2Trainer(ac, aux, cfg, device="cpu")

        def env_creator():
            wm = MockWorldModel(n_agents=3, n_actions=5)
            return _PufferLibCompatEnv(
                world_model=wm,
                human_policy_prior=mock_human_policy_prior,
                goal_sampler=mock_goal_sampler,
                human_agent_indices=[0, 1],
                robot_agent_indices=[2],
                config=cfg,
                obs_dim=3,
            )

        # Workaround: PufferLib 3.0 bug — torch.nan is a float, so
        # torch.nan.item() raises AttributeError when var_y == 0.
        _nan = torch.nan
        torch.nan = torch.tensor(float("nan"))
        try:
            metrics = trainer.train(env_creator, num_iterations=2)
        finally:
            torch.nan = _nan

        assert len(metrics) == 2
        for m in metrics:
            assert "iteration" in m
            assert "global_step" in m


# ======================================================================
# 5. Isolation check: no existing Phase 2 files were modified
# ======================================================================


class TestIsolation:
    """Verify the PPO code doesn't modify existing Phase 2 files."""

    def test_phase2_config_unchanged(self):
        """Phase2Config should still be importable and unchanged."""
        from empo.learning_based.phase2.config import Phase2Config

        cfg = Phase2Config()
        assert hasattr(cfg, "gamma_r")
        assert hasattr(cfg, "beta_r")
        assert hasattr(cfg, "lr_q_r")
        # PPO fields should NOT be on Phase2Config
        assert not hasattr(cfg, "ppo_clip_coef")
        assert not hasattr(cfg, "ppo_rollout_length")

    def test_replay_buffer_unchanged(self):
        """Phase2ReplayBuffer push() API should be unchanged."""
        buf = Phase2ReplayBuffer(capacity=10)
        buf.push(
            state=(0,),
            robot_action=(0,),
            goals={0: "g"},
            goal_weights={0: 1.0},
            human_actions=[0],
            next_state=(1,),
        )
        assert len(buf) == 1

    def test_no_ppo_imports_in_phase2_init(self):
        """The DQN-path __init__.py should NOT import from phase2_ppo."""
        init_path = os.path.join(
            os.path.dirname(__file__),
            "..", "src", "empo", "learning_based", "phase2", "__init__.py"
        )
        # Normalise the path
        init_path = os.path.normpath(init_path)
        if os.path.exists(init_path):
            with open(init_path) as f:
                content = f.read()
            # Check for actual imports/references to the PPO module, not just
            # any mention of "ppo" (which could appear in comments/docs).
            assert "phase2_ppo" not in content, (
                "The DQN-path __init__.py should not reference phase2_ppo"
            )

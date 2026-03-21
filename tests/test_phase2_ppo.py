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
    3. EMPOWorldModelEnv — reset, step, observation / reward / info contract
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

pytest.importorskip("pufferlib")
# Side-effect imports: ensure PufferLib submodules are registered for the trainer.
import pufferlib.emulation  # noqa: E402,F401
import pufferlib.vector  # noqa: E402,F401

# ── PPO-path imports (new code under test) ──────────────────────────────
from empo.learning_based.phase2_ppo.config import PPOPhase2Config
from empo.learning_based.phase2_ppo.actor_critic import EMPOActorCritic
from empo.learning_based.phase2_ppo.env_wrapper import EMPOWorldModelEnv
from empo.learning_based.phase2_ppo.trainer import (
    PPOPhase2Trainer,
    PPOAuxiliaryNetworks,
    HAS_TENSORBOARD,
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
        new_state = tuple((s + a) % 10 for s, a in zip(self._state, joint_action))
        self._state = new_state
        terminated = self._step_count >= 50
        return 0, 0.0, terminated, False, {}

    def transition_probabilities(self, state, joint_action):
        # Deterministic: single successor with prob 1.0
        new_state = tuple((s + a) % 10 for s, a in zip(state, joint_action))
        return [(1.0, new_state)]


def mock_human_policy_prior(state, human_idx, goal, world_model):
    """Uniform random policy prior (adapts to world_model.n_actions)."""
    n = getattr(world_model, "n_actions", 5)
    return [1.0 / n] * n


def mock_goal_sampler(state, human_idx):
    """Returns a dummy goal and weight."""
    return f"goal_{human_idx}", 1.0


class _ZeroObsEnv(EMPOWorldModelEnv):
    """Test subclass that provides a trivial zero-vector observation encoder.

    ``EMPOWorldModelEnv._state_to_obs()`` deliberately raises
    ``NotImplementedError`` to prevent silent training on constant zeros.
    This subclass provides the minimal override needed for tests.
    """

    def _state_to_obs(self, state):
        return np.zeros(self.observation_space.shape, dtype=np.float32)


class _PufferLibCompatEnv(_ZeroObsEnv):
    """Testing shim that asserts a numeric-only top-level ``info`` dict.

    ``EMPOWorldModelEnv.step()`` already follows the contract that all
    top-level ``info`` values are scalar numerics, and stores any richer
    auxiliary data in its internal ``_aux_buffer`` attribute instead.

    PufferLib's Serial vectorisation backend calls ``np.mean()`` on every
    top-level ``info`` value, which would break on non-numeric data.  This
    wrapper **asserts** that the contract holds (failing the test if a
    regression reintroduces complex objects in ``info``), rather than
    silently filtering non-numeric entries.

    Trainers should continue to read rich auxiliary data from ``_aux_buffer``
    after ``evaluate()``; this wrapper does not modify that behaviour.
    """

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        for key, value in info.items():
            assert isinstance(value, (int, float, np.integer, np.floating)), (
                f"EMPOWorldModelEnv.step() returned non-numeric info "
                f"value for key {key!r}: {type(value).__name__}. "
                f"Top-level info must contain only scalar numerics; "
                f"rich auxiliary data belongs in _aux_buffer."
            )
        return obs, reward, terminated, truncated, info


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
        cfg = PPOPhase2Config(gamma_r=0.95, num_actions=3, num_robots=2)
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
            device="cpu",
            seed=42,
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
        # device and seed should come from config fields, not be hardcoded
        assert d["device"] == "cpu"
        assert d["seed"] == 42
        # Should contain all keys expected by PuffeRL
        for key in [
            "seed",
            "total_timesteps",
            "compile",
            "use_rnn",
            "device",
            "anneal_lr",
        ]:
            assert key in d, f"Missing PufferLib key: {key}"
        # Verify critical defaults are sensible
        assert d["device"] == "cpu"
        assert d["use_rnn"] is False
        assert isinstance(d["seed"], int)

    def test_warmup_stage_boundaries(self):
        """get_warmup_stage() returns correct stage at boundary values."""
        cfg = PPOPhase2Config(
            warmup_v_h_e_steps=100,
            warmup_x_h_steps=200,
            warmup_u_r_steps=300,
        )
        assert cfg.get_warmup_stage(0) == 0
        assert cfg.get_warmup_stage(99) == 0
        assert cfg.get_warmup_stage(100) == 1
        assert cfg.get_warmup_stage(199) == 1
        assert cfg.get_warmup_stage(200) == 2
        assert cfg.get_warmup_stage(299) == 2
        assert cfg.get_warmup_stage(300) == 3
        assert cfg.get_warmup_stage(1000) == 3

    def test_active_aux_networks(self):
        """get_active_aux_networks() returns the correct set per stage."""
        cfg = PPOPhase2Config(
            warmup_v_h_e_steps=10,
            warmup_x_h_steps=20,
            warmup_u_r_steps=30,
        )
        assert cfg.get_active_aux_networks(0) == {"v_h_e"}
        assert cfg.get_active_aux_networks(10) == {"v_h_e", "x_h"}
        assert cfg.get_active_aux_networks(20) == {"v_h_e", "x_h", "u_r"}
        assert cfg.get_active_aux_networks(30) == {"v_h_e", "x_h", "u_r"}

    def test_is_in_warmup(self):
        cfg = PPOPhase2Config(
            warmup_v_h_e_steps=30,
            warmup_x_h_steps=60,
            warmup_u_r_steps=100,
        )
        assert cfg.is_in_warmup(0) is True
        assert cfg.is_in_warmup(99) is True
        assert cfg.is_in_warmup(100) is False

    def test_warmup_stage_name(self):
        cfg = PPOPhase2Config(
            warmup_v_h_e_steps=10,
            warmup_x_h_steps=20,
            warmup_u_r_steps=30,
        )
        assert "V_h^e" in cfg.get_warmup_stage_name(0)
        assert "X_h" in cfg.get_warmup_stage_name(15)
        assert "U_r" in cfg.get_warmup_stage_name(25)
        assert "PPO" in cfg.get_warmup_stage_name(30).upper() or "full" in cfg.get_warmup_stage_name(30)

    def test_warmup_threshold_validation(self):
        """warmup thresholds must be non-decreasing."""
        with pytest.raises(ValueError, match="non-decreasing"):
            PPOPhase2Config(
                warmup_v_h_e_steps=200,
                warmup_x_h_steps=100,
                warmup_u_r_steps=300,
            )

    def test_zero_warmup(self):
        """All warmup steps=0 means no warm-up (stage 3 immediately)."""
        cfg = PPOPhase2Config(
            warmup_v_h_e_steps=0,
            warmup_x_h_steps=0,
            warmup_u_r_steps=0,
        )
        assert cfg.is_in_warmup(0) is False
        assert cfg.get_warmup_stage(0) == 3
        assert cfg.get_total_warmup_steps() == 0


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
# 3. EMPOWorldModelEnv tests
# ======================================================================


class TestEMPOWorldModelEnv:
    """Tests for the Gymnasium-compatible environment wrapper."""

    def _make_env(self, config=None, **kwargs):
        wm = MockWorldModel(n_agents=3, n_actions=5)
        cfg = config or PPOPhase2Config(
            num_actions=5, num_robots=1, steps_per_episode=50
        )
        return _ZeroObsEnv(
            world_model=wm,
            human_policy_prior=mock_human_policy_prior,
            goal_sampler=mock_goal_sampler,
            human_agent_indices=[0, 1],
            robot_agent_indices=[2],
            config=cfg,
            obs_dim=3,
            **kwargs,
        )

    def test_base_state_to_obs_raises(self):
        """Base EMPOWorldModelEnv._state_to_obs() raises NotImplementedError."""
        wm = MockWorldModel(n_agents=3, n_actions=5)
        cfg = PPOPhase2Config(num_actions=5, num_robots=1, steps_per_episode=50)
        env = EMPOWorldModelEnv(
            world_model=wm,
            human_policy_prior=mock_human_policy_prior,
            goal_sampler=mock_goal_sampler,
            human_agent_indices=[0, 1],
            robot_agent_indices=[2],
            config=cfg,
            obs_dim=3,
        )
        with pytest.raises(NotImplementedError):
            env.reset()

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
        """info has only numeric fields; rich data is in _aux_buffer."""
        env = self._make_env()
        env.reset()
        _, _, _, _, info = env.step(2)
        # info must only contain scalar-numeric keys for PufferLib compat.
        assert "u_r" in info
        assert "env_reward" in info
        # Rich auxiliary data lives in the internal _aux_buffer.
        assert len(env._aux_buffer) == 1
        aux = env._aux_buffer[0]
        assert "state" in aux
        assert "next_state" in aux
        assert "goals" in aux
        assert "goal_weights" in aux
        assert "human_actions" in aux
        assert "terminal" in aux
        # robot_action should be a per-robot tuple
        assert isinstance(aux["robot_action"], tuple)

    def test_transition_probs_per_robot_action(self):
        cfg = PPOPhase2Config(
            num_actions=5,
            num_robots=1,
            steps_per_episode=50,
            compute_transition_probs=True,
        )
        env = self._make_env(config=cfg)
        env.reset()
        _, _, _, _, _ = env.step(0)
        aux = env._aux_buffer[0]
        tp = aux["transition_probs"]
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
            _, _, terminated, truncated, _ = env.step(env.action_space.sample())
            if terminated or truncated:
                break
        assert terminated  # MockWorldModel terminates after 50 steps

    def test_episode_truncates_at_steps_per_episode(self):
        """Episode is truncated when step count reaches steps_per_episode."""
        wm = MockWorldModel(n_agents=3, n_actions=5)
        cfg = PPOPhase2Config(num_actions=5, num_robots=1, steps_per_episode=10)
        env = _ZeroObsEnv(
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
                aux = env._aux_buffer[-1]
                trajectory.append((aux["next_state"], aux["human_actions"]))
            results.append(trajectory)
        assert results[0] == results[1]

    def test_non_contiguous_agent_indices(self):
        """Handles non-contiguous agent indices (e.g. [1, 3] human, [5] robot)."""
        wm = MockWorldModel(n_agents=6, n_actions=5)
        cfg = PPOPhase2Config(num_actions=5, num_robots=1, steps_per_episode=50)
        env = _ZeroObsEnv(
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
        aux = env._aux_buffer[0]
        ha = aux["human_actions"]
        assert len(ha) == 6  # max(1,3,5) + 1 = 6
        # Human agents should have been assigned actions from the policy prior
        # (uniform over 5 actions, so 0-4 are all valid)
        assert 0 <= ha[1] < 5, f"Human at index 1 got invalid action {ha[1]}"
        assert 0 <= ha[3] < 5, f"Human at index 3 got invalid action {ha[3]}"

    def test_multi_robot_transition_probs(self):
        """Transition probs correctly iterate over joint actions for 2 robots."""
        wm = MockWorldModel(n_agents=4, n_actions=3)
        cfg = PPOPhase2Config(
            num_actions=3,
            num_robots=2,
            steps_per_episode=50,
            compute_transition_probs=True,
        )
        env = _ZeroObsEnv(
            world_model=wm,
            human_policy_prior=mock_human_policy_prior,
            goal_sampler=mock_goal_sampler,
            human_agent_indices=[0, 1],
            robot_agent_indices=[2, 3],
            config=cfg,
            obs_dim=3,
        )
        env.reset(seed=0)
        # (0, 1) → 0 + 1*3 = 3 in the flat index
        flat_action = 0 + 1 * 3  # action_tuple_to_index((0, 1))
        _, _, _, _, _ = env.step(flat_action)
        aux = env._aux_buffer[0]
        tp = aux["transition_probs"]
        # Should have 3^2 = 9 joint actions
        assert len(tp) == 9, f"Expected 9 joint actions, got {len(tp)}"
        # Each entry should be a valid transition probability list
        for idx, transitions in tp.items():
            assert isinstance(transitions, list)
            assert len(transitions) > 0
            probs_sum = sum(p for p, _ in transitions)
            assert abs(probs_sum - 1.0) < 1e-6

    def test_multi_robot_action_space_is_flat_discrete(self):
        """Multi-robot env has flat Discrete(num_actions**num_robots) action space."""
        wm = MockWorldModel(n_agents=4, n_actions=3)
        cfg = PPOPhase2Config(num_actions=3, num_robots=2, steps_per_episode=50)
        env = EMPOWorldModelEnv(
            world_model=wm,
            human_policy_prior=mock_human_policy_prior,
            goal_sampler=mock_goal_sampler,
            human_agent_indices=[0, 1],
            robot_agent_indices=[2, 3],
            config=cfg,
            obs_dim=3,
        )
        assert isinstance(env.action_space, gymnasium.spaces.Discrete)
        assert env.action_space.n == 3**2  # num_actions ** num_robots = 9


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
        mock_v_h_e.parameters.return_value = iter([torch.zeros(1, requires_grad=True)])
        aux = PPOAuxiliaryNetworks(v_h_e=mock_v_h_e)
        return PPOPhase2Trainer(ac, aux, cfg, device="cpu")

    def test_creation(self):
        trainer = self._make_trainer()
        assert trainer.global_env_step == 0
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
            num_actions=3,
            num_robots=2,
            hidden_dim=32,
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
            num_actions=5,
            num_robots=1,
            hidden_dim=16,
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
            # Skip warm-up for this test (tested separately)
            warmup_v_h_e_steps=0,
            warmup_x_h_steps=0,
            warmup_u_r_steps=0,
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
        import pytest

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(torch, "nan", torch.tensor(float("nan")))
            metrics = trainer.train(env_creator, num_iterations=2)

        assert len(metrics) == 2
        for m in metrics:
            assert "iteration" in m
            assert "global_step" in m

    def test_warmup_runs_before_ppo(self):
        """Warm-up produces metrics with warmup_stage before PPO metrics."""
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
            # Short warm-up so test is fast.
            warmup_v_h_e_steps=10,
            warmup_x_h_steps=10,
            warmup_u_r_steps=20,
        )
        ac = EMPOActorCritic(
            state_encoder=None, hidden_dim=16, num_actions=5, obs_dim=3,
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

        import pytest as _pytest

        with _pytest.MonkeyPatch.context() as mp:
            mp.setattr(torch, "nan", torch.tensor(float("nan")))
            metrics = trainer.train(env_creator, num_iterations=2)

        # Warm-up metrics have a warmup_stage key; PPO metrics have iteration
        warmup_metrics = [m for m in metrics if "warmup_stage" in m]
        ppo_metrics = [m for m in metrics if "iteration" in m]
        assert len(warmup_metrics) > 0, "Expected warm-up metrics"
        assert len(ppo_metrics) == 2, "Expected 2 PPO iteration metrics"
        # aux_training_step should have advanced beyond warm-up
        assert trainer.aux_training_step >= 20

    def test_active_networks_filter(self):
        """train_auxiliary_step with active_networks filters which nets train."""
        cfg = PPOPhase2Config(
            num_actions=5, num_robots=1, hidden_dim=16,
            batch_size=2, aux_buffer_size=100,
        )
        v_h_e = _MockVhE()
        from empo.learning_based.phase2.aggregate_goal_ability import (
            BaseAggregateGoalAbilityNetwork,
        )

        class _MockXh(BaseAggregateGoalAbilityNetwork):
            def __init__(self):
                super().__init__(zeta=2.0)
                self._linear = torch.nn.Linear(1, 1)

            def forward(self, state, world_model, human_agent_idx, device="cpu"):
                return self.apply_hard_clamp(
                    self._linear(torch.zeros(1, device=device))
                )

            def get_config(self):
                return {"type": "mock_x_h"}

        x_h = _MockXh()
        ac = EMPOActorCritic(
            state_encoder=None, hidden_dim=16, num_actions=5, obs_dim=3,
        )
        aux = PPOAuxiliaryNetworks(v_h_e=v_h_e, x_h=x_h)
        trainer = PPOPhase2Trainer(ac, aux, cfg, device="cpu")

        # Populate the buffer with transitions
        for _ in range(5):
            trainer.push_transition_to_aux_buffer(
                state=(0,), next_state=(1,), robot_action=(0,),
                goals={0: "g"}, goal_weights={0: 1.0},
                human_actions=[0], transition_probs=None, terminal=False,
            )

        # With active_networks={"v_h_e"}, only v_h_e should be trained
        losses = trainer.train_auxiliary_step(
            world_model=None, active_networks={"v_h_e"}
        )
        assert "x_h_loss" not in losses

    def test_checkpoint_save_load(self):
        """save_checkpoint / load_checkpoint roundtrips trainer state."""
        import tempfile

        cfg = PPOPhase2Config(
            num_actions=5, num_robots=1, hidden_dim=16,
            warmup_v_h_e_steps=0, warmup_x_h_steps=0, warmup_u_r_steps=0,
        )
        ac = EMPOActorCritic(
            state_encoder=None, hidden_dim=16, num_actions=5, obs_dim=3,
        )
        v_h_e = _MockVhE()
        aux = PPOAuxiliaryNetworks(v_h_e=v_h_e)
        trainer = PPOPhase2Trainer(ac, aux, cfg, device="cpu")
        trainer.global_env_step = 42
        trainer.ppo_iteration = 7
        trainer.aux_training_step = 99

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            ckpt_path = f.name

        try:
            saved_path = trainer.save_checkpoint(ckpt_path)
            assert os.path.isfile(saved_path)

            # Create a fresh trainer and load
            ac2 = EMPOActorCritic(
                state_encoder=None, hidden_dim=16, num_actions=5, obs_dim=3,
            )
            v_h_e2 = _MockVhE()
            aux2 = PPOAuxiliaryNetworks(v_h_e=v_h_e2)
            trainer2 = PPOPhase2Trainer(ac2, aux2, cfg, device="cpu")
            trainer2.load_checkpoint(saved_path)

            assert trainer2.global_env_step == 42
            assert trainer2.ppo_iteration == 7
            assert trainer2.aux_training_step == 99
        finally:
            if os.path.isfile(ckpt_path):
                os.unlink(ckpt_path)

    def test_tensorboard_writer_initialised(self):
        """writer is initialised when tensorboard_dir is set."""
        import tempfile

        cfg = PPOPhase2Config(
            num_actions=5, num_robots=1, hidden_dim=16,
        )
        ac = EMPOActorCritic(
            state_encoder=None, hidden_dim=16, num_actions=5, obs_dim=3,
        )
        v_h_e = _MockVhE()
        aux = PPOAuxiliaryNetworks(v_h_e=v_h_e)
        trainer = PPOPhase2Trainer(ac, aux, cfg, device="cpu")
        assert trainer.writer is None  # before init

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer.config.tensorboard_dir = tmpdir
            trainer._init_tensorboard()
            if HAS_TENSORBOARD:
                assert trainer.writer is not None
                trainer.writer.close()
            else:
                assert trainer.writer is None


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
            "..",
            "src",
            "empo",
            "learning_based",
            "phase2",
            "__init__.py",
        )
        # Normalise the path
        init_path = os.path.normpath(init_path)
        if os.path.exists(init_path):
            with open(init_path) as f:
                content = f.read()
            # Check for actual imports/references to the PPO module, not just
            # any mention of "ppo" (which could appear in comments/docs).
            assert (
                "phase2_ppo" not in content
            ), "The DQN-path __init__.py should not reference phase2_ppo"

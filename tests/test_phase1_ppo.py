"""
Tests for the PPO-based Phase 1 implementation (PufferLib-backed).

These tests verify the **new** PPO code path without touching or importing
anything from the existing DQN-based Phase 1 trainer.  The only shared
import is the stable ``learning_based/phase1/__init__.py`` for the
isolation check.

Test categories:
    1. PPOPhase1Config — creation, validation, helpers, PufferLib config
    2. GoalConditionedActorCritic — forward (PufferLib convention),
       encode/decode, get_action_and_value, get_value
    3. Phase1PPOEnv — reset, step, observation / reward / info contract
    4. PPOPhase1Trainer — creation, PufferLib training loop,
       checkpoint save/load
    5. Isolation check — no existing Phase 1 files modified
"""

import os
import tempfile

import numpy as np
import pytest
import torch
import gymnasium

pytest.importorskip("pufferlib")
# Side-effect imports: ensure PufferLib submodules are registered for the trainer.
import pufferlib.emulation  # noqa: E402,F401
import pufferlib.vector  # noqa: E402,F401

# ── PPO-path imports (new code under test) ──────────────────────────────
from empo.learning_based.phase1_ppo.config import PPOPhase1Config
from empo.learning_based.phase1_ppo.actor_critic import GoalConditionedActorCritic
from empo.learning_based.phase1_ppo.env_wrapper import Phase1PPOEnv
from empo.learning_based.phase1_ppo.trainer import PPOPhase1Trainer

# ======================================================================
# Fixtures & mocks
# ======================================================================


class _DummyGoal:
    """Minimal goal: achieved when the first element of state >= 5."""

    def is_achieved(self, state):
        return 1 if state[0] >= 5 else 0

    def __hash__(self):
        return 42

    def __eq__(self, other):
        return isinstance(other, _DummyGoal)


class MockWorldModel:
    """Minimal WorldModel mock for testing the PPO env wrapper."""

    def __init__(self, n_agents: int = 2, n_actions: int = 4):
        self.n_agents = n_agents
        self.n_actions = n_actions
        self._state = (0,) * n_agents  # positions
        self._step_count = 0

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


def mock_goal_sampler(state, human_idx):
    """Returns a dummy goal and weight."""
    return _DummyGoal(), 1.0


def mock_other_policy(state, agent_idx):
    """Returns a fixed action (0) for simplicity."""
    return 0


class _ZeroObsEnv(Phase1PPOEnv):
    """Test subclass that provides a trivial zero-vector observation encoder.

    ``Phase1PPOEnv._state_to_obs()`` deliberately raises
    ``NotImplementedError`` to prevent silent training on constant zeros.
    This subclass provides the minimal override needed for tests.
    """

    def _state_to_obs(self, state, goal):
        return np.zeros(self.observation_space.shape, dtype=np.float32)


class _PufferLibCompatEnv(_ZeroObsEnv):
    """Testing shim that asserts a numeric-only top-level ``info`` dict.

    PufferLib's Serial vectorisation backend calls ``np.mean()`` on every
    top-level ``info`` value, which would break on non-numeric data.  This
    wrapper **asserts** that the contract holds (failing the test if a
    regression reintroduces complex objects in ``info``).
    """

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        for key, value in info.items():
            assert isinstance(value, (int, float, np.integer, np.floating)), (
                f"Phase1PPOEnv.step() returned non-numeric info "
                f"value for key {key!r}: {type(value).__name__}. "
                f"Top-level info must contain only scalar numerics."
            )
        return obs, reward, terminated, truncated, info


# ======================================================================
# 1. PPOPhase1Config tests
# ======================================================================


class TestPPOPhase1Config:
    """Tests for the standalone PPO Phase 1 config class."""

    def test_default_creation(self):
        cfg = PPOPhase1Config()
        assert cfg.gamma_h == 0.99
        assert cfg.beta_h == 1.0
        assert cfg.ppo_clip_coef == 0.2
        assert cfg.ppo_vf_coef == 0.5
        assert cfg.ppo_max_grad_norm == 0.5
        assert cfg.ppo_gae_lambda == 0.95
        assert cfg.lr == 3e-4
        assert cfg.num_envs == 16
        assert cfg.num_ppo_iterations == 10_000
        assert cfg.ppo_ent_coef_start == 0.1
        assert cfg.ppo_ent_coef_end == 0.01
        assert cfg.hidden_dim == 256
        assert cfg.steps_per_episode == 50
        assert cfg.num_actions == 7
        assert cfg.device == "cpu"
        assert cfg.seed == 1
        assert cfg.log_interval == 1
        assert cfg.reward_shaping_coef == 0.0

    def test_custom_values(self):
        cfg = PPOPhase1Config(
            gamma_h=0.95,
            beta_h=2.0,
            num_actions=3,
            hidden_dim=64,
            lr=1e-3,
            num_envs=8,
        )
        assert cfg.gamma_h == 0.95
        assert cfg.beta_h == 2.0
        assert cfg.num_actions == 3
        assert cfg.hidden_dim == 64
        assert cfg.lr == 1e-3
        assert cfg.num_envs == 8

    def test_gamma_h_validation(self):
        with pytest.raises(ValueError, match="gamma_h"):
            PPOPhase1Config(gamma_h=-0.1)
        with pytest.raises(ValueError, match="gamma_h"):
            PPOPhase1Config(gamma_h=1.5)

    def test_beta_h_validation(self):
        with pytest.raises(ValueError, match="beta_h"):
            PPOPhase1Config(beta_h=0.0)
        with pytest.raises(ValueError, match="beta_h"):
            PPOPhase1Config(beta_h=-1.0)

    def test_log_interval_validation(self):
        with pytest.raises(ValueError, match="log_interval"):
            PPOPhase1Config(log_interval=0)
        with pytest.raises(ValueError, match="log_interval"):
            PPOPhase1Config(log_interval=-1)

    def test_entropy_schedule(self):
        cfg = PPOPhase1Config(
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

    def test_to_pufferlib_config(self):
        """to_pufferlib_config() produces a valid PufferLib config dict."""
        cfg = PPOPhase1Config(
            num_envs=4,
            ppo_rollout_length=32,
            ppo_num_minibatches=2,
            ppo_update_epochs=3,
            gamma_h=0.97,
            ppo_gae_lambda=0.9,
            ppo_clip_coef=0.1,
            ppo_vf_coef=0.25,
            ppo_max_grad_norm=1.0,
            lr=1e-3,
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
        assert d["use_rnn"] is False
        assert isinstance(d["seed"], int)


# ======================================================================
# 2. GoalConditionedActorCritic tests
# ======================================================================


class TestGoalConditionedActorCritic:
    """Tests for the goal-conditioned actor-critic network."""

    def test_forward_returns_pufferlib_convention(self):
        """forward() returns (logits, value) with PufferLib-expected shapes."""
        ac = GoalConditionedActorCritic(
            obs_dim=10,
            hidden_dim=32,
            num_actions=6,
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
        ac = GoalConditionedActorCritic(
            obs_dim=8,
            hidden_dim=16,
            num_actions=4,
        )
        obs = torch.randn(2, 8)
        assert hasattr(ac, "forward_eval"), "forward_eval method missing"

        logits_fwd, value_fwd = ac.forward(obs)
        logits_eval, value_eval = ac.forward_eval(obs)

        assert torch.equal(logits_fwd, logits_eval)
        assert torch.equal(value_fwd, value_eval)

    def test_encode_decode_roundtrip(self):
        """encode_observations + decode_actions matches forward."""
        ac = GoalConditionedActorCritic(
            obs_dim=8,
            hidden_dim=32,
            num_actions=5,
        )
        obs = torch.randn(4, 8)
        logits_fwd, value_fwd = ac.forward(obs)

        hidden = ac.encode_observations(obs)
        logits_dec, value_dec = ac.decode_actions(hidden)

        assert torch.equal(logits_fwd, logits_dec)
        assert torch.equal(value_fwd, value_dec)

    def test_get_action_and_value(self):
        """get_action_and_value returns (action, log_prob, entropy, value)."""
        ac = GoalConditionedActorCritic(
            obs_dim=8,
            hidden_dim=16,
            num_actions=4,
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
        """Providing an action returns that exact action."""
        ac = GoalConditionedActorCritic(
            obs_dim=8,
            hidden_dim=16,
            num_actions=4,
        )
        obs = torch.randn(3, 8)
        fixed_action = torch.tensor([0, 1, 2])
        action, lp, ent, val = ac.get_action_and_value(obs, fixed_action)
        assert torch.equal(action, fixed_action)

    def test_get_value(self):
        """get_value returns scalar value per batch element."""
        ac = GoalConditionedActorCritic(
            obs_dim=8,
            hidden_dim=16,
            num_actions=4,
        )
        obs = torch.randn(2, 8)
        val = ac.get_value(obs)
        assert val.shape == (2,)


# ======================================================================
# 3. Phase1PPOEnv tests
# ======================================================================


class TestPhase1PPOEnv:
    """Tests for the Gymnasium-compatible Phase 1 environment wrapper."""

    def _make_env(self, config=None, **kwargs):
        wm = MockWorldModel(n_agents=2, n_actions=4)
        cfg = config or PPOPhase1Config(
            num_actions=4, steps_per_episode=50
        )
        return _ZeroObsEnv(
            world_model=wm,
            goal_sampler=mock_goal_sampler,
            training_human_index=0,
            other_agent_policies={1: mock_other_policy},
            config=cfg,
            obs_dim=3,
            **kwargs,
        )

    def test_base_state_to_obs_raises(self):
        """Base Phase1PPOEnv._state_to_obs() raises NotImplementedError
        when no callback is provided."""
        wm = MockWorldModel(n_agents=2, n_actions=4)
        cfg = PPOPhase1Config(num_actions=4, steps_per_episode=50)
        env = Phase1PPOEnv(
            world_model=wm,
            goal_sampler=mock_goal_sampler,
            training_human_index=0,
            other_agent_policies={1: mock_other_policy},
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
        assert isinstance(info, dict)

    def test_step_returns_five_tuple(self):
        env = self._make_env()
        env.reset()
        obs, reward, terminated, truncated, info = env.step(0)
        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_reward_is_goal_achievement(self):
        """Reward is 0 or 1 based on goal.is_achieved(state)."""
        env = self._make_env()
        env.reset()
        # With default MockWorldModel starting at (0,0), goal needs state[0] >= 5
        _, reward, _, _, _ = env.step(0)
        assert reward == 0.0  # state[0] = 0, not >= 5

        # Force state so goal is achieved
        env.world_model._state = (5, 0)
        _, reward, _, _, _ = env.step(0)
        # After step, state[0] = (5 + 0) % 10 = 5, goal achieved
        assert reward == 1.0

    def test_action_space_discrete(self):
        env = self._make_env()
        assert isinstance(env.action_space, gymnasium.spaces.Discrete)
        assert env.action_space.n == 4

    def test_episode_truncates_at_steps_per_episode(self):
        """Episode is truncated when step count reaches steps_per_episode."""
        wm = MockWorldModel(n_agents=2, n_actions=4)
        cfg = PPOPhase1Config(num_actions=4, steps_per_episode=10)
        env = _ZeroObsEnv(
            world_model=wm,
            goal_sampler=mock_goal_sampler,
            training_human_index=0,
            other_agent_policies={1: mock_other_policy},
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

    def test_goal_achieved_in_info(self):
        """info['goal_achieved'] matches the reward."""
        env = self._make_env()
        env.reset()
        _, reward, _, _, info = env.step(0)
        assert "goal_achieved" in info
        assert info["goal_achieved"] == reward

    def test_seeded_rng_reproducibility(self):
        """Two envs with the same seed produce identical trajectories."""
        results = []
        for _ in range(2):
            env = self._make_env()
            env.reset(seed=42)
            trajectory = []
            for _ in range(5):
                obs, reward, _, _, info = env.step(0)
                trajectory.append((obs.tolist(), reward))
            results.append(trajectory)
        assert results[0] == results[1]


# ======================================================================
# 4. PPOPhase1Trainer tests
# ======================================================================


class TestPPOPhase1Trainer:
    """Tests for the PPO Phase 1 trainer (PufferLib-backed)."""

    @staticmethod
    def _make_trainer(**config_overrides):
        defaults = dict(
            num_actions=4,
            hidden_dim=16,
            ppo_rollout_length=8,
            ppo_num_minibatches=2,
            ppo_update_epochs=1,
            num_envs=4,
            num_ppo_iterations=2,
            steps_per_episode=50,
        )
        defaults.update(config_overrides)
        cfg = PPOPhase1Config(**defaults)
        ac = GoalConditionedActorCritic(
            obs_dim=3,
            hidden_dim=defaults["hidden_dim"],
            num_actions=defaults["num_actions"],
        )
        return PPOPhase1Trainer(ac, cfg, device="cpu")

    def test_creation(self):
        trainer = self._make_trainer()
        assert trainer.global_env_step == 0
        assert trainer.ppo_iteration == 0

    def test_pufferlib_training_loop(self):
        """End-to-end: PufferLib training loop completes 2 iterations."""
        cfg = PPOPhase1Config(
            num_actions=4,
            hidden_dim=16,
            ppo_rollout_length=8,
            ppo_num_minibatches=2,
            ppo_update_epochs=1,
            num_envs=4,
            num_ppo_iterations=2,
            steps_per_episode=50,
        )
        ac = GoalConditionedActorCritic(
            obs_dim=3,
            hidden_dim=16,
            num_actions=4,
        )
        trainer = PPOPhase1Trainer(ac, cfg, device="cpu")

        def env_creator():
            wm = MockWorldModel(n_agents=2, n_actions=4)
            return _PufferLibCompatEnv(
                world_model=wm,
                goal_sampler=mock_goal_sampler,
                training_human_index=0,
                other_agent_policies={1: mock_other_policy},
                config=cfg,
                obs_dim=3,
            )

        # Workaround: PufferLib 3.0 bug — torch.nan is a float, so
        # torch.nan.item() raises AttributeError when var_y == 0.
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(torch, "nan", torch.tensor(float("nan")))
            metrics = trainer.train(env_creator, num_iterations=2)

        assert len(metrics) == 2
        for m in metrics:
            assert "iteration" in m
            assert "global_env_step" in m

    def test_checkpoint_save_load(self):
        """save_checkpoint / load_checkpoint roundtrips trainer state."""
        cfg = PPOPhase1Config(
            num_actions=4,
            hidden_dim=16,
        )
        ac = GoalConditionedActorCritic(
            obs_dim=3,
            hidden_dim=16,
            num_actions=4,
        )
        trainer = PPOPhase1Trainer(ac, cfg, device="cpu")
        trainer.global_env_step = 42
        trainer.ppo_iteration = 7

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            ckpt_path = f.name

        try:
            saved_path = trainer.save_checkpoint(ckpt_path)
            assert os.path.isfile(saved_path)

            # Create a fresh trainer and load
            ac2 = GoalConditionedActorCritic(
                obs_dim=3,
                hidden_dim=16,
                num_actions=4,
            )
            trainer2 = PPOPhase1Trainer(ac2, cfg, device="cpu")
            trainer2.load_checkpoint(saved_path)

            assert trainer2.global_env_step == 42
            assert trainer2.ppo_iteration == 7
        finally:
            if os.path.isfile(ckpt_path):
                os.unlink(ckpt_path)


# ======================================================================
# 5. Isolation check: no existing Phase 1 files were modified
# ======================================================================


class TestIsolation:
    """Verify the PPO code doesn't modify existing Phase 1 files."""

    def test_no_ppo_imports_in_phase1_init(self):
        """The DQN-path __init__.py should NOT import from phase1_ppo."""
        init_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "src",
            "empo",
            "learning_based",
            "phase1",
            "__init__.py",
        )
        init_path = os.path.normpath(init_path)
        if os.path.exists(init_path):
            with open(init_path) as f:
                content = f.read()
            # Check for actual imports/references to the PPO module, not just
            # any mention of "ppo" (which could appear in comments/docs).
            assert (
                "phase1_ppo" not in content
            ), "The DQN-path __init__.py should not reference phase1_ppo"

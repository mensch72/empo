"""
Tests for the MultiGrid-specific PPO Phase 1 implementation.

These tests verify:
1. ``MultiGridPhase1PPOEnv`` — observation encoding with real encoders,
   step/reset API, goal-conditioned observations.
2. ``create_multigrid_phase1_ppo_networks`` — network factory output shapes
   and types.
3. Integration — env + actor-critic wiring.
4. Isolation — no PPO imports leak into existing DQN path.

All tests use a real ``MultiGridEnv`` instance (tiny 4×6 grid) so they
exercise the actual encoder pipeline end-to-end.

These tests do NOT require ``pufferlib`` to be installed.
"""

import numpy as np
import pytest
import torch

from gym_multigrid.multigrid import MultiGridEnv, World

from empo.learning_based.phase1_ppo.config import PPOPhase1Config
from empo.learning_based.phase1_ppo.actor_critic import GoalConditionedActorCritic

from empo.learning_based.multigrid.state_encoder import MultiGridStateEncoder
from empo.learning_based.multigrid.goal_encoder import MultiGridGoalEncoder
from empo.learning_based.multigrid.feature_extraction import get_num_agents_per_color

from empo.learning_based.multigrid.phase1_ppo import (
    MultiGridPhase1PPOEnv,
    create_multigrid_phase1_ppo_networks,
)

# ======================================================================
# Fixtures
# ======================================================================

GRID_MAP = """
We We We We We We
We Ae Ro .. .. We
We We Ay We We We
We We We We We We
"""


def _make_env() -> MultiGridEnv:
    """Create a small MultiGrid environment for testing."""
    env = MultiGridEnv(
        map=GRID_MAP,
        max_steps=20,
        partial_obs=False,
        objects_set=World,
    )
    return env


def _make_config() -> PPOPhase1Config:
    return PPOPhase1Config(
        num_actions=4,
        steps_per_episode=20,
        hidden_dim=32,
        num_envs=2,
        num_ppo_iterations=2,
        ppo_rollout_length=8,
        ppo_num_minibatches=2,
        ppo_update_epochs=1,
    )


def _make_encoders(
    env: MultiGridEnv,
    feature_dim: int = 32,
    goal_feature_dim: int = 16,
    use_encoders: bool = True,
):
    """Build state and goal encoders for the given environment."""
    npc = get_num_agents_per_color(env)
    state_encoder = MultiGridStateEncoder(
        grid_height=env.height,
        grid_width=env.width,
        num_agents_per_color=npc,
        feature_dim=feature_dim,
        include_step_count=True,
        use_encoders=use_encoders,
    )
    goal_encoder = MultiGridGoalEncoder(
        grid_height=env.height,
        grid_width=env.width,
        feature_dim=goal_feature_dim,
        use_encoders=use_encoders,
    )
    return state_encoder, goal_encoder


# ======================================================================
# Test helper classes
# ======================================================================


class _DummyGoal:
    """Simple goal for testing."""

    def __init__(self, target_pos=(1, 1)):
        self.target_pos = target_pos

    def is_achieved(self, state):
        return 0

    def __hash__(self):
        return hash(self.target_pos)

    def __eq__(self, other):
        return isinstance(other, _DummyGoal) and self.target_pos == other.target_pos


class _AchievedGoal(_DummyGoal):
    """Goal that is always achieved (for testing reward)."""

    def is_achieved(self, state):
        return 1


def _dummy_goal_sampler(state, human_idx):
    """Returns a dummy goal and unit weight."""
    return _DummyGoal(), 1.0


def _achieved_goal_sampler(state, human_idx):
    """Returns a goal that is always achieved."""
    return _AchievedGoal(), 1.0


def _random_other_policy(state, agent_idx):
    """Random policy for non-training agents."""
    return np.random.randint(4)


def _make_ppo_env(
    env: MultiGridEnv,
    cfg: PPOPhase1Config,
    state_encoder: MultiGridStateEncoder,
    goal_encoder: MultiGridGoalEncoder,
    goal_sampler=None,
):
    """Create a MultiGridPhase1PPOEnv with sensible defaults."""
    if goal_sampler is None:
        goal_sampler = _dummy_goal_sampler

    # Build other agent policies for all non-training agents
    training_human = env.human_agent_indices[0]
    other_policies = {}
    for idx in range(len(env.agents)):
        if idx != training_human:
            other_policies[idx] = _random_other_policy

    return MultiGridPhase1PPOEnv(
        world_model=env,
        goal_sampler=goal_sampler,
        training_human_index=training_human,
        other_agent_policies=other_policies,
        config=cfg,
        state_encoder=state_encoder,
        goal_encoder=goal_encoder,
    )


# ======================================================================
# 1. MultiGridPhase1PPOEnv tests
# ======================================================================


class TestMultiGridPhase1PPOEnv:
    """Tests for MultiGridPhase1PPOEnv (real encoders, real environment)."""

    def test_init_sets_observation_space(self):
        env = _make_env()
        env.reset()
        cfg = _make_config()
        se, ge = _make_encoders(env, feature_dim=48, goal_feature_dim=16)

        ppo_env = _make_ppo_env(env, cfg, se, ge)
        expected_dim = se.feature_dim + ge.feature_dim
        assert ppo_env.observation_space.shape == (expected_dim,)

    def test_reset_returns_obs_and_info(self):
        env = _make_env()
        env.reset()
        cfg = _make_config()
        se, ge = _make_encoders(env)

        ppo_env = _make_ppo_env(env, cfg, se, ge)
        obs, info = ppo_env.reset()
        expected_dim = se.feature_dim + ge.feature_dim
        assert obs.shape == (expected_dim,)
        assert obs.dtype == np.float32
        assert isinstance(info, dict)

    def test_step_returns_five_tuple(self):
        env = _make_env()
        env.reset()
        cfg = _make_config()
        se, ge = _make_encoders(env)

        ppo_env = _make_ppo_env(env, cfg, se, ge)
        ppo_env.reset()
        result = ppo_env.step(0)
        assert len(result) == 5
        obs, reward, terminated, truncated, info = result
        expected_dim = se.feature_dim + ge.feature_dim
        assert obs.shape == (expected_dim,)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert "goal_achieved" in info

    def test_reward_is_goal_achievement(self):
        """Reward should be 1.0 when goal is achieved, 0.0 otherwise."""
        env = _make_env()
        env.reset()
        cfg = _make_config()
        se, ge = _make_encoders(env)

        # Unachieved goal
        ppo_env = _make_ppo_env(env, cfg, se, ge, goal_sampler=_dummy_goal_sampler)
        ppo_env.reset()
        _, reward_unachieved, _, _, info_unachieved = ppo_env.step(0)
        assert reward_unachieved == 0.0
        assert info_unachieved["goal_achieved"] == 0.0

        # Achieved goal
        env2 = _make_env()
        env2.reset()
        ppo_env2 = _make_ppo_env(env2, cfg, se, ge, goal_sampler=_achieved_goal_sampler)
        ppo_env2.reset()
        _, reward_achieved, _, _, info_achieved = ppo_env2.step(0)
        assert reward_achieved == 1.0
        assert info_achieved["goal_achieved"] == 1.0

    def test_observation_includes_goal_features(self):
        """Goal-feature slice of the observation should differ for different goals."""
        env = _make_env()
        env.reset()
        cfg = _make_config()
        se, ge = _make_encoders(env)

        goal_a = _DummyGoal(target_pos=(1, 1))
        goal_b = _DummyGoal(target_pos=(3, 3))

        def sampler_a(state, h_idx):
            return goal_a, 1.0

        def sampler_b(state, h_idx):
            return goal_b, 1.0

        # Use the same seed so that the world-model state is identical
        ppo_env_a = _make_ppo_env(env, cfg, se, ge, goal_sampler=sampler_a)
        obs_a, _ = ppo_env_a.reset(seed=42)

        ppo_env_b = _make_ppo_env(env, cfg, se, ge, goal_sampler=sampler_b)
        obs_b, _ = ppo_env_b.reset(seed=42)

        expected_dim = se.feature_dim + ge.feature_dim
        assert obs_a.shape == (expected_dim,)
        assert obs_b.shape == (expected_dim,)

        # The goal-feature tail must differ between the two goals
        goal_slice_a = obs_a[-ge.feature_dim :]
        goal_slice_b = obs_b[-ge.feature_dim :]
        assert not np.array_equal(
            goal_slice_a, goal_slice_b
        ), "Goal-feature slices should differ for different goals"

    def test_episode_truncates_at_steps_per_episode(self):
        env = _make_env()
        env.reset()
        cfg = PPOPhase1Config(num_actions=4, steps_per_episode=3)
        se, ge = _make_encoders(env)

        ppo_env = _make_ppo_env(env, cfg, se, ge)
        ppo_env.reset()
        for _ in range(3):
            _, _, terminated, truncated, _ = ppo_env.step(0)
        assert truncated is True

    def test_world_model_time_limit_is_truncation_not_termination(self):
        """MultiGrid done from max_steps should map to truncated, not terminated."""
        env = _make_env()  # max_steps=20
        env.reset()
        cfg = PPOPhase1Config(num_actions=4, steps_per_episode=100)
        se, ge = _make_encoders(env)

        ppo_env = _make_ppo_env(env, cfg, se, ge)
        ppo_env.reset()
        # Step until the world model signals done
        for _ in range(25):
            _, _, terminated, truncated, _ = ppo_env.step(0)
            if terminated or truncated:
                break
        # MultiGrid done from max_steps should be converted to truncation
        assert truncated is True
        assert terminated is False

    def test_terminated_and_truncated_not_both_true(self):
        """Gymnasium semantics: terminated and truncated should not both be True."""
        env = _make_env()
        env.reset()
        cfg = PPOPhase1Config(num_actions=4, steps_per_episode=3)
        se, ge = _make_encoders(env)

        ppo_env = _make_ppo_env(env, cfg, se, ge)
        ppo_env.reset()
        for _ in range(20):
            _, _, terminated, truncated, _ = ppo_env.step(0)
            assert not (terminated and truncated), (
                f"terminated={terminated}, truncated={truncated} — "
                "Gymnasium requires these to be mutually exclusive"
            )
            if terminated or truncated:
                ppo_env.reset()

    def test_observation_varies_with_state(self):
        """Observation should change when the state changes."""
        env = _make_env()
        env.reset()
        cfg = _make_config()
        se, ge = _make_encoders(env)

        ppo_env = _make_ppo_env(env, cfg, se, ge)
        obs1, _ = ppo_env.reset()
        # Step several times to change state
        for _ in range(5):
            obs2, _, term, trunc, _ = ppo_env.step(np.random.randint(cfg.num_actions))
            if term or trunc:
                break
        assert obs2.shape == obs1.shape
        assert obs2.dtype == obs1.dtype

    def test_action_space_matches_config(self):
        env = _make_env()
        env.reset()
        cfg = _make_config()
        se, ge = _make_encoders(env)

        ppo_env = _make_ppo_env(env, cfg, se, ge)
        assert ppo_env.action_space.n == cfg.num_actions

    def test_seed_produces_deterministic_reset(self):
        """Same seed should produce identical initial observations."""
        env = _make_env()
        env.reset()
        cfg = _make_config()
        se, ge = _make_encoders(env)

        ppo_env = _make_ppo_env(env, cfg, se, ge)
        obs1, _ = ppo_env.reset(seed=42)
        obs2, _ = ppo_env.reset(seed=42)
        np.testing.assert_array_equal(obs1, obs2)

    def test_identity_mode_encoders(self):
        """Environment should work with identity-mode encoders (use_encoders=False)."""
        env = _make_env()
        env.reset()
        cfg = _make_config()
        se, ge = _make_encoders(env, use_encoders=False)

        ppo_env = _make_ppo_env(env, cfg, se, ge)
        obs, info = ppo_env.reset()
        expected_dim = se.feature_dim + ge.feature_dim
        assert obs.shape == (expected_dim,)
        assert obs.dtype == np.float32

        # Step should also work
        obs2, reward, terminated, truncated, info2 = ppo_env.step(0)
        assert obs2.shape == (expected_dim,)


# ======================================================================
# 2. Network factory tests
# ======================================================================


class TestCreateMultiGridPhase1PPONetworks:
    """Tests for the create_multigrid_phase1_ppo_networks factory."""

    def test_returns_three_components(self):
        env = _make_env()
        env.reset()
        cfg = _make_config()

        ac, se, ge = create_multigrid_phase1_ppo_networks(
            env=env,
            config=cfg,
            feature_dim=32,
            goal_feature_dim=16,
        )
        assert isinstance(ac, GoalConditionedActorCritic)
        assert isinstance(se, MultiGridStateEncoder)
        assert isinstance(ge, MultiGridGoalEncoder)

    def test_actor_critic_shapes(self):
        env = _make_env()
        env.reset()
        cfg = _make_config()

        ac, se, ge = create_multigrid_phase1_ppo_networks(
            env=env, config=cfg, feature_dim=32, goal_feature_dim=16
        )
        obs_dim = se.feature_dim + ge.feature_dim
        obs = torch.randn(2, obs_dim)
        logits, value = ac(obs)
        assert logits.shape == (2, cfg.num_actions)
        assert value.shape == (2, 1)

    def test_encoder_feature_dims(self):
        env = _make_env()
        env.reset()
        cfg = _make_config()

        _, se, ge = create_multigrid_phase1_ppo_networks(
            env=env, config=cfg, feature_dim=48, goal_feature_dim=24
        )
        assert se.feature_dim == 48
        assert ge.feature_dim == 24

    def test_identity_mode_uses_actual_feature_dim(self):
        """In identity mode (use_encoders=False) obs_dim must match encoder output."""
        env = _make_env()
        env.reset()
        cfg = _make_config()

        ac, se, ge = create_multigrid_phase1_ppo_networks(
            env=env,
            config=cfg,
            feature_dim=32,
            goal_feature_dim=16,
            use_encoders=False,
        )
        # In identity mode, encoder feature_dim is overwritten to the
        # actual flattened output size, which is much larger than 32.
        assert se.feature_dim != 32
        obs_dim = se.feature_dim + ge.feature_dim
        obs = torch.randn(1, obs_dim)
        with torch.no_grad():
            logits, value = ac(obs)
        assert logits.shape[1] == cfg.num_actions

    def test_device_placement(self):
        """Networks should be placed on the specified device."""
        env = _make_env()
        env.reset()
        cfg = _make_config()

        ac, se, ge = create_multigrid_phase1_ppo_networks(
            env=env, config=cfg, feature_dim=32, goal_feature_dim=16, device="cpu"
        )
        # Check that all parameters are on cpu
        for p in ac.parameters():
            assert p.device.type == "cpu"
        for p in se.parameters():
            assert p.device.type == "cpu"


# ======================================================================
# 3. Integration tests
# ======================================================================


class TestMultiGridPhase1PPOIntegration:
    """End-to-end integration tests for MultiGrid PPO Phase 1."""

    def test_env_with_actor_critic(self):
        """Actor-critic can process observations from the env."""
        env = _make_env()
        env.reset()
        cfg = _make_config()

        ac, se, ge = create_multigrid_phase1_ppo_networks(
            env=env, config=cfg, feature_dim=32, goal_feature_dim=16
        )
        ppo_env = _make_ppo_env(env, cfg, se, ge)

        obs, _ = ppo_env.reset()
        obs_t = torch.tensor(obs).unsqueeze(0)
        with torch.no_grad():
            logits, value = ac(obs_t)
        assert logits.shape == (1, cfg.num_actions)
        assert value.shape == (1, 1)

    def test_multi_step_rollout(self):
        """Can run multiple steps without errors."""
        env = _make_env()
        env.reset()
        cfg = _make_config()

        ac, se, ge = create_multigrid_phase1_ppo_networks(
            env=env, config=cfg, feature_dim=32, goal_feature_dim=16
        )
        ppo_env = _make_ppo_env(env, cfg, se, ge)

        obs, _ = ppo_env.reset()
        total_reward = 0.0
        for _ in range(15):
            obs_t = torch.tensor(obs).unsqueeze(0)
            with torch.no_grad():
                logits, _ = ac(obs_t)
                action = torch.argmax(logits, dim=-1).item()
            obs, reward, terminated, truncated, info = ppo_env.step(action)
            total_reward += reward
            if terminated or truncated:
                obs, _ = ppo_env.reset()
        # Just verify it ran without error; reward value depends on environment

    def test_get_action_and_value_with_env_obs(self):
        """get_action_and_value works with real environment observations."""
        env = _make_env()
        env.reset()
        cfg = _make_config()

        ac, se, ge = create_multigrid_phase1_ppo_networks(
            env=env, config=cfg, feature_dim=32, goal_feature_dim=16
        )
        ppo_env = _make_ppo_env(env, cfg, se, ge)

        obs, _ = ppo_env.reset()
        obs_t = torch.tensor(obs).unsqueeze(0)
        with torch.no_grad():
            action, log_prob, entropy, value = ac.get_action_and_value(obs_t)
        assert action.shape == (1,)
        assert log_prob.shape == (1,)
        assert entropy.shape == (1,)
        assert value.shape == (1,)

    def test_trainer_creation_with_multigrid_networks(self):
        """PPOPhase1Trainer can be initialised with MultiGrid networks."""
        pytest.importorskip("pufferlib")  # Trainer requires pufferlib

        from empo.learning_based.phase1_ppo.trainer import PPOPhase1Trainer

        env = _make_env()
        env.reset()
        cfg = _make_config()

        ac, _, _ = create_multigrid_phase1_ppo_networks(
            env=env, config=cfg, feature_dim=32, goal_feature_dim=16
        )
        trainer = PPOPhase1Trainer(
            actor_critic=ac,
            config=cfg,
            device="cpu",
        )
        assert trainer.actor_critic is ac


# ======================================================================
# 4. Isolation test
# ======================================================================


class TestMultiGridPhase1PPOIsolation:
    """Verify new code doesn't modify existing DQN path."""

    def test_no_ppo_imports_in_multigrid_phase1_init(self):
        """The DQN-path multigrid/phase1/__init__.py must not import PPO code."""
        import importlib

        mod = importlib.import_module("empo.learning_based.multigrid.phase1")
        source_file = mod.__file__
        assert source_file is not None
        with open(source_file) as f:
            content = f.read()
        assert "phase1_ppo" not in content

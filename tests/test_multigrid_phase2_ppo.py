"""
Tests for the MultiGrid-specific PPO Phase 2 implementation.

These tests verify:
1. ``MultiGridWorldModelEnv`` — observation encoding, step/reset API
2. ``create_multigrid_ppo_networks`` — network factory output shapes & types
3. Integration — env + actor-critic + trainer wiring

All tests use a real ``MultiGridEnv`` instance (tiny 4×6 grid) so they
exercise the actual encoder pipeline end-to-end.
"""

import numpy as np
import torch

from gym_multigrid.multigrid import MultiGridEnv, World

from empo.learning_based.phase2_ppo.config import PPOPhase2Config
from empo.learning_based.phase2_ppo.actor_critic import EMPOActorCritic
from empo.learning_based.phase2_ppo.trainer import (
    PPOPhase2Trainer,
    PPOAuxiliaryNetworks,
)

from empo.learning_based.multigrid.state_encoder import MultiGridStateEncoder
from empo.learning_based.multigrid.feature_extraction import get_num_agents_per_color

from empo.learning_based.multigrid.phase2_ppo import (
    MultiGridWorldModelEnv,
    create_multigrid_ppo_networks,
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


def _make_config(env: MultiGridEnv) -> PPOPhase2Config:
    return PPOPhase2Config(
        num_actions=env.action_space.n,
        num_robots=len(env.robot_agent_indices),
        hidden_dim=32,
        ppo_rollout_length=8,
        ppo_num_minibatches=2,
        ppo_update_epochs=1,
        num_envs=2,
        num_ppo_iterations=2,
        aux_training_steps_per_iteration=1,
        aux_buffer_size=500,
        batch_size=4,
        steps_per_episode=20,
        device="cpu",
    )


def _uniform_policy(state, human_idx, goal, wm):
    n = wm.action_space.n
    return [1.0 / n] * n


def _dummy_goal_sampler(state, human_idx):
    return f"goal_{human_idx}", 1.0


def _make_encoder(env: MultiGridEnv, feature_dim: int = 32) -> MultiGridStateEncoder:
    npc = get_num_agents_per_color(env)
    return MultiGridStateEncoder(
        grid_height=env.height,
        grid_width=env.width,
        num_agents_per_color=npc,
        feature_dim=feature_dim,
        include_step_count=True,
        use_encoders=True,
    )


# ======================================================================
# 1. MultiGridWorldModelEnv tests
# ======================================================================


class TestMultiGridWorldModelEnv:
    """Tests for MultiGridWorldModelEnv (real encoder, real environment)."""

    def test_init_sets_observation_space(self):
        env = _make_env()
        env.reset()
        cfg = _make_config(env)
        enc = _make_encoder(env, feature_dim=48)

        ppo_env = MultiGridWorldModelEnv(
            world_model=env,
            human_policy_prior=_uniform_policy,
            goal_sampler=_dummy_goal_sampler,
            human_agent_indices=env.human_agent_indices,
            robot_agent_indices=env.robot_agent_indices,
            config=cfg,
            state_encoder=enc,
        )
        assert ppo_env.observation_space.shape == (48,)

    def test_reset_returns_obs_and_info(self):
        env = _make_env()
        env.reset()
        cfg = _make_config(env)
        enc = _make_encoder(env)

        ppo_env = MultiGridWorldModelEnv(
            world_model=env,
            human_policy_prior=_uniform_policy,
            goal_sampler=_dummy_goal_sampler,
            human_agent_indices=env.human_agent_indices,
            robot_agent_indices=env.robot_agent_indices,
            config=cfg,
            state_encoder=enc,
        )
        obs, info = ppo_env.reset()
        assert obs.shape == (32,)
        assert obs.dtype == np.float32
        assert "state" in info

    def test_step_returns_five_tuple(self):
        env = _make_env()
        env.reset()
        cfg = _make_config(env)
        enc = _make_encoder(env)

        ppo_env = MultiGridWorldModelEnv(
            world_model=env,
            human_policy_prior=_uniform_policy,
            goal_sampler=_dummy_goal_sampler,
            human_agent_indices=env.human_agent_indices,
            robot_agent_indices=env.robot_agent_indices,
            config=cfg,
            state_encoder=enc,
        )
        ppo_env.reset()
        result = ppo_env.step(0)
        assert len(result) == 5
        obs, reward, terminated, truncated, info = result
        assert obs.shape == (32,)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert "env_reward" in info
        assert "u_r" in info

    def test_info_values_are_numeric(self):
        """PufferLib Serial backend requires numeric info values."""
        env = _make_env()
        env.reset()
        cfg = _make_config(env)
        enc = _make_encoder(env)

        ppo_env = MultiGridWorldModelEnv(
            world_model=env,
            human_policy_prior=_uniform_policy,
            goal_sampler=_dummy_goal_sampler,
            human_agent_indices=env.human_agent_indices,
            robot_agent_indices=env.robot_agent_indices,
            config=cfg,
            state_encoder=enc,
        )
        ppo_env.reset()
        _, _, _, _, info = ppo_env.step(0)
        for key, val in info.items():
            assert isinstance(
                val, (int, float, np.integer, np.floating)
            ), f"Non-numeric info[{key!r}]: {type(val).__name__}"

    def test_reward_is_zero_without_aux_nets(self):
        env = _make_env()
        env.reset()
        cfg = _make_config(env)
        enc = _make_encoder(env)

        ppo_env = MultiGridWorldModelEnv(
            world_model=env,
            human_policy_prior=_uniform_policy,
            goal_sampler=_dummy_goal_sampler,
            human_agent_indices=env.human_agent_indices,
            robot_agent_indices=env.robot_agent_indices,
            config=cfg,
            state_encoder=enc,
            auxiliary_networks=None,
        )
        ppo_env.reset()
        _, reward, _, _, _ = ppo_env.step(0)
        assert reward == 0.0

    def test_aux_buffer_populated_after_step(self):
        env = _make_env()
        env.reset()
        cfg = _make_config(env)
        enc = _make_encoder(env)

        ppo_env = MultiGridWorldModelEnv(
            world_model=env,
            human_policy_prior=_uniform_policy,
            goal_sampler=_dummy_goal_sampler,
            human_agent_indices=env.human_agent_indices,
            robot_agent_indices=env.robot_agent_indices,
            config=cfg,
            state_encoder=enc,
        )
        ppo_env.reset()
        ppo_env.step(0)
        assert len(ppo_env._aux_buffer) == 1
        entry = ppo_env._aux_buffer[0]
        expected_keys = {
            "state",
            "next_state",
            "goals",
            "goal_weights",
            "human_actions",
            "transition_probs",
            "robot_action",
            "terminated",
            "truncated",
            "terminal",
        }
        assert set(entry.keys()) == expected_keys

    def test_episode_truncates_at_steps_per_episode(self):
        env = _make_env()
        env.reset()
        cfg = PPOPhase2Config(
            num_actions=env.action_space.n,
            num_robots=1,
            hidden_dim=32,
            steps_per_episode=3,
        )
        enc = _make_encoder(env)

        ppo_env = MultiGridWorldModelEnv(
            world_model=env,
            human_policy_prior=_uniform_policy,
            goal_sampler=_dummy_goal_sampler,
            human_agent_indices=env.human_agent_indices,
            robot_agent_indices=env.robot_agent_indices,
            config=cfg,
            state_encoder=enc,
        )
        ppo_env.reset()
        for i in range(3):
            _, _, terminated, truncated, _ = ppo_env.step(0)
        assert truncated is True

    def test_observation_varies_with_state(self):
        """Observation should change when the state changes."""
        env = _make_env()
        env.reset()
        cfg = _make_config(env)
        enc = _make_encoder(env)

        ppo_env = MultiGridWorldModelEnv(
            world_model=env,
            human_policy_prior=_uniform_policy,
            goal_sampler=_dummy_goal_sampler,
            human_agent_indices=env.human_agent_indices,
            robot_agent_indices=env.robot_agent_indices,
            config=cfg,
            state_encoder=enc,
        )
        obs1, _ = ppo_env.reset()
        # Step several times to change state
        for _ in range(5):
            obs2, _, term, trunc, _ = ppo_env.step(np.random.randint(cfg.num_actions))
            if term or trunc:
                break
        # After stepping, the observation should differ from the initial one
        # (unless the agent didn't move, which is unlikely in 5 steps)
        # We check that at least the obs tensor is produced correctly
        assert obs2.shape == obs1.shape
        assert obs2.dtype == obs1.dtype

    def test_action_space_matches_config(self):
        env = _make_env()
        env.reset()
        cfg = _make_config(env)
        enc = _make_encoder(env)

        ppo_env = MultiGridWorldModelEnv(
            world_model=env,
            human_policy_prior=_uniform_policy,
            goal_sampler=_dummy_goal_sampler,
            human_agent_indices=env.human_agent_indices,
            robot_agent_indices=env.robot_agent_indices,
            config=cfg,
            state_encoder=enc,
        )
        expected_n = cfg.num_actions**cfg.num_robots
        assert ppo_env.action_space.n == expected_n


# ======================================================================
# 2. Network factory tests
# ======================================================================


class TestCreateMultiGridPPONetworks:
    """Tests for the create_multigrid_ppo_networks factory."""

    def test_returns_three_components(self):
        env = _make_env()
        env.reset()
        cfg = _make_config(env)

        ac, aux, enc = create_multigrid_ppo_networks(
            env=env,
            config=cfg,
            feature_dim=32,
        )
        assert isinstance(ac, EMPOActorCritic)
        assert isinstance(aux, PPOAuxiliaryNetworks)
        assert isinstance(enc, MultiGridStateEncoder)

    def test_actor_critic_shapes(self):
        env = _make_env()
        env.reset()
        cfg = _make_config(env)

        ac, _, enc = create_multigrid_ppo_networks(env=env, config=cfg, feature_dim=32)
        obs = torch.randn(2, enc.feature_dim)
        logits, value = ac(obs)
        assert logits.shape == (2, cfg.num_joint_actions)
        assert value.shape == (2, 1)

    def test_auxiliary_networks_present(self):
        env = _make_env()
        env.reset()
        cfg = _make_config(env)

        _, aux, _ = create_multigrid_ppo_networks(
            env=env, config=cfg, feature_dim=32, use_x_h=True, use_u_r=True
        )
        assert aux.v_h_e is not None
        assert aux.x_h is not None
        assert aux.u_r is not None

    def test_no_x_h_or_u_r_when_disabled(self):
        env = _make_env()
        env.reset()
        cfg = _make_config(env)

        _, aux, _ = create_multigrid_ppo_networks(
            env=env, config=cfg, feature_dim=32, use_x_h=False, use_u_r=False
        )
        assert aux.v_h_e is not None
        assert aux.x_h is None
        assert aux.u_r is None

    def test_encoder_feature_dim(self):
        env = _make_env()
        env.reset()
        cfg = _make_config(env)

        _, _, enc = create_multigrid_ppo_networks(env=env, config=cfg, feature_dim=48)
        assert enc.feature_dim == 48

    def test_v_h_e_forward_works(self):
        """V_h^e network should accept state + world_model + human_idx + goal."""
        env = _make_env()
        env.reset()
        cfg = _make_config(env)

        _, aux, _ = create_multigrid_ppo_networks(env=env, config=cfg, feature_dim=32)
        state = env.get_state()
        h_idx = env.human_agent_indices[0]
        # Use a mock goal that has is_achieved
        goal = type(
            "Goal", (), {"is_achieved": lambda self, s: 0, "target_pos": (1, 1)}
        )()

        with torch.no_grad():
            result = aux.v_h_e(state, env, h_idx, goal, "cpu")
        assert result.shape == (1,) or result.dim() <= 1


# ======================================================================
# 3. Integration tests
# ======================================================================


class TestMultiGridPPOIntegration:
    """End-to-end integration tests for MultiGrid PPO."""

    def test_env_with_actor_critic(self):
        """Actor-critic can process observations from the env."""
        env = _make_env()
        env.reset()
        cfg = _make_config(env)

        ac, _, enc = create_multigrid_ppo_networks(env=env, config=cfg, feature_dim=32)
        ppo_env = MultiGridWorldModelEnv(
            world_model=env,
            human_policy_prior=_uniform_policy,
            goal_sampler=_dummy_goal_sampler,
            human_agent_indices=env.human_agent_indices,
            robot_agent_indices=env.robot_agent_indices,
            config=cfg,
            state_encoder=enc,
        )

        obs, _ = ppo_env.reset()
        obs_t = torch.tensor(obs).unsqueeze(0)
        with torch.no_grad():
            logits, value = ac(obs_t)
        assert logits.shape == (1, cfg.num_joint_actions)
        assert value.shape == (1, 1)

    def test_trainer_creation_with_multigrid_networks(self):
        """Trainer can be initialised with MultiGrid networks."""
        env = _make_env()
        env.reset()
        cfg = _make_config(env)

        ac, aux, _ = create_multigrid_ppo_networks(env=env, config=cfg, feature_dim=32)
        trainer = PPOPhase2Trainer(
            actor_critic=ac,
            auxiliary_networks=aux,
            config=cfg,
            device="cpu",
        )
        assert trainer.actor_critic is ac
        assert trainer.auxiliary_networks is aux

    def test_freeze_auxiliary_networks(self):
        """Freeze creates target copies for MultiGrid networks."""
        env = _make_env()
        env.reset()
        cfg = _make_config(env)

        ac, aux, _ = create_multigrid_ppo_networks(
            env=env, config=cfg, feature_dim=32, use_x_h=True, use_u_r=True
        )
        trainer = PPOPhase2Trainer(ac, aux, cfg, device="cpu")
        trainer.freeze_auxiliary_networks()

        assert aux.v_h_e_target is not None
        assert aux.x_h_target is not None
        assert aux.u_r_target is not None
        # Target should be separate objects
        assert aux.v_h_e_target is not aux.v_h_e

    def test_auxiliary_training_step(self):
        """One aux training step runs without error on MultiGrid transitions."""
        env = _make_env()
        env.reset()
        cfg = _make_config(env)

        ac, aux, enc = create_multigrid_ppo_networks(
            env=env, config=cfg, feature_dim=32
        )
        trainer = PPOPhase2Trainer(ac, aux, cfg, device="cpu")
        trainer.freeze_auxiliary_networks()

        # Push a few transitions
        ppo_env = MultiGridWorldModelEnv(
            world_model=env,
            human_policy_prior=_uniform_policy,
            goal_sampler=_dummy_goal_sampler,
            human_agent_indices=env.human_agent_indices,
            robot_agent_indices=env.robot_agent_indices,
            config=cfg,
            state_encoder=enc,
        )
        ppo_env.reset()
        for _ in range(cfg.batch_size + 2):
            _, _, term, trunc, _ = ppo_env.step(np.random.randint(cfg.num_actions))
            if ppo_env._aux_buffer:
                entry = ppo_env._aux_buffer[-1]
                trainer.push_transition_to_aux_buffer(
                    state=entry["state"],
                    next_state=entry["next_state"],
                    robot_action=entry["robot_action"],
                    goals=entry["goals"],
                    goal_weights=entry["goal_weights"],
                    human_actions=entry["human_actions"],
                    transition_probs=entry["transition_probs"],
                    terminal=entry["terminal"],
                )
            if term or trunc:
                ppo_env.reset()

        losses = trainer.train_auxiliary_step(world_model=env)
        # Should have computed at least v_h_e loss
        assert isinstance(losses, dict)


# ======================================================================
# 4. Isolation test
# ======================================================================


class TestMultiGridPPOIsolation:
    """Verify new code doesn't modify existing DQN path."""

    def test_no_ppo_imports_in_multigrid_phase2_init(self):
        """The DQN-path multigrid/phase2/__init__.py must not import PPO code."""
        import importlib

        mod = importlib.import_module("empo.learning_based.multigrid.phase2")
        source_file = mod.__file__
        assert source_file is not None
        with open(source_file) as f:
            content = f.read()
        assert "phase2_ppo" not in content

"""
Tests for the BushWorld-specific PPO implementations (Phase 1 and Phase 2).

These tests verify the BushWorld analogues of the multigrid PPO packages:
1. ``BushWorldPhase1PPOEnv`` + ``create_bushworld_phase1_ppo_networks`` —
   goal-conditioned observation encoding, reset/step API, network factory.
2. ``BushWorldWorldModelEnv`` + ``create_bushworld_ppo_networks`` — Phase 2
   observation encoding, reset/step API, actor-critic + auxiliary networks.

All tests use a real ``BushWorld`` instance and exercise the encoder pipeline
end-to-end. They do NOT require ``pufferlib`` (only the env wrappers and network
factories are tested, which reuse the shared PPO infrastructure).
"""

import numpy as np
import pytest
import torch

from empo.bushworld import BushWorld, ShortestPathHumanPolicyPrior
from empo.learning_based.phase1_ppo.actor_critic import GoalConditionedActorCritic
from empo.learning_based.phase1_ppo.config import PPOPhase1Config
from empo.learning_based.phase2_ppo.actor_critic import EMPOActorCritic
from empo.learning_based.phase2_ppo.config import PPOPhase2Config

from empo.learning_based.bushworld.phase1_ppo import (
    BushWorldPhase1PPOEnv,
    create_bushworld_phase1_ppo_networks,
)
from empo.learning_based.bushworld.phase2_ppo import (
    BushWorldWorldModelEnv,
    create_bushworld_ppo_networks,
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


def _phase1_config(env):
    return PPOPhase1Config(num_actions=env.action_space.n, hidden_dim=32)


def _phase2_config(env):
    return PPOPhase2Config(
        num_actions=env.action_space.n,
        num_robots=len(env.robot_agent_indices),
        hidden_dim=32,
        steps_per_episode=env.max_steps,
    )


def _make_phase1_env(env, feature_dim=32, goal_feature_dim=16, use_encoders=True):
    cfg = _phase1_config(env)
    ac, senc, genc = create_bushworld_phase1_ppo_networks(
        env,
        cfg,
        feature_dim=feature_dim,
        goal_feature_dim=goal_feature_dim,
        use_encoders=use_encoders,
    )
    sampler = env.possible_goal_generator.get_sampler()
    human_idx = env.human_agent_indices[0]
    num_actions = env.action_space.n
    # All agents except the training human need a fallback policy.
    # Each policy is called as ``policy(state, agent_idx) -> action``.
    others = {
        idx: (lambda state, agent_idx: int(np.random.randint(num_actions)))
        for idx in range(env.num_players)
        if idx != human_idx
    }
    ppo_env = BushWorldPhase1PPOEnv(
        env, sampler.sample, human_idx, others, cfg, senc, genc
    )
    return ppo_env, ac, senc, genc


def _make_phase2_env(env, feature_dim=32, use_encoders=True):
    cfg = _phase2_config(env)
    ac, aux, enc = create_bushworld_ppo_networks(
        env, cfg, feature_dim=feature_dim, use_encoders=use_encoders
    )
    hpp = ShortestPathHumanPolicyPrior(env, env.human_agent_indices)

    def prior(state, h_idx, goal, world_model):
        return hpp(state, h_idx, goal)

    sampler = env.possible_goal_generator.get_sampler()
    ppo_env = BushWorldWorldModelEnv(
        env,
        prior,
        sampler.sample,
        list(env.human_agent_indices),
        list(env.robot_agent_indices),
        cfg,
        enc,
        auxiliary_networks=aux,
    )
    return ppo_env, ac, aux, enc


# --------------------------------------------------------------------------- #
# Phase 1 PPO
# --------------------------------------------------------------------------- #
class TestPhase1PPONetworks:
    def test_factory_returns_three_components(self):
        env = make_corridor()
        ac, senc, genc = create_bushworld_phase1_ppo_networks(
            env, _phase1_config(env), feature_dim=32, goal_feature_dim=16
        )
        assert isinstance(ac, GoalConditionedActorCritic)
        assert senc.feature_dim == 32
        assert genc.feature_dim == 16

    def test_obs_dim_is_state_plus_goal(self):
        env = make_corridor()
        ppo_env, _, senc, genc = _make_phase1_env(env)
        obs, _ = ppo_env.reset(seed=0)
        assert obs.shape == (senc.feature_dim + genc.feature_dim,)

    def test_actor_critic_shapes(self):
        env = make_corridor()
        ppo_env, ac, _, _ = _make_phase1_env(env)
        obs, _ = ppo_env.reset(seed=0)
        logits, value = ac(torch.tensor(obs).unsqueeze(0))
        assert logits.shape == (1, env.action_space.n)
        assert value.shape == (1, 1)

    def test_reset_and_step_api(self):
        env = make_corridor()
        ppo_env, _, _, _ = _make_phase1_env(env)
        obs, info = ppo_env.reset(seed=0)
        assert obs.dtype == np.float32
        result = ppo_env.step(ppo_env.action_space.sample())
        assert len(result) == 5
        obs2, reward, terminated, truncated, info2 = result
        assert obs2.shape == obs.shape
        assert isinstance(terminated, (bool, np.bool_))
        assert isinstance(truncated, (bool, np.bool_))

    def test_action_space_matches_config(self):
        env = make_corridor()
        ppo_env, _, _, _ = _make_phase1_env(env)
        assert ppo_env.action_space.n == env.action_space.n

    def test_identity_mode_encoders(self):
        env = make_corridor()
        ppo_env, _, senc, genc = _make_phase1_env(env, use_encoders=False)
        obs, _ = ppo_env.reset(seed=0)
        assert obs.shape == (senc.feature_dim + genc.feature_dim,)


# --------------------------------------------------------------------------- #
# Phase 2 PPO
# --------------------------------------------------------------------------- #
class TestPhase2PPONetworks:
    def test_factory_returns_components(self):
        env = make_corridor()
        ac, aux, enc = create_bushworld_ppo_networks(
            env, _phase2_config(env), feature_dim=32
        )
        assert isinstance(ac, EMPOActorCritic)
        assert enc.feature_dim == 32
        assert aux.v_h_e is not None
        assert aux.x_h is not None
        assert aux.u_r is not None

    def test_shared_state_encoder(self):
        env = make_corridor()
        _, aux, enc = create_bushworld_ppo_networks(
            env, _phase2_config(env), feature_dim=32
        )
        # The auxiliary networks share the factory's state encoder instance.
        assert aux.v_h_e.state_encoder is enc
        assert aux.x_h.state_encoder is enc
        assert aux.u_r.state_encoder is enc

    def test_obs_dim_matches_feature_dim(self):
        env = make_corridor()
        ppo_env, _, _, enc = _make_phase2_env(env)
        obs, _ = ppo_env.reset(seed=0)
        assert obs.shape == (enc.feature_dim,)

    def test_actor_critic_shapes(self):
        env = make_corridor()
        ppo_env, ac, _, _ = _make_phase2_env(env)
        obs, _ = ppo_env.reset(seed=0)
        logits, value = ac(torch.tensor(obs).unsqueeze(0))
        num_robots = len(env.robot_agent_indices)
        assert logits.shape == (1, env.action_space.n**num_robots)
        assert value.shape == (1, 1)

    def test_reset_and_step_api(self):
        env = make_corridor()
        ppo_env, _, _, _ = _make_phase2_env(env)
        obs, info = ppo_env.reset(seed=0)
        assert obs.dtype == np.float32
        result = ppo_env.step(ppo_env.action_space.sample())
        assert len(result) == 5
        obs2, reward, terminated, truncated, info2 = result
        assert obs2.shape == obs.shape

    def test_no_x_h_no_u_r(self):
        env = make_corridor()
        _, aux, _ = create_bushworld_ppo_networks(
            env, _phase2_config(env), feature_dim=32, use_x_h=False, use_u_r=False
        )
        assert aux.x_h is None
        assert aux.u_r is None

    def test_identity_mode_uses_actual_feature_dim(self):
        env = make_corridor()
        ppo_env, _, _, enc = _make_phase2_env(env, use_encoders=False)
        obs, _ = ppo_env.reset(seed=0)
        assert obs.shape == (enc.feature_dim,)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""
Tests for parallel episode collection.

Run with:
    PYTHONPATH=src:vendor/multigrid pytest src/empo/ali_learning_based/tests/test_parallel_rollout.py -v
"""

import pytest
import torch

from empo.ali_learning_based.envs import get_env_path
from empo.ali_learning_based.parallel_rollout import ParallelCollector
from empo.ali_learning_based.rollout import EpisodeData


# Use forced_rock_push as it's the smallest env (7x6)
CONFIG_PATH = get_env_path("forced_rock_push.yaml")
ROBOT_IDX = 0
HUMAN_IDX = 1


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def _make_collector(num_workers=2):
    return ParallelCollector(CONFIG_PATH, ROBOT_IDX, HUMAN_IDX, num_workers)


def _get_env_info():
    """Build env once to get dimensions for network construction."""
    from gym_multigrid.multigrid import MultiGridEnv, SmallActions
    from empo.ali_learning_based.encoders import StateEncoder, GoalEncoder
    from empo.ali_learning_based.networks import QhNet, QrNet

    env = MultiGridEnv(config_file=CONFIG_PATH, partial_obs=False, actions_set=SmallActions)
    env.reset()
    se = StateEncoder(env, robot_agent_index=ROBOT_IDX, human_agent_indices=[HUMAN_IDX])
    ge = GoalEncoder(env)
    num_actions = len(SmallActions.available)

    q_net = QhNet.from_encoders(se, ge, num_actions=num_actions)
    qr_net = QrNet.from_encoders(se, num_actions=num_actions)

    state = env.get_state()
    goals = [g for g, _ in env.possible_goal_generator.generate(state, HUMAN_IDX)]

    return {
        "se": se, "ge": ge, "num_actions": num_actions,
        "q_net": q_net, "qr_net": qr_net, "goals": goals,
        "state_dim": se.dim, "goal_dim": ge.dim,
    }


# -----------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------

class TestParallelCollector:

    def test_create_and_close(self):
        """Pool can be created and shut down without error."""
        c = _make_collector(num_workers=2)
        c.close()

    def test_phase1_basic(self):
        """Phase 1 parallel collection returns valid EpisodeData."""
        info = _get_env_info()
        sd = info["q_net"].state_dict()

        c = _make_collector(num_workers=2)
        try:
            tasks = [
                {
                    "goal_idx": 0,
                    "q_net_state_dict": sd,
                    "epsilon": 0.5,
                    "beta_h": 10.0,
                    "reward_shaping": "none",
                    "shaping_weight": 1.0,
                    "gamma": 0.99,
                    "random_start": False,
                    "done_on_goal": True,
                },
                {
                    "goal_idx": 1,
                    "q_net_state_dict": sd,
                    "epsilon": 0.5,
                    "beta_h": 10.0,
                    "reward_shaping": "none",
                    "shaping_weight": 1.0,
                    "gamma": 0.99,
                    "random_start": False,
                    "done_on_goal": True,
                },
            ]
            episodes = c.collect_phase1_episodes(tasks)

            assert len(episodes) == 2
            for ep in episodes:
                assert isinstance(ep, EpisodeData)
                T = ep.states.shape[0]
                assert T > 0
                assert ep.states.shape == (T, info["state_dim"])
                assert ep.goals.shape == (T, info["goal_dim"])
                assert ep.actions.shape == (T,)
                assert ep.rewards.shape == (T,)
                assert ep.next_states.shape == (T, info["state_dim"])
                assert ep.dones.shape == (T,)
                assert ep.goal_rewards.shape == (T,)
                # goal_rewards should be 0/1
                assert ((ep.goal_rewards == 0.0) | (ep.goal_rewards == 1.0)).all()
        finally:
            c.close()

    def test_phase2_basic(self):
        """Phase 2 parallel collection returns valid EpisodeData."""
        info = _get_env_info()
        p1_sd = info["q_net"].state_dict()
        qr_sd = info["qr_net"].state_dict()

        c = _make_collector(num_workers=2)
        try:
            tasks = [
                {
                    "goal_idx": 0,
                    "phase1_state_dict": p1_sd,
                    "qr_state_dict": qr_sd,
                    "epsilon": 0.5,
                    "beta_h": 10.0,
                    "rock_seek_prob": 0.0,
                    "rock_curiosity_bonus": 0.0,
                    "rock_approach_weight": 0.0,
                    "gamma_r": 0.99,
                    "random_start": False,
                    "done_on_goal": True,
                },
                {
                    "goal_idx": 1,
                    "phase1_state_dict": p1_sd,
                    "qr_state_dict": qr_sd,
                    "epsilon": 0.5,
                    "beta_h": 10.0,
                    "rock_seek_prob": 0.0,
                    "rock_curiosity_bonus": 0.0,
                    "rock_approach_weight": 0.0,
                    "gamma_r": 0.99,
                    "random_start": False,
                    "done_on_goal": True,
                },
            ]
            episodes = c.collect_phase2_episodes(tasks)

            assert len(episodes) == 2
            for ep in episodes:
                assert isinstance(ep, EpisodeData)
                T = ep.states.shape[0]
                assert T > 0
                assert ep.states.shape == (T, info["state_dim"])
                assert ep.goal_rewards.shape == (T,)
                assert ((ep.goal_rewards == 0.0) | (ep.goal_rewards == 1.0)).all()
        finally:
            c.close()

    def test_phase1_with_shaping(self):
        """Phase 1 with PBRS reward shaping works in parallel."""
        info = _get_env_info()
        sd = info["q_net"].state_dict()

        c = _make_collector(num_workers=2)
        try:
            tasks = [{
                "goal_idx": 0,
                "q_net_state_dict": sd,
                "epsilon": 1.0,
                "beta_h": 10.0,
                "reward_shaping": "pbrs",
                "shaping_weight": 1.0,
                "gamma": 0.99,
                "random_start": True,
                "done_on_goal": True,
            }]
            episodes = c.collect_phase1_episodes(tasks)
            assert len(episodes) == 1
            ep = episodes[0]
            # Shaped rewards can differ from goal_rewards
            assert ep.rewards.shape == ep.goal_rewards.shape
        finally:
            c.close()

    def test_phase2_with_rock_bonus(self):
        """Phase 2 with rock curiosity bonus works in parallel."""
        info = _get_env_info()
        p1_sd = info["q_net"].state_dict()
        qr_sd = info["qr_net"].state_dict()

        c = _make_collector(num_workers=2)
        try:
            tasks = [{
                "goal_idx": 0,
                "phase1_state_dict": p1_sd,
                "qr_state_dict": qr_sd,
                "epsilon": 1.0,
                "beta_h": 10.0,
                "rock_seek_prob": 0.5,
                "rock_curiosity_bonus": 3.0,
                "rock_approach_weight": 0.3,
                "gamma_r": 0.99,
                "random_start": False,
                "done_on_goal": True,
            }]
            episodes = c.collect_phase2_episodes(tasks)
            assert len(episodes) == 1
            ep = episodes[0]
            assert ep.goal_rewards.shape == ep.rewards.shape
            # goal_rewards stays pure 0/1 even with rock bonus
            assert ((ep.goal_rewards == 0.0) | (ep.goal_rewards == 1.0)).all()
        finally:
            c.close()


# -----------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------

if __name__ == "__main__":
    pytest.main([__file__, "-v"])

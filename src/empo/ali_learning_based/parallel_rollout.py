"""
Parallel episode collection using multiprocessing.

Uses a persistent ``multiprocessing.Pool`` so worker processes are created once
and reused across training iterations.  Each worker owns its own env, encoders,
and goal list.  Network weights are serialized to CPU and sent per collection
batch (cheap for our small networks).

Usage:
    collector = ParallelCollector(config_path, robot_idx, human_idx, num_workers=4)
    episodes = collector.collect_phase1_episodes(tasks)
    for ep in episodes:
        buffer.push_batch(ep.states, ep.goals, ...)
    collector.close()

Limitations:
    - UCB exploration is not supported (visit_counts need global state).
      Phase1Trainer falls back to sequential collection for UCB.
    - Workers run CPU inference (no MPS/CUDA).  Fine for small MLPs.
"""

import multiprocessing as mp
import random
from typing import Any, Dict, List, Optional

import torch

from empo.ali_learning_based.rollout import EpisodeData


# ---------------------------------------------------------------------------
# Worker-local globals (set once by _init_worker, reused across episodes)
# ---------------------------------------------------------------------------

_w_env = None      # MultiGridEnv instance
_w_se = None       # StateEncoder
_w_ge = None       # GoalEncoder
_w_goals = None    # List of goal objects
_w_num_actions = None
_w_num_agents = None
_w_robot_idx = None
_w_human_idx = None


def _init_worker(
    config_path: str,
    robot_idx: int,
    human_idx: int,
):
    """Initialise per-worker env, encoders, and goals.

    Called once by each pool worker process at creation time.
    """
    global _w_env, _w_se, _w_ge, _w_goals
    global _w_num_actions, _w_num_agents, _w_robot_idx, _w_human_idx

    from gym_multigrid.multigrid import MultiGridEnv, SmallActions
    from empo.ali_learning_based.encoders import StateEncoder, GoalEncoder

    _w_env = MultiGridEnv(
        config_file=config_path,
        partial_obs=False,
        actions_set=SmallActions,
    )
    _w_env.reset()

    _w_robot_idx = robot_idx
    _w_human_idx = human_idx
    _w_num_actions = len(SmallActions.available)
    _w_num_agents = len(_w_env.agents)

    _w_se = StateEncoder(
        _w_env,
        robot_agent_index=robot_idx,
        human_agent_indices=[human_idx],
    )
    _w_ge = GoalEncoder(_w_env)

    state = _w_env.get_state()
    _w_goals = [g for g, _ in _w_env.possible_goal_generator.generate(state, human_idx)]


# ---------------------------------------------------------------------------
# Phase 1 worker
# ---------------------------------------------------------------------------

def _phase1_worker(task: dict) -> dict:
    """Collect one Phase 1 episode.  Runs in a worker process.

    Args (via task dict):
        goal_idx:           Index into the worker's goal list.
        q_net_state_dict:   CPU state_dict for the Q-network.
        epsilon:            Exploration probability.
        beta_h:             Boltzmann temperature.
        reward_shaping:     "none", "pbrs", or "pbrs_path".
        shaping_weight:     Multiplier for PBRS shaping.
        gamma:              Discount factor (for PBRS).
        random_start:       Whether to randomise the human's start position.
        done_on_goal:       Whether to terminate on goal achievement.

    Returns:
        Dict of tensors ready for EpisodeData(**result).
    """
    from empo.ali_learning_based.networks import QhNet
    from empo.ali_learning_based.phase1 import make_phase1_policy_fn
    from empo.ali_learning_based.rollout import (
        collect_episode, make_pbrs_shaper, make_path_pbrs_shaper,
    )

    goal = _w_goals[task["goal_idx"]]

    # Build Q-network on CPU, load weights
    q_net = QhNet.from_encoders(_w_se, _w_ge, num_actions=_w_num_actions)
    q_net.load_state_dict(task["q_net_state_dict"])
    q_net.eval()

    policy_fn = make_phase1_policy_fn(
        q_net=q_net,
        state_encoder=_w_se,
        goal_encoder=_w_ge,
        goal=goal,
        beta_h=task["beta_h"],
        epsilon=task["epsilon"],
        num_actions=_w_num_actions,
        num_agents=_w_num_agents,
        human_agent_idx=_w_human_idx,
        device=None,  # CPU
    )

    # Reward shaper
    shaper = None
    if task["reward_shaping"] == "pbrs":
        max_dist = _w_env.width + _w_env.height
        shaper = make_pbrs_shaper(
            goal, _w_human_idx, task["gamma"], max_dist, task["shaping_weight"],
        )
    elif task["reward_shaping"] == "pbrs_path":
        shaper = make_path_pbrs_shaper(
            goal, _w_human_idx, task["gamma"], _w_env, task["shaping_weight"],
        )

    randomize_agent = _w_human_idx if task["random_start"] else None

    ep = collect_episode(
        world_model=_w_env,
        state_encoder=_w_se,
        goal_encoder=_w_ge,
        goal=goal,
        policy_fn=policy_fn,
        record_agent_idx=_w_human_idx,
        reward_shaper=shaper,
        randomize_start_agent=randomize_agent,
        done_on_goal=task["done_on_goal"],
    )

    return {
        "states": ep.states,
        "goals": ep.goals,
        "actions": ep.actions,
        "rewards": ep.rewards,
        "next_states": ep.next_states,
        "dones": ep.dones,
        "goal_rewards": ep.goal_rewards,
    }


# ---------------------------------------------------------------------------
# Phase 2 worker
# ---------------------------------------------------------------------------

class _FrozenPhase1:
    """Lightweight stand-in for Phase1Trainer in worker processes.

    Implements the ``.get_policy(state_tuple, goal)`` and ``.beta_h``
    interface that ``make_phase2_policy_fn`` expects.
    """

    def __init__(self, q_net, state_encoder, goal_encoder, beta_h):
        self.q_net = q_net
        self.state_encoder = state_encoder
        self.goal_encoder = goal_encoder
        self.beta_h = beta_h

    def get_policy(self, state_tuple, goal):
        from empo.ali_learning_based.phase1 import boltzmann_probs
        state_enc = self.state_encoder.encode(state_tuple).unsqueeze(0)
        goal_enc = self.goal_encoder.encode(goal).unsqueeze(0)
        with torch.no_grad():
            q = self.q_net(state_enc, goal_enc).squeeze(0)
        return boltzmann_probs(q.unsqueeze(0), self.beta_h).squeeze(0)


def _phase2_worker(task: dict) -> dict:
    """Collect one Phase 2 episode.  Runs in a worker process.

    Args (via task dict):
        goal_idx:                Index into the worker's goal list.
        phase1_state_dict:       CPU state_dict for the frozen Phase 1 Q-net.
        qr_state_dict:           CPU state_dict for the Q_r network.
        epsilon:                 Robot exploration probability.
        beta_h:                  Phase 1 Boltzmann temperature.
        rock_seek_prob:          Probability of rock-seeking actions.
        rock_curiosity_bonus:    Push bonus for rock shaper.
        rock_approach_weight:    Approach PBRS weight for rock shaper.
        gamma_r:                 Robot discount factor.
        random_start:            Whether to randomise the robot's start.
        done_on_goal:            Whether to terminate on goal achievement.

    Returns:
        Dict of tensors ready for EpisodeData(**result).
    """
    from empo.ali_learning_based.networks import QhNet, QrNet
    from empo.ali_learning_based.phase2 import make_phase2_policy_fn
    from empo.ali_learning_based.rollout import (
        collect_episode, make_rock_curiosity_shaper,
    )

    goal = _w_goals[task["goal_idx"]]

    # Rebuild frozen Phase 1 network
    p1_net = QhNet.from_encoders(_w_se, _w_ge, num_actions=_w_num_actions)
    p1_net.load_state_dict(task["phase1_state_dict"])
    p1_net.eval()
    frozen_p1 = _FrozenPhase1(p1_net, _w_se, _w_ge, task["beta_h"])

    # Rebuild Q_r network
    qr_net = QrNet.from_encoders(_w_se, num_actions=_w_num_actions)
    qr_net.load_state_dict(task["qr_state_dict"])
    qr_net.eval()

    policy_fn = make_phase2_policy_fn(
        phase1_trainer=frozen_p1,
        q_r_net=qr_net,
        state_encoder=_w_se,
        goals=_w_goals,
        epsilon=task["epsilon"],
        num_actions=_w_num_actions,
        num_agents=_w_num_agents,
        robot_agent_idx=_w_robot_idx,
        human_agent_idx=_w_human_idx,
        human_goal=goal,
        device=None,  # CPU
        rock_seek_prob=task["rock_seek_prob"],
    )

    # Rock curiosity shaper
    shaper = None
    if task["rock_curiosity_bonus"] > 0 or task["rock_approach_weight"] > 0:
        shaper = make_rock_curiosity_shaper(
            push_bonus=task["rock_curiosity_bonus"],
            approach_weight=task["rock_approach_weight"],
            robot_agent_idx=_w_robot_idx,
            gamma=task["gamma_r"],
        )

    randomize_agent = _w_robot_idx if task["random_start"] else None

    ep = collect_episode(
        world_model=_w_env,
        state_encoder=_w_se,
        goal_encoder=_w_ge,
        goal=goal,
        policy_fn=policy_fn,
        record_agent_idx=_w_robot_idx,
        reward_shaper=shaper,
        randomize_start_agent=randomize_agent,
        done_on_goal=task["done_on_goal"],
    )

    return {
        "states": ep.states,
        "goals": ep.goals,
        "actions": ep.actions,
        "rewards": ep.rewards,
        "next_states": ep.next_states,
        "dones": ep.dones,
        "goal_rewards": ep.goal_rewards,
    }


# ---------------------------------------------------------------------------
# ParallelCollector
# ---------------------------------------------------------------------------

class ParallelCollector:
    """Manages a persistent multiprocessing pool for episode collection.

    Args:
        config_path: Absolute path to the YAML environment config.
        robot_idx: Robot agent index.
        human_idx: Human agent index.
        num_workers: Number of worker processes.
    """

    def __init__(
        self,
        config_path: str,
        robot_idx: int,
        human_idx: int,
        num_workers: int,
    ):
        self.num_workers = num_workers
        # Use 'spawn' to avoid fork-safety issues on macOS
        ctx = mp.get_context("spawn")
        self.pool = ctx.Pool(
            processes=num_workers,
            initializer=_init_worker,
            initargs=(config_path, robot_idx, human_idx),
        )

    def collect_phase1_episodes(self, tasks: List[dict]) -> List[EpisodeData]:
        """Collect Phase 1 episodes in parallel.

        Args:
            tasks: List of task dicts (see ``_phase1_worker`` for keys).

        Returns:
            List of EpisodeData, one per task.
        """
        results = self.pool.map(_phase1_worker, tasks)
        return [EpisodeData(**r) for r in results]

    def collect_phase2_episodes(self, tasks: List[dict]) -> List[EpisodeData]:
        """Collect Phase 2 episodes in parallel.

        Args:
            tasks: List of task dicts (see ``_phase2_worker`` for keys).

        Returns:
            List of EpisodeData, one per task.
        """
        results = self.pool.map(_phase2_worker, tasks)
        return [EpisodeData(**r) for r in results]

    def close(self):
        """Shut down the worker pool."""
        self.pool.terminate()
        self.pool.join()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

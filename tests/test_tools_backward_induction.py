"""
Integration test: Tools WorldModel with backward induction (Phase 1 + Phase 2).

Uses a small environment (2 agents, 2 tools, 3 steps) to keep the state space
manageable so the test finishes quickly.
"""

import numpy as np

from empo.backward_induction import compute_human_policy_prior, compute_robot_policy
from empo.world_specific_helpers.tools import (
    ToolsGoalGenerator,
    create_tools_env,
)


def test_tools_backward_induction_small():
    """End-to-end test: create small tools env, run Phase 1 + Phase 2 backward induction."""
    env = create_tools_env(n_agents=2, n_tools=2, max_steps=3, seed=42)
    env.reset(seed=42)

    goal_gen = ToolsGoalGenerator(env)

    # Phase 1
    human_policy = compute_human_policy_prior(
        env,
        env.human_agent_indices,
        goal_gen,
        beta_h=5.0,
        quiet=True,
    )

    # Phase 2
    robot_policy = compute_robot_policy(
        env,
        env.human_agent_indices,
        env.robot_agent_indices,
        goal_gen,
        human_policy,
        beta_r=5.0,
        quiet=True,
    )

    # Verify robot policy produces valid actions
    state = env.initial_state()
    action = robot_policy.sample(state)
    assert action is not None
    assert isinstance(action, tuple)
    assert len(action) == len(env.robot_agent_indices)
    for i, r_idx in enumerate(env.robot_agent_indices):
        assert 0 <= action[i] < env.n_actions_per_agent[r_idx]

    # Verify human policy produces valid distributions
    for hi in env.human_agent_indices:
        dist = human_policy(state, hi)
        assert dist is not None
        assert len(dist) == env.n_actions_per_agent[hi]
        assert abs(dist.sum() - 1.0) < 1e-6


def test_tools_backward_induction_rollout():
    """Verify a short rollout works with backward-induction policies."""
    env = create_tools_env(n_agents=2, n_tools=2, max_steps=3, seed=42)
    env.reset(seed=42)

    goal_gen = ToolsGoalGenerator(env)

    human_policy = compute_human_policy_prior(
        env,
        env.human_agent_indices,
        goal_gen,
        beta_h=5.0,
        quiet=True,
    )
    robot_policy = compute_robot_policy(
        env,
        env.human_agent_indices,
        env.robot_agent_indices,
        goal_gen,
        human_policy,
        beta_r=5.0,
        quiet=True,
    )

    env.reset(seed=42)
    for _step in range(env.max_steps):
        state = env.get_state()
        robot_action = robot_policy.sample(state)
        actions = [0] * env.n_agents
        for i, r_idx in enumerate(env.robot_agent_indices):
            actions[r_idx] = robot_action[i]
        for hi in env.human_agent_indices:
            dist = human_policy(state, hi)
            if dist is not None and dist.sum() > 0:
                actions[hi] = np.random.choice(len(dist), p=dist)
        _obs, _reward, terminated, _truncated, _info = env.step(actions)
        if terminated:
            break

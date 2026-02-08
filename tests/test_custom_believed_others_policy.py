#!/usr/bin/env python3
"""
Test to verify that custom believed_others_policy functions work in parallel mode.

This tests the fix for BUG-002: Custom believed_others_policy not supported in parallel mode.
The fix uses cloudpickle to serialize custom functions (including lambdas and closures)
for use in forked worker processes.
"""

import sys
import numpy as np
from itertools import product
from pathlib import Path

current_file = Path(__file__).resolve()
repo_root = current_file.parent.parent
multigrid_path = repo_root / "vendor" / "multigrid"

# Avoiding unfound modules
for p in [str(repo_root), str(multigrid_path)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from multigrid_worlds.one_or_three_chambers import SmallOneOrThreeChambersMapEnv
from empo.possible_goal import PossibleGoal, PossibleGoalGenerator
from empo.backward_induction import compute_human_policy_prior


class ReachCellGoal(PossibleGoal):
    """A goal where a specific human agent tries to reach a specific cell."""

    def __init__(self, world_model, human_agent_index: int, target_pos: tuple):
        super().__init__(world_model)
        self.human_agent_index = human_agent_index
        self.target_pos = np.array(target_pos)
        self._hash = hash((self.human_agent_index, tuple(self.target_pos)))
        super()._freeze()  # Make immutable

    def is_achieved(self, state) -> int:
        """Returns 1 if the specific human agent is at the target position, 0 otherwise."""
        step_count, agent_states, mobile_objects, mutable_objects = state
        agent_state = agent_states[self.human_agent_index]
        agent_x, agent_y = agent_state[0], agent_state[1]
        if agent_x == self.target_pos[0] and agent_y == self.target_pos[1]:
            return 1
        return 0

    def __str__(self):
        return f"ReachCell(agent_{self.human_agent_index}_to_{self.target_pos[0]},{self.target_pos[1]})"

    def __hash__(self):
        return self._hash

    def __eq__(self, other):
        return (
            isinstance(other, ReachCellGoal)
            and self.human_agent_index == other.human_agent_index
            and np.array_equal(self.target_pos, other.target_pos)
        )


class SimpleGoalGenerator(PossibleGoalGenerator):
    """Generates a small number of goals for testing."""

    def __init__(self, world_model, target_cells):
        super().__init__(world_model)
        self.target_cells = target_cells

    def generate(self, state, human_agent_index: int):
        for pos in self.target_cells:
            goal = ReachCellGoal(self.env, human_agent_index, pos)
            yield (goal, 1.0 / len(self.target_cells))


def create_custom_believed_others_policy(
    num_agents, num_actions, robot_agent_indices=None
):
    """
    Creates a custom believed_others_policy function using a closure.

    This function returns a policy that is functionally equivalent to the default
    uniform policy, but tests the cloudpickle serialization path because it
    captures num_agents and num_actions in its closure.

    Args:
        num_agents: Total number of agents
        num_actions: Number of actions per agent
        robot_agent_indices: List of robot agent indices (their actions will be set to -1)
    """
    if robot_agent_indices is None:
        robot_agent_indices = []
    robot_set = set(robot_agent_indices)
    all_actions = list(range(num_actions))
    num_other_humans = num_agents - 1 - len(robot_agent_indices)
    uniform_p = 1 / (num_actions**num_other_humans) if num_other_humans > 0 else 1.0

    def custom_policy(state, agent_index, action):
        """Custom policy with captured closure variables."""
        return [
            (uniform_p, np.array(action_profile, dtype=np.int64))
            for action_profile in product(
                *[
                    [-1] if (idx == agent_index or idx in robot_set) else all_actions
                    for idx in range(num_agents)
                ]
            )
        ]

    return custom_policy


def compare_policies(pol1, pol2, rtol=1e-5, atol=1e-8):
    """Compare two policy dictionaries and return differences."""
    differences = []

    # Check all states in pol1
    for state in pol1:
        if state not in pol2:
            differences.append(f"State {state} in pol1 but not in pol2")
            continue

        for agent_idx in pol1[state]:
            if agent_idx not in pol2[state]:
                differences.append(
                    f"Agent {agent_idx} missing in pol2 for state {state}"
                )
                continue

            for goal in pol1[state][agent_idx]:
                if goal not in pol2[state][agent_idx]:
                    differences.append(
                        f"Goal {goal} missing in pol2 for state {state}, agent {agent_idx}"
                    )
                    continue

                p1 = pol1[state][agent_idx][goal]
                p2 = pol2[state][agent_idx][goal]

                if not np.allclose(p1, p2, rtol=rtol, atol=atol):
                    max_diff = np.max(np.abs(p1 - p2))
                    differences.append(
                        f"Policy mismatch for state {state}, agent {agent_idx}, goal {goal}: "
                        f"max_diff={max_diff:.2e}"
                    )

    # Check for extra states in pol2
    for state in pol2:
        if state not in pol1:
            differences.append(f"State {state} in pol2 but not in pol1")

    return differences


def test_custom_believed_others_policy_parallel():
    """Test that custom believed_others_policy works in parallel mode."""
    # Create environment
    wm = SmallOneOrThreeChambersMapEnv()
    wm.max_steps = 2
    wm.reset()

    num_agents = len(wm.agents)
    num_actions = wm.action_space.n

    target_cells = [(0, 0), (5, 5)]
    goal_gen = SimpleGoalGenerator(wm, target_cells)
    human_agent_indices = [0, 1]
    robot_agent_indices = [i for i in range(num_agents) if i not in human_agent_indices]

    # Create a custom believed_others_policy using a closure
    custom_policy = create_custom_believed_others_policy(
        num_agents, num_actions, robot_agent_indices
    )

    # Run with custom policy in sequential mode
    result_seq = compute_human_policy_prior(
        wm,
        human_agent_indices,
        goal_gen,
        believed_others_policy=custom_policy,
        parallel=False,
    )

    # Reset and run with custom policy in parallel mode
    wm.reset()
    result_par = compute_human_policy_prior(
        wm,
        human_agent_indices,
        goal_gen,
        believed_others_policy=custom_policy,
        parallel=True,
        level_fct=lambda s: s[0],
    )

    # Compare results
    differences = compare_policies(result_seq.values, result_par.values)

    assert len(differences) == 0, (
        f"Found {len(differences)} differences: {differences[:5]}"
    )
    assert len(result_seq.values) == len(result_par.values)


def test_custom_believed_others_policy_consistency():
    """Test that custom and default policies produce different (but valid) results."""
    # Create environment
    wm = SmallOneOrThreeChambersMapEnv()
    wm.max_steps = 2
    wm.reset()

    num_agents = len(wm.agents)
    num_actions = wm.action_space.n

    target_cells = [(0, 0), (5, 5)]
    goal_gen = SimpleGoalGenerator(wm, target_cells)
    human_agent_indices = [0, 1]
    robot_agent_indices = [i for i in range(num_agents) if i not in human_agent_indices]

    # Run with default policy
    result_default = compute_human_policy_prior(
        wm,
        human_agent_indices,
        goal_gen,
        believed_others_policy=None,  # Use default
        parallel=True,
        level_fct=lambda s: s[0],
    )

    # Reset and run with custom policy (equivalent to default)
    wm.reset()
    custom_policy = create_custom_believed_others_policy(
        num_agents, num_actions, robot_agent_indices
    )
    result_custom = compute_human_policy_prior(
        wm,
        human_agent_indices,
        goal_gen,
        believed_others_policy=custom_policy,
        parallel=True,
        level_fct=lambda s: s[0],
    )

    # Results should be identical since the custom policy is equivalent to default
    differences = compare_policies(result_default.values, result_custom.values)

    assert len(differences) == 0, (
        f"Found {len(differences)} differences: {differences[:5]}"
    )


def test_lambda_believed_others_policy():
    """Test that lambda-based believed_others_policy works in parallel mode."""
    # Create environment
    wm = SmallOneOrThreeChambersMapEnv()
    wm.max_steps = 2
    wm.reset()

    num_agents = len(wm.agents)
    num_actions = wm.action_space.n

    target_cells = [(0, 0), (5, 5)]
    goal_gen = SimpleGoalGenerator(wm, target_cells)
    human_agent_indices = [0, 1]
    robot_agent_indices = [i for i in range(num_agents) if i not in human_agent_indices]
    robot_set = set(robot_agent_indices)

    # Create a lambda-based policy (these are harder to pickle with standard pickle)
    all_actions = list(range(num_actions))
    num_other_humans = num_agents - 1 - len(robot_agent_indices)
    uniform_p = 1 / (num_actions**num_other_humans) if num_other_humans > 0 else 1.0

    lambda_policy = lambda state, agent_index, action: [
        (uniform_p, np.array(action_profile, dtype=np.int64))
        for action_profile in product(
            *[
                [-1] if (idx == agent_index or idx in robot_set) else all_actions
                for idx in range(num_agents)
            ]
        )
    ]

    # This should NOT raise NotImplementedError anymore
    result = compute_human_policy_prior(
        wm,
        human_agent_indices,
        goal_gen,
        believed_others_policy=lambda_policy,
        parallel=True,
        level_fct=lambda s: s[0],
    )

    # Should have policies computed
    assert len(result.values) > 0


if __name__ == "__main__":
    print("Testing custom believed_others_policy in parallel mode...")

    print("\n1. Testing custom policy parallel execution...")
    test_custom_believed_others_policy_parallel()
    print("   PASSED!")

    print("\n2. Testing custom vs default policy consistency...")
    test_custom_believed_others_policy_consistency()
    print("   PASSED!")

    print("\n3. Testing lambda-based policy...")
    test_lambda_believed_others_policy()
    print("   PASSED!")

    print("\nAll tests passed!")

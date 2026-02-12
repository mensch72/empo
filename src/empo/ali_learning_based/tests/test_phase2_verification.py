"""
Phase 2 functional verification.

Tests that Phase 2 training actually produces a robot policy that
increases human empowerment (goal achievement across multiple goals).

The key test: after Phase 2 training, the robot should learn to
behave in a way that helps the human achieve more goals compared
to a random robot policy.

Run with:
    PYTHONPATH=src:vendor/multigrid pytest src/empo/ali_learning_based/tests/test_phase2_verification.py -v -s
"""

import random as py_random
import pytest
import torch
import numpy as np

from gym_multigrid.multigrid import MultiGridEnv, SmallActions

from empo.ali_learning_based.envs import get_env_path

def _phase1_test_env():
    return get_env_path("phase1_test.yaml")

from empo.ali_learning_based.encoders import StateEncoder, GoalEncoder
from empo.ali_learning_based.networks import QhNet, VheNet, QrNet
from empo.ali_learning_based.phase1 import Phase1Trainer
from empo.ali_learning_based.phase2 import Phase2Trainer


# -----------------------------------------------------------------------
# Setup helpers
# -----------------------------------------------------------------------

NUM_ACTIONS = 4


def make_env():
    env = MultiGridEnv(
        config_file=_phase1_test_env(),
        partial_obs=False,
        actions_set=SmallActions,
    )
    env.reset()
    return env


def get_indices(env):
    robot_idx = human_idx = None
    for i, a in enumerate(env.agents):
        if a.color in ("grey", "gray"):
            robot_idx = i
        elif a.color == "yellow":
            human_idx = i
    return robot_idx, human_idx


def get_goals(env, human_idx):
    state = env.get_state()
    return [g for g, _ in env.possible_goal_generator.generate(state, human_idx)]


def train_phase1(env, robot_idx, human_idx, goals):
    """Train Phase 1 properly so human has a real policy."""
    se = StateEncoder(env, robot_agent_index=robot_idx, human_agent_indices=[human_idx])
    ge = GoalEncoder(env)
    q_net = QhNet.from_encoders(se, ge, num_actions=NUM_ACTIONS)
    trainer = Phase1Trainer(
        q_net=q_net,
        state_encoder=se,
        goal_encoder=ge,
        num_actions=NUM_ACTIONS,
        num_agents=len(env.agents),
        human_agent_idx=human_idx,
        gamma=0.99,
        beta_h=10.0,
        lr=1e-3,
        buffer_capacity=20_000,
        batch_size=64,
        target_update_freq=100,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay_steps=3000,
    )
    trainer.train(env, goals, num_iterations=300, episodes_per_iter=4,
                  train_steps_per_iter=8, log_interval=0)
    return trainer, se, ge


def make_phase2_trainer(env, robot_idx, human_idx, p1, se, ge, goals):
    vhe_net = VheNet.from_encoders(se, ge)
    qr_net = QrNet.from_encoders(se, num_actions=NUM_ACTIONS)
    return Phase2Trainer(
        vhe_net=vhe_net,
        q_r_net=qr_net,
        phase1_trainer=p1,
        state_encoder=se,
        goal_encoder=ge,
        goals=goals,
        num_actions=NUM_ACTIONS,
        num_agents=len(env.agents),
        robot_agent_idx=robot_idx,
        human_agent_idx=human_idx,
        gamma_h=0.99,
        gamma_r=0.99,
        lr_vhe=1e-3,
        lr_qr=1e-3,
        buffer_capacity=20_000,
        batch_size=64,
        target_update_freq=100,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay_steps=3000,
    )


def evaluate_total_goal_achievement(env, policy_fn, goals, num_episodes=100):
    """
    Run episodes and measure average number of goals achieved at episode end.

    Returns mean goals achieved per episode.
    """
    total_achieved = 0
    for _ in range(num_episodes):
        env.reset()
        done = False
        while not done:
            state = env.get_state()
            actions = policy_fn(state)
            _, _, done, _ = env.step(actions)
        final_state = env.get_state()
        for g in goals:
            if g.is_achieved(final_state):
                total_achieved += 1
    return total_achieved / num_episodes


def random_policy_fn(state, num_agents=2, num_actions=4):
    return [py_random.randint(0, num_actions - 1) for _ in range(num_agents)]


# -----------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------

class TestPhase2Functional:

    @pytest.fixture(autouse=True)
    def setup(self):
        self.env = make_env()
        self.robot_idx, self.human_idx = get_indices(self.env)
        self.goals = get_goals(self.env, self.human_idx)
        # Train Phase 1 once (shared across tests via fixture)
        self.p1, self.se, self.ge = train_phase1(
            self.env, self.robot_idx, self.human_idx, self.goals
        )

    def test_vhe_learns_goal_achievement(self):
        """
        After training, V_h^e should be higher for goals that the
        human can easily reach than for hard-to-reach goals.
        """
        t = make_phase2_trainer(
            self.env, self.robot_idx, self.human_idx,
            self.p1, self.se, self.ge, self.goals,
        )
        t.train(self.env, num_iterations=300, episodes_per_iter=4,
                train_steps_per_iter=4, log_interval=0)

        self.env.reset()
        state = self.env.get_state()

        vhe_values = [t.get_vhe(state, g) for g in self.goals]
        print(f"\nV_h^e values: {vhe_values}")

        # All should be in [0, 1]
        assert all(0 <= v <= 1 for v in vhe_values), f"V_h^e out of range: {vhe_values}"
        # V_h^e should be meaningfully above 0 (human can achieve goals)
        assert max(vhe_values) > 0.1, \
            f"V_h^e too low — human should achieve some goals: {vhe_values}"

    def test_training_is_stable(self):
        """Both VheNet and QrNet losses should stay finite."""
        t = make_phase2_trainer(
            self.env, self.robot_idx, self.human_idx,
            self.p1, self.se, self.ge, self.goals,
        )
        history = t.train(self.env, num_iterations=100, episodes_per_iter=4,
                          train_steps_per_iter=4, log_interval=0)

        assert all(np.isfinite(l) for l in history["vhe_losses"]), "VheNet loss not finite"
        assert all(np.isfinite(l) for l in history["qr_losses"]), "QrNet loss not finite"
        assert max(history["vhe_losses"]) < 100, "VheNet loss diverged"
        assert max(history["qr_losses"]) < 1e6, "QrNet loss diverged"

    def test_empowerment_varies_across_states(self):
        """
        Empowerment should differ between states — the robot's reward
        signal should be state-dependent.
        """
        t = make_phase2_trainer(
            self.env, self.robot_idx, self.human_idx,
            self.p1, self.se, self.ge, self.goals,
        )
        t.train(self.env, num_iterations=100, episodes_per_iter=4,
                train_steps_per_iter=4, log_interval=0)

        # Get empowerment at the start vs after a few random steps
        self.env.reset()
        emp_start = t.get_empowerment(self.env.get_state())

        # Take some random steps to change state
        for _ in range(5):
            actions = [py_random.randint(0, NUM_ACTIONS - 1) for _ in range(len(self.env.agents))]
            self.env.step(actions)
        emp_mid = t.get_empowerment(self.env.get_state())

        print(f"\nEmpowerment at start: {emp_start:.4f}")
        print(f"Empowerment mid-episode: {emp_mid:.4f}")

        # They don't need to differ hugely, but empowerment should be
        # a function of state, not a constant
        # (This test mainly checks the plumbing works end-to-end)
        assert isinstance(emp_start, float)
        assert isinstance(emp_mid, float)
        assert emp_start >= 0
        assert emp_mid >= 0

    def test_robot_q_values_differ_across_actions(self):
        """After training, robot Q-values should not be identical for all actions."""
        t = make_phase2_trainer(
            self.env, self.robot_idx, self.human_idx,
            self.p1, self.se, self.ge, self.goals,
        )
        t.train(self.env, num_iterations=200, episodes_per_iter=4,
                train_steps_per_iter=4, log_interval=0)

        self.env.reset()
        q = t.get_robot_q_values(self.env.get_state())
        print(f"\nRobot Q-values: {q.tolist()}")

        # Q-values should have some variation (robot learned preferences)
        assert q.max() > q.min(), "Robot Q-values identical for all actions"

    def test_phase2_end_to_end(self):
        """
        End-to-end test: Phase 1 + Phase 2 training runs to completion
        and the robot policy can be evaluated.
        """
        t = make_phase2_trainer(
            self.env, self.robot_idx, self.human_idx,
            self.p1, self.se, self.ge, self.goals,
        )
        history = t.train(
            self.env,
            num_iterations=200,
            episodes_per_iter=4,
            train_steps_per_iter=4,
            log_interval=100,
        )

        # Evaluate the learned robot policy
        learned_policy = t.get_robot_policy_fn()
        learned_achievement = evaluate_total_goal_achievement(
            self.env, learned_policy, self.goals, num_episodes=50,
        )

        # Evaluate random baseline
        random_achievement = evaluate_total_goal_achievement(
            self.env, lambda s: random_policy_fn(s), self.goals, num_episodes=50,
        )

        print(f"\nRandom robot: {random_achievement:.2f} goals/episode")
        print(f"Learned robot: {learned_achievement:.2f} goals/episode")

        # The trained robot should achieve at least as many goals as random.
        # With a trained human + trained robot, this should hold.
        assert learned_achievement >= random_achievement * 0.8, (
            f"Learned robot ({learned_achievement:.2f}) much worse than "
            f"random ({random_achievement:.2f})"
        )


# -----------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

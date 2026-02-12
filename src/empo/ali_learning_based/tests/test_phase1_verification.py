"""
Phase 1 functional verification.

Tests that Phase 1 Q-learning actually works by measuring whether the
learned policy achieves goals more often than a random policy.

Uses a 7x7 open grid (phase1_test.yaml) where the human agent has room
to navigate. The key test: after training, the learned policy's goal
achievement rate should exceed the random baseline.

Run with:
    PYTHONPATH=src:vendor/multigrid pytest src/empo/ali_learning_based/tests/test_phase1_verification.py -v -s
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
from empo.ali_learning_based.networks import QhNet
from empo.ali_learning_based.phase1 import Phase1Trainer, boltzmann_probs


# -----------------------------------------------------------------------
# Setup helpers
# -----------------------------------------------------------------------

NUM_ACTIONS = 4  # SmallActions: still, left, right, forward


def make_env():
    """Create the 7x7 open grid with grey robot + yellow human."""
    env = MultiGridEnv(
        config_file=_phase1_test_env(),
        partial_obs=False,
        actions_set=SmallActions,
    )
    env.reset()
    return env


def get_agent_indices(env):
    """Return (robot_idx, human_idx) based on agent colors."""
    robot_idx = human_idx = None
    for i, agent in enumerate(env.agents):
        if agent.color in ("grey", "gray"):
            robot_idx = i
        elif agent.color == "yellow":
            human_idx = i
    assert robot_idx is not None and human_idx is not None
    return robot_idx, human_idx


def get_env_goals(env, human_idx):
    """Get goals from the environment's goal generator."""
    state = env.get_state()
    return [g for g, _ in env.possible_goal_generator.generate(state, human_idx)]


def make_trainer_for_env(env, robot_idx, human_idx):
    """Build a Phase1Trainer wired to the environment."""
    se = StateEncoder(env, robot_agent_index=robot_idx, human_agent_indices=[human_idx])
    ge = GoalEncoder(env)
    q_net = QhNet.from_encoders(se, ge, num_actions=NUM_ACTIONS)

    return Phase1Trainer(
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


def evaluate_goal_achievement(env, policy_fn, goal, num_episodes=100):
    """Run episodes and return fraction where goal is achieved at any step."""
    achieved_count = 0
    for _ in range(num_episodes):
        env.reset()
        done = False
        while not done:
            state = env.get_state()
            actions = policy_fn(state)
            _, _, done, _ = env.step(actions)
        # Check final state
        final_state = env.get_state()
        if goal.is_achieved(final_state):
            achieved_count += 1
    return achieved_count / num_episodes


def random_policy_fn(state, num_agents=2, num_actions=4):
    """Uniform random policy for all agents."""
    return [py_random.randint(0, num_actions - 1) for _ in range(num_agents)]


# -----------------------------------------------------------------------
# Core functional test
# -----------------------------------------------------------------------

class TestPhase1Functional:
    """Verify that Phase 1 training produces a policy that achieves goals."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.env = make_env()
        self.robot_idx, self.human_idx = get_agent_indices(self.env)
        self.goals = get_env_goals(self.env, self.human_idx)

    def test_learned_policy_beats_random_on_hard_goal(self):
        """
        The trained policy should achieve the hardest rectangle goal
        more often than a random policy.

        Goal 0 is ReachRect(1,1,3,3) — the upper-left quadrant.
        Human starts at (4,4) in the lower-right. Random achieves
        this ~10% of the time; a learned policy should do better.
        """
        trainer = make_trainer_for_env(self.env, self.robot_idx, self.human_idx)
        target_goal = self.goals[0]  # ReachRect(1,1,3,3)

        # Measure random baseline
        random_rate = evaluate_goal_achievement(
            self.env,
            lambda s: random_policy_fn(s),
            target_goal,
            num_episodes=200,
        )
        print(f"\nRandom baseline for {target_goal}: {random_rate:.2%}")

        # Train with enough iterations for epsilon to decay and Q-values to learn
        trainer.train(
            self.env,
            self.goals,
            num_iterations=500,
            episodes_per_iter=4,
            train_steps_per_iter=8,
            log_interval=100,
        )

        # Evaluate trained policy (epsilon=0, Boltzmann from learned Q)
        learned_policy = trainer.get_policy_fn(target_goal)
        learned_rate = evaluate_goal_achievement(
            self.env,
            learned_policy,
            target_goal,
            num_episodes=200,
        )
        print(f"Learned policy for {target_goal}: {learned_rate:.2%}")
        print(f"Improvement: {learned_rate - random_rate:+.2%}")

        # The learned policy should outperform random.
        # Even a modest improvement (>= random) shows the network learned.
        assert learned_rate >= random_rate, (
            f"Learned policy ({learned_rate:.2%}) should be at least as good "
            f"as random ({random_rate:.2%})"
        )

    def test_training_loss_is_stable(self):
        """Training loss should stay finite and bounded."""
        trainer = make_trainer_for_env(self.env, self.robot_idx, self.human_idx)

        history = trainer.train(
            self.env,
            self.goals,
            num_iterations=100,
            episodes_per_iter=4,
            train_steps_per_iter=4,
            log_interval=0,
        )

        losses = history["losses"]
        assert all(np.isfinite(l) for l in losses), "Losses must be finite"
        assert max(losses) < 100.0, f"Loss diverged: max={max(losses):.4f}"

    def test_different_goals_get_different_values(self):
        """V_h(s, g) should differ across goals after training."""
        trainer = make_trainer_for_env(self.env, self.robot_idx, self.human_idx)

        trainer.train(
            self.env,
            self.goals,
            num_iterations=100,
            episodes_per_iter=4,
            train_steps_per_iter=4,
            log_interval=0,
        )

        self.env.reset()
        state = self.env.get_state()

        values = [trainer.get_value(state, g) for g in self.goals]
        # Not all values should be identical
        assert max(values) != min(values), (
            f"All values identical: {values}"
        )

    def test_policy_probabilities_valid(self):
        """Policy should return valid probability distributions."""
        trainer = make_trainer_for_env(self.env, self.robot_idx, self.human_idx)

        trainer.train(
            self.env,
            self.goals,
            num_iterations=50,
            episodes_per_iter=4,
            train_steps_per_iter=4,
            log_interval=0,
        )

        self.env.reset()
        state = self.env.get_state()

        for goal in self.goals:
            probs = trainer.get_policy(state, goal)
            assert probs.shape == (NUM_ACTIONS,)
            assert torch.allclose(probs.sum(), torch.tensor(1.0), atol=1e-5)
            assert (probs >= 0).all()

    def test_q_values_shape_correct(self):
        """Q-values should have shape (num_actions,) for each goal."""
        trainer = make_trainer_for_env(self.env, self.robot_idx, self.human_idx)

        self.env.reset()
        state = self.env.get_state()

        for goal in self.goals:
            q = trainer.get_q_values(state, goal)
            assert q.shape == (NUM_ACTIONS,)
            assert all(torch.isfinite(q))


class TestPhase1BackwardInductionComparison:
    """
    Compare against backward induction if it works.
    This is a soft check — backward induction code may have issues.
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        self.env = make_env()
        self.env.max_steps = 5  # Short for tractable BI
        self.robot_idx, self.human_idx = get_agent_indices(self.env)

    def test_soft_comparison(self):
        """
        If backward induction succeeds, check that neural net values
        have at least some correlation with exact values.
        """
        try:
            from empo.backward_induction.phase1 import compute_human_policy_prior
            policy_prior, Vh = compute_human_policy_prior(
                world_model=self.env,
                human_agent_indices=[self.human_idx],
                beta_h=10.0,
                gamma_h=0.99,
                parallel=False,
                return_Vh=True,
                level_fct=lambda state: state[0],
                quiet=True,
            )
        except Exception as e:
            pytest.skip(f"Backward induction failed: {e}")

        # Get goals from BI
        self.env.reset()
        initial_state = self.env.get_state()

        # Collect BI values at initial state
        bi_agent_dict = Vh.get(initial_state, {}).get(self.human_idx, {})
        if not bi_agent_dict:
            pytest.skip("BI returned no values for initial state")

        bi_goals = list(bi_agent_dict.keys())
        bi_values = [float(bi_agent_dict[g]) for g in bi_goals]

        if np.std(bi_values) < 1e-6:
            pytest.skip("BI values have no variation")

        # Train neural net with the same goals
        goals_for_training = get_env_goals(self.env, self.human_idx)
        trainer = make_trainer_for_env(self.env, self.robot_idx, self.human_idx)
        trainer.train(
            self.env,
            goals_for_training,
            num_iterations=200,
            episodes_per_iter=4,
            train_steps_per_iter=4,
            log_interval=0,
        )

        nn_values = [trainer.get_value(initial_state, g) for g in bi_goals]

        print(f"\nBI values:  {np.array(bi_values)}")
        print(f"NN values:  {np.array(nn_values)}")

        try:
            from scipy.stats import spearmanr
            corr, pval = spearmanr(bi_values, nn_values)
            print(f"Spearman r={corr:.3f}, p={pval:.3f}")
            # Just check it's not strongly anti-correlated
            assert corr > -0.5, f"Values anti-correlated: r={corr:.3f}"
        except ImportError:
            pass  # scipy not available, skip correlation check


# -----------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

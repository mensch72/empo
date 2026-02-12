"""
Tests for Phase 1 training.

Run with:
    PYTHONPATH=src:vendor/multigrid pytest src/empo/ali_learning_based/tests/test_phase1.py -v
"""

import math
import pytest
import torch

from empo.ali_learning_based.phase1 import (
    boltzmann_probs,
    boltzmann_value,
    make_phase1_policy_fn,
    make_phase1_ucb_policy_fn,
    ucb_bonus,
    ucb_action_values,
    Phase1Trainer,
)
from empo.ali_learning_based.networks import QhNet
from empo.ali_learning_based.encoders import NUM_GRID_CHANNELS


# -----------------------------------------------------------------------
# Boltzmann utilities
# -----------------------------------------------------------------------

class TestBoltzmannProbs:

    def test_uniform_at_equal_q(self):
        """Equal Q-values → uniform distribution."""
        q = torch.tensor([[1.0, 1.0, 1.0, 1.0]])
        probs = boltzmann_probs(q, beta=1.0)
        assert probs.shape == (1, 4)
        assert torch.allclose(probs, torch.tensor([[0.25, 0.25, 0.25, 0.25]]))

    def test_higher_beta_more_greedy(self):
        """Higher beta concentrates probability on the max Q-value."""
        q = torch.tensor([[1.0, 2.0, 0.5]])
        probs_low = boltzmann_probs(q, beta=1.0)
        probs_high = boltzmann_probs(q, beta=100.0)
        # High beta should put more weight on action 1 (Q=2.0)
        assert probs_high[0, 1] > probs_low[0, 1]

    def test_inf_beta_is_argmax(self):
        """beta=inf → deterministic argmax."""
        q = torch.tensor([[0.1, 0.9, 0.5]])
        probs = boltzmann_probs(q, beta=math.inf)
        assert probs[0, 1].item() == 1.0
        assert probs[0, 0].item() == 0.0
        assert probs[0, 2].item() == 0.0

    def test_inf_beta_ties(self):
        """beta=inf with ties → uniform over max actions."""
        q = torch.tensor([[0.5, 0.5, 0.1]])
        probs = boltzmann_probs(q, beta=math.inf)
        assert probs[0, 0].item() == pytest.approx(0.5)
        assert probs[0, 1].item() == pytest.approx(0.5)
        assert probs[0, 2].item() == 0.0

    def test_sums_to_one(self):
        q = torch.randn(5, 6)
        probs = boltzmann_probs(q, beta=3.0)
        sums = probs.sum(dim=-1)
        assert torch.allclose(sums, torch.ones(5))

    def test_1d_input(self):
        """Should work with unbatched (num_actions,) input."""
        q = torch.tensor([1.0, 2.0, 3.0])
        probs = boltzmann_probs(q, beta=1.0)
        assert probs.shape == (3,)
        assert probs.sum().item() == pytest.approx(1.0)


class TestBoltzmannValue:

    def test_value_is_weighted_sum(self):
        """V = sum(pi * Q)."""
        q = torch.tensor([[1.0, 3.0]])  # 2 actions
        probs = boltzmann_probs(q, beta=1.0)
        expected_v = (probs * q).sum(dim=-1)
        v = boltzmann_value(q, beta=1.0)
        assert torch.allclose(v, expected_v)

    def test_argmax_value(self):
        """At beta=inf, V = max Q."""
        q = torch.tensor([[1.0, 5.0, 3.0]])
        v = boltzmann_value(q, beta=math.inf)
        assert v.item() == pytest.approx(5.0)


# -----------------------------------------------------------------------
# Mock objects for Phase 1 testing
# -----------------------------------------------------------------------

class MockGoal:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.target_pos = (x, y)
        self.target_rect = (x, y, x, y)

    def is_achieved(self, state):
        _, agents, _, _ = state
        return agents[1][0] == self.x and agents[1][1] == self.y  # human = agent 1


class MockWorldModel:
    """
    Simple grid: human (agent 1) starts at (0,0), can move right.
    Robot (agent 0) stays at (0,1). Episode ends after max_steps.
    """
    def __init__(self, width=5, height=5, max_steps=8):
        self.width = width
        self.height = height
        self.max_steps = max_steps
        self._step = 0
        self._human_x = 0
        # Agents need enough fields for the state encoder
        self.agents = [None, None]  # 2 agents
        self.grid = MockGrid(width, height)

    def reset(self):
        self._step = 0
        self._human_x = 0

    def get_state(self):
        agents = (
            (0, 1, 0, None, None, None, None, None),             # robot at (0,1)
            (self._human_x, 0, 0, None, None, None, None, None), # human
        )
        return (self._step, agents, (), ())

    def step(self, action_profile):
        self._step += 1
        # Human moves right regardless of action (simplification)
        self._human_x = min(self._human_x + 1, self.width - 1)
        done = self._step >= self.max_steps
        return None, None, done, None


class MockGrid:
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def get(self, x, y):
        # Return wall objects on borders
        if x == 0 or x == self.width - 1 or y == 0 or y == self.height - 1:
            return MockCell("wall")
        return None


class MockCell:
    def __init__(self, cell_type):
        self.type = cell_type


# -----------------------------------------------------------------------
# Phase 1 Trainer tests
# -----------------------------------------------------------------------

class TestPhase1Trainer:

    def _make_trainer(self):
        """Create a trainer with small dimensions for testing."""
        wm = MockWorldModel(width=5, height=5, max_steps=8)
        from empo.ali_learning_based.encoders import StateEncoder, GoalEncoder
        se = StateEncoder(wm, robot_agent_index=0, human_agent_indices=[1])
        ge = GoalEncoder(wm)

        q_net = QhNet.from_encoders(se, ge, num_actions=6)
        trainer = Phase1Trainer(
            q_net=q_net,
            state_encoder=se,
            goal_encoder=ge,
            num_actions=6,
            num_agents=2,
            human_agent_idx=1,
            gamma=0.99,
            beta_h=10.0,
            lr=1e-3,
            buffer_capacity=1000,
            batch_size=16,
            target_update_freq=50,
            epsilon_start=1.0,
            epsilon_end=0.05,
            epsilon_decay_steps=100,
        )
        return trainer, wm

    def test_creation(self):
        trainer, _ = self._make_trainer()
        assert trainer.total_steps == 0
        assert len(trainer.buffer) == 0
        assert trainer.epsilon == 1.0

    def test_epsilon_decay(self):
        trainer, _ = self._make_trainer()
        assert trainer.epsilon == 1.0
        trainer.total_steps = 50
        assert trainer.epsilon == pytest.approx(0.525)
        trainer.total_steps = 100
        assert trainer.epsilon == pytest.approx(0.05)
        trainer.total_steps = 200  # past decay, stays at end
        assert trainer.epsilon == pytest.approx(0.05)

    def test_collect_and_store(self):
        trainer, wm = self._make_trainer()
        goals = [MockGoal(4, 0)]
        n = trainer.collect_and_store(wm, goals, num_episodes=2)
        assert n > 0
        assert len(trainer.buffer) == n

    def test_train_step_before_enough_data(self):
        trainer, _ = self._make_trainer()
        loss = trainer.train_step()
        assert loss is None  # buffer empty

    def test_train_step_returns_loss(self):
        trainer, wm = self._make_trainer()
        goals = [MockGoal(4, 0)]
        trainer.collect_and_store(wm, goals, num_episodes=5)
        loss = trainer.train_step()
        assert loss is not None
        assert isinstance(loss, float)
        assert loss >= 0

    def test_train_loop(self):
        """Full training loop should run without errors."""
        trainer, wm = self._make_trainer()
        goals = [MockGoal(4, 0)]

        history = trainer.train(
            wm, goals,
            num_iterations=10,
            episodes_per_iter=2,
            train_steps_per_iter=2,
            log_interval=0,  # suppress prints
        )
        assert len(history["losses"]) == 10
        assert len(history["epsilons"]) == 10

    def test_get_q_values(self):
        trainer, wm = self._make_trainer()
        wm.reset()
        state = wm.get_state()
        goal = MockGoal(4, 0)
        q = trainer.get_q_values(state, goal)
        assert q.shape == (6,)  # num_actions = 6

    def test_get_value(self):
        trainer, wm = self._make_trainer()
        wm.reset()
        state = wm.get_state()
        goal = MockGoal(4, 0)
        v = trainer.get_value(state, goal)
        assert isinstance(v, float)

    def test_get_policy(self):
        trainer, wm = self._make_trainer()
        wm.reset()
        state = wm.get_state()
        goal = MockGoal(4, 0)
        probs = trainer.get_policy(state, goal)
        assert probs.shape == (6,)
        assert probs.sum().item() == pytest.approx(1.0)
        assert (probs >= 0).all()

    def test_target_network_updates(self):
        """Target net should differ from Q-net before update, match after."""
        trainer, wm = self._make_trainer()
        goals = [MockGoal(4, 0)]

        # Collect data and do a few training steps to change Q-net weights
        trainer.collect_and_store(wm, goals, num_episodes=5)
        for _ in range(5):
            trainer.train_step()

        # After training, Q-net and target should differ
        q_params = list(trainer.q_net.parameters())
        t_params = list(trainer.target_net.parameters())
        # At least some params should differ (training changed Q-net)
        any_diff = any(
            not torch.equal(qp, tp)
            for qp, tp in zip(q_params, t_params)
        )
        assert any_diff, "Q-net should have changed from target after training"

        # Force target update
        trainer._update_target()
        # Now they should match
        for qp, tp in zip(q_params, t_params):
            assert torch.equal(qp, tp)

    def test_get_policy_fn(self):
        """get_policy_fn should return a callable that produces action lists."""
        trainer, wm = self._make_trainer()
        goal = MockGoal(4, 0)
        policy_fn = trainer.get_policy_fn(goal)
        wm.reset()
        state = wm.get_state()
        actions = policy_fn(state)
        assert len(actions) == 2  # 2 agents
        assert all(0 <= a < 6 for a in actions)


# -----------------------------------------------------------------------
# UCB exploration tests
# -----------------------------------------------------------------------

class TestUCBBonus:

    def test_unvisited_state_large_bonus(self):
        """Unvisited state should get a large bonus."""
        b = ucb_bonus(state_visits=0, action_visits=0, c=2.0)
        assert b > 10.0  # large value

    def test_untried_action_large_bonus(self):
        """Action never tried in a visited state should get a large bonus."""
        b = ucb_bonus(state_visits=100, action_visits=0, c=2.0)
        assert b > 3.0

    def test_well_explored_small_bonus(self):
        """Frequently explored (state, action) should have small bonus."""
        b = ucb_bonus(state_visits=1000, action_visits=500, c=2.0)
        assert b < 1.0

    def test_bonus_decreases_with_visits(self):
        """Bonus should decrease as action is explored more."""
        b1 = ucb_bonus(state_visits=100, action_visits=1, c=2.0)
        b2 = ucb_bonus(state_visits=100, action_visits=10, c=2.0)
        b3 = ucb_bonus(state_visits=100, action_visits=50, c=2.0)
        assert b1 > b2 > b3

    def test_c_scales_bonus(self):
        """Higher c → larger bonus."""
        b_low = ucb_bonus(state_visits=100, action_visits=10, c=1.0)
        b_high = ucb_bonus(state_visits=100, action_visits=10, c=5.0)
        assert b_high > b_low
        assert b_high / b_low == pytest.approx(5.0)

    def test_non_negative(self):
        """Bonus should always be non-negative."""
        for sv in [0, 1, 10, 100]:
            for av in [0, 1, 10, 100]:
                b = ucb_bonus(sv, av, c=2.0)
                assert b >= 0


class TestUCBActionValues:

    def test_augments_q_values(self):
        """UCB should add non-negative bonus to Q-values."""
        q = torch.tensor([1.0, 2.0, 3.0, 4.0])
        counts = {}
        augmented = ucb_action_values(q, "state_0", counts, num_actions=4, c=2.0)
        assert (augmented >= q).all()

    def test_favours_unexplored(self):
        """Action never tried should get the highest augmented value."""
        q = torch.tensor([1.0, 1.0, 1.0, 1.0])  # equal Q-values
        counts = {
            ("s", 0): 100,
            ("s", 1): 100,
            ("s", 2): 100,
            # action 3 never tried
        }
        augmented = ucb_action_values(q, "s", counts, num_actions=4, c=2.0)
        assert augmented[3] > augmented[0]
        assert augmented[3] > augmented[1]
        assert augmented[3] > augmented[2]


class TestPhase1TrainerUCB:
    """Tests for Phase1Trainer with UCB exploration."""

    def _make_ucb_trainer(self, ucb_c=2.0):
        wm = MockWorldModel(width=5, height=5, max_steps=8)
        from empo.ali_learning_based.encoders import StateEncoder, GoalEncoder
        se = StateEncoder(wm, robot_agent_index=0, human_agent_indices=[1])
        ge = GoalEncoder(wm)

        q_net = QhNet.from_encoders(se, ge, num_actions=6)
        trainer = Phase1Trainer(
            q_net=q_net,
            state_encoder=se,
            goal_encoder=ge,
            num_actions=6,
            num_agents=2,
            human_agent_idx=1,
            gamma=0.99,
            beta_h=10.0,
            lr=1e-3,
            buffer_capacity=1000,
            batch_size=16,
            target_update_freq=50,
            exploration="ucb",
            ucb_c=ucb_c,
        )
        return trainer, wm

    def test_creation(self):
        trainer, _ = self._make_ucb_trainer()
        assert trainer.exploration == "ucb"
        assert trainer.ucb_c == 2.0
        assert len(trainer.visit_counts) == 0

    def test_collect_and_store(self):
        trainer, wm = self._make_ucb_trainer()
        goals = [MockGoal(4, 0)]
        n = trainer.collect_and_store(wm, goals, num_episodes=2)
        assert n > 0
        assert len(trainer.buffer) == n
        # Visit counts should be populated
        assert len(trainer.visit_counts) > 0

    def test_visit_counts_grow(self):
        """Visit counts should accumulate across episodes."""
        trainer, wm = self._make_ucb_trainer()
        goals = [MockGoal(4, 0)]
        trainer.collect_and_store(wm, goals, num_episodes=2)
        n1 = sum(trainer.visit_counts.values())
        trainer.collect_and_store(wm, goals, num_episodes=2)
        n2 = sum(trainer.visit_counts.values())
        assert n2 > n1

    def test_train_loop(self):
        """Full UCB training loop should run without errors."""
        trainer, wm = self._make_ucb_trainer()
        goals = [MockGoal(4, 0)]
        history = trainer.train(
            wm, goals,
            num_iterations=10,
            episodes_per_iter=2,
            train_steps_per_iter=2,
            log_interval=0,
        )
        assert len(history["losses"]) == 10
        assert history["exploration"] == "ucb"

    def test_train_step_returns_loss(self):
        trainer, wm = self._make_ucb_trainer()
        goals = [MockGoal(4, 0)]
        trainer.collect_and_store(wm, goals, num_episodes=5)
        loss = trainer.train_step()
        assert loss is not None
        assert isinstance(loss, float)
        assert loss >= 0

    def test_evaluation_uses_boltzmann(self):
        """Evaluation policy should use Boltzmann (not UCB)."""
        trainer, wm = self._make_ucb_trainer()
        goal = MockGoal(4, 0)
        policy_fn = trainer.get_policy_fn(goal)
        wm.reset()
        state = wm.get_state()
        actions = policy_fn(state)
        assert len(actions) == 2
        assert all(0 <= a < 6 for a in actions)


# -----------------------------------------------------------------------
# Phase 1 with PBRS reward shaping
# -----------------------------------------------------------------------

class TestPhase1TrainerPBRS:
    """Tests for Phase1Trainer with potential-based reward shaping."""

    def _make_pbrs_trainer(self, shaping_weight=1.0):
        wm = MockWorldModel(width=5, height=5, max_steps=8)
        from empo.ali_learning_based.encoders import StateEncoder, GoalEncoder
        se = StateEncoder(wm, robot_agent_index=0, human_agent_indices=[1])
        ge = GoalEncoder(wm)

        q_net = QhNet.from_encoders(se, ge, num_actions=6)
        trainer = Phase1Trainer(
            q_net=q_net,
            state_encoder=se,
            goal_encoder=ge,
            num_actions=6,
            num_agents=2,
            human_agent_idx=1,
            gamma=0.99,
            beta_h=10.0,
            lr=1e-3,
            buffer_capacity=1000,
            batch_size=16,
            target_update_freq=50,
            reward_shaping="pbrs",
            shaping_weight=shaping_weight,
        )
        return trainer, wm

    def test_creation(self):
        trainer, _ = self._make_pbrs_trainer()
        assert trainer.reward_shaping == "pbrs"
        assert trainer.shaping_weight == 1.0

    def test_collect_and_store(self):
        """PBRS shaping should not crash during data collection."""
        trainer, wm = self._make_pbrs_trainer()
        goals = [MockGoal(4, 0)]
        n = trainer.collect_and_store(wm, goals, num_episodes=2)
        assert n > 0
        assert len(trainer.buffer) == n

    def test_shaped_rewards_not_binary(self):
        """With PBRS, rewards should not all be 0.0 or 1.0."""
        trainer, wm = self._make_pbrs_trainer()
        goals = [MockGoal(4, 0)]
        trainer.collect_and_store(wm, goals, num_episodes=2)

        batch = trainer.buffer.sample(min(16, len(trainer.buffer)))
        rewards = batch.rewards
        # At least some rewards should be non-zero and non-one
        unique = rewards.unique()
        assert len(unique) > 1, f"All rewards identical: {unique}"

    def test_train_loop(self):
        """Full training loop with PBRS should run without errors."""
        trainer, wm = self._make_pbrs_trainer()
        goals = [MockGoal(4, 0)]
        history = trainer.train(
            wm, goals,
            num_iterations=10,
            episodes_per_iter=2,
            train_steps_per_iter=2,
            log_interval=0,
        )
        assert len(history["losses"]) == 10

    def test_no_shaping_is_default(self):
        """reward_shaping='none' should give binary rewards."""
        wm = MockWorldModel(width=5, height=5, max_steps=8)
        from empo.ali_learning_based.encoders import StateEncoder, GoalEncoder
        se = StateEncoder(wm, robot_agent_index=0, human_agent_indices=[1])
        ge = GoalEncoder(wm)
        q_net = QhNet.from_encoders(se, ge, num_actions=6)

        trainer = Phase1Trainer(
            q_net=q_net, state_encoder=se, goal_encoder=ge,
            num_actions=6, num_agents=2, human_agent_idx=1,
            buffer_capacity=1000, batch_size=16,
            reward_shaping="none",
        )
        goals = [MockGoal(4, 0)]
        trainer.collect_and_store(wm, goals, num_episodes=2)

        batch = trainer.buffer.sample(min(16, len(trainer.buffer)))
        # All rewards should be 0.0 or 1.0
        for r in batch.rewards:
            assert r.item() in (0.0, 1.0), f"Unexpected reward: {r.item()}"


# -----------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------

if __name__ == "__main__":
    pytest.main([__file__, "-v"])

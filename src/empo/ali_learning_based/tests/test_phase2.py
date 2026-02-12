"""
Tests for Phase 2 training.

Run with:
    PYTHONPATH=src:vendor/multigrid pytest src/empo/ali_learning_based/tests/test_phase2.py -v
"""

import pytest
import torch

from gym_multigrid.multigrid import MultiGridEnv, SmallActions
from empo.ali_learning_based.envs import get_env_path

def _phase1_test_env():
    return get_env_path("phase1_test.yaml")

from empo.ali_learning_based.encoders import StateEncoder, GoalEncoder
from empo.ali_learning_based.networks import QhNet, VheNet, QrNet
from empo.ali_learning_based.phase1 import Phase1Trainer
from empo.ali_learning_based.phase2 import (
    Phase2Trainer,
    compute_empowerment,
    make_phase2_policy_fn,
)


# -----------------------------------------------------------------------
# Helpers
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


def make_phase1(env, robot_idx, human_idx):
    """Create and minimally train a Phase 1 trainer."""
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
        buffer_capacity=5000,
        batch_size=32,
        target_update_freq=50,
        epsilon_start=1.0,
        epsilon_end=0.1,
        epsilon_decay_steps=500,
    )
    goals = get_goals(env, human_idx)
    trainer.train(env, goals, num_iterations=20, episodes_per_iter=2,
                  train_steps_per_iter=2, log_interval=0)
    return trainer, se, ge


def make_phase2(env, robot_idx, human_idx, phase1_trainer, se, ge, goals):
    """Create a Phase 2 trainer."""
    vhe_net = VheNet.from_encoders(se, ge)
    qr_net = QrNet.from_encoders(se, num_actions=NUM_ACTIONS)
    return Phase2Trainer(
        vhe_net=vhe_net,
        q_r_net=qr_net,
        phase1_trainer=phase1_trainer,
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
        buffer_capacity=5000,
        batch_size=32,
        target_update_freq=50,
        epsilon_start=1.0,
        epsilon_end=0.1,
        epsilon_decay_steps=500,
    )


# -----------------------------------------------------------------------
# compute_empowerment tests
# -----------------------------------------------------------------------

class TestComputeEmpowerment:

    def test_output_shape(self):
        env = make_env()
        robot_idx, human_idx = get_indices(env)
        se = StateEncoder(env, robot_agent_index=robot_idx, human_agent_indices=[human_idx])
        ge = GoalEncoder(env)
        vhe = VheNet.from_encoders(se, ge)
        goals = get_goals(env, human_idx)
        goal_encs = torch.stack([ge.encode(g) for g in goals])

        env.reset()
        state_enc = se.encode(env.get_state()).unsqueeze(0)  # (1, D)

        emp = compute_empowerment(vhe, state_enc, goal_encs)
        assert emp.shape == (1,)

    def test_batch(self):
        env = make_env()
        robot_idx, human_idx = get_indices(env)
        se = StateEncoder(env, robot_agent_index=robot_idx, human_agent_indices=[human_idx])
        ge = GoalEncoder(env)
        vhe = VheNet.from_encoders(se, ge)
        goals = get_goals(env, human_idx)
        goal_encs = torch.stack([ge.encode(g) for g in goals])

        # Batch of 5 states
        states = torch.randn(5, se.dim)
        emp = compute_empowerment(vhe, states, goal_encs)
        assert emp.shape == (5,)

    def test_non_negative(self):
        """VheNet uses sigmoid, so V_h^e >= 0, thus empowerment >= 0."""
        env = make_env()
        robot_idx, human_idx = get_indices(env)
        se = StateEncoder(env, robot_agent_index=robot_idx, human_agent_indices=[human_idx])
        ge = GoalEncoder(env)
        vhe = VheNet.from_encoders(se, ge)
        goals = get_goals(env, human_idx)
        goal_encs = torch.stack([ge.encode(g) for g in goals])

        states = torch.randn(10, se.dim)
        emp = compute_empowerment(vhe, states, goal_encs)
        assert (emp >= 0).all()


# -----------------------------------------------------------------------
# Phase 2 Trainer tests
# -----------------------------------------------------------------------

class TestPhase2Trainer:

    @pytest.fixture(autouse=True)
    def setup(self):
        self.env = make_env()
        self.robot_idx, self.human_idx = get_indices(self.env)
        self.goals = get_goals(self.env, self.human_idx)
        self.p1, self.se, self.ge = make_phase1(
            self.env, self.robot_idx, self.human_idx
        )

    def _make_trainer(self):
        return make_phase2(
            self.env, self.robot_idx, self.human_idx,
            self.p1, self.se, self.ge, self.goals,
        )

    def test_creation(self):
        t = self._make_trainer()
        assert t.total_steps == 0
        assert len(t.buffer) == 0

    def test_collect_and_store(self):
        t = self._make_trainer()
        n = t.collect_and_store(self.env, num_episodes=2)
        assert n > 0
        assert len(t.buffer) == n

    def test_train_step_before_data(self):
        t = self._make_trainer()
        assert t.train_step() is None

    def test_train_step_returns_losses(self):
        t = self._make_trainer()
        t.collect_and_store(self.env, num_episodes=5)
        result = t.train_step()
        assert result is not None
        vhe_loss, qr_loss = result
        assert isinstance(vhe_loss, float)
        assert isinstance(qr_loss, float)
        assert vhe_loss >= 0
        assert qr_loss >= 0

    def test_train_loop(self):
        t = self._make_trainer()
        history = t.train(
            self.env,
            num_iterations=10,
            episodes_per_iter=2,
            train_steps_per_iter=2,
            log_interval=0,
        )
        assert len(history["vhe_losses"]) == 10
        assert len(history["qr_losses"]) == 10

    def test_get_vhe(self):
        t = self._make_trainer()
        t.collect_and_store(self.env, num_episodes=3)
        self.env.reset()
        state = self.env.get_state()
        v = t.get_vhe(state, self.goals[0])
        assert isinstance(v, float)
        assert 0.0 <= v <= 1.0  # sigmoid output

    def test_get_empowerment(self):
        t = self._make_trainer()
        self.env.reset()
        state = self.env.get_state()
        emp = t.get_empowerment(state)
        assert isinstance(emp, float)
        assert emp >= 0

    def test_get_robot_q_values(self):
        t = self._make_trainer()
        self.env.reset()
        state = self.env.get_state()
        q = t.get_robot_q_values(state)
        assert q.shape == (NUM_ACTIONS,)

    def test_get_robot_policy_fn(self):
        t = self._make_trainer()
        t.collect_and_store(self.env, num_episodes=3)
        policy = t.get_robot_policy_fn()
        self.env.reset()
        state = self.env.get_state()
        actions = policy(state)
        assert len(actions) == len(self.env.agents)
        assert all(0 <= a < NUM_ACTIONS for a in actions)

    def test_target_networks_exist(self):
        t = self._make_trainer()
        # Target nets should be copies
        for p, tp in zip(t.vhe_net.parameters(), t.vhe_target.parameters()):
            assert torch.equal(p, tp)
        for p, tp in zip(t.q_r_net.parameters(), t.qr_target.parameters()):
            assert torch.equal(p, tp)

    def test_target_diverges_after_training(self):
        t = self._make_trainer()
        t.collect_and_store(self.env, num_episodes=20)
        for _ in range(5):
            t.train_step()
        # After training, Q_r net should differ from target
        any_diff = any(
            not torch.equal(p, tp)
            for p, tp in zip(t.q_r_net.parameters(), t.qr_target.parameters())
        )
        assert any_diff


# -----------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------

if __name__ == "__main__":
    pytest.main([__file__, "-v"])

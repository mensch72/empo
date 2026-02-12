"""
Tests for the rollout collection module.

Run with:
    PYTHONPATH=src:vendor/multigrid pytest src/empo/ali_learning_based/tests/test_rollout.py -v
"""

import pytest
import torch

from empo.ali_learning_based.rollout import (
    collect_episode,
    collect_episodes,
    EpisodeData,
    manhattan_distance_to_rect,
    make_pbrs_shaper,
)


# -----------------------------------------------------------------------
# Mock objects
# -----------------------------------------------------------------------

class MockStateEncoder:
    """Encodes state tuples as [step_count, agent_x, agent_y]."""
    dim = 3
    height = 5
    width = 5

    def encode(self, state):
        step, agents, _, _ = state
        x, y = agents[0][:2]
        return torch.tensor([float(step), float(x), float(y)], dtype=torch.float32)


class MockGoalEncoder:
    dim = 2

    def encode(self, goal):
        return torch.tensor([float(goal.x), float(goal.y)], dtype=torch.float32)


class MockGoal:
    """Goal at a fixed position. Achieved when agent 0 is at that position."""
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.target_rect = (x, y, x, y)

    def is_achieved(self, state):
        _, agents, _, _ = state
        return agents[0][0] == self.x and agents[0][1] == self.y


class MockWorldModel:
    """
    A simple 1-D world: agent starts at x=0 and moves right each step.
    Episode ends after `max_steps` steps or when agent reaches `width - 1`.
    Two agents: agent 0 (robot) and agent 1 (human).
    """
    def __init__(self, width=5, max_steps=10):
        self.width = width
        self.max_steps = max_steps
        self._step = 0
        self._x = 0

    def reset(self):
        self._step = 0
        self._x = 0

    def get_state(self):
        agents = (
            (self._x, 0, 0, None, None, None, None, None),  # agent 0
            (0, 0, 0, None, None, None, None, None),          # agent 1
        )
        return (self._step, agents, (), ())

    def step(self, action_profile):
        self._step += 1
        # Agent 0 moves right regardless of action
        self._x = min(self._x + 1, self.width - 1)
        done = self._step >= self.max_steps or self._x >= self.width - 1
        return None, None, done, None


# -----------------------------------------------------------------------
# collect_episode tests
# -----------------------------------------------------------------------

class TestCollectEpisode:

    def test_returns_episode_data(self):
        wm = MockWorldModel(width=5, max_steps=10)
        se = MockStateEncoder()
        ge = MockGoalEncoder()
        goal = MockGoal(4, 0)  # at x=4
        policy_fn = lambda state: [0, 0]

        ep = collect_episode(wm, se, ge, goal, policy_fn, record_agent_idx=0)
        assert isinstance(ep, EpisodeData)

    def test_tensor_shapes(self):
        wm = MockWorldModel(width=5, max_steps=10)
        se = MockStateEncoder()
        ge = MockGoalEncoder()
        goal = MockGoal(4, 0)
        policy_fn = lambda state: [0, 0]

        ep = collect_episode(wm, se, ge, goal, policy_fn, record_agent_idx=0)
        T = ep.states.shape[0]
        assert T > 0
        assert ep.states.shape == (T, 3)    # state_dim = 3
        assert ep.goals.shape == (T, 2)     # goal_dim = 2
        assert ep.actions.shape == (T,)
        assert ep.rewards.shape == (T,)
        assert ep.next_states.shape == (T, 3)
        assert ep.dones.shape == (T,)
        assert ep.goal_rewards.shape == (T,)

    def test_episode_length(self):
        """Agent reaches x=4 after 4 steps (0→1→2→3→4)."""
        wm = MockWorldModel(width=5, max_steps=10)
        se = MockStateEncoder()
        ge = MockGoalEncoder()
        goal = MockGoal(4, 0)
        policy_fn = lambda state: [0, 0]

        ep = collect_episode(wm, se, ge, goal, policy_fn, record_agent_idx=0)
        assert ep.states.shape[0] == 4  # steps: x goes 0→1→2→3→4, done at x=4

    def test_rewards_match_goal(self):
        """Goal at x=4. Reward should be 1.0 only at the last step."""
        wm = MockWorldModel(width=5, max_steps=10)
        se = MockStateEncoder()
        ge = MockGoalEncoder()
        goal = MockGoal(4, 0)
        policy_fn = lambda state: [0, 0]

        ep = collect_episode(wm, se, ge, goal, policy_fn, record_agent_idx=0)
        # Only the last step should have reward 1.0 (when agent reaches x=4)
        assert ep.rewards[-1].item() == 1.0
        # Earlier steps should have reward 0.0
        if len(ep.rewards) > 1:
            assert (ep.rewards[:-1] == 0.0).all()

    def test_goal_rewards_pure_binary(self):
        """goal_rewards should be pure 0/1, matching rewards when no shaper."""
        wm = MockWorldModel(width=5, max_steps=10)
        se = MockStateEncoder()
        ge = MockGoalEncoder()
        goal = MockGoal(4, 0)
        policy_fn = lambda state: [0, 0]

        ep = collect_episode(wm, se, ge, goal, policy_fn, record_agent_idx=0)
        # Without a reward shaper, goal_rewards == rewards (both are pure 0/1)
        assert torch.equal(ep.goal_rewards, ep.rewards)
        # goal_rewards should only contain 0.0 or 1.0
        assert ((ep.goal_rewards == 0.0) | (ep.goal_rewards == 1.0)).all()

    def test_goal_rewards_separate_from_shaped(self):
        """goal_rewards should stay 0/1 even when rewards are shaped."""
        wm = MockWorldModel(width=5, max_steps=10)
        se = MockStateEncoder()
        ge = MockGoalEncoder()
        goal = MockGoal(4, 0)
        policy_fn = lambda state: [0, 0]

        def add_bonus(state, next_state, original_reward):
            return original_reward + 5.0  # big bonus every step

        ep = collect_episode(wm, se, ge, goal, policy_fn,
                             record_agent_idx=0, reward_shaper=add_bonus)
        # Shaped rewards should all be >= 5.0
        assert (ep.rewards >= 5.0).all()
        # goal_rewards should still be pure 0/1
        assert ((ep.goal_rewards == 0.0) | (ep.goal_rewards == 1.0)).all()
        # Last step goal_rewards should be 1.0 (goal achieved)
        assert ep.goal_rewards[-1].item() == 1.0

    def test_dones_flag(self):
        """Last transition should have done=True."""
        wm = MockWorldModel(width=5, max_steps=10)
        se = MockStateEncoder()
        ge = MockGoalEncoder()
        goal = MockGoal(4, 0)
        policy_fn = lambda state: [0, 0]

        ep = collect_episode(wm, se, ge, goal, policy_fn, record_agent_idx=0)
        assert ep.dones[-1].item() == True
        if len(ep.dones) > 1:
            assert (ep.dones[:-1] == False).all()

    def test_record_agent_idx(self):
        """Actions should come from the specified agent index."""
        wm = MockWorldModel(width=5, max_steps=10)
        se = MockStateEncoder()
        ge = MockGoalEncoder()
        goal = MockGoal(4, 0)
        # Agent 0 action=3, agent 1 action=5
        policy_fn = lambda state: [3, 5]

        ep0 = collect_episode(wm, se, ge, goal, policy_fn, record_agent_idx=0)
        assert (ep0.actions == 3).all()

        ep1 = collect_episode(wm, se, ge, goal, policy_fn, record_agent_idx=1)
        assert (ep1.actions == 5).all()

    def test_goal_encoding_repeated(self):
        """Goal encoding should be the same for every step in the episode."""
        wm = MockWorldModel(width=5, max_steps=10)
        se = MockStateEncoder()
        ge = MockGoalEncoder()
        goal = MockGoal(4, 0)
        policy_fn = lambda state: [0, 0]

        ep = collect_episode(wm, se, ge, goal, policy_fn, record_agent_idx=0)
        # All goal rows should be identical
        for i in range(ep.goals.shape[0]):
            assert torch.equal(ep.goals[i], ep.goals[0])

    def test_state_next_state_chain(self):
        """next_state[t] should equal state[t+1] (encoding reuse)."""
        wm = MockWorldModel(width=5, max_steps=10)
        se = MockStateEncoder()
        ge = MockGoalEncoder()
        goal = MockGoal(4, 0)
        policy_fn = lambda state: [0, 0]

        ep = collect_episode(wm, se, ge, goal, policy_fn, record_agent_idx=0)
        T = ep.states.shape[0]
        for t in range(T - 1):
            assert torch.equal(ep.next_states[t], ep.states[t + 1])

    def test_dtypes(self):
        wm = MockWorldModel(width=5, max_steps=10)
        se = MockStateEncoder()
        ge = MockGoalEncoder()
        goal = MockGoal(4, 0)
        policy_fn = lambda state: [0, 0]

        ep = collect_episode(wm, se, ge, goal, policy_fn, record_agent_idx=0)
        assert ep.states.dtype == torch.float32
        assert ep.goals.dtype == torch.float32
        assert ep.actions.dtype == torch.int64
        assert ep.rewards.dtype == torch.float32
        assert ep.dones.dtype == torch.bool
        assert ep.goal_rewards.dtype == torch.float32


# -----------------------------------------------------------------------
# collect_episodes tests
# -----------------------------------------------------------------------

class TestCollectEpisodes:

    def test_concatenates_multiple_episodes(self):
        wm = MockWorldModel(width=5, max_steps=10)
        se = MockStateEncoder()
        ge = MockGoalEncoder()
        goals = [MockGoal(4, 0)]
        policy_fn = lambda state: [0, 0]

        ep = collect_episodes(wm, se, ge, goals, policy_fn, record_agent_idx=0, num_episodes=3)
        # Each episode is 4 steps, so 3 episodes = 12 steps
        assert ep.states.shape[0] == 12

    def test_goal_cycling(self):
        """Goals should cycle round-robin."""
        wm = MockWorldModel(width=5, max_steps=10)
        se = MockStateEncoder()
        ge = MockGoalEncoder()
        goal_a = MockGoal(4, 0)
        goal_b = MockGoal(3, 0)
        goals = [goal_a, goal_b]
        policy_fn = lambda state: [0, 0]

        # 3 episodes: goal_a, goal_b, goal_a
        ep = collect_episodes(wm, se, ge, goals, policy_fn, record_agent_idx=0, num_episodes=3)
        assert ep.states.shape[0] > 0

    def test_shapes_consistent(self):
        wm = MockWorldModel(width=5, max_steps=10)
        se = MockStateEncoder()
        ge = MockGoalEncoder()
        goals = [MockGoal(4, 0)]
        policy_fn = lambda state: [0, 0]

        ep = collect_episodes(wm, se, ge, goals, policy_fn, record_agent_idx=0, num_episodes=2)
        T = ep.states.shape[0]
        assert ep.goals.shape == (T, 2)
        assert ep.actions.shape == (T,)
        assert ep.rewards.shape == (T,)
        assert ep.next_states.shape == (T, 3)
        assert ep.dones.shape == (T,)
        assert ep.goal_rewards.shape == (T,)


# -----------------------------------------------------------------------
# manhattan_distance_to_rect tests
# -----------------------------------------------------------------------

class TestManhattanDistanceToRect:

    def test_inside_rect(self):
        assert manhattan_distance_to_rect(2, 3, 1, 1, 4, 4) == 0

    def test_on_border(self):
        assert manhattan_distance_to_rect(1, 1, 1, 1, 4, 4) == 0
        assert manhattan_distance_to_rect(4, 4, 1, 1, 4, 4) == 0

    def test_left_of_rect(self):
        assert manhattan_distance_to_rect(0, 2, 2, 1, 4, 4) == 2

    def test_above_rect(self):
        assert manhattan_distance_to_rect(3, 0, 2, 2, 4, 4) == 2

    def test_diagonal(self):
        # (0,0) to rect [2,2,4,4]: dx=2, dy=2, manhattan=4
        assert manhattan_distance_to_rect(0, 0, 2, 2, 4, 4) == 4

    def test_right_of_rect(self):
        assert manhattan_distance_to_rect(6, 3, 1, 1, 4, 4) == 2

    def test_point_rect(self):
        # Single-cell rect at (3,3)
        assert manhattan_distance_to_rect(3, 3, 3, 3, 3, 3) == 0
        assert manhattan_distance_to_rect(1, 3, 3, 3, 3, 3) == 2

    def test_symmetric(self):
        # Distance from left and right should be the same
        assert (manhattan_distance_to_rect(0, 3, 2, 2, 4, 4) ==
                manhattan_distance_to_rect(6, 3, 2, 2, 4, 4))


# -----------------------------------------------------------------------
# PBRS shaper tests
# -----------------------------------------------------------------------

class MockRectGoal:
    """Goal with a target rectangle, used for testing PBRS shaper."""
    def __init__(self, x1, y1, x2, y2):
        self.target_rect = (x1, y1, x2, y2)

    def is_achieved(self, state):
        _, agents, _, _ = state
        x, y = int(agents[0][0]), int(agents[0][1])
        x1, y1, x2, y2 = self.target_rect
        return int(x1 <= x <= x2 and y1 <= y <= y2)


class TestPBRSShaper:

    def _make_state(self, x, y, step=0):
        """Helper to create a state tuple with agent 0 at (x, y)."""
        agents = (
            (x, y, 0, None, None, None, None, None, None),
        )
        return (step, agents, (), ())

    def test_positive_when_approaching(self):
        """Moving closer to goal should give positive shaping."""
        goal = MockRectGoal(4, 4, 4, 4)
        shaper = make_pbrs_shaper(goal, human_agent_idx=0, gamma=0.99,
                                  max_dist=10.0, weight=1.0)
        s0 = self._make_state(0, 0)
        s1 = self._make_state(1, 0)  # closer to (4,4)
        reward = shaper(s0, s1, 0.0)
        assert reward > 0.0

    def test_negative_when_retreating(self):
        """Moving away from goal should give negative shaping."""
        goal = MockRectGoal(4, 4, 4, 4)
        shaper = make_pbrs_shaper(goal, human_agent_idx=0, gamma=0.99,
                                  max_dist=10.0, weight=1.0)
        s0 = self._make_state(2, 2)
        s1 = self._make_state(1, 2)  # further from (4,4)
        reward = shaper(s0, s1, 0.0)
        assert reward < 0.0

    def test_zero_when_at_goal(self):
        """When already at goal, potential should be 0."""
        goal = MockRectGoal(3, 3, 5, 5)
        shaper = make_pbrs_shaper(goal, human_agent_idx=0, gamma=1.0,
                                  max_dist=10.0, weight=1.0)
        s0 = self._make_state(4, 4)  # inside rect
        s1 = self._make_state(4, 4)  # still inside
        reward = shaper(s0, s1, 0.0)
        assert abs(reward) < 1e-6

    def test_original_reward_passes_through(self):
        """Original reward should be included in shaped reward."""
        goal = MockRectGoal(4, 4, 4, 4)
        shaper = make_pbrs_shaper(goal, human_agent_idx=0, gamma=0.99,
                                  max_dist=10.0, weight=1.0)
        s0 = self._make_state(3, 3)
        s1 = self._make_state(3, 3)  # same position
        reward = shaper(s0, s1, 1.0)
        # Should include the original 1.0 plus a small shaping term
        assert reward >= 0.9

    def test_weight_scales_shaping(self):
        """Higher weight should give larger shaping magnitude."""
        goal = MockRectGoal(4, 4, 4, 4)
        s0 = self._make_state(0, 0)
        s1 = self._make_state(1, 0)

        shaper_small = make_pbrs_shaper(goal, 0, 0.99, 10.0, weight=0.1)
        r_small = shaper_small(s0, s1, 0.0)

        shaper_large = make_pbrs_shaper(goal, 0, 0.99, 10.0, weight=10.0)
        r_large = shaper_large(s0, s1, 0.0)

        assert abs(r_large) > abs(r_small)

    def test_stateful_across_steps(self):
        """Shaper should track previous potential across calls."""
        goal = MockRectGoal(4, 0, 4, 0)
        shaper = make_pbrs_shaper(goal, 0, 1.0, 10.0, weight=1.0)

        # Step 1: 0→1 (closer)
        s0 = self._make_state(0, 0)
        s1 = self._make_state(1, 0)
        r1 = shaper(s0, s1, 0.0)

        # Step 2: 1→2 (closer again)
        s2 = self._make_state(2, 0)
        r2 = shaper(s1, s2, 0.0)

        # Both should be positive (moving closer)
        assert r1 > 0.0
        assert r2 > 0.0


# -----------------------------------------------------------------------
# collect_episode with reward shaper tests
# -----------------------------------------------------------------------

class TestCollectEpisodeWithShaper:

    def test_shaper_modifies_rewards(self):
        """When a reward shaper is provided, rewards should differ from binary."""
        wm = MockWorldModel(width=5, max_steps=10)
        se = MockStateEncoder()
        ge = MockGoalEncoder()
        goal = MockGoal(4, 0)
        policy_fn = lambda state: [0, 0]

        # Without shaper: rewards are 0.0 until goal reached (1.0)
        ep_no_shaping = collect_episode(wm, se, ge, goal, policy_fn,
                                        record_agent_idx=0)

        # With a simple shaper that adds 0.5 to every reward
        def add_half(state, next_state, original_reward):
            return original_reward + 0.5

        ep_shaped = collect_episode(wm, se, ge, goal, policy_fn,
                                    record_agent_idx=0,
                                    reward_shaper=add_half)

        # Shaped rewards should be 0.5 higher than unshaped
        diff = ep_shaped.rewards - ep_no_shaping.rewards
        assert torch.allclose(diff, torch.full_like(diff, 0.5))

    def test_none_shaper_is_noop(self):
        """reward_shaper=None should produce identical results to no shaper."""
        wm = MockWorldModel(width=5, max_steps=10)
        se = MockStateEncoder()
        ge = MockGoalEncoder()
        goal = MockGoal(4, 0)
        policy_fn = lambda state: [0, 0]

        ep1 = collect_episode(wm, se, ge, goal, policy_fn,
                              record_agent_idx=0)
        ep2 = collect_episode(wm, se, ge, goal, policy_fn,
                              record_agent_idx=0, reward_shaper=None)

        assert torch.equal(ep1.rewards, ep2.rewards)


# -----------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------

if __name__ == "__main__":
    pytest.main([__file__, "-v"])

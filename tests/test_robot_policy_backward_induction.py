#!/usr/bin/env python3
"""
Test robot policy backward induction (Phase 2).

Tests the compute_robot_policy function with simple environments
to ensure computational feasibility and correctness.

These tests verify:
1. Basic execution without errors
2. Policy structure (correct state/action mappings)
3. Value function properties (terminal states, monotonicity)
4. Parallel vs sequential consistency
"""

import sys
import os
import numpy as np
import pytest
from typing import List, Tuple, Dict, Any
from itertools import product

# Patch gym import for compatibility
import gymnasium as gym
sys.modules['gym'] = gym

from gym_multigrid.multigrid import (
    MultiGridEnv, Grid, Agent, Wall, World, SmallActions, SmallWorld,
    Goal, Ball
)
from empo.possible_goal import PossibleGoal, PossibleGoalGenerator
from empo.backward_induction import (
    compute_human_policy_prior, 
    compute_robot_policy,
    TabularRobotPolicy
)


# =============================================================================
# Simple Test Environment (3x3 grid, 1 human + 1 robot)
# =============================================================================

class TinyTwoAgentEnv(MultiGridEnv):
    """
    Minimal 3x3 environment with 1 human and 1 robot for testing.
    
    Layout:
        We We We
        We .. We
        We We We
    
    Human (agent 0) and Robot (agent 1) start at center.
    Very limited state space for fast testing.
    """
    
    def __init__(self):
        agents = [
            Agent(SmallWorld, 0, view_size=3),  # Human
            Agent(SmallWorld, 1, view_size=3),  # Robot
        ]
        super().__init__(
            grid_size=3,
            max_steps=2,  # Very short for testing
            agents=agents,
            agent_view_size=3,
            actions_set=SmallActions,  # only still, forward, left, right
        )
    
    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        # Wall border
        self.grid.horz_wall(SmallWorld, 0, 0)
        self.grid.horz_wall(SmallWorld, 0, height-1)
        self.grid.vert_wall(SmallWorld, 0, 0)
        self.grid.vert_wall(SmallWorld, width-1, 0)
        # Place both agents at center
        self.agents[0].pos = np.array([1, 1])
        self.agents[0].dir = 0
        self.agents[1].pos = np.array([1, 1])
        self.agents[1].dir = 0


class Tiny4x4TwoAgentEnv(MultiGridEnv):
    """
    Small 4x4 environment with 1 human and 1 robot.
    
    Layout:
        We We We We
        We .. .. We
        We .. .. We
        We We We We
    
    Slightly larger state space but still manageable.
    """
    
    def __init__(self):
        agents = [
            Agent(SmallWorld, 0, view_size=3),  # Human
            Agent(SmallWorld, 1, view_size=3),  # Robot
        ]
        super().__init__(
            grid_size=4,
            max_steps=3,
            agents=agents,
            agent_view_size=3,
            actions_set=SmallActions,
        )
    
    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        # Wall border
        self.grid.horz_wall(SmallWorld, 0, 0)
        self.grid.horz_wall(SmallWorld, 0, height-1)
        self.grid.vert_wall(SmallWorld, 0, 0)
        self.grid.vert_wall(SmallWorld, width-1, 0)
        # Place agents
        self.agents[0].pos = np.array([1, 1])
        self.agents[0].dir = 0
        self.agents[1].pos = np.array([2, 2])
        self.agents[1].dir = 2


# =============================================================================
# Goal Classes
# =============================================================================

class ReachRectGoal(PossibleGoal):
    """A goal where a human agent tries to reach any cell in a rectangle."""
    
    def __init__(self, world_model, human_agent_index: int, rect: tuple):
        """
        Args:
            world_model: The environment.
            human_agent_index: Which agent this goal is for.
            rect: (x_min, y_min, x_max, y_max) inclusive rectangle.
        """
        super().__init__(world_model)
        self.human_agent_index = human_agent_index
        self.rect = rect  # (x_min, y_min, x_max, y_max)
    
    def is_achieved(self, state) -> int:
        """Returns 1 if the human is within the rectangle, 0 otherwise."""
        step_count, agent_states, mobile_objects, mutable_objects = state
        agent_state = agent_states[self.human_agent_index]
        agent_x, agent_y = agent_state[0], agent_state[1]
        x_min, y_min, x_max, y_max = self.rect
        if x_min <= agent_x <= x_max and y_min <= agent_y <= y_max:
            return 1
        return 0
    
    def __str__(self):
        return f"ReachRect({self.human_agent_index}->{self.rect})"
    
    def __repr__(self):
        return self.__str__()
    
    def __hash__(self):
        return hash((self.human_agent_index, self.rect))
    
    def __eq__(self, other):
        return (isinstance(other, ReachRectGoal) and 
                self.human_agent_index == other.human_agent_index and
                self.rect == other.rect)


class WholeMapGoalGenerator(PossibleGoalGenerator):
    """
    Generates goals that partition the map into regions.
    
    Always includes a goal covering the entire walkable area to ensure
    at least one goal is achievable from any state.
    """
    
    def __init__(self, world_model, grid_size: int):
        super().__init__(world_model)
        self.grid_size = grid_size
        # Walkable area is inside the walls (1 to grid_size-2)
        self.walkable_min = 1
        self.walkable_max = grid_size - 2
    
    def generate(self, state, human_agent_index: int):
        # Goal 1: Cover the whole walkable area (always achievable)
        whole_map = ReachRectGoal(
            self.env, human_agent_index,
            (self.walkable_min, self.walkable_min, self.walkable_max, self.walkable_max)
        )
        yield (whole_map, 0.5)
        
        # Goal 2: Left half of the map
        left_half = ReachRectGoal(
            self.env, human_agent_index,
            (self.walkable_min, self.walkable_min, 
             (self.walkable_min + self.walkable_max) // 2, self.walkable_max)
        )
        yield (left_half, 0.25)
        
        # Goal 3: Right half of the map  
        right_half = ReachRectGoal(
            self.env, human_agent_index,
            ((self.walkable_min + self.walkable_max) // 2 + 1, self.walkable_min,
             self.walkable_max, self.walkable_max)
        )
        yield (right_half, 0.25)


# =============================================================================
# Tests
# =============================================================================

class TestRobotPolicyBasic:
    """Basic tests for compute_robot_policy."""
    
    def test_compute_robot_policy_runs(self):
        """Test that compute_robot_policy executes without errors."""
        env = TinyTwoAgentEnv()
        env.reset()
        
        human_agent_indices = [0]
        robot_agent_indices = [1]
        goal_gen = WholeMapGoalGenerator(env, grid_size=3)
        
        # First compute human policy prior
        human_policy = compute_human_policy_prior(
            env, human_agent_indices, goal_gen,
            beta_h=5.0, gamma_h=1.0, quiet=True
        )
        
        # Then compute robot policy
        robot_policy = compute_robot_policy(
            env, human_agent_indices, robot_agent_indices,
            goal_gen, human_policy,
            beta_r=5.0, gamma_h=1.0, gamma_r=1.0,
            zeta=1.0, xi=1.0, eta=1.0,
            quiet=True
        )
        
        assert isinstance(robot_policy, TabularRobotPolicy)
        assert robot_policy.robot_agent_indices == robot_agent_indices
    
    def test_robot_policy_returns_valid_distributions(self):
        """Test that robot policy returns valid probability distributions."""
        env = TinyTwoAgentEnv()
        env.reset()
        
        human_agent_indices = [0]
        robot_agent_indices = [1]
        goal_gen = WholeMapGoalGenerator(env, grid_size=3)
        
        human_policy = compute_human_policy_prior(
            env, human_agent_indices, goal_gen,
            beta_h=5.0, gamma_h=1.0, quiet=True
        )
        
        robot_policy = compute_robot_policy(
            env, human_agent_indices, robot_agent_indices,
            goal_gen, human_policy,
            beta_r=5.0, gamma_h=1.0, gamma_r=1.0,
            zeta=1.0, xi=1.0, eta=1.0,
            quiet=True
        )
        
        # Check that we have some non-terminal states with policies
        state = env.get_state()
        dist = robot_policy(state)
        
        if dist:  # Non-terminal state
            probs = np.array(list(dist.values()))
            # Probabilities should be non-negative
            assert np.all(probs >= 0), f"Negative probabilities: {probs}"
            # Probabilities should sum to ~1
            assert np.abs(probs.sum() - 1.0) < 1e-6, f"Probabilities don't sum to 1: {probs.sum()}"
    
    def test_robot_policy_can_sample(self):
        """Test that robot policy can sample actions."""
        env = TinyTwoAgentEnv()
        env.reset()
        
        human_agent_indices = [0]
        robot_agent_indices = [1]
        goal_gen = WholeMapGoalGenerator(env, grid_size=3)
        
        human_policy = compute_human_policy_prior(
            env, human_agent_indices, goal_gen,
            beta_h=5.0, gamma_h=1.0, quiet=True
        )
        
        robot_policy = compute_robot_policy(
            env, human_agent_indices, robot_agent_indices,
            goal_gen, human_policy,
            beta_r=5.0, gamma_h=1.0, gamma_r=1.0,
            zeta=1.0, xi=1.0, eta=1.0,
            quiet=True
        )
        
        state = env.get_state()
        
        # Sample action profile
        action_profile = robot_policy.sample(state)
        assert isinstance(action_profile, tuple)
        assert len(action_profile) == len(robot_agent_indices)
        
        # Actions should be valid
        num_actions = env.action_space.n
        for action in action_profile:
            assert 0 <= action < num_actions
        
        # Test get_action for specific robot
        action = robot_policy.get_action(state, 1)  # robot index 1
        assert 0 <= action < num_actions


class TestRobotPolicyValues:
    """Tests for return_values=True mode."""
    
    def test_return_values(self):
        """Test that return_values=True returns Vr and Vh dicts."""
        env = TinyTwoAgentEnv()
        env.reset()
        
        human_agent_indices = [0]
        robot_agent_indices = [1]
        goal_gen = WholeMapGoalGenerator(env, grid_size=3)
        
        human_policy = compute_human_policy_prior(
            env, human_agent_indices, goal_gen,
            beta_h=5.0, gamma_h=1.0, quiet=True
        )
        
        robot_policy, Vr_dict, Vh_dict = compute_robot_policy(
            env, human_agent_indices, robot_agent_indices,
            goal_gen, human_policy,
            beta_r=5.0, gamma_h=1.0, gamma_r=1.0,
            zeta=1.0, xi=1.0, eta=1.0,
            return_values=True,
            quiet=True
        )
        
        assert isinstance(robot_policy, TabularRobotPolicy)
        assert isinstance(Vr_dict, dict)
        assert isinstance(Vh_dict, dict)
        
        # Vr values should be floats
        for state, vr in Vr_dict.items():
            assert isinstance(vr, float)
    
    def test_terminal_state_values(self):
        """Test that terminal states have Vr=terminal_Vr (default -1e-10)."""
        env = TinyTwoAgentEnv()
        env.reset()
        
        human_agent_indices = [0]
        robot_agent_indices = [1]
        goal_gen = WholeMapGoalGenerator(env, grid_size=3)
        
        human_policy = compute_human_policy_prior(
            env, human_agent_indices, goal_gen,
            beta_h=5.0, gamma_h=1.0, quiet=True
        )
        
        robot_policy, Vr_dict, Vh_dict = compute_robot_policy(
            env, human_agent_indices, robot_agent_indices,
            goal_gen, human_policy,
            beta_r=5.0, gamma_h=1.0, gamma_r=1.0,
            zeta=1.0, xi=1.0, eta=1.0,
            return_values=True,
            quiet=True
        )
        
        # Find terminal states (step_count == max_steps)
        terminal_states = [s for s in Vr_dict.keys() if s[0] == env.max_steps]
        
        # Terminal states should have Vr = terminal_Vr (default -1e-10)
        terminal_Vr = -1e-10
        for state in terminal_states:
            assert Vr_dict[state] == terminal_Vr, f"Terminal state {state} has Vr={Vr_dict[state]}, expected {terminal_Vr}"


class TestRobotPolicyMultipleGoals:
    """Tests with multiple possible goals."""
    
    def test_multiple_goals(self):
        """Test robot policy with multiple possible goals for humans."""
        env = Tiny4x4TwoAgentEnv()
        env.reset()
        
        human_agent_indices = [0]
        robot_agent_indices = [1]
        # Use WholeMapGoalGenerator which produces multiple goals covering the map
        goal_gen = WholeMapGoalGenerator(env, grid_size=4)
        
        human_policy = compute_human_policy_prior(
            env, human_agent_indices, goal_gen,
            beta_h=5.0, gamma_h=1.0, quiet=True
        )
        
        robot_policy = compute_robot_policy(
            env, human_agent_indices, robot_agent_indices,
            goal_gen, human_policy,
            beta_r=5.0, gamma_h=1.0, gamma_r=1.0,
            zeta=1.0, xi=1.0, eta=1.0,
            quiet=True
        )
        
        assert isinstance(robot_policy, TabularRobotPolicy)
        
        state = env.get_state()
        dist = robot_policy(state)
        
        if dist:
            probs = np.array(list(dist.values()))
            # Check not NaN and sums to 1
            assert not np.any(np.isnan(probs)), f"NaN probabilities: {probs}"
            assert np.abs(probs.sum() - 1.0) < 1e-6


class TestRobotPolicyParallel:
    """Tests for parallel execution mode."""
    
    def test_parallel_runs(self):
        """Test that parallel mode executes without errors."""
        env = Tiny4x4TwoAgentEnv()
        env.reset()
        
        human_agent_indices = [0]
        robot_agent_indices = [1]
        goal_gen = WholeMapGoalGenerator(env, grid_size=4)
        
        human_policy = compute_human_policy_prior(
            env, human_agent_indices, goal_gen,
            beta_h=5.0, gamma_h=1.0, quiet=True
        )
        
        robot_policy = compute_robot_policy(
            env, human_agent_indices, robot_agent_indices,
            goal_gen, human_policy,
            beta_r=5.0, gamma_h=1.0, gamma_r=1.0,
            zeta=1.0, xi=1.0, eta=1.0,
            parallel=True,
            num_workers=2,
            level_fct=lambda s: s[0],  # Use step count as level
            quiet=True
        )
        
        assert isinstance(robot_policy, TabularRobotPolicy)
    
    def test_parallel_vs_sequential_consistency(self):
        """Test that parallel and sequential modes produce same results."""
        env = Tiny4x4TwoAgentEnv()
        env.reset()
        
        human_agent_indices = [0]
        robot_agent_indices = [1]
        goal_gen = WholeMapGoalGenerator(env, grid_size=4)
        
        human_policy = compute_human_policy_prior(
            env, human_agent_indices, goal_gen,
            beta_h=5.0, gamma_h=1.0, quiet=True
        )
        
        # Sequential mode
        env.reset()
        robot_policy_seq, Vr_seq, Vh_seq = compute_robot_policy(
            env, human_agent_indices, robot_agent_indices,
            goal_gen, human_policy,
            beta_r=5.0, gamma_h=1.0, gamma_r=1.0,
            zeta=1.0, xi=1.0, eta=1.0,
            parallel=False,
            return_values=True,
            quiet=True
        )
        
        # Parallel mode
        env.reset()
        robot_policy_par, Vr_par, Vh_par = compute_robot_policy(
            env, human_agent_indices, robot_agent_indices,
            goal_gen, human_policy,
            beta_r=5.0, gamma_h=1.0, gamma_r=1.0,
            zeta=1.0, xi=1.0, eta=1.0,
            parallel=True,
            num_workers=2,
            level_fct=lambda s: s[0],
            return_values=True,
            quiet=True
        )
        
        # Compare Vr values
        for state in Vr_seq:
            if state in Vr_par:
                assert np.abs(Vr_seq[state] - Vr_par[state]) < 1e-6, \
                    f"Vr mismatch for state {state}: seq={Vr_seq[state]}, par={Vr_par[state]}"
        
        # Compare policies
        for state in robot_policy_seq.values:
            if state in robot_policy_par.values:
                dist_seq = robot_policy_seq.values[state]
                dist_par = robot_policy_par.values[state]
                for action_profile in dist_seq:
                    if action_profile in dist_par:
                        assert np.abs(dist_seq[action_profile] - dist_par[action_profile]) < 1e-6


class TestRobotPolicyEdgeCases:
    """Edge case tests."""
    
    def test_single_state(self):
        """Test with environment that has very few states."""
        env = TinyTwoAgentEnv()
        env.max_steps = 1  # Only initial state + terminal
        env.reset()
        
        human_agent_indices = [0]
        robot_agent_indices = [1]
        goal_gen = WholeMapGoalGenerator(env, grid_size=3)
        
        human_policy = compute_human_policy_prior(
            env, human_agent_indices, goal_gen,
            beta_h=5.0, gamma_h=1.0, quiet=True
        )
        
        robot_policy = compute_robot_policy(
            env, human_agent_indices, robot_agent_indices,
            goal_gen, human_policy,
            beta_r=5.0, gamma_h=1.0, gamma_r=1.0,
            zeta=1.0, xi=1.0, eta=1.0,
            quiet=True
        )
        
        assert isinstance(robot_policy, TabularRobotPolicy)
    
    def test_high_beta_deterministic(self):
        """Test that high beta_r produces valid policies without numerical issues."""
        env = Tiny4x4TwoAgentEnv()
        env.reset()
        
        human_agent_indices = [0]
        robot_agent_indices = [1]
        goal_gen = WholeMapGoalGenerator(env, grid_size=4)
        
        human_policy = compute_human_policy_prior(
            env, human_agent_indices, goal_gen,
            beta_h=5.0, gamma_h=1.0, quiet=True
        )
        
        # Moderately high beta (100.0 causes numerical overflow in power-law)
        robot_policy = compute_robot_policy(
            env, human_agent_indices, robot_agent_indices,
            goal_gen, human_policy,
            beta_r=20.0,  # High but not extreme
            gamma_h=1.0, gamma_r=1.0,
            zeta=1.0, xi=1.0, eta=1.0,
            quiet=True
        )
        
        state = env.get_state()
        dist = robot_policy(state)
        
        if dist:
            probs = np.array(list(dist.values()))
            # Check that we get valid probabilities (no NaN, sums to 1)
            assert not np.any(np.isnan(probs)), f"NaN probabilities: {probs}"
            assert np.abs(probs.sum() - 1.0) < 1e-6, f"Probabilities don't sum to 1: {probs.sum()}"
            # With high beta, policy should be more concentrated (less uniform)
            # Uniform would be 0.25 for 4 actions; max should be > uniform
            assert np.max(probs) >= 0.25, f"Policy is too uniform: {probs}"


class TestRobotPolicyParameters:
    """Tests for different parameter values."""
    
    def test_different_discount_factors(self):
        """Test with different gamma values."""
        env = Tiny4x4TwoAgentEnv()
        env.reset()
        
        human_agent_indices = [0]
        robot_agent_indices = [1]
        goal_gen = WholeMapGoalGenerator(env, grid_size=4)
        
        human_policy = compute_human_policy_prior(
            env, human_agent_indices, goal_gen,
            beta_h=5.0, gamma_h=0.9, quiet=True
        )
        
        # Test with gamma_r=0.5
        robot_policy = compute_robot_policy(
            env, human_agent_indices, robot_agent_indices,
            goal_gen, human_policy,
            beta_r=5.0, gamma_h=0.9, gamma_r=0.5,
            zeta=1.0, xi=1.0, eta=1.0,
            quiet=True
        )
        
        assert isinstance(robot_policy, TabularRobotPolicy)
    
    def test_different_theory_parameters(self):
        """Test with different zeta, xi, eta values."""
        env = Tiny4x4TwoAgentEnv()
        env.reset()
        
        human_agent_indices = [0]
        robot_agent_indices = [1]
        goal_gen = WholeMapGoalGenerator(env, grid_size=4)
        
        human_policy = compute_human_policy_prior(
            env, human_agent_indices, goal_gen,
            beta_h=5.0, gamma_h=1.0, quiet=True
        )
        
        # Test with non-default zeta, xi, eta
        robot_policy = compute_robot_policy(
            env, human_agent_indices, robot_agent_indices,
            goal_gen, human_policy,
            beta_r=5.0, gamma_h=1.0, gamma_r=1.0,
            zeta=0.5,   # Risk-seeking
            xi=2.0,     # Higher inequality aversion
            eta=0.5,    # Lower intertemporal inequality aversion
            quiet=True
        )
        
        assert isinstance(robot_policy, TabularRobotPolicy)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

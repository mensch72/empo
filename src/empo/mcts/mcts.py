from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Iterator
import math
import random
import time

import numpy as np

from empo.possible_goal import PossibleGoal, PossibleGoalGenerator


# Action indices: 0=Still, 1=Left, 2=Right, 3=Forward, 4=Pickup, 5=Drop, 6=Toggle, 7=Done
# Inverse actions: Left <-> Right (turning back and forth is redundant)
INVERSE_ACTIONS: Dict[int, int] = {
    1: 2,  # Left -> Right
    2: 1,  # Right -> Left
}


class _MCTSReachCellGoal(PossibleGoal):
    """Simple reach-cell goal for MCTS default goal generation."""

    def __init__(self, world_model: Any, human_agent_index: int, target_pos: Tuple[int, int]):
        super().__init__(world_model)
        self.human_agent_index = human_agent_index
        self.target_pos = (int(target_pos[0]), int(target_pos[1]))
        self._hash = hash((self.human_agent_index, self.target_pos))
        super()._freeze()

    def is_achieved(self, state) -> int:
        step_count, agent_states, mobile_objects, mutable_objects = state
        if self.human_agent_index < len(agent_states):
            agent_state = agent_states[self.human_agent_index]
            x, y = int(agent_state[0]), int(agent_state[1])
            if x == self.target_pos[0] and y == self.target_pos[1]:
                return 1
        return 0

    def __hash__(self):
        return self._hash

    def __eq__(self, other):
        if not isinstance(other, _MCTSReachCellGoal):
            return False
        return (self.human_agent_index == other.human_agent_index and
                self.target_pos == other.target_pos)


class _MCTSDefaultGoalGenerator(PossibleGoalGenerator):
    """
    Default goal generator for MCTS that generates reach-cell goals
    for all walkable (non-wall) cells in the grid.
    """

    def __init__(self, world_model: Any):
        super().__init__(world_model)
        self._walkable_cells: Optional[List[Tuple[int, int]]] = None

    def _get_walkable_cells(self) -> List[Tuple[int, int]]:
        if self._walkable_cells is not None:
            return self._walkable_cells

        cells = []
        env = self.world_model
        if not hasattr(env, 'grid') or not hasattr(env, 'width') or not hasattr(env, 'height'):
            self._walkable_cells = cells
            return cells

        # Exclude immutable obstacles AND rocks (rocks can move but aren't walkable)
        immutable_types = {'wall', 'lava', 'magicwall', 'rock'}
        for x in range(env.width):
            for y in range(env.height):
                cell = env.grid.get(x, y)
                cell_type = getattr(cell, 'type', None) if cell else None
                if cell_type not in immutable_types:
                    cells.append((x, y))

        self._walkable_cells = cells
        return cells

    def generate(self, state, human_agent_index: int) -> Iterator[Tuple[PossibleGoal, float]]:
        """
        Generate goals for ALL walkable cells.

        The goal_attainment will be 1.0 if reachable, 0.0 if blocked.
        This way, pushing rocks that open new areas increases human power.
        """
        cells = self._get_walkable_cells()
        if not cells:
            return

        # Generate goal for every walkable cell
        # Reachability is checked in _compute_goal_attainment()
        weight = 1.0 / len(cells)
        for cell in cells:
            goal = _MCTSReachCellGoal(self.world_model, human_agent_index, cell)
            yield goal, weight

    def _is_cell_reachable(self, start_pos: Tuple[int, int], target_pos: Tuple[int, int], max_depth: int = 50) -> bool:
        """Check if target cell is reachable from start via BFS."""
        if start_pos == target_pos:
            return True

        visited = set()
        queue = [start_pos]
        visited.add(start_pos)
        depth = 0

        while queue and depth < max_depth:
            depth += 1
            next_queue = []
            for pos in queue:
                for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                    nx, ny = pos[0] + dx, pos[1] + dy
                    next_pos = (nx, ny)

                    if next_pos in visited:
                        continue

                    # Check if this cell is passable
                    cell = self.world_model.grid.get(nx, ny)
                    is_passable = cell is None or (hasattr(cell, "can_overlap") and cell.can_overlap())

                    # Only mark as reachable if target is actually passable
                    if next_pos == target_pos:
                        return is_passable

                    if is_passable:
                        visited.add(next_pos)
                        next_queue.append(next_pos)

            queue = next_queue

        return False


@dataclass
class MCTSConfig:
    """
    Configuration for MCTS planner aligned with the EMPO framework.

    EMPO Parameters (from paper):
        zeta (ζ): Risk aversion parameter for goal aggregation. ζ > 1 means
            the robot prefers humans having reliable outcomes over uncertain ones.
            Default: 2.0 (as recommended in paper Table 2)

        xi (ξ): Inequality aversion parameter. ξ ≥ 1 protects each human's
            "last bit of power". Default: 1.0

        eta (η): Intertemporal inequality aversion. η > 1 limits trading off
            current vs later human power. Default: 1.1

        beta_r: Robot's softmax rationality. β_r < ∞ enables soft optimization
            and exploration. Default: 5.0

    MCTS Parameters:
        num_simulations: Number of MCTS simulations per search
        max_depth: Maximum rollout depth
        exploration_c: UCT exploration constant
        gamma_h: Human's discount factor
        gamma_r: Robot's discount factor
        num_goal_samples: Goals sampled per human for power estimation
    """
    # MCTS parameters
    num_simulations: int = 200
    max_depth: int = 20
    exploration_c: float = 7.0  # Higher = more exploration of untried actions

    # EMPO discount factors
    gamma_h: float = 0.99
    gamma_r: float = 0.99

    # EMPO power metric parameters (from paper Table 2)
    zeta: float = 2.0      # ζ: risk aversion, preference for reliability
    xi: float = 1.0        # ξ: inequality aversion across humans
    eta: float = 1.1       # η: intertemporal inequality aversion
    beta_r: float = 5.0    # β_r: robot softmax rationality

    # Goal sampling
    num_goal_samples: int = 10

    # Logging
    verbose: bool = False
    log_every: int = 50

    # Performance options
    use_transition_probabilities: bool = True
    slow_transition_secs: float = 2.0
    debug_timing: bool = False

    # Rollout policy
    greedy_rollout: bool = False  # Use greedy rollout (pick action that maximizes U_r) instead of random - WARNING: greedy is 4x slower!

    # Numerical stability
    epsilon_x: float = 1e-6  # Added to X_h to avoid division by zero


@dataclass
class MCTSSearchResult:
    """Results from an MCTS search for visualization and analysis."""
    action_distribution: Dict[Tuple[int, ...], float]
    best_action: Tuple[int, ...]
    visit_counts: Dict[Tuple[int, ...], int]
    q_values: Dict[Tuple[int, ...], float]
    total_simulations: int
    human_power_estimates: Dict[int, float] = field(default_factory=dict)  # X_h per human
    aggregate_power: float = 0.0  # U_r
    search_time_secs: float = 0.0


class MCTSPlanner:
    """
    MCTS planner implementing the EMPO framework for human power maximization.

    This planner branches only on robot actions while sampling human actions
    from a provided policy prior. The robot's objective is to maximize
    aggregate human power (ICCEA power) as defined in the EMPO paper.

    Key EMPO Concepts:
        - V_h^e(s, g_h): Effective goal attainment probability for human h
        - X_h(s) = Σ V_h^e(s, g_h)^ζ: Individual power metric
        - U_r(s) = -(Σ_h X_h(s)^{-ξ})^η: Robot's intrinsic reward
        - W_h(s) = log₂(X_h): Power in bits (effective binary choices)

    Reference:
        Heitzig & Potham (2025). "Model-Based Soft Maximization of Suitable
        Metrics of Long-Term Human Power." arXiv:2508.00159v2
    """

    def __init__(
        self,
        world_model: Any,
        human_policy_prior: Any,
        human_agent_indices: List[int],
        robot_agent_indices: List[int],
        goal_sampler: Optional[Any] = None,
        goal_generator: Optional[Any] = None,
        config: Optional[MCTSConfig] = None,
    ):
        self.world_model = world_model
        self.human_policy_prior = human_policy_prior
        self.human_agent_indices = human_agent_indices
        self.robot_agent_indices = robot_agent_indices
        self.goal_sampler = goal_sampler
        self.config = config or MCTSConfig()

        # Auto-create default goal generator if none provided
        if goal_generator is None and goal_sampler is None:
            self.goal_generator = _MCTSDefaultGoalGenerator(world_model)
            if self.config.verbose:
                print("[MCTS] No goal generator provided, using default ReachCell goals")
        else:
            self.goal_generator = goal_generator

        self.num_actions = world_model.action_space.n
        self.num_robots = len(robot_agent_indices)
        self.robot_action_tuples = self._enumerate_robot_actions()

        # MCTS statistics
        self.N: Dict[Tuple[Any, Tuple[int, ...]], int] = {}
        self.W: Dict[Tuple[Any, Tuple[int, ...]], float] = {}
        self._sim_counter = 0

        # Track previous action to filter redundant moves
        self._prev_action: Optional[Tuple[int, ...]] = None

        # Cache for goal lists per human
        self._goal_cache: Dict[int, List[Tuple[PossibleGoal, float]]] = {}

    def _enumerate_robot_actions(self) -> List[Tuple[int, ...]]:
        actions = []
        total = self.num_actions ** self.num_robots
        for idx in range(total):
            rem = idx
            tup = []
            for _ in range(self.num_robots):
                tup.append(rem % self.num_actions)
                rem //= self.num_actions
            actions.append(tuple(tup))
        return actions

    def _filter_redundant_actions(
        self, actions: List[Tuple[int, ...]], prev_action: Optional[Tuple[int, ...]]
    ) -> List[Tuple[int, ...]]:
        """
        Filter out actions that are inverse of the previous action.

        For single-robot case: filters Left after Right, Right after Left.
        For multi-robot case: filters if ANY robot would do inverse of its prev action.
        """
        if prev_action is None:
            return actions

        filtered = []
        for action in actions:
            is_redundant = False
            for i in range(len(action)):
                prev_a = prev_action[i] if i < len(prev_action) else None
                curr_a = action[i]
                if prev_a is not None and INVERSE_ACTIONS.get(prev_a) == curr_a:
                    is_redundant = True
                    break
            if not is_redundant:
                filtered.append(action)

        # Always return at least one action (fallback to all if everything filtered)
        return filtered if filtered else actions

    def _combine_actions(
        self, human_actions: List[int], robot_actions: Tuple[int, ...]
    ) -> List[int]:
        num_agents = len(self.human_agent_indices) + len(self.robot_agent_indices)
        full = [0] * num_agents
        for i, idx in enumerate(self.human_agent_indices):
            full[idx] = human_actions[i]
        for i, idx in enumerate(self.robot_agent_indices):
            full[idx] = robot_actions[i]
        return full

    def _sample_human_actions(
        self, state: Any, goals: Optional[Dict[int, Any]] = None
    ) -> List[int]:
        actions = []
        for agent_idx in self.human_agent_indices:
            if goals is not None and agent_idx in goals:
                action = self.human_policy_prior.sample(state, agent_idx, goals[agent_idx])
            else:
                action = self.human_policy_prior.sample(state, agent_idx)
            actions.append(action)
        return actions

    def _sample_goals(self, state: Any) -> Optional[Dict[int, Any]]:
        if self.goal_sampler is None and self.goal_generator is None:
            return None
        goals: Dict[int, Any] = {}
        for agent_idx in self.human_agent_indices:
            if self.goal_sampler is not None:
                goal, _ = self.goal_sampler.sample(state, agent_idx)
                goals[agent_idx] = goal
            else:
                for goal, _ in self.goal_generator.generate(state, agent_idx):
                    goals[agent_idx] = goal
                    break
        return goals

    def _get_goals_for_human(self, state: Any, human_idx: int) -> List[Tuple[PossibleGoal, float]]:
        """
        Get list of (goal, weight) pairs for a human.

        NOTE: We don't cache goals because reachability changes as rocks are pushed.
        For goal generators, we use ALL generated goals (not limited by num_goal_samples).
        For goal samplers, we limit to num_goal_samples.
        """
        goals = []
        if self.goal_generator is not None:
            # Use all goals from generator (don't limit to num_goal_samples)
            for goal, weight in self.goal_generator.generate(state, human_idx):
                goals.append((goal, weight))
        elif self.goal_sampler is not None:
            # Limit sampled goals to num_goal_samples
            for _ in range(self.config.num_goal_samples):
                goal, weight = self.goal_sampler.sample(state, human_idx)
                goals.append((goal, weight))

        return goals

    def _transition(
        self, state: Any, robot_action: Tuple[int, ...], goals: Optional[Dict[int, Any]]
    ) -> Any:
        t0 = time.time()
        saved_state = self.world_model.get_state()
        self.world_model.set_state(state)
        human_actions = self._sample_human_actions(state, goals)
        full_actions = self._combine_actions(human_actions, robot_action)

        if self.config.use_transition_probabilities:
            transitions = self.world_model.transition_probabilities(state, full_actions)
            if transitions is None or len(transitions) == 0:
                next_state = state
            else:
                probs = [p for p, _ in transitions]
                next_state = random.choices([s for _, s in transitions], weights=probs, k=1)[0]
        else:
            step_result = self.world_model.step(full_actions)
            if len(step_result) == 5:
                _, _, terminated, truncated, _ = step_result
            else:
                _, _, done, _ = step_result
                terminated = done
                truncated = False
            next_state = self.world_model.get_state()
        self.world_model.set_state(saved_state)
        if self.config.debug_timing:
            dt = time.time() - t0
            if dt >= self.config.slow_transition_secs:
                print(f"[MCTS] slow transition: {dt:.3f}s action={robot_action}")
        return next_state

    def _compute_goal_attainment(self, state: Any, human_idx: int, goal: PossibleGoal, _state_is_set: bool = False) -> float:
        """
        Estimate V_h^e(s, g_h) - the effective goal attainment probability.

        Uses BFS-based reachability: 1.0 if goal is reachable, 0.0 otherwise.
        This is a heuristic approximation of the true value function.

        Args:
            state: The state to evaluate
            human_idx: Index of the human agent
            goal: The goal to check
            _state_is_set: Internal flag - if True, assumes world_model state is already set
        """
        # Check if goal is already achieved
        if goal.is_achieved(state):
            return 1.0

        # Check if goal is reachable via BFS
        if not hasattr(goal, 'target_pos'):
            # Not a ReachCell goal, conservative estimate
            return 0.0

        target_pos = goal.target_pos
        agent_state = state[1][human_idx]
        start_pos = (int(agent_state[0]), int(agent_state[1]))

        # CRITICAL: Set world_model state so grid reflects correct rock positions!
        # But only if caller hasn't already set it (for performance)
        if not _state_is_set:
            saved_state = self.world_model.get_state()
            self.world_model.set_state(state)

        # Simple BFS to check reachability (pass state for agent positions)
        reachable = self._is_reachable(start_pos, target_pos, state)

        # Restore previous state if we changed it
        if not _state_is_set:
            self.world_model.set_state(saved_state)

        return 1.0 if reachable else 0.0

    def _is_reachable(self, start_pos: Tuple[int, int], target_pos: Tuple[int, int], state: Any, max_depth: int = 50) -> bool:
        """Check if target_pos is reachable from start_pos via BFS.

        CRITICAL: Must check that target is not only grid-passable but also not
        occupied by an agent or rock. Gets rock positions from state tuple to
        avoid MultiGridEnv.set_state() bug that doesn't clear old agent positions.

        Args:
            start_pos: Starting position
            target_pos: Target position to reach
            state: Current state (step_count, agent_states, mobile_objects, mutable_objects)
            max_depth: Maximum BFS depth
        """
        if start_pos == target_pos:
            return True

        # Get agent and rock positions from state tuple (NOT from grid!)
        _, agent_states, mobile_objects, _ = state

        agent_positions = set()
        for agent_state in agent_states:
            if agent_state is not None and len(agent_state) >= 2:
                agent_pos = (int(agent_state[0]), int(agent_state[1]))
                agent_positions.add(agent_pos)

        rock_positions = set()
        for mobile_obj in mobile_objects:
            if mobile_obj[0] == 'rock':
                rock_pos = (int(mobile_obj[1]), int(mobile_obj[2]))
                rock_positions.add(rock_pos)

        visited = set()
        queue = [start_pos]
        visited.add(start_pos)
        depth = 0

        while queue and depth < max_depth:
            depth += 1
            next_queue = []
            for pos in queue:
                # Check 4 neighbors
                for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                    nx, ny = pos[0] + dx, pos[1] + dy
                    next_pos = (nx, ny)

                    if next_pos in visited:
                        continue

                    # Check if blocked by rock or agent
                    if next_pos in rock_positions or next_pos in agent_positions:
                        continue

                    # Check if blocked by static obstacles (walls)
                    # Use immutable_obstacles from world_model which doesn't change
                    cell = self.world_model.grid.get(nx, ny)
                    if cell is not None and hasattr(cell, 'type') and cell.type in ['wall', 'lava', 'magicwall']:
                        continue

                    # Cell is passable!
                    if next_pos == target_pos:
                        return True

                    visited.add(next_pos)
                    next_queue.append(next_pos)

            queue = next_queue

        return False

    def _compute_human_power(self, state: Any, human_idx: int) -> float:
        """
        Compute X_h(s) = Σ_{g_h} V_h^e(s, g_h)^ζ for a single human.

        This is the EMPO individual power metric (Equation 7 in paper).
        If the human is terminated, returns 0 (no power).

        CRITICAL: Being alive adds bonus equal to sum of reachable goals to massively
        penalize killing humans.
        """
        # Check if human is terminated
        if hasattr(self.world_model, 'agents'):
            agent = self.world_model.agents[human_idx]
            if hasattr(agent, 'terminated') and agent.terminated:
                return 0.0  # Terminated human has no power

        goals = self._get_goals_for_human(state, human_idx)
        if not goals:
            return self.config.epsilon_x

        # OPTIMIZATION: Set world_model state once for all goal checks
        # This avoids expensive set_state/get_state for every goal
        saved_state = self.world_model.get_state()
        self.world_model.set_state(state)

        zeta = self.config.zeta
        x_h = 0.0
        num_reachable = 0

        for goal, weight in goals:
            # Pass _state_is_set=True to skip redundant state setting
            v_e = self._compute_goal_attainment(state, human_idx, goal, _state_is_set=True)
            # X_h = Σ V_h^e(s, g_h)^ζ (weighted by goal probability)
            x_h += weight * (v_e ** zeta)
            if v_e > 0:
                num_reachable += 1

        # Restore previous world_model state
        self.world_model.set_state(saved_state)

        # CRITICAL: Add massive bonus for being alive = number of reachable goals
        # This ensures killing a human is always worse than any possible goal rearrangement
        x_h += num_reachable

        # Ensure X_h > 0 for numerical stability
        return max(x_h, self.config.epsilon_x)

    def _compute_intrinsic_reward(self, state: Any) -> float:
        """
        Compute U_r(s) = -(Σ_h X_h(s)^{-ξ})^η

        This is the EMPO aggregate power metric (Equation 8 in paper).
        The robot's intrinsic reward encourages:
        - Increasing individual human power (X_h)
        - Reducing inequality (via ξ)
        - Limiting intertemporal power trading (via η)

        CRITICAL: If any human is terminated, returns very negative value (-1e9).
        """
        xi = self.config.xi
        eta = self.config.eta
        epsilon = self.config.epsilon_x

        # Check for terminated humans - huge penalty
        if hasattr(self.world_model, 'agents'):
            for human_idx in self.human_agent_indices:
                agent = self.world_model.agents[human_idx]
                if hasattr(agent, 'terminated') and agent.terminated:
                    return -1e9  # Massive penalty for killing humans!

        # Compute X_h for each human
        sum_x_neg_xi = 0.0
        for human_idx in self.human_agent_indices:
            x_h = self._compute_human_power(state, human_idx)
            # X_h^{-ξ} (inequality-averse aggregation)
            sum_x_neg_xi += (x_h + epsilon) ** (-xi)

        if sum_x_neg_xi <= 0:
            return 0.0

        # U_r = -(Σ_h X_h^{-ξ})^η
        u_r = -((sum_x_neg_xi) ** eta)

        return u_r

    def _compute_human_power_bits(self, state: Any, human_idx: int) -> float:
        """
        Compute W_h(s) = log₂(X_h(s)) - power in bits.

        This represents the "effective number of binary choices" the human has.
        """
        x_h = self._compute_human_power(state, human_idx)
        if x_h <= 0:
            return float('-inf')
        return math.log2(x_h)

    def _rollout(self, state: Any) -> float:
        """
        Perform a rollout and accumulate discounted intrinsic rewards.

        Returns the discounted sum of FUTURE U_r values along the rollout trajectory.
        Does NOT include U_r(state) since that's already counted in _simulate.

        Uses greedy policy if config.greedy_rollout=True, otherwise random.
        """
        total = 0.0
        gamma = self.config.gamma_r  # Start with gamma, not 1.0
        goals = self._sample_goals(state)
        current = state

        for _ in range(self.config.max_depth):
            if self.config.greedy_rollout:
                # Greedy: pick action that maximizes U_r(next_state)
                best_action = None
                best_u_r = float('-inf')

                for action in self.robot_action_tuples:
                    test_next = self._transition(current, action, goals)
                    u_r_test = self._compute_intrinsic_reward(test_next)
                    if u_r_test > best_u_r:
                        best_u_r = u_r_test
                        best_action = action

                robot_action = best_action
            else:
                # Random rollout
                robot_action = random.choice(self.robot_action_tuples)

            next_state = self._transition(current, robot_action, goals)

            # Compute intrinsic reward at next state
            u_r = self._compute_intrinsic_reward(next_state)
            total += gamma * u_r
            gamma *= self.config.gamma_r

            current = next_state

        return total

    def _uct_score(self, state: Any, action: Tuple[int, ...]) -> float:
        """Compute UCT score for action selection."""
        n_sa = self.N.get((state, action), 0)
        w_sa = self.W.get((state, action), 0.0)
        n_s = sum(self.N.get((state, a), 0) for a in self.robot_action_tuples)

        if n_sa == 0:
            return float("inf")

        # Q-value (average return)
        q = w_sa / n_sa

        # UCT exploration bonus
        u = self.config.exploration_c * math.sqrt(math.log(n_s + 1) / n_sa)

        return q + u

    def _select_action(self, state: Any) -> Tuple[int, ...]:
        """Select action using UCT, filtering redundant actions."""
        available = self._filter_redundant_actions(self.robot_action_tuples, self._prev_action)
        return max(available, key=lambda a: self._uct_score(state, a))

    def _compute_softmax_policy(self, state: Any) -> Dict[Tuple[int, ...], float]:
        """
        Compute π_r(s)(a) ∝ (-Q_r(s, a))^{-β_r} as per EMPO Equation 5.

        This implements soft maximization with the power-law form.
        """
        beta_r = self.config.beta_r

        # Get Q-values from visit statistics
        q_values = {}
        for action in self.robot_action_tuples:
            n_sa = self.N.get((state, action), 0)
            w_sa = self.W.get((state, action), 0.0)
            if n_sa > 0:
                q_values[action] = w_sa / n_sa
            else:
                q_values[action] = float('-inf')

        # Compute softmax with power-law: π(a) ∝ (-Q(s,a))^{-β_r}
        # Note: Q values are negative (since U_r < 0), so -Q > 0
        weights = {}
        for action, q in q_values.items():
            if q == float('-inf'):
                weights[action] = 0.0
            else:
                # -Q_r is positive, raise to power -β_r
                neg_q = -q
                if neg_q > 0:
                    weights[action] = neg_q ** (-beta_r)
                else:
                    # Q = 0 case, use small weight
                    weights[action] = 1e-10

        # Normalize to get probabilities
        total_weight = sum(weights.values())
        if total_weight <= 0:
            # Uniform fallback
            n = len(self.robot_action_tuples)
            return {a: 1.0/n for a in self.robot_action_tuples}

        return {a: w / total_weight for a, w in weights.items()}

    def clear_statistics(self, clear_prev_action: bool = False) -> None:
        """Clear all accumulated MCTS statistics."""
        self.N.clear()
        self.W.clear()
        self._sim_counter = 0
        self._goal_cache.clear()
        if clear_prev_action:
            self._prev_action = None

    def set_prev_action(self, action: Optional[Tuple[int, ...]]) -> None:
        """Set the previous action for redundancy filtering."""
        self._prev_action = action

    def get_prev_action(self) -> Optional[Tuple[int, ...]]:
        """Get the previous action."""
        return self._prev_action

    def search(self, state: Any, clear_stats: bool = True) -> Dict[Tuple[int, ...], float]:
        """
        Run MCTS search from the given state.

        Args:
            state: The current state to search from.
            clear_stats: If True, clear statistics before searching.

        Returns:
            Dictionary mapping robot action tuples to visit proportions.
        """
        if clear_stats:
            self.clear_statistics()

        for i in range(self.config.num_simulations):
            self._simulate(state)
            if self.config.verbose and (i + 1) % self.config.log_every == 0:
                print(f"[MCTS] simulations: {i + 1}/{self.config.num_simulations}")

        # Filter to non-redundant actions
        available_actions = self._filter_redundant_actions(self.robot_action_tuples, self._prev_action)

        counts = {a: self.N.get((state, a), 0) for a in available_actions}
        total = sum(counts.values()) or 1
        action_dist = {a: c / total for a, c in counts.items()}

        # Update previous action for next search
        if action_dist:
            best = max(action_dist.items(), key=lambda kv: kv[1])[0]
            self._prev_action = best

        return action_dist

    def search_with_result(self, state: Any, clear_stats: bool = True) -> MCTSSearchResult:
        """
        Run MCTS search and return detailed results for visualization.

        Returns:
            MCTSSearchResult with action distribution, Q-values, power estimates, etc.
        """
        start_time = time.time()

        if clear_stats:
            self.clear_statistics()

        for i in range(self.config.num_simulations):
            self._simulate(state)
            if self.config.verbose and (i + 1) % self.config.log_every == 0:
                print(f"[MCTS] simulations: {i + 1}/{self.config.num_simulations}")

        # Filter to non-redundant actions
        available_actions = self._filter_redundant_actions(self.robot_action_tuples, self._prev_action)

        # Compute action distribution from visit counts (only for available actions)
        counts = {a: self.N.get((state, a), 0) for a in available_actions}
        total_visits = sum(counts.values()) or 1
        action_dist = {a: c / total_visits for a, c in counts.items()}

        # Compute Q-values (for all actions, for analysis)
        q_values = {}
        for action in self.robot_action_tuples:
            n_sa = self.N.get((state, action), 0)
            w_sa = self.W.get((state, action), 0.0)
            q_values[action] = w_sa / n_sa if n_sa > 0 else 0.0

        # Find best action from available (non-redundant) actions
        best_action = max(action_dist.items(), key=lambda kv: kv[1])[0]

        # Update previous action for next search
        self._prev_action = best_action

        # Compute human power estimates
        human_power = {}
        for human_idx in self.human_agent_indices:
            human_power[human_idx] = self._compute_human_power(state, human_idx)

        # Compute aggregate power (U_r)
        aggregate_power = self._compute_intrinsic_reward(state)

        search_time = time.time() - start_time

        return MCTSSearchResult(
            action_distribution=action_dist,
            best_action=best_action,
            visit_counts=counts,
            q_values=q_values,
            total_simulations=self.config.num_simulations,
            human_power_estimates=human_power,
            aggregate_power=aggregate_power,
            search_time_secs=search_time
        )

    def best_action(self, state: Any) -> Tuple[int, ...]:
        """Run search and return the best action."""
        dist = self.search(state)
        return max(dist.items(), key=lambda kv: kv[1])[0]

    def _simulate(self, state: Any, depth: int = 0) -> float:
        """
        Run a single MCTS simulation with recursive tree expansion.

        Returns the discounted return from this state.
        """
        if depth == 0:
            self._sim_counter += 1
            if self.config.verbose and self._sim_counter % self.config.log_every == 0:
                print(f"[MCTS] simulate {self._sim_counter}")

        # Terminal condition: max depth reached
        if depth >= self.config.max_depth:
            return self._compute_intrinsic_reward(state)

        # Select action using UCT
        action = self._select_action(state)
        goals = self._sample_goals(state)
        next_state = self._transition(state, action, goals)

        # Check if this (state, action) has been visited before
        key = (state, action)
        n_sa = self.N.get(key, 0)

        if n_sa == 0:
            # First visit to this (state, action) - expand with rollout
            immediate_reward = self._compute_intrinsic_reward(next_state)
            rollout_value = self._rollout(next_state)
            total_return = immediate_reward + self.config.gamma_r * rollout_value
        else:
            # Already expanded - recurse deeper
            immediate_reward = self._compute_intrinsic_reward(next_state)
            future_value = self._simulate(next_state, depth + 1)
            total_return = immediate_reward + self.config.gamma_r * future_value

        # Update statistics for this (state, action) pair
        self.N[key] = self.N.get(key, 0) + 1
        self.W[key] = self.W.get(key, 0.0) + total_return

        return total_return

    def get_power_analysis(self, state: Any) -> Dict[str, Any]:
        """
        Get detailed power analysis for current state.

        Returns dictionary with:
            - human_power_X: X_h values for each human
            - human_power_bits: W_h = log₂(X_h) for each human
            - aggregate_power_U: U_r value
            - goal_achievement: Which goals are achieved per human
        """
        analysis = {
            'human_power_X': {},
            'human_power_bits': {},
            'goal_achievement': {},
        }

        for human_idx in self.human_agent_indices:
            x_h = self._compute_human_power(state, human_idx)
            w_h = math.log2(x_h) if x_h > 0 else float('-inf')

            analysis['human_power_X'][human_idx] = x_h
            analysis['human_power_bits'][human_idx] = w_h

            # Check goal achievement
            goals = self._get_goals_for_human(state, human_idx)
            achieved = sum(1 for g, _ in goals if g.is_achieved(state))
            analysis['goal_achievement'][human_idx] = {
                'achieved': achieved,
                'total': len(goals)
            }

        analysis['aggregate_power_U'] = self._compute_intrinsic_reward(state)

        return analysis

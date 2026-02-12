"""
Rollout collection: run episodes and collect transitions.

The rollout module is phase-agnostic. It runs episodes using a caller-provided
policy function and records transitions for the replay buffer. The caller
decides:
    - Which policy each agent uses (via policy_fn)
    - Which agent's action to record (via record_agent_idx)
    - Which goal to use (via goal argument)

Phase 1 usage:
    Human explores with ε-greedy from QhNet, robot acts randomly.
    Record the human's action. Reward = goal achieved at next state.

Phase 2 usage:
    Human acts with frozen Phase 1 policy, robot explores from QrNet.
    Record the robot's action. Reward = goal achieved at next state.
    (The actual Q_r target is computed at training time from V_h^e.)

Design choices:
    - policy_fn(state) → list[int] returns actions for ALL agents.
      This keeps rollout decoupled from phase-specific logic.
    - EpisodeData is a struct-of-tensors that can be pushed directly
      to the replay buffer via push_batch().
    - State encodings are computed once and reused: the next_state of
      step t becomes the state of step t+1, avoiding double encoding.
    - Goal encoding is computed once and repeated for all transitions
      (one goal per episode).
"""

import random

import numpy as np
import torch
from typing import Any, Callable, List, NamedTuple, Optional

from empo.ali_learning_based.encoders import StateEncoder, GoalEncoder


# ---------------------------------------------------------------------------
# Distance utilities for reward shaping
# ---------------------------------------------------------------------------

def manhattan_distance_to_rect(
    x: int, y: int, x1: int, y1: int, x2: int, y2: int,
) -> int:
    """
    Manhattan distance from point (x, y) to the nearest point in
    rectangle [x1, y1] x [x2, y2] (inclusive).

    Returns 0 if (x, y) is inside or on the border of the rectangle.
    """
    dx = max(x1 - x, 0, x - x2)
    dy = max(y1 - y, 0, y - y2)
    return dx + dy


def make_pbrs_shaper(
    goal: Any,
    human_agent_idx: int,
    gamma: float,
    max_dist: float,
    weight: float = 1.0,
) -> Callable:
    """
    Create a potential-based reward shaping (PBRS) function for one episode.

    Potential:  Phi(s) = -manhattan_distance(human_pos, goal_rect) / max_dist
    Shaping:    F(s,s') = gamma * Phi(s') - Phi(s)
    Result:     r_shaped = r_original + weight * F(s,s')

    PBRS provably preserves the optimal policy (Ng et al. 1999).
    Must create a new shaper per episode (it tracks internal state).

    Args:
        goal: PossibleGoal with target_rect attribute.
        human_agent_idx: Index of the human agent in the state tuple.
        gamma: Discount factor (same as Q-learning gamma).
        max_dist: Normalizer for distance (e.g. env.width + env.height).
        weight: Scaling factor for the shaping term.
    """
    x1, y1, x2, y2 = goal.target_rect
    prev_phi = [None]  # mutable closure state — reset per episode

    def _phi(state):
        _, agent_states, _, _ = state
        ax = int(agent_states[human_agent_idx][0])
        ay = int(agent_states[human_agent_idx][1])
        dist = manhattan_distance_to_rect(ax, ay, x1, y1, x2, y2)
        return -dist / max_dist

    def shaper(state, next_state, original_reward):
        phi_next = _phi(next_state)
        if prev_phi[0] is None:
            prev_phi[0] = _phi(state)
        shaping = gamma * phi_next - prev_phi[0]
        prev_phi[0] = phi_next
        return original_reward + weight * shaping

    return shaper


def make_path_pbrs_shaper(
    goal: Any,
    human_agent_idx: int,
    gamma: float,
    world_model: Any,
    weight: float = 1.0,
    path_calc: Any = None,
) -> Callable:
    """
    Create a PBRS shaper using BFS path distance (respects walls).

    Like make_pbrs_shaper but uses BFS shortest-path distance instead of
    Manhattan distance.  This gives accurate distance signals in environments
    with walls where Manhattan distance is misleading (e.g. rock_gateway
    where Bot-Left is far by path but close by Manhattan).

    The BFS distance grid is precomputed once per goal rectangle, making
    per-step lookup O(1).

    Args:
        goal: PossibleGoal with target_rect attribute.
        human_agent_idx: Index of the human agent in the state tuple.
        gamma: Discount factor (same as Q-learning gamma).
        world_model: Multigrid environment (for grid/wall info).
        weight: Scaling factor for the shaping term.
        path_calc: Optional shared PathDistanceCalculator (for caching
            across episodes).  If None, one is created internally.
    """
    from empo.learning_based.multigrid.path_distance import PathDistanceCalculator

    x1, y1, x2, y2 = goal.target_rect

    # Build or reuse calculator
    if path_calc is None:
        path_calc = PathDistanceCalculator(
            world_model.height, world_model.width,
        )

    # Extract impassable obstacles from the environment grid
    # Include walls, rocks, closed doors — anything the agent can't walk through
    obstacles: set = set()
    if hasattr(world_model, 'grid') and world_model.grid is not None:
        for _y in range(world_model.height):
            for _x in range(world_model.width):
                cell = world_model.grid.get(_x, _y)
                if cell is not None:
                    cell_type = getattr(cell, 'type', None)
                    if cell_type in ('wall', 'magicwall', 'lava', 'rock'):
                        obstacles.add((_x, _y))
                    elif cell_type == 'door' and not getattr(cell, 'is_open', False):
                        obstacles.add((_x, _y))

    # Precompute BFS distance grid for this goal rectangle (cached inside calc)
    dist_grid = path_calc.compute_distances_to_rectangle(
        (x1, y1, x2, y2), obstacles,
    )

    # Normalizer: max finite distance in the grid
    import numpy as np
    finite_dists = dist_grid[dist_grid < float('inf')]
    max_dist = float(finite_dists.max()) if len(finite_dists) > 0 else 1.0

    prev_phi: list = [None]

    def _phi(state):
        _, agent_states, _, _ = state
        ax = int(agent_states[human_agent_idx][0])
        ay = int(agent_states[human_agent_idx][1])
        d = dist_grid[ay, ax]
        if d == float('inf'):
            return -1.0
        return -d / max_dist

    def shaper(state, next_state, original_reward):
        phi_next = _phi(next_state)
        if prev_phi[0] is None:
            prev_phi[0] = _phi(state)
        shaping = gamma * phi_next - prev_phi[0]
        prev_phi[0] = phi_next
        return original_reward + weight * shaping

    return shaper


# ---------------------------------------------------------------------------
# Rock curiosity utilities
# ---------------------------------------------------------------------------

def get_rock_positions(state: tuple) -> List[tuple]:
    """
    Extract rock positions from a state tuple.

    Args:
        state: The 4-tuple (step_count, agent_states, mobile_objects, mutable_objects).

    Returns:
        List of (x, y) positions for all rocks in the state.
    """
    _, _, mobile_objects, _ = state
    rocks = []
    for obj in mobile_objects:
        if obj[0] == "rock":
            rocks.append((int(obj[1]), int(obj[2])))
    return rocks


def make_rock_curiosity_shaper(
    push_bonus: float = 1.0,
    approach_weight: float = 0.0,
    robot_agent_idx: int = 0,
    gamma: float = 0.99,
) -> Callable:
    """
    Create a reward shaper that incentivizes rock interaction.

    Provides two forms of intrinsic motivation:
    1. Push bonus: One-time reward when the robot successfully pushes a rock.
    2. Approach shaping: PBRS-style reward for getting closer to rocks.

    Args:
        push_bonus: Reward for moving a rock (default 1.0).
        approach_weight: Weight for distance-to-rock PBRS shaping. When > 0,
            robot gets reward for reducing distance to nearest rock.
        robot_agent_idx: Index of the robot in agent_states.
        gamma: Discount factor for PBRS (should match Q_r's gamma).

    Returns:
        A shaper function: shaper(state, next_state, original_reward) → shaped_reward.
    """
    prev_rocks: List[List[tuple]] = [[]]
    prev_phi: List[float] = [0.0]

    def _get_robot_pos(state: tuple) -> tuple:
        """Get robot (x, y) from state."""
        _, agent_states, _, _ = state
        agent = agent_states[robot_agent_idx]
        return (int(agent[0]), int(agent[1]))

    def _min_dist_to_rock(pos: tuple, rocks: List[tuple]) -> float:
        """Manhattan distance to nearest rock, or large value if no rocks."""
        if not rocks:
            return 100.0
        return min(abs(pos[0] - r[0]) + abs(pos[1] - r[1]) for r in rocks)

    def _phi(state: tuple) -> float:
        """Potential: negative distance to nearest rock (higher when closer)."""
        rocks = get_rock_positions(state)
        if not rocks:
            return 0.0
        robot_pos = _get_robot_pos(state)
        return -_min_dist_to_rock(robot_pos, rocks)

    def shaper(state: tuple, next_state: tuple, original_reward: float) -> float:
        reward = original_reward

        # --- Push bonus ---
        if not prev_rocks[0]:
            prev_rocks[0] = get_rock_positions(state)

        current_rocks = get_rock_positions(next_state)
        rock_moved = set(prev_rocks[0]) != set(current_rocks)
        prev_rocks[0] = current_rocks

        if rock_moved and push_bonus > 0:
            reward += push_bonus

        # --- Approach shaping (PBRS) ---
        if approach_weight > 0:
            phi_next = _phi(next_state)
            if prev_phi[0] == 0.0:
                prev_phi[0] = _phi(state)
            shaping = gamma * phi_next - prev_phi[0]
            prev_phi[0] = phi_next
            reward += approach_weight * shaping

        return reward

    return shaper


# ---------------------------------------------------------------------------
# Random start state utilities
# ---------------------------------------------------------------------------

def randomize_agent_position(env: Any, agent_idx: int) -> np.ndarray:
    """
    Teleport an agent to a random empty cell after env.reset().

    Finds all empty (None) cells on the grid and moves the agent to one
    chosen uniformly at random.  Uses the environment's built-in
    ``_move_agent_to_cell`` to handle grid and terrain tracking.

    Args:
        env: Multigrid environment (already reset).
        agent_idx: Which agent to move.

    Returns:
        The agent's new position as a numpy array.
    """
    empty: List[tuple] = []
    for y in range(env.height):
        for x in range(env.width):
            if env.grid.get(x, y) is None:
                empty.append((x, y))

    if not empty:
        return env.agents[agent_idx].pos

    pos = random.choice(empty)
    target_cell = env.grid.get(*pos)  # None for empty cell
    env._move_agent_to_cell(agent_idx, pos, target_cell)
    return env.agents[agent_idx].pos


class EpisodeData(NamedTuple):
    """Transitions from a single episode, ready for replay buffer push_batch."""
    states: torch.Tensor        # (T, state_dim)
    goals: torch.Tensor         # (T, goal_dim)
    actions: torch.Tensor       # (T,) int64
    rewards: torch.Tensor       # (T,) float32  — shaped reward (may include bonuses)
    next_states: torch.Tensor   # (T, state_dim)
    dones: torch.Tensor         # (T,) bool
    goal_rewards: torch.Tensor  # (T,) float32  — pure goal achievement (0/1)


def collect_episode(
    world_model: Any,
    state_encoder: StateEncoder,
    goal_encoder: GoalEncoder,
    goal: Any,
    policy_fn: Callable,
    record_agent_idx: int,
    reward_shaper: Optional[Callable] = None,
    randomize_start_agent: Optional[int] = None,
    done_on_goal: bool = False,
) -> EpisodeData:
    """
    Run one episode and collect transitions.

    Args:
        world_model: Multigrid environment (has reset, step, get_state).
        state_encoder: Encodes state tuples to tensors.
        goal_encoder: Encodes goal objects to tensors.
        goal: A PossibleGoal instance (fixed for the whole episode).
        policy_fn: Called as policy_fn(state_tuple) → list[int].
            Must return one action per agent.
        record_agent_idx: Which agent's action to store in transitions
            (0 = robot for Phase 2, 1 = human for Phase 1, etc.).
        reward_shaper: Optional callable(state, next_state, original_reward) → shaped_reward.
            When provided, rewards are transformed before storing. Used for
            PBRS in Phase 1 to add distance-based shaping (see make_pbrs_shaper).
        randomize_start_agent: If set, teleport this agent to a random empty
            cell after reset. Used for random-start exploration in Phase 1.
        done_on_goal: If True, terminate the episode as soon as the goal is
            achieved. This prevents post-achievement drift and makes Q-values
            represent the discounted probability of reaching the goal.

    Returns:
        EpisodeData with T transitions (T = episode length).
    """
    goal_enc = goal_encoder.encode(goal)

    states_list: List[torch.Tensor] = []
    actions_list: List[int] = []
    rewards_list: List[float] = []
    goal_rewards_list: List[float] = []
    next_states_list: List[torch.Tensor] = []
    dones_list: List[bool] = []

    world_model.reset()
    if randomize_start_agent is not None:
        randomize_agent_position(world_model, randomize_start_agent)
    state = world_model.get_state()
    state_enc = state_encoder.encode(state)

    done = False
    while not done:
        # Select actions for all agents
        action_profile = policy_fn(state)

        # Step the environment
        _, _, done, _ = world_model.step(action_profile)

        # Observe next state
        next_state = world_model.get_state()
        next_state_enc = state_encoder.encode(next_state)

        # Reward: did the goal get achieved at the next state?
        goal_achieved = float(goal.is_achieved(next_state))
        reward = (reward_shaper(state, next_state, goal_achieved)
                  if reward_shaper else goal_achieved)

        # Early termination on goal achievement
        if done_on_goal and goal_achieved > 0:
            done = True

        # Record transition
        states_list.append(state_enc)
        actions_list.append(action_profile[record_agent_idx])
        rewards_list.append(reward)
        goal_rewards_list.append(goal_achieved)
        next_states_list.append(next_state_enc)
        dones_list.append(done)

        # Advance: reuse next_state encoding as current for next step
        state = next_state
        state_enc = next_state_enc

    T = len(states_list)
    return EpisodeData(
        states=torch.stack(states_list),
        goals=goal_enc.unsqueeze(0).expand(T, -1).clone(),
        actions=torch.tensor(actions_list, dtype=torch.long),
        rewards=torch.tensor(rewards_list, dtype=torch.float32),
        next_states=torch.stack(next_states_list),
        dones=torch.tensor(dones_list, dtype=torch.bool),
        goal_rewards=torch.tensor(goal_rewards_list, dtype=torch.float32),
    )


def collect_episodes(
    world_model: Any,
    state_encoder: StateEncoder,
    goal_encoder: GoalEncoder,
    goals: List[Any],
    policy_fn: Callable,
    record_agent_idx: int,
    num_episodes: int,
) -> EpisodeData:
    """
    Run multiple episodes and concatenate all transitions.

    Each episode uses a goal from the `goals` list, cycling through
    them round-robin if num_episodes > len(goals).

    Args:
        world_model: Multigrid environment.
        state_encoder: StateEncoder instance.
        goal_encoder: GoalEncoder instance.
        goals: List of PossibleGoal instances to cycle through.
        policy_fn: Action selection function (see collect_episode).
        record_agent_idx: Which agent's action to record.
        num_episodes: Number of episodes to run.

    Returns:
        EpisodeData with all transitions concatenated.
    """
    all_states = []
    all_goals = []
    all_actions = []
    all_rewards = []
    all_goal_rewards = []
    all_next_states = []
    all_dones = []

    for i in range(num_episodes):
        goal = goals[i % len(goals)]
        ep = collect_episode(
            world_model, state_encoder, goal_encoder,
            goal, policy_fn, record_agent_idx,
        )
        all_states.append(ep.states)
        all_goals.append(ep.goals)
        all_actions.append(ep.actions)
        all_rewards.append(ep.rewards)
        all_goal_rewards.append(ep.goal_rewards)
        all_next_states.append(ep.next_states)
        all_dones.append(ep.dones)

    return EpisodeData(
        states=torch.cat(all_states),
        goals=torch.cat(all_goals),
        actions=torch.cat(all_actions),
        rewards=torch.cat(all_rewards),
        next_states=torch.cat(all_next_states),
        dones=torch.cat(all_dones),
        goal_rewards=torch.cat(all_goal_rewards),
    )

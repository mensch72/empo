"""
Phase 1 training: learn Q_h^m(s, g, a_h) — the human's goal-conditioned Q-values.

This module implements DQN training for the human agent's Q-function.
The human explores using epsilon-greedy OR UCB (Upper Confidence Bound)
with Boltzmann action selection, while the robot acts randomly.
Transitions are collected and stored in a replay buffer, then used for
off-policy TD learning.

Equation (from backward induction):
    Q_h(s, a, g) = E[reward(s') + gamma * (1 - done) * V_h(s', g)]
    V_h(s, g) = sum_a pi(a|s,g) * Q_h(s, a, g)
    pi(a|s,g) = softmax(beta_h * Q_h(s, a, g))  (Boltzmann policy)

    reward(s') = 1.0 if goal g is achieved at state s', else 0.0

Exploration strategies:
    - **epsilon-greedy**: With probability epsilon, take random action;
      otherwise follow Boltzmann policy. Epsilon decays linearly.
    - **UCB** (Upper Confidence Bound): Choose actions by
      a = argmax_a [Q(s,a,g) + c * sqrt(ln(N(s)) / (N(s,a) + 1))]
      where N(s) = visit count for state s, N(s,a) = visit count for
      (state, action) pair. UCB favours under-explored actions,
      providing more directed exploration than epsilon-greedy.

Design choices:
    - Boltzmann policy (not epsilon-greedy for the *learned* policy) matches
      the backward induction formulation exactly. The temperature beta_h
      controls exploration vs exploitation.
    - Exploration (epsilon-greedy or UCB) is used only during *data collection*.
      The Boltzmann policy is used to compute V_h targets and for evaluation.
    - Target network with periodic hard updates for training stability.
    - UCB visit counts are goal-agnostic (state, action) pairs to encourage
      broad state-space coverage regardless of the current goal.
"""

import copy
import math
import random
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

from empo.ali_learning_based.encoders import StateEncoder, GoalEncoder
from empo.ali_learning_based.networks import QhNet
from empo.ali_learning_based.rollout import make_pbrs_shaper, make_path_pbrs_shaper
from empo.ali_learning_based.replay_buffer import ReplayBuffer
from empo.ali_learning_based.rollout import collect_episode, EpisodeData


# ---------------------------------------------------------------------------
# Boltzmann policy utilities
# ---------------------------------------------------------------------------

def boltzmann_probs(q_values: torch.Tensor, beta: float) -> torch.Tensor:
    """
    Compute Boltzmann (softmax) action probabilities from Q-values.

    Args:
        q_values: (batch, num_actions) or (num_actions,) Q-values.
        beta: Inverse temperature. Higher = more greedy.
              math.inf → deterministic argmax.

    Returns:
        Same shape as q_values, probabilities summing to 1 along last dim.
    """
    if beta == math.inf:
        # Deterministic argmax (break ties uniformly)
        max_q = q_values.max(dim=-1, keepdim=True).values
        mask = (q_values == max_q).float()
        return mask / mask.sum(dim=-1, keepdim=True)
    else:
        return torch.softmax(beta * q_values, dim=-1)


def boltzmann_value(q_values: torch.Tensor, beta: float) -> torch.Tensor:
    """
    Compute soft state value: V(s,g) = sum_a pi(a|s,g) * Q(s,a,g).

    Args:
        q_values: (batch, num_actions) Q-values.
        beta: Inverse temperature.

    Returns:
        (batch,) value for each state.
    """
    probs = boltzmann_probs(q_values, beta)
    return (probs * q_values).sum(dim=-1)


# ---------------------------------------------------------------------------
# UCB exploration utilities
# ---------------------------------------------------------------------------

def ucb_bonus(
    state_visits: int,
    action_visits: int,
    c: float = 2.0,
) -> float:
    """
    Compute the UCB exploration bonus for a (state, action) pair.

    bonus = c * sqrt(ln(N(s)) / (N(s,a) + 1))

    When N(s) = 0 (unvisited state), returns a large bonus.
    When N(s,a) = 0 (untried action), returns a large bonus.

    Args:
        state_visits: N(s) — number of times state s has been visited.
        action_visits: N(s,a) — number of times action a was taken in state s.
        c: Exploration constant (default 2.0). Higher = more exploration.

    Returns:
        Non-negative exploration bonus.
    """
    if state_visits <= 0:
        return c * 10.0  # large bonus for unvisited states
    return c * math.sqrt(math.log(state_visits) / (action_visits + 1))


def ucb_action_values(
    q_values: torch.Tensor,
    state_hash: Any,
    visit_counts: Dict,
    num_actions: int,
    c: float = 2.0,
) -> torch.Tensor:
    """
    Compute Q(s,a) + UCB bonus for each action.

    Args:
        q_values: (num_actions,) Q-values for the current state.
        state_hash: Hashable state identifier.
        visit_counts: Dict mapping (state_hash, action) → visit count.
        num_actions: Number of possible actions.
        c: UCB exploration constant.

    Returns:
        (num_actions,) tensor of Q-values augmented with UCB bonus.
    """
    state_n = sum(visit_counts.get((state_hash, a), 0) for a in range(num_actions))
    bonuses = torch.zeros_like(q_values)
    for a in range(num_actions):
        action_n = visit_counts.get((state_hash, a), 0)
        bonuses[a] = ucb_bonus(state_n, action_n, c)
    return q_values + bonuses


# ---------------------------------------------------------------------------
# Phase 1 policy function (for rollout)
# ---------------------------------------------------------------------------

def make_phase1_policy_fn(
    q_net: QhNet,
    state_encoder: StateEncoder,
    goal_encoder: GoalEncoder,
    goal: Any,
    beta_h: float,
    epsilon: float,
    num_actions: int,
    num_agents: int,
    human_agent_idx: int = 1,
    device: Optional[torch.device] = None,
) -> Callable:
    """
    Build a policy_fn for rollout.collect_episode during Phase 1.

    Human: epsilon-greedy with Boltzmann policy from q_net.
    Robot: uniform random.
    Other agents: uniform random.

    Args:
        q_net: Current Q-network for human.
        state_encoder: StateEncoder instance.
        goal_encoder: GoalEncoder instance.
        goal: The goal for this episode.
        beta_h: Boltzmann temperature for human policy.
        epsilon: Exploration probability for human.
        num_actions: Number of possible actions per agent.
        num_agents: Total number of agents.
        human_agent_idx: Index of the human agent (default 1).
        device: Device for network inference (None = CPU).

    Returns:
        A function: state_tuple → list[int] of actions for all agents.
    """
    _dev = device or torch.device("cpu")
    goal_enc = goal_encoder.encode(goal).unsqueeze(0).to(_dev)  # (1, goal_dim)

    def policy_fn(state_tuple):
        actions = [random.randint(0, num_actions - 1) for _ in range(num_agents)]

        # Human action: epsilon-greedy with Boltzmann
        if random.random() < epsilon:
            actions[human_agent_idx] = random.randint(0, num_actions - 1)
        else:
            state_enc = state_encoder.encode(state_tuple).unsqueeze(0).to(_dev)
            with torch.no_grad():
                q_values = q_net(state_enc, goal_enc)  # (1, num_actions)
                probs = boltzmann_probs(q_values, beta_h).squeeze(0)  # (num_actions,)
                actions[human_agent_idx] = torch.multinomial(probs.cpu(), 1).item()

        return actions

    return policy_fn


def make_phase1_ucb_policy_fn(
    q_net: QhNet,
    state_encoder: StateEncoder,
    goal_encoder: GoalEncoder,
    goal: Any,
    visit_counts: Dict,
    ucb_c: float,
    num_actions: int,
    num_agents: int,
    human_agent_idx: int = 1,
    device: Optional[torch.device] = None,
) -> Callable:
    """
    Build a UCB-based policy_fn for Phase 1 data collection.

    Human: UCB action selection — argmax_a [Q(s,a,g) + c*sqrt(ln(N(s))/(N(s,a)+1))].
    Robot: uniform random.

    The visit_counts dict is updated in-place as actions are taken.

    Args:
        q_net: Current Q-network for human.
        state_encoder: StateEncoder instance.
        goal_encoder: GoalEncoder instance.
        goal: The goal for this episode.
        visit_counts: Mutable dict mapping (state_hash, action) → count.
        ucb_c: UCB exploration constant.
        num_actions: Number of possible actions per agent.
        num_agents: Total number of agents.
        human_agent_idx: Index of the human agent.
        device: Device for network inference (None = CPU).

    Returns:
        A function: state_tuple → list[int] of actions for all agents.
    """
    _dev = device or torch.device("cpu")
    goal_enc = goal_encoder.encode(goal).unsqueeze(0).to(_dev)

    def policy_fn(state_tuple):
        actions = [random.randint(0, num_actions - 1) for _ in range(num_agents)]

        # Human action: UCB
        state_enc = state_encoder.encode(state_tuple).unsqueeze(0).to(_dev)
        with torch.no_grad():
            q_values = q_net(state_enc, goal_enc).squeeze(0).cpu()  # (num_actions,)

        augmented = ucb_action_values(
            q_values, state_tuple, visit_counts, num_actions, ucb_c,
        )
        chosen = augmented.argmax().item()
        actions[human_agent_idx] = chosen

        # Update visit counts
        visit_counts[(state_tuple, chosen)] = (
            visit_counts.get((state_tuple, chosen), 0) + 1
        )

        return actions

    return policy_fn


# ---------------------------------------------------------------------------
# Phase 1 Trainer
# ---------------------------------------------------------------------------

class Phase1Trainer:
    """
    DQN trainer for Phase 1: learning Q_h^m(s, g, a_h).

    Training loop:
        1. Collect episodes using epsilon-greedy + Boltzmann human policy
        2. Push transitions to replay buffer
        3. Sample batch, compute TD loss, update Q-network
        4. Periodically hard-copy weights to target network

    Args:
        q_net: The QhNet to train.
        state_encoder: StateEncoder for the environment.
        goal_encoder: GoalEncoder for goals.
        num_actions: Number of actions per agent.
        num_agents: Number of agents in the environment.
        human_agent_idx: Which agent is the human (default 1).
        gamma: Discount factor (default 0.99).
        beta_h: Boltzmann inverse temperature (default 10.0).
        lr: Learning rate (default 1e-3).
        buffer_capacity: Replay buffer size (default 50000).
        batch_size: Training batch size (default 64).
        target_update_freq: Steps between target network updates (default 200).
        exploration: Exploration strategy: "epsilon_greedy" or "ucb" (default "epsilon_greedy").
        epsilon_start: Initial exploration rate for epsilon-greedy (default 1.0).
        epsilon_end: Final exploration rate for epsilon-greedy (default 0.05).
        epsilon_decay_steps: Steps over which epsilon decays linearly (default 5000).
        ucb_c: UCB exploration constant (default 2.0). Only used when exploration="ucb".
        reward_shaping: Reward shaping strategy: "none" or "pbrs" (default "none").
            "pbrs" adds potential-based reward shaping using Manhattan distance
            to the goal rectangle. Preserves optimal policy (Ng et al. 1999).
        shaping_weight: Scaling factor for the PBRS shaping term (default 1.0).
    """

    def __init__(
        self,
        q_net: QhNet,
        state_encoder: StateEncoder,
        goal_encoder: GoalEncoder,
        num_actions: int,
        num_agents: int,
        human_agent_idx: int = 1,
        gamma: float = 0.99,
        beta_h: float = 10.0,
        lr: float = 1e-3,
        buffer_capacity: int = 50_000,
        batch_size: int = 64,
        target_update_freq: int = 200,
        exploration: str = "epsilon_greedy",
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay_steps: int = 5000,
        ucb_c: float = 2.0,
        reward_shaping: str = "none",
        shaping_weight: float = 1.0,
        random_start_prob: float = 0.0,
        done_on_goal: bool = True,
        device: Optional[torch.device] = None,
        parallel_collector=None,
    ):
        self.device = device or torch.device("cpu")
        self.parallel_collector = parallel_collector
        self.q_net = q_net.to(self.device)
        self.target_net = copy.deepcopy(q_net).to(self.device)
        self.target_net.eval()

        self.state_encoder = state_encoder
        self.goal_encoder = goal_encoder
        self.num_actions = num_actions
        self.num_agents = num_agents
        self.human_agent_idx = human_agent_idx

        self.gamma = gamma
        self.beta_h = beta_h
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        self.exploration = exploration
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.ucb_c = ucb_c
        self.visit_counts: Dict = {}  # (state_hash, action) → count
        self.reward_shaping = reward_shaping
        self.shaping_weight = shaping_weight
        self.random_start_prob = random_start_prob
        self.done_on_goal = done_on_goal

        self.optimizer = optim.Adam(q_net.parameters(), lr=lr)
        self.buffer = ReplayBuffer(
            capacity=buffer_capacity,
            state_dim=state_encoder.dim,
            goal_dim=goal_encoder.dim,
            device=self.device,
        )

        self.total_steps = 0
        self.train_losses: List[float] = []

    @property
    def epsilon(self) -> float:
        """Current epsilon (linear decay)."""
        progress = min(1.0, self.total_steps / max(1, self.epsilon_decay_steps))
        return self.epsilon_start + (self.epsilon_end - self.epsilon_start) * progress

    def _update_target(self):
        """Hard copy Q-net weights to target network."""
        self.target_net.load_state_dict(self.q_net.state_dict())

    def collect_and_store(
        self,
        world_model: Any,
        goals: List[Any],
        num_episodes: int = 1,
    ) -> int:
        """
        Collect episodes and push transitions to the replay buffer.

        Args:
            world_model: The environment.
            goals: List of goals to cycle through.
            num_episodes: How many episodes to run.

        Returns:
            Total number of transitions collected.
        """
        total_transitions = 0

        # Parallel collection (not supported for UCB — needs shared visit_counts)
        if (self.parallel_collector is not None
                and self.exploration != "ucb"
                and num_episodes > 1):
            return self._collect_parallel(goals, num_episodes)

        for i in range(num_episodes):
            goal = random.choice(goals)

            if self.exploration == "ucb":
                policy_fn = make_phase1_ucb_policy_fn(
                    q_net=self.q_net,
                    state_encoder=self.state_encoder,
                    goal_encoder=self.goal_encoder,
                    goal=goal,
                    visit_counts=self.visit_counts,
                    ucb_c=self.ucb_c,
                    num_actions=self.num_actions,
                    num_agents=self.num_agents,
                    human_agent_idx=self.human_agent_idx,
                    device=self.device,
                )
            else:
                policy_fn = make_phase1_policy_fn(
                    q_net=self.q_net,
                    state_encoder=self.state_encoder,
                    goal_encoder=self.goal_encoder,
                    goal=goal,
                    beta_h=self.beta_h,
                    epsilon=self.epsilon,
                    num_actions=self.num_actions,
                    num_agents=self.num_agents,
                    human_agent_idx=self.human_agent_idx,
                    device=self.device,
                )

            # Reward shaping: create a fresh PBRS shaper per episode
            reward_shaper = None
            if self.reward_shaping == "pbrs":
                max_dist = world_model.width + world_model.height
                reward_shaper = make_pbrs_shaper(
                    goal, self.human_agent_idx, self.gamma,
                    max_dist, self.shaping_weight,
                )
            elif self.reward_shaping == "pbrs_path":
                reward_shaper = make_path_pbrs_shaper(
                    goal, self.human_agent_idx, self.gamma,
                    world_model, self.shaping_weight,
                )

            # Random start: with some probability, teleport human to a
            # random empty cell.  This bootstraps Q-values for distant goals.
            randomize_agent = None
            if self.random_start_prob > 0 and random.random() < self.random_start_prob:
                randomize_agent = self.human_agent_idx

            ep = collect_episode(
                world_model=world_model,
                state_encoder=self.state_encoder,
                goal_encoder=self.goal_encoder,
                goal=goal,
                policy_fn=policy_fn,
                record_agent_idx=self.human_agent_idx,
                reward_shaper=reward_shaper,
                randomize_start_agent=randomize_agent,
                done_on_goal=self.done_on_goal,
            )

            self.buffer.push_batch(
                ep.states, ep.goals, ep.actions,
                ep.rewards, ep.next_states, ep.dones,
                ep.goal_rewards,
            )
            total_transitions += ep.states.shape[0]

        return total_transitions

    def _collect_parallel(self, goals: List[Any], num_episodes: int) -> int:
        """Collect episodes in parallel using the ParallelCollector."""
        num_goals = len(goals)
        sd = {k: v.cpu() for k, v in self.q_net.state_dict().items()}

        tasks = []
        for _ in range(num_episodes):
            tasks.append({
                "goal_idx": random.randint(0, num_goals - 1),
                "q_net_state_dict": sd,
                "epsilon": self.epsilon,
                "beta_h": self.beta_h,
                "reward_shaping": self.reward_shaping,
                "shaping_weight": self.shaping_weight,
                "gamma": self.gamma,
                "random_start": (
                    self.random_start_prob > 0
                    and random.random() < self.random_start_prob
                ),
                "done_on_goal": self.done_on_goal,
            })

        episodes = self.parallel_collector.collect_phase1_episodes(tasks)

        total = 0
        for ep in episodes:
            self.buffer.push_batch(
                ep.states, ep.goals, ep.actions,
                ep.rewards, ep.next_states, ep.dones,
                ep.goal_rewards,
            )
            total += ep.states.shape[0]
        return total

    def train_step(self) -> Optional[float]:
        """
        One gradient step on a batch from the replay buffer.

        Returns:
            Loss value, or None if buffer has too few samples.
        """
        if len(self.buffer) < self.batch_size:
            return None

        batch = self.buffer.sample(self.batch_size)

        # Current Q-values: Q(s, a, g) for the actions taken
        q_all = self.q_net(batch.states, batch.goals)       # (B, num_actions)
        q_taken = q_all.gather(1, batch.actions.unsqueeze(1)).squeeze(1)  # (B,)

        # Target: r + gamma * (1 - done) * V_target(s', g)
        with torch.no_grad():
            q_next = self.target_net(batch.next_states, batch.goals)  # (B, num_actions)
            v_next = boltzmann_value(q_next, self.beta_h)             # (B,)
            target = batch.rewards + self.gamma * (~batch.dones).float() * v_next

        loss = nn.functional.mse_loss(q_taken, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.total_steps += 1
        if self.total_steps % self.target_update_freq == 0:
            self._update_target()

        loss_val = loss.item()
        self.train_losses.append(loss_val)
        return loss_val

    def train(
        self,
        world_model: Any,
        goals: List[Any],
        num_iterations: int = 1000,
        episodes_per_iter: int = 4,
        train_steps_per_iter: int = 4,
        log_interval: int = 100,
        log_fn: Optional[Callable[[Dict], None]] = None,
    ) -> Dict[str, List[float]]:
        """
        Full Phase 1 training loop.

        Each iteration:
            1. Collect `episodes_per_iter` episodes
            2. Perform `train_steps_per_iter` gradient updates

        Args:
            world_model: The environment.
            goals: List of goals.
            num_iterations: Number of collect-then-train iterations.
            episodes_per_iter: Episodes to collect per iteration.
            train_steps_per_iter: Gradient steps per iteration.
            log_interval: Print stats every N iterations.
            log_fn: Optional callback for custom logging. Receives a dict
                with keys: iteration, epsilon, buffer_size, mean_loss.

        Returns:
            Dict with training history: {"losses": [...], "epsilons": [...]}.
        """
        history = {"losses": [], "epsilons": [], "exploration": self.exploration}

        for it in range(num_iterations):
            # Collect
            self.collect_and_store(world_model, goals, episodes_per_iter)

            # Train
            iter_losses = []
            for _ in range(train_steps_per_iter):
                loss = self.train_step()
                if loss is not None:
                    iter_losses.append(loss)

            mean_loss = sum(iter_losses) / len(iter_losses) if iter_losses else 0.0
            history["losses"].append(mean_loss)
            history["epsilons"].append(self.epsilon)

            if log_fn is not None:
                log_fn({
                    "iteration": it,
                    "epsilon": self.epsilon,
                    "buffer_size": len(self.buffer),
                    "mean_loss": mean_loss,
                })

            if log_interval and (it + 1) % log_interval == 0:
                if self.exploration == "ucb":
                    n_states = len(set(s for s, _ in self.visit_counts))
                    print(
                        f"Phase1 iter {it+1}/{num_iterations} | "
                        f"ucb(c={self.ucb_c}) | states={n_states} | "
                        f"buf={len(self.buffer)} | loss={mean_loss:.6f}"
                    )
                else:
                    print(
                        f"Phase1 iter {it+1}/{num_iterations} | "
                        f"eps={self.epsilon:.3f} | buf={len(self.buffer)} | "
                        f"loss={mean_loss:.6f}"
                    )

        return history

    def get_policy_fn(self, goal: Any) -> Callable:
        """
        Return a greedy policy function for evaluation (epsilon=0).

        The returned function has signature: state_tuple → list[int].
        Human uses the learned Boltzmann policy; robot acts randomly.
        """
        return make_phase1_policy_fn(
            q_net=self.q_net,
            state_encoder=self.state_encoder,
            goal_encoder=self.goal_encoder,
            goal=goal,
            beta_h=self.beta_h,
            epsilon=0.0,
            num_actions=self.num_actions,
            num_agents=self.num_agents,
            human_agent_idx=self.human_agent_idx,
            device=self.device,
        )

    def get_q_values(
        self, state_tuple: Any, goal: Any
    ) -> torch.Tensor:
        """
        Get Q-values for a single state-goal pair.

        Args:
            state_tuple: Raw state from world_model.get_state().
            goal: A PossibleGoal instance.

        Returns:
            (num_actions,) tensor of Q-values (on CPU).
        """
        state_enc = self.state_encoder.encode(state_tuple).unsqueeze(0).to(self.device)
        goal_enc = self.goal_encoder.encode(goal).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return self.q_net(state_enc, goal_enc).squeeze(0).cpu()

    def get_value(self, state_tuple: Any, goal: Any) -> float:
        """
        Get V_h(s, g) = sum_a pi(a|s,g) * Q(s,a,g).

        Args:
            state_tuple: Raw state.
            goal: A PossibleGoal.

        Returns:
            Scalar value.
        """
        q = self.get_q_values(state_tuple, goal)
        return boltzmann_value(q.unsqueeze(0), self.beta_h).item()

    def get_policy(self, state_tuple: Any, goal: Any) -> torch.Tensor:
        """
        Get pi(a|s, g) — Boltzmann action probabilities.

        Args:
            state_tuple: Raw state.
            goal: A PossibleGoal.

        Returns:
            (num_actions,) probability tensor.
        """
        q = self.get_q_values(state_tuple, goal)
        return boltzmann_probs(q.unsqueeze(0), self.beta_h).squeeze(0)

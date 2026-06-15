"""
Neural human policy prior for BushWorld environments (Phase 1, DQN path).

Extends :class:`~empo.learning_based.phase1.neural_policy_prior.BaseNeuralHumanPolicyPrior`
with BushWorld-specific network creation and loading, and provides
:func:`train_bushworld_neural_policy_prior`, which reuses the shared, generic
:class:`~empo.learning_based.phase1.trainer.Trainer` (Q-learning with experience
replay) — no new training algorithm is introduced.

This is the BushWorld analogue of
:mod:`empo.learning_based.multigrid.phase1.neural_policy_prior`. BushWorld's
human prior is normally the heuristic
:class:`~empo.bushworld.human_policy.ShortestPathHumanPolicyPrior`; this module
provides the *learned* alternative so BushWorld has the same Phase 1 learning
path as multigrid.
"""

import random
from collections import deque
from typing import Any, Dict, List, Optional

import torch
import torch.optim as optim

from empo.possible_goal import PossibleGoalSampler
from empo.learning_based.phase1.neural_policy_prior import BaseNeuralHumanPolicyPrior
from empo.learning_based.phase1.replay_buffer import ReplayBuffer
from empo.learning_based.phase1.trainer import Trainer

from .direct_phi_network import BushWorldDirectPhiNetwork
from .policy_prior_network import BushWorldPolicyPriorNetwork
from .q_network import BushWorldQNetwork

# Numerical stability constant for log operations.
LOG_EPS = 1e-10


def _action_encoding_from_env(world_model: Any) -> Dict[int, str]:
    """Build ``{index: action_name}`` from a BushWorld environment's actions."""
    actions = getattr(world_model, "actions", None)
    if actions is None:
        return {}
    if hasattr(actions, "__members__"):
        return {i: name.lower() for i, name in enumerate(actions.__members__.keys())}
    if hasattr(actions, "__iter__"):
        return {i: a.name.lower() for i, a in enumerate(actions)}
    return {}


class BushWorldNeuralHumanPolicyPrior(BaseNeuralHumanPolicyPrior):
    """Neural policy prior for BushWorld environments."""

    def __init__(
        self,
        q_network: BushWorldQNetwork,
        world_model: Any,
        human_agent_indices: List[int],
        goal_sampler: Optional[PossibleGoalSampler] = None,
        action_encoding: Optional[Dict[int, str]] = None,
        device: str = "cpu",
        direct_phi_network: Optional[BushWorldDirectPhiNetwork] = None,
        marginal_goal_samples: int = 10,
    ):
        policy_network = BushWorldPolicyPriorNetwork(q_network)
        super().__init__(
            q_network=q_network,
            policy_network=policy_network,
            world_model=world_model,
            human_agent_indices=human_agent_indices,
            goal_sampler=goal_sampler,
            action_encoding=action_encoding or _action_encoding_from_env(world_model),
            device=device,
        )
        self.marginal_goal_samples = marginal_goal_samples
        self.direct_phi_network = direct_phi_network
        if self.direct_phi_network is not None:
            self.direct_phi_network.to(device)
            self.direct_phi_network.eval()

    def _compute_marginal_policy(self, state: Any, agent_idx: int) -> torch.Tensor:
        """Compute the marginal policy over goals.

        Uses the direct phi network when available (fast path); otherwise
        averages the goal-conditioned policies over goals sampled from the
        goal sampler (slow path).
        """
        if self.direct_phi_network is not None:
            with torch.no_grad():
                probs = self.direct_phi_network.forward(
                    state, self.world_model, agent_idx, self.device
                )
                return probs.squeeze(0)

        goals: List[Any] = []
        if self.goal_sampler is not None:
            for _ in range(self.marginal_goal_samples):
                try:
                    goal, _weight = self.goal_sampler.sample(state, agent_idx)
                except (ValueError, RuntimeError, IndexError):
                    continue
                if goal is not None:
                    goals.append(goal)

        if not goals:
            probs = torch.ones(self.q_network.num_actions, device=self.device)
            return probs / probs.sum()

        return self.policy_network.compute_marginal(
            state, self.world_model, agent_idx, goals, device=self.device
        )

    @classmethod
    def load(
        cls,
        filepath: str,
        world_model: Any,
        human_agent_indices: List[int],
        goal_sampler: Optional[PossibleGoalSampler] = None,
        infeasible_actions_become: Optional[int] = None,
        device: str = "cpu",
    ) -> "BushWorldNeuralHumanPolicyPrior":
        """Load a saved BushWorld neural policy prior."""
        checkpoint = torch.load(filepath, map_location=device, weights_only=False)
        config = checkpoint["config"]

        saved_encoding = config.get("action_encoding") or _action_encoding_from_env(
            world_model
        )
        cls._validate_action_encoding(saved_encoding, world_model)

        q_network = BushWorldQNetwork(
            grid_height=config["grid_height"],
            grid_width=config["grid_width"],
            B=config["B"],
            num_robots=config["num_robots"],
            max_steps=config["max_steps"],
            num_actions=config["num_actions"],
            state_feature_dim=config.get("state_feature_dim", 128),
            goal_feature_dim=config.get("goal_feature_dim", 32),
            hidden_dim=config.get("hidden_dim", 128),
            beta=config.get("beta", 1.0),
            feasible_range=config.get("feasible_range", None),
            use_encoders=config.get("use_encoders", True),
        )
        q_network.load_state_dict(checkpoint["q_network_state_dict"])

        prior = cls(
            q_network=q_network,
            world_model=world_model,
            human_agent_indices=human_agent_indices,
            goal_sampler=goal_sampler,
            action_encoding=saved_encoding,
            device=device,
        )
        if infeasible_actions_become is not None:
            prior._infeasible_actions_become = infeasible_actions_become
        return prior


def _train_phi_network_step(
    phi_network: BushWorldDirectPhiNetwork,
    phi_optimizer: optim.Optimizer,
    q_network: BushWorldQNetwork,
    state_buffer: deque,
    goal_sampler: PossibleGoalSampler,
    batch_size: int,
    num_goal_samples: int,
    device: str,
) -> float:
    """Train the direct phi network to match the Q-network's goal-averaged marginal."""
    if len(state_buffer) < batch_size:
        return 0.0

    indices = random.sample(range(len(state_buffer)), batch_size)
    phi_network.train()
    q_network.eval()

    total_loss = torch.tensor(0.0, device=device)
    counted = 0
    for idx in indices:
        item = state_buffer[idx]
        state = item["state"]
        agent_idx = item["agent_idx"]
        world_model = item["world_model"]

        with torch.no_grad():
            marginal_probs = torch.zeros(q_network.num_actions, device=device)
            valid_samples = 0
            for _ in range(num_goal_samples):
                try:
                    goal, _ = goal_sampler.sample(state, agent_idx)
                except (ValueError, RuntimeError, IndexError):
                    continue
                if goal is None:
                    continue
                q_values = q_network.forward(state, world_model, agent_idx, goal, device)
                marginal_probs += q_network.get_policy(q_values).squeeze(0)
                valid_samples += 1
            if valid_samples == 0:
                continue
            target_marginal = marginal_probs / valid_samples

        phi_probs = phi_network.forward(state, world_model, agent_idx, device).squeeze(0)
        loss = -torch.sum(target_marginal * torch.log(phi_probs + LOG_EPS))
        total_loss = total_loss + loss
        counted += 1

    if counted == 0:
        return 0.0

    avg_loss = total_loss / counted
    phi_optimizer.zero_grad()
    avg_loss.backward()
    phi_optimizer.step()
    return avg_loss.item()


def train_bushworld_neural_policy_prior(
    world_model: Any,
    human_agent_indices: List[int],
    goal_sampler: PossibleGoalSampler,
    num_episodes: int = 200,
    steps_per_episode: int = 20,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    gamma: float = 0.99,
    beta: float = 1.0,
    buffer_capacity: int = 100000,
    target_update_freq: int = 100,
    state_feature_dim: int = 128,
    goal_feature_dim: int = 32,
    hidden_dim: int = 128,
    use_encoders: bool = True,
    epsilon: float = 0.3,
    updates_per_episode: int = 1,
    train_phi_network: bool = False,
    phi_learning_rate: float = 1e-3,
    phi_num_goal_samples: int = 10,
    device: str = "cpu",
    verbose: bool = True,
) -> BushWorldNeuralHumanPolicyPrior:
    """Train a neural human policy prior for a BushWorld environment.

    Uses the shared, generic DQN :class:`Trainer` (Q-learning with experience
    replay). Data is collected by repeatedly sampling a human and a goal,
    acting epsilon-greedily for that human, and letting all other players act
    randomly. Optionally trains a direct phi network for fast marginal queries.

    Returns a :class:`BushWorldNeuralHumanPolicyPrior`.
    """
    env = world_model
    num_robots = len(getattr(env, "robot_agent_indices", []))
    grid_height = env.height
    grid_width = env.width
    B = env.B
    max_steps = env.max_steps

    if hasattr(env, "action_space") and hasattr(env.action_space, "n"):
        num_actions = env.action_space.n
    else:
        num_actions = len(env.actions)

    num_players = getattr(env, "num_players", len(human_agent_indices) + num_robots)

    # Without reward shaping, Q-values are bounded by discounted binary rewards.
    feasible_range = (0.0, 1.0)

    q_network = BushWorldQNetwork(
        grid_height=grid_height,
        grid_width=grid_width,
        B=B,
        num_robots=num_robots,
        max_steps=max_steps,
        num_actions=num_actions,
        state_feature_dim=state_feature_dim,
        goal_feature_dim=goal_feature_dim,
        hidden_dim=hidden_dim,
        beta=beta,
        feasible_range=feasible_range,
        use_encoders=use_encoders,
    ).to(device)

    target_network = BushWorldQNetwork(
        grid_height=grid_height,
        grid_width=grid_width,
        B=B,
        num_robots=num_robots,
        max_steps=max_steps,
        num_actions=num_actions,
        state_feature_dim=state_feature_dim,
        goal_feature_dim=goal_feature_dim,
        hidden_dim=hidden_dim,
        beta=beta,
        feasible_range=feasible_range,
        use_encoders=use_encoders,
    ).to(device)
    target_network.load_state_dict(q_network.state_dict())

    optimizer = optim.Adam(q_network.parameters(), lr=learning_rate)
    replay_buffer = ReplayBuffer(buffer_capacity)

    phi_network = None
    phi_optimizer = None
    phi_state_buffer = None
    if train_phi_network:
        phi_network = BushWorldDirectPhiNetwork(
            grid_height=grid_height,
            grid_width=grid_width,
            B=B,
            num_robots=num_robots,
            max_steps=max_steps,
            num_actions=num_actions,
            state_feature_dim=state_feature_dim,
            hidden_dim=hidden_dim,
            use_encoders=use_encoders,
        ).to(device)
        phi_optimizer = optim.Adam(phi_network.parameters(), lr=phi_learning_rate)
        phi_state_buffer = deque(maxlen=buffer_capacity)

    trainer = Trainer(
        q_network=q_network,
        target_network=target_network,
        optimizer=optimizer,
        replay_buffer=replay_buffer,
        gamma=gamma,
        target_update_freq=target_update_freq,
        device=device,
    )

    for episode in range(num_episodes):
        env.reset()
        state = env.get_state()

        for _ in range(steps_per_episode):
            if env.is_terminal(state):
                break
            agent_idx = random.choice(human_agent_indices)
            try:
                goal, _ = goal_sampler.sample(state, agent_idx)
            except (ValueError, RuntimeError, IndexError):
                continue
            if goal is None:
                continue

            action = trainer.sample_action(state, env, agent_idx, goal, epsilon=epsilon)

            # Other players act randomly.
            actions = [random.randrange(num_actions) for _ in range(num_players)]
            actions[agent_idx] = action

            env.step(actions)
            next_state = env.get_state()

            trainer.store_transition(state, action, next_state, agent_idx, goal)
            if train_phi_network:
                phi_state_buffer.append(
                    {"state": state, "agent_idx": agent_idx, "world_model": env}
                )
            state = next_state

        for _ in range(updates_per_episode):
            trainer.train_step(batch_size)
            if train_phi_network and len(phi_state_buffer) >= batch_size:
                _train_phi_network_step(
                    phi_network=phi_network,
                    phi_optimizer=phi_optimizer,
                    q_network=q_network,
                    state_buffer=phi_state_buffer,
                    goal_sampler=goal_sampler,
                    batch_size=batch_size,
                    num_goal_samples=phi_num_goal_samples,
                    device=device,
                )

        if verbose and (episode + 1) % max(1, num_episodes // 10) == 0:
            print(f"  Phase 1 episode {episode + 1}/{num_episodes}")

    return BushWorldNeuralHumanPolicyPrior(
        q_network=q_network,
        world_model=env,
        human_agent_indices=human_agent_indices,
        goal_sampler=goal_sampler,
        action_encoding=_action_encoding_from_env(env),
        device=device,
        direct_phi_network=phi_network,
    )

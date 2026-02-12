"""
Phase 2 training: learn V_h^e(s, g) and Q_r(s, a_r).

Phase 2 trains two networks jointly:

    VheNet — V_h^e(s, g): probability human achieves goal g from state s,
        given frozen human policy from Phase 1 and current robot policy.
        TD target: achieved(s',g) + (1 - achieved(s',g)) * γ_h * V_h^e_target(s', g)

    QrNet — Q_r(s, a_r): robot Q-values for empowerment maximization.
        Intrinsic reward: U_r(s) = empowerment(V_h^e(s, g) for all goals)
        TD target: U_r(s) + γ_r * (1 - done) * max_a' Q_r_target(s', a')

Empowerment reward (single human, default params zeta=xi=eta=1):
    U_r(s) = Σ_g V_h^e(s, g)
    (Proportional to the total expected goal achievement probability.
     Maximizing this is equivalent to maximizing empowerment for the human.)

Design choices:
    - The human policy is frozen from Phase 1. During data collection the
      human acts with the trained Phase 1 Boltzmann policy (no epsilon).
    - The robot explores with epsilon-greedy from QrNet.
    - V_h^e and Q_r are trained from the same transition data.
    - V_h^e targets are computed per-goal: one VheNet handles all goals
      via its goal input.
    - Both networks use target networks with periodic hard updates.
    - Empowerment uses sum of V_h^e (stable, avoids 1/mean singularity).
"""

import copy
import random
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

from empo.ali_learning_based.encoders import StateEncoder, GoalEncoder
from empo.ali_learning_based.networks import VheNet, QrNet
from empo.ali_learning_based.replay_buffer import ReplayBuffer
from empo.ali_learning_based.rollout import (
    collect_episode,
    EpisodeData,
    make_rock_curiosity_shaper,
)


# ---------------------------------------------------------------------------
# Phase 2 policy function (for rollout)
# ---------------------------------------------------------------------------

def _get_action_toward_rock(state_tuple: Any, robot_agent_idx: int) -> Optional[int]:
    """
    Get action that moves robot toward and pushes the nearest rock.

    Uses SmallActions which are RELATIVE to agent's facing direction:
        0: still
        1: left (turn left)
        2: right (turn right)
        3: forward (move in facing direction)

    Agent direction encoding: 0=E(+x), 1=S(+y), 2=W(-x), 3=N(-y)

    This heuristic:
    1. Finds the nearest rock
    2. Determines desired absolute direction to reach rock
    3. If facing correct direction, move forward; otherwise turn

    Returns None if no rocks found.
    """
    from empo.ali_learning_based.rollout import get_rock_positions

    rocks = get_rock_positions(state_tuple)
    if not rocks:
        return None

    _, agent_states, _, _ = state_tuple
    robot = agent_states[robot_agent_idx]
    rx, ry = int(robot[0]), int(robot[1])
    robot_dir = int(robot[2])  # 0=E, 1=S, 2=W, 3=N

    # Find nearest rock
    rock_x, rock_y = min(rocks, key=lambda r: abs(rx - r[0]) + abs(ry - r[1]))

    dx = rock_x - rx
    dy = rock_y - ry

    # If adjacent to rock and facing it, move forward to push
    if abs(dx) + abs(dy) == 1:
        # Determine which direction the rock is in
        if dx == 1:
            desired_dir = 0  # East
        elif dx == -1:
            desired_dir = 2  # West
        elif dy == 1:
            desired_dir = 1  # South
        else:  # dy == -1
            desired_dir = 3  # North

        if robot_dir == desired_dir:
            return 3  # forward - push the rock!
        else:
            # Turn to face the rock
            return _turn_toward(robot_dir, desired_dir)

    # Not adjacent - navigate toward the rock
    # Determine desired direction to move
    if abs(dx) >= abs(dy):
        # Prioritize horizontal movement
        desired_dir = 0 if dx > 0 else 2  # East or West
    else:
        # Prioritize vertical movement
        desired_dir = 1 if dy > 0 else 3  # South or North

    # Random exploration (50%) to escape stuck states
    if random.random() < 0.5:
        return random.randint(0, 3)

    if robot_dir == desired_dir:
        return 3  # forward
    else:
        return _turn_toward(robot_dir, desired_dir)


def _turn_toward(current_dir: int, target_dir: int) -> int:
    """
    Return action (1=left, 2=right) to turn from current_dir toward target_dir.

    Directions: 0=E, 1=S, 2=W, 3=N (clockwise from East)
    """
    diff = (target_dir - current_dir) % 4
    if diff == 1:
        return 2  # turn right (clockwise)
    else:
        return 1  # turn left (counter-clockwise)


def make_phase2_policy_fn(
    phase1_trainer: Any,
    q_r_net: QrNet,
    state_encoder: StateEncoder,
    goals: List[Any],
    epsilon: float,
    num_actions: int,
    num_agents: int,
    robot_agent_idx: int = 0,
    human_agent_idx: int = 1,
    human_goal: Optional[Any] = None,
    device: Optional[torch.device] = None,
    rock_seek_prob: float = 0.0,
) -> Callable:
    """
    Build a policy_fn for rollout.collect_episode during Phase 2.

    Human: frozen Phase 1 policy (Boltzmann from trained QhNet, epsilon=0).
        The human pursues a fixed goal for the entire episode:
        - If human_goal is provided, the human pursues that goal.
        - Otherwise, a random goal is sampled once at creation time.

        This produces coherent human trajectories where the human
        commits to a goal and makes real progress toward it.

    Robot: epsilon-greedy from Q_r net, with optional rock-seeking behavior.

    Args:
        rock_seek_prob: Probability of taking an action toward the rock
            instead of epsilon-greedy. Used during warmup to seed the
            replay buffer with rock-push experiences.
    """
    _dev = device or torch.device("cpu")
    goal = human_goal if human_goal is not None else random.choice(goals)

    def policy_fn(state_tuple):
        actions = [0] * num_agents

        # Human: Phase 1 Boltzmann for the fixed episode goal
        human_policy = phase1_trainer.get_policy(state_tuple, goal)
        actions[human_agent_idx] = torch.multinomial(human_policy, 1).item()

        # Robot: rock-seeking or epsilon-greedy
        if rock_seek_prob > 0 and random.random() < rock_seek_prob:
            rock_action = _get_action_toward_rock(state_tuple, robot_agent_idx)
            if rock_action is not None:
                actions[robot_agent_idx] = rock_action
                return actions
            # Fall through to epsilon-greedy if no rock action

        if random.random() < epsilon:
            actions[robot_agent_idx] = random.randint(0, num_actions - 1)
        else:
            state_enc = state_encoder.encode(state_tuple).unsqueeze(0).to(_dev)
            with torch.no_grad():
                q_vals = q_r_net(state_enc).squeeze(0)  # (num_actions,)
                actions[robot_agent_idx] = q_vals.argmax().item()

        return actions

    return policy_fn


# ---------------------------------------------------------------------------
# Empowerment reward computation
# ---------------------------------------------------------------------------

def compute_empowerment(
    vhe_net: VheNet,
    state_enc: torch.Tensor,
    goal_encs: torch.Tensor,
) -> torch.Tensor:
    """
    Compute empowerment reward U_r(s) = Σ_g V_h^e(s, g).

    Args:
        vhe_net: The V_h^e network.
        state_enc: (batch, state_dim) encoded states.
        goal_encs: (num_goals, goal_dim) encoded goals.

    Returns:
        (batch,) empowerment for each state.
    """
    batch_size = state_enc.shape[0]
    num_goals = goal_encs.shape[0]

    # Expand: state (B, 1, D) x goals (1, G, D_g) → (B, G, ...)
    state_exp = state_enc.unsqueeze(1).expand(batch_size, num_goals, -1)
    goal_exp = goal_encs.unsqueeze(0).expand(batch_size, num_goals, -1)

    # Flatten to (B*G, D), (B*G, D_g) for the network
    state_flat = state_exp.reshape(batch_size * num_goals, -1)
    goal_flat = goal_exp.reshape(batch_size * num_goals, -1)

    # V_h^e for all (state, goal) pairs
    vhe_flat = vhe_net(state_flat, goal_flat)  # (B*G,)
    vhe = vhe_flat.reshape(batch_size, num_goals)  # (B, G)

    # Empowerment = sum over goals
    return vhe.sum(dim=1)  # (B,)


# ---------------------------------------------------------------------------
# Phase 2 Trainer
# ---------------------------------------------------------------------------

class Phase2Trainer:
    """
    Joint DQN trainer for Phase 2: learning V_h^e and Q_r.

    Training loop per iteration:
        1. Collect episodes with frozen human policy + robot epsilon-greedy
        2. Push transitions to replay buffer
        3. Sample batch, compute V_h^e TD loss + Q_r TD loss, update both
        4. Periodically update target networks

    Args:
        vhe_net: VheNet to train.
        q_r_net: QrNet to train.
        phase1_trainer: Frozen Phase 1 trainer (provides human policy).
        state_encoder: StateEncoder for the environment.
        goal_encoder: GoalEncoder for goals.
        goals: List of PossibleGoal instances.
        num_actions: Number of actions per agent.
        num_agents: Number of agents.
        robot_agent_idx: Robot agent index (default 0).
        human_agent_idx: Human agent index (default 1).
        gamma_h: Discount factor for V_h^e (default 0.99).
        gamma_r: Discount factor for Q_r (default 0.99).
        lr_vhe: Learning rate for VheNet (default 1e-3).
        lr_qr: Learning rate for QrNet (default 1e-3).
        buffer_capacity: Replay buffer size (default 50000).
        batch_size: Training batch size (default 64).
        target_update_freq: Steps between target updates (default 200).
        epsilon_start: Initial robot exploration rate (default 1.0).
        epsilon_end: Final robot exploration rate (default 0.05).
        epsilon_decay_steps: Steps for epsilon decay (default 5000).
    """

    def __init__(
        self,
        vhe_net: VheNet,
        q_r_net: QrNet,
        phase1_trainer: Any,
        state_encoder: StateEncoder,
        goal_encoder: GoalEncoder,
        goals: List[Any],
        num_actions: int,
        num_agents: int,
        robot_agent_idx: int = 0,
        human_agent_idx: int = 1,
        gamma_h: float = 0.99,
        gamma_r: float = 0.99,
        lr_vhe: float = 1e-3,
        lr_qr: float = 1e-3,
        buffer_capacity: int = 50_000,
        batch_size: int = 64,
        target_update_freq: int = 200,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay_steps: int = 5000,
        done_on_goal: bool = True,
        random_start_prob: float = 0.0,
        rock_curiosity_bonus: float = 0.0,
        rock_approach_weight: float = 0.0,
        device: Optional[torch.device] = None,
        parallel_collector=None,
    ):
        self.device = device or torch.device("cpu")
        self.parallel_collector = parallel_collector
        self.vhe_net = vhe_net.to(self.device)
        self.q_r_net = q_r_net.to(self.device)
        self.vhe_target = copy.deepcopy(vhe_net).to(self.device)
        self.vhe_target.eval()
        self.qr_target = copy.deepcopy(q_r_net).to(self.device)
        self.qr_target.eval()

        self.phase1_trainer = phase1_trainer
        # Freeze Phase 1 Q-net so it can't be accidentally modified during Phase 2
        for p in self.phase1_trainer.q_net.parameters():
            p.requires_grad_(False)
        self.state_encoder = state_encoder
        self.goal_encoder = goal_encoder
        self.goals = goals
        self.num_actions = num_actions
        self.num_agents = num_agents
        self.robot_agent_idx = robot_agent_idx
        self.human_agent_idx = human_agent_idx

        self.gamma_h = gamma_h
        self.gamma_r = gamma_r
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.done_on_goal = done_on_goal
        self.random_start_prob = random_start_prob
        self.rock_curiosity_bonus = rock_curiosity_bonus
        self.rock_approach_weight = rock_approach_weight

        self.optimizer_vhe = optim.Adam(vhe_net.parameters(), lr=lr_vhe)
        self.optimizer_qr = optim.Adam(q_r_net.parameters(), lr=lr_qr)
        self.buffer = ReplayBuffer(
            capacity=buffer_capacity,
            state_dim=state_encoder.dim,
            goal_dim=goal_encoder.dim,
            device=self.device,
        )

        # Pre-encode all goals (fixed for the whole training) — on device
        self._goal_encs = torch.stack(
            [goal_encoder.encode(g) for g in goals]
        ).to(self.device)

        self.total_steps = 0
        self.vhe_losses: List[float] = []
        self.qr_losses: List[float] = []

    @property
    def epsilon(self) -> float:
        """Current robot exploration epsilon."""
        progress = min(1.0, self.total_steps / max(1, self.epsilon_decay_steps))
        return self.epsilon_start + (self.epsilon_end - self.epsilon_start) * progress

    def _update_targets(self):
        """Hard copy weights to target networks."""
        self.vhe_target.load_state_dict(self.vhe_net.state_dict())
        self.qr_target.load_state_dict(self.q_r_net.state_dict())

    def collect_and_store(
        self,
        world_model: Any,
        num_episodes: int = 1,
        rock_seek_prob: float = 0.0,
    ) -> int:
        """
        Collect episodes and push to replay buffer.

        Robot's action is recorded (record_agent_idx = robot_agent_idx).
        Goal reward = goal.is_achieved(next_state) for a randomly chosen goal.

        Args:
            rock_seek_prob: Probability of robot taking rock-seeking actions.
                Used during warmup to seed buffer with rock-push experiences.

        Returns total transitions collected.
        """
        total = 0

        # Parallel collection
        if self.parallel_collector is not None and num_episodes > 1:
            return self._collect_parallel(num_episodes, rock_seek_prob)

        for i in range(num_episodes):
            goal = random.choice(self.goals)
            policy_fn = make_phase2_policy_fn(
                phase1_trainer=self.phase1_trainer,
                q_r_net=self.q_r_net,
                state_encoder=self.state_encoder,
                goals=self.goals,
                epsilon=self.epsilon,
                num_actions=self.num_actions,
                num_agents=self.num_agents,
                robot_agent_idx=self.robot_agent_idx,
                human_agent_idx=self.human_agent_idx,
                human_goal=goal,  # human pursues same goal as reward check
                device=self.device,
                rock_seek_prob=rock_seek_prob,
            )

            randomize_agent = None
            if self.random_start_prob > 0 and random.random() < self.random_start_prob:
                randomize_agent = self.robot_agent_idx

            reward_shaper = None
            if self.rock_curiosity_bonus > 0 or self.rock_approach_weight > 0:
                reward_shaper = make_rock_curiosity_shaper(
                    push_bonus=self.rock_curiosity_bonus,
                    approach_weight=self.rock_approach_weight,
                    robot_agent_idx=self.robot_agent_idx,
                    gamma=self.gamma_r,
                )

            ep = collect_episode(
                world_model=world_model,
                state_encoder=self.state_encoder,
                goal_encoder=self.goal_encoder,
                goal=goal,
                policy_fn=policy_fn,
                record_agent_idx=self.robot_agent_idx,
                reward_shaper=reward_shaper,
                randomize_start_agent=randomize_agent,
                done_on_goal=self.done_on_goal,
            )

            self.buffer.push_batch(
                ep.states, ep.goals, ep.actions,
                ep.rewards, ep.next_states, ep.dones,
                ep.goal_rewards,
            )
            total += ep.states.shape[0]

        return total

    def _collect_parallel(self, num_episodes: int, rock_seek_prob: float) -> int:
        """Collect episodes in parallel using the ParallelCollector."""
        num_goals = len(self.goals)
        p1_sd = {k: v.cpu() for k, v in self.phase1_trainer.q_net.state_dict().items()}
        qr_sd = {k: v.cpu() for k, v in self.q_r_net.state_dict().items()}

        tasks = []
        for _ in range(num_episodes):
            tasks.append({
                "goal_idx": random.randint(0, num_goals - 1),
                "phase1_state_dict": p1_sd,
                "qr_state_dict": qr_sd,
                "epsilon": self.epsilon,
                "beta_h": self.phase1_trainer.beta_h,
                "rock_seek_prob": rock_seek_prob,
                "rock_curiosity_bonus": self.rock_curiosity_bonus,
                "rock_approach_weight": self.rock_approach_weight,
                "gamma_r": self.gamma_r,
                "random_start": (
                    self.random_start_prob > 0
                    and random.random() < self.random_start_prob
                ),
                "done_on_goal": self.done_on_goal,
            })

        episodes = self.parallel_collector.collect_phase2_episodes(tasks)

        total = 0
        for ep in episodes:
            self.buffer.push_batch(
                ep.states, ep.goals, ep.actions,
                ep.rewards, ep.next_states, ep.dones,
                ep.goal_rewards,
            )
            total += ep.states.shape[0]
        return total

    def train_step(self, vhe_only: bool = False) -> Optional[Tuple[float, float]]:
        """
        One gradient step for V_h^e (and optionally Q_r).

        Args:
            vhe_only: If True, skip the Q_r update. Used during Vhe warmup
                so Q_r doesn't train on noisy empowerment signals.

        Returns (vhe_loss, qr_loss) or None if buffer too small.
        """
        if len(self.buffer) < self.batch_size:
            return None

        batch = self.buffer.sample(self.batch_size)

        # --- V_h^e update ---
        vhe_loss = self._vhe_loss(batch)
        self.optimizer_vhe.zero_grad()
        vhe_loss.backward()
        self.optimizer_vhe.step()

        # --- Q_r update (skip during warmup) ---
        if vhe_only:
            qr_l = 0.0
        else:
            qr_loss = self._qr_loss(batch)
            self.optimizer_qr.zero_grad()
            qr_loss.backward()
            self.optimizer_qr.step()
            qr_l = qr_loss.item()

        self.total_steps += 1
        if self.total_steps % self.target_update_freq == 0:
            self._update_targets()

        vhe_l = vhe_loss.item()
        self.vhe_losses.append(vhe_l)
        self.qr_losses.append(qr_l)
        return vhe_l, qr_l

    def _vhe_loss(self, batch) -> torch.Tensor:
        """
        V_h^e TD loss for the batch's stored goal.

        For each goal g:
            target = achieved(s',g) + (1-achieved(s',g)) * (1-done) * γ_h * V_h^e_target(s', g)
            loss += (V_h^e(s, g) - target)^2

        Uses goal_rewards (pure 0/1 goal achievement) NOT shaped rewards,
        because V_h^e estimates goal achievement probability in [0, 1].
        Shaped rewards (with rock bonus, approach shaping) would corrupt this.
        """
        states = batch.states          # (B, state_dim)
        next_states = batch.next_states
        goals = batch.goals            # (B, goal_dim)
        goal_rewards = batch.goal_rewards  # (B,) — pure goal achieved (0/1)
        dones = batch.dones            # (B,)

        # Predict V_h^e for the batch's goal
        vhe_pred = self.vhe_net(states, goals)  # (B,)

        # Target
        with torch.no_grad():
            vhe_next = self.vhe_target(next_states, goals)  # (B,)
            # Bootstrap only if: goal not yet achieved AND episode not over.
            # When done=True but goal_reward=0, the human failed — target is 0.
            target = goal_rewards + (1.0 - goal_rewards) * (~dones).float() * self.gamma_h * vhe_next

        return nn.functional.mse_loss(vhe_pred, target)

    def _qr_loss(self, batch) -> torch.Tensor:
        """
        Double DQN loss for Q_r.

        Uses the online network to select the best next action and the
        target network to evaluate it, reducing overestimation bias:

            a* = argmax_a' Q_r_online(s', a')
            target = U_r(s) + intrinsic_bonus + γ_r * (1 - done) * Q_r_target(s', a*)

        U_r(s) = Σ_g V_h^e(s, g) (empowerment as sum of goal achievement probs).
        intrinsic_bonus = shaped_reward - goal_reward (rock bonus + approach shaping).
        """
        states = batch.states
        next_states = batch.next_states
        actions = batch.actions        # (B,) robot actions
        rewards = batch.rewards        # (B,) shaped reward (goal + rock + approach)
        goal_rewards = batch.goal_rewards  # (B,) pure goal achievement (0/1)
        dones = batch.dones

        # Current Q_r values for the actions taken
        qr_all = self.q_r_net(states)  # (B, num_actions)
        qr_taken = qr_all.gather(1, actions.unsqueeze(1)).squeeze(1)  # (B,)

        with torch.no_grad():
            # Empowerment at current state (use target net for stable reward signal)
            ur = compute_empowerment(self.vhe_target, states, self._goal_encs)  # (B,)

            # Intrinsic bonus = total shaped reward minus pure goal reward
            # This exactly captures rock_push_bonus + approach_shaping
            intrinsic_bonus = rewards - goal_rewards
            robot_reward = ur + intrinsic_bonus

            # Double DQN: online net selects action, target net evaluates
            qr_next_online = self.q_r_net(next_states)  # (B, num_actions)
            best_actions = qr_next_online.argmax(dim=1, keepdim=True)  # (B, 1)
            qr_next_target = self.qr_target(next_states)  # (B, num_actions)
            qr_next_val = qr_next_target.gather(1, best_actions).squeeze(1)  # (B,)

            target = robot_reward + self.gamma_r * (~dones).float() * qr_next_val

        return nn.functional.mse_loss(qr_taken, target)

    def train(
        self,
        world_model: Any,
        num_iterations: int = 1000,
        episodes_per_iter: int = 4,
        train_steps_per_iter: int = 4,
        vhe_warmup_iters: int = 0,
        rock_seek_during_warmup: float = 0.0,
        log_interval: int = 100,
        log_fn: Optional[Callable[[Dict], None]] = None,
    ) -> Dict[str, List[float]]:
        """
        Full Phase 2 training loop.

        Args:
            vhe_warmup_iters: Number of initial iterations where only V_h^e
                trains (Q_r is frozen, robot acts randomly). This lets V_h^e
                converge to reasonable values before Q_r starts using
                empowerment as its reward signal.
            rock_seek_during_warmup: During warmup, probability of robot taking
                rock-seeking actions instead of random. Seeds replay buffer
                with rock-push experiences so Q_r can learn from them.

        Returns dict with training history.
        """
        history = {"vhe_losses": [], "qr_losses": [], "epsilons": []}
        total_iters = vhe_warmup_iters + num_iterations

        for it in range(total_iters):
            warming_up = it < vhe_warmup_iters
            rock_seek = rock_seek_during_warmup if warming_up else 0.0
            self.collect_and_store(world_model, episodes_per_iter, rock_seek_prob=rock_seek)

            iter_vhe = []
            iter_qr = []
            for _ in range(train_steps_per_iter):
                result = self.train_step(vhe_only=warming_up)
                if result is not None:
                    iter_vhe.append(result[0])
                    iter_qr.append(result[1])

            mean_vhe = sum(iter_vhe) / len(iter_vhe) if iter_vhe else 0.0
            mean_qr = sum(iter_qr) / len(iter_qr) if iter_qr else 0.0
            history["vhe_losses"].append(mean_vhe)
            history["qr_losses"].append(mean_qr)
            history["epsilons"].append(self.epsilon)

            if log_fn is not None:
                log_fn({
                    "iteration": it,
                    "epsilon": self.epsilon,
                    "buffer_size": len(self.buffer),
                    "vhe_loss": mean_vhe,
                    "qr_loss": mean_qr,
                })

            if log_interval and (it + 1) % log_interval == 0:
                phase = "warmup" if warming_up else "joint"
                print(
                    f"Phase2 iter {it+1}/{total_iters} ({phase}) | "
                    f"eps={self.epsilon:.3f} | buf={len(self.buffer)} | "
                    f"vhe={mean_vhe:.6f} | qr={mean_qr:.6f}"
                )

        return history

    def get_robot_policy_fn(self) -> Callable:
        """
        Return greedy robot policy for evaluation (epsilon=0).

        Human still uses frozen Phase 1 Boltzmann policy.
        """
        return make_phase2_policy_fn(
            phase1_trainer=self.phase1_trainer,
            q_r_net=self.q_r_net,
            state_encoder=self.state_encoder,
            goals=self.goals,
            epsilon=0.0,
            num_actions=self.num_actions,
            num_agents=self.num_agents,
            robot_agent_idx=self.robot_agent_idx,
            human_agent_idx=self.human_agent_idx,
            device=self.device,
        )

    def get_empowerment(self, state_tuple: Any) -> float:
        """Compute U_r(s) = Σ_g V_h^e(s, g) for a raw state."""
        state_enc = self.state_encoder.encode(state_tuple).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return compute_empowerment(
                self.vhe_net, state_enc, self._goal_encs
            ).item()

    def get_vhe(self, state_tuple: Any, goal: Any) -> float:
        """Get V_h^e(s, g) for a single state-goal pair."""
        state_enc = self.state_encoder.encode(state_tuple).unsqueeze(0).to(self.device)
        goal_enc = self.goal_encoder.encode(goal).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return self.vhe_net(state_enc, goal_enc).item()

    def get_robot_q_values(self, state_tuple: Any) -> torch.Tensor:
        """Get Q_r(s, a) for all robot actions."""
        state_enc = self.state_encoder.encode(state_tuple).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return self.q_r_net(state_enc).squeeze(0).cpu()

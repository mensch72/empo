"""
End-to-end EMPO learning pipeline.

Runs Phase 1 (human Q-learning) then Phase 2 (V_h^e + robot Q-learning)
on a multigrid environment and returns the trained system.

Usage:
    from empo.ali_learning_based.pipeline import run_empo_learning

    from empo.ali_learning_based.envs import get_env_path

    result = run_empo_learning(
        config_file=get_env_path("phase1_test.yaml"),
        num_phase1_iters=500,
        num_phase2_iters=500,
    )

    # Use the trained robot policy
    policy_fn = result.robot_policy_fn
    env = result.env
    env.reset()
    actions = policy_fn(env.get_state())
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import torch

from gym_multigrid.multigrid import MultiGridEnv, SmallActions

from empo.ali_learning_based.encoders import StateEncoder, GoalEncoder
from empo.ali_learning_based.networks import QhNet, VheNet, QrNet
from empo.ali_learning_based.phase1 import Phase1Trainer
from empo.ali_learning_based.phase2 import Phase2Trainer


@dataclass
class EMPOResult:
    """Result of running the full EMPO learning pipeline."""
    env: Any
    phase1_trainer: Phase1Trainer
    phase2_trainer: Phase2Trainer
    state_encoder: StateEncoder
    goal_encoder: GoalEncoder
    goals: List[Any]
    robot_agent_idx: int
    human_agent_idx: int
    phase1_history: Dict[str, List[float]] = field(default_factory=dict)
    phase2_history: Dict[str, List[float]] = field(default_factory=dict)

    @property
    def robot_policy_fn(self) -> Callable:
        """Greedy robot policy (human uses frozen Phase 1 Boltzmann)."""
        return self.phase2_trainer.get_robot_policy_fn()

    def get_empowerment(self, state_tuple: Any) -> float:
        """U_r(s) = sum of V_h^e across goals."""
        return self.phase2_trainer.get_empowerment(state_tuple)

    def get_vhe(self, state_tuple: Any, goal: Any) -> float:
        """V_h^e(s, g) — probability human achieves goal g."""
        return self.phase2_trainer.get_vhe(state_tuple, goal)

    def get_human_policy(self, state_tuple: Any, goal: Any) -> torch.Tensor:
        """Human action probabilities for a given goal."""
        return self.phase1_trainer.get_policy(state_tuple, goal)

    def get_robot_q_values(self, state_tuple: Any) -> torch.Tensor:
        """Robot Q-values for all actions."""
        return self.phase2_trainer.get_robot_q_values(state_tuple)


def run_empo_learning(
    config_file: str,
    *,
    # Phase 1 params
    num_phase1_iters: int = 500,
    phase1_episodes_per_iter: int = 4,
    phase1_train_steps_per_iter: int = 8,
    beta_h: float = 10.0,
    gamma_h: float = 0.99,
    phase1_lr: float = 1e-3,
    phase1_buffer: int = 100_000,
    phase1_batch: int = 64,
    phase1_target_freq: int = 100,
    phase1_exploration: str = "epsilon_greedy",
    phase1_eps_start: float = 1.0,
    phase1_eps_end: float = 0.05,
    phase1_eps_decay: int = 3000,
    phase1_ucb_c: float = 2.0,
    phase1_reward_shaping: str = "none",
    phase1_shaping_weight: float = 1.0,
    phase1_random_start_prob: float = 0.0,
    phase1_done_on_goal: bool = True,
    # Phase 2 params
    num_phase2_iters: int = 500,
    phase2_episodes_per_iter: int = 4,
    phase2_train_steps_per_iter: int = 4,
    gamma_r: float = 0.99,
    phase2_lr_vhe: float = 1e-3,
    phase2_lr_qr: float = 1e-3,
    phase2_buffer: int = 100_000,
    phase2_batch: int = 64,
    phase2_target_freq: int = 100,
    phase2_eps_start: float = 1.0,
    phase2_eps_end: float = 0.05,
    phase2_eps_decay: int = 3000,
    phase2_done_on_goal: bool = True,
    phase2_random_start_prob: float = 0.0,
    phase2_rock_curiosity: float = 0.0,
    phase2_rock_approach: float = 0.0,
    phase2_rock_seek_warmup: float = 0.0,
    phase2_vhe_warmup: int = 0,
    # General
    log_interval: int = 100,
    quiet: bool = False,
    device: str = "auto",
    num_workers: int = 0,
) -> EMPOResult:
    """
    Run the full EMPO learning pipeline on a multigrid environment.

    Args:
        config_file: Path to YAML environment config.
        num_phase1_iters: Phase 1 training iterations.
        num_phase2_iters: Phase 2 training iterations.
        (other args control hyperparameters for each phase)
        log_interval: Print progress every N iterations (0 = silent).
        quiet: Suppress all output.
        device: "auto" (default) detects MPS/CUDA, or "cpu"/"mps"/"cuda".
        num_workers: Number of parallel workers for episode collection.
            0 = auto (cpu_count - 1, min 1), 1 = sequential (no parallelism).

    Returns:
        EMPOResult with trained networks and evaluation methods.
    """
    log = 0 if quiet else log_interval

    # --- Device selection ---
    if device == "auto":
        if torch.backends.mps.is_available():
            dev = torch.device("mps")
        elif torch.cuda.is_available():
            dev = torch.device("cuda")
        else:
            dev = torch.device("cpu")
    else:
        dev = torch.device(device)

    if not quiet:
        print(f"Device: {dev}")

    # --- Environment setup ---
    env = MultiGridEnv(
        config_file=config_file,
        partial_obs=False,
        actions_set=SmallActions,
    )
    env.reset()

    num_actions = len(SmallActions.available)
    num_agents = len(env.agents)

    # Detect agent roles by color
    robot_idx = human_idx = None
    for i, agent in enumerate(env.agents):
        if agent.color in ("grey", "gray"):
            robot_idx = i
        elif agent.color == "yellow":
            human_idx = i

    if robot_idx is None or human_idx is None:
        raise ValueError(
            f"Environment must have a grey (robot) and yellow (human) agent. "
            f"Found: {[a.color for a in env.agents]}"
        )

    # Get goals from the environment
    state = env.get_state()
    goals = [g for g, _ in env.possible_goal_generator.generate(state, human_idx)]
    if not goals:
        raise ValueError("No goals found for the human agent")

    if not quiet:
        print(f"Environment: {env.width}x{env.height}, {num_agents} agents, "
              f"{num_actions} actions, {len(goals)} goals, max_steps={env.max_steps}")

    # --- Encoders ---
    se = StateEncoder(env, robot_agent_index=robot_idx, human_agent_indices=[human_idx])
    ge = GoalEncoder(env)

    # --- Parallel collector ---
    import os
    from empo.ali_learning_based.parallel_rollout import ParallelCollector

    if num_workers == 0:
        # Auto: use half of CPU cores, capped at 4 to avoid memory pressure
        num_workers = min(4, max(1, (os.cpu_count() or 2) // 2))

    parallel_collector = None
    if num_workers > 1:
        parallel_collector = ParallelCollector(
            config_path=config_file,
            robot_idx=robot_idx,
            human_idx=human_idx,
            num_workers=num_workers,
        )
        if not quiet:
            print(f"Parallel collection: {num_workers} workers")

    # --- Phase 1: Human Q-learning ---
    if not quiet:
        print(f"\n--- Phase 1: Human Q-learning ({num_phase1_iters} iterations) ---")

    q_h_net = QhNet.from_encoders(se, ge, num_actions=num_actions)
    p1 = Phase1Trainer(
        q_net=q_h_net,
        state_encoder=se,
        goal_encoder=ge,
        num_actions=num_actions,
        num_agents=num_agents,
        human_agent_idx=human_idx,
        gamma=gamma_h,
        beta_h=beta_h,
        lr=phase1_lr,
        buffer_capacity=phase1_buffer,
        batch_size=phase1_batch,
        target_update_freq=phase1_target_freq,
        exploration=phase1_exploration,
        epsilon_start=phase1_eps_start,
        epsilon_end=phase1_eps_end,
        epsilon_decay_steps=phase1_eps_decay,
        ucb_c=phase1_ucb_c,
        reward_shaping=phase1_reward_shaping,
        shaping_weight=phase1_shaping_weight,
        random_start_prob=phase1_random_start_prob,
        done_on_goal=phase1_done_on_goal,
        device=dev,
        parallel_collector=parallel_collector,
    )
    p1_history = p1.train(
        env, goals,
        num_iterations=num_phase1_iters,
        episodes_per_iter=phase1_episodes_per_iter,
        train_steps_per_iter=phase1_train_steps_per_iter,
        log_interval=log,
    )

    # --- Phase 2: V_h^e + Robot Q-learning ---
    if not quiet:
        print(f"\n--- Phase 2: Robot empowerment learning ({num_phase2_iters} iterations) ---")

    vhe_net = VheNet.from_encoders(se, ge)
    qr_net = QrNet.from_encoders(se, num_actions=num_actions)
    p2 = Phase2Trainer(
        vhe_net=vhe_net,
        q_r_net=qr_net,
        phase1_trainer=p1,
        state_encoder=se,
        goal_encoder=ge,
        goals=goals,
        num_actions=num_actions,
        num_agents=num_agents,
        robot_agent_idx=robot_idx,
        human_agent_idx=human_idx,
        gamma_h=gamma_h,
        gamma_r=gamma_r,
        lr_vhe=phase2_lr_vhe,
        lr_qr=phase2_lr_qr,
        buffer_capacity=phase2_buffer,
        batch_size=phase2_batch,
        target_update_freq=phase2_target_freq,
        epsilon_start=phase2_eps_start,
        epsilon_end=phase2_eps_end,
        epsilon_decay_steps=phase2_eps_decay,
        done_on_goal=phase2_done_on_goal,
        random_start_prob=phase2_random_start_prob,
        rock_curiosity_bonus=phase2_rock_curiosity,
        rock_approach_weight=phase2_rock_approach,
        device=dev,
        parallel_collector=parallel_collector,
    )
    p2_history = p2.train(
        env,
        num_iterations=num_phase2_iters,
        episodes_per_iter=phase2_episodes_per_iter,
        train_steps_per_iter=phase2_train_steps_per_iter,
        vhe_warmup_iters=phase2_vhe_warmup,
        rock_seek_during_warmup=phase2_rock_seek_warmup,
        log_interval=log,
    )

    if parallel_collector is not None:
        parallel_collector.close()

    if not quiet:
        print("\n--- Training complete ---")

    return EMPOResult(
        env=env,
        phase1_trainer=p1,
        phase2_trainer=p2,
        state_encoder=se,
        goal_encoder=ge,
        goals=goals,
        robot_agent_idx=robot_idx,
        human_agent_idx=human_idx,
        phase1_history=p1_history,
        phase2_history=p2_history,
    )

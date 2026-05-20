#!/usr/bin/env python3
"""
Phase 2 open-milestone evaluation: trajectory-target modes × pi_r modes.

Compares:
- target modes: one_step, n_step, episode
- pi_r_mode: direct, mcts

For each run, reports:
- wall-clock throughput metrics,
- search-cost metrics,
- stability metrics from recent training losses.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import statistics
import sys
import time
from datetime import datetime, timezone
from typing import Dict, List, TypeAlias

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.join(_SCRIPT_DIR, os.pardir, os.pardir)
for _subdir in ("src", "vendor/multigrid", "vendor/ai_transport", "multigrid_worlds"):
    _path = os.path.join(_PROJECT_ROOT, _subdir)
    if _path not in sys.path:
        sys.path.insert(0, _path)

import numpy as np
import torch

from gym_multigrid.multigrid import MultiGridEnv, SmallActions, World

from empo.human_policy_prior import HeuristicPotentialPolicy
from empo.learning_based.multigrid import PathDistanceCalculator
from empo.learning_based.multigrid.phase2 import train_multigrid_phase2
from empo.learning_based.phase2.config import Phase2Config
from empo.world_specific_helpers.multigrid import ReachCellGoal
from empo.possible_goal import TabularGoalSampler

GRID_MAP = """
We We We We We We
We Ae Ro .. .. We
We We Ay We We We
We We We We We We
"""

RunMetricValue: TypeAlias = float | str
RunMetricRow: TypeAlias = Dict[str, RunMetricValue]
JsonMetricValue: TypeAlias = float | str | None


def _set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _create_world_model(max_steps: int) -> MultiGridEnv:
    return MultiGridEnv(
        map=GRID_MAP,
        max_steps=max_steps,
        partial_obs=False,
        objects_set=World,
        actions_set=SmallActions,
    )


def _create_goal_sampler(env: MultiGridEnv) -> TabularGoalSampler:
    goal_specs = [
        (1, (2, 1)),
        (1, (1, 1)),
        (1, (3, 1)),
        (1, (2, 2)),
        (1, (1, 2)),
    ]
    goals = [ReachCellGoal(env, human_idx, pos) for human_idx, pos in goal_specs]
    return TabularGoalSampler(goals)


def _create_human_policy(env: MultiGridEnv, human_indices: List[int]) -> HeuristicPotentialPolicy:
    path_calc = PathDistanceCalculator(
        grid_height=env.height,
        grid_width=env.width,
        world_model=env,
    )
    return HeuristicPotentialPolicy(
        world_model=env,
        human_agent_indices=human_indices,
        path_calculator=path_calc,
        beta=1000.0,
    )


def _tail_stats(history: List[Dict[str, float]], key: str, tail_k: int) -> Dict[str, float]:
    values = [float(entry[key]) for entry in history if key in entry]
    if not values:
        return {
            f"{key}_tail_count": 0.0,
            f"{key}_tail_mean": float("nan"),
            f"{key}_tail_std": float("nan"),
            f"{key}_tail_cv": float("nan"),
        }
    tail = values[-tail_k:]
    mean_val = float(statistics.fmean(tail))
    std_val = float(statistics.pstdev(tail)) if len(tail) > 1 else 0.0
    cv = std_val / (abs(mean_val) + 1e-12)
    return {
        f"{key}_tail_count": float(len(tail)),
        f"{key}_tail_mean": mean_val,
        f"{key}_tail_std": std_val,
        f"{key}_tail_cv": float(cv),
    }


def _search_policy_entropy(policy: List[float]) -> float:
    entropy = 0.0
    for prob in policy:
        p = max(float(prob), 1e-12)
        entropy -= p * math.log(p)
    return float(entropy)


def run_single_experiment(
    target_mode: str,
    pi_r_mode: str,
    seed: int,
    steps: int,
    n_step: int,
    mcts_num_simulations: int,
    mcts_max_depth: int,
    mcts_c_puct: float,
) -> RunMetricRow:
    _set_global_seed(seed)
    env = _create_world_model(max_steps=10)
    env.reset()

    human_indices = [i for i, a in enumerate(env.agents) if a.color == "yellow"]
    robot_indices = [i for i, a in enumerate(env.agents) if a.color == "grey"]
    goal_sampler = _create_goal_sampler(env)
    human_policy = _create_human_policy(env, human_indices)

    config = Phase2Config(
        gamma_r=0.99,
        gamma_h=0.99,
        gamma_h_curriculum=True,
        gamma_h_start=0.0,
        gamma_h_rampup_steps=max(1, int(0.10 * steps)),
        gamma_r_curriculum=True,
        gamma_r_start=0.0,
        gamma_r_rampup_steps=max(1, int(0.10 * steps)),
        zeta=2.0,
        xi=1.0,
        eta=1.1,
        beta_r=1000.0,
        epsilon_r_start=1.0,
        epsilon_r_end=0.0,
        epsilon_r_decay_steps=max(10, steps // 2),
        epsilon_h_start=1.0,
        epsilon_h_end=0.0,
        epsilon_h_decay_steps=max(10, steps // 2),
        lr_q_r=1e-4,
        lr_v_r=1e-4,
        lr_v_h_e=1e-3,
        lr_x_h=1e-4,
        lr_u_r=1e-4,
        buffer_size=20_000,
        batch_size=32,
        x_h_batch_size=64,
        num_training_steps=steps,
        steps_per_episode=env.max_steps,
        training_steps_per_env_step=0.1,
        goal_resample_prob=0.1,
        x_h_use_network=False,
        warmup_v_h_e_steps=max(1, int(0.10 * steps)),
        warmup_x_h_steps=max(1, int(0.10 * steps)),
        warmup_u_r_steps=max(1, int(0.05 * steps)),
        warmup_q_r_steps=max(1, int(0.10 * steps)),
        warmup_v_r_steps=max(1, int(0.05 * steps)),
        beta_r_rampup_steps=max(1, int(0.20 * steps)),
        use_encoders=True,
        use_lookup_tables=False,
        use_rnd=False,
        use_human_action_rnd=False,
        use_count_based_curiosity=False,
        use_z_space_transform=True,
        use_z_based_loss=False,
        v_h_e_target_mode=target_mode,
        q_r_target_mode=target_mode,
        v_h_e_n_step=n_step,
        q_r_n_step=n_step,
        pi_r_mode=pi_r_mode,
        mcts_num_simulations=(mcts_num_simulations if pi_r_mode == "mcts" else 0),
        mcts_max_depth=mcts_max_depth,
        mcts_c_puct=mcts_c_puct,
        mcts_temperature=1.0,
        mcts_root_noise_frac=0.0,
        mcts_enable_after_training_step=0,
    )

    wall_start = time.perf_counter()
    tensorboard_dir = os.path.join(
        _PROJECT_ROOT,
        "outputs",
        "phase2_open_eval_tensorboard",
        f"{target_mode}_{pi_r_mode}_seed{seed}",
    )
    os.makedirs(tensorboard_dir, exist_ok=True)

    _, _, history, trainer = train_multigrid_phase2(
        world_model=env,
        human_agent_indices=human_indices,
        robot_agent_indices=robot_indices,
        human_policy_prior=human_policy,
        goal_sampler=goal_sampler,
        config=config,
        hidden_dim=32,
        goal_feature_dim=16,
        agent_embedding_dim=8,
        device="cpu",
        verbose=False,
        debug=False,
        tensorboard_dir=tensorboard_dir,
        restore_networks_path=None,
        world_model_factory=None,
        robot_exploration_policy=None,
        human_exploration_policy=None,
    )
    wall_seconds = float(time.perf_counter() - wall_start)

    replay_transitions = list(trainer.replay_buffer.buffer)
    transition_count = len(replay_transitions)
    search_policies = [
        list(transition.search_policy)
        for transition in replay_transitions
        if transition.search_policy is not None
    ]
    searched_transition_count = len(search_policies)

    if search_policies:
        mean_search_entropy = float(
            statistics.fmean(_search_policy_entropy(policy) for policy in search_policies)
        )
    else:
        mean_search_entropy = float("nan")

    approx_search_simulations = (
        searched_transition_count * config.mcts_num_simulations
    )

    run_metrics: RunMetricRow = {
        "target_mode": target_mode,
        "pi_r_mode": pi_r_mode,
        "seed": float(seed),
        "num_training_steps": float(trainer.training_step_count),
        "total_env_steps": float(trainer.total_env_steps),
        "history_points": float(len(history)),
        "wall_clock_seconds": wall_seconds,
        "training_steps_per_second": float(
            trainer.training_step_count / max(wall_seconds, 1e-12)
        ),
        "env_steps_per_second": float(trainer.total_env_steps / max(wall_seconds, 1e-12)),
        "replay_transitions": float(transition_count),
        "searched_transitions": float(searched_transition_count),
        "searched_transition_rate": float(
            searched_transition_count / max(transition_count, 1)
        ),
        "mean_search_policy_entropy": mean_search_entropy,
        "approx_search_simulations": float(approx_search_simulations),
        "approx_search_simulations_per_second": float(
            approx_search_simulations / max(wall_seconds, 1e-12)
        ),
        "approx_search_simulations_per_training_step": float(
            approx_search_simulations / max(trainer.training_step_count, 1)
        ),
    }
    run_metrics.update(_tail_stats(history, "q_r", tail_k=5))
    run_metrics.update(_tail_stats(history, "v_h_e", tail_k=5))
    return run_metrics


def _float_for_json(value: RunMetricValue) -> JsonMetricValue:
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 2 open evaluation: target horizon modes × pi_r modes"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick run (small budget) for fast reproducibility checks.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=1000,
        help="Training steps per experiment run (default: 1000).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed (default: 42).",
    )
    parser.add_argument(
        "--n-step",
        type=int,
        default=5,
        help="n-step horizon used when target_mode=n_step (default: 5).",
    )
    parser.add_argument(
        "--mcts-num-simulations",
        type=int,
        default=32,
        help="MCTS simulations per acted state for pi_r_mode=mcts (default: 32).",
    )
    parser.add_argument(
        "--mcts-max-depth",
        type=int,
        default=6,
        help="MCTS max depth for pi_r_mode=mcts (default: 6).",
    )
    parser.add_argument(
        "--mcts-c-puct",
        type=float,
        default=1.5,
        help="MCTS c_puct (default: 1.5).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for JSON/CSV summary outputs.",
    )
    args = parser.parse_args()

    steps = 200 if args.quick else args.steps
    output_dir = args.output_dir or os.path.join(
        _PROJECT_ROOT, "outputs", "phase2_open_eval"
    )
    os.makedirs(output_dir, exist_ok=True)

    target_modes = ["one_step", "n_step", "episode"]
    pi_r_modes = ["direct", "mcts"]
    rows: List[RunMetricRow] = []

    print("=" * 78)
    print("Phase 2 open evaluation")
    print(f"  steps/run={steps}, n_step={args.n_step}")
    print(f"  mcts_num_simulations={args.mcts_num_simulations}")
    print("=" * 78)

    run_idx = 0
    for target_mode in target_modes:
        for pi_r_mode in pi_r_modes:
            run_seed = args.seed + run_idx
            print(
                f"[run {run_idx + 1}/6] target_mode={target_mode}, "
                f"pi_r_mode={pi_r_mode}, seed={run_seed}"
            )
            row = run_single_experiment(
                target_mode=target_mode,
                pi_r_mode=pi_r_mode,
                seed=run_seed,
                steps=steps,
                n_step=args.n_step,
                mcts_num_simulations=args.mcts_num_simulations,
                mcts_max_depth=args.mcts_max_depth,
                mcts_c_puct=args.mcts_c_puct,
            )
            rows.append(row)
            run_idx += 1

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = os.path.join(output_dir, f"open_eval_summary_{timestamp}.json")
    csv_path = os.path.join(output_dir, f"open_eval_summary_{timestamp}.csv")

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "steps_per_run": steps,
        "n_step": args.n_step,
        "mcts_num_simulations": args.mcts_num_simulations,
        "mcts_max_depth": args.mcts_max_depth,
        "mcts_c_puct": args.mcts_c_puct,
        "results": [
            {k: _float_for_json(v) for k, v in row.items()}
            for row in rows
        ],
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)

    csv_fields = [
        "target_mode",
        "pi_r_mode",
        "seed",
        "num_training_steps",
        "total_env_steps",
        "history_points",
        "wall_clock_seconds",
        "training_steps_per_second",
        "env_steps_per_second",
        "replay_transitions",
        "searched_transitions",
        "searched_transition_rate",
        "mean_search_policy_entropy",
        "approx_search_simulations",
        "approx_search_simulations_per_second",
        "approx_search_simulations_per_training_step",
        "q_r_tail_count",
        "q_r_tail_mean",
        "q_r_tail_std",
        "q_r_tail_cv",
        "v_h_e_tail_count",
        "v_h_e_tail_mean",
        "v_h_e_tail_std",
        "v_h_e_tail_cv",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in csv_fields})

    print("\nSummary:")
    for row in rows:
        target_mode_label = str(row["target_mode"])
        pi_r_mode_label = str(row["pi_r_mode"])
        wall_seconds = float(row["wall_clock_seconds"])
        searched_transition_rate = float(row["searched_transition_rate"])
        q_r_tail_std = float(row["q_r_tail_std"])
        v_h_e_tail_std = float(row["v_h_e_tail_std"])
        print(
            f"  {target_mode_label:>8} | {pi_r_mode_label:>6} | "
            f"wall={wall_seconds:.2f}s | "
            f"search_rate={searched_transition_rate:.3f} | "
            f"q_r_std={q_r_tail_std if not math.isnan(q_r_tail_std) else float('nan'):.4f} | "
            f"v_h_e_std={v_h_e_tail_std if not math.isnan(v_h_e_tail_std) else float('nan'):.4f}"
        )

    print(f"\nSaved JSON: {json_path}")
    print(f"Saved CSV : {csv_path}")


if __name__ == "__main__":
    main()

"""
Phase 2 Hyperparameter Optimisation (HPO) example using Optuna.

Tunes *training* hyperparameters (learning rates, warm-up steps, batch size, …)
for the Phase 2 robot-policy training loop.  Theory parameters (beta_r, gamma_h,
gamma_r, zeta, xi, eta) are intentionally **fixed**

Usage (outside Docker):
    PYTHONPATH=src:vendor/multigrid:vendor/ai_transport:multigrid_worlds \\
        python examples/phase2/hpo_phase2_example.py --trials 20 --training-steps 200

Usage (inside Docker / make shell):
    python examples/phase2/hpo_phase2_example.py --trials 20 --training-steps 200
"""

import argparse
import gc
import os
import random
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import optuna
except ImportError as exc:
    raise ImportError(
        "The Phase 2 HPO example requires the 'optuna' package. "
        "Install it with:\n\n    pip install optuna\n"
    ) from exc

# ---------------------------------------------------------------------------
# Path setup – mirrors the PYTHONPATH used by the project's examples
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for _p in ("src", "vendor/multigrid", "vendor/ai_transport", "multigrid_worlds"):
    sys.path.insert(0, os.path.join(_REPO_ROOT, _p))

# ---------------------------------------------------------------------------
# Project imports (after path setup)
# ---------------------------------------------------------------------------
from gym_multigrid.multigrid import MultiGridEnv, World, SmallActions  # noqa: E402

from empo.human_policy_prior import HeuristicPotentialPolicy              # noqa: E402
from empo.learning_based.multigrid import PathDistanceCalculator          # noqa: E402
from empo.learning_based.multigrid.phase2 import train_multigrid_phase2   # noqa: E402
from empo.learning_based.phase2.config import Phase2Config                # noqa: E402
from empo.possible_goal import TabularGoalSampler                         # noqa: E402
from empo.world_specific_helpers.multigrid import ReachCellGoal           # noqa: E402


## CUDA
device = "cuda" if (optuna.integration.pytorch.is_available() and torch.cuda.is_available()) else "cpu"

# ---------------------------------------------------------------------------
# Default grid map – tiny 4×6 "trivial" world (same as demo's trivial mode)
# ---------------------------------------------------------------------------
TRIVIAL_GRID_MAP = """
We We We We We We
We Ae Ro .. .. We
We We Ay We We We
We We We We We We
"""
TRIVIAL_MAX_STEPS = 10

# Fixed theory parameters (not tuned – see copilot-instructions.md)
BETA_R_FINAL = 1000.0
GAMMA_H = 0.99
GAMMA_R = 0.99
ZETA = 2.0
XI = 1.0
ETA = 1.1


# ---------------------------------------------------------------------------
# Data class for HPO settings
# ---------------------------------------------------------------------------
@dataclass
class HPOSettings:
    trials: int
    total_training_steps: int
    chunk_size: int
    grid_map: str
    max_steps: int
    study_name: str
    storage: Optional[str]
    n_startup_trials: int
    n_warmup_reports: int
    seed: int


# ---------------------------------------------------------------------------
# Environment + prior + goal-sampler helpers
# ---------------------------------------------------------------------------

def _make_env(grid_map: str, max_steps: int) -> MultiGridEnv:
    """Create a fresh MultiGridEnv from a grid-map string."""
    return MultiGridEnv(
        map=grid_map,
        max_steps=max_steps,
        partial_obs=False,
        objects_set=World,
        actions_set=SmallActions,
    )


def _detect_agents(env: MultiGridEnv) -> Tuple[List[int], List[int]]:
    """Return (human_indices, robot_indices) by agent colour convention."""
    human_indices = [i for i, a in enumerate(env.agents) if a.color == "yellow"]
    robot_indices = [i for i, a in enumerate(env.agents) if a.color == "grey"]
    return human_indices, robot_indices


def _make_goal_sampler(env: MultiGridEnv, human_indices: List[int]) -> TabularGoalSampler:
    """
    Build a uniform TabularGoalSampler over all empty interior cells,
    one ReachCellGoal per human agent.
    """
    walkable: List[Tuple[int, int]] = []
    for x in range(1, env.width - 1):
        for y in range(1, env.height - 1):
            cell = env.grid.get(x, y)
            if cell is None:
                walkable.append((x, y))

    if not walkable:
        walkable = [(1, 1)]

    goals = [
        ReachCellGoal(env, h_idx, pos)
        for h_idx in human_indices
        for pos in walkable
    ]
    return TabularGoalSampler(goals)


def _make_human_policy(
    env: MultiGridEnv,
    human_indices: List[int],
) -> HeuristicPotentialPolicy:
    """Build a (fairly deterministic) heuristic human policy prior."""
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


# ---------------------------------------------------------------------------
# Objective helpers
# ---------------------------------------------------------------------------

def _aggregate_objective(history: List[Dict[str, float]], window: int = 20) -> float:
    """
    Compute HPO objective from trainer loss history (lower = better).

    Centers on v_h_e (human goal-achievement approximation quality).
    q_r values are also checked for numerical stability; if any are NaN/inf
    the trial is penalised.
    """
    if not history:
        return float("inf")

    tail = history[-window:]
    v_h_e_vals = [h["v_h_e"] for h in tail if "v_h_e" in h and np.isfinite(h["v_h_e"])]
    q_r_vals = [h["q_r"] for h in tail if "q_r" in h and np.isfinite(h["q_r"])]

    if not v_h_e_vals:
        return float("inf")

    v_h_e_mean = float(np.mean(v_h_e_vals))
    q_r_mean = float(np.mean(q_r_vals)) if q_r_vals else 0.0

    # q_r values are large early in training (beta_r=1000 scales them to 10k+);
    # only flag actual numerical explosions, not normal large-but-finite values.
    score = v_h_e_mean

    if np.isnan(score) or np.isinf(score) or np.isnan(q_r_mean) or np.isinf(q_r_mean):
        return float("inf")

    return score


# ---------------------------------------------------------------------------
# Optuna search space → Phase2Config
# ---------------------------------------------------------------------------

def _suggest_config(trial: optuna.Trial, total_training_steps: int) -> Phase2Config:
    """
    Map Optuna suggestions to a Phase2Config.

    Only *training* hyperparameters are tuned here.  Theory parameters
    (beta_r, gamma_h, gamma_r, zeta, xi, eta) are kept at fixed values.

    Hyperparameters are grouped into four categories:
      1. Learning rates          – how fast networks update
      2. Warm-up / ramp-up steps – how long each network trains before full use
      3. Network update schedule – target-network refresh and LR decay shape
      4. Training efficiency     – replay ratio and replay-buffer batch size
    """
    # ------------------------------------------------------------------ #
    # Category 1: Learning rates                                          #
    # ------------------------------------------------------------------ #
    lr_v_h_e = trial.suggest_float("lr_v_h_e", 1e-4, 1e-2, log=True)
    lr_q_r = trial.suggest_float("lr_q_r", 1e-5, 1e-3, log=True)

    # ------------------------------------------------------------------ #
    # Category 2: Warm-up / ramp-up steps                                #
    # Cap at 20 % of budget so the main training phase still runs.       #
    # ------------------------------------------------------------------ #
    max_warmup = max(50, (total_training_steps // 5 // 50) * 50)
    warmup_v_h_e_steps = trial.suggest_int("warmup_v_h_e_steps", 50, max_warmup, step=50)
    warmup_q_r_steps = trial.suggest_int("warmup_q_r_steps", 50, max_warmup, step=50)
    max_rampup = max(100, (total_training_steps // 3 // 100) * 100)
    beta_r_rampup_steps = trial.suggest_int("beta_r_rampup_steps", 100, max_rampup, step=100)

    # ------------------------------------------------------------------ #
    # Category 3: Network update schedule                                 #
    # ------------------------------------------------------------------ #
    target_update_interval = trial.suggest_int("target_update_interval", 10, 200, step=10)
    lr_constant_fraction = trial.suggest_float("lr_constant_fraction", 0.4, 0.95)

    # ------------------------------------------------------------------ #
    # Category 4: Training efficiency                                     #
    # ------------------------------------------------------------------ #
    # training_steps_per_env_step ≥ 0.5 means at least one env step per  #
    # two training steps, keeping the replay buffer well-populated.       #
    training_steps_per_env_step = trial.suggest_float("training_steps_per_env_step", 0.5, 4.0)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])

    epsilon_decay_steps = int(total_training_steps * 2 / 3)

    return Phase2Config(
        # ---- Fixed theory parameters ----
        beta_r=BETA_R_FINAL,
        gamma_h=GAMMA_H,
        gamma_r=GAMMA_R,
        zeta=ZETA,
        xi=XI,
        eta=ETA,
        # ---- Category 1: Learning rates ----
        lr_v_h_e=lr_v_h_e,
        lr_q_r=lr_q_r,
        lr_x_h=lr_v_h_e,
        lr_u_r=lr_q_r,
        lr_v_r=lr_q_r,
        # ---- Category 2: Warm-up / ramp-up steps ----
        warmup_v_h_e_steps=warmup_v_h_e_steps,
        warmup_x_h_steps=warmup_v_h_e_steps,
        warmup_q_r_steps=warmup_q_r_steps,
        beta_r_rampup_steps=beta_r_rampup_steps,
        # ---- Category 3: Network update schedule ----
        v_h_e_target_update_interval=target_update_interval,
        q_r_target_update_interval=target_update_interval,
        x_h_target_update_interval=target_update_interval,
        lr_constant_fraction=lr_constant_fraction,
        constant_lr_then_1_over_t=True,
        # ---- Category 4: Training efficiency ----
        training_steps_per_env_step=training_steps_per_env_step,
        batch_size=batch_size,
        buffer_size=2_000,
        # ---- Exploration (derived from total steps, not tuned) ----
        epsilon_r_start=1.0,
        epsilon_r_end=0.0,
        epsilon_r_decay_steps=epsilon_decay_steps,
        epsilon_h_start=1.0,
        epsilon_h_end=0.0,
        epsilon_h_decay_steps=epsilon_decay_steps,
        num_training_steps=total_training_steps,
        steps_per_episode=TRIVIAL_MAX_STEPS,
        # ---- Network configuration ----
        x_h_use_network=True,
        u_r_use_network=False,
        v_r_use_network=False,
        use_encoders=True,
        use_lookup_tables=False,
        # ---- Disable extras that would slow HPO ----
        use_rnd=False,
        async_training=False,
        checkpoint_interval=0,
    )


# ---------------------------------------------------------------------------
# Optuna objective
# ---------------------------------------------------------------------------

def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        print("Warning: PyTorch not available, skipping torch seeding.")


def make_objective(settings: HPOSettings):
    def objective(trial: optuna.Trial) -> float:
        _seed_everything(settings.seed + trial.number)

        env: Optional[MultiGridEnv] = None
        history_accum: List[Dict[str, float]] = []

        try:
            env = _make_env(settings.grid_map, settings.max_steps)
            env.reset()

            human_indices, robot_indices = _detect_agents(env)
            if not human_indices or not robot_indices:
                trial.set_user_attr("error", "No human or robot agents found in grid map")
                return float("inf")

            goal_sampler = _make_goal_sampler(env, human_indices)
            human_policy = _make_human_policy(env, human_indices)
            config = _suggest_config(trial, settings.total_training_steps)

            # train_multigrid_phase2 creates networks internally and returns
            # (q_r_network, all_networks, history, trainer).
            # First chunk: bootstrap trainer via train_multigrid_phase2.
            # Clamp to total_training_steps so we never exceed the budget.
            first_chunk_size = min(settings.chunk_size, settings.total_training_steps)
            _q_r, _networks, first_history, trainer = train_multigrid_phase2(
                world_model=env,
                human_agent_indices=human_indices,
                robot_agent_indices=robot_indices,
                human_policy_prior=human_policy,
                goal_sampler=goal_sampler,
                config=config,
                num_training_steps=first_chunk_size,
                device="cpu",
                verbose=False,
            )
            if first_history:
                history_accum.extend(first_history)

            partial_score = _aggregate_objective(history_accum, window=10)
            trial.report(partial_score, step=first_chunk_size)
            if trial.should_prune():
                raise optuna.TrialPruned()

            # Subsequent chunks up to (but not including) total_training_steps.
            target = settings.chunk_size * 2
            while target < settings.total_training_steps:
                chunk_history = trainer.train(num_training_steps=target)
                if chunk_history:
                    history_accum.extend(chunk_history)

                partial_score = _aggregate_objective(history_accum, window=10)
                trial.report(partial_score, step=target)
                if trial.should_prune():
                    raise optuna.TrialPruned()

                target += settings.chunk_size

            # Final chunk: always train up to the exact total requested so
            # the --training-steps CLI flag is respected even when total is
            # not an exact multiple of chunk_size.
            if first_chunk_size < settings.total_training_steps:
                chunk_history = trainer.train(
                    num_training_steps=settings.total_training_steps
                )
                if chunk_history:
                    history_accum.extend(chunk_history)

                partial_score = _aggregate_objective(history_accum, window=10)
                trial.report(partial_score, step=settings.total_training_steps)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            return _aggregate_objective(history_accum, window=20)

        except optuna.TrialPruned:
            raise
        except Exception as exc:
            trial.set_user_attr("error", str(exc))
            return float("inf")
        finally:
            try:
                if env is not None:
                    env.close()
            except Exception:
                pass
            gc.collect()

    return objective


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> HPOSettings:
    parser = argparse.ArgumentParser(description="Phase 2 HPO example (Optuna)")
    parser.add_argument("--trials", type=int, default=20,
                        help="Number of Optuna trials (default: 20)")
    parser.add_argument("--training-steps", type=int, default=3000,
                        help="Total training_steps per trial (default: 3000)")
    parser.add_argument("--chunk-size", type=int, default=500,
                        help="training_step chunk size for pruning/reporting (default: 500)")
    parser.add_argument("--study-name", type=str, default="phase2_hpo",
                        help="Optuna study name (default: phase2_hpo)")
    parser.add_argument("--storage", type=str, default=None,
                        help="Optuna storage URI, e.g. sqlite:///hpo.db (optional)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Base random seed for sampler and per-trial seeding (default: 42)")
    parser.add_argument("--startup-trials", type=int, default=5,
                        help="Median pruner: startup trials before pruning (default: 5)")
    parser.add_argument("--warmup-reports", type=int, default=1,
                        help="Median pruner: warmup reporting steps before pruning (default: 1)")
    args = parser.parse_args()

    return HPOSettings(
        trials=args.trials,
        total_training_steps=args.training_steps,
        chunk_size=args.chunk_size,
        grid_map=TRIVIAL_GRID_MAP,
        max_steps=TRIVIAL_MAX_STEPS,
        study_name=args.study_name,
        storage=args.storage,
        n_startup_trials=args.startup_trials,
        n_warmup_reports=args.warmup_reports,
        seed=args.seed,
    )


def main() -> None:
    settings = _parse_args()

    print("=== Phase 2 HPO ===")
    print(f"  Trials:          {settings.trials}")
    print(f"  Training steps:  {settings.total_training_steps}")
    print(f"  Chunk size:      {settings.chunk_size}")
    print(f"  Study:           {settings.study_name}")
    print(f"  Storage:         {settings.storage or '(in-memory)'}")
    print()

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    sampler = optuna.samplers.TPESampler(seed=settings.seed, multivariate=True)
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=settings.n_startup_trials,
        n_warmup_steps=settings.n_warmup_reports,
    )

    study = optuna.create_study(
        study_name=settings.study_name,
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        storage=settings.storage,
        load_if_exists=True,
    )

    objective = make_objective(settings)
    study.optimize(objective, n_trials=settings.trials, show_progress_bar=True)

    print("\n=== HPO Complete ===")
    if study.best_trial is not None:
        print(f"Best objective: {study.best_value:.6f}")
        print("Best hyperparameters:")
        for key, value in study.best_trial.params.items():
            print(f"  {key}: {value}")
    else:
        print("No successful trials completed.")


if __name__ == "__main__":
    main()

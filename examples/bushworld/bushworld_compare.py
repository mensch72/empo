#!/usr/bin/env python3
"""
BushWorld Phase 2 comparison: backward induction vs. a learned robot policy.

This example computes a robot policy for a BushWorld world two ways and compares
them:

1. **Backward induction** (exact tabular Phase 2 over the reachable-state DAG),
   via :func:`empo.backward_induction.phase2.compute_robot_policy`.
2. **Learning-based Phase 2** (``method='tabular'`` fitted value iteration, or
   ``method='dqn'`` / ``'alphazero'`` neural fitted value iteration), via
   :func:`empo.bushworld.learning.train_bushworld_phase2`.

It demonstrates the full lifecycle requested for a new world model:

* **Checkpointing & recovery** — the learner saves a checkpoint and, on a second
  run (or after interruption), resumes from it (``--checkpoint-dir`` / ``--resume``).
* **Final-policy persistence** — the learned policy is saved to disk and reloaded
  to verify round-tripping.
* **Optional extra rollouts** — with ``--extra-rollouts N`` the script loads the
  *saved* final policy and runs additional rollouts from it (e.g. to keep
  generating trajectories without recomputing the policy).
* **Movies** — rollouts under each policy are rendered to annotated movies (with
  per-human goal overlays and a side text panel).

Usage::

    # Inside the dev container (PYTHONPATH already set):
    python examples/bushworld/bushworld_compare.py --quick

    # Outside the container:
    PYTHONPATH=src:vendor/multigrid:vendor/ai_transport:multigrid_worlds \
        python examples/bushworld/bushworld_compare.py \
        --world bushworld_worlds/two_humans_one_robot.yaml --method tabular --rollouts 4

    # Neural learner with checkpoint recovery:
    python examples/bushworld/bushworld_compare.py --method dqn --neural-iterations 600

    # Re-run to resume training from the checkpoint, then run extra rollouts from
    # the saved final policy without recomputing it:
    python examples/bushworld/bushworld_compare.py --method dqn --extra-rollouts 3
"""

import argparse
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

# Allow running without setting PYTHONPATH explicitly.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
for _p in ("src", "vendor/multigrid", "vendor/ai_transport", "multigrid_worlds"):
    _full = os.path.join(_REPO_ROOT, _p)
    if _full not in sys.path:
        sys.path.insert(0, _full)

from empo.backward_induction.phase2 import compute_robot_policy  # noqa: E402
from empo.bushworld import (  # noqa: E402
    ShortestPathHumanPolicyPrior,
    load_bushworld,
)
from empo.bushworld.learning import (  # noqa: E402
    Phase2Params,
    enumerate_reachable_states,
    load_policy,
    save_policy,
    train_bushworld_phase2,
)


# --------------------------------------------------------------------------- #
# Rollouts and movies
# --------------------------------------------------------------------------- #
def sample_human_goals(env, goal_sampler, rng) -> Dict[int, object]:
    """Sample one goal per human (for visualization/overlays during a rollout)."""
    goals: Dict[int, object] = {}
    for h in env.human_agent_indices:
        goal, _prob = goal_sampler.sample(env.get_state(), h)
        goals[h] = goal
    return goals


def rollout(
    env,
    robot_policy,
    human_policy_prior,
    *,
    goal_sampler,
    seed: int,
    render: bool = True,
    label: str = "",
) -> Tuple[List[np.ndarray], dict]:
    """Run one episode under ``robot_policy`` and sampled human behavior.

    Returns ``(frames, metrics)``. ``metrics`` reports the total bush density the
    robot cleared and the total human travel distance (a crude empowerment proxy:
    a helpful robot clears bushes so humans can move more freely).
    """
    rng = np.random.default_rng(seed)
    env.reset(seed=seed)
    state = env.initial_state()
    env.set_state(state)
    robot_policy.reset(env)

    human_goals = sample_human_goals(env, goal_sampler, rng)
    frames: List[np.ndarray] = []
    initial_density = sum(state[2])
    prev_positions = {h: state[1][h] for h in env.human_agent_indices}
    human_travel = 0

    num_actions = env.action_space.n
    step = 0
    while not env.is_terminal(state):
        robot_profile = robot_policy.sample(state)
        # Humans act under their (goal-conditioned) prior toward their sampled goal.
        human_actions: List[int] = []
        for h in env.human_agent_indices:
            dist = human_policy_prior(state, h, human_goals[h])
            human_actions.append(int(rng.choice(num_actions, p=dist)))
        actions = list(robot_profile) + human_actions

        if render:
            annotation = [
                f"{label}",
                f"step {step}/{env.max_steps}",
                f"robot a={list(robot_profile)}",
                f"humans a={human_actions}",
            ]
            frames.append(
                env.render(annotation_text=annotation, goal_overlays=human_goals)
            )

        env.set_state(state)
        env.step(actions)
        state = env.get_state()
        for h in env.human_agent_indices:
            x0, y0 = prev_positions[h]
            x1, y1 = state[1][h]
            human_travel += abs(x1 - x0) + abs(y1 - y0)
            prev_positions[h] = (x1, y1)
        step += 1

    if render:
        annotation = [f"{label}", f"step {step}/{env.max_steps} (terminal)"]
        frames.append(env.render(annotation_text=annotation, goal_overlays=human_goals))

    final_density = sum(state[2])
    metrics = {
        "bush_cleared": int(initial_density - final_density),
        "human_travel": int(human_travel),
    }
    return frames, metrics


def make_movie(frames: List[np.ndarray], path: str, fps: int = 2) -> Optional[str]:
    """Save ``frames`` to a movie file, returning the path or None on failure."""
    if not frames:
        return None
    try:
        from empo.bushworld.rendering import save_movie

        return save_movie(frames, path, fps=fps)
    except Exception as exc:  # pragma: no cover - depends on imageio/codecs
        print(f"  [warn] could not write movie {path}: {exc}")
        return None


def run_rollouts(
    env, policy, human_policy_prior, goal_sampler, *, n: int, base_seed: int,
    output_dir: str, tag: str, make_movies: bool, movie_ext: str = "gif",
) -> dict:
    """Run ``n`` rollouts under ``policy`` and (optionally) save movies."""
    cleared, travel = [], []
    for i in range(n):
        frames, metrics = rollout(
            env, policy, human_policy_prior, goal_sampler=goal_sampler,
            seed=base_seed + i, render=make_movies, label=tag,
        )
        cleared.append(metrics["bush_cleared"])
        travel.append(metrics["human_travel"])
        if make_movies:
            path = os.path.join(output_dir, f"{tag}_rollout_{i:02d}.{movie_ext}")
            saved = make_movie(frames, path)
            if saved:
                print(f"  saved movie: {saved}")
    summary = {
        "rollouts": n,
        "mean_bush_cleared": float(np.mean(cleared)) if cleared else 0.0,
        "mean_human_travel": float(np.mean(travel)) if travel else 0.0,
    }
    print(
        f"  [{tag}] mean bush cleared = {summary['mean_bush_cleared']:.2f}, "
        f"mean human travel = {summary['mean_human_travel']:.2f}"
    )
    return summary


# --------------------------------------------------------------------------- #
# Policy comparison
# --------------------------------------------------------------------------- #
def compare_policies(env, human_policy_prior, policy_a, policy_b) -> dict:
    """Compare two robot policies over reachable non-terminal states."""
    states = [
        s for s in enumerate_reachable_states(env, human_policy_prior)
        if not env.is_terminal(s)
    ]
    max_diff = 0.0
    argmax_agree = 0
    for s in states:
        da, db = policy_a(s), policy_b(s)
        for k in set(da) | set(db):
            max_diff = max(max_diff, abs(da.get(k, 0.0) - db.get(k, 0.0)))
        if da and db and max(da, key=da.get) == max(db, key=db.get):
            argmax_agree += 1
    return {
        "num_states": len(states),
        "max_prob_diff": max_diff,
        "argmax_agreement": argmax_agree / len(states) if states else 1.0,
    }


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def build_params(args) -> Phase2Params:
    return Phase2Params(
        beta_r=args.beta_r,
        gamma_h=args.gamma_h,
        gamma_r=args.gamma_r,
        zeta=args.zeta,
        xi=args.xi,
        eta=args.eta,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--world", default="bushworld_worlds/two_humans_one_robot.yaml",
                        help="Path to a BushWorld YAML world.")
    parser.add_argument("--method", default="tabular",
                        choices=["tabular", "value_iteration", "neural", "dqn", "alphazero"],
                        help="Learning version for the learned policy.")
    parser.add_argument("--output-dir", default="outputs/bushworld_compare",
                        help="Directory for movies, checkpoints, and the saved policy.")
    parser.add_argument("--checkpoint-dir", default=None,
                        help="Directory for training checkpoints (defaults to <output-dir>/checkpoints).")
    parser.add_argument("--no-resume", action="store_true",
                        help="Ignore any existing checkpoint and start training fresh.")
    parser.add_argument("--rollouts", type=int, default=3, help="Rollouts per policy.")
    parser.add_argument("--extra-rollouts", type=int, default=0,
                        help="Additional rollouts from the *saved* final learned policy.")
    parser.add_argument("--no-movie", action="store_true", help="Disable movie rendering.")
    parser.add_argument("--movie-format", default="gif", choices=["gif", "mp4"],
                        help="Movie container (gif needs no ffmpeg; mp4 needs imageio[ffmpeg]).")
    parser.add_argument("--seed", type=int, default=0)
    # Theory parameters.
    parser.add_argument("--beta-r", type=float, default=5.0)
    parser.add_argument("--gamma-h", type=float, default=0.95)
    parser.add_argument("--gamma-r", type=float, default=0.95)
    parser.add_argument("--zeta", type=float, default=1.0)
    parser.add_argument("--xi", type=float, default=1.0)
    parser.add_argument("--eta", type=float, default=1.0)
    # Learner controls.
    parser.add_argument("--neural-iterations", type=int, default=600,
                        help="Training iterations for the neural learner.")
    parser.add_argument("--max-iterations", type=int, default=1000,
                        help="Max sweeps for the tabular learner.")
    parser.add_argument("--quick", action="store_true",
                        help="Fast settings (fewer rollouts / iterations) for smoke testing.")
    args = parser.parse_args()

    if args.quick:
        args.rollouts = min(args.rollouts, 2)
        args.neural_iterations = min(args.neural_iterations, 150)

    os.makedirs(args.output_dir, exist_ok=True)
    checkpoint_dir = args.checkpoint_dir or os.path.join(args.output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    make_movies = not args.no_movie

    np.random.seed(args.seed)

    # --- Load world and human policy prior -------------------------------- #
    world_path = args.world
    if not os.path.isabs(world_path):
        world_path = os.path.join(_REPO_ROOT, world_path)
    print(f"Loading BushWorld from {world_path}")
    env = load_bushworld(world_path)
    print(f"  {env!r}")
    human_policy_prior = ShortestPathHumanPolicyPrior(env, env.human_agent_indices)
    goal_generator = env.possible_goal_generator
    goal_sampler = goal_generator.get_sampler()
    params = build_params(args)
    print(f"  Phase 2 params: {params}")

    # --- Backward induction (exact reference) ----------------------------- #
    print("\n=== Backward induction (exact Phase 2) ===")
    bi_policy = compute_robot_policy(
        env, list(env.human_agent_indices), list(env.robot_agent_indices),
        goal_generator, human_policy_prior,
        beta_r=params.beta_r, gamma_h=params.gamma_h, gamma_r=params.gamma_r,
        zeta=params.zeta, xi=params.xi, eta=params.eta,
        level_fct=lambda s: s[0], quiet=True,
    )
    print("  backward induction policy computed.")

    # --- Learned policy with checkpointing -------------------------------- #
    resolved = "neural" if args.method in ("neural", "dqn", "alphazero") else "tabular"
    ckpt_name = "neural.pt" if resolved == "neural" else "tabular.pkl"
    checkpoint_path = os.path.join(checkpoint_dir, ckpt_name)
    resume = not args.no_resume
    if resume and os.path.exists(checkpoint_path):
        print(f"\n=== Learned policy ({args.method}) — resuming from {checkpoint_path} ===")
    else:
        print(f"\n=== Learned policy ({args.method}) — training from scratch ===")

    learn_kwargs = dict(
        method=args.method,
        checkpoint_path=checkpoint_path,
        resume=resume,
        quiet=False,
    )
    if resolved == "neural":
        learn_kwargs.update(num_iterations=args.neural_iterations, seed=args.seed)
    else:
        learn_kwargs.update(max_iterations=args.max_iterations)

    learned_policy, history = train_bushworld_phase2(
        env, human_policy_prior, params, **learn_kwargs
    )
    print(f"  checkpoint saved to {checkpoint_path}")

    # --- Save final policy and reload to verify round-trip ---------------- #
    policy_path = os.path.join(args.output_dir, "final_policy" + (".pt" if resolved == "neural" else ".pkl"))
    save_policy(learned_policy, policy_path)
    print(f"  final policy saved to {policy_path}")
    reloaded_policy = load_policy(policy_path, env)
    print("  final policy reloaded from disk (verifying save/restore).")

    # --- Compare policies -------------------------------------------------- #
    print("\n=== Policy comparison (learned vs backward induction) ===")
    cmp = compare_policies(env, human_policy_prior, bi_policy, reloaded_policy)
    print(f"  states compared: {cmp['num_states']}")
    print(f"  max |pi_r diff|: {cmp['max_prob_diff']:.4e}")
    print(f"  argmax agreement: {cmp['argmax_agreement'] * 100:.1f}%")
    if resolved == "tabular":
        print("  (tabular learner converges to the backward-induction fixed point.)")
    else:
        print("  (neural learner is an approximation; expect some disagreement.)")

    # --- Rollouts and movies ---------------------------------------------- #
    print("\n=== Rollouts ===")
    bi_summary = run_rollouts(
        env, bi_policy, human_policy_prior, goal_sampler,
        n=args.rollouts, base_seed=args.seed, output_dir=args.output_dir,
        tag="backward_induction", make_movies=make_movies, movie_ext=args.movie_format,
    )
    learned_summary = run_rollouts(
        env, reloaded_policy, human_policy_prior, goal_sampler,
        n=args.rollouts, base_seed=args.seed + 1000, output_dir=args.output_dir,
        tag=f"learned_{args.method}", make_movies=make_movies, movie_ext=args.movie_format,
    )

    # --- Optional extra rollouts from the saved final policy --------------- #
    if args.extra_rollouts > 0:
        print(f"\n=== Extra rollouts from saved final policy ({args.extra_rollouts}) ===")
        extra_policy = load_policy(policy_path, env)
        run_rollouts(
            env, extra_policy, human_policy_prior, goal_sampler,
            n=args.extra_rollouts, base_seed=args.seed + 2000,
            output_dir=args.output_dir, tag=f"extra_{args.method}",
            make_movies=make_movies, movie_ext=args.movie_format,
        )

    print("\nDone.")
    print(f"  Outputs in: {args.output_dir}")
    print(f"  Backward induction rollouts: {bi_summary}")
    print(f"  Learned rollouts:            {learned_summary}")


if __name__ == "__main__":
    main()

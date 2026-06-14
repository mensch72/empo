#!/usr/bin/env python3
"""
BushWorld Phase 2 comparison: backward induction vs. a learned robot policy.

This example computes a robot policy for a BushWorld world two ways and compares
them:

1. **Backward induction** (exact tabular Phase 2 over the reachable-state DAG),
   via :func:`empo.backward_induction.phase2.compute_robot_policy`.
2. **Learning-based Phase 2** through the shared training infrastructure (the same
   ``BasePhase2Trainer`` used for MultiGrid), via
   :func:`empo.learning_based.bushworld.phase2.train_bushworld_phase2`
   — ``--method lookup`` (lookup-table fitted value iteration) or
   ``--method neural`` (neural networks).

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

    # Outside the container (single-map mode):
    PYTHONPATH=src:vendor/multigrid:vendor/ai_transport:multigrid_worlds \
        python examples/bushworld/bushworld_compare.py \
        --world bushworld_worlds/two_humans_one_robot.yaml --method lookup --rollouts 4

    # Random-map-ensemble mode: train across several worlds (pass multiple files
    # or a directory of worlds). The exact backward-induction reference is skipped
    # automatically; the learner draws a fresh world each episode.
    python examples/bushworld/bushworld_compare.py --method neural \
        --world bushworld_worlds/two_humans_7x3_ensemble

    # Neural learner with checkpoint recovery:
    python examples/bushworld/bushworld_compare.py --method neural --neural-iterations 600

    # Re-run to resume training from the checkpoint, then run extra rollouts from
    # the saved final policy without recomputing it:
    python examples/bushworld/bushworld_compare.py --method neural --extra-rollouts 3

Movies default to ``.mp4`` (easy to pause/scrub frame by frame); pass
``--movie-format gif`` for a dependency-free animated GIF instead.
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
from empo.learning_based.phase2.config import Phase2Config  # noqa: E402
from empo.learning_based.phase2.world_model_factory import (  # noqa: E402
    EnsembleWorldModelFactory,
)
from empo.learning_based.bushworld.phase2 import (  # noqa: E402
    BushWorldRobotPolicy,
    train_bushworld_phase2,
)


class _BushWorldEnsembleLoader:
    """Picklable callable that loads a random BushWorld from a fixed list of paths.

    Used as the factory function for
    :class:`~empo.learning_based.phase2.world_model_factory.EnsembleWorldModelFactory`
    so the Phase 2 trainer can draw a fresh world each episode in
    *random-map-ensemble* mode. It stores only the (absolute) world paths and a
    seed, so it remains picklable for async actor processes.
    """

    def __init__(self, world_paths, seed: int = 0):
        self._world_paths = [os.path.abspath(p) for p in world_paths]
        self._seed = int(seed)
        self._rng = np.random.default_rng(self._seed)

    def __call__(self):
        idx = int(self._rng.integers(len(self._world_paths)))
        return load_bushworld(self._world_paths[idx])

    def __getstate__(self):
        # Drop the (unpicklable-by-value, non-deterministic) RNG; rebuild on load.
        return {"_world_paths": self._world_paths, "_seed": self._seed}

    def __setstate__(self, state):
        self._world_paths = state["_world_paths"]
        self._seed = state["_seed"]
        self._rng = np.random.default_rng(self._seed)


def resolve_worlds(world_args, repo_root: str) -> List[str]:
    """Expand ``--world`` entries into an ordered list of absolute YAML paths.

    Each entry may be a single YAML file or a directory; directories are
    expanded to every ``*.yaml`` / ``*.yml`` file they contain (sorted). A
    single resulting path means *single-map* mode; multiple paths mean
    *random-map-ensemble* mode.
    """
    import glob as _glob

    candidates: List[str] = []
    for entry in world_args:
        full = entry if os.path.isabs(entry) else os.path.join(repo_root, entry)
        if os.path.isdir(full):
            for ext in ("*.yaml", "*.yml"):
                candidates.extend(sorted(_glob.glob(os.path.join(full, ext))))
        else:
            candidates.append(full)

    seen = set()
    resolved: List[str] = []
    for p in candidates:
        ap = os.path.abspath(p)
        if ap not in seen:
            seen.add(ap)
            resolved.append(ap)
    if not resolved:
        raise FileNotFoundError(
            f"No BushWorld YAML files found for --world {world_args!r}"
        )
    return resolved


def _reachable_states(env, max_profiles=64):
    """Breadth-first enumeration of reachable states from the initial state."""
    import itertools

    seen = set()
    frontier = [env.initial_state()]
    profiles = list(
        itertools.product(range(env.action_space.n), repeat=env.num_players)
    )[:max_profiles]
    while frontier:
        s = frontier.pop()
        if s in seen:
            continue
        seen.add(s)
        if env.is_terminal(s):
            continue
        for pr in profiles:
            for p, ns in env.transition_probabilities(s, list(pr)):
                if p > 0 and ns not in seen:
                    frontier.append(ns)
    return seen


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
    env_components_fn, policy, *, n: int, base_seed: int,
    output_dir: str, tag: str, make_movies: bool, movie_ext: str = "mp4",
) -> dict:
    """Run ``n`` rollouts under ``policy`` and (optionally) save movies.

    ``env_components_fn(seed)`` returns the ``(env, human_policy_prior,
    goal_sampler)`` triple to use for that rollout. In single-map mode it always
    returns the same world; in random-map-ensemble mode it draws a fresh world
    each call.
    """
    cleared, travel = [], []
    for i in range(n):
        env, human_policy_prior, goal_sampler = env_components_fn(base_seed + i)
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
def _policy_distribution(policy, state) -> dict:
    """Return ``{action_profile: prob}`` for either policy type."""
    if hasattr(policy, "get_distribution"):
        return policy.get_distribution(state)
    return policy(state)


def compare_policies(env, human_policy_prior, policy_a, policy_b) -> dict:
    """Compare two robot policies over reachable non-terminal states."""
    states = [
        s for s in _reachable_states(env)
        if not env.is_terminal(s)
    ]
    max_diff = 0.0
    argmax_agree = 0
    for s in states:
        da, db = _policy_distribution(policy_a, s), _policy_distribution(policy_b, s)
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
def build_config(args, resolved: str) -> Phase2Config:
    """Build a :class:`Phase2Config` for the lookup or neural learner."""
    common = dict(
        gamma_r=args.gamma_r,
        gamma_h=args.gamma_h,
        beta_r=args.beta_r,
        steps_per_episode=8,
        epsilon_r_start=0.6,
        epsilon_r_end=0.1,
    )
    if resolved == "lookup":
        n = args.lookup_iterations
        return Phase2Config(
            use_lookup_tables=True,
            use_lookup_q_r=True,
            use_lookup_v_h_e=True,
            use_lookup_x_h=True,
            use_lookup_u_r=True,
            use_lookup_v_r=True,
            u_r_use_network=True,
            v_r_use_network=True,
            lookup_use_adaptive_lr=True,
            use_count_based_curiosity=True,
            warmup_v_h_e_steps=max(1, n // 10),
            warmup_x_h_steps=max(1, n // 10),
            warmup_u_r_steps=max(1, n // 10),
            warmup_q_r_steps=max(1, n // 10),
            beta_r_rampup_steps=max(1, n // 10),
            num_training_steps=n,
            buffer_size=1000,
            batch_size=64,
            epsilon_r_decay_steps=max(1, n // 2),
            **common,
        )
    n = args.neural_iterations
    return Phase2Config(
        use_lookup_tables=False,
        use_encoders=True,
        u_r_use_network=False,
        v_r_use_network=False,
        x_h_use_network=True,
        hidden_dim=64,
        goal_feature_dim=32,
        warmup_v_h_e_steps=max(1, n // 10),
        warmup_x_h_steps=max(1, n // 10),
        warmup_u_r_steps=0,
        warmup_q_r_steps=max(1, n // 10),
        beta_r_rampup_steps=max(1, n // 10),
        num_training_steps=n,
        buffer_size=2000,
        batch_size=64,
        epsilon_r_decay_steps=max(1, n // 2),
        **common,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--world", nargs="+",
        default=["bushworld_worlds/two_humans_one_robot.yaml"],
        help="One or more BushWorld YAML worlds (or directories of them). A "
             "single world runs in single-map mode; two or more (or a "
             "directory) run in random-map-ensemble mode, where the learner "
             "draws a fresh world each episode.")
    parser.add_argument("--method", default="lookup",
                        choices=["lookup", "neural"],
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
    parser.add_argument("--movie-format", default="mp4", choices=["gif", "mp4"],
                        help="Movie container. Default mp4 (easy to scrub/pause; "
                             "needs imageio[ffmpeg]); gif needs no ffmpeg.")
    parser.add_argument("--skip-bi", action="store_true",
                        help="Skip the exact backward-induction reference (and the "
                             "learned-vs-exact comparison). Automatically enabled in "
                             "random-map-ensemble mode, where the exact solver is "
                             "per-map and can be expensive on larger worlds.")
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
                        help="Training steps for the neural learner.")
    parser.add_argument("--lookup-iterations", type=int, default=600,
                        help="Training steps for the lookup-table learner.")
    parser.add_argument("--quick", action="store_true",
                        help="Fast settings (fewer rollouts / iterations) for smoke testing.")
    args = parser.parse_args()

    if args.quick:
        args.rollouts = min(args.rollouts, 2)
        args.neural_iterations = min(args.neural_iterations, 150)
        args.lookup_iterations = min(args.lookup_iterations, 150)

    os.makedirs(args.output_dir, exist_ok=True)
    checkpoint_dir = args.checkpoint_dir or os.path.join(args.output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    make_movies = not args.no_movie

    np.random.seed(args.seed)

    # --- Load world(s) and human policy prior ----------------------------- #
    world_paths = resolve_worlds(args.world, _REPO_ROOT)
    ensemble = len(world_paths) > 1
    mode = "random-map-ensemble" if ensemble else "single-map"
    print(f"Mode: {mode} ({len(world_paths)} world(s))")
    for p in world_paths:
        print(f"  - {os.path.relpath(p, _REPO_ROOT)}")

    # Primary world: networks, the human prior and goal sampler are built from it.
    env = load_bushworld(world_paths[0])
    print(f"  primary: {env!r}")

    if ensemble:
        # Validate that ensemble worlds are mutually compatible. The learned
        # networks (especially neural encoders) assume a fixed grid size and a
        # fixed number of players across episodes.
        for p in world_paths[1:]:
            other = load_bushworld(p)
            if (other.width, other.height, other.num_robots, other.num_humans) != (
                env.width, env.height, env.num_robots, env.num_humans
            ):
                raise ValueError(
                    "All ensemble worlds must share grid size and player counts. "
                    f"{os.path.relpath(world_paths[0], _REPO_ROOT)} is "
                    f"{env.width}x{env.height} with {env.num_robots} robot(s)/"
                    f"{env.num_humans} human(s), but "
                    f"{os.path.relpath(p, _REPO_ROOT)} is "
                    f"{other.width}x{other.height} with {other.num_robots} robot(s)/"
                    f"{other.num_humans} human(s)."
                )

    human_policy_prior = ShortestPathHumanPolicyPrior(env, env.human_agent_indices)
    goal_generator = env.possible_goal_generator
    goal_sampler = goal_generator.get_sampler()
    resolved = args.method
    config = build_config(args, resolved)
    print(f"  beta_r={args.beta_r}, gamma_h={args.gamma_h}, gamma_r={args.gamma_r}")

    # The exact backward-induction reference is per-map; skip it for ensembles.
    skip_bi = args.skip_bi or ensemble

    # In ensemble mode, draw a fresh world each training episode.
    world_model_factory = None
    if ensemble:
        world_model_factory = EnsembleWorldModelFactory(
            _BushWorldEnsembleLoader(world_paths, seed=args.seed),
            episodes_per_env=1,
        )

    # --- Backward induction (exact reference) ----------------------------- #
    bi_policy = None
    if skip_bi:
        reason = "ensemble mode" if ensemble else "--skip-bi"
        print(f"\n=== Backward induction skipped ({reason}) ===")
    else:
        print("\n=== Backward induction (exact Phase 2) ===")
        bi_policy = compute_robot_policy(
            env, list(env.human_agent_indices), list(env.robot_agent_indices),
            goal_generator, human_policy_prior,
            beta_r=args.beta_r, gamma_h=args.gamma_h, gamma_r=args.gamma_r,
            zeta=args.zeta, xi=args.xi, eta=args.eta,
            level_fct=lambda s: s[0], quiet=True,
        )
        print("  backward induction policy computed.")

    # --- Learned policy with checkpointing -------------------------------- #
    ckpt_name = "neural.pt" if resolved == "neural" else "lookup.pt"
    checkpoint_path = os.path.join(checkpoint_dir, ckpt_name)
    tb_dir = os.path.join(args.output_dir, "tensorboard")
    resume = not args.no_resume and os.path.exists(checkpoint_path)
    if resume:
        print(f"\n=== Learned policy ({resolved}) — resuming from {checkpoint_path} ===")
    else:
        print(f"\n=== Learned policy ({resolved}) — training from scratch ===")

    _q_r, _networks, history, trainer = train_bushworld_phase2(
        env,
        list(env.human_agent_indices),
        list(env.robot_agent_indices),
        human_policy_prior,
        goal_sampler,
        config=config,
        device="cpu",
        verbose=True,
        tensorboard_dir=tb_dir,
        world_model_factory=world_model_factory,
        checkpoint_path=checkpoint_path,
        checkpoint_interval=max(1, config.num_training_steps // 4),
        restore_networks_path=checkpoint_path if resume else None,
    )
    print(f"  checkpoint saved to {checkpoint_path}")

    # --- Save final policy and reload to verify round-trip ---------------- #
    policy_path = os.path.join(args.output_dir, "final_policy.pt")
    trainer.save_policy(policy_path)
    print(f"  final policy saved to {policy_path}")
    reloaded_policy = BushWorldRobotPolicy(path=policy_path, beta_r=args.beta_r)
    reloaded_policy.reset(env)
    print("  final policy reloaded from disk (verifying save/restore).")

    # --- Compare policies -------------------------------------------------- #
    if bi_policy is not None:
        print("\n=== Policy comparison (learned vs backward induction) ===")
        cmp = compare_policies(env, human_policy_prior, bi_policy, reloaded_policy)
        print(f"  states compared: {cmp['num_states']}")
        print(f"  max |pi_r diff|: {cmp['max_prob_diff']:.4e}")
        print(f"  argmax agreement: {cmp['argmax_agreement'] * 100:.1f}%")
        print("  (the learned policy uses the shared sampling-based trainer; it "
              "approximates the backward-induction fixed point.)")

    # --- World providers for rollouts ------------------------------------- #
    # In single-map mode every rollout uses the one loaded world; in ensemble
    # mode each rollout draws a fresh world (with its own human prior / goal
    # sampler) so the movies showcase the policy across the ensemble.
    primary_components = (env, human_policy_prior, goal_sampler)

    def single_map_components(_seed):
        return primary_components

    if ensemble:
        _rollout_rng = np.random.default_rng(args.seed + 7)

        def ensemble_components(_seed):
            path = world_paths[int(_rollout_rng.integers(len(world_paths)))]
            e = load_bushworld(path)
            prior = ShortestPathHumanPolicyPrior(e, e.human_agent_indices)
            sampler = e.possible_goal_generator.get_sampler()
            return e, prior, sampler

        env_components_fn = ensemble_components
    else:
        env_components_fn = single_map_components

    # --- Rollouts and movies ---------------------------------------------- #
    print("\n=== Rollouts ===")
    bi_summary = None
    if bi_policy is not None:
        bi_summary = run_rollouts(
            single_map_components, bi_policy,
            n=args.rollouts, base_seed=args.seed, output_dir=args.output_dir,
            tag="backward_induction", make_movies=make_movies, movie_ext=args.movie_format,
        )
    learned_summary = run_rollouts(
        env_components_fn, reloaded_policy,
        n=args.rollouts, base_seed=args.seed + 1000, output_dir=args.output_dir,
        tag=f"learned_{resolved}", make_movies=make_movies, movie_ext=args.movie_format,
    )

    # --- Optional extra rollouts from the saved final policy --------------- #
    if args.extra_rollouts > 0:
        print(f"\n=== Extra rollouts from saved final policy ({args.extra_rollouts}) ===")
        extra_policy = BushWorldRobotPolicy(path=policy_path, beta_r=args.beta_r)
        extra_policy.reset(env)
        run_rollouts(
            env_components_fn, extra_policy,
            n=args.extra_rollouts, base_seed=args.seed + 2000,
            output_dir=args.output_dir, tag=f"extra_{resolved}",
            make_movies=make_movies, movie_ext=args.movie_format,
        )

    print("\nDone.")
    print(f"  Outputs in: {args.output_dir}")
    if bi_summary is not None:
        print(f"  Backward induction rollouts: {bi_summary}")
    print(f"  Learned rollouts:            {learned_summary}")


if __name__ == "__main__":
    main()

# Phase 2 Open Evaluation (Trajectory Targets × `pi_r_mode`)

This directory codifies the Phase 2 open milestone from `docs/plans/phase2_multistep_targets_and_mcts.md`.

## Reproduce

```bash
PYTHONPATH=src:vendor/multigrid:vendor/ai_transport:multigrid_worlds \
  python examples/phase2/phase2_multistep_mcts_open_eval.py --quick \
  --output-dir docs/evaluations/phase2_open_eval
```

## Latest committed run

- Timestamp: `2026-05-18T11:49:31Z`
- Steps per run: `200`
- Grid: `one_step/n_step/episode × direct/mcts` (6 runs total)
- Artifacts:
  - `open_eval_summary_20260518T114931Z.json`
  - `open_eval_summary_20260518T114931Z.csv`

## Quick summary (from the CSV)

| target_mode | pi_r_mode | wall_clock_seconds | searched_transition_rate | q_r_tail_std | v_h_e_tail_std |
|---|---:|---:|---:|---:|---:|
| one_step | direct | 14.30 | 0.0000 | 408315.09 | 0.0012 |
| one_step | mcts | 122.24 | 0.9825 | 360.67 | 0.0147 |
| n_step | direct | 15.90 | 0.0000 | 465352.38 | 0.0102 |
| n_step | mcts | 122.41 | 0.9850 | 27992666.25 | 0.0110 |
| episode | direct | 16.69 | 0.0000 | 50378906.94 | 0.0285 |
| episode | mcts | 121.98 | 0.9875 | 22885815.88 | 0.0486 |

Interpretation is intentionally left to follow-up analysis runs with larger training budgets; this quick run is primarily for reproducible codification of the milestone comparison protocol and output metrics.


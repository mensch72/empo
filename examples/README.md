# Examples Directory

This directory contains example scripts demonstrating empo functionality.

**Currently the most important one is `phase2_robot_policy_demo.py`.**

## Running Examples

Most examples can be run with:
```bash
PYTHONPATH=src:vendor/multigrid python examples/<script_name>.py
```

For transport environment examples:
```bash
PYTHONPATH=src:vendor/multigrid:vendor/ai_transport python examples/transport_random_demo.py
```

## Development Guidelines

### Quick Test Mode

**All long-running example scripts include a `--quick` or `-q` flag** that reduces training episodes, environments, and other time-consuming parameters. This allows developers to quickly verify scripts work without waiting for full runs.

Example:
```bash
# Full run (may take several minutes)
python examples/random_multigrid_ensemble_demo.py

# Quick test (completes in ~10-30 seconds)
python examples/random_multigrid_ensemble_demo.py --quick
```

When creating new examples, follow the established pattern:
- Use `argparse` to add `--quick/-q` flag
- Define separate constants for full vs quick mode (e.g., `NUM_EPISODES_FULL` vs `NUM_EPISODES_QUICK`)
- Document both modes in the script's docstring

## Available Examples

### Quick-running Examples (complete in < 30 seconds)

These examples run quickly and don't require the `--quick` flag:

- `simple_example.py` - Basic framework demonstration
- `dag_visualization_example.py` - DAG computation and visualization
- `state_management_demo.py` - Environment state get/set operations
- `debug_value_function.py` - Value function tracing for debugging
- `magic_wall_demo.py` - Magic wall cell type demonstration
- `path_distance_visualization.py` - Path-based distance calculation
- `unsteady_ground_animation.py` - Unsteady ground cell type demo
- `blocks_rocks_animation.py` - Block and rock pushing mechanics

### Long-running Examples (support `--quick` flag)

These examples take longer to run in full mode. Use `--quick` for faster testing:

| Example | Full Mode | Quick Mode | Description |
|---------|-----------|------------|-------------|
| `bellman_backward_induction.py` | ~minutes | ~10s | Backward induction on gridworld |
| `neural_policy_prior_demo.py` | 5000 episodes | 100 episodes | Neural network policy learning |
| `control_button_demo.py` | 2000 episodes | 50 episodes | Control button + neural learning |
| `human_policy_prior_example.py` | 3 time steps | 2 time steps | Human policy prior computation |
| `one_or_three_chambers_random_play.py` | 1000 steps | 50 steps | Random play video |
| `benchmark_parallel_dag.py` | Multiple tests | Single test | DAG computation benchmarks |
| `profile_transitions.py` | 8 time steps | 4 time steps | Transition profiling |
| `cooperative_puzzle_demo.py` | 200 steps | 30 steps | Multi-agent puzzle demo |
| `random_multigrid_ensemble_demo.py` | 500 episodes | 50 episodes | Random multigrid ensemble |
| `dag_and_episode_example.py` | Full DAG | Full DAG | DAG computation + episode GIF |
| `single_agent_value_function.py` | 5 beta values | 5 beta values | Value function visualization |
| `transport_random_demo.py` | 100 steps | 20 steps | AI transport environment demo |
| `transport_learning_demo.py` | 500 episodes | 50 episodes | Transport human policy learning |

### Usage Examples

```bash
# Quick test all long-running examples
PYTHONPATH=src:vendor/multigrid python examples/bellman_backward_induction.py --quick
PYTHONPATH=src:vendor/multigrid python examples/neural_policy_prior_demo.py --quick
PYTHONPATH=src:vendor/multigrid python examples/control_button_demo.py --quick

# Transport environment demo
PYTHONPATH=src:vendor/multigrid:vendor/ai_transport python examples/transport_random_demo.py --quick
PYTHONPATH=src:vendor/multigrid:vendor/ai_transport python examples/transport_random_demo.py --render  # with visualization
PYTHONPATH=src:vendor/multigrid:vendor/ai_transport python examples/transport_learning_demo.py --quick  # policy learning

# Run full examples (takes longer)
PYTHONPATH=src:vendor/multigrid python examples/bellman_backward_induction.py
```

### Output Files

Most examples generate output files in the `outputs/` directory:
- Animation GIFs and MP4s
- Visualization PNGs
- Profiling results

Note: MP4 files require FFmpeg. If FFmpeg is not available, examples will fall back to GIF format.

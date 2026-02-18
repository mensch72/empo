# Examples Directory

This directory contains categorized example scripts demonstrating empo functionality.

**Currently the most important one is `examples/phase2/phase2_robot_policy_demo.py`.**

## Layout

- `examples/diagnostics/` - DAG computation, profiling, debugging
- `examples/multigrid/` - MultiGrid environment demos
- `examples/phase1/` - Phase 1 (human policy prior) demos
- `examples/phase2/` - Phase 2 (robot policy) demos
- `examples/transport/` - Transport environment demos
- `examples/visualization/` - Animation and visualization scripts
- `examples/llm/` - LLM-related comparisons

## Running Examples

Most examples can be run with:
```bash
PYTHONPATH=src:vendor/multigrid python examples/<category>/<script_name>.py
```

For transport environment examples:
```bash
PYTHONPATH=src:vendor/multigrid:vendor/ai_transport python examples/transport/transport_random_demo.py
```

## Development Guidelines

### Quick Test Mode

**All long-running example scripts include a `--quick` or `-q` flag** that reduces training episodes, environments, and other time-consuming parameters. This allows developers to quickly verify scripts work without waiting for full runs.

Example:
```bash
# Full run (may take several minutes)
python examples/multigrid/random_multigrid_ensemble_demo.py

# Quick test (completes in ~10-30 seconds)
python examples/multigrid/random_multigrid_ensemble_demo.py --quick
```

When creating new examples, follow the established pattern:
- Use `argparse` to add `--quick/-q` flag
- Define separate constants for full vs quick mode (e.g., `NUM_EPISODES_FULL` vs `NUM_EPISODES_QUICK`)
- Document both modes in the script's docstring

## Available Examples

### Quick-running Examples (complete in < 30 seconds)

These examples run quickly and don't require the `--quick` flag:

- `examples/multigrid/simple_example.py` - Basic framework demonstration
- `examples/diagnostics/dag_visualization_example.py` - DAG computation and visualization
- `examples/multigrid/state_management_demo.py` - Environment state get/set operations
- `examples/diagnostics/debug_value_function.py` - Value function tracing for debugging
- `examples/multigrid/magic_wall_demo.py` - Magic wall cell type demonstration
- `examples/visualization/path_distance_visualization.py` - Path-based distance calculation
- `examples/visualization/unsteady_ground_animation.py` - Unsteady ground cell type demo
- `examples/visualization/blocks_rocks_animation.py` - Block and rock pushing mechanics

### Long-running Examples (support `--quick` flag)

These examples take longer to run in full mode. Use `--quick` for faster testing:

| Example | Full Mode | Quick Mode | Description |
|---------|-----------|------------|-------------|
| `examples/diagnostics/bellman_backward_induction.py` | ~minutes | ~10s | Backward induction on gridworld |
| `examples/phase1/neural_policy_prior_demo.py` | 5000 episodes | 100 episodes | Neural network policy learning |
| `examples/multigrid/control_button_demo.py` | 2000 episodes | 50 episodes | Control button + neural learning |
| `examples/phase1/human_policy_prior_example.py` | 3 time steps | 2 time steps | Human policy prior computation |
| `examples/multigrid/one_or_three_chambers_random_play.py` | 1000 steps | 50 steps | Random play video |
| `examples/diagnostics/benchmark_parallel_dag.py` | Multiple tests | Single test | DAG computation benchmarks |
| `examples/diagnostics/profile_transitions.py` | 8 time steps | 4 time steps | Transition profiling |
| `examples/multigrid/cooperative_puzzle_demo.py` | 200 steps | 30 steps | Multi-agent puzzle demo |
| `examples/multigrid/random_multigrid_ensemble_demo.py` | 500 episodes | 50 episodes | Random multigrid ensemble |
| `examples/diagnostics/dag_and_episode_example.py` | Full DAG | Full DAG | DAG computation + episode GIF |
| `examples/visualization/single_agent_value_function.py` | 5 beta values | 5 beta values | Value function visualization |
| `examples/transport/transport_random_demo.py` | 100 steps | 20 steps | AI transport environment demo |
| `examples/transport/transport_learning_demo.py` | 500 episodes | 50 episodes | Transport human policy learning |

### Usage Examples

```bash
# Quick test all long-running examples
PYTHONPATH=src:vendor/multigrid python examples/diagnostics/bellman_backward_induction.py --quick
PYTHONPATH=src:vendor/multigrid python examples/phase1/neural_policy_prior_demo.py --quick
PYTHONPATH=src:vendor/multigrid python examples/multigrid/control_button_demo.py --quick

# Transport environment demo
PYTHONPATH=src:vendor/multigrid:vendor/ai_transport python examples/transport/transport_random_demo.py --quick
PYTHONPATH=src:vendor/multigrid:vendor/ai_transport python examples/transport/transport_random_demo.py --render  # with visualization
PYTHONPATH=src:vendor/multigrid:vendor/ai_transport python examples/transport/transport_learning_demo.py --quick  # policy learning

# Run full examples (takes longer)
PYTHONPATH=src:vendor/multigrid python examples/diagnostics/bellman_backward_induction.py
```

### Output Files

Most examples generate output files in the `outputs/` directory:
- Animation GIFs and MP4s
- Visualization PNGs
- Profiling results

Note: MP4 files require FFmpeg. If FFmpeg is not available, examples will fall back to GIF format.

# Examples Directory

Examples are now organized by purpose to make them easier to find:

```
examples/
  phase1/         # Human policy prior (Phase 1)
  phase2/         # Robot policy + power metric (Phase 2)
  multigrid/      # General multigrid demos
  transport/      # Transport environment demos
  diagnostics/    # Profiling, DAG, and debugging utilities
  mcts/           # MCTS-based demos
  visualization/  # Animations and plots
```

**Most important Phase 2 demo:**  
`examples/phase2/phase2_robot_policy_demo.py`

## Running Examples

Most Multigrid examples:
```bash
PYTHONPATH=src:vendor/multigrid python examples/multigrid/simple_example.py
```

Phase 1:
```bash
PYTHONPATH=src:vendor/multigrid python examples/phase1/neural_policy_prior_demo.py --quick
```

Phase 2:
```bash
PYTHONPATH=src:vendor/multigrid python examples/phase2/phase2_robot_policy_demo.py --quick
```

MCTS:
```bash
PYTHONPATH=src:vendor/multigrid python examples/mcts/mcts_simple_demo.py
```

Transport examples:
```bash
PYTHONPATH=src:vendor/multigrid:vendor/ai_transport python examples/transport/transport_random_demo.py --quick
```

## Quick Test Mode

Most longer scripts support `--quick` / `-q` to reduce runtime.

Example:
```bash
python examples/phase2/phase2_robot_policy_demo.py --quick
```

## Contents

### Phase 1 (Human Policy Prior)
- `phase1/human_policy_prior_example.py`
- `phase1/neural_policy_prior_demo.py`
- `phase1/policy_prior_transfer_demo.py`
- `phase1/phi_network_ensemble_demo.py`

### Phase 2 (Robot Policy + Power Metric)
- `phase2/phase2_robot_policy_demo.py`
- `phase2/phase2_backward_induction.py`
- `phase2/lookup_table_phase2_demo.py`

### Multigrid Demos
- `multigrid/hello_world.py`
- `multigrid/simple_example.py`
- `multigrid/state_management_demo.py`
- `multigrid/one_or_three_chambers_random_play.py`
- `multigrid/cooperative_puzzle_demo.py`
- `multigrid/control_button_demo.py`
- `multigrid/magic_wall_demo.py`
- `multigrid/simple_rock_push_demo.py`
- `multigrid/heuristic_key_door_demo.py`
- `multigrid/cross_grid_policy_demo.py`
- `multigrid/random_multigrid_ensemble_demo.py`
- `multigrid/random_ensemble_heuristic_exploration_demo.py`
- `multigrid/heuristic_multigrid_ensemble_demo.py`

### Diagnostics
- `diagnostics/bellman_backward_induction.py`
- `diagnostics/benchmark_parallel_dag.py`
- `diagnostics/dag_visualization_example.py`
- `diagnostics/dag_and_episode_example.py`
- `diagnostics/profile_transitions.py`
- `diagnostics/debug_value_function.py`

### Visualization
- `visualization/blocks_rocks_animation.py`
- `visualization/unsteady_ground_animation.py`
- `visualization/path_distance_visualization.py`
- `visualization/rectangle_goal_demo.py`
- `visualization/single_agent_value_function.py`

### Transport
- `transport/transport_handcrafted_demo.py`
- `transport/transport_random_demo.py`
- `transport/transport_learning_demo.py`
- `transport/transport_stress_test_demo.py`
- `transport/transport_two_cluster_demo.py`

### MCTS
- `mcts/mcts_simple_demo.py`

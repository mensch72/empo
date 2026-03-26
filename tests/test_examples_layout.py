from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXAMPLES_DIR = PROJECT_ROOT / "examples"


EXPECTED_CATEGORIES = {
    "diagnostics": {
        "bellman_backward_induction.py",
        "benchmark_parallel_dag.py",
        "dag_and_episode_example.py",
        "dag_visualization_example.py",
        "debug_value_function.py",
        "profile_transitions.py",
    },
    "llm": {
        "llm_comparison.py",
        "llm_comparison.ipynb",
    },
    "multigrid": {
        "control_button_demo.py",
        "cooperative_puzzle_demo.py",
        "cross_grid_policy_demo.py",
        "hello_world.py",
        "heuristic_key_door_demo.py",
        "heuristic_multigrid_ensemble_demo.py",
        "magic_wall_demo.py",
        "one_or_three_chambers_random_play.py",
        "random_ensemble_heuristic_exploration_demo.py",
        "random_multigrid_ensemble_demo.py",
        "simple_example.py",
        "simple_rock_push_demo.py",
        "state_management_demo.py",
    },
    "phase1": {
        "human_policy_prior_example.py",
        "neural_policy_prior_demo.py",
        "phi_network_ensemble_demo.py",
        "policy_prior_transfer_demo.py",
    },
    "phase2": {
        "lookup_table_phase2_demo.py",
        "phase2_backward_induction.py",
        "phase2_robot_policy_demo.py",
    },
    "transport": {
        "transport_handcrafted_demo.py",
        "transport_learning_demo.py",
        "transport_random_demo.py",
        "transport_stress_test_demo.py",
        "transport_two_cluster_demo.py",
    },
    "visualization": {
        "blocks_rocks_animation.py",
        "path_distance_visualization.py",
        "rectangle_goal_demo.py",
        "single_agent_value_function.py",
        "unsteady_ground_animation.py",
    },
}


def test_examples_categories_exist():
    for category in EXPECTED_CATEGORIES:
        category_dir = EXAMPLES_DIR / category
        assert category_dir.is_dir(), f"Missing examples category directory: {category_dir}"


def test_expected_examples_present():
    for category, files in EXPECTED_CATEGORIES.items():
        for filename in files:
            path = EXAMPLES_DIR / category / filename
            assert path.exists(), f"Missing expected example file: {path}"


def test_no_legacy_top_level_example_scripts():
    top_level_scripts = sorted(
        p.name
        for p in EXAMPLES_DIR.glob("*.py")
        if p.is_file()
    )
    assert top_level_scripts == [], (
        "Found legacy top-level example scripts in examples/: "
        f"{top_level_scripts}"
    )

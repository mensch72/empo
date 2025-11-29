"""Debug parallel DAG computation to find missing states."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'vendor', 'multigrid'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from gym_multigrid.multigrid import MultiGridEnv, World


if __name__ == '__main__':
    env = MultiGridEnv(
        map='''
    We We We We We We We
    We .. .. .. Un Un We
    We .. .. .. Un Un We
    We .. .. Ay Un Un We
    We .. .. .. Un Un We
    We .. .. .. Un Un We
    We We We We We We We
    ''',
        objects_set=World,
        orientations=['n'],
        max_steps=4,
        partial_obs=False
    )

    print("Computing DAG sequentially...")
    states_seq, state_to_idx_seq, successors_seq = env.get_dag(return_probabilities=False)
    print(f"  States: {len(states_seq)}")

    print("\nComputing DAG in parallel...")
    states_par, state_to_idx_par, successors_par = env.get_dag_parallel(return_probabilities=False, num_workers=2)
    print(f"  States: {len(states_par)}")

    states_set_seq = set(states_seq)
    states_set_par = set(states_par)

    missing_from_parallel = states_set_seq - states_set_par
    print(f"\nStates missing from parallel version: {len(missing_from_parallel)}")

    if missing_from_parallel:
        print("\nFirst few missing states:")
        for i, state in enumerate(list(missing_from_parallel)[:5]):
            print(f"  {i+1}. {state}")
            
            # Find which state(s) lead to this missing state
            for parent_state in states_seq:
                parent_idx = state_to_idx_seq[parent_state]
                for succ_idx in successors_seq[parent_idx]:
                    if states_seq[succ_idx] == state:
                        print(f"      → Parent: {parent_state}")
                        break

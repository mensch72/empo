"""
Test parallel DAG computation.

Verifies that parallel and sequential DAG computation produce identical results.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'vendor', 'multigrid'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from gym_multigrid.multigrid import MultiGridEnv, World


def test_parallel_dag_correctness():
    """Test that parallel and sequential DAG computation give identical results."""
    
    # Create a moderately complex environment
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
    
    print("\n" + "="*70)
    print("Testing Parallel DAG Computation Correctness")
    print("="*70)
    
    # Compute DAG sequentially
    print("\nComputing DAG sequentially...")
    states_seq, state_to_idx_seq, successors_seq = env.get_dag(return_probabilities=False)
    print(f"  States: {len(states_seq)}")
    
    # Compute DAG in parallel
    print("\nComputing DAG in parallel...")
    states_par, state_to_idx_par, successors_par = env.get_dag_parallel(return_probabilities=False, num_workers=4)
    print(f"  States: {len(states_par)}")
    
    # Verify same number of states
    assert len(states_seq) == len(states_par), \
        f"Different number of states: seq={len(states_seq)}, par={len(states_par)}"
    
    # Verify same states (as sets, since order might differ)
    states_set_seq = set(states_seq)
    states_set_par = set(states_par)
    assert states_set_seq == states_set_par, "Different state sets"
    
    # Verify same successors (need to map through indices)
    print("\nVerifying successor structure...")
    for state in states_seq:
        idx_seq = state_to_idx_seq[state]
        idx_par = state_to_idx_par[state]
        
        # Get successor states (not indices) for comparison
        succ_states_seq = set(states_seq[s_idx] for s_idx in successors_seq[idx_seq])
        succ_states_par = set(states_par[s_idx] for s_idx in successors_par[idx_par])
        
        assert succ_states_seq == succ_states_par, \
            f"Different successors for state {state}"
    
    # Verify root state is the same
    assert states_seq[0] == states_par[0], "Different root states"
    
    # Verify topological ordering (all successors have higher indices)
    print("\nVerifying topological ordering...")
    for idx, succ_list in enumerate(successors_par):
        for succ_idx in succ_list:
            assert idx < succ_idx, f"Topological order violated: {idx} -> {succ_idx}"
    
    print("\n✓ All checks passed! Parallel and sequential DAG computation are identical.")
    print("="*70)


def test_parallel_dag_with_probabilities():
    """Test parallel DAG computation with probability tracking."""
    
    env = MultiGridEnv(
        map='''
We We We We We
We .. .. Ar We
We .. .. .. We
We We We We We
''',
        objects_set=World,
        orientations=['e'],
        max_steps=3,
        partial_obs=False
    )
    
    print("\n" + "="*70)
    print("Testing Parallel DAG with Probabilities")
    print("="*70)
    
    # Sequential with probabilities
    print("\nComputing DAG sequentially (with probabilities)...")
    states_seq, state_to_idx_seq, successors_seq, trans_seq = env.get_dag(return_probabilities=True)
    print(f"  States: {len(states_seq)}")
    
    # Parallel with probabilities
    print("\nComputing DAG in parallel (with probabilities)...")
    states_par, state_to_idx_par, successors_par, trans_par = env.get_dag_parallel(
        return_probabilities=True, num_workers=2
    )
    print(f"  States: {len(states_par)}")
    
    # Verify same states
    assert len(states_seq) == len(states_par)
    assert set(states_seq) == set(states_par)
    
    print("\nVerifying transition probabilities...")
    # For each state, verify transitions match
    for state in states_seq:
        idx_seq = state_to_idx_seq[state]
        idx_par = state_to_idx_par[state]
        
        # Get transitions as sets of (action, successor_state, prob) tuples
        trans_set_seq = set()
        for action, probs, succ_indices in trans_seq[idx_seq]:
            for prob, succ_idx in zip(probs, succ_indices):
                succ_state = states_seq[succ_idx]
                trans_set_seq.add((action, succ_state, prob))
        
        trans_set_par = set()
        for action, probs, succ_indices in trans_par[idx_par]:
            for prob, succ_idx in zip(probs, succ_indices):
                succ_state = states_par[succ_idx]
                trans_set_par.add((action, succ_state, prob))
        
        assert trans_set_seq == trans_set_par, \
            f"Different transitions for state {state}"
    
    print("\n✓ Probability tracking verified! Sequential and parallel match.")
    print("="*70)


if __name__ == '__main__':
    test_parallel_dag_correctness()
    test_parallel_dag_with_probabilities()
    print("\n" + "="*70)
    print("All tests passed!")
    print("="*70)

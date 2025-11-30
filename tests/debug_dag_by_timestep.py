"""Debug parallel DAG with detailed logging."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'vendor', 'multigrid'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from gym_multigrid.multigrid import MultiGridEnv, World

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

print("="*70)
print("Detailed DAG Comparison")
print("="*70)

print("\nComputing sequential DAG...")
states_seq, state_to_idx_seq, successors_seq = env.get_dag(return_probabilities=False)
print(f"Sequential: {len(states_seq)} states")

# Group by timestep
seq_by_time = {}
for state in states_seq:
    t = state[0]
    if t not in seq_by_time:
        seq_by_time[t] = []
    seq_by_time[t].append(state)

for t in sorted(seq_by_time.keys()):
    print(f"  t={t}: {len(seq_by_time[t])} states")

print("\nComputing parallel DAG...")
states_par, state_to_idx_par, successors_par = env.get_dag_parallel(return_probabilities=False, num_workers=2)
print(f"Parallel: {len(states_par)} states")

# Group by timestep
par_by_time = {}
for state in states_par:
    t = state[0]
    if t not in par_by_time:
        par_by_time[t] = []
    par_by_time[t].append(state)

for t in sorted(par_by_time.keys()):
    print(f"  t={t}: {len(par_by_time[t])} states")

print("\nComparing by timestep...")
all_times = sorted(set(list(seq_by_time.keys()) + list(par_by_time.keys())))

for t in all_times:
    seq_states = set(seq_by_time.get(t, []))
    par_states = set(par_by_time.get(t, []))
    
    if seq_states == par_states:
        print(f"  t={t}: ✓ Same ({len(seq_states)} states)")
    else:
        print(f"  t={t}: ✗ DIFFERENT - seq:{len(seq_states)}, par:{len(par_states)}")
        missing = seq_states - par_states
        extra = par_states - seq_states
        if missing:
            print(f"      Missing from parallel: {len(missing)}")
            for s in list(missing)[:2]:
                print(f"        {s}")
        if extra:
            print(f"      Extra in parallel: {len(extra)}")
            for s in list(extra)[:2]:
                print(f"        {s}")

"""Test the _process_state_actions worker function directly."""

import sys
import os
import pickle
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'vendor', 'multigrid'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from gym_multigrid.multigrid import MultiGridEnv, World
from empo.world_model import _init_dag_worker, _process_state_actions
import empo.world_model as wm

env = MultiGridEnv(
    map='''
We We We We We
We .. Ar .. We
We .. .. .. We
We We We We We
''',
    objects_set=World,
    orientations=['e'],
    max_steps=2,
    partial_obs=False
)

env.reset()
root_state = env.get_state()

print(f"Root state: {root_state}")
print(f"Num agents: {len(env.agents)}")
print(f"Num actions: {env.action_space.n}")

# Initialize worker globals using pickle (as the parallel version does)
env_pickle = pickle.dumps(env)
_init_dag_worker(env_pickle)

total_combinations = env.action_space.n ** len(env.agents)
print(f"\nTotal action combinations: {total_combinations}")

# Process all combinations sequentially
print("\nSequential processing:")
env.set_state(root_state)
seq_successors = set()
for combo_idx in range(total_combinations):
    action_profile = []
    temp = combo_idx
    for _ in range(len(env.agents)):
        action_profile.append(temp % env.action_space.n)
        temp //= env.action_space.n
    
    trans_result = env.transition_probabilities(root_state, tuple(action_profile))
    if trans_result:
        for prob, succ_state in trans_result:
            seq_successors.add(succ_state)

print(f"Sequential successors: {len(seq_successors)}")

# Process using worker function (processes all actions for a single state)
print("\nWorker function processing:")
state, worker_successors, action_data = _process_state_actions(root_state)
print(f"Worker successors: {len(worker_successors)}")

# Compare
print(f"\nMatch: {seq_successors == worker_successors}")

if seq_successors != worker_successors:
    print(f"Missing from worker: {seq_successors - worker_successors}")
    print(f"Extra in worker: {worker_successors - seq_successors}")

#!/usr/bin/env python3
"""Test multiprocessing spawn in Docker with use_encoders=False."""

import pickle
import time
import torch
import numpy as np
import multiprocessing as mp
from gym_multigrid.multigrid import MultiGridEnv, World, SmallActions
from empo.nn_based.phase2.config import Phase2Config
from empo.nn_based.phase2.world_model_factory import CachedWorldModelFactory
from empo.nn_based.multigrid.phase2.trainer import (
    MultiGridPhase2Trainer, create_phase2_networks
)
from empo.possible_goal import TabularGoalSampler
from empo.multigrid import ReachCellGoal
from empo.human_policy_prior import HumanPolicyPrior


GRID_MAP = '''
We We We We We We
We Ar .. .. .. We
We We Ay We We We
We We We We We We
'''


class UniformPrior(HumanPolicyPrior):
    """Simple uniform policy for testing."""
    
    def __init__(self, num_actions=4):
        self.num_actions = num_actions
        
    def get_action_probabilities(self, state, goal, agent_index):
        return np.ones(self.num_actions) / self.num_actions
        
    def __call__(self, state, goal, agent_index):
        return self.get_action_probabilities(state, goal, agent_index)


class EnvCreator:
    """Picklable callable for creating env."""
    def __init__(self, grid_map):
        self.grid_map = grid_map
        
    def __call__(self):
        env = MultiGridEnv(
            map=self.grid_map,
            max_steps=10,
            partial_obs=False,
            objects_set=World,
            actions_set=SmallActions
        )
        return env


def call_method_wrapper(method_name, trainer_pkl, args):
    """Wrapper to call trainer method in spawned process."""
    import pickle
    import sys
    print(f'[Worker] ENTRY - unpickling trainer for {method_name}...', flush=True)
    sys.stdout.flush()
    
    trainer = pickle.loads(trainer_pkl)
    print(f'[Worker] Trainer unpickled, calling {method_name}...', flush=True)
    
    method = getattr(trainer, method_name)
    method(*args)


def minimal_worker(trainer_pkl, actor_id):
    """Minimal worker function to test trainer unpickling."""
    import pickle
    import sys
    print(f'[Worker {actor_id}] ENTRY - unpickling...', flush=True)
    sys.stdout.flush()
    
    trainer = pickle.loads(trainer_pkl)
    print(f'[Worker {actor_id}] Trainer unpickled!', flush=True)
    print(f'[Worker {actor_id}] use_encoders: {trainer.config.use_encoders}', flush=True)
    print(f'[Worker {actor_id}] Q_r feature_dim: {trainer.networks.q_r.state_encoder.feature_dim}', flush=True)
    
    # Test forward pass
    print(f'[Worker {actor_id}] Testing forward pass...', flush=True)
    actor_state = trainer._init_actor_state()
    print(f'[Worker {actor_id}] Actor state initialized!', flush=True)
    
    action = trainer.sample_robot_action(actor_state.state)
    print(f'[Worker {actor_id}] Sampled action: {action}', flush=True)
    
    return True


def main():
    print('Creating environment...')
    env = MultiGridEnv(
        map=GRID_MAP, 
        max_steps=10, 
        partial_obs=False, 
        objects_set=World, 
        actions_set=SmallActions
    )
    env.reset()
    
    print(f'Created env: {env.width}x{env.height}')

    print('Creating config with use_encoders=False...')
    config = Phase2Config(
        use_encoders=False,  # Test identity mode
        num_training_steps=100,
        steps_per_episode=10,
        async_training=True,
        num_actors=1,
    )
    print(f'use_encoders = {config.use_encoders}')

    print('Creating networks...')
    networks = create_phase2_networks(
        env=env,
        config=config,
        num_robots=1,
        num_actions=4,
        device='cpu'
    )
    print(f'Q_r state_encoder.feature_dim: {networks.q_r.state_encoder.feature_dim}')
    print(f'Q_r q_head input: {networks.q_r.q_head[0].in_features}')

    # Create goal sampler
    goal_sampler = TabularGoalSampler([
        ReachCellGoal(env, 1, (2,1)),
    ])
    
    # Create human policy (uniform for simplicity)
    human_policy = UniformPrior(num_actions=4)

    # Factory for creating env in workers
    factory = CachedWorldModelFactory(EnvCreator(GRID_MAP))

    # Create trainer
    print("Creating trainer...")
    trainer = MultiGridPhase2Trainer(
        env=env,
        networks=networks,
        config=config,
        human_agent_indices=[1],
        robot_agent_indices=[0],
        human_policy_prior=human_policy,
        goal_sampler=goal_sampler,
        device='cpu',
        verbose=True,
        world_model_factory=factory,
    )
    print(f'Trainer created.')
    print(f'Trainer config.use_encoders: {trainer.config.use_encoders}')

    # Local test
    print('')
    print('[Main] Local test...')
    actor_state = trainer._init_actor_state()
    action = trainer.sample_robot_action(actor_state.state)
    print(f'[Main] Local action: {action}')

    # Test spawning with manually pickled trainer
    print('')
    print('[Main] Testing spawn with manually pickled trainer...')
    
    trainer_pkl = pickle.dumps(trainer)
    print(f'[Main] Trainer pickled: {len(trainer_pkl)} bytes')
    
    ctx = mp.get_context('spawn')
    
    p = ctx.Process(
        target=minimal_worker,
        args=(trainer_pkl, 0),
        daemon=True
    )
    print("[Main] Starting process...")
    p.start()
    
    # Wait for process
    p.join(timeout=10.0)
    
    if p.exitcode == 0:
        print("[Main] Manual pickle test: SUCCESS!")
    else:
        print(f"[Main] Manual pickle test: FAILED - exit code: {p.exitcode}")
        return
    
    # Now test the actual _actor_process_entry call pattern
    print('')
    print('[Main] Testing _actor_process_entry call pattern...')
    
    transition_queue = ctx.Queue(maxsize=100)
    stop_event = ctx.Event()
    shared_training_steps = ctx.Value('i', 0)
    shared_env_steps = ctx.Value('i', 0)
    
    # Test without manager first
    print('[Main] Creating manager...')
    manager = ctx.Manager()
    print('[Main] Manager created.')
    
    policy_lock = manager.Lock()
    print('[Main] Lock created.')
    
    shared_policy = manager.dict()
    print('[Main] Dict created.')
    
    shared_policy['state_dict'] = trainer._serialize_policy_state()
    shared_policy['version'] = 0
    print('[Main] Policy serialized.')
    
    # First test: just pickle the bound method to see if that crashes
    print('[Main] Testing pickle of bound method...')
    import sys
    try:
        bound_method_pkl = pickle.dumps(trainer._actor_process_entry)
        print(f'[Main] Bound method pickled: {len(bound_method_pkl)} bytes')
    except Exception as e:
        print(f'[Main] Bound method pickle failed: {e}')
        return
    
    # Test with our wrapper instead of bound method
    print('[Main] Testing wrapper method...')
    args = (
        0,  # actor_id
        transition_queue,
        stop_event,
        shared_training_steps,
        shared_env_steps,
        shared_policy,
        policy_lock,
    )
    p_wrapper = ctx.Process(
        target=call_method_wrapper,
        args=('_actor_process_entry', trainer_pkl, args),
        daemon=True
    )
    p_wrapper.start()
    time.sleep(5.0)
    if p_wrapper.is_alive():
        print('[Main] Wrapper test: SUCCESS - process running!')
        stop_event.set()
        p_wrapper.join(timeout=2.0)
        if p_wrapper.is_alive():
            p_wrapper.terminate()
    else:
        print(f'[Main] Wrapper test: FAILED - exit code: {p_wrapper.exitcode}')
        return
    
    # Reset stop event for next test
    stop_event = ctx.Event()
    shared_policy['version'] = 0
    
    # Now test the real Process call with bound method
    print('')
    print('[Main] Now testing actual Process call with bound method...')
    
    # Start actor process like the real code does
    p = ctx.Process(
        target=trainer._actor_process_entry,
        args=(
            0,  # actor_id
            transition_queue,
            stop_event,
            shared_training_steps,
            shared_env_steps,
            shared_policy,
            policy_lock,
        ),
        daemon=True
    )
    print("[Main] Starting actor process...")
    sys.stdout.flush()
    p.start()
    print(f"[Main] Process started, pid={p.pid}")
    sys.stdout.flush()
    
    # Wait a bit for actor to start
    time.sleep(5.0)
    
    # Check if process is still alive
    if p.is_alive():
        print("[Main] Actor process is running!")
        stop_event.set()
        p.join(timeout=2.0)
        if p.is_alive():
            print("[Main] Force terminating...")
            p.terminate()
        print("[Main] Actor process entry test: SUCCESS!")
    else:
        print(f"[Main] Actor process died! Exit code: {p.exitcode}")
        print("[Main] Actor process entry test: FAILED")


if __name__ == '__main__':
    main()

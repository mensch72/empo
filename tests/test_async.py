#!/usr/bin/env python3
"""Test full async training setup with use_encoders=False."""

import torch
import multiprocessing as mp
import sys
import traceback


def actor_entry(trainer, actor_id, queue, stop_event, shared_steps, shared_env, shared_policy, lock):
    """Simple actor entry point for testing."""
    print(f"[Actor {actor_id}] Starting...")
    print(f"[Actor {actor_id}] Config use_encoders: {trainer.config.use_encoders}")
    print(f"[Actor {actor_id}] Q_r state_encoder.feature_dim: {trainer.networks.q_r.state_encoder.feature_dim}")
    print(f"[Actor {actor_id}] Q_r q_head input: {trainer.networks.q_r.q_head[0].in_features}")
    
    # Try to init actor state (this creates env and samples first state)
    try:
        print(f"[Actor {actor_id}] Initializing actor state...")
        actor_state = trainer._init_actor_state()
        print(f"[Actor {actor_id}] Actor state initialized.")
        
        # Try sample_robot_action
        print(f"[Actor {actor_id}] Sampling robot action...")
        action = trainer.sample_robot_action(actor_state.state)
        print(f"[Actor {actor_id}] Sampled action: {action}")
        
        print(f"[Actor {actor_id}] SUCCESS!")
    except Exception as e:
        print(f"[Actor {actor_id}] FAILED: {e}")
        traceback.print_exc()
    
    stop_event.set()


if __name__ == '__main__':
    from gym_multigrid.multigrid import MultiGridEnv, World, SmallActions
    from empo.learning_based.phase2.config import Phase2Config
    from empo.learning_based.phase2.world_model_factory import CachedWorldModelFactory
    from empo.learning_based.multigrid.phase2.trainer import (
        MultiGridPhase2Trainer, create_phase2_networks
    )
    from empo.possible_goal import TabularGoalSampler
    from empo.world_specific_helpers.multigrid import ReachCellGoal
    from empo.human_policy_prior import HeuristicPotentialPolicy
    from empo.learning_based.multigrid import PathDistanceCalculator
    
    GRID_MAP = '''
    We We We We We We
    We Ae Ro .. .. We
    We We Ay We We We
    We We We We We We
    '''

    # Create env
    env = MultiGridEnv(
        map=GRID_MAP,
        max_steps=10,
        partial_obs=False,
        objects_set=World,
        actions_set=SmallActions
    )
    env.reset()
    
    print(f'Created env: {env.width}x{env.height}')

    # Config with use_encoders=False
    config = Phase2Config(
        use_encoders=False,
        num_training_steps=100,
        steps_per_episode=10,
        async_training=True,
        num_actors=1,
    )
    print(f'use_encoders = {config.use_encoders}')

    # Create networks
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
    
    # Create human policy
    path_calc = PathDistanceCalculator(env)
    human_policy = HeuristicPotentialPolicy(
        env, [1], path_calc, beta=10.0
    )
    
    # Factory for creating env in workers
    factory = CachedWorldModelFactory(
        map=GRID_MAP,
        max_steps=10,
        partial_obs=False,
        objects_set=World,
        actions_set=SmallActions
    )

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

    # Test spawn
    print("\nTesting spawn...")
    ctx = mp.get_context('spawn')
    
    queue = ctx.Queue(maxsize=100)
    stop_event = ctx.Event()
    shared_steps = ctx.Value('i', 0)
    shared_env = ctx.Value('i', 0)
    manager = ctx.Manager()
    lock = manager.Lock()
    shared_policy = manager.dict()
    shared_policy['state_dict'] = trainer._serialize_policy_state()
    shared_policy['version'] = 0
    
    p = ctx.Process(
        target=actor_entry,
        args=(trainer, 0, queue, stop_event, shared_steps, shared_env, shared_policy, lock),
        daemon=True
    )
    p.start()
    p.join(timeout=30)
    
    if p.is_alive():
        print("Process hung!")
        p.terminate()
    
    print(f'Worker exit code: {p.exitcode}')

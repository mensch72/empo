#!/usr/bin/env python3
"""Test multiprocessing spawn with use_encoders=False."""

import torch
import multiprocessing as mp

def worker_test(q_r):
    print(f'Worker Q_r state_encoder feature_dim: {q_r.state_encoder.feature_dim}')
    print(f'Worker Q_r q_head input: {q_r.q_head[0].in_features}')
    # Create a simple test input
    batch_size = 1
    grid = torch.zeros(batch_size, q_r.state_encoder.num_grid_channels, 
                       q_r.state_encoder.grid_height, q_r.state_encoder.grid_width)
    glob = torch.zeros(batch_size, q_r.state_encoder._global_features_size)
    agent = torch.zeros(batch_size, q_r.state_encoder._agent_input_size)
    inter = torch.zeros(batch_size, q_r.state_encoder._interactive_input_size)
    # Try forward
    try:
        out = q_r.forward(grid, glob, agent, inter)
        print(f'Worker forward succeeded: {out.shape}')
    except Exception as e:
        import traceback
        print(f'Worker forward FAILED: {e}')
        traceback.print_exc()


if __name__ == '__main__':
    from gym_multigrid.multigrid import MultiGridEnv, World, SmallActions
    from empo.nn_based.phase2.config import Phase2Config
    from empo.nn_based.multigrid.phase2.trainer import create_phase2_networks
    
    GRID_MAP = '''
    We We We We We We
    We Ae Ro .. .. We
    We We Ay We We We
    We We We We We We
    '''

    env = MultiGridEnv(
        map=GRID_MAP,
        max_steps=10,
        partial_obs=False,
        objects_set=World,
        actions_set=SmallActions
    )
    env.reset()

    config = Phase2Config(use_encoders=False)
    print(f'use_encoders = {config.use_encoders}')

    networks = create_phase2_networks(
        env=env,
        config=config,
        num_robots=1,
        num_actions=4,
        device='cpu'
    )

    print(f'Main process Q_r state_encoder feature_dim: {networks.q_r.state_encoder.feature_dim}')
    print(f'Main process Q_r q_head input: {networks.q_r.q_head[0].in_features}')

    print('Testing spawn...')
    ctx = mp.get_context('spawn')
    p = ctx.Process(target=worker_test, args=(networks.q_r,))
    p.start()
    p.join()
    print(f'Worker exit code: {p.exitcode}')

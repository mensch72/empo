#!/usr/bin/env python
"""Minimal test: full train() flow with 0 warmup."""
import sys
import os
import time
import logging

os.environ["MPLCONFIGDIR"] = "/tmp/mpl"
os.makedirs("/tmp/mpl", exist_ok=True)
logging.basicConfig(level=logging.WARNING, format="%(name)s %(message)s", stream=sys.stdout)

print("[1] start", flush=True)
t0 = time.time()

from gym_multigrid.multigrid import MultiGridEnv, SmallActions
from empo.human_policy_prior import HeuristicPotentialPolicy
from empo.learning_based.multigrid import PathDistanceCalculator
from empo.learning_based.phase2_ppo.config import PPOPhase2Config
from empo.learning_based.phase2_ppo.trainer import PPOPhase2Trainer
from empo.learning_based.multigrid.phase2_ppo.networks import create_multigrid_ppo_networks
from empo.learning_based.multigrid.phase2_ppo.env_wrapper import MultiGridWorldModelEnv
import numpy as np

print(f"[2] imports: {time.time()-t0:.1f}s", flush=True)

world_yaml = os.path.join(os.getcwd(), "multigrid_worlds", "trivial.yaml")
ref_env = MultiGridEnv(config_file=world_yaml, partial_obs=False, actions_set=SmallActions)
ref_env.reset()
num_actions = ref_env.action_space.n
robot_idx = ref_env.robot_agent_indices
human_idx = ref_env.human_agent_indices
print(f"[3] env: robots={robot_idx}, humans={human_idx}, actions={num_actions}", flush=True)

cfg = PPOPhase2Config(
    num_actions=num_actions,
    num_robots=len(robot_idx),
    ppo_rollout_length=16,
    num_envs=2,
    num_ppo_iterations=2,
    batch_size=8,
    warmup_v_h_e_steps=0,
    warmup_x_h_steps=0,
    warmup_u_r_steps=0,
    steps_per_episode=ref_env.max_steps,
    device="cpu",
    seed=42,
    aux_training_steps_per_iteration=1,
)

ac, aux, enc = create_multigrid_ppo_networks(
    env=ref_env, config=cfg, feature_dim=32, use_x_h=True, use_u_r=False, device="cpu",
)
trainer = PPOPhase2Trainer(config=cfg, actor_critic=ac, auxiliary_networks=aux, device="cpu")
print(f"[4] setup done: {time.time()-t0:.1f}s", flush=True)

# Build env_creator
path_calc = PathDistanceCalculator(
    grid_height=ref_env.height, grid_width=ref_env.width, world_model=ref_env
)
hpp = HeuristicPotentialPolicy(
    world_model=ref_env,
    human_agent_indices=human_idx,
    beta=1000.0,
    path_calculator=path_calc,
)


def hp_fn(state, h_idx, goal, world_model):
    return hpp.get_action_probabilities(state, h_idx, goal, world_model)


pg = ref_env.possible_goal_generator


def gs_fn(state, h_idx):
    goals = list(pg.generate(state, h_idx))
    if goals:
        g = goals[np.random.randint(len(goals))]
        return g, 1.0
    from empo.possible_goal import PossibleGoal

    class DG(PossibleGoal):
        def is_achieved(self, s):
            return 0

        def __hash__(self):
            return 0

        def __eq__(self, o):
            return isinstance(o, DG)

    return DG(), 1.0


def env_creator():
    wm = MultiGridEnv(
        config_file=world_yaml, partial_obs=False, actions_set=SmallActions
    )
    wm.reset()
    return MultiGridWorldModelEnv(
        world_model=wm,
        human_policy_prior=hp_fn,
        goal_sampler=gs_fn,
        human_agent_indices=human_idx,
        robot_agent_indices=robot_idx,
        config=cfg,
        state_encoder=enc,
    )


# Quick test
te = env_creator()
obs, info = te.reset()
obs2, r, d, t2, i2 = te.step(0)
print(
    f"[5] env test: obs={obs.shape}, buf={len(te._aux_buffer)}, "
    f"goals={list(te._goals.keys())}",
    flush=True,
)
te.close()

print("[6] starting train...", flush=True)
t1 = time.time()
metrics = trainer.train(env_creator, num_iterations=2)
t2 = time.time()
print(f"[7] train done: {t2-t1:.1f}s", flush=True)
if metrics:
    last = metrics[-1]
    print(f"  keys: {list(last.keys())}", flush=True)
    vl = last.get("v_h_e_loss", "N/A")
    xl = last.get("x_h_loss", "N/A")
    print(f"  v_h_e_loss: {vl}", flush=True)
    print(f"  x_h_loss: {xl}", flush=True)
else:
    print("  No metrics!", flush=True)

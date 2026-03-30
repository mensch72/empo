#!/usr/bin/env python
"""Ultra-minimal test: can we even create and step the EMPO env via PufferLib?"""
import sys, os, time
os.environ["MPLCONFIGDIR"] = "/tmp/mpl"
os.makedirs("/tmp/mpl", exist_ok=True)

print("[1] Importing...", flush=True)
t0 = time.time()

from gym_multigrid.multigrid import MultiGridEnv, SmallActions
from empo.human_policy_prior import HeuristicPotentialPolicy
from empo.learning_based.multigrid import PathDistanceCalculator

from empo.learning_based.phase2_ppo.config import PPOPhase2Config
from empo.learning_based.multigrid.phase2_ppo.env_wrapper import MultiGridWorldModelEnv
from empo.learning_based.multigrid.phase2_ppo.networks import create_multigrid_ppo_networks

import numpy as np
import pufferlib.emulation
import pufferlib.vector
import pufferlib.pufferl

print(f"[2] Imports done in {time.time()-t0:.1f}s", flush=True)

# Create reference env
world_yaml = os.path.join(os.getcwd(), "multigrid_worlds", "trivial.yaml")
ref_env = MultiGridEnv(config_file=world_yaml, partial_obs=False, actions_set=SmallActions)
ref_env.reset()
print(f"[3] Env created: {ref_env.width}x{ref_env.height}, agents={len(ref_env.agents)}", flush=True)

num_actions = ref_env.action_space.n
robot_indices = ref_env.robot_agent_indices
human_indices = ref_env.human_agent_indices
print(f"    robots={robot_indices}, humans={human_indices}, actions={num_actions}", flush=True)

# Config with NO warmup
cfg = PPOPhase2Config(
    num_actions=num_actions,
    num_robots=len(robot_indices),
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
print(f"[4] Config: batch_size={cfg.batch_size}, warmup={cfg.get_total_warmup_steps()}", flush=True)

# Create networks
actor_critic, aux_nets, state_encoder = create_multigrid_ppo_networks(
    env=ref_env,
    config=cfg,
    feature_dim=32,
    use_x_h=True,
    use_u_r=False,
    device="cpu",
)
print(f"[5] Networks created", flush=True)

# Create trainer
from empo.learning_based.phase2_ppo.trainer import PPOPhase2Trainer

trainer = PPOPhase2Trainer(
    config=cfg,
    actor_critic=actor_critic,
    auxiliary_networks=aux_nets,
    device="cpu",
)
print(f"[6] Trainer created", flush=True)

# Create env_creator
path_calc = PathDistanceCalculator(
    grid_height=ref_env.height,
    grid_width=ref_env.width,
    world_model=ref_env,
)
hpp = HeuristicPotentialPolicy(
    world_model=ref_env,
    human_agent_indices=human_indices,
    beta=1000.0,
    path_calculator=path_calc,
)

def hp_fn(state, h_idx, goal, world_model):
    return hpp.get_action_probabilities(state, h_idx, goal, world_model)

pg_gen = ref_env.possible_goal_generator

def gs_fn(state, h_idx):
    goals = pg_gen.generate(state, h_idx, ref_env)
    if goals:
        g = goals[np.random.randint(len(goals))]
        return g, 1.0
    from empo.possible_goal import PossibleGoal
    class DummyGoal(PossibleGoal):
        def is_achieved(self, s): return 0
        def __hash__(self): return 0
        def __eq__(self, o): return type(o) == type(self)
    return DummyGoal(), 1.0

def env_creator():
    wm = MultiGridEnv(config_file=world_yaml, partial_obs=False, actions_set=SmallActions)
    wm.reset()
    return MultiGridWorldModelEnv(
        world_model=wm,
        human_policy_prior=hp_fn,
        goal_sampler=gs_fn,
        human_agent_indices=human_indices,
        robot_agent_indices=robot_indices,
        config=cfg,
        state_encoder=state_encoder,
    )

# Test env_creator
print("[7] Testing env_creator...", flush=True)
test_env = env_creator()
obs, info = test_env.reset()
print(f"    obs shape={obs.shape}, goals={dict(test_env._goals)}", flush=True)
obs2, rew, term, trunc, info2 = test_env.step(0)
print(f"    step: rew={rew}, term={term}, buf_len={len(test_env._aux_buffer)}", flush=True)
test_env.close()

# Now test via trainer.train()
print("[8] Starting trainer.train()...", flush=True)
t1 = time.time()
metrics = trainer.train(env_creator, num_iterations=2)
t2 = time.time()
print(f"[9] Training done in {t2-t1:.1f}s", flush=True)
if metrics:
    last = metrics[-1]
    print(f"    Last metrics keys: {list(last.keys())}", flush=True)
    print(f"    v_h_e_loss: {last.get('v_h_e_loss', 'N/A')}", flush=True)
    print(f"    x_h_loss: {last.get('x_h_loss', 'N/A')}", flush=True)
else:
    print("    No metrics returned!", flush=True)

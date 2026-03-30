#!/usr/bin/env python
"""Debug script to trace aux buffer population in Phase 2 PPO."""
import sys
import os

os.environ["MPLCONFIGDIR"] = "/tmp/mpl"
os.makedirs("/tmp/mpl", exist_ok=True)

# Monkey-patch before imports
from empo.learning_based.phase2_ppo import trainer as trainer_mod

OrigClass = trainer_mod.PPOPhase2Trainer

_orig_collect = OrigClass._collect_aux_data_from_rollout

def _debug_collect(self, pufferl, vecenv):
    envs = getattr(vecenv, "envs", None)
    print(f"[DBG] _collect_aux_data: envs={envs is not None}", flush=True)
    if envs:
        print(f"[DBG]   num envs = {len(envs)}", flush=True)
        for i, env in enumerate(envs[:2]):
            inner = env
            depth = 0
            while hasattr(inner, "env") and depth < 20:
                inner = inner.env
                depth += 1
            buf = getattr(inner, "_aux_buffer", None)
            buf_len = len(buf) if buf is not None else -1
            print(
                f"[DBG]   env[{i}] type={type(inner).__name__} depth={depth} "
                f"has_buf={buf is not None} buf_len={buf_len}",
                flush=True,
            )
    before = len(self.aux_replay_buffer)
    _orig_collect(self, pufferl, vecenv)
    after = len(self.aux_replay_buffer)
    print(f"[DBG]   replay_buffer: {before} -> {after}", flush=True)

OrigClass._collect_aux_data_from_rollout = _debug_collect

_orig_train_aux = OrigClass.train_auxiliary_step

def _debug_train_aux(self, **kwargs):
    losses = _orig_train_aux(self, **kwargs)
    buf_len = len(self.aux_replay_buffer)
    bs = self.config.batch_size
    if not losses:
        print(f"[DBG] train_aux: EMPTY (buf={buf_len} < bs={bs})", flush=True)
    else:
        print(f"[DBG] train_aux: {losses} (buf={buf_len})", flush=True)
    return losses

OrigClass.train_auxiliary_step = _debug_train_aux

# Now run the actual script
sys.argv = [
    "debug_aux.py",
    "--iters", "3",
    "--world", "trivial.yaml",
    "--num-envs", "2",
]

# Import and run
import importlib.util

spec = importlib.util.spec_from_file_location(
    "script", "examples/phase2/phase2_ppo_robot_policy.py"
)
mod = importlib.util.module_from_spec(spec)
mod.__file__ = "examples/phase2/phase2_ppo_robot_policy.py"
spec.loader.exec_module(mod)

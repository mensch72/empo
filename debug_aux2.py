#!/usr/bin/env python
"""Minimal test: skip warmup, check if aux buffer populates during PPO."""
import sys, os
os.environ["MPLCONFIGDIR"] = "/tmp/mpl"
os.makedirs("/tmp/mpl", exist_ok=True)

sys.argv = [
    "test.py", "--iters", "3", "--world", "trivial.yaml",
    "--num-envs", "2",
]

# Patch config to skip warmup
from empo.learning_based.phase2_ppo.config import PPOPhase2Config
_orig_init = PPOPhase2Config.__init__
def _patched_init(self, **kw):
    kw.setdefault("warmup_v_h_e_steps", 0)
    kw.setdefault("warmup_x_h_steps", 0)
    kw.setdefault("warmup_u_r_steps", 0)
    _orig_init(self, **kw)
PPOPhase2Config.__init__ = _patched_init

# Patch trainer to add debug prints 
from empo.learning_based.phase2_ppo.trainer import PPOPhase2Trainer

_orig_collect = PPOPhase2Trainer._collect_aux_data_from_rollout
def _debug_collect(self, pufferl, vecenv):
    envs = getattr(vecenv, 'envs', None)
    n = len(envs) if envs else 0
    bufs = []
    if envs:
        for env in envs:
            inner = env
            for _ in range(20):
                if not hasattr(inner, 'env'):
                    break
                inner = inner.env
            b = getattr(inner, '_aux_buffer', None)
            bufs.append(len(b) if b is not None else -1)
    before = len(self.aux_replay_buffer)
    _orig_collect(self, pufferl, vecenv)
    after = len(self.aux_replay_buffer)
    print(f"[DBG] collect: {n} envs, bufs_before={bufs}, replay {before}->{after}", flush=True)
PPOPhase2Trainer._collect_aux_data_from_rollout = _debug_collect

_orig_train = PPOPhase2Trainer.train_auxiliary_step
def _debug_train(self, **kw):
    r = _orig_train(self, **kw)
    if not r:
        print(f"[DBG] train_aux empty: buf={len(self.aux_replay_buffer)} bs={self.config.batch_size}", flush=True)
    else:
        print(f"[DBG] train_aux: {list(r.keys())}", flush=True)
    return r
PPOPhase2Trainer.train_auxiliary_step = _debug_train

import importlib.util
spec = importlib.util.spec_from_file_location("script", "examples/phase2/phase2_ppo_robot_policy.py")
mod = importlib.util.module_from_spec(spec)
mod.__file__ = "examples/phase2/phase2_ppo_robot_policy.py"
spec.loader.exec_module(mod)

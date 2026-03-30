#!/usr/bin/env python
"""Quick test: verify aux losses with 0 warmup and 5 PPO iters."""
import sys
import os
import time

os.environ["MPLCONFIGDIR"] = "/tmp/mpl"
os.makedirs("/tmp/mpl", exist_ok=True)

# Monkey-patch warmup to 0 before importing config
from empo.learning_based.phase2_ppo import config as cfg_mod
_orig = cfg_mod.PPOPhase2Config.__init__
def _patched_init(self, **kw):
    kw["warmup_v_h_e_steps"] = 0
    kw["warmup_x_h_steps"] = 0
    kw["warmup_u_r_steps"] = 0
    _orig(self, **kw)
cfg_mod.PPOPhase2Config.__init__ = _patched_init

# Run the actual script
sys.argv = [
    "test.py", "--iters", "5", "--world", "trivial.yaml",
    "--num-envs", "2",
]

import importlib.util
spec = importlib.util.spec_from_file_location(
    "script", "examples/phase2/phase2_ppo_robot_policy.py"
)
mod = importlib.util.module_from_spec(spec)
mod.__file__ = "examples/phase2/phase2_ppo_robot_policy.py"
spec.loader.exec_module(mod)

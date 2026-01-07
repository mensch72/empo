# Backward Compatibility Re-exports

This document tracks backward compatibility re-exports in `__init__.py` files throughout the codebase.

## Purpose

Re-exports in `__init__.py` files allow users to import items from higher-level package namespaces rather than having to know the exact internal module structure. This provides:
- Cleaner, more intuitive imports
- Protection against internal reorganization
- Clear public API definition

## Current Re-exports

### `/src/empo/__init__.py`
Main package-level exports providing the public API:

```python
from empo.world_model import WorldModel
from empo.possible_goal import PossibleGoal, PossibleGoalGenerator, PossibleGoalSampler
from empo.human_policy_prior import HumanPolicyPrior, TabularHumanPolicyPrior
from empo.robot_policy import RobotPolicy
from empo.backward_induction import compute_human_policy_prior, TabularRobotPolicy
from empo.util.memory_monitor import MemoryMonitor, check_memory
```

**Usage:**
```python
from empo import WorldModel, MemoryMonitor  # Instead of from empo.world_model, from empo.util.memory_monitor
```

### `/src/empo/util/__init__.py`
Utility module exports:

```python
from .memory_monitor import MemoryMonitor, check_memory
```

**Usage:**
```python
from empo.util import MemoryMonitor  # Instead of from empo.util.memory_monitor
```

### `/src/empo/backward_induction/__init__.py`
Backward induction exports:

```python
from .phase1 import compute_human_policy_prior, TabularHumanPolicyPrior
from .phase2 import compute_robot_policy, TabularRobotPolicy
```

**Note:** Full list in the actual file - provides high-level functions without exposing internal module structure.

### `/src/empo/learning_based/__init__.py`
Neural network-based implementations:

Re-exports base classes from `phase1/` subdirectory:
```python
from .phase1.state_encoder import BaseStateEncoder
from .phase1.goal_encoder import BaseGoalEncoder
from .phase1.q_network import BaseQNetwork
from .phase1.policy_prior_network import BasePolicyPriorNetwork
from .phase1.neural_policy_prior import BaseNeuralHumanPolicyPrior
from .phase1.replay_buffer import ReplayBuffer
from .phase1.trainer import Trainer
```

**Usage:**
```python
from empo.learning_based import BaseStateEncoder  # Instead of from empo.learning_based.phase1.state_encoder
```

### `/src/empo/learning_based/phase1/__init__.py`
Phase 1 (human policy prior) exports - mirrors parent package structure.

### `/src/empo/learning_based/phase2/__init__.py`
Phase 2 (robot policy) exports:

Re-exports from various submodules including:
- Base network classes
- Lookup table implementations  
- Network factory functions
- Curiosity-driven exploration modules

**Note:** This is a comprehensive re-export providing a unified interface to Phase 2 components.

### `/src/empo/learning_based/multigrid/__init__.py`
MultiGrid-specific implementations:

Re-exports constants, feature extraction utilities, and neural networks:
```python
from .constants import OBJECT_TYPE_TO_CHANNEL, ...
from .feature_extraction import extract_agent_features, ...
from .state_encoder import MultiGridStateEncoder
from .goal_encoder import MultiGridGoalEncoder
from .phase1.q_network import MultiGridQNetwork
# ... and more
```

**Usage:**
```python
from empo.learning_based.multigrid import MultiGridStateEncoder  # Clean import path
```

### `/src/empo/learning_based/transport/__init__.py`
Transport environment-specific implementations - similar pattern to multigrid.

## Maintenance Guidelines

When reorganizing code:

1. **Keep existing re-exports** unless deprecating functionality
2. **Add new re-exports** to maintain clean import paths
3. **Update `__all__`** to document public API
4. **Test imports** after reorganization to ensure backward compatibility
5. **Document changes** in this file

## Testing Imports

To verify backward compatibility after changes:
```bash
python -c "from empo import WorldModel, MemoryMonitor, compute_human_policy_prior"
python -c "from empo.learning_based import BaseStateEncoder, BaseQNetwork"
python -c "from empo.learning_based.multigrid import MultiGridStateEncoder"
```

## Recent Fixes

### 2026-01-07: Duplicate memory_monitor.py Removed
- **Issue:** `memory_monitor.py` existed in both `src/empo/` and `src/empo/util/`
- **Resolution:** Removed `src/empo/memory_monitor.py`, kept `src/empo/util/memory_monitor.py`
- **Impact:** No code changes needed - all imports already used `empo.util.memory_monitor`
- **Backward compatibility:** Maintained via re-export in `src/empo/__init__.py`:
  ```python
  from empo.util.memory_monitor import MemoryMonitor, check_memory
  ```
  This allows users to continue using `from empo import MemoryMonitor`.

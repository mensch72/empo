# EMPO Copilot Instructions

## Project Overview

EMPO ("Human Empowerment AI Agents") implements a framework for computing AI policies that softly maximize aggregate human power, based on [this theoretical paper](https://arxiv.org/html/2508.00159v2). This is **model-based planning**, not standard RL—we solve equations that define policies, not optimize utility.

## Critical Conceptual Distinctions

**This is NOT standard RL:**
- `beta_h`, `beta_r`, `gamma_h`, `gamma_r` are **theory parameters**, not training hyperparameters
- "Rewards" are goal-achievement signals (0/1) or intrinsic power metrics—never environment scores
- Policies are **computed/approximated**, not "learned" or "optimized"
- Training happens in the robot's world model, not the real world

**Two-Phase Architecture:**
1. **Phase 1**: Compute human policy priors (goal-conditioned Boltzmann policies via backward induction)
2. **Phase 2**: Compute robot policy simultaneously with power metric (mutual dependency loop, equations 4-9)

## Code Structure

```
src/empo/                      # Core framework
├── world_model.py             # WorldModel base: get_state(), set_state(), transition_probabilities()
├── possible_goal.py           # PossibleGoal, PossibleGoalGenerator, PossibleGoalSampler
├── backward_induction.py      # compute_human_policy_prior() - Phase 1
├── human_policy_prior.py      # TabularHumanPolicyPrior
└── nn_based/                  # Neural network implementations
    ├── phase2/                # Base classes for Phase 2 networks
    └── multigrid/phase2/      # MultiGrid-specific Phase 2 implementations

src/envs/                      # Environment configurations
vendor/multigrid/              # Extended MultiGrid with state management
vendor/ai_transport/           # Transport environment
```

## Key Patterns

### WorldModel Interface
All environments must implement `WorldModel` (extends `gym.Env`):
```python
def get_state(self) -> Hashable:     # Complete, hashable state
def set_state(self, state) -> None:  # Restore to exact state
def transition_probabilities(self, state, actions) -> List[Tuple[float, State]]  # Exact transitions
```

### Goal Definition
Goals return 0/1 reward via `is_achieved(state)` and must be hashable:
```python
class MyGoal(PossibleGoal):
    def is_achieved(self, state) -> int:
        return 1 if condition_met(state) else 0
    def __hash__(self): ...
    def __eq__(self, other): ...
```

### Encoder Synchronization
When modifying `vendor/multigrid/gym_multigrid/multigrid.py` (object types, agent attributes), update:
- `src/empo/nn_based/neural_policy_prior.py` (OBJECT_TYPE_TO_CHANNEL)
- `docs/ENCODER_ARCHITECTURE.md`

## Terminology (Follow Precisely)

Precise terminology prevents confusion in this codebase:
- **"phase"**: Use ONLY for "Phase 1" (human policy prior) vs "Phase 2" (robot policy). Never for training phases/stages.
- **"training_step"** vs **"env_step"**: Never use bare "step". Use `training_step` for gradient updates, `env_step` for environment transitions.
- **"world_model"** preferred over "env": Emphasizes prediction of `transition_probabilities()`, not just simulation.
- **"compute/approximate"** not "learn/optimize": Policies are solutions to equations, not optimized objectives.

## Development Workflow

### Setup (Docker recommended)
```bash
make up          # Start dev environment (auto-detects GPU)
make shell       # Enter container
make test        # Run tests
```

### Running Examples (IMPORTANT)
Always use `PYTHONPATH` to avoid import errors:
```bash
# Outside Docker container:
PYTHONPATH=src:vendor/multigrid:vendor/ai_transport python examples/human_policy_prior_example.py

# Inside Docker container (make shell):
python examples/human_policy_prior_example.py
```

### Tests
```bash
make test                           # All tests (inside container)
PYTHONPATH=src:vendor/multigrid:vendor/ai_transport python -m pytest tests/test_phase2.py  # Outside container
```

## Phase 2 Network Warm-up

Phase 2 training uses staged warm-up to break mutual dependencies—see [docs/WARMUP_DESIGN.md](docs/WARMUP_DESIGN.md) for rationale.

During warm-up: `beta_r=0` (uniform random robot). After warm-up: `beta_r` ramps to nominal.

## Common Pitfalls

- **Don't confuse epsilon (exploration hyperparameter) with beta (theory parameter)**
- **State must be hashable**—use tuples, not lists/arrays
- **Goals are hypothetical**—robot considers ALL possible goals, not "actual" human goals
- **Parallel DAG computation** uses worker processes with pickled environments—ensure serializability
- **Shared encoders** in Phase 2 networks cache raw tensors (not NN outputs) for gradient flow

## Important Workflow Notes

- **NEVER delete code unrelated to the current task.** This has caused problems requiring recovery from git history. When editing files, preserve all unrelated functions, classes, and imports.
- **Remind the user to make intermediate commits** after completing each logical unit of work. This makes it easier to revert mistakes and track progress.
- Don't use `sed` or other text-replacement tools that change files directly on disk, since this will conflict with the VS code editor's in-memory state. Instead, make changes through your built-in editing capability.
- When implementing changes, ensure that all related files are updated consistently to avoid discrepancies in functionality or documentation.
- When adding new features or modifying existing ones, make sure to do so for all execution modes (sync and async, with and without encoders, with neural networks and with lookup tables) unless the feature is explicitly meant for specific modes only, and update the relevant documentation files to reflect these changes accurately.

## Key Files for Understanding

- [README.md](README.md) - Theory explanation (read "What this is and what this is NOT")
- [docs/API.md](docs/API.md) - API reference
- [docs/ENCODER_ARCHITECTURE.md](docs/ENCODER_ARCHITECTURE.md) - Neural encoder design
- [docs/WARMUP_DESIGN.md](docs/WARMUP_DESIGN.md) - Phase 2 warm-up rationale
- [examples/human_policy_prior_example.py](examples/human_policy_prior_example.py) - Phase 1 usage
- [examples/phase2_robot_policy_demo.py](examples/phase2_robot_policy_demo.py) - Phase 2 usage

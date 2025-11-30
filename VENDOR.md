# Vendored Dependencies

This document describes the vendored (bundled) dependencies in this repository and how to manage them.

## Multigrid

We vendor the Multigrid library source code to allow local modifications without rebuilding the Docker container.

### Location

The Multigrid source code is located in:
```
vendor/multigrid/
```

The main Python package is at:
```
vendor/multigrid/gym_multigrid/
```

### How It Works

Unlike traditional pip installation, the vendored Multigrid is imported via **PYTHONPATH**:

- The Dockerfile sets: `PYTHONPATH=/workspace:/workspace/vendor/multigrid`
- This allows Python to import `gym_multigrid` directly from the vendored source
- **No container rebuild needed** when you modify the source code
- Changes are immediately reflected when you restart Python or re-import the module

## EMPO-Specific Modifications

The vendored Multigrid has been extensively modified for the EMPO framework. Key changes:

### 1. WorldModel Integration (`multigrid.py`)

The `MultiGridEnv` class now inherits from `empo.WorldModel` providing:

- **`get_state()`**: Returns compact hashable state tuple containing:
  - Step count
  - Agent states (position, direction, terminated, started, paused, carrying)
  - Mobile objects (blocks, rocks) with positions
  - Mutable objects (doors, boxes, magic walls) with their mutable state

- **`set_state(state)`**: Restores environment to any previously observed state

- **`transition_probabilities(state, actions)`**: Computes exact transition probabilities using conflict block partitioning algorithm (see `PROBABILISTIC_TRANSITIONS.md`)

### 2. New Object Types

| Object | File Location | Description |
|--------|---------------|-------------|
| **Block** | `multigrid.py:589` | Pushable by any agent |
| **Rock** | `multigrid.py:605` | Pushable only by authorized agents |
| **UnsteadyGround** | `multigrid.py:262` | Agents may stumble with configurable probability |
| **MagicWall** | `multigrid.py:331` | Probabilistic entry from one direction |

### 3. New Agent Attributes

```python
class Agent(WorldObj):
    def __init__(self, ..., can_enter_magic_walls=False, can_push_rocks=False):
        ...
        self.can_enter_magic_walls = can_enter_magic_walls  # Can attempt magic wall entry
        self.can_push_rocks = can_push_rocks  # Can push rocks (immutable)
        self.on_unsteady_ground = False  # Derived from position
```

### 4. Map-Based Environment Specification

New map parsing system allows ASCII-based environment definition:

```python
MAP = """
WeWeWeWeWe
We..Ay..We
WeRo....We
We..Ae..We
WeWeWeWeWe
"""

class MyEnv(MultiGridEnv):
    def __init__(self):
        super().__init__(
            map=MAP,
            max_steps=100,
            can_push_rocks='e'  # Grey agents can push rocks
        )
```

**Cell codes:**
- `..` : Empty cell
- `We` : Grey wall
- `Ay/Ar/Ae` : Agent (yellow/red/grey)
- `Ro` : Rock
- `Bl` : Block
- `Un` : Unsteady ground
- `Mn/Ms/Me/Mw` : Magic wall (north/south/east/west entry side)

### 5. Optimized Transition Computation

The `transition_probabilities()` method uses a conflict block partitioning algorithm:

1. **Identify active agents** - agents that will actually act
2. **Early exit** for deterministic cases (≤1 agent or all rotations)
3. **Partition into conflict blocks** - agents competing for same resource
4. **Cartesian product** of block winners instead of k! permutations
5. **Handle stochastic elements** - unsteady ground and magic walls

See `vendor/multigrid/PROBABILISTIC_TRANSITIONS.md` for detailed documentation.

### 6. Object Caching

Added `_build_object_cache()` for O(1) state computation:
- `_mobile_objects`: List of (position, object) for blocks/rocks
- `_mutable_objects`: List of (position, object) for doors/boxes/magic walls

### 7. Helper Methods

- `_categorize_agents(actions)`: Categorize agents into normal/unsteady/magic-wall groups
- `_execute_single_agent_action(agent_idx, action, rewards)`: Execute single action
- `_process_unsteady_forward_agents(...)`: Handle stumbling stochasticity
- `_identify_conflict_blocks(actions, active_agents)`: Partition agents by contested resource
- `_compute_successor_state(state, actions, ordering)`: Deterministic successor computation
- `_compute_successor_state_with_unsteady(...)`: Successor with stochastic outcomes

## Making Local Modifications

You can freely modify the source code in `vendor/multigrid/`. Changes take effect immediately without rebuilding the container.

**Common files to modify:**
- `vendor/multigrid/gym_multigrid/envs/` - Environment definitions
- `vendor/multigrid/gym_multigrid/multigrid.py` - Core multigrid classes
- `vendor/multigrid/gym_multigrid/rendering.py` - Rendering utilities

**Workflow:**
1. Edit files in `vendor/multigrid/gym_multigrid/`
2. Restart your Python interpreter or re-import the module
3. Changes are immediately available (no rebuild required)

### For Local Development (Outside Docker)

If you want to use the vendored Multigrid outside of Docker:

```bash
# Option 1: Set PYTHONPATH
export PYTHONPATH=/path/to/empo:/path/to/empo/vendor/multigrid:$PYTHONPATH

# Option 2: Install in editable mode
pip install -e ./vendor/multigrid
```

### Pulling Upstream Updates

To pull the latest changes from the upstream Multigrid repository:

```bash
# Pull updates from upstream
git subtree pull --prefix=vendor/multigrid https://github.com/ArnaudFickinger/gym-multigrid.git master --squash
```

**Important Notes:**
- The `--squash` flag combines all upstream commits into a single commit
- If you have local modifications, there may be merge conflicts that need to be resolved
- Always test after pulling upstream updates to ensure compatibility
- No container rebuild needed after pulling updates

### Pushing Local Changes Upstream (Optional)

If you want to contribute your local changes back to the Multigrid project:

1. Create a separate branch with your changes:
   ```bash
   git subtree split --prefix=vendor/multigrid -b multigrid-changes
   ```

2. Push to your fork of Multigrid:
   ```bash
   git push your-multigrid-fork multigrid-changes:feature-branch
   ```

3. Create a pull request on the upstream Multigrid repository

### Version Information

To check the current vendored version, look at the source code or check the git history:
```bash
git log --oneline vendor/multigrid/ | head -5
```

### Troubleshooting

**Issue: Import errors after modifying the source**
- Restart your Python interpreter or container
- Verify PYTHONPATH is set: `echo $PYTHONPATH`
- Check that `vendor/multigrid` is in PYTHONPATH

**Issue: Changes not reflected**
- Ensure you restart Python or re-import the module
- Python caches imported modules - use `importlib.reload()` if needed
- For interactive development, consider using `%load_ext autoreload` in Jupyter

**Issue: Merge conflicts when pulling updates**
- Resolve conflicts in `vendor/multigrid/` like normal git conflicts
- Test thoroughly after resolution
- Consider stashing your changes, pulling updates, then reapplying

**Issue: Want to revert to a specific upstream version**
```bash
# Check available tags/commits
git ls-remote https://github.com/ArnaudFickinger/gym-multigrid.git

# Pull a specific commit
git subtree pull --prefix=vendor/multigrid https://github.com/ArnaudFickinger/gym-multigrid.git <commit-hash> --squash
```

## Why Use PYTHONPATH Instead of pip install?

For actively developed dependencies like Multigrid, using PYTHONPATH provides several advantages:

- **No rebuild required**: Edit code and immediately see changes
- **Fast iteration**: Perfect for making extensive modifications
- **Simple workflow**: Just edit files and restart Python
- **Docker-friendly**: Changes persist across container restarts via bind mount
- **Same as development**: Identical behavior in Docker and local development

## Alternative: pip install -e (Editable Install)

If you prefer traditional editable installs, you can modify the Dockerfile:

```dockerfile
# Copy vendored dependencies
COPY vendor /tmp/vendor

# Install in editable mode
RUN pip install -e /tmp/vendor/multigrid
```

However, this requires:
- Rebuilding the container when switching between modified/unmodified versions
- More complex workflow
- Not recommended for extensive changes

## Why Use Git Subtree?

Git subtree was chosen over alternatives for these reasons:

- **No submodule complexity**: No need for `git submodule update --init`
- **Full source in repo**: All code is in the main repository
- **Easy for collaborators**: No special git commands needed for basic usage
- **Flexible modifications**: Can modify vendored code freely
- **Upstream sync**: Can pull updates from upstream when needed

## Upstream Repository

- **Repository**: https://github.com/ArnaudFickinger/gym-multigrid
- **License**: Apache 2.0 (see vendor/multigrid/LICENSE)
- **Description**: Multi-agent gridworld environments for reinforcement learning

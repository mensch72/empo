# Archival Mechanism Summary

## What Was Added

Added automatic value function archival during backward induction to reduce memory usage when disk slicing is enabled.

### Key Components

1. **`detect_archivable_levels()`** in `helpers.py`:
   - Determines which levels can be safely archived based on dependency analysis
   - A level k can be archived when no future computations will need it
   - Optional `quiet` parameter to suppress debug output

2. **`archive_value_slices()`** in `helpers.py`:
   - Archives value function slices to disk using pickle
   - Supports both `List[List[Dict]]` (vh_values, vr_values) and `List[Dict]` (vhe_values) structures
   - Optionally clears archived data from memory to free RAM
   - **Output** (when `quiet=False`): 
     - Prints archival summary: states archived, level count, filename, and memory status

3. **Phase 1 Integration** (`phase1.py`):
   - Automatically archives `Vh_values` when `use_disk_slicing=True`
   - Archives happen after each level completes
   - Files saved to `disk_dag.output_dir / "vh_values.pkl"`
   - Respects the `quiet` parameter from `compute_human_policy_prior()`

4. **Phase 2 Status** (`phase2.py`):
   - Infrastructure ready but **NOT YET ACTIVE**
   - Phase 2 doesn't support `use_disk_slicing` parameter yet
   - TODO comments show where archival will hook in once disk slicing is added to Phase 2

## Current State

### ✅ Working in Phase 1
- `examples/phase2/phase2_backward_induction.py` uses `use_disk_slicing=True` in Phase 1
- Archival happens automatically during Phase 1 computation
- Output visible when script runs (quiet defaults to False)

### ❌ Not Yet in Phase 2
- Phase 2 doesn't have `use_disk_slicing` parameter
- Phase 2 doesn't have `disk_dag` initialization
- Archival code is commented out with TODO markers

## How to See Archival in Action

Run the example script:

```bash
PYTHONPATH=src:vendor/multigrid:vendor/ai_transport:multigrid_worlds \
    python examples/phase2/phase2_backward_induction.py --steps 5
```

Look for output during Phase 1:
```
    Archived 127 states from 3 level(s) to vh_values.pkl
    Archived 89 states from 2 level(s) to vh_values.pkl
```

To suppress archival messages, add `quiet=True` to `compute_human_policy_prior()`.

## Algorithm Details

### When Can a Level Be Archived?

Backward induction processes levels in **descending** order (high→low). A level k can be archived when:

```
max_successor_levels[l] < k  for all l <= current_level_value
```

This means:
- All future computations only need successors at levels < k
- Level k's data will never be read again
- Safe to archive and free from memory

### Example

Processing levels: [5, 4, 3, 2, 1, 0]

```
max_successor_levels = {
    5: 4,   # Level 5 states have successors up to level 4
    4: 3,
    3: 2,
    2: 1,
    1: 0,
    0: -1   # Terminal states have no successors
}
```

**At current_level=3:**
- Future levels: [3, 2, 1, 0]
- Max future successor: max(2, 1, 0, -1) = 2
- Archivable: levels > 2 → **[5, 4, 3]**
- Why include 3? We just finished it, won't need it again

**At current_level=1:**
- Future levels: [1, 0]
- Max future successor: max(0, -1) = 0
- Archivable: levels > 0 → **[5, 4, 3, 2, 1]**

## File Format

Archived files are pickle dumps containing:
```python
{
    'level_value': int,           # The level being archived
    'state_indices': List[int],   # Which states are at this level
    'data': List[...]             # The actual value data for these states
}
```

Multiple levels are appended to the same file sequentially.

## Memory Management

With `return_values=False`, archived slices are replaced with empty structures:
- `List[List[Dict]]` → `[{}, {}, ...]`
- `List[Dict]` → `{}`

This frees the memory while maintaining the array structure for any remaining reads.

## Testing

Run the test script:
```bash
PYTHONPATH=src:vendor/multigrid:vendor/ai_transport:multigrid_worlds \
    python test_archival_logging.py
```

Expected output shows:
- Level detection logic working correctly
- Logging messages appearing
- Files being created with archived data

## Future Work

To enable Phase 2 archival:
1. Add `use_disk_slicing` parameter to `compute_robot_policy()`
2. Add `disk_dag` initialization like Phase 1
3. Uncomment archival code in phase2.py (already written, just needs disk_dag)
4. Test with both Vh_values and Vr_values archival

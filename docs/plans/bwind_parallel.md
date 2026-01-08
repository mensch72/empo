SEE https://gemini.google.com/share/f574f2ad80bc for a much better approach: no prepopulated lists, no manager, just batched returns that get pickled and then flattened into a merged list.

MEMORY improvements: everything float16, attainments int8, action index int8, short hashes?

# Backward Induction Parallelization Plan

**Status:** Planning (Implementation Pending)  
**Author:** GitHub Copilot  
**Date:** 2026-01-08

## 1. Overview

This document outlines the design for parallelizing backward induction computation across dependency levels in both Phase 1 (human policy prior) and Phase 2 (robot policy). Both phases currently have broken parallel implementations.

### 1.1 Current State

- Sequential backward induction works correctly in both phases
- Parallel implementations exist in both `phase1.py` and `phase2.py` but are broken
- **Critical lesson:** Separate parallel code paths lead to divergence and breakage

### 1.2 Goal

Fix and improve parallel computation in both Phase 1 and Phase 2 by:
- Processing states within each dependency level in parallel across multiple workers
- Integrating parallel mode into sequential code (no code divergence)
- Sharing DAG data, attainment cache, and value functions efficiently
- Efficiently merging worker results back into main value functions
- Supporting incremental archival of completed levels

### 1.3 Scope

This plan applies to:
- `src/empo/backward_induction/phase1.py` (human policy prior computation)
- `src/empo/backward_induction/phase2.py` (robot policy computation)

Both files should be refactored using the same principles to avoid code divergence between phases.

---

## 2. Design Principles

### 2.1 Level-Based Parallelization

**Core insight:** States within the same dependency level can be computed independently in parallel since they don't depend on each other—only on states in higher (i.e., temporally later) levels. In many types of environments, including the multigrid worlds, the simplest level function is the time step, as all quantities from one time step only depend on quantities from the same time step or the very next time step, but no later time steps.

**Sequential structure to preserve:**
- Main computation loop iterates over dependency levels in reverse order (highest to lowest)
- Each level is fully completed before moving to the next
- Value functions are updated after each level completes
- Archival checks occur after level completion

**Parallel enhancement:**
- Within each level, partition states into batches for parallel processing
- Each worker processes its batch independently
- Workers share read access to current level's DAG slice and higher levels' value functions
- Main process merges results after all workers complete.

### 2.2 No Code Divergence

**Critical requirement:** The parallel option MUST be integrated into the existing sequential code, NOT as a separate code branch.

**Anti-pattern to avoid (what's broken in current code):**
- Separate parallel and sequential code paths that diverge over time
- Different logic for parallel vs sequential modes
- Code that becomes untested in one mode or the other
- Parallel code in phase1.py and phase2.py that's broken while sequential works

**Required pattern:**
- Single unified code path with minimal `if parallel` branching
- Same helper functions used in both modes
- Same data structures and algorithms
- Parallel mode is a "batching wrapper" around sequential logic

### 2.3 Fork-Based Multiprocessing

**Context:** Use standard Python multiprocessing patterns (fork context on Linux).

**Implementation considerations:**
- Use `multiprocessing` with fork context (Linux default)
- Use `ProcessPoolExecutor` for worker management
- Consider executor lifecycle: per-level vs persistent
- Consider data sharing: module globals vs shared memory vs pickling

---

## 3. Algorithm Structure

### 3.1 Sequential Code Structure (Current)

The current sequential implementation already has the structure needed:

**Level generation:**
- Use `level_fct` to assign levels to states
- Use `compute_dependency_levels_fast` or `compute_dependency_levels_general` to compute dependency ordering
- Store levels as list of state-index lists

**Main loop:**
- Iterate over levels in reverse topological order
- Process each state in the level
- Update value functions after processing each state
- Check for archival opportunities when transitioning between levels

**Value function updates:**
- Read successor values from later levels
- Compute new values for current state
- Write values back to value function arrays

### 3.2 Parallel Enhancement Design

**High-level structure:**

```
Initialize shared data structures
Compute dependency levels using level_fct
Initialize value functions

For each level in reverse topological order:
    # Determine whether to batch this level
    if parallel and len(level) > threshold:
        batches = split_into_batches(level, n_workers)
    else:
        batches = [level]  # Single batch = sequential
    
    # Process batches (same code path for sequential and parallel)
    if len(batches) > 1:
        # Multiple batches: use worker pool
        results = process_batches_parallel(batches)
    else:
        # Single batch: process inline
        results = process_batch_sequential(batches[0])
    
    # Merge results (same code path for sequential and parallel)
    merge_results_into_value_functions(results)
    
    # Archival (same code path for sequential and parallel)
    check_and_archive_completed_levels()
```

**Key insight:** Sequential is just "parallel with 1 batch processed inline"

**Key design decisions:**

**Decision 1: When to parallelize**
- Add configurable threshold (e.g., states_per_level > 2 * num_workers)
- Small levels process sequentially to avoid parallelization overhead
- Current phase2.py has: `if len(level) <= num_workers: sequential else: parallel`

**Decision 2: Worker pool lifecycle**
- Consider two options: (A) recreate per level, (B) persistent with explicit data updates
- Option A: Fresh fork per level ensures workers see latest values
- Option B: Persistent pool with explicit data passing (more complex but potentially faster)
- Recommendation: Start with Option A (simpler), profile and optimize later

**Decision 3: Batch assignment**
- Use helper function to split states into batches
- Reuse `split_into_batches` from helpers module
- Distribute states evenly across workers
- Each batch is an interval of consecutive state indices

---

## 4. Shared Memory Management

### 4.1 Data to Share

**Read-only data (shared across all levels):**
- States list
- Theory parameters (beta_h, gamma_h, etc.)
- Configuration parameters

**Level-specific data (updated per level):**
- Value functions:
  - Phase 1: Vh_values (human value per goal)
  - Phase 2: Vh_values (expected achievement) + Vr_values (robot value)
- Attainment cache slices (goal achievement arrays)
- Current DAG slice (if using disk-based slicing)

### 4.2 Sharing Strategy

**Current implementation approach (smart COW avoidance):**

The current code uses module-level globals with a clever nested list structure to avoid copy-on-write overhead:

**For attainment cache / slice cache:**
- Pre-allocate outer list: `cache = [[] for _ in range(num_states)]`
- Each inner list is empty before fork
- Workers only append to their assigned state's inner list
- **Key insight:** Outer list is never modified (no COW trigger)
- Inner lists get copied on first write, but they're empty so copy is O(1)
- This avoids copying large data structures!

**For read-only data (DAG, value functions):**
- Stored in module-level globals
- Workers only read, never write
- COW is triggered by Python reference counting, but data is truly read-only so not actually copied in practice

**For SharedMemory (optional):**
- `SharedDAG` class available for very large DAGs
- Stores pickled states/transitions in shared memory blocks
- More complex but can help for massive state spaces
- Currently exists but may need debugging

**Recommendation:** Keep the clever nested-list approach for cache, it's efficient and works well

### 4.3 Initialization Pattern

**Before each level's parallel processing:**

```
Call _init_shared_data with:
    - States (via SharedDAG if use_shared_memory=True)
    - Transitions (via SharedDAG if use_shared_memory=True)  
    - Current value functions (via module globals)
    - Theory parameters (via module globals)
    - Current attainment cache slices (via module globals)
    - Number of action profiles (for cache creation)
```

**Workers access shared data:**

```
In worker initialization:
    - Attach to SharedDAG if present
    - Otherwise use module-level globals
    - Access value functions from globals
    - Access attainment cache from globals
```

---

## 5. Worker Batch Processing

### 5.1 Batch Processing Function

**Function signature pattern:**

```
def _process_state_batch(state_indices: List[int]) -> BatchResults:
    """Process a batch of states from the current level.
    
    Returns:
        - value_results: Dict mapping state_index to computed values
        - cache_slice: SliceCache for this batch
        - batch_metadata: Timing, stats, etc.
    """
```

**Implementation pattern:**

**Worker setup:**
- Retrieve shared DAG from SharedMemory or globals
- Retrieve current value functions from globals
- Retrieve current cache slices from globals
- Create local cache slice for this batch

**State processing loop:**
- For each state index in batch:
  - Call unified `_process_single_state` helper
  - Store results in local value dict
  - Store cache entries in local cache slice

**Return results:**
- Return local value dict (sparse, only batch states)
- Return local cache slice (for merging into global cache)
- Return timing statistics

### 5.2 Unified State Processing Helper

**Critical requirement:** Use the same `_process_single_state` function in sequential and parallel modes.

**Helper function characteristics:**
- Takes state and all dependencies as explicit parameters
- No global state access inside helper
- Returns computed values (doesn't modify globals)
- Accepts optional slice_cache parameter for writing

**Advantages:**
- Single implementation to test and maintain
- Guaranteed consistency between modes
- Easier debugging and profiling

---

## 6. Result Merging

### 6.1 Value Function Updates

**No merging needed!**

With the clever nested-list approach from section 4.2, workers write directly to the pre-allocated value function lists:

**How it works:**
- Value functions pre-allocated: `Vh_values = [[{} for _ in agents] for _ in states]`
- Workers write directly to their assigned states' dicts
- No explicit merge step required
- Main process just waits for all workers to complete

**What workers return:**
- Workers can return minimal metadata (timing, stats)
- No need to return computed values since they're already written
- Or return slice_id for cache management

**Synchronization:**
- Use `as_completed` or similar to wait for all workers
- All writes are complete when all workers finish
- Check memory after all workers complete if monitoring enabled

### 6.2 Attainment Cache Updates

**No merging needed here either!**

Just like value functions, the attainment cache uses the clever nested-list approach:

**How it works:**
- Cache pre-allocated: `cache = [[] for _ in range(num_states)]`
- Workers append directly to their assigned states' inner lists
- No explicit merge or storage step required
- Cache is already populated when workers finish

**What this means:**
- No `make_slice_id` needed
- No `store_slice` calls needed
- Workers just write and return
- Main process has complete cache when all workers done

**Phase 2 cache reuse:**
- Cache is saved to disk along with DAG slices in Phase 1
- Phase 2 loads the cache from disk (via `disk_dag.load_cache_slice()`)
- Only the human policy prior remains in memory between phases
- Cache is reloaded level-by-level as needed in Phase 2

---

## 7. Archival Integration

### 7.1 Archival Trigger Points

**When to check for archival:**
- After completing each level
- Before moving to next (earlier) level
- At the end of all levels (final cleanup)

**Archival logic pattern:**

```
After level completion:
    current_level_value = level_fct(first_state_in_current_level)
    
    Compute which levels are archivable:
        archivable_levels = detect_archivable_levels(
            current_level_value, 
            max_successor_levels
        )
    
    Filter to new levels only:
        new_archivable = [lvl for lvl in archivable 
                         if lvl not in archived_levels]
    
    If new_archivable:
        Archive value slices to disk
        Mark as archived: archived_levels.update(new_archivable)
```

### 7.2 Archival Prerequisites

**Data structures needed:**

**max_successor_levels:**
- Dict mapping level value to maximum successor level
- Computed once at start via `compute_dependency_levels_fast`
- Used to determine when level is no longer needed

**archived_levels set:**
- Track which levels have already been archived
- Prevents duplicate archival
- Persists across entire computation

**level_values_list:**
- List of level values in dependency order
- Maps level_idx to level_value
- Needed for archival detection

### 7.3 Parallel Archival Considerations

**Timing:**
- Archive between levels, not during level processing
- All workers must complete before archival
- Archival happens in main process only

**Consistency:**
- Ensure all value updates are merged before archival
- Check that no workers are still processing
- Verify slice caches are stored if needed

---

## 8. Memory Management

### 8.1 Memory Monitoring

**Current implementation in both phases:**

**MemoryMonitor usage:**
- Create monitor with thresholds
- Check after each batch completion
- Use `force=True` for per-batch checks (bypass interval)
- Pause or abort if memory critical

**Check points:**
- Before collecting worker result
- After merging worker result
- After level completion

### 8.2 Memory Optimization Strategies

**Shared memory for DAG:**
- Use SharedDAG when `use_shared_memory=True`
- Reduces memory from O(workers * dag_size) to O(dag_size)
- Essential for large state spaces

**Sliced cache strategy:**
- Keep slices separate (don't merge)
- Archive slices to disk if too large
- Use DiskBasedDAG for cache storage if needed

**Value function slicing:**
- Archive completed levels to disk
- Free archived levels from memory
- Re-load from disk if needed later (rare)

**Worker cleanup:**
- Explicitly cleanup executor after each level
- Call `cleanup_shared_dag()` when done
- Free shared memory blocks

---

## 9. Implementation Phases

### 9.1 Phase A: Audit and Refactor Sequential Code

**Objective:** Prepare sequential code for parallelization without adding parallelization yet.

**Tasks:**
- Extract state-processing logic into standalone `_process_single_state` function
- Ensure function takes all dependencies as explicit parameters (no globals)
- Ensure function returns results (doesn't modify globals directly)
- Verify sequential code works through this refactored function
- Add tests that verify correctness of refactored sequential code

**Success criteria:**
- Sequential code passes all existing tests
- `_process_single_state` is pure (no side effects)
- No global state access within core processing

### 9.2 Phase B: Add Batch Processing (Sequential Mode)

**Objective:** Add batching infrastructure in sequential mode only.

**Tasks:**
- Add batch creation: `batches = [level]` (single batch)
- Add batch processing loop that calls `_process_single_state`
- Add result collection and merging
- Verify sequential still works identically

**Success criteria:**
- Sequential code still passes all tests
- Batching infrastructure in place but not yet used for parallelization
- No performance regression

### 9.3 Phase C: Add Worker Processing Option

**Objective:** Add ability to process batches in workers.

**Tasks:**
- Add `parallel` parameter (default False)
- Add conditional: if parallel, create multiple batches
- Add worker batch processing function
- Add worker pool creation and management
- Keep parallel=False as default

**Success criteria:**
- Sequential mode (parallel=False) still works
- Parallel mode (parallel=True) executes without errors
- Small test cases produce identical results in both modes

### 9.4 Phase D: Optimization and Testing

**Objective:** Optimize parallel performance and verify correctness.

**Tasks:**
- Add shared memory for large data if needed
- Tune batch sizes and thresholds
- Add comprehensive parallel vs sequential tests
- Profile and optimize bottlenecks

**Success criteria:**
- All tests pass in both modes
- Parallel shows speedup for large problems
- Memory usage is reasonable

### 9.5 Phase E: Integration and Documentation

**Objective:** Document and deploy.

**Tasks:**
- Update API documentation
- Add usage examples
- Document performance characteristics
- Mark as ready for production use

**Success criteria:**
- Documentation complete
- Examples demonstrate usage
- CI tests run both modes

---

## 10. Testing Strategy

### 10.1 Correctness Tests

**Comparison tests:**
- Run same problem with parallel=False and parallel=True
- Compare final value functions (should match exactly)
- Compare final policies (should match exactly)
- Compare archived files (should match exactly)

**Edge cases:**
- Single state per level (no parallelization benefit)
- All states in one level (maximum parallelization)
- Levels with 1, 2, n_workers, n_workers+1 states
- Empty levels (terminal states)

### 10.2 Performance Tests

**Speedup measurement:**
- Measure wall-clock time for varying n_workers
- Expect near-linear speedup for large levels
- Expect overhead dominates for small levels

**Memory measurement:**
- Track peak memory usage
- Verify shared memory reduces memory vs fork overhead
- Verify archival reduces memory for completed levels

### 10.3 Stress Tests

**Large state spaces:**
- Test with disk-based slicing
- Test with archival enabled
- Test with memory monitoring

**Worker failures:**
- Test graceful handling of worker exceptions
- Verify cleanup happens on error
- Test keyboard interrupt handling

---

## 11. Configuration Parameters

### 11.1 New Parameters

**parallel: bool**
- Enable parallel processing
- Default: False (sequential mode)

**num_workers: Optional[int]**
- Number of worker processes
- Default: None (use cpu_count)

**use_shared_memory: bool**
- Use SharedDAG for states/transitions
- Default: True when parallel=True
- Recommended for large state spaces

**parallel_threshold: int**
- Minimum states per level to parallelize
- Default: 2 * num_workers
- Small levels run sequentially for efficiency

### 11.2 Existing Parameters (Reused)

**level_fct: Callable[[State], int]**
- Required for parallel mode
- Maps state to level value
- Used for dependency computation and archival

**archive_dir: Optional[str]**
- Directory for archived value slices
- Same as sequential mode
- Works with parallel archival

**memory monitoring parameters:**
- min_free_memory_fraction
- memory_check_interval  
- memory_pause_duration
- Same behavior as sequential

---

## 12. Open Questions

### 12.1 Batch Size Optimization

**Question:** How to determine optimal batch size?

**Options:**
- Fixed batches: split into exactly n_workers batches
- Dynamic batches: adjust based on level size
- Load balancing: use larger pool, more smaller batches

**Recommendation:** Start with fixed n_workers batches, optimize later if needed.

### 12.2 Cache Slice Management

**Question:** When to merge cache slices vs keep separate?

**Current pattern:** Keep separate (sliced cache approach)

**Alternative:** Merge if needed for sequential phase access

**Recommendation:** Keep separate initially, add merge option if profiling shows benefit.

### 12.3 Archival Granularity

**Question:** Archive per level or per level-group?

**Options:**
- Archive each level immediately when archivable
- Buffer multiple levels and archive in batches
- Archive only when memory pressure detected

**Recommendation:** Archive immediately when archivable, add buffering if I/O overhead is significant.

---

## 13. Avoiding Code Divergence

### 13.1 Root Cause of Current Breakage

**Problem:** Separate code paths for parallel vs sequential modes means:
- Sequential code gets maintained and tested
- Parallel code bitrotss as sequential evolves
- Bugs in parallel code go unnoticed until someone tries to use it
- Fixing bugs requires understanding two different implementations

### 13.2 Prevention Strategy

**Single source of truth:**
- One `process_state` function used in both modes
- One result-merging function used in both modes
- One archival function used in both modes
- Parallelization is only about:
  - Batching states
  - Distributing batches to workers
  - Collecting results

**Testability:**
- Every test should run in both parallel=False and parallel=True modes
- Results must be identical (within floating-point tolerance)
- Performance tests measure speedup, not correctness

**Code review checklist:**
- Does this change affect only batching/distribution logic?
- Or does it change the core algorithm?
- If core algorithm: ensure both modes use the same code
- If batching logic: document why parallel differs from sequential

### 13.3 Refactoring Existing Code

**Before adding parallelization:**
- Extract core state-processing logic into a standalone function
- Ensure this function has no global state dependencies
- Test this function in isolation
- Verify sequential code uses only this function

**Only then add parallelization:**
- Create batch-processing wrapper that calls core function
- Use batch-processing wrapper in parallel mode
- Keep sequential mode using the same core function

## 14. Relationship to Existing Code

### 13.1 Reusable Components

**From helpers.py:**
- `compute_dependency_levels_fast`
- `compute_dependency_levels_general`
- `split_into_batches`
- `detect_archivable_levels`
- `archive_value_slices`
- `make_slice_id`

**From shared_dag.py:**
- `SharedDAG` class
- `init_shared_dag`
- `attach_shared_dag`
- `get_shared_dag`
- `cleanup_shared_dag`

**From phase1.py / phase2.py (existing broken parallel code):**
- ProcessPoolExecutor usage pattern (general idea, but buggy implementation)
- Batch processing structure (concept exists, but has issues)
- Result merging approach (attempted but broken)
- Memory monitoring integration (this part actually works)

**Important:** Study the broken parallel code to understand what went wrong, then refactor both files using the same approach

### 14.2 Refactoring Needed in Both Files

**Apply to both phase1.py and phase2.py:**
- Extract unified `_process_single_state` helper (phase1: `_hpp_process_single_state`, phase2: `_rp_process_single_state`)
- Refactor `_process_state_batch` functions to use unified helper
- Integrate parallel branching into main computation loop
- Ensure identical code structure between both files where possible

**Phase-specific components:**
- Phase 1: Human policy prior computation logic
- Phase 2: Robot policy + expected achievement computation logic
- But parallelization infrastructure should be the same pattern in both

---

## 15. Success Criteria

### 15.1 Functional Requirements (Mandatory)

- **Identical results:** Parallel mode produces bit-identical results to sequential mode
- **No code divergence:** Core algorithm uses same code path in both modes
- **Shared helper:** Single `_process_single_state` function used by both modes
- **All tests pass:** Both modes pass all existing and new tests
- **Graceful errors:** Errors are handled gracefully, cleanup happens correctly

### 15.2 Performance Requirements (Goals)

- Overhead for small levels < 10% of sequential time
- Speedup for large levels > 0.5 * num_workers (modest target)
- Memory usage scales reasonably with workers
- No memory leaks during extended runs

### 15.3 Code Quality Requirements (Mandatory)

- **Minimal branching:** `if parallel` appears only for batching/worker management
- **No duplicate logic:** Algorithm logic never duplicated between modes
- **Type hints:** All functions have complete type hints
- **Documentation:** Clear docstrings explain parallel behavior
- **Testing:** CI runs tests in both modes automatically

### 15.4 Maintainability Requirements (Critical)

- **Single implementation:** Changing algorithm requires changing one function only
- **Clear separation:** Parallelization logic is isolated from algorithm logic
- **Reviewer checklist:** Code review template prevents divergence
- **No bitrot:** Parallel mode tested as frequently as sequential

---

## 16. Future Enhancements

### 16.1 Get It Working First

**Priority 1:** Correctness and no divergence
- Make parallel work correctly
- Keep code maintainable
- Don't optimize prematurely

### 16.2 Then Optimize If Needed

**Priority 2:** Performance improvements (only if profiling shows benefit)

**Advanced Scheduling:**
- Work-stealing or dynamic task queue
- Better load balancing for heterogeneous state costs

**Shared Memory Optimization:**
- Reduce pickling overhead for large data
- Use memory mapping for very large DAGs

**Adaptive Thresholding:**
- Auto-tune parallel threshold based on profiling
- Optimal performance without manual tuning

### 16.3 Don't Do (Probably)

**GPU Acceleration:**
- Backward induction is inherently sequential across levels
- Only within-level parallelism possible
- Likely not worth GPU overhead

**Distributed Computing:**
- Adds massive complexity
- State synchronization is hard
- Only needed for truly massive problems
- If state space is that large, use neural approximation instead

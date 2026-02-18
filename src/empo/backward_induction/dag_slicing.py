"""
Disk-based DAG Slicing for Memory-Efficient Backward Induction.

This module provides functionality to slice a DAG by timestep and save/load
slices to/from disk, dramatically reducing memory usage for large state spaces.

Key idea: During backward induction, we only need transitions FROM states at
timestep t TO states at timestep t+1. By organizing states by timestep and
saving slices to disk, we can load only the necessary slice at each iteration.

Memory reduction: ~10-20x for typical problems.
"""

import os
import gc
import pickle
import tempfile
import shutil
from typing import List, Dict, Tuple, Any, Optional
import numpy as np

# Type aliases
State = Any
ActionProfile = Tuple[int, ...]
TransitionData = Tuple[ActionProfile, List[float], List[int]]


def get_optimal_cache_dir(preferred_dir: Optional[str] = None, min_free_gb: float = 1.0) -> str:
    """
    Determine optimal cache directory with cross-platform tmpfs detection.
    
    Tries locations in order:
    1. User-specified preferred_dir
    2. /dev/shm (Linux tmpfs - RAM-based)
    3. /tmp (standard fallback)
    4. System temp directory
    
    Args:
        preferred_dir: User-specified directory (highest priority)
        min_free_gb: Minimum free space required in GB
    
    Returns:
        Best available cache directory path
    """
    import platform
    
    candidates = []
    
    # User preference always wins
    if preferred_dir:
        candidates.append(preferred_dir)
    
    # Platform-specific tmpfs locations
    system = platform.system()
    
    if system == "Linux":
        # Try /dev/shm first (RAM-based tmpfs)
        if os.path.exists("/dev/shm") and os.access("/dev/shm", os.W_OK):
            candidates.append("/dev/shm/empo_cache")
        # Fall back to /tmp
        candidates.append("/tmp/empo_cache")
    
    elif system == "Darwin":  # macOS
        # macOS doesn't have /dev/shm, use /tmp (which is actually tmpfs on modern macOS)
        candidates.append("/tmp/empo_cache")
    
    elif system == "Windows":
        # Windows: use TEMP directory
        temp_dir = os.environ.get('TEMP', os.environ.get('TMP', 'C:\\Temp'))
        candidates.append(os.path.join(temp_dir, 'empo_cache'))
    
    # Universal fallback
    candidates.append(os.path.join(tempfile.gettempdir(), "empo_cache"))
    
    # Check each candidate
    min_free_bytes = min_free_gb * (1024 ** 3)
    
    for path in candidates:
        try:
            # Create directory if needed
            os.makedirs(path, exist_ok=True)
            
            # Check if writable
            if not os.access(path, os.W_OK):
                continue
            
            # Check free space
            stat = os.statvfs(path) if hasattr(os, 'statvfs') else None
            if stat:
                free_bytes = stat.f_bavail * stat.f_frsize
                if free_bytes < min_free_bytes:
                    continue
            
            # Found a good location
            return path
            
        except (OSError, PermissionError):
            continue
    
    # Last resort: use tempfile
    return tempfile.mkdtemp(prefix="empo_cache_")



class DAGSlice:
    """
    A single timestep slice of the DAG.
    
    Contains:
    - state_indices: List of state indices in this timestep
    - transitions: Transition data for states in this timestep (indexed by local slice index)
    - timestep: The timestep number
    """
    
    def __init__(self, timestep: int, state_indices: List[int], 
                 transitions: List[List[TransitionData]]):
        self.timestep = timestep
        self.state_indices = state_indices  # Global indices
        self.transitions = transitions  # Indexed by position in state_indices
        
        # Create mapping from global index to local index within this slice
        self.global_to_local: Dict[int, int] = {
            global_idx: local_idx 
            for local_idx, global_idx in enumerate(state_indices)
        }
    
    def get_transitions(self, global_state_idx: int) -> List[TransitionData]:
        """Get transitions for a state by its global index."""
        local_idx = self.global_to_local.get(global_state_idx)
        if local_idx is None:
            raise KeyError(f"State {global_state_idx} not in timestep {self.timestep}")
        return self.transitions[local_idx]
    
    def memory_size_mb(self) -> float:
        """Estimate memory size in MB."""
        # Rough estimate: count objects
        num_transitions = sum(len(t) for t in self.transitions)
        # Each transition: ~180 bytes (action_profile + probs + indices with overhead)
        bytes_estimate = (
            len(self.state_indices) * 8 +  # state_indices list
            num_transitions * 180 +  # transitions
            len(self.global_to_local) * 40  # dict entries
        )
        return bytes_estimate / (1024**2)


class DiskBasedDAG:
    """
    Disk-based DAG storage for memory-efficient backward induction.
    
    Organizes states by timestep and saves each timestep slice to disk.
    During backward induction, only the current and next timestep slices
    are kept in memory.
    
    Also manages attainment cache slices on disk to avoid the 6+ GB memory overhead.
    Workers write cache slices directly to disk to avoid master-worker copying.
    
    Usage:
        # After computing DAG:
        disk_dag = DiskBasedDAG.from_dag(states, transitions, level_fct)
        
        # During backward induction (iterate from last timestep to first):
        for timestep in range(disk_dag.max_timestep, -1, -1):
            slice = disk_dag.load_slice(timestep)
            # Process states in slice...
            disk_dag.unload_slice(timestep)  # Free memory
    """
    
    def __init__(self, cache_dir: Optional[str] = None, use_compression: bool = False,
                 num_action_profiles: int = 0):
        """
        Initialize disk-based DAG.
        
        Args:
            cache_dir: Directory to store slices (None = auto-select optimal location)
            use_compression: If True, compress slices with gzip (slower but smaller)
            num_action_profiles: Number of action profiles (for cache initialization)
        """
        if cache_dir is None:
            # Auto-detect optimal cache location (tmpfs on Linux, fallback to /tmp)
            self.cache_dir = get_optimal_cache_dir()
            self._temp_dir = True
        else:
            self.cache_dir = cache_dir
            os.makedirs(cache_dir, exist_ok=True)
            self._temp_dir = False
        
        self.use_compression = use_compression
        self.num_action_profiles = num_action_profiles
        self.max_timestep: int = -1
        self._loaded_slices: Dict[int, DAGSlice] = {}
        self._slice_files: Dict[int, str] = {}
        
        # Attainment cache management
        self._cache_slice_files: Dict[int, str] = {}  # timestep -> cache file path
        self._loaded_cache_slices: Dict[int, Any] = {}  # timestep -> cache dict
    
    @classmethod
    def from_dag(cls, states: List[State], 
                 transitions: List[List[TransitionData]],
                 level_fct: Any,
                 cache_dir: Optional[str] = None,
                 use_compression: bool = False,
                 use_float16: bool = True,
                 num_action_profiles: int = 0,
                 quiet: bool = False) -> 'DiskBasedDAG':
        """
        Create disk-based DAG from full DAG.
        
        Args:
            states: List of all states
            transitions: Transition data for all states
            level_fct: Function that returns timestep for a state
            cache_dir: Where to save slices (None = temp dir)
            use_compression: Compress slices with gzip
            use_float16: Convert probability floats to float16 for 50% memory savings
            num_action_profiles: Number of action profiles (for cache initialization)
            quiet: Suppress progress messages
        
        Returns:
            DiskBasedDAG object with slices saved to disk
        """
        disk_dag = cls(cache_dir=cache_dir, use_compression=use_compression,
                      num_action_profiles=num_action_profiles)
        
        # Group states by timestep
        timestep_states: Dict[int, List[int]] = {}
        for state_idx, state in enumerate(states):
            timestep = level_fct(state)
            if timestep not in timestep_states:
                timestep_states[timestep] = []
            timestep_states[timestep].append(state_idx)
        
        disk_dag.max_timestep = max(timestep_states.keys())
        
        if not quiet:
            # Detect if using tmpfs for performance info
            is_tmpfs = "/dev/shm" in disk_dag.cache_dir or (
                hasattr(os, 'statvfs') and 
                os.path.exists(disk_dag.cache_dir) and
                os.statvfs(disk_dag.cache_dir).f_type == 0x01021994  # TMPFS_MAGIC on Linux
            )
            storage_type = "tmpfs (RAM)" if is_tmpfs else "disk"
            
            print(f"Slicing DAG into {len(timestep_states)} timesteps...")
            print(f"  Cache directory: {disk_dag.cache_dir}")
            print(f"  Storage type: {storage_type}")
            print(f"  Compression: {use_compression}")
            print(f"  Float16: {use_float16}")
        
        # Create and save each slice
        for timestep in sorted(timestep_states.keys()):
            state_indices = timestep_states[timestep]
            
            # Sort state indices to ensure consistent Kahn ordering within slice
            state_indices.sort()
            
            # Extract transitions for this timestep
            slice_transitions = []
            for global_idx in state_indices:
                trans_list = transitions[global_idx]
                
                # Optionally convert to float16
                if use_float16:
                    trans_list_converted = []
                    for action_profile, probs, succ_indices in trans_list:
                        probs_f16 = [np.float16(p) for p in probs]
                        trans_list_converted.append((action_profile, probs_f16, succ_indices))
                    slice_transitions.append(trans_list_converted)
                else:
                    # Create a COPY of the transitions to avoid keeping references to original
                    trans_list_copy = [
                        (action_profile, list(probs), list(succ_indices))
                        for action_profile, probs, succ_indices in trans_list
                    ]
                    slice_transitions.append(trans_list_copy)
            
            # Create slice
            dag_slice = DAGSlice(timestep, state_indices, slice_transitions)
            
            # Save to disk
            disk_dag._save_slice(dag_slice)
            
            if not quiet and timestep % max(1, len(timestep_states) // 10) == 0:
                print(f"  Saved timestep {timestep}: {len(state_indices)} states, "
                      f"{dag_slice.memory_size_mb():.1f} MB")
            
            # Free memory immediately after saving to disk
            del dag_slice
            del slice_transitions
        
        # Force garbage collection to free memory from transitions
        gc.collect()
        
        if not quiet:
            total_size_mb = sum(
                os.path.getsize(f) / (1024**2) 
                for f in disk_dag._slice_files.values()
            )
            print(f"  Total disk size: {total_size_mb:.1f} MB")
        
        return disk_dag
    
    def _get_slice_path(self, timestep: int) -> str:
        """Get file path for a timestep slice."""
        ext = ".pkl.gz" if self.use_compression else ".pkl"
        return os.path.join(self.cache_dir, f"slice_t{timestep:04d}{ext}")
    
    def _save_slice(self, dag_slice: DAGSlice) -> None:
        """Save a slice to disk."""
        filepath = self._get_slice_path(dag_slice.timestep)
        
        if self.use_compression:
            import gzip
            with gzip.open(filepath, 'wb') as f:
                pickle.dump(dag_slice, f, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            with open(filepath, 'wb') as f:
                pickle.dump(dag_slice, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        self._slice_files[dag_slice.timestep] = filepath
    
    def load_slice(self, timestep: int) -> DAGSlice:
        """
        Load a timestep slice from disk.
        
        Args:
            timestep: Timestep to load
        
        Returns:
            DAGSlice object
        """
        # Check if already loaded
        if timestep in self._loaded_slices:
            return self._loaded_slices[timestep]
        
        # Load from disk
        filepath = self._slice_files.get(timestep)
        if filepath is None:
            raise KeyError(f"No slice found for timestep {timestep}")
        
        if self.use_compression:
            import gzip
            with gzip.open(filepath, 'rb') as f:
                dag_slice = pickle.load(f)
        else:
            with open(filepath, 'rb') as f:
                dag_slice = pickle.load(f)
        
        self._loaded_slices[timestep] = dag_slice
        return dag_slice
    
    def unload_slice(self, timestep: int) -> None:
        """
        Unload a slice from memory (but keep on disk).
        
        Args:
            timestep: Timestep to unload
        """
        if timestep in self._loaded_slices:
            del self._loaded_slices[timestep]
    
    def get_loaded_memory_mb(self) -> float:
        """Get total memory used by loaded slices."""
        return sum(s.memory_size_mb() for s in self._loaded_slices.values())
    
    def cleanup(self) -> None:
        """Remove temporary directory if created."""
        if self._temp_dir:
            shutil.rmtree(self.cache_dir, ignore_errors=True)
    
    def __del__(self):
        """Cleanup on deletion."""
        self.cleanup()
    
    def __enter__(self):
        """Context manager support."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Cleanup on context manager exit."""
        self.cleanup()
        return False
    
    # ==================== Attainment Cache Methods ====================
    
    def get_cache_slice_path(self, timestep: int) -> str:
        """Get file path for attainment cache slice."""
        ext = ".cache.pkl.gz" if self.use_compression else ".cache.pkl"
        return os.path.join(self.cache_dir, f"cache_t{timestep:04d}{ext}")
    
    def create_cache_slice_for_states(self, state_indices: List[int]) -> Dict[int, List[Dict]]:
        """
        Create empty attainment cache structure for given states.
        
        This is used by workers to create their local cache before processing.
        
        Args:
            state_indices: List of state indices
        
        Returns:
            Cache dict: {state_idx: [{}  for _ in range(num_action_profiles)]}
        """
        return {
            state_idx: [{} for _ in range(self.num_action_profiles)]
            for state_idx in state_indices
        }
    
    def save_cache_slice(self, timestep: int, cache_data: Dict[int, List[Dict]]) -> None:
        """
        Save attainment cache slice to disk.
        
        Called by workers after processing a timestep to write cache directly to disk.
        
        Args:
            timestep: Timestep number
            cache_data: Cache dictionary for this timestep's states
        """
        filepath = self.get_cache_slice_path(timestep)
        
        # Ensure cache directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        if self.use_compression:
            import gzip
            with gzip.open(filepath, 'wb') as f:
                pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            with open(filepath, 'wb') as f:
                pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        self._cache_slice_files[timestep] = filepath
    
    def load_cache_slice(self, timestep: int) -> Optional[Dict[int, List[Dict]]]:
        """
        Load attainment cache slice from disk.
        
        Args:
            timestep: Timestep to load
        
        Returns:
            Cache dictionary if found, None otherwise
        """
        # Check if already loaded
        if timestep in self._loaded_cache_slices:
            return self._loaded_cache_slices[timestep]
        
        # Load from disk
        filepath = self._cache_slice_files.get(timestep)
        if filepath is None or not os.path.exists(filepath):
            return None
        
        if self.use_compression:
            import gzip
            with gzip.open(filepath, 'rb') as f:
                cache_data = pickle.load(f)
        else:
            with open(filepath, 'rb') as f:
                cache_data = pickle.load(f)
        
        self._loaded_cache_slices[timestep] = cache_data
        return cache_data
    
    def unload_cache_slice(self, timestep: int) -> None:
        """
        Unload cache slice from memory (keeps on disk).
        
        Args:
            timestep: Timestep to unload
        """
        if timestep in self._loaded_cache_slices:
            del self._loaded_cache_slices[timestep]
    
    def get_cache_memory_mb(self) -> float:
        """Get memory used by loaded cache slices."""
        # Rough estimate
        total_entries = 0
        for cache in self._loaded_cache_slices.values():
            for state_cache in cache.values():
                for ap_cache in state_cache:
                    total_entries += len(ap_cache)
        # Each entry: ~82 bytes (2 bytes data + 80 bytes overhead)
        return (total_entries * 82) / (1024**2)


def convert_transitions_to_float16(transitions: List[List[TransitionData]]) -> List[List[TransitionData]]:
    """
    Convert transition probabilities from float64 to float16.
    
    Args:
        transitions: Original transitions with float64 probabilities
    
    Returns:
        New transitions list with float16 probabilities
    """
    converted = []
    for state_trans in transitions:
        state_trans_f16 = []
        for action_profile, probs, succ_indices in state_trans:
            probs_f16 = [np.float16(p) for p in probs]
            state_trans_f16.append((action_profile, probs_f16, succ_indices))
        converted.append(state_trans_f16)
    return converted


def estimate_dag_memory(states: List[State], 
                       transitions: List[List[TransitionData]],
                       num_agents: int,
                       num_goals: int,
                       num_actions: int,
                       include_cache: bool = True,
                       num_humans: Optional[int] = None,
                       num_robots: Optional[int] = None) -> Dict[str, float]:
    """
    Estimate memory consumption of DAG and backward induction data structures.
    
    Calculates memory for all major data structures:
    - States list and transitions (shared)
    - Phase 1: Vh_values + human policies (system2_policies)
    - Phase 2: Vh_values (separate!) + Vr_values + robot_policy
    - Attainment cache (optional)
    - Python interpreter baseline
    
    Args:
        states: List of states
        transitions: Transition data
        num_agents: Total number of agents
        num_goals: Number of goals per agent
        num_actions: Number of actions per agent
        include_cache: If True, include attainment cache estimate
        num_humans: Number of human agents (default: num_agents - 1)
        num_robots: Number of robot agents (default: 1)
    
    Returns dict with memory estimates in MB:
        - states_mb: States list
        - transitions_mb: Transition probability data
        - phase1_vh_mb: Phase 1 Vh_values
        - phase1_policies_mb: Phase 1 human policies
        - phase2_vh_mb: Phase 2 Vh_values (expected human achievement)
        - phase2_vr_mb: Phase 2 Vr_values
        - phase2_robot_policy_mb: Phase 2 robot policy
        - attainment_cache_mb: Attainment cache (if include_cache=True)
        - python_baseline_mb: Python interpreter + imports
        - total_mb: Total memory estimate
    """
    num_states = len(states)
    
    # Default agent counts
    if num_humans is None:
        num_humans = max(1, num_agents - 1)
    if num_robots is None:
        num_robots = 1
    
    # Python dict overhead constants (CPython 3.10+)
    # WARNING: These values are specific to CPython 3.10+ implementation and may
    # be inaccurate on other Python versions (e.g., 3.8, 3.9) or implementations
    # like PyPy. The estimates should be used as rough guidelines only.
    # See: https://github.com/python/cpython/blob/main/Objects/dictobject.c
    DICT_EMPTY_SIZE = 64  # bytes for empty dict
    DICT_ENTRY_SIZE = 56  # bytes per entry (hash table overhead)
    LIST_EMPTY_SIZE = 56  # bytes for empty list
    LIST_ENTRY_SIZE = 8   # bytes per list element (pointer)
    NUMPY_HEADER = 128    # bytes numpy array overhead
    
    # --- States list ---
    # Each state is a tuple. Estimate avg 200 bytes per state (grid state tuples)
    avg_state_size = 200
    states_mb = (num_states * (avg_state_size + LIST_ENTRY_SIZE)) / (1024**2)
    
    # --- Transitions ---
    # Each transition: (action_profile tuple, probs list, indices list)
    num_transitions = sum(len(t) for t in transitions)
    avg_successors = num_transitions / max(1, sum(1 for t in transitions if t))
    # tuple: ~56 bytes + 8*num_agents, two lists: ~56 + 8*avg_successors each
    transition_size = 56 + 8*num_agents + 2*(56 + 8*avg_successors)
    transitions_mb = (num_transitions * transition_size + num_states * LIST_EMPTY_SIZE) / (1024**2)
    
    # --- Phase 1 Vh_values: List[Dict[agent, Dict[goal, float16]]] ---
    # Outer list: num_states entries
    # Middle dict (per state): num_humans entries -> inner dicts
    # Inner dict (per human): num_goals entries -> float16 values
    # float16 stored as numpy scalar: ~24 bytes
    inner_dict_size = DICT_EMPTY_SIZE + num_goals * (DICT_ENTRY_SIZE + 24)
    middle_dict_size = DICT_EMPTY_SIZE + num_humans * (DICT_ENTRY_SIZE + inner_dict_size)
    phase1_vh_mb = (num_states * middle_dict_size) / (1024**2)
    
    # --- Phase 1 policies: Dict[state, Dict[agent, Dict[goal, ndarray]]] ---
    # Outermost dict: num_states entries
    # Middle dict (per state): num_humans entries
    # Inner dict (per human): num_goals entries -> ndarray[num_actions]
    policy_array_size = NUMPY_HEADER + num_actions * 8  # float64 array
    inner_policy_dict_size = DICT_EMPTY_SIZE + num_goals * (DICT_ENTRY_SIZE + policy_array_size)
    middle_policy_dict_size = DICT_EMPTY_SIZE + num_humans * (DICT_ENTRY_SIZE + inner_policy_dict_size)
    phase1_policies_mb = (DICT_EMPTY_SIZE + num_states * (DICT_ENTRY_SIZE + avg_state_size + middle_policy_dict_size)) / (1024**2)
    
    # --- Phase 2 Vh_values: List[Dict[agent, Dict[goal, float16]]] ---
    # Same structure as Phase 1 - BOTH exist simultaneously!
    phase2_vh_mb = phase1_vh_mb
    
    # --- Phase 2 Vr_values: ndarray[num_states] ---
    phase2_vr_mb = (NUMPY_HEADER + num_states * 8) / (1024**2)  # float64
    
    # --- Phase 2 robot_policy: Dict[RobotActionProfile, float] per state ---
    # Stored as Dict[state, Dict[RobotActionProfile, float]]
    num_robot_action_profiles = num_actions ** num_robots
    robot_policy_per_state = DICT_EMPTY_SIZE + num_robot_action_profiles * (DICT_ENTRY_SIZE + 8)
    phase2_robot_policy_mb = (DICT_EMPTY_SIZE + num_states * (DICT_ENTRY_SIZE + avg_state_size + robot_policy_per_state)) / (1024**2)
    
    # --- Python baseline ---
    python_baseline_mb = 150.0  # interpreter + numpy + scipy + env imports
    
    # --- Totals ---
    data_structures_total = (states_mb + transitions_mb + 
                            phase1_vh_mb + phase1_policies_mb +
                            phase2_vh_mb + phase2_vr_mb + phase2_robot_policy_mb)
    
    result = {
        'num_states': num_states,
        'num_transitions': num_transitions,
        'avg_successors_per_transition': avg_successors,
        'states_mb': states_mb,
        'transitions_mb': transitions_mb,
        'phase1_vh_mb': phase1_vh_mb,
        'phase1_policies_mb': phase1_policies_mb,
        'phase2_vh_mb': phase2_vh_mb,
        'phase2_vr_mb': phase2_vr_mb,
        'phase2_robot_policy_mb': phase2_robot_policy_mb,
        'python_baseline_mb': python_baseline_mb,
        'total_mb': data_structures_total + python_baseline_mb,
    }
    
    # Attainment cache estimate (if used)
    if include_cache:
        num_action_profiles = num_actions ** num_agents
        # Cache is per state, per action profile, per goal: stores int8 arrays
        # But uses dicts, so overhead is significant
        cache_entries = num_states * num_action_profiles * num_goals
        # Each entry: ~2 bytes (int8 value) + dict overhead amortized
        cache_mb = (num_states * num_action_profiles * (DICT_EMPTY_SIZE + num_goals * (DICT_ENTRY_SIZE + 2))) / (1024**2)
        result['attainment_cache_mb'] = cache_mb
        result['total_mb'] += cache_mb
    
    # Add 20% overhead for temporaries during computation (Q-value arrays, etc.)
    result['total_mb'] *= 1.2
    
    return result

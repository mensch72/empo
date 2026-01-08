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
import pickle
import tempfile
import shutil
from typing import List, Dict, Tuple, Any, Optional
from pathlib import Path
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
                    slice_transitions.append(trans_list)
            
            # Create slice
            dag_slice = DAGSlice(timestep, state_indices, slice_transitions)
            
            # Save to disk
            disk_dag._save_slice(dag_slice)
            
            if not quiet and timestep % max(1, len(timestep_states) // 10) == 0:
                print(f"  Saved timestep {timestep}: {len(state_indices)} states, "
                      f"{dag_slice.memory_size_mb():.1f} MB")
        
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
                       include_cache: bool = True) -> Dict[str, float]:
    """
    Estimate memory consumption of DAG and backward induction data structures.
    
    Args:
        states: List of states
        transitions: Transition data
        num_agents: Number of agents
        num_goals: Number of goals per agent
        num_actions: Number of actions per agent
        include_cache: If True, include attainment cache estimate
    
    Returns dict with memory estimates in MB:
        - transitions_mb: Transition probability data
        - vh_values_mb: Value function data (Vh_values)
        - policies_mb: Policy data (system2_policies)
        - attainment_cache_mb: Attainment cache (if include_cache=True)
        - total_mb: Total estimated memory
    """
    num_states = len(states)
    
    # Count actual transitions
    num_transitions = sum(len(t) for t in transitions)
    avg_successors = num_transitions / max(1, sum(1 for t in transitions if t))
    
    # Transitions: each is (action_profile, probs, indices)
    # Estimate ~180 bytes per transition with Python overhead
    transitions_mb = (num_transitions * 180) / (1024**2)
    
    # Vh_values: List[List[Dict[goal, float]]]
    # Each dict entry: ~16 bytes (key + value) + dict overhead
    vh_values_mb = (num_states * num_agents * (num_goals * 16 + 1000)) / (1024**2)
    
    # Policies: Dict[state, Dict[agent, Dict[goal, ndarray[num_actions]]]]
    # Each ndarray: num_actions * 8 bytes + ~100 bytes overhead
    policies_mb = (num_states * num_agents * num_goals * (num_actions * 8 + 100)) / (1024**2)
    
    result = {
        'num_states': num_states,
        'num_transitions': num_transitions,
        'avg_successors_per_transition': avg_successors,
        'transitions_mb': transitions_mb,
        'vh_values_mb': vh_values_mb,
        'policies_mb': policies_mb,
        'total_mb': transitions_mb + vh_values_mb + policies_mb,
    }
    
    # Attainment cache estimate
    if include_cache:
        num_action_profiles = num_actions ** num_agents
        cache_entries = num_states * num_action_profiles * num_goals
        # Each entry: ~2 bytes (int8 array) + 80 bytes (dict overhead)
        cache_mb = (cache_entries * 82) / (1024**2)
        result['attainment_cache_mb'] = cache_mb
        result['total_mb'] += cache_mb
    
    return result

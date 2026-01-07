"""
Shared Memory Attainment Cache for Parallel Backward Induction.

This module provides utilities to store goal attainment caches in shared memory
that can be accessed by multiple processes without copy-on-write overhead.

The attainment cache stores precomputed is_achieved() results for (state, action_profile, goal)
combinations. This avoids redundant computation between Phase 1 and Phase 2, and within
parallel workers.

Structure: cache[state_index][action_profile_index][goal] -> np.ndarray of attainment values.
The first two indices are lists (for efficient O(1) access with known ranges), and the last
is a dict (for flexible goal keys).

Usage:
    # In parent process (Phase 1):
    from empo.backward_induction.helpers import create_attainment_cache
    cache = create_attainment_cache(num_states, num_action_profiles)
    # ... populate cache during backward induction ...
    
    # Store on world_model for automatic reuse:
    world_model._attainment_cache = cache
    
    # For parallel Phase 2, create shared memory version:
    shared_cache = init_shared_attainment_cache(cache)
    
    # In worker processes (after fork):
    cache = get_shared_attainment_cache()
    
    # Cleanup when done:
    cleanup_shared_attainment_cache()
"""

import pickle
from multiprocessing import shared_memory
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING
import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from empo.possible_goal import PossibleGoal

# Type alias for the attainment cache structure (imported from helpers for consistency)
# List[state_index -> List[action_profile_index -> Dict[goal, np.ndarray of int8]]]
AttainmentCache = List[List[Dict["PossibleGoal", npt.NDArray[np.int8]]]]


class SharedAttainmentCache:
    """
    Store attainment cache in shared memory for efficient multi-process access.
    
    This class serializes the attainment cache into a shared memory block,
    allowing multiple forked processes to access it without triggering
    copy-on-write overhead.
    """
    
    def __init__(
        self,
        cache_shm: shared_memory.SharedMemory,
        cache_size: int,
    ):
        """
        Initialize with existing shared memory block.
        
        Args:
            cache_shm: Shared memory block containing pickled cache
            cache_size: Size of pickled cache data in bytes
        """
        self._cache_shm = cache_shm
        self._cache_size = cache_size
        
        # Cache for deserialized data (lazy loading)
        self._cache: Optional[AttainmentCache] = None
    
    @classmethod
    def from_cache(cls, cache: AttainmentCache) -> 'SharedAttainmentCache':
        """
        Create a SharedAttainmentCache from an existing cache dict.
        
        This pickles the cache and stores it in a shared memory block.
        Should be called in the parent process before forking workers.
        
        Args:
            cache: The attainment cache dict
            
        Returns:
            SharedAttainmentCache instance with data in shared memory
        """
        cache_bytes = pickle.dumps(cache)
        cache_shm = shared_memory.SharedMemory(create=True, size=len(cache_bytes))
        cache_shm.buf[:len(cache_bytes)] = cache_bytes
        
        return cls(
            cache_shm=cache_shm,
            cache_size=len(cache_bytes),
        )
    
    def get_cache(self) -> AttainmentCache:
        """
        Get the attainment cache from shared memory.
        
        The result is cached after the first call.
        """
        if self._cache is None:
            self._cache = pickle.loads(bytes(self._cache_shm.buf[:self._cache_size]))
        return self._cache
    
    def get_shm_name(self) -> str:
        """Get the name of the shared memory block."""
        return self._cache_shm.name
    
    def get_size(self) -> int:
        """Get the size of the serialized data."""
        return self._cache_size
    
    @classmethod
    def attach(cls, shm_name: str, size: int) -> 'SharedAttainmentCache':
        """
        Attach to existing shared memory block by name.
        
        This should be called in worker processes to access the shared cache.
        
        Args:
            shm_name: Name of the shared memory block
            size: Size of the cache data
            
        Returns:
            SharedAttainmentCache instance attached to the existing shared memory
        """
        cache_shm = shared_memory.SharedMemory(name=shm_name)
        return cls(cache_shm=cache_shm, cache_size=size)
    
    def close(self) -> None:
        """
        Close this process's view of the shared memory.
        
        Call this in worker processes when done. Does not destroy the shared memory.
        """
        self._cache_shm.close()
    
    def cleanup(self) -> None:
        """
        Unlink (destroy) the shared memory block.
        
        Call this only once, in the parent process, after all workers are done.
        """
        self._cache_shm.close()
        self._cache_shm.unlink()


# Global for workers to access shared attainment cache
_shared_attainment_cache: Optional[SharedAttainmentCache] = None
_shared_attainment_cache_info: Optional[Tuple[str, int]] = None


def init_shared_attainment_cache(cache: AttainmentCache) -> SharedAttainmentCache:
    """
    Initialize shared attainment cache in the parent process.
    
    This creates a shared memory block and stores the cache data.
    Call this before forking worker processes.
    
    Args:
        cache: The attainment cache dict
        
    Returns:
        SharedAttainmentCache instance
    """
    global _shared_attainment_cache, _shared_attainment_cache_info
    
    # Clean up any existing shared memory
    cleanup_shared_attainment_cache()
    
    _shared_attainment_cache = SharedAttainmentCache.from_cache(cache)
    _shared_attainment_cache_info = (
        _shared_attainment_cache.get_shm_name(),
        _shared_attainment_cache.get_size(),
    )
    
    return _shared_attainment_cache


def get_shared_attainment_cache_info() -> Optional[Tuple[str, int]]:
    """Get the shared memory info for passing to workers."""
    return _shared_attainment_cache_info


def attach_shared_attainment_cache() -> Optional[AttainmentCache]:
    """
    Attach to shared attainment cache from a worker process.
    
    Uses the global _shared_attainment_cache_info set in the parent process.
    
    Returns:
        The attainment cache dict, or None if no shared cache exists
    """
    global _shared_attainment_cache
    
    if _shared_attainment_cache_info is None:
        return None
    
    shm_name, size = _shared_attainment_cache_info
    _shared_attainment_cache = SharedAttainmentCache.attach(shm_name, size)
    return _shared_attainment_cache.get_cache()


def get_shared_attainment_cache() -> Optional[AttainmentCache]:
    """Get the current shared attainment cache (for use in workers after attach)."""
    if _shared_attainment_cache is None:
        return None
    return _shared_attainment_cache.get_cache()


def cleanup_shared_attainment_cache() -> None:
    """Clean up shared attainment cache memory. Call in parent process when done."""
    global _shared_attainment_cache, _shared_attainment_cache_info
    
    if _shared_attainment_cache is not None:
        try:
            _shared_attainment_cache.cleanup()
        except Exception:
            pass
        _shared_attainment_cache = None
        _shared_attainment_cache_info = None

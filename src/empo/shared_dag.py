"""
Shared Memory DAG Storage for Parallel Backward Induction.

This module provides utilities to store DAG data (states, transitions, probabilities)
in shared memory that can be accessed by multiple processes without copy-on-write overhead.

The key insight is that Python's fork-based multiprocessing triggers copy-on-write
even for read-only access due to reference counting. By storing data in raw shared
memory blocks (via multiprocessing.shared_memory), we avoid this overhead.

Usage:
    # In parent process:
    shared_dag = SharedDAG.from_dag(states, transitions)
    
    # In worker processes (after fork):
    states = shared_dag.get_states()
    transitions = shared_dag.get_transitions()
    
    # Cleanup when done:
    shared_dag.cleanup()
"""

import pickle
from multiprocessing import shared_memory
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import numpy.typing as npt


# Type aliases
State = Any
TransitionData = Tuple[Tuple[int, ...], List[float], List[int]]


class SharedDAG:
    """
    Store DAG data in shared memory for efficient multi-process access.
    
    This class serializes the DAG data structures into shared memory blocks,
    allowing multiple forked processes to access them without triggering
    copy-on-write overhead.
    
    The DAG is stored as:
    - states: pickled list of State objects
    - transitions: pickled list of transition data
    
    While pickling has overhead, it's done once in the parent process.
    Workers then access the shared memory directly without copying.
    """
    
    def __init__(
        self,
        states_shm: shared_memory.SharedMemory,
        states_size: int,
        transitions_shm: Optional[shared_memory.SharedMemory],
        transitions_size: int,
    ):
        """
        Initialize with existing shared memory blocks.
        
        Args:
            states_shm: Shared memory block containing pickled states
            states_size: Size of pickled states data in bytes
            transitions_shm: Shared memory block containing pickled transitions (optional)
            transitions_size: Size of pickled transitions data in bytes
        """
        self._states_shm = states_shm
        self._states_size = states_size
        self._transitions_shm = transitions_shm
        self._transitions_size = transitions_size
        
        # Cache for deserialized data (lazy loading)
        self._states_cache: Optional[List[State]] = None
        self._transitions_cache: Optional[List[List[TransitionData]]] = None
    
    @classmethod
    def from_dag(
        cls,
        states: List[State],
        transitions: Optional[List[List[TransitionData]]] = None,
    ) -> 'SharedDAG':
        """
        Create a SharedDAG from DAG data structures.
        
        This pickles the data and stores it in shared memory blocks.
        Should be called in the parent process before forking workers.
        
        Args:
            states: List of state objects
            transitions: Optional list of transition data per state
            
        Returns:
            SharedDAG instance with data in shared memory
        """
        # Pickle states
        states_bytes = pickle.dumps(states)
        states_shm = shared_memory.SharedMemory(create=True, size=len(states_bytes))
        states_shm.buf[:len(states_bytes)] = states_bytes
        
        # Pickle transitions if provided
        transitions_shm = None
        transitions_size = 0
        if transitions is not None:
            transitions_bytes = pickle.dumps(transitions)
            transitions_shm = shared_memory.SharedMemory(create=True, size=len(transitions_bytes))
            transitions_shm.buf[:len(transitions_bytes)] = transitions_bytes
            transitions_size = len(transitions_bytes)
        
        return cls(
            states_shm=states_shm,
            states_size=len(states_bytes),
            transitions_shm=transitions_shm,
            transitions_size=transitions_size,
        )
    
    def get_states(self) -> List[State]:
        """
        Get the list of states from shared memory.
        
        The result is cached after the first call.
        """
        if self._states_cache is None:
            self._states_cache = pickle.loads(bytes(self._states_shm.buf[:self._states_size]))
        return self._states_cache
    
    def get_transitions(self) -> Optional[List[List[TransitionData]]]:
        """
        Get the transitions from shared memory.
        
        The result is cached after the first call.
        Returns None if transitions were not stored.
        """
        if self._transitions_shm is None:
            return None
        if self._transitions_cache is None:
            self._transitions_cache = pickle.loads(bytes(self._transitions_shm.buf[:self._transitions_size]))
        return self._transitions_cache
    
    def get_shm_names(self) -> Tuple[str, Optional[str]]:
        """
        Get the names of the shared memory blocks.
        
        These can be used to attach to the shared memory from other processes.
        """
        transitions_name = self._transitions_shm.name if self._transitions_shm else None
        return (self._states_shm.name, transitions_name)
    
    def get_sizes(self) -> Tuple[int, int]:
        """Get the sizes of the serialized data."""
        return (self._states_size, self._transitions_size)
    
    @classmethod
    def attach(
        cls,
        states_shm_name: str,
        states_size: int,
        transitions_shm_name: Optional[str],
        transitions_size: int,
    ) -> 'SharedDAG':
        """
        Attach to existing shared memory blocks by name.
        
        This should be called in worker processes to access the shared data.
        
        Args:
            states_shm_name: Name of the states shared memory block
            states_size: Size of the states data
            transitions_shm_name: Name of the transitions shared memory block (or None)
            transitions_size: Size of the transitions data
            
        Returns:
            SharedDAG instance attached to the existing shared memory
        """
        states_shm = shared_memory.SharedMemory(name=states_shm_name)
        transitions_shm = None
        if transitions_shm_name:
            transitions_shm = shared_memory.SharedMemory(name=transitions_shm_name)
        
        return cls(
            states_shm=states_shm,
            states_size=states_size,
            transitions_shm=transitions_shm,
            transitions_size=transitions_size,
        )
    
    def close(self) -> None:
        """
        Close this process's view of the shared memory.
        
        Call this in worker processes when done. Does not destroy the shared memory.
        """
        self._states_shm.close()
        if self._transitions_shm:
            self._transitions_shm.close()
    
    def cleanup(self) -> None:
        """
        Unlink (destroy) the shared memory blocks.
        
        Call this only once, in the parent process, after all workers are done.
        """
        self._states_shm.close()
        self._states_shm.unlink()
        if self._transitions_shm:
            self._transitions_shm.close()
            self._transitions_shm.unlink()


# Global for workers to access shared DAG
_shared_dag: Optional[SharedDAG] = None
_shared_dag_info: Optional[Tuple[str, int, Optional[str], int]] = None


def init_shared_dag(states: List[State], transitions: Optional[List[List[TransitionData]]] = None) -> SharedDAG:
    """
    Initialize shared DAG in the parent process.
    
    This creates shared memory blocks and stores the DAG data.
    Call this before forking worker processes.
    
    Args:
        states: List of state objects
        transitions: Optional list of transition data
        
    Returns:
        SharedDAG instance
    """
    global _shared_dag, _shared_dag_info
    
    # Clean up any existing shared memory
    if _shared_dag is not None:
        try:
            _shared_dag.cleanup()
        except Exception:
            pass
    
    _shared_dag = SharedDAG.from_dag(states, transitions)
    names = _shared_dag.get_shm_names()
    sizes = _shared_dag.get_sizes()
    _shared_dag_info = (names[0], sizes[0], names[1], sizes[1])
    
    return _shared_dag


def get_shared_dag_info() -> Optional[Tuple[str, int, Optional[str], int]]:
    """Get the shared memory info for passing to workers."""
    return _shared_dag_info


def attach_shared_dag() -> Optional[SharedDAG]:
    """
    Attach to shared DAG from a worker process.
    
    Uses the global _shared_dag_info set in the parent process.
    """
    global _shared_dag
    
    if _shared_dag_info is None:
        return None
    
    states_name, states_size, transitions_name, transitions_size = _shared_dag_info
    _shared_dag = SharedDAG.attach(states_name, states_size, transitions_name, transitions_size)
    return _shared_dag


def get_shared_dag() -> Optional[SharedDAG]:
    """Get the current shared DAG (for use in workers after attach)."""
    return _shared_dag


def cleanup_shared_dag() -> None:
    """Clean up shared DAG memory. Call in parent process when done."""
    global _shared_dag, _shared_dag_info
    
    if _shared_dag is not None:
        try:
            _shared_dag.cleanup()
        except Exception:
            pass
        _shared_dag = None
        _shared_dag_info = None

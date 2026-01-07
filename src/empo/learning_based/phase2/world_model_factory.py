"""
World Model Factory for Phase 2 Async Training.

Provides a factory protocol that creates world models for actor processes.
This allows async training to work with environments that can't be pickled.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional


class WorldModelFactory(ABC):
    """
    Abstract factory for creating world models in actor processes.
    
    This is used for async training where environments cannot be pickled
    and must be created fresh in each actor process.
    
    Two modes are supported:
    - Fixed: Create once, cache, return same world model every call
    - Ensemble: Create new world model every call (or every k calls)
    
    Implementations must be picklable (store only configuration, not the env).
    """
    
    @abstractmethod
    def create(self) -> Any:
        """
        Create or return a world model.
        
        Returns:
            A world model (environment) instance.
        """
    
    def reset(self) -> None:
        """
        Reset the factory state (e.g., clear cache for fixed mode).
        
        Called when actor process starts to ensure fresh state.
        """


class CachedWorldModelFactory(WorldModelFactory):
    """
    Factory that creates the world model once and caches it.
    
    Use this for fixed-env training where all episodes use the same environment.
    
    Args:
        factory_fn: Callable that creates a world model when called with no args.
    """
    
    def __init__(self, factory_fn):
        """
        Initialize with a factory function.
        
        Args:
            factory_fn: Callable that creates a world model. Must be picklable
                (e.g., a module-level function or lambda-free callable).
        """
        self._factory_fn = factory_fn
        self._cached_world_model: Optional[Any] = None
    
    def create(self) -> Any:
        """Return cached world model, creating it on first call."""
        if self._cached_world_model is None:
            self._cached_world_model = self._factory_fn()
        return self._cached_world_model
    
    def reset(self) -> None:
        """Clear the cache."""
        self._cached_world_model = None
    
    def __getstate__(self):
        """Exclude cached world model from pickling."""
        state = self.__dict__.copy()
        state['_cached_world_model'] = None
        return state
    
    def __setstate__(self, state):
        """Restore state after unpickling."""
        self.__dict__.update(state)


class EnsembleWorldModelFactory(WorldModelFactory):
    """
    Factory that creates a new world model each call (or every k calls).
    
    Use this for ensemble training where each episode uses a different environment.
    
    Args:
        factory_fn: Callable that creates a world model when called with no args.
        episodes_per_env: Number of episodes to use same env before creating new one.
            Default 1 means new env every episode.
    """
    
    def __init__(self, factory_fn, episodes_per_env: int = 1):
        """
        Initialize with a factory function.
        
        Args:
            factory_fn: Callable that creates a world model. Must be picklable.
            episodes_per_env: How many episodes to run on same env.
        """
        self._factory_fn = factory_fn
        self._episodes_per_env = episodes_per_env
        self._cached_world_model: Optional[Any] = None
        self._episode_count: int = 0
    
    def create(self) -> Any:
        """
        Return world model, creating new one every episodes_per_env calls.
        """
        if self._cached_world_model is None or self._episode_count >= self._episodes_per_env:
            self._cached_world_model = self._factory_fn()
            self._episode_count = 0
        self._episode_count += 1
        return self._cached_world_model
    
    def reset(self) -> None:
        """Clear state for fresh start."""
        self._cached_world_model = None
        self._episode_count = 0
    
    def __getstate__(self):
        """Exclude cached world model from pickling."""
        state = self.__dict__.copy()
        state['_cached_world_model'] = None
        state['_episode_count'] = 0
        return state
    
    def __setstate__(self, state):
        """Restore state after unpickling."""
        self.__dict__.update(state)

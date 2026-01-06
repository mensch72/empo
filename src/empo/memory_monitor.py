"""
Memory Monitor for EMPO Framework.

Provides a general-purpose memory monitoring utility that can be used by both
the Phase 2 trainer and backward induction code to prevent out-of-memory crashes.

The monitor periodically checks free system memory and when it falls below a
configurable threshold:
1. Pauses computation for a configurable duration (default 60 seconds)
2. Checks again, and if still below threshold, raises KeyboardInterrupt

This allows long-running computations to gracefully handle memory pressure,
either by waiting for other processes to release memory or by triggering
a clean shutdown with checkpoint saving.
"""

import time
from typing import Optional

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


class MemoryMonitor:
    """
    Monitor system memory and handle low-memory situations.
    
    This class provides a general-purpose memory monitoring mechanism that:
    - Periodically checks free memory as a fraction of total system memory
    - When free memory falls below a configurable threshold, pauses computation
    - If memory is still low after pausing, raises KeyboardInterrupt
    
    The monitor is designed to be used in long-running computations like
    Phase 2 training or backward induction.
    
    Attributes:
        min_free_fraction: Minimum free memory fraction (0.0-1.0). Default 0.1 (10%).
        check_interval: How often to check memory (in iterations). Default 100.
        pause_duration: How long to pause when memory is low (seconds). Default 60.
        verbose: Whether to print status messages. Default True.
        enabled: Whether memory monitoring is enabled. Default True.
        
    Example usage:
        >>> monitor = MemoryMonitor(min_free_fraction=0.1, check_interval=100)
        >>> for step in range(1000000):
        ...     # Your computation here
        ...     monitor.check(step)  # Will pause or interrupt if memory is low
        
        Or with context manager for automatic resource cleanup:
        >>> with MemoryMonitor(min_free_fraction=0.1) as monitor:
        ...     for step in range(1000000):
        ...         # Your computation here
        ...         monitor.check(step)
    """
    
    def __init__(
        self,
        min_free_fraction: float = 0.1,
        check_interval: int = 100,
        pause_duration: float = 60.0,
        verbose: bool = True,
        enabled: bool = True
    ):
        """
        Initialize the memory monitor.
        
        Args:
            min_free_fraction: Minimum free memory as fraction of total (0.0-1.0).
                When free memory falls below this threshold, the monitor takes action.
                Default is 0.1 (10% free).
            check_interval: How often to check memory, in iterations/steps.
                Lower values catch memory issues faster but add overhead.
                Default is 100.
            pause_duration: How long to pause (in seconds) when memory is low.
                After pausing, memory is checked again. Default is 60 seconds.
            verbose: Whether to print status messages. Default is True.
            enabled: Whether memory monitoring is enabled. Set to False to
                disable all checks. Default is True.
        """
        self.min_free_fraction = min_free_fraction
        self.check_interval = check_interval
        self.pause_duration = pause_duration
        self.verbose = verbose
        self.enabled = enabled
        
        # Track last check step to support manual check calls
        self._last_check_step: int = -1
        
        # Warn if psutil is not available
        if not HAS_PSUTIL and enabled:
            if verbose:
                print("[MemoryMonitor] Warning: psutil not available. Memory monitoring disabled.")
            self.enabled = False
    
    def __enter__(self) -> 'MemoryMonitor':
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """Context manager exit. Does not suppress exceptions."""
        return False
    
    def get_free_memory_fraction(self) -> Optional[float]:
        """
        Get current free memory as a fraction of total system memory.
        
        Returns:
            Free memory fraction (0.0-1.0), or None if psutil is not available.
        """
        if not HAS_PSUTIL:
            return None
        
        mem_info = psutil.virtual_memory()
        # available is the memory that can be given to processes without swapping
        return mem_info.available / mem_info.total
    
    def get_memory_usage_fraction(self) -> Optional[float]:
        """
        Get current memory usage as a fraction of total system memory.
        
        Returns:
            Memory usage fraction (0.0-1.0), or None if psutil is not available.
        """
        free = self.get_free_memory_fraction()
        if free is None:
            return None
        return 1.0 - free
    
    def is_memory_low(self) -> bool:
        """
        Check if free memory is below the configured threshold.
        
        Returns:
            True if free memory is below min_free_fraction, False otherwise.
            Returns False if psutil is not available or monitoring is disabled.
        """
        if not self.enabled or not HAS_PSUTIL:
            return False
        
        free = self.get_free_memory_fraction()
        if free is None:
            return False
        
        return free < self.min_free_fraction
    
    def check(self, step: int, force: bool = False) -> None:
        """
        Check memory at the given step and handle low-memory situations.
        
        This method should be called regularly during computation (e.g., in a loop).
        It only actually checks memory every `check_interval` steps to minimize overhead.
        
        When memory is low:
        1. Prints a warning (if verbose)
        2. Pauses for `pause_duration` seconds
        3. Checks again - if still low, raises KeyboardInterrupt
        
        Args:
            step: Current iteration/step number. Memory is only checked when
                step is a multiple of check_interval.
            force: If True, check memory regardless of step number.
            
        Raises:
            KeyboardInterrupt: If memory is still low after pausing.
        """
        if not self.enabled:
            return
        
        # Only check at configured interval (unless forced)
        if not force and step % self.check_interval != 0:
            return
        
        # Skip if we already checked at this step
        if step == self._last_check_step and not force:
            return
        self._last_check_step = step
        
        if not self.is_memory_low():
            return
        
        # Memory is low - take action
        free = self.get_free_memory_fraction()
        if self.verbose:
            free_pct = free * 100 if free is not None else 0
            print(f"\n[MemoryMonitor] Warning: Free memory ({free_pct:.1f}%) is below "
                  f"threshold ({self.min_free_fraction * 100:.1f}%). "
                  f"Pausing for {self.pause_duration:.0f} seconds...")
        
        # Pause and wait for memory to potentially free up
        time.sleep(self.pause_duration)
        
        # Check again
        if self.is_memory_low():
            free = self.get_free_memory_fraction()
            free_pct = free * 100 if free is not None else 0
            if self.verbose:
                print(f"[MemoryMonitor] Free memory ({free_pct:.1f}%) still below threshold "
                      f"after pause. Raising KeyboardInterrupt for graceful shutdown...")
            
            # Raise KeyboardInterrupt for graceful shutdown
            # This allows try/except KeyboardInterrupt handlers to save checkpoints
            raise KeyboardInterrupt("Memory limit exceeded - free memory below threshold")
        else:
            if self.verbose:
                free = self.get_free_memory_fraction()
                free_pct = free * 100 if free is not None else 0
                print(f"[MemoryMonitor] Memory recovered ({free_pct:.1f}% free). Resuming computation...")


def check_memory(
    step: int,
    min_free_fraction: float = 0.1,
    check_interval: int = 100,
    pause_duration: float = 60.0,
    verbose: bool = True
) -> None:
    """
    Convenience function for one-off memory checks without creating a MemoryMonitor instance.
    
    This is a simpler interface for cases where you don't need the full MemoryMonitor class.
    
    Args:
        step: Current iteration/step number.
        min_free_fraction: Minimum free memory as fraction of total (0.0-1.0).
        check_interval: How often to check memory, in iterations/steps.
        pause_duration: How long to pause (seconds) when memory is low.
        verbose: Whether to print status messages.
        
    Raises:
        KeyboardInterrupt: If memory is still low after pausing.
        
    Example:
        >>> for step in range(1000000):
        ...     # Your computation here
        ...     check_memory(step, min_free_fraction=0.1, check_interval=100)
    """
    if not HAS_PSUTIL:
        return
    
    if step % check_interval != 0:
        return
    
    if min_free_fraction <= 0.0:
        return
    
    mem_info = psutil.virtual_memory()
    free_fraction = mem_info.available / mem_info.total
    
    if free_fraction >= min_free_fraction:
        return
    
    # Memory is low
    if verbose:
        print(f"\n[MemoryMonitor] Warning: Free memory ({free_fraction * 100:.1f}%) is below "
              f"threshold ({min_free_fraction * 100:.1f}%). "
              f"Pausing for {pause_duration:.0f} seconds...")
    
    time.sleep(pause_duration)
    
    # Check again
    mem_info = psutil.virtual_memory()
    free_fraction = mem_info.available / mem_info.total
    
    if free_fraction < min_free_fraction:
        if verbose:
            print(f"[MemoryMonitor] Free memory ({free_fraction * 100:.1f}%) still below threshold "
                  f"after pause. Raising KeyboardInterrupt for graceful shutdown...")
        raise KeyboardInterrupt("Memory limit exceeded - free memory below threshold")
    else:
        if verbose:
            print(f"[MemoryMonitor] Memory recovered ({free_fraction * 100:.1f}% free). Resuming computation...")

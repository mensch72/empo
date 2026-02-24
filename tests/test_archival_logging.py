#!/usr/bin/env python3
"""Quick test to verify archival output works."""

import tempfile
from pathlib import Path

from empo.backward_induction.helpers import detect_archivable_levels, archive_value_slices

# Test detect_archivable_levels
print("=" * 70)
print("Testing detect_archivable_levels")
print("=" * 70)

# Levels processing in descending order: [5, 4, 3, 2, 1, 0]
# Level 5 successors: max=4
# Level 4 successors: max=3
# Level 3 successors: max=2
# ...
max_successor_levels = {
    5: 4,  # Level 5 has max successor at level 4
    4: 3,  # Level 4 has max successor at level 3
    3: 2,
    2: 1,
    1: 0,
    0: -1,  # Terminal level has no successors
}

# When processing level 3, future levels are [3, 2, 1, 0]
# Max future successor = max(2, 1, 0, -1) = 2
# So levels > 2 can be archived: [5, 4, 3] (but 3 is current, so really [5, 4])
print("\nAt level 3 (just completed):")
archivable = detect_archivable_levels(3, max_successor_levels, quiet=False)
print(f"Result: {archivable}")
print(f"Expected: [5, 4, 3] (levels no longer needed)")

print("\nAt level 1 (just completed):")
archivable = detect_archivable_levels(1, max_successor_levels, quiet=False)
print(f"Result: {archivable}")
print(f"Expected: [5, 4, 3, 2, 1] (all levels > max future successor of 0)")

# Test archive_value_slices
print("\n" + "=" * 70)
print("Testing archive_value_slices")
print("=" * 70)

# Create mock data
states = [
    (5, 0, 0),  # state 0 at level 5
    (5, 1, 0),  # state 1 at level 5
    (4, 0, 0),  # state 2 at level 4
    (4, 1, 0),  # state 3 at level 4
    (3, 0, 0),  # state 4 at level 3
]

values = [
    [{'goal1': 1.0}, {'goal2': 2.0}],  # state 0
    [{'goal1': 1.5}, {'goal2': 2.5}],  # state 1
    [{'goal1': 2.0}, {'goal2': 3.0}],  # state 2
    [{'goal1': 2.5}, {'goal2': 3.5}],  # state 3
    [{'goal1': 3.0}, {'goal2': 4.0}],  # state 4
]

level_fct = lambda state: state[0]  # First element is level

with tempfile.TemporaryDirectory() as tmpdir:
    filepath = Path(tmpdir) / "test_values.pkl"
    
    print(f"\nArchiving levels [5, 4] to {filepath}")
    archive_value_slices(
        values=values,
        states=states,
        level_fct=level_fct,
        archivable_levels=[5, 4],
        filepath=filepath,
        return_values=True,  # Keep in memory
        quiet=False  # Show output
    )
    
    print(f"\nFile created: {filepath.exists()}")
    print(f"File size: {filepath.stat().st_size} bytes")

print("\nDone!")

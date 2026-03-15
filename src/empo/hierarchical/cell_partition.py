"""Agglomerative rectangular partitioning of grid cells.

Partitions a set of grid positions into maximal rectangular macro-cells
using an agglomerative clustering algorithm. Each walkable cell starts as
a singleton 1x1 block. Adjacent blocks whose union forms a rectangle are
iteratively merged, preferring merges with minimal merged area. The
algorithm stops when no further rectangular merges are possible.

This module is independent of any specific grid environment and operates
purely on a set of (x, y) integer positions.
"""

from typing import Dict, FrozenSet, List, Optional, Set, Tuple

import heapq
import random


# (x_min, y_min, x_max, y_max) — inclusive on both ends
Rectangle = Tuple[int, int, int, int]


def _rect_area(r: Rectangle) -> int:
    """Return the area (number of cells) of a rectangle."""
    return (r[2] - r[0] + 1) * (r[3] - r[1] + 1)


def _merged_rect(a: Rectangle, b: Rectangle) -> Rectangle:
    """Return the bounding-box rectangle of two rectangles."""
    return (min(a[0], b[0]), min(a[1], b[1]),
            max(a[2], b[2]), max(a[3], b[3]))


class CellPartition:
    """Partition of grid cells into maximal non-overlapping rectangles.

    Uses agglomerative hierarchical clustering: starts with each cell as a
    singleton 1x1 block, then iteratively merges adjacent blocks whose union
    is still rectangular, preferring merges with minimal area (ties broken
    at random). Stops when no further rectangular merges are possible.

    Attributes:
        rectangles: List of (x_min, y_min, x_max, y_max) tuples, one per
            macro-cell.  Coordinates are inclusive on both ends.
        num_cells: Number of macro-cells.
        adjacency: Dict mapping cell index to frozenset of adjacent cell
            indices.
    """

    def __init__(
        self,
        walkable: Set[Tuple[int, int]],
        *,
        seed: Optional[int] = None,
    ):
        """Compute an agglomerative rectangular partition.

        Args:
            walkable: Set of (x, y) positions to partition.
            seed: Random seed for tie-breaking (default: None for
                non-deterministic).
        """
        self._rng = random.Random(seed)
        self._walkable: FrozenSet[Tuple[int, int]] = frozenset(walkable)

        # Computed by _compute_partition
        self._cell_map: Dict[Tuple[int, int], int] = {}
        self.rectangles: List[Rectangle] = []

        if walkable:
            self._compute_partition(walkable)

        self.num_cells: int = len(self.rectangles)

        # Computed by _compute_adjacency_and_borders
        self.adjacency: Dict[int, FrozenSet[int]] = {}
        self._border_pairs: Dict[
            Tuple[int, int],
            List[Tuple[Tuple[int, int], Tuple[int, int]]]
        ] = {}
        self._compute_adjacency_and_borders()

    # ------------------------------------------------------------------
    # Public query API
    # ------------------------------------------------------------------

    def cell_of(self, x: int, y: int) -> int:
        """Return the macro-cell index containing position (x, y).

        Raises:
            KeyError: If (x, y) is not in the partition.
        """
        return self._cell_map[(x, y)]

    def cell_positions(self, cell_index: int) -> FrozenSet[Tuple[int, int]]:
        """Return all (x, y) positions belonging to *cell_index*."""
        x_min, y_min, x_max, y_max = self.rectangles[cell_index]
        return frozenset(
            (x, y)
            for x in range(x_min, x_max + 1)
            for y in range(y_min, y_max + 1)
        )

    def border_pairs(
        self, cell_i: int, cell_j: int,
    ) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """Return border cell pairs between adjacent macro-cells *i* and *j*.

        Each pair is ``((x1, y1), (x2, y2))`` where ``(x1, y1)`` is in
        *cell_i* and ``(x2, y2)`` is in *cell_j*, and the two positions
        are grid-neighbours (horizontally or vertically adjacent).

        Returns an empty list if the cells are not adjacent.
        """
        key = (min(cell_i, cell_j), max(cell_i, cell_j))
        pairs = self._border_pairs.get(key, [])
        if cell_i <= cell_j:
            return list(pairs)
        return [(b, a) for a, b in pairs]

    def estimated_distance(self, cell_i: int, cell_j: int) -> float:
        """Manhattan distance between the centres of two macro-cells."""
        if cell_i == cell_j:
            return 0.0
        ri = self.rectangles[cell_i]
        rj = self.rectangles[cell_j]
        ci_x = (ri[0] + ri[2]) / 2.0
        ci_y = (ri[1] + ri[3]) / 2.0
        cj_x = (rj[0] + rj[2]) / 2.0
        cj_y = (rj[1] + rj[3]) / 2.0
        return abs(ci_x - cj_x) + abs(ci_y - cj_y)

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_grid(
        cls,
        grid,
        width: int,
        height: int,
        *,
        excluded_types: FrozenSet[str] = frozenset({'wall', 'lava'}),
        seed: Optional[int] = None,
    ) -> 'CellPartition':
        """Create a partition from a grid with ``.get(x, y)`` access.

        Args:
            grid: Object supporting ``grid.get(x, y)`` that returns
                ``None`` for empty cells or an object with a ``.type``
                attribute.
            width: Grid width.
            height: Grid height.
            excluded_types: Cell types to exclude from the partition.
            seed: Random seed for tie-breaking.

        Returns:
            A ``CellPartition`` covering all non-excluded cells.
        """
        walkable: Set[Tuple[int, int]] = set()
        for x in range(width):
            for y in range(height):
                cell = grid.get(x, y)
                cell_type = getattr(cell, 'type', None) if cell is not None else None
                if cell_type not in excluded_types:
                    walkable.add((x, y))
        return cls(walkable, seed=seed)

    # ------------------------------------------------------------------
    # Internal: agglomerative clustering
    # ------------------------------------------------------------------

    def _compute_partition(self, walkable: Set[Tuple[int, int]]) -> None:
        """Run the agglomerative rectangular clustering algorithm."""

        # 1. Initialise each walkable cell as a singleton block.
        blocks: Dict[int, Rectangle] = {}
        pos_to_block: Dict[Tuple[int, int], int] = {}
        alive: Set[int] = set()
        next_id = 0

        for pos in walkable:
            blocks[next_id] = (pos[0], pos[1], pos[0], pos[1])
            pos_to_block[pos] = next_id
            alive.add(next_id)
            next_id += 1

        # 2. Compute initial block adjacency.
        block_adj: Dict[int, Set[int]] = {bid: set() for bid in alive}
        _DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1))
        for pos in walkable:
            bid = pos_to_block[pos]
            for dx, dy in _DIRS:
                npos = (pos[0] + dx, pos[1] + dy)
                if npos in pos_to_block:
                    nbid = pos_to_block[npos]
                    if nbid != bid:
                        block_adj[bid].add(nbid)
                        block_adj[nbid].add(bid)

        # 3. Build a min-heap of candidate merges.
        #    Entry: (merged_area, random_tiebreaker, id_lo, id_hi)
        heap: list = []
        enqueued: Set[Tuple[int, int]] = set()

        def _try_enqueue(a: int, b: int) -> None:
            lo, hi = (a, b) if a < b else (b, a)
            if (lo, hi) in enqueued:
                return
            ra, rb = blocks[lo], blocks[hi]
            merged = _merged_rect(ra, rb)
            if _rect_area(merged) == _rect_area(ra) + _rect_area(rb):
                enqueued.add((lo, hi))
                heapq.heappush(
                    heap,
                    (_rect_area(merged), self._rng.random(), lo, hi),
                )

        for bid in alive:
            for nbid in block_adj[bid]:
                _try_enqueue(bid, nbid)

        # 4. Iteratively pop the minimal-area merge and apply it.
        while heap:
            _, _, a, b = heapq.heappop(heap)

            if a not in alive or b not in alive:
                continue

            ra, rb = blocks[a], blocks[b]
            merged = _merged_rect(ra, rb)
            if _rect_area(merged) != _rect_area(ra) + _rect_area(rb):
                continue  # blocks changed since enqueue; skip

            # Create the merged block.
            new_id = next_id
            next_id += 1
            blocks[new_id] = merged
            alive.add(new_id)
            alive.discard(a)
            alive.discard(b)

            # Update pos_to_block for every cell in the merged rectangle.
            for x in range(merged[0], merged[2] + 1):
                for y in range(merged[1], merged[3] + 1):
                    pos_to_block[(x, y)] = new_id

            # Update adjacency.
            new_adj = (block_adj.get(a, set()) | block_adj.get(b, set())) - {a, b}
            block_adj[new_id] = new_adj
            for nbid in new_adj:
                block_adj[nbid].discard(a)
                block_adj[nbid].discard(b)
                block_adj[nbid].add(new_id)
            block_adj.pop(a, None)
            block_adj.pop(b, None)

            # Enqueue candidate merges with the new block.
            for nbid in new_adj:
                if nbid in alive:
                    _try_enqueue(new_id, nbid)

        # 5. Collect final blocks, sort deterministically, and assign indices.
        final_blocks = sorted(
            [blocks[bid] for bid in alive],
            key=lambda r: (r[1], r[0], r[3], r[2]),  # y_min, x_min, …
        )
        self.rectangles = final_blocks

        for idx, rect in enumerate(final_blocks):
            for x in range(rect[0], rect[2] + 1):
                for y in range(rect[1], rect[3] + 1):
                    self._cell_map[(x, y)] = idx

    # ------------------------------------------------------------------
    # Internal: adjacency and border computation
    # ------------------------------------------------------------------

    def _compute_adjacency_and_borders(self) -> None:
        """Compute adjacency graph and border pairs between macro-cells."""
        adj: Dict[int, Set[int]] = {i: set() for i in range(self.num_cells)}
        borders: Dict[
            Tuple[int, int],
            List[Tuple[Tuple[int, int], Tuple[int, int]]]
        ] = {}

        _DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1))
        for pos in self._walkable:
            cell_i = self._cell_map[pos]
            for dx, dy in _DIRS:
                npos = (pos[0] + dx, pos[1] + dy)
                if npos in self._cell_map:
                    cell_j = self._cell_map[npos]
                    if cell_i != cell_j:
                        adj[cell_i].add(cell_j)
                        key = (min(cell_i, cell_j), max(cell_i, cell_j))
                        if cell_i < cell_j:
                            pair = (pos, npos)
                        else:
                            pair = (npos, pos)
                        if key not in borders:
                            borders[key] = [pair]
                        elif pair not in borders[key]:
                            borders[key].append(pair)

        self.adjacency = {i: frozenset(s) for i, s in adj.items()}
        self._border_pairs = borders

"""Tests for cell_partition.py — agglomerative rectangular partitioning.

Covers:
- Single room (all walkable → one rectangle)
- Two rooms separated by a wall row
- Two rooms connected by a door (walkable cell in the wall)
- L-shaped room (must produce ≥2 rectangles)
- Single cell and empty partition edge cases
- Partition validity properties (coverage, non-overlap, rectangularity)
- Adjacency symmetry and border pair correctness
- Distance estimates
- from_grid factory
- Deterministic output with seed
"""

import pytest
from empo.hierarchical.cell_partition import CellPartition, _rect_area


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _assert_valid_partition(part: CellPartition, walkable):
    """Assert fundamental partition invariants."""
    walkable = set(walkable)

    # Every walkable cell must be in exactly one macro-cell.
    covered = set()
    for idx in range(part.num_cells):
        positions = part.cell_positions(idx)
        assert positions, f"macro-cell {idx} is empty"
        assert positions.isdisjoint(covered), \
            f"macro-cell {idx} overlaps with previous cells"
        covered |= positions
    assert covered == walkable, \
        f"partition does not cover all walkable cells: " \
        f"missing={walkable - covered}, extra={covered - walkable}"

    # Each rectangle must be axis-aligned and consistent with cell_of.
    for idx, (x0, y0, x1, y1) in enumerate(part.rectangles):
        assert x0 <= x1 and y0 <= y1, "invalid rectangle bounds"
        for x in range(x0, x1 + 1):
            for y in range(y0, y1 + 1):
                assert part.cell_of(x, y) == idx

    # Adjacency must be symmetric.
    for i, neighbours in part.adjacency.items():
        for j in neighbours:
            assert i in part.adjacency[j], \
                f"adjacency not symmetric: {i}→{j} but not {j}→{i}"

    # Border pairs must reference cells from the correct macro-cells.
    for i in range(part.num_cells):
        for j in part.adjacency[i]:
            pairs = part.border_pairs(i, j)
            for pos_i, pos_j in pairs:
                assert part.cell_of(*pos_i) == i
                assert part.cell_of(*pos_j) == j
                # Must be grid neighbours.
                assert abs(pos_i[0] - pos_j[0]) + abs(pos_i[1] - pos_j[1]) == 1

    # num_cells consistency
    assert part.num_cells == len(part.rectangles)


def _rect_from_corners(x0, y0, x1, y1):
    """Return the set of (x, y) positions in a rectangle."""
    return {(x, y) for x in range(x0, x1 + 1) for y in range(y0, y1 + 1)}


# ---------------------------------------------------------------------------
# Tests — basic layouts
# ---------------------------------------------------------------------------

class TestSingleRoom:
    """All walkable cells form one rectangle → single macro-cell."""

    def test_3x3(self):
        walkable = _rect_from_corners(0, 0, 2, 2)
        part = CellPartition(walkable, seed=42)
        _assert_valid_partition(part, walkable)
        assert part.num_cells == 1
        assert part.rectangles[0] == (0, 0, 2, 2)
        assert part.adjacency[0] == frozenset()

    def test_5x3(self):
        walkable = _rect_from_corners(0, 0, 4, 2)
        part = CellPartition(walkable, seed=42)
        _assert_valid_partition(part, walkable)
        assert part.num_cells == 1

    def test_1x5(self):
        walkable = _rect_from_corners(0, 0, 0, 4)
        part = CellPartition(walkable, seed=42)
        _assert_valid_partition(part, walkable)
        assert part.num_cells == 1


class TestTwoRooms:
    """Two rectangular rooms separated by a full wall row (gap in walkable)."""

    def test_horizontal_split(self):
        """Two 3x2 rooms separated by a missing row y=2."""
        top = _rect_from_corners(0, 0, 2, 1)
        bottom = _rect_from_corners(0, 3, 2, 4)
        walkable = top | bottom
        part = CellPartition(walkable, seed=42)
        _assert_valid_partition(part, walkable)
        assert part.num_cells == 2
        # No adjacency because rooms are separated.
        for i in range(part.num_cells):
            assert part.adjacency[i] == frozenset()

    def test_vertical_split(self):
        """Two 2x3 rooms separated by a missing column x=2."""
        left = _rect_from_corners(0, 0, 1, 2)
        right = _rect_from_corners(3, 0, 4, 2)
        walkable = left | right
        part = CellPartition(walkable, seed=42)
        _assert_valid_partition(part, walkable)
        assert part.num_cells == 2
        for i in range(part.num_cells):
            assert part.adjacency[i] == frozenset()


class TestTwoRoomsWithDoor:
    """Two rooms connected by a single walkable cell (door position)."""

    def test_door_connects_rooms(self):
        r"""Layout (W=wall, .=walkable):

        . . . W . . .
        . . . D . . .
        . . . W . . .

        D is a door at (3,1) — walkable.
        """
        left = _rect_from_corners(0, 0, 2, 2)
        right = _rect_from_corners(4, 0, 6, 2)
        door = {(3, 1)}
        walkable = left | right | door
        part = CellPartition(walkable, seed=42)
        _assert_valid_partition(part, walkable)

        # The door cell prevents the two rooms from merging into one
        # rectangle, so we expect ≥2 macro-cells.
        assert part.num_cells >= 2

        # The door cell's macro-cell should be adjacent to at least one
        # macro-cell from each room.
        door_cell = part.cell_of(3, 1)
        left_cell = part.cell_of(1, 1)
        right_cell = part.cell_of(5, 1)

        # There should be connectivity from left to right via door.
        # Either door merged with one side or is its own cell.
        if left_cell != door_cell and right_cell != door_cell:
            # Door is its own cell — must be adjacent to both.
            assert left_cell in part.adjacency[door_cell]
            assert right_cell in part.adjacency[door_cell]


class TestLShape:
    """L-shaped room cannot be a single rectangle."""

    def test_l_shape(self):
        r"""Layout:

        . .
        . .
        . . . .
        """
        walkable = (
            _rect_from_corners(0, 0, 1, 1) |
            _rect_from_corners(0, 2, 3, 2)
        )
        part = CellPartition(walkable, seed=42)
        _assert_valid_partition(part, walkable)
        # An L-shape needs at least 2 rectangles.
        assert part.num_cells >= 2


class TestUShape:
    """U-shaped room."""

    def test_u_shape(self):
        r"""Layout:

        . . . . .
        .       .
        . . . . .
        """
        walkable = (
            _rect_from_corners(0, 0, 4, 0) |    # top row
            {(0, 1), (4, 1)} |                    # sides
            _rect_from_corners(0, 2, 4, 2)        # bottom row
        )
        part = CellPartition(walkable, seed=42)
        _assert_valid_partition(part, walkable)
        assert part.num_cells >= 2


# ---------------------------------------------------------------------------
# Tests — edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def test_single_cell(self):
        walkable = {(5, 7)}
        part = CellPartition(walkable, seed=42)
        _assert_valid_partition(part, walkable)
        assert part.num_cells == 1
        assert part.rectangles[0] == (5, 7, 5, 7)
        assert part.cell_of(5, 7) == 0

    def test_empty(self):
        part = CellPartition(set(), seed=42)
        assert part.num_cells == 0
        assert part.rectangles == []
        assert part.adjacency == {}

    def test_diagonal_cells(self):
        """Two diagonally adjacent cells cannot merge."""
        walkable = {(0, 0), (1, 1)}
        part = CellPartition(walkable, seed=42)
        _assert_valid_partition(part, walkable)
        assert part.num_cells == 2
        # No adjacency — diagonal doesn't count.
        for i in range(2):
            assert part.adjacency[i] == frozenset()

    def test_two_adjacent_cells(self):
        walkable = {(0, 0), (1, 0)}
        part = CellPartition(walkable, seed=42)
        _assert_valid_partition(part, walkable)
        assert part.num_cells == 1
        assert _rect_area(part.rectangles[0]) == 2


# ---------------------------------------------------------------------------
# Tests — partition properties
# ---------------------------------------------------------------------------

class TestPartitionProperties:
    """Verify structural properties hold on a more complex layout."""

    @pytest.fixture
    def complex_partition(self):
        r"""Multi-room layout:

        Rows 0-2: 5-wide room
        Row 3:    wall except x=2 (door)
        Rows 4-6: 5-wide room
        """
        top = _rect_from_corners(0, 0, 4, 2)
        door = {(2, 3)}
        bottom = _rect_from_corners(0, 4, 4, 6)
        walkable = top | door | bottom
        return CellPartition(walkable, seed=42), walkable

    def test_full_coverage(self, complex_partition):
        part, walkable = complex_partition
        covered = set()
        for i in range(part.num_cells):
            covered |= part.cell_positions(i)
        assert covered == walkable

    def test_non_overlap(self, complex_partition):
        part, _ = complex_partition
        all_pos = []
        for i in range(part.num_cells):
            all_pos.extend(part.cell_positions(i))
        assert len(all_pos) == len(set(all_pos))

    def test_rectangularity(self, complex_partition):
        part, _ = complex_partition
        for rect in part.rectangles:
            assert _rect_area(rect) > 0
            x0, y0, x1, y1 = rect
            assert x0 <= x1 and y0 <= y1

    def test_cell_of_inverse(self, complex_partition):
        part, walkable = complex_partition
        for pos in walkable:
            idx = part.cell_of(*pos)
            assert pos in part.cell_positions(idx)

    def test_cell_of_raises_for_non_walkable(self, complex_partition):
        part, _ = complex_partition
        with pytest.raises(KeyError):
            part.cell_of(0, 3)  # wall position
        with pytest.raises(KeyError):
            part.cell_of(100, 100)  # out of bounds


# ---------------------------------------------------------------------------
# Tests — adjacency and borders
# ---------------------------------------------------------------------------

class TestAdjacencyAndBorders:

    def test_adjacent_rooms(self):
        """Two rooms sharing a complete edge are adjacent."""
        left = _rect_from_corners(0, 0, 1, 2)
        right = _rect_from_corners(2, 0, 3, 2)
        walkable = left | right
        part = CellPartition(walkable, seed=42)
        _assert_valid_partition(part, walkable)
        # The rooms should merge since their union is a rectangle.
        if part.num_cells == 2:
            assert 1 in part.adjacency[0]
            assert 0 in part.adjacency[1]

    def test_border_pairs_content(self):
        """Border pairs between adjacent cells are valid grid-neighbour pairs."""
        # An L-shape guarantees ≥2 macro-cells with known adjacency.
        top = _rect_from_corners(0, 0, 2, 1)    # 3×2
        leg = _rect_from_corners(0, 2, 0, 3)    # 1×2
        walkable = top | leg
        part = CellPartition(walkable, seed=0)
        _assert_valid_partition(part, walkable)
        assert part.num_cells >= 2

        # Every border pair must have cells from the correct macro-cells
        # and must be grid-neighbours.
        for i in range(part.num_cells):
            for j in part.adjacency[i]:
                for pos_i, pos_j in part.border_pairs(i, j):
                    assert part.cell_of(*pos_i) == i
                    assert part.cell_of(*pos_j) == j
                    assert abs(pos_i[0] - pos_j[0]) + abs(pos_i[1] - pos_j[1]) == 1

    def test_border_pairs_between_separated_rooms(self):
        """Two rooms connected by a narrow bridge."""
        left = _rect_from_corners(0, 0, 2, 2)
        bridge = {(3, 1)}
        right = _rect_from_corners(4, 0, 6, 2)
        walkable = left | right | bridge
        part = CellPartition(walkable, seed=42)
        _assert_valid_partition(part, walkable)

        bridge_cell = part.cell_of(3, 1)
        left_cell = part.cell_of(1, 1)
        right_cell = part.cell_of(5, 1)

        # Bridge should be adjacent to at least one room on each side.
        if bridge_cell != left_cell:
            pairs = part.border_pairs(bridge_cell, left_cell)
            assert len(pairs) > 0
        if bridge_cell != right_cell:
            pairs = part.border_pairs(bridge_cell, right_cell)
            assert len(pairs) > 0

    def test_border_pairs_order(self):
        """border_pairs(i, j) swaps correctly vs border_pairs(j, i)."""
        top = _rect_from_corners(0, 0, 2, 0)
        bottom = _rect_from_corners(0, 2, 2, 2)
        bridge = {(1, 1)}
        walkable = top | bottom | bridge
        part = CellPartition(walkable, seed=42)
        _assert_valid_partition(part, walkable)

        for i in range(part.num_cells):
            for j in part.adjacency[i]:
                pairs_ij = part.border_pairs(i, j)
                pairs_ji = part.border_pairs(j, i)
                # Swapped pairs should match.
                assert len(pairs_ij) == len(pairs_ji)
                for (a, b), (c, d) in zip(sorted(pairs_ij), sorted(pairs_ji)):
                    assert a == d and b == c


# ---------------------------------------------------------------------------
# Tests — distance estimation
# ---------------------------------------------------------------------------

class TestDistanceEstimation:

    def test_same_cell(self):
        walkable = _rect_from_corners(0, 0, 2, 2)
        part = CellPartition(walkable, seed=42)
        assert part.estimated_distance(0, 0) == 0.0

    def test_adjacent_cells(self):
        """Distance between adjacent 1x1 cells should be 1."""
        walkable = {(0, 0), (1, 0)}
        # These merge into one cell, so use a case that doesn't merge.
        # Use an L-shape to get two cells.
        top = _rect_from_corners(0, 0, 2, 0)
        bottom = _rect_from_corners(0, 1, 0, 1)
        walkable = top | bottom
        part = CellPartition(walkable, seed=42)
        if part.num_cells >= 2:
            c0 = part.cell_of(1, 0)
            c1 = part.cell_of(0, 1)
            if c0 != c1:
                dist = part.estimated_distance(c0, c1)
                assert dist > 0


# ---------------------------------------------------------------------------
# Tests — deterministic seed
# ---------------------------------------------------------------------------

class TestSeedDeterminism:

    def test_same_seed_same_result(self):
        """Partitions with the same seed must produce identical results."""
        walkable = (
            _rect_from_corners(0, 0, 3, 1) |
            _rect_from_corners(0, 2, 1, 3)
        )
        p1 = CellPartition(walkable, seed=123)
        p2 = CellPartition(walkable, seed=123)
        assert p1.rectangles == p2.rectangles
        assert p1.num_cells == p2.num_cells
        for pos in walkable:
            assert p1.cell_of(*pos) == p2.cell_of(*pos)


# ---------------------------------------------------------------------------
# Tests — from_grid factory
# ---------------------------------------------------------------------------

class _FakeObj:
    """Minimal object with a .type attribute."""
    def __init__(self, obj_type):
        self.type = obj_type


class _FakeGrid:
    """Minimal grid with .get(x, y) access."""
    def __init__(self, width, height, cells=None):
        self.width = width
        self.height = height
        self._cells = cells or {}

    def get(self, x, y):
        return self._cells.get((x, y))


class TestFromGrid:

    def test_empty_grid(self):
        """Grid with all None (empty) cells → all walkable."""
        grid = _FakeGrid(3, 3)
        part = CellPartition.from_grid(grid, 3, 3, seed=42)
        assert part.num_cells == 1
        assert _rect_area(part.rectangles[0]) == 9

    def test_walls_excluded(self):
        """Walls are excluded from the partition."""
        cells = {
            (0, 0): _FakeObj('wall'),
            (1, 0): _FakeObj('wall'),
            (2, 0): _FakeObj('wall'),
        }
        grid = _FakeGrid(3, 3, cells)
        part = CellPartition.from_grid(grid, 3, 3, seed=42)
        _assert_valid_partition(part, {
            (x, y) for x in range(3) for y in range(3)
            if (x, y) not in cells
        })
        # (0,0), (1,0), (2,0) are walls, 6 walkable cells remain.
        total = sum(_rect_area(r) for r in part.rectangles)
        assert total == 6

    def test_lava_excluded(self):
        """Lava cells are excluded from the partition."""
        cells = {(1, 1): _FakeObj('lava')}
        grid = _FakeGrid(3, 3, cells)
        part = CellPartition.from_grid(grid, 3, 3, seed=42)
        total = sum(_rect_area(r) for r in part.rectangles)
        assert total == 8
        with pytest.raises(KeyError):
            part.cell_of(1, 1)

    def test_doors_included(self):
        """Door cells are included (not excluded by default)."""
        cells = {(1, 1): _FakeObj('door')}
        grid = _FakeGrid(3, 3, cells)
        part = CellPartition.from_grid(grid, 3, 3, seed=42)
        assert part.cell_of(1, 1) is not None
        total = sum(_rect_area(r) for r in part.rectangles)
        assert total == 9

    def test_custom_excluded_types(self):
        """Custom excluded_types parameter."""
        cells = {(0, 0): _FakeObj('danger')}
        grid = _FakeGrid(3, 3, cells)
        # By default, 'danger' is NOT excluded.
        p1 = CellPartition.from_grid(grid, 3, 3, seed=42)
        assert sum(_rect_area(r) for r in p1.rectangles) == 9
        # Exclude 'danger' explicitly.
        p2 = CellPartition.from_grid(
            grid, 3, 3,
            excluded_types=frozenset({'wall', 'lava', 'danger'}),
            seed=42,
        )
        assert sum(_rect_area(r) for r in p2.rectangles) == 8

    def test_walled_rooms(self):
        """Realistic two-room layout with walls and a door."""
        # 7x5 grid:
        #   W W W W W W W
        #   W . . W . . W    (row y=1)
        #   W . . D . . W    (row y=2, door at x=3)
        #   W . . W . . W    (row y=3)
        #   W W W W W W W
        cells = {}
        for x in range(7):
            cells[(x, 0)] = _FakeObj('wall')
            cells[(x, 4)] = _FakeObj('wall')
        for y in range(5):
            cells[(0, y)] = _FakeObj('wall')
            cells[(6, y)] = _FakeObj('wall')
        # Middle wall column at x=3, except door at (3,2).
        cells[(3, 1)] = _FakeObj('wall')
        cells[(3, 3)] = _FakeObj('wall')
        cells[(3, 2)] = _FakeObj('door')  # door — walkable

        grid = _FakeGrid(7, 5, cells)
        part = CellPartition.from_grid(grid, 7, 5, seed=42)

        # Walkable: 5*3 interior minus 2 wall cells at (3,1), (3,3) = 15 - 2 = 13
        # Actually, interior is y=1..3, x=1..5 = 5*3 = 15.
        # Minus walls at (3,1) and (3,3) = 13 walkable.
        walkable = {
            (x, y) for x in range(7) for y in range(5)
            if (x, y) not in cells or cells[(x, y)].type not in ('wall', 'lava')
        }
        assert len(walkable) == 13
        _assert_valid_partition(part, walkable)

        # The door cell's macro-cell index should be valid.
        door_idx = part.cell_of(3, 2)
        assert 0 <= door_idx < part.num_cells

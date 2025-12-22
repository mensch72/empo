"""
Test map parsing functionality for MultiGridEnv.
"""
import sys
import os

import numpy as np
from gym_multigrid.multigrid import (
    MultiGridEnv, Grid, Agent, Block, Rock, Wall, World,
    parse_map_string, create_object_from_spec
)


def test_parse_map_string_basic():
    """Test basic map string parsing."""
    test_map = """
    We We We We We
    We .. .. .. We
    We .. Ar .. We
    We .. .. .. We
    We We We We We
    """
    
    width, height, cells, agents = parse_map_string(test_map, World)
    
    assert width == 5
    assert height == 5
    assert len(agents) == 1
    assert agents[0] == (2, 2, {'color': 'red'})


def test_parse_map_string_no_whitespace():
    """Test map parsing without whitespace between cells."""
    test_map = """
    WeWeWeWeWe
    We......We
    We..Ar..We
    We......We
    WeWeWeWeWe
    """
    
    width, height, cells, agents = parse_map_string(test_map, World)
    
    assert width == 5
    assert height == 5
    assert len(agents) == 1


def test_parse_map_list_of_strings():
    """Test parsing with list of strings."""
    test_map = [
        "We We We We We",
        "We .. .. .. We",
        "We .. Ar .. We",
        "We .. .. .. We",
        "We We We We We"
    ]
    
    width, height, cells, agents = parse_map_string(test_map, World)
    
    assert width == 5
    assert height == 5


def test_parse_map_list_of_lists():
    """Test parsing with list of lists."""
    test_map = [
        ['We', 'We', 'We', 'We', 'We'],
        ['We', '..', '..', '..', 'We'],
        ['We', '..', 'Ar', '..', 'We'],
        ['We', '..', '..', '..', 'We'],
        ['We', 'We', 'We', 'We', 'We']
    ]
    
    width, height, cells, agents = parse_map_string(test_map, World)
    
    assert width == 5
    assert height == 5


def test_all_color_codes():
    """Test all color codes work correctly."""
    test_map = """
    We We We We We We We We
    We Ar Ag Ab Ap Ay Ae We
    We We We We We We We We
    """
    
    width, height, cells, agents = parse_map_string(test_map, World)
    
    assert len(agents) == 6
    expected_colors = ['red', 'green', 'blue', 'purple', 'yellow', 'grey']
    for i, (x, y, params) in enumerate(agents):
        assert params['color'] == expected_colors[i]


def test_all_object_types():
    """Test all object types are parsed correctly."""
    test_map = """
    We We We We We We We We We We
    We Bl Ro La Sw Un .. .. .. We
    We Gr Kr Br Xr .. .. .. .. We
    We Lr Cr Or .. .. .. .. .. We
    We Mn Ms Mw Me Ma .. .. .. We
    We .. Ar .. .. .. .. .. .. We
    We We We We We We We We We We
    """
    
    width, height, cells, agents = parse_map_string(test_map, World)
    
    # Check block
    assert cells[1][1] == ('block', {})
    
    # Check rock
    assert cells[1][2] == ('rock', {})
    
    # Check lava
    assert cells[1][3] == ('lava', {})
    
    # Check switch
    assert cells[1][4] == ('switch', {})
    
    # Check unsteady
    assert cells[1][5] == ('unsteady', {})
    
    # Check goal
    assert cells[2][1] == ('goal', {'color': 'red'})
    
    # Check key
    assert cells[2][2] == ('key', {'color': 'red'})
    
    # Check ball
    assert cells[2][3] == ('ball', {'color': 'red'})
    
    # Check box
    assert cells[2][4] == ('box', {'color': 'red'})
    
    # Check locked door
    assert cells[3][1] == ('door', {'color': 'red', 'is_locked': True, 'is_open': False})
    
    # Check closed door
    assert cells[3][2] == ('door', {'color': 'red', 'is_locked': False, 'is_open': False})
    
    # Check open door
    assert cells[3][3] == ('door', {'color': 'red', 'is_locked': False, 'is_open': True})
    
    # Check magic walls
    assert cells[4][1] == ('magicwall', {'magic_side': 3})  # north
    assert cells[4][2] == ('magicwall', {'magic_side': 1})  # south
    assert cells[4][3] == ('magicwall', {'magic_side': 2})  # west
    assert cells[4][4] == ('magicwall', {'magic_side': 0})  # east
    assert cells[4][5] == ('magicwall', {'magic_side': 4})  # all


def test_create_environment_with_map():
    """Test creating an environment with a map specification."""
    test_map = """
    We We We We We We We
    We .. .. .. .. .. We
    We .. Ar .. Ag .. We
    We .. .. .. .. .. We
    We .. Bl .. Ro .. We
    We .. .. .. .. .. We
    We We We We We We We
    """
    
    env = MultiGridEnv(
        map=test_map,
        max_steps=100,
        partial_obs=False,
        objects_set=World
    )
    
    assert env.width == 7
    assert env.height == 7
    assert len(env.agents) == 2
    
    # Check agent positions
    assert np.array_equal(env.agents[0].pos, [2, 2])
    assert np.array_equal(env.agents[1].pos, [4, 2])
    
    # Check agent colors
    assert env.agents[0].color == 'red'
    assert env.agents[1].color == 'green'
    
    # Check block
    block = env.grid.get(2, 4)
    assert block is not None
    assert block.type == 'block'
    
    # Check rock
    rock = env.grid.get(4, 4)
    assert rock is not None
    assert rock.type == 'rock'


def test_environment_with_doors():
    """Test environment with all door types."""
    test_map = """
    We We We We We We We
    We Lr .. Cr .. Or We
    We .. .. .. .. .. We
    We .. .. Ar .. .. We
    We .. .. .. .. .. We
    We We We We We We We
    """
    
    env = MultiGridEnv(
        map=test_map,
        max_steps=100,
        partial_obs=False,
        objects_set=World
    )
    
    # Check locked door
    locked_door = env.grid.get(1, 1)
    assert locked_door is not None
    assert locked_door.type == 'door'
    assert locked_door.is_locked
    assert not locked_door.is_open
    
    # Check closed door
    closed_door = env.grid.get(3, 1)
    assert closed_door is not None
    assert closed_door.type == 'door'
    assert not closed_door.is_locked
    assert not closed_door.is_open
    
    # Check open door
    open_door = env.grid.get(5, 1)
    assert open_door is not None
    assert open_door.type == 'door'
    assert not open_door.is_locked
    assert open_door.is_open


def test_environment_with_magic_walls():
    """Test environment with magic walls in all directions."""
    test_map = """
    We We We We We We
    We Mn .. Ms .. We
    We .. .. .. .. We
    We Mw .. Me .. We
    We .. Ar .. .. We
    We We We We We We
    """
    
    env = MultiGridEnv(
        map=test_map,
        max_steps=100,
        partial_obs=False,
        objects_set=World
    )
    
    # Check north magic wall
    mw_north = env.grid.get(1, 1)
    assert mw_north is not None
    assert mw_north.type == 'magicwall'
    assert mw_north.magic_side == 3  # north/up
    
    # Check south magic wall
    mw_south = env.grid.get(3, 1)
    assert mw_south is not None
    assert mw_south.type == 'magicwall'
    assert mw_south.magic_side == 1  # south/down
    
    # Check west magic wall
    mw_west = env.grid.get(1, 3)
    assert mw_west is not None
    assert mw_west.type == 'magicwall'
    assert mw_west.magic_side == 2  # west/left
    
    # Check east magic wall
    mw_east = env.grid.get(3, 3)
    assert mw_east is not None
    assert mw_east.type == 'magicwall'
    assert mw_east.magic_side == 0  # east/right


def test_environment_step():
    """Test that environment with map can be stepped."""
    test_map = """
    We We We We We
    We .. .. .. We
    We .. Ar .. We
    We .. .. .. We
    We We We We We
    """
    
    env = MultiGridEnv(
        map=test_map,
        max_steps=100,
        partial_obs=False,
        objects_set=World,
        orientations=['e']  # Explicitly set orientation to east for predictable test
    )
    
    # Take a step
    obs, rewards, done, info = env.step([3])  # forward
    
    # Agent should have moved east (from [2,2] to [3,2])
    assert np.array_equal(env.agents[0].pos, [3, 2])


def test_environment_reset():
    """Test that environment with map can be reset."""
    test_map = """
    We We We We We
    We .. .. .. We
    We .. Ar .. We
    We .. .. .. We
    We We We We We
    """
    
    env = MultiGridEnv(
        map=test_map,
        max_steps=100,
        partial_obs=False,
        objects_set=World,
        orientations=['e']  # Explicitly set orientation for predictable test
    )
    
    # Take some steps
    env.step([3])  # forward
    env.step([3])  # forward
    
    # Reset
    env.reset()
    
    # Agent should be back at original position
    assert np.array_equal(env.agents[0].pos, [2, 2])


def test_one_or_three_chambers_map_env():
    """Test that OneOrThreeChambersMapEnv produces the same layout as the original."""
    from envs.one_or_three_chambers import OneOrThreeChambersEnv, OneOrThreeChambersMapEnv
    
    # Create both environments
    env_original = OneOrThreeChambersEnv()
    env_map = OneOrThreeChambersMapEnv()
    
    # Check dimensions match
    assert env_original.width == env_map.width
    assert env_original.height == env_map.height
    
    # Check agent counts match
    assert len(env_original.agents) == len(env_map.agents)
    assert env_original.num_humans == env_map.num_humans
    assert env_original.num_robots == env_map.num_robots
    
    # Check that walls are in the same positions
    for y in range(env_original.height):
        for x in range(env_original.width):
            orig_cell = env_original.grid.get(x, y)
            map_cell = env_map.grid.get(x, y)
            
            orig_is_wall = orig_cell is not None and orig_cell.type == 'wall'
            map_is_wall = map_cell is not None and map_cell.type == 'wall'
            
            assert orig_is_wall == map_is_wall, f"Wall mismatch at ({x}, {y})"
    
    # Check that block and rock are in the same positions
    orig_block = None
    orig_rock = None
    map_block = None
    map_rock = None
    
    for y in range(env_original.height):
        for x in range(env_original.width):
            orig_cell = env_original.grid.get(x, y)
            map_cell = env_map.grid.get(x, y)
            
            if orig_cell and orig_cell.type == 'block':
                orig_block = (x, y)
            if orig_cell and orig_cell.type == 'rock':
                orig_rock = (x, y)
            if map_cell and map_cell.type == 'block':
                map_block = (x, y)
            if map_cell and map_cell.type == 'rock':
                map_rock = (x, y)
    
    assert orig_block == map_block, f"Block position mismatch: {orig_block} vs {map_block}"
    assert orig_rock == map_rock, f"Rock position mismatch: {orig_rock} vs {map_rock}"


def test_orientations_parameter():
    """Test that orientations parameter sets agent directions correctly."""
    test_map = """
    We We We We We We We
    We .. Ar .. Ag .. We
    We .. .. .. .. .. We
    We We We We We We We
    """
    
    # Test with explicit orientations
    env = MultiGridEnv(
        map=test_map,
        max_steps=100,
        partial_obs=False,
        objects_set=World,
        orientations=['n', 's']  # north and south
    )
    
    # Check orientations: n=3 (north/up), s=1 (south/down)
    assert env.agents[0].dir == 3, f"Expected dir=3 (north), got {env.agents[0].dir}"
    assert env.agents[1].dir == 1, f"Expected dir=1 (south), got {env.agents[1].dir}"
    
    # Test with all orientations
    env2 = MultiGridEnv(
        map=test_map,
        max_steps=100,
        partial_obs=False,
        objects_set=World,
        orientations=['e', 'w']  # east and west
    )
    
    assert env2.agents[0].dir == 0, f"Expected dir=0 (east), got {env2.agents[0].dir}"
    assert env2.agents[1].dir == 2, f"Expected dir=2 (west), got {env2.agents[1].dir}"


def test_can_push_rocks_parameter():
    """Test that can_push_rocks parameter controls which agents can push rocks."""
    test_map = """
    We We We We We We We
    We .. Ar Ro .. .. We
    We .. Ae Ro .. .. We
    We We We We We We We
    """
    
    # Default: only grey (e) agents can push rocks
    env = MultiGridEnv(
        map=test_map,
        max_steps=100,
        partial_obs=False,
        objects_set=World,
        orientations=['e', 'e']  # Both facing east
    )
    
    # Check that rocks exist
    rock1 = env.grid.get(3, 1)
    rock2 = env.grid.get(3, 2)
    
    assert rock1 is not None and rock1.type == 'rock'
    assert rock2 is not None and rock2.type == 'rock'
    
    # Grey agent (index 1) should have can_push_rocks=True, red agent (index 0) should not
    assert not env.agents[0].can_push_rocks, "Red agent should not have can_push_rocks"
    assert env.agents[1].can_push_rocks, "Grey agent should have can_push_rocks"
    
    # Verify this affects rock pushing
    assert not rock1.can_be_pushed_by(env.agents[0]), "Red agent should not push rocks"
    assert rock1.can_be_pushed_by(env.agents[1]), "Grey agent should be able to push rocks"


def test_can_push_rocks_multiple_colors():
    """Test that can_push_rocks with multiple color codes works."""
    test_map = """
    We We We We We We We
    We .. Ar Ro .. .. We
    We .. Ag Ro .. .. We
    We We We We We We We
    """
    
    # Both red and green agents can push rocks
    env = MultiGridEnv(
        map=test_map,
        max_steps=100,
        partial_obs=False,
        objects_set=World,
        orientations=['e', 'e'],
        can_push_rocks='rg'  # red and green
    )
    
    rock1 = env.grid.get(3, 1)
    
    # Both agents should be able to push
    assert rock1.can_be_pushed_by(env.agents[0]), "Red agent should push rocks when 'r' in can_push_rocks"
    assert rock1.can_be_pushed_by(env.agents[1]), "Green agent should push rocks when 'g' in can_push_rocks"


if __name__ == '__main__':
    test_parse_map_string_basic()
    print("✓ test_parse_map_string_basic passed")
    
    test_parse_map_string_no_whitespace()
    print("✓ test_parse_map_string_no_whitespace passed")
    
    test_parse_map_list_of_strings()
    print("✓ test_parse_map_list_of_strings passed")
    
    test_parse_map_list_of_lists()
    print("✓ test_parse_map_list_of_lists passed")
    
    test_all_color_codes()
    print("✓ test_all_color_codes passed")
    
    test_all_object_types()
    print("✓ test_all_object_types passed")
    
    test_create_environment_with_map()
    print("✓ test_create_environment_with_map passed")
    
    test_environment_with_doors()
    print("✓ test_environment_with_doors passed")
    
    test_environment_with_magic_walls()
    print("✓ test_environment_with_magic_walls passed")
    
    test_environment_step()
    print("✓ test_environment_step passed")
    
    test_environment_reset()
    print("✓ test_environment_reset passed")
    
    test_one_or_three_chambers_map_env()
    print("✓ test_one_or_three_chambers_map_env passed")
    
    print("\nAll tests passed! ✓")

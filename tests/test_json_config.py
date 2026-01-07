"""
Test config file loading for MultiGridEnv (JSON and YAML).
"""
import os
import json
import tempfile

from gym_multigrid.multigrid import MultiGridEnv


def test_load_config_file_basic():
    """Test loading a basic config file."""
    config_path = os.path.join(
        os.path.dirname(__file__), '..', 'multigrid_worlds', 'basic', 'empty_5x5.yaml'
    )
    
    env = MultiGridEnv(config_file=config_path)
    
    assert env.width == 5, f"Expected width 5, got {env.width}"
    assert env.height == 5, f"Expected height 5, got {env.height}"
    assert len(env.agents) == 1, f"Expected 1 agent, got {len(env.agents)}"
    assert env.max_steps == 100, f"Expected max_steps 100, got {env.max_steps}"
    
    # Check metadata was stored
    assert hasattr(env, '_config_metadata')
    assert env._config_metadata.get('name') == 'Empty 5x5'
    
    print("✓ test_load_config_file_basic passed")


def test_load_config_file_with_override():
    """Test that __init__ params override config file values."""
    config_path = os.path.join(
        os.path.dirname(__file__), '..', 'multigrid_worlds', 'basic', 'empty_5x5.yaml'
    )
    
    # Override max_steps from config
    env = MultiGridEnv(config_file=config_path, max_steps=50)
    
    assert env.max_steps == 50, f"Expected max_steps 50 (override), got {env.max_steps}"
    
    print("✓ test_load_config_file_with_override passed")


def test_load_config_file_multiagent():
    """Test loading a multi-agent config file."""
    config_path = os.path.join(
        os.path.dirname(__file__), '..', 'multigrid_worlds', 'basic', 'two_agents.yaml'
    )
    
    env = MultiGridEnv(config_file=config_path)
    
    assert len(env.agents) == 2, f"Expected 2 agents, got {len(env.agents)}"
    
    # Check agent colors
    colors = [agent.color for agent in env.agents]
    assert 'red' in colors, "Expected a red agent"
    assert 'green' in colors, "Expected a green agent"
    
    print("✓ test_load_config_file_multiagent passed")


def test_load_config_file_with_objects():
    """Test loading a config file with blocks and rocks."""
    config_path = os.path.join(
        os.path.dirname(__file__), '..', 'multigrid_worlds', 'puzzles', 'block_rock_test.yaml'
    )
    
    env = MultiGridEnv(config_file=config_path)
    
    # Check that block and rock exist in the grid
    has_block = False
    has_rock = False
    for y in range(env.height):
        for x in range(env.width):
            cell = env.grid.get(x, y)
            if cell is not None:
                if cell.type == 'block':
                    has_block = True
                if cell.type == 'rock':
                    has_rock = True
    
    assert has_block, "Expected a block in the grid"
    assert has_rock, "Expected a rock in the grid"
    
    print("✓ test_load_config_file_with_objects passed")


def test_load_config_file_large():
    """Test loading a larger multi-chamber config file."""
    config_path = os.path.join(
        os.path.dirname(__file__), '..', 'multigrid_worlds', 'puzzles', 'small_one_or_three_chambers.yaml'
    )
    
    env = MultiGridEnv(config_file=config_path)
    
    # Check environment was created
    assert env.width > 0
    assert env.height > 0
    assert len(env.agents) > 0
    
    # Check can step
    actions = [0] * len(env.agents)
    obs, rewards, done, info = env.step(actions)
    
    print("✓ test_load_config_file_large passed")


def test_load_config_file_not_found():
    """Test that FileNotFoundError is raised for missing config file."""
    try:
        env = MultiGridEnv(config_file='/nonexistent/path/config.yaml')
        assert False, "Expected FileNotFoundError"
    except FileNotFoundError:
        pass
    
    print("✓ test_load_config_file_not_found passed")


def test_load_config_file_invalid_json():
    """Test that ValueError is raised for invalid JSON."""
    # Create a temp file with invalid JSON
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        f.write("{ invalid json }")
        temp_path = f.name
    
    try:
        env = MultiGridEnv(config_file=temp_path)
        assert False, "Expected ValueError for invalid JSON"
    except ValueError as e:
        assert "Invalid JSON" in str(e)
    finally:
        os.unlink(temp_path)
    
    print("✓ test_load_config_file_invalid_json passed")


def test_load_config_file_missing_map():
    """Test that ValueError is raised for config without map field."""
    # Create a temp file without map field
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump({"max_steps": 100}, f)
        temp_path = f.name
    
    try:
        env = MultiGridEnv(config_file=temp_path)
        assert False, "Expected ValueError for missing map field"
    except ValueError as e:
        assert "must contain a 'map' field" in str(e)
    finally:
        os.unlink(temp_path)
    
    print("✓ test_load_config_file_missing_map passed")


def test_config_file_all_worlds():
    """Test loading all config files in multigrid_worlds directory."""
    worlds_dir = os.path.join(os.path.dirname(__file__), '..', 'multigrid_worlds')
    
    count = 0
    for root, dirs, files in os.walk(worlds_dir):
        for filename in files:
            if filename.endswith('.yaml') or filename.endswith('.yml') or filename.endswith('.json'):
                config_path = os.path.join(root, filename)
                try:
                    env = MultiGridEnv(config_file=config_path)
                    env.reset()
                    # Take a step to verify environment is functional
                    actions = [0] * len(env.agents)
                    env.step(actions)
                    count += 1
                    print(f"  ✓ {os.path.relpath(config_path, worlds_dir)}")
                except Exception as e:
                    print(f"  ✗ {os.path.relpath(config_path, worlds_dir)}: {e}")
                    raise
    
    print(f"✓ test_config_file_all_worlds passed ({count} worlds)")


if __name__ == '__main__':
    test_load_config_file_basic()
    test_load_config_file_with_override()
    test_load_config_file_multiagent()
    test_load_config_file_with_objects()
    test_load_config_file_large()
    test_load_config_file_not_found()
    test_load_config_file_invalid_json()
    test_load_config_file_missing_map()
    
    print("\n--- Testing all world config files ---")
    test_config_file_all_worlds()
    
    print("\n" + "=" * 50)
    print("All tests passed! ✓")
    print("=" * 50)

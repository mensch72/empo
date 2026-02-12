"""
Environment YAML configs used by the learning-based solver.

Usage:
    from empo.ali_learning_based.envs import get_env_path

    env = MultiGridEnv(config_file=get_env_path("rock_gateway_demo.yaml"))
"""

from pathlib import Path

_ENV_DIR = Path(__file__).parent


def get_env_path(name: str) -> str:
    """Return the absolute path to an environment YAML file.

    Args:
        name: Filename (e.g. "phase1_test.yaml", "rock_gateway_demo.yaml").

    Returns:
        Absolute path string suitable for MultiGridEnv(config_file=...).

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    path = _ENV_DIR / name
    if not path.exists():
        available = sorted(p.name for p in _ENV_DIR.glob("*.yaml"))
        raise FileNotFoundError(
            f"Environment '{name}' not found in {_ENV_DIR}. "
            f"Available: {available}"
        )
    return str(path)


def load_env_from_yaml(name: str):
    """
    Load a MultiGridEnv from a YAML config file.

    Args:
        name: Name of the YAML file (see get_env_path for format).

    Returns:
        A MultiGridEnv instance, reset and ready to use.
    """
    from gym_multigrid.multigrid import MultiGridEnv, SmallActions

    config_path = get_env_path(name)
    env = MultiGridEnv(
        config_file=config_path,
        partial_obs=False,
        actions_set=SmallActions,
    )
    env.reset()
    return env

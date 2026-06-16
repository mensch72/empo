"""
YAML loader for BushWorld, mirroring the multigrid gridworld loader.

A BushWorld YAML file looks like::

    map: |
      Hu 01 01 Ro 01 01 Hu
    B: 1
    max_steps: 12
    fill_density: 1          # density of cells occupied by a player
    possible_goals:          # optional; defaults to every single cell
      - [0, 0]
      - [6, 0]

Map token encoding (whitespace separated, one token per cell):

- ``Ro``  : a robot body (lowest agent ids, in row-major order of appearance)
- ``Hu``  : a human (agent ids after the robots, in row-major order)
- ``.`` / ``..`` : an empty cell (bush density 0)
- a non-negative integer (e.g. ``0``, ``1``, ``12``) : a bush of that density

Cells occupied by a player carry the bush density given by ``fill_density``
(default 0), because the player token cannot also encode a number.
"""

from __future__ import annotations

import os
from typing import List, Optional, Tuple

import yaml

from empo.bushworld.env import BushWorld

Position = Tuple[int, int]


def parse_bushworld_map(
    map_spec, fill_density: int = 0
) -> Tuple[int, int, List[List[int]], List[Position], List[Position]]:
    """Parse a BushWorld map string into dimensions, densities and positions.

    Args:
        map_spec: The map as a single string (rows separated by newlines) or a
            list of row strings.
        fill_density: Density assigned to cells that contain a player token.

    Returns:
        Tuple ``(width, height, densities, robot_positions, human_positions)``
        where ``densities`` is a list of rows (each a list of ints).
    """
    if isinstance(map_spec, str):
        rows = [r for r in map_spec.splitlines() if r.strip() != ""]
    else:
        rows = [r for r in map_spec if str(r).strip() != ""]

    tokenized = [row.split() for row in rows]
    height = len(tokenized)
    if height == 0:
        raise ValueError("BushWorld map is empty")
    width = len(tokenized[0])
    if any(len(r) != width for r in tokenized):
        raise ValueError("All BushWorld map rows must have the same number of cells")

    densities: List[List[int]] = [[0] * width for _ in range(height)]
    robot_positions: List[Position] = []
    human_positions: List[Position] = []

    for y, row in enumerate(tokenized):
        for x, token in enumerate(row):
            tok = token.strip()
            if tok in (".", ".."):
                densities[y][x] = 0
            elif tok == "Ro":
                robot_positions.append((x, y))
                densities[y][x] = fill_density
            elif tok == "Hu":
                human_positions.append((x, y))
                densities[y][x] = fill_density
            else:
                try:
                    densities[y][x] = int(tok)
                except ValueError as exc:
                    raise ValueError(
                        f"Unrecognized BushWorld map token {tok!r} at ({x}, {y})"
                    ) from exc

    return width, height, densities, robot_positions, human_positions


def load_bushworld_config(config: dict, **overrides) -> BushWorld:
    """Build a :class:`BushWorld` from a parsed YAML config dict.

    Recognized keys: ``map`` (required), ``B``, ``max_steps``, ``fill_density``,
    ``possible_goals``, ``seed``, ``render_tile_size``. Any keyword in
    ``overrides`` takes precedence over the config file.
    """
    cfg = dict(config)
    cfg.update({k: v for k, v in overrides.items() if v is not None})

    if "map" not in cfg:
        raise ValueError("BushWorld config must contain a 'map' key")

    B = int(cfg.get("B", 1))
    fill_density = int(cfg.get("fill_density", 0))

    width, height, densities, robot_positions, human_positions = parse_bushworld_map(
        cfg["map"], fill_density=fill_density
    )

    if not robot_positions and not human_positions:
        raise ValueError("BushWorld map must contain at least one player (Ro/Hu)")

    possible_goals = cfg.get("possible_goals")
    if possible_goals is not None:
        possible_goals = [tuple(int(c) for c in spec) for spec in possible_goals]

    return BushWorld(
        width=width,
        height=height,
        num_robots=len(robot_positions),
        num_humans=len(human_positions),
        max_steps=int(cfg.get("max_steps", 10)),
        B=B,
        robot_positions=robot_positions,
        human_positions=human_positions,
        initial_densities=densities,
        seed=cfg.get("seed"),
        possible_goals=possible_goals,
        render_tile_size=int(cfg.get("render_tile_size", 48)),
    )


def load_bushworld(path: str, **overrides) -> BushWorld:
    """Load a :class:`BushWorld` from a YAML file.

    Args:
        path: Path to the YAML file. If it does not exist as given and does not
            end in ``.yaml``/``.yml``, a ``.yaml`` extension is appended and the
            file is also looked up under the repository ``bushworld_worlds/``
            directory.
        **overrides: Keyword overrides forwarded to :func:`load_bushworld_config`.
    """
    resolved = _resolve_world_path(path)
    with open(resolved, "r") as f:
        config = yaml.safe_load(f)
    return load_bushworld_config(config, **overrides)


def _resolve_world_path(path: str) -> str:
    candidates = [path]
    if not path.endswith((".yaml", ".yml")):
        candidates.append(path + ".yaml")
    # Also look under the repo-level bushworld_worlds directory.
    worlds_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
        "bushworld_worlds",
    )
    for c in list(candidates):
        candidates.append(os.path.join(worlds_dir, c))
    for c in candidates:
        if os.path.isfile(c):
            return c
    raise FileNotFoundError(
        f"Could not find BushWorld YAML for {path!r}. Tried: {candidates}"
    )

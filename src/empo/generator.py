"""Environment Generators for MultiGrid

Base classes and implementations for generating MultiGrid environments.

Usage:
    from empo.generator import MazeGenerator
    generator = MazeGenerator(width=15, height=15, num_locked_agents=2, seed=42)
    env = generator.generate()
    env.reset()
"""

import random
import numpy as np
from abc import ABC, abstractmethod
from typing import Optional

from gym_multigrid.multigrid import MultiGridEnv, World, SmallActions


class EnvironmentGenerator(ABC):
    """Base class for MultiGrid environment generators."""

    COLORS = ['r', 'g', 'b', 'p', 'y']
    COLOR_NAMES = ['red', 'green', 'blue', 'purple', 'yellow']

    def __init__(
        self, 
        width: int, 
        height: int, 
        seed: Optional[int] = 42,
        max_steps: int = 50
    ):
        self.width = width
        self.height = height
        self.seed = seed
        self.max_steps = max_steps
        if seed is not None:
            self._set_seed(seed)
        self._validate_inputs()

    def _set_seed(self, seed: int):
        random.seed(seed)
        np.random.seed(seed)

    @abstractmethod
    def _validate_inputs(self):
        pass

    @abstractmethod
    def generate(self) -> MultiGridEnv:
        """Generate and return a MultiGridEnv."""
        pass


class MazeGenerator(EnvironmentGenerator):
    """Generates random maze environments using DFS with backtracking."""

    DIRECTIONS = [(0, 1), (1, 0), (0, -1), (-1, 0)]

    def __init__(
        self, 
        width: int, height: int,         
        seed: Optional[int] = 42,
        max_steps: int = 50, 
        num_locked_agents: int = 1
    ):
        self.num_locked_agents = num_locked_agents
        super().__init__(width, height, seed, max_steps)

    def _validate_inputs(self):
        if self.width < 5 or self.height < 5:
            raise ValueError(f"Width and height must be at least 5. Got: {self.width}x{self.height}")

        if self.width % 2 == 0 or self.height % 2 == 0:
            raise ValueError(f"Width and height must be odd numbers. Got: {self.width}x{self.height}")

        if self.num_locked_agents < 1 or self.num_locked_agents > 6:
            raise ValueError(f"Number of locked agents must be between 1 and 6. Got: {self.num_locked_agents}")

        min_cells = (self.width - 2) * (self.height - 2)
        if min_cells < self.num_locked_agents * 4:
            raise ValueError(f"Maze is too small for {self.num_locked_agents} locked agents.")

    def generate(self) -> MultiGridEnv:
        """Generate a maze environment."""
        maze = np.zeros((self.height, self.width), dtype=int)
        self._generate_maze_dfs(maze)

        dead_ends = self._find_dead_ends(maze)
        if len(dead_ends) < self.num_locked_agents * 2:
            raise RuntimeError(
                f"Generated maze has only {len(dead_ends)} dead ends, "
                f"need at least {self.num_locked_agents * 2}. Try increasing size."
            )

        agent_positions, door_positions = self._place_agents_and_doors(maze, dead_ends)
        key_positions = self._place_keys(maze, dead_ends, agent_positions, door_positions)
        robot_position = self._place_robot(maze, dead_ends, agent_positions, door_positions, key_positions)

        map_string = self._generate_map_string(maze, agent_positions, door_positions,
                                               key_positions, robot_position)

        env = MultiGridEnv(
            map=map_string,
            max_steps=self.max_steps,
            partial_obs=False,
            objects_set=World,
            actions_set=SmallActions
        )

        env.maze_info = {
            'dead_ends': dead_ends,
            'agent_positions': agent_positions,
            'door_positions': door_positions,
            'key_positions': key_positions,
            'robot_position': robot_position,
            'width': self.width,
            'height': self.height,
            'num_locked_agents': self.num_locked_agents,
            'seed': self.seed
        }
        env.map_str = map_string

        return env

    def _generate_maze_dfs(self, maze):
        maze.fill(0)
        start_row = random.randrange(1, self.height - 1, 2)
        start_col = random.randrange(1, self.width - 1, 2)

        stack = [(start_row, start_col)]
        maze[start_row, start_col] = 1

        while stack:
            current_row, current_col = stack[-1]

            neighbors = []
            for dr, dc in self.DIRECTIONS:
                new_row = current_row + dr * 2
                new_col = current_col + dc * 2
                if (1 <= new_row < self.height - 1 and
                    1 <= new_col < self.width - 1 and
                    maze[new_row, new_col] == 0):
                    neighbors.append((new_row, new_col, dr, dc))

            if neighbors:
                new_row, new_col, dr, dc = random.choice(neighbors)
                maze[current_row + dr, current_col + dc] = 1
                maze[new_row, new_col] = 1
                stack.append((new_row, new_col))
            else:
                stack.pop()

    def _find_dead_ends(self, maze):
        dead_ends = []
        for row in range(1, self.height - 1):
            for col in range(1, self.width - 1):
                if maze[row, col] == 1:
                    path_neighbors = sum(
                        1 for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]
                        if 0 <= row + dr < self.height and 0 <= col + dc < self.width
                        and maze[row + dr, col + dc] == 1
                    )
                    if path_neighbors == 1:
                        dead_ends.append((row, col))
        return dead_ends

    def _place_agents_and_doors(self, maze, dead_ends):
        available_dead_ends = dead_ends.copy()
        random.shuffle(available_dead_ends)

        agent_positions = []
        door_positions = []

        for i in range(self.num_locked_agents):
            agent_pos = available_dead_ends.pop()
            agent_positions.append(agent_pos)

            agent_row, agent_col = agent_pos
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = agent_row + dr, agent_col + dc
                if (0 <= nr < self.height and 0 <= nc < self.width
                    and maze[nr, nc] == 1):
                    door_positions.append((nr, nc, self.COLORS[i]))
                    break

        return agent_positions, door_positions

    def _place_keys(self, maze, dead_ends, agent_positions, door_positions):
        used_positions = set(agent_positions)
        used_positions.update((row, col) for row, col, _ in door_positions)
        available_dead_ends = [pos for pos in dead_ends if pos not in used_positions]
        random.shuffle(available_dead_ends)

        key_positions = []
        for i in range(self.num_locked_agents):
            if available_dead_ends:
                key_pos = available_dead_ends.pop()
                key_positions.append((key_pos[0], key_pos[1], self.COLORS[i]))
            else:
                available_cells = [
                    (row, col) for row in range(1, self.height - 1)
                    for col in range(1, self.width - 1)
                    if maze[row, col] == 1 and (row, col) not in used_positions
                ]
                if available_cells:
                    key_pos = random.choice(available_cells)
                    key_positions.append((key_pos[0], key_pos[1], self.COLORS[i]))
                    used_positions.add(key_pos)

        return key_positions

    def _place_robot(self, maze, dead_ends, agent_positions, door_positions, key_positions):
        used_positions = set(agent_positions)
        used_positions.update((row, col) for row, col, _ in door_positions)
        used_positions.update((row, col) for row, col, _ in key_positions)
        dead_end_set = set(dead_ends)

        available_cells = [
            (row, col) for row in range(1, self.height - 1)
            for col in range(1, self.width - 1)
            if (maze[row, col] == 1 and (row, col) not in dead_end_set
                and (row, col) not in used_positions)
        ]

        if not available_cells:
            available_cells = [
                (row, col) for row in range(1, self.height - 1)
                for col in range(1, self.width - 1)
                if maze[row, col] == 1 and (row, col) not in used_positions
            ]

        if available_cells:
            return random.choice(available_cells)
        else:
            path_cells = [(r, c) for r in range(self.height) for c in range(self.width)
                         if maze[r, c] == 1]
            return random.choice(path_cells) if path_cells else (1, 1)

    def _generate_map_string(self, maze, agent_positions, door_positions, key_positions, robot_position):
        map_grid = [['We' for _ in range(self.width)] for _ in range(self.height)]

        for row in range(self.height):
            for col in range(self.width):
                if maze[row, col] == 1:
                    map_grid[row][col] = '..'

        for row, col, color in door_positions:
            map_grid[row][col] = f'L{color}'

        for row, col, color in key_positions:
            map_grid[row][col] = f'K{color}'

        for i, (row, col) in enumerate(agent_positions):
            map_grid[row][col] = f'A{self.COLORS[i]}'

        if robot_position:
            row, col = robot_position
            robot_color = "e"
            map_grid[row][col] = f'A{robot_color}'

        return '\n'.join(' '.join(row) for row in map_grid)


def generate_maze_env(
    width: int = 15,
    height: int = 15,
    num_locked_agents: int = 2,
    max_steps: int = 200,
    seed: Optional[int] = None
) -> MultiGridEnv:
    """Generate a random maze environment."""
    generator = MazeGenerator(
        width=width,
        height=height,
        seed=seed,
        max_steps=max_steps,
        num_locked_agents=num_locked_agents
    )
    return generator.generate()


def print_maze_info(info):
    """Pretty print maze information from a generated environment or info dict."""
    if isinstance(info, MultiGridEnv):
        if not hasattr(info, 'maze_info'):
            print("Environment does not have maze_info")
            return
        info = info.maze_info
    print("\n" + "=" * 60)
    print("MAZE INFORMATION")
    print("=" * 60)
    print(f"Size: {info['width']}x{info['height']}")
    print(f"Locked Agents: {info['num_locked_agents']}")
    print(f"Dead Ends: {len(info['dead_ends'])}")
    print(f"Seed: {info['seed']}")
    print("\nAgent Positions:")
    for i, (row, col) in enumerate(info['agent_positions']):
        print(f"  Agent {i+1} ({MazeGenerator.COLOR_NAMES[i]}): row={row}, col={col}")
    print("\nDoor Positions:")
    for row, col, color in info['door_positions']:
        color_idx = MazeGenerator.COLORS.index(color)
        print(f"  Locked door ({MazeGenerator.COLOR_NAMES[color_idx]}): row={row}, col={col}")
    print("\nKey Positions:")
    for row, col, color in info['key_positions']:
        color_idx = MazeGenerator.COLORS.index(color)
        print(f"  Key ({MazeGenerator.COLOR_NAMES[color_idx]}): row={row}, col={col}")
    print(f"\nRobot Position: row={info['robot_position'][0]}, col={info['robot_position'][1]}")
    print("=" * 60 + "\n")
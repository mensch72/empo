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


class RockMazeGenerator(EnvironmentGenerator):
    """Generates maze environments where humans are trapped by rocks that robots must push."""

    DIRECTIONS = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # right, down, left, up

    def __init__(
        self,
        width: int,
        height: int,
        seed: Optional[int] = 42,
        max_steps: int = 50,
        num_trapped_agents: int = 1,
        robot_distance: float = 0.0
    ):
        """
        Args:
            width: Width of the maze (must be odd)
            height: Height of the maze (must be odd)
            seed: Random seed
            max_steps: Maximum steps per episode
            num_trapped_agents: Number of human agents trapped by rocks
            robot_distance: Normalized distance of robot from rock (0.0 = adjacent, 1.0 = max map distance)
        """
        self.num_trapped_agents = num_trapped_agents
        self.robot_distance = robot_distance
        super().__init__(width, height, seed, max_steps)

    def _validate_inputs(self):
        if self.width < 5 or self.height < 5:
            raise ValueError(f"Width and height must be at least 5. Got: {self.width}x{self.height}")

        if self.width % 2 == 0 or self.height % 2 == 0:
            raise ValueError(f"Width and height must be odd numbers. Got: {self.width}x{self.height}")

        if self.num_trapped_agents < 1 or self.num_trapped_agents > 4:
            raise ValueError(f"Number of trapped agents must be between 1 and 4. Got: {self.num_trapped_agents}")

        if not (0.0 <= self.robot_distance <= 1.0):
            raise ValueError(f"Robot distance must be between 0.0 and 1.0. Got: {self.robot_distance}")

    def generate(self) -> MultiGridEnv:
        """Generate a maze environment with rock-trapped humans."""
        maze = np.zeros((self.height, self.width), dtype=int)
        self._generate_maze_dfs(maze)

        # Find suitable trap locations (dead ends or corners)
        trap_locations = self._find_trap_locations(maze)
        if len(trap_locations) < self.num_trapped_agents:
            raise RuntimeError(
                f"Generated maze has only {len(trap_locations)} suitable trap locations, "
                f"need at least {self.num_trapped_agents}. Try increasing size."
            )

        # Place trapped agents, rocks, and robot
        agent_positions, rock_positions, rock_directions = self._place_agents_and_rocks(trap_locations)

        # Create push space by replacing orthogonal walls with empty paths
        self._create_push_space(maze, agent_positions, rock_positions, rock_directions)

        robot_position = self._place_robot(maze, rock_positions, rock_directions)

        map_string = self._generate_map_string(
            maze, agent_positions, rock_positions, robot_position
        )

        env = MultiGridEnv(
            map=map_string,
            max_steps=self.max_steps,
            partial_obs=False,
            objects_set=World,
            actions_set=SmallActions
        )

        # CRITICAL: Enable rock pushing for grey (robot) agents
        for agent in env.agents:
            if agent.color == 'grey':
                agent.can_push_rock = True

        env.maze_info = {
            'agent_positions': agent_positions,
            'rock_positions': rock_positions,
            'robot_position': robot_position,
            'robot_distance': self.robot_distance,
            'width': self.width,
            'height': self.height,
            'num_trapped_agents': self.num_trapped_agents,
            'seed': self.seed
        }
        env.map_str = map_string

        return env

    def _generate_maze_dfs(self, maze):
        """Generate maze using depth-first search with backtracking."""
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

    def _find_trap_locations(self, maze):
        """
        Find locations suitable for trapping agents with rocks.

        A valid trap has:
        - Human position: surrounded by walls on 3 sides (dead end)
        - Rock position: blocks the 4th side (the only exit)
        - Push space will be created by replacing orthogonal walls later
        """
        trap_locations = []

        for row in range(1, self.height - 1):
            for col in range(1, self.width - 1):
                if maze[row, col] != 1:
                    continue

                # Check each direction for potential trap
                for dr, dc in self.DIRECTIONS:
                    rock_row = row + dr
                    rock_col = col + dc

                    # Check rock position is valid path
                    if not (0 <= rock_row < self.height and 0 <= rock_col < self.width):
                        continue
                    if maze[rock_row, rock_col] != 1:
                        continue

                    # Count how many sides are blocked (should be exactly 3 for proper dead end)
                    blocked_sides = 0
                    for check_dr, check_dc in self.DIRECTIONS:
                        # Skip the direction we're checking as exit
                        if (check_dr, check_dc) == (dr, dc):
                            continue

                        check_row = row + check_dr
                        check_col = col + check_dc

                        # Count walls (blocked) vs paths (open)
                        if not (0 <= check_row < self.height and 0 <= check_col < self.width):
                            blocked_sides += 1
                        elif maze[check_row, check_col] == 0:
                            blocked_sides += 1

                    # Good trap: exactly 3 sides blocked, 1 side open (for rock)
                    if blocked_sides == 3:
                        trap_locations.append((row, col, dr, dc))

        return trap_locations

    def _create_push_space(self, maze, agent_positions, rock_positions, rock_directions):
        """
        Create push space by replacing walls orthogonal to the human-rock axis with paths.

        If human-rock are vertically aligned (same column), replace left/right walls.
        If human-rock are horizontally aligned (same row), replace top/bottom walls.
        """
        for i in range(len(rock_positions)):
            rock_row, rock_col = rock_positions[i]
            dr, dc = rock_directions[i]

            # Determine orthogonal directions
            if dr != 0:  # Vertical alignment (same column)
                orthogonal_dirs = [(0, 1), (0, -1)]  # Left and right
            else:  # Horizontal alignment (same row)
                orthogonal_dirs = [(1, 0), (-1, 0)]  # Up and down

            # Replace orthogonal walls with paths
            for orth_dr, orth_dc in orthogonal_dirs:
                wall_row = rock_row + orth_dr
                wall_col = rock_col + orth_dc

                # Check if this is a wall and within bounds
                if (0 <= wall_row < self.height and 0 <= wall_col < self.width
                        and maze[wall_row, wall_col] == 0):
                    # Replace wall with path to create push space
                    maze[wall_row, wall_col] = 1

    def _place_agents_and_rocks(self, trap_locations):
        """Place trapped agents and rocks blocking them."""
        random.shuffle(trap_locations)

        agent_positions = []
        rock_positions = []
        rock_directions = []  # Direction from agent to rock

        for i in range(self.num_trapped_agents):
            # Get trap location: (agent_row, agent_col, rock_dr, rock_dc)
            agent_row, agent_col, dr, dc = trap_locations[i]
            agent_positions.append((agent_row, agent_col))

            # Place rock in the exit direction
            rock_row = agent_row + dr
            rock_col = agent_col + dc
            rock_positions.append((rock_row, rock_col))
            rock_directions.append((dr, dc))

        return agent_positions, rock_positions, rock_directions

    def _place_robot(self, maze, rock_positions, rock_directions):
        """
        Place robot at normalized distance from a rock in a perpendicular push direction.

        The robot_distance is normalized to map size:
        - 0.0 = adjacent to rock (ready to push)
        - 1.0 = maximum distance from rock

        Uses BFS to find positions at the correct path distance.
        """
        if not rock_positions:
            # Fallback: place anywhere
            available_cells = [
                (row, col) for row in range(1, self.height - 1)
                for col in range(1, self.width - 1)
                if maze[row, col] == 1
            ]
            return random.choice(available_cells) if available_cells else (1, 1)

        # Choose a random rock to place robot near
        target_idx = random.randrange(len(rock_positions))
        rock_row, rock_col = rock_positions[target_idx]
        agent_to_rock_dr, _ = rock_directions[target_idx]

        # Determine perpendicular directions (for filtering candidates)
        if agent_to_rock_dr != 0:  # Vertical alignment
            perpendicular_dirs = [(0, 1), (0, -1)]  # Left and right
        else:  # Horizontal alignment
            perpendicular_dirs = [(1, 0), (-1, 0)]  # Up and down

        # Use BFS to find all reachable positions and their distances from the rock
        from collections import deque
        visited = set()
        queue = deque([(rock_row, rock_col, 0)])
        visited.add((rock_row, rock_col))
        positions_by_distance = {}
        max_reachable_distance = 0

        while queue:
            row, col, dist = queue.popleft()

            if dist > 0:
                if dist not in positions_by_distance:
                    positions_by_distance[dist] = []
                positions_by_distance[dist].append((row, col))
                max_reachable_distance = max(max_reachable_distance, dist)

            for dr, dc in self.DIRECTIONS:
                new_row, new_col = row + dr, col + dc
                if ((new_row, new_col) not in visited and
                    0 < new_row < self.height - 1 and
                    0 < new_col < self.width - 1 and
                    maze[new_row, new_col] == 1):
                    visited.add((new_row, new_col))
                    queue.append((new_row, new_col, dist + 1))

        # Calculate target distance based on normalized robot_distance
        # 0.0 = adjacent (distance 1), 1.0 = maximum reachable distance
        if max_reachable_distance > 0:
            target_distance = int(1 + self.robot_distance * (max_reachable_distance - 1))
        else:
            target_distance = 1

        # Try to find a position at the target distance, preferring perpendicular directions
        for dist in range(target_distance, 0, -1):
            if dist not in positions_by_distance:
                continue

            candidates = positions_by_distance[dist]

            # Filter candidates in perpendicular directions (preferred)
            perpendicular_candidates = []
            for pos_row, pos_col in candidates:
                # Check if position is roughly perpendicular to agent-rock axis
                dr_to_pos = pos_row - rock_row
                dc_to_pos = pos_col - rock_col
                for perp_dr, perp_dc in perpendicular_dirs:
                    # Position is perpendicular if it's more aligned with perp direction
                    if abs(dr_to_pos * perp_dr + dc_to_pos * perp_dc) > abs(dr_to_pos + dc_to_pos) * 0.5:
                        perpendicular_candidates.append((pos_row, pos_col))
                        break

            if perpendicular_candidates:
                return random.choice(perpendicular_candidates)

            # If no perpendicular candidates, use any candidate
            if candidates:
                return random.choice(candidates)

        # Fallback: find any valid position
        available_cells = [
            (row, col) for row in range(1, self.height - 1)
            for col in range(1, self.width - 1)
            if maze[row, col] == 1 and (row, col) not in set(rock_positions)
        ]

        return random.choice(available_cells) if available_cells else (1, 1)

    def _generate_map_string(self, maze, agent_positions, rock_positions, robot_position):
        """Generate map string for MultiGridEnv."""
        map_grid = [['We' for _ in range(self.width)] for _ in range(self.height)]

        # Fill in maze paths
        for row in range(self.height):
            for col in range(self.width):
                if maze[row, col] == 1:
                    map_grid[row][col] = '..'

        # Place rocks
        for row, col in rock_positions:
            map_grid[row][col] = 'Ro'

        # Place trapped agents (yellow = human)
        for row, col in agent_positions:
            map_grid[row][col] = 'Ay'  # Yellow agent (human)

        # Place robot (grey)
        if robot_position:
            row, col = robot_position
            map_grid[row][col] = 'Ae'  # Grey agent (robot with can_push_rock)

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


def generate_rock_maze_env(
    width: int = 15,
    height: int = 15,
    num_trapped_agents: int = 1,
    robot_distance: float = 0.0,
    max_steps: int = 200,
    seed: Optional[int] = None
) -> MultiGridEnv:
    """
    Generate a maze where humans are trapped by rocks that the robot must push.

    Args:
        width: Maze width (must be odd)
        height: Maze height (must be odd)
        num_trapped_agents: Number of trapped human agents
        robot_distance: Normalized distance from rock (0.0 = adjacent, 1.0 = maximum distance)
        max_steps: Maximum episode steps
        seed: Random seed

    Returns:
        MultiGridEnv with trapped humans and pushing robot
    """
    generator = RockMazeGenerator(
        width=width,
        height=height,
        seed=seed,
        max_steps=max_steps,
        num_trapped_agents=num_trapped_agents,
        robot_distance=robot_distance
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
    print(f"Seed: {info['seed']}")

    # Check if this is a rock maze or door/key maze
    if 'rock_positions' in info:
        # Rock maze
        print(f"Trapped Agents: {info['num_trapped_agents']}")
        print(f"Robot Distance from Rock: {info['robot_distance']} (cells away)")
        print("\nTrapped Agent Positions:")
        for i, (row, col) in enumerate(info['agent_positions']):
            print(f"  Human {i+1} (yellow): row={row}, col={col}")
        print("\nRock Positions (blocking humans):")
        for i, (row, col) in enumerate(info['rock_positions']):
            print(f"  Rock {i+1}: row={row}, col={col}")
        print(f"\nRobot Position: row={info['robot_position'][0]}, col={info['robot_position'][1]}")
    else:
        # Door/key maze
        print(f"Locked Agents: {info['num_locked_agents']}")
        print(f"Dead Ends: {len(info['dead_ends'])}")
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
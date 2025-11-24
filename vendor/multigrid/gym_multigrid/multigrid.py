import math
import gymnasium as gym
from enum import IntEnum
import numpy as np
from gymnasium import error, spaces, utils
from gymnasium.utils import seeding
from .rendering import *
from .window import Window
import numpy as np
from itertools import product

# Size in pixels of a tile in the full-scale human view
TILE_PIXELS = 32

# Map of color names to RGB values
COLORS = {
    'red': np.array([255, 0, 0]),
    'green': np.array([0, 255, 0]),
    'blue': np.array([0, 0, 255]),
    'purple': np.array([112, 39, 195]),
    'yellow': np.array([255, 255, 0]),
    'grey': np.array([100, 100, 100]),
    'brown': np.array([139, 90, 43])
}

COLOR_NAMES = sorted(list(COLORS.keys()))

class World:

    encode_dim = 6

    normalize_obs = 1

    # Used to map colors to integers
    COLOR_TO_IDX = {
        'red': 0,
        'green': 1,
        'blue': 2,
        'purple': 3,
        'yellow': 4,
        'grey': 5,
        'brown': 6
    }

    IDX_TO_COLOR = dict(zip(COLOR_TO_IDX.values(), COLOR_TO_IDX.keys()))

    # Map of object type to integers
    OBJECT_TO_IDX = {
        'unseen': 0,
        'empty': 1,
        'wall': 2,
        'floor': 3,
        'door': 4,
        'key': 5,
        'ball': 6,
        'box': 7,
        'goal': 8,
        'lava': 9,
        'agent': 10,
        'objgoal': 11,
        'switch': 12,
        'block': 13,
        'rock': 14
    }
    IDX_TO_OBJECT = dict(zip(OBJECT_TO_IDX.values(), OBJECT_TO_IDX.keys()))


class SmallWorld:

    encode_dim = 3

    normalize_obs = 1/3

    COLOR_TO_IDX = {
        'red': 0,
        'green': 1,
        'blue': 2,
        'grey': 3
    }

    IDX_TO_COLOR = dict(zip(COLOR_TO_IDX.values(), COLOR_TO_IDX.keys()))

    OBJECT_TO_IDX = {
        'unseen': 0,
        'empty': 1,
        'wall': 2,
        'agent': 3
    }

    IDX_TO_OBJECT = dict(zip(OBJECT_TO_IDX.values(), OBJECT_TO_IDX.keys()))


# Map of state names to integers
STATE_TO_IDX = {
    'open': 0,
    'closed': 1,
    'locked': 2,
}

# Map of agent direction indices to vectors
DIR_TO_VEC = [
    # Pointing right (positive X)
    np.array((1, 0)),
    # Down (positive Y)
    np.array((0, 1)),
    # Pointing left (negative X)
    np.array((-1, 0)),
    # Up (negative Y)
    np.array((0, -1)),
]


class WorldObj:
    """
    Base class for grid world objects
    """

    def __init__(self, world, type, color):
        assert type in world.OBJECT_TO_IDX, type
        assert color in world.COLOR_TO_IDX, color
        self.type = type
        self.color = color
        self.contains = None

        # Initial position of the object
        self.init_pos = None

        # Current position of the object
        self.cur_pos = None

    def can_overlap(self):
        """Can the agent overlap with this?"""
        return False

    def can_pickup(self):
        """Can the agent pick this up?"""
        return False

    def can_contain(self):
        """Can this contain another object?"""
        return False

    def see_behind(self):
        """Can the agent see behind this object?"""
        return True

    def toggle(self, env, pos):
        """Method to trigger/toggle an action this object performs"""
        return False

    def encode(self, world, current_agent=False):
        """Encode the a description of this object as a 3-tuple of integers"""
        if world.encode_dim==3:
            return (world.OBJECT_TO_IDX[self.type], world.COLOR_TO_IDX[self.color], 0)
        else:
            return (world.OBJECT_TO_IDX[self.type], world.COLOR_TO_IDX[self.color], 0, 0, 0, 0)

    @staticmethod
    def decode(type_idx, color_idx, state):
        assert False, "not implemented"

    def render(self, r):
        """Draw this object with the given renderer"""
        raise NotImplementedError


class ObjectGoal(WorldObj):
    def __init__(self, world, index, target_type='ball', reward=1, color=None):
        if color is None:
            super().__init__(world, 'objgoal', world.IDX_TO_COLOR[index])
        else:
            super().__init__(world, 'objgoal', world.IDX_TO_COLOR[color])
        self.target_type = target_type
        self.index = index
        self.reward = reward

    def can_overlap(self):
        return False

    def render(self, img):
        fill_coords(img, point_in_rect(0, 1, 0, 1), COLORS[self.color])


class Goal(WorldObj):
    def __init__(self, world, index, reward=1, color=None):
        if color is None:
            super().__init__(world, 'goal', world.IDX_TO_COLOR[index])
        else:
            super().__init__(world, 'goal', world.IDX_TO_COLOR[color])
        self.index = index
        self.reward = reward

    def can_overlap(self):
        return True

    def render(self, img):
        fill_coords(img, point_in_rect(0, 1, 0, 1), COLORS[self.color])

class Switch(WorldObj):
    def __init__(self, world):
        super().__init__(world, 'switch', world.IDX_TO_COLOR[0])

    def can_overlap(self):
        return True

    def render(self, img):
        fill_coords(img, point_in_rect(0, 1, 0, 1), COLORS[self.color])


class Floor(WorldObj):
    """
    Colored floor tile the agent can walk over
    """

    def __init__(self, world, color='blue'):
        super().__init__(world, 'floor', color)

    def can_overlap(self):
        return True

    def render(self, r):
        # Give the floor a pale color
        c = COLORS[self.color]
        r.setLineColor(100, 100, 100, 0)
        r.setColor(*c / 2)
        r.drawPolygon([
            (1, TILE_PIXELS),
            (TILE_PIXELS, TILE_PIXELS),
            (TILE_PIXELS, 1),
            (1, 1)
        ])


class Lava(WorldObj):
    def __init__(self, world):
        super().__init__(world, 'lava', 'red')

    def can_overlap(self):
        return True

    def render(self, img):
        c = (255, 128, 0)

        # Background color
        fill_coords(img, point_in_rect(0, 1, 0, 1), c)

        # Little waves
        for i in range(3):
            ylo = 0.3 + 0.2 * i
            yhi = 0.4 + 0.2 * i
            fill_coords(img, point_in_line(0.1, ylo, 0.3, yhi, r=0.03), (0, 0, 0))
            fill_coords(img, point_in_line(0.3, yhi, 0.5, ylo, r=0.03), (0, 0, 0))
            fill_coords(img, point_in_line(0.5, ylo, 0.7, yhi, r=0.03), (0, 0, 0))
            fill_coords(img, point_in_line(0.7, yhi, 0.9, ylo, r=0.03), (0, 0, 0))


class Wall(WorldObj):
    def __init__(self, world, color='grey'):
        super().__init__(world, 'wall', color)

    def see_behind(self):
        return False

    def render(self, img):
        fill_coords(img, point_in_rect(0, 1, 0, 1), COLORS[self.color])


class Door(WorldObj):
    def __init__(self, world, color, is_open=False, is_locked=False):
        super().__init__(world, 'door', color)
        self.is_open = is_open
        self.is_locked = is_locked

    def can_overlap(self):
        """The agent can only walk over this cell when the door is open"""
        return self.is_open

    def see_behind(self):
        return self.is_open

    def toggle(self, env, pos):
        # If the player has the right key to open the door
        if self.is_locked:
            if isinstance(env.carrying, Key) and env.carrying.color == self.color:
                self.is_locked = False
                self.is_open = True
                return True
            return False

        self.is_open = not self.is_open
        return True

    def encode(self, world, current_agent=False):
        """Encode the a description of this object as a 3-tuple of integers"""

        # State, 0: open, 1: closed, 2: locked
        if self.is_open:
            state = 0
        elif self.is_locked:
            state = 2
        elif not self.is_open:
            state = 1

        return (world.OBJECT_TO_IDX[self.type], world.COLOR_TO_IDX[self.color], state, 0, 0, 0)

    def render(self, img):
        c = COLORS[self.color]

        if self.is_open:
            fill_coords(img, point_in_rect(0.88, 1.00, 0.00, 1.00), c)
            fill_coords(img, point_in_rect(0.92, 0.96, 0.04, 0.96), (0, 0, 0))
            return

        # Door frame and door
        if self.is_locked:
            fill_coords(img, point_in_rect(0.00, 1.00, 0.00, 1.00), c)
            fill_coords(img, point_in_rect(0.06, 0.94, 0.06, 0.94), 0.45 * np.array(c))

            # Draw key slot
            fill_coords(img, point_in_rect(0.52, 0.75, 0.50, 0.56), c)
        else:
            fill_coords(img, point_in_rect(0.00, 1.00, 0.00, 1.00), c)
            fill_coords(img, point_in_rect(0.04, 0.96, 0.04, 0.96), (0, 0, 0))
            fill_coords(img, point_in_rect(0.08, 0.92, 0.08, 0.92), c)
            fill_coords(img, point_in_rect(0.12, 0.88, 0.12, 0.88), (0, 0, 0))

            # Draw door handle
            fill_coords(img, point_in_circle(cx=0.75, cy=0.50, r=0.08), c)


class Key(WorldObj):
    def __init__(self, world, color='blue'):
        super(Key, self).__init__(world, 'key', color)

    def can_pickup(self):
        return True

    def render(self, img):
        c = COLORS[self.color]

        # Vertical quad
        fill_coords(img, point_in_rect(0.50, 0.63, 0.31, 0.88), c)

        # Teeth
        fill_coords(img, point_in_rect(0.38, 0.50, 0.59, 0.66), c)
        fill_coords(img, point_in_rect(0.38, 0.50, 0.81, 0.88), c)

        # Ring
        fill_coords(img, point_in_circle(cx=0.56, cy=0.28, r=0.190), c)
        fill_coords(img, point_in_circle(cx=0.56, cy=0.28, r=0.064), (0, 0, 0))


class Ball(WorldObj):
    def __init__(self, world, index=0, reward=1):
        super(Ball, self).__init__(world, 'ball', world.IDX_TO_COLOR[index])
        self.index = index
        self.reward = reward

    def can_pickup(self):
        return True

    def render(self, img):
        fill_coords(img, point_in_circle(0.5, 0.5, 0.31), COLORS[self.color])


class Box(WorldObj):
    def __init__(self, world, color, contains=None):
        super(Box, self).__init__(world, 'box', color)
        self.contains = contains

    def can_pickup(self):
        return True

    def render(self, img):
        c = COLORS[self.color]

        # Outline
        fill_coords(img, point_in_rect(0.12, 0.88, 0.12, 0.88), c)
        fill_coords(img, point_in_rect(0.18, 0.82, 0.18, 0.82), (0, 0, 0))

        # Horizontal slit
        fill_coords(img, point_in_rect(0.16, 0.84, 0.47, 0.53), c)

    def toggle(self, env, pos):
        # Replace the box by its contents
        env.grid.set(*pos, self.contains)
        return True


class Block(WorldObj):
    def __init__(self, world):
        super(Block, self).__init__(world, 'block', 'brown')

    def can_overlap(self):
        return False

    def can_pickup(self):
        return False

    def render(self, img):
        # Light brown square
        c = COLORS[self.color]
        fill_coords(img, point_in_rect(0, 1, 0, 1), c)


class Rock(WorldObj):
    def __init__(self, world, pushable_by=None):
        super(Rock, self).__init__(world, 'rock', 'grey')
        # pushable_by can be an agent index, a list of indices, or None (pushable by all)
        self.pushable_by = pushable_by

    def can_overlap(self):
        return False

    def can_pickup(self):
        return False
    
    def can_be_pushed_by(self, agent):
        """
        Check if this rock can be pushed by the given agent.
        
        Args:
            agent: The Agent object attempting to push this rock
            
        Returns:
            bool: True if the agent is authorized to push this rock, False otherwise
        """
        if self.pushable_by is None:
            return True
        if isinstance(self.pushable_by, list):
            return agent.index in self.pushable_by
        return agent.index == self.pushable_by

    def render(self, img):
        # Medium grey irregular rock shape
        c = COLORS[self.color]
        # Create an irregular rock-like shape using multiple overlapping shapes
        # Base rock body
        fill_coords(img, point_in_circle(0.45, 0.5, 0.35), c)
        fill_coords(img, point_in_circle(0.55, 0.45, 0.30), c)
        fill_coords(img, point_in_circle(0.50, 0.60, 0.28), c)
        # Add some texture/detail with darker grey
        darker_grey = np.array([70, 70, 70])  # Darker shade for texture
        fill_coords(img, point_in_circle(0.35, 0.40, 0.12), darker_grey)
        fill_coords(img, point_in_circle(0.65, 0.55, 0.10), darker_grey)


class Agent(WorldObj):
    def __init__(self, world, index=0, view_size=7):
        super(Agent, self).__init__(world, 'agent', world.IDX_TO_COLOR[index])
        self.pos = None
        self.dir = None
        self.index = index
        self.view_size = view_size
        self.carrying = None
        self.terminated = False
        self.started = True
        self.paused = False

    def render(self, img):
        c = COLORS[self.color]
        tri_fn = point_in_triangle(
            (0.12, 0.19),
            (0.87, 0.50),
            (0.12, 0.81),
        )
        # Rotate the agent based on its direction
        tri_fn = rotate_fn(tri_fn, cx=0.5, cy=0.5, theta=0.5 * math.pi * self.dir)
        fill_coords(img, tri_fn, c)

    def encode(self, world, current_agent=False):
        """Encode the a description of this object as a 3-tuple of integers"""
        if world.encode_dim==3:
            return (world.OBJECT_TO_IDX[self.type], world.COLOR_TO_IDX[self.color], self.dir)
        elif self.carrying:
            if current_agent:
                return (world.OBJECT_TO_IDX[self.type], world.COLOR_TO_IDX[self.color], world.OBJECT_TO_IDX[self.carrying.type],
                        world.COLOR_TO_IDX[self.carrying.color], self.dir, 1)
            else:
                return (world.OBJECT_TO_IDX[self.type], world.COLOR_TO_IDX[self.color], world.OBJECT_TO_IDX[self.carrying.type],
                        world.COLOR_TO_IDX[self.carrying.color], self.dir, 0)

        else:
            if current_agent:
                return (world.OBJECT_TO_IDX[self.type], world.COLOR_TO_IDX[self.color], 0, 0, self.dir, 1)
            else:
                return (world.OBJECT_TO_IDX[self.type], world.COLOR_TO_IDX[self.color], 0, 0, self.dir, 0)

    @property
    def dir_vec(self):
        """
        Get the direction vector for the agent, pointing in the direction
        of forward movement.
        """

        assert self.dir >= 0 and self.dir < 4
        return DIR_TO_VEC[self.dir]

    @property
    def right_vec(self):
        """
        Get the vector pointing to the right of the agent.
        """

        dx, dy = self.dir_vec
        return np.array((-dy, dx))

    @property
    def front_pos(self):
        """
        Get the position of the cell that is right in front of the agent
        """

        return self.pos + self.dir_vec

    def get_view_coords(self, i, j):
        """
        Translate and rotate absolute grid coordinates (i, j) into the
        agent's partially observable view (sub-grid). Note that the resulting
        coordinates may be negative or outside of the agent's view size.
        """

        ax, ay = self.pos
        dx, dy = self.dir_vec
        rx, ry = self.right_vec

        # Compute the absolute coordinates of the top-left view corner
        sz = self.view_size
        hs = self.view_size // 2
        tx = ax + (dx * (sz - 1)) - (rx * hs)
        ty = ay + (dy * (sz - 1)) - (ry * hs)

        lx = i - tx
        ly = j - ty

        # Project the coordinates of the object relative to the top-left
        # corner onto the agent's own coordinate system
        vx = (rx * lx + ry * ly)
        vy = -(dx * lx + dy * ly)

        return vx, vy

    def get_view_exts(self):
        """
        Get the extents of the square set of tiles visible to the agent
        Note: the bottom extent indices are not included in the set
        """

        # Facing right
        if self.dir == 0:
            topX = self.pos[0]
            topY = self.pos[1] - self.view_size // 2
        # Facing down
        elif self.dir == 1:
            topX = self.pos[0] - self.view_size // 2
            topY = self.pos[1]
        # Facing left
        elif self.dir == 2:
            topX = self.pos[0] - self.view_size + 1
            topY = self.pos[1] - self.view_size // 2
        # Facing up
        elif self.dir == 3:
            topX = self.pos[0] - self.view_size // 2
            topY = self.pos[1] - self.view_size + 1
        else:
            assert False, "invalid agent direction"

        botX = topX + self.view_size
        botY = topY + self.view_size

        return (topX, topY, botX, botY)

    def relative_coords(self, x, y):
        """
        Check if a grid position belongs to the agent's field of view, and returns the corresponding coordinates
        """

        vx, vy = self.get_view_coords(x, y)

        if vx < 0 or vy < 0 or vx >= self.view_size or vy >= self.view_size:
            return None

        return vx, vy

    def in_view(self, x, y):
        """
        check if a grid position is visible to the agent
        """

        return self.relative_coords(x, y) is not None


class Grid:
    """
    Represent a grid and operations on it
    """

    # Static cache of pre-renderer tiles
    tile_cache = {}

    def __init__(self, width, height):
        assert width >= 3
        assert height >= 3

        self.width = width
        self.height = height

        self.grid = [None] * width * height

    def __contains__(self, key):
        if isinstance(key, WorldObj):
            for e in self.grid:
                if e is key:
                    return True
        elif isinstance(key, tuple):
            for e in self.grid:
                if e is None:
                    continue
                if (e.color, e.type) == key:
                    return True
                if key[0] is None and key[1] == e.type:
                    return True
        return False

    def __eq__(self, other):
        grid1 = self.encode()
        grid2 = other.encode()
        return np.array_equal(grid2, grid1)

    def __ne__(self, other):
        return not self == other

    def copy(self):
        from copy import deepcopy
        return deepcopy(self)

    def set(self, i, j, v):
        assert i >= 0 and i < self.width
        assert j >= 0 and j < self.height
        self.grid[j * self.width + i] = v

    def get(self, i, j):
        assert i >= 0 and i < self.width
        assert j >= 0 and j < self.height
        return self.grid[j * self.width + i]

    def horz_wall(self, world, x, y, length=None, obj_type=Wall):
        if length is None:
            length = self.width - x
        for i in range(0, length):
            self.set(x + i, y, obj_type(world))

    def vert_wall(self, world, x, y, length=None, obj_type=Wall):
        if length is None:
            length = self.height - y
        for j in range(0, length):
            self.set(x, y + j, obj_type(world))

    def wall_rect(self, x, y, w, h):
        self.horz_wall(x, y, w)
        self.horz_wall(x, y + h - 1, w)
        self.vert_wall(x, y, h)
        self.vert_wall(x + w - 1, y, h)

    def rotate_left(self):
        """
        Rotate the grid to the left (counter-clockwise)
        """

        grid = Grid(self.height, self.width)

        for i in range(self.width):
            for j in range(self.height):
                v = self.get(i, j)
                grid.set(j, grid.height - 1 - i, v)

        return grid

    def slice(self, world, topX, topY, width, height):
        """
        Get a subset of the grid
        """

        grid = Grid(width, height)

        for j in range(0, height):
            for i in range(0, width):
                x = topX + i
                y = topY + j

                if x >= 0 and x < self.width and \
                        y >= 0 and y < self.height:
                    v = self.get(x, y)
                else:
                    v = Wall(world)

                grid.set(i, j, v)

        return grid

    @classmethod
    def render_tile(
            cls,
            world,
            obj,
            highlights=[],
            tile_size=TILE_PIXELS,
            subdivs=3
    ):
        """
        Render a tile and cache the result
        """

        key = (*highlights, tile_size)
        key = obj.encode(world) + key if obj else key

        if key in cls.tile_cache:
            return cls.tile_cache[key]

        img = np.zeros(shape=(tile_size * subdivs, tile_size * subdivs, 3), dtype=np.uint8)

        # Draw the grid lines (top and left edges)
        fill_coords(img, point_in_rect(0, 0.031, 0, 1), (100, 100, 100))
        fill_coords(img, point_in_rect(0, 1, 0, 0.031), (100, 100, 100))

        if obj != None:
            obj.render(img)

        # Highlight the cell  if needed
        if len(highlights) > 0:
            for h in highlights:
                highlight_img(img, color=COLORS[world.IDX_TO_COLOR[h%len(world.IDX_TO_COLOR)]])

        # Downsample the image to perform supersampling/anti-aliasing
        img = downsample(img, subdivs)

        # Cache the rendered tile
        cls.tile_cache[key] = img

        return img

    def render(
            self,
            world,
            tile_size,
            highlight_masks=None
    ):
        """
        Render this grid at a given scale
        :param r: target renderer object
        :param tile_size: tile size in pixels
        """

        # Compute the total grid size
        width_px = self.width * tile_size
        height_px = self.height * tile_size

        img = np.zeros(shape=(height_px, width_px, 3), dtype=np.uint8)

        # Render the grid
        for j in range(0, self.height):
            for i in range(0, self.width):
                cell = self.get(i, j)

                # agent_here = np.array_equal(agent_pos, (i, j))
                tile_img = Grid.render_tile(
                    world,
                    cell,
                    highlights=[] if highlight_masks is None else highlight_masks[i, j],
                    tile_size=tile_size
                )

                ymin = j * tile_size
                ymax = (j + 1) * tile_size
                xmin = i * tile_size
                xmax = (i + 1) * tile_size
                img[ymin:ymax, xmin:xmax, :] = tile_img

        return img

    def encode(self, world, vis_mask=None):
        """
        Produce a compact numpy encoding of the grid
        """

        if vis_mask is None:
            vis_mask = np.ones((self.width, self.height), dtype=bool)

        array = np.zeros((self.width, self.height, world.encode_dim), dtype='uint8')

        for i in range(self.width):
            for j in range(self.height):
                if vis_mask[i, j]:
                    v = self.get(i, j)

                    if v is None:
                        array[i, j, 0] = world.OBJECT_TO_IDX['empty']
                        array[i, j, 1] = 0
                        array[i, j, 2] = 0
                        if world.encode_dim > 3:
                            array[i, j, 3] = 0
                            array[i, j, 4] = 0
                            array[i, j, 5] = 0

                    else:
                        array[i, j, :] = v.encode(world)

        return array

    def encode_for_agents(self, world, agent_pos, vis_mask=None):
        """
        Produce a compact numpy encoding of the grid
        """
        if vis_mask is None:
            vis_mask = np.ones((self.width, self.height), dtype=bool)

        array = np.zeros((self.width, self.height, world.encode_dim), dtype='uint8')

        for i in range(self.width):
            for j in range(self.height):
                if vis_mask[i, j]:
                    v = self.get(i, j)

                    if v is None:
                        array[i, j, 0] = world.OBJECT_TO_IDX['empty']
                        array[i, j, 1] = 0
                        array[i, j, 2] = 0
                        if world.encode_dim > 3:
                            array[i, j, 3] = 0
                            array[i, j, 4] = 0
                            array[i, j, 5] = 0

                    else:
                        array[i, j, :] = v.encode(world, current_agent=np.array_equal(agent_pos, (i, j)))

        return array

    # @staticmethod
    # def decode(array):
    #     """
    #     Decode an array grid encoding back into a grid
    #     """
    #
    #     width, height, channels = array.shape
    #     assert channels == 3
    #
    #     vis_mask = np.ones(shape=(width, height), dtype=np.bool)
    #
    #     grid = Grid(width, height)
    #     for i in range(width):
    #         for j in range(height):
    #             type_idx, color_idx, state = array[i, j]
    #             v = WorldObj.decode(type_idx, color_idx, state)
    #             grid.set(i, j, v)
    #             vis_mask[i, j] = (type_idx != OBJECT_TO_IDX['unseen'])
    #
    #     return grid, vis_mask

    def process_vis(grid, agent_pos):
        mask = np.zeros(shape=(grid.width, grid.height), dtype=np.bool)

        mask[agent_pos[0], agent_pos[1]] = True

        for j in reversed(range(0, grid.height)):
            for i in range(0, grid.width - 1):
                if not mask[i, j]:
                    continue

                cell = grid.get(i, j)
                if cell and not cell.see_behind():
                    continue

                mask[i + 1, j] = True
                if j > 0:
                    mask[i + 1, j - 1] = True
                    mask[i, j - 1] = True

            for i in reversed(range(1, grid.width)):
                if not mask[i, j]:
                    continue

                cell = grid.get(i, j)
                if cell and not cell.see_behind():
                    continue

                mask[i - 1, j] = True
                if j > 0:
                    mask[i - 1, j - 1] = True
                    mask[i, j - 1] = True

        for j in range(0, grid.height):
            for i in range(0, grid.width):
                if not mask[i, j]:
                    grid.set(i, j, None)

        return mask

class Actions:
    available=['still', 'left', 'right', 'forward', 'pickup', 'drop', 'toggle', 'done']

    still = 0
    # Turn left, turn right, move forward
    left = 1
    right = 2
    forward = 3

    # Pick up an object
    pickup = 4
    # Drop an object
    drop = 5
    # Toggle/activate an object
    toggle = 6

    # Done completing task
    done = 7

class SmallActions:
    available=['still', 'left', 'right', 'forward']

    # Turn left, turn right, move forward
    still = 0
    left = 1
    right = 2
    forward = 3

class MineActions:
    available=['still', 'left', 'right', 'forward', 'build']

    still = 0
    left = 1
    right = 2
    forward = 3
    build = 4

class MultiGridEnv(gym.Env):
    """
    2D grid world game environment
    """

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 10
    }

    # Enumeration of possible actions

    def __init__(
            self,
            grid_size=None,
            width=None,
            height=None,
            max_steps=100,
            see_through_walls=False,
            seed=2,
            agents=None,
            partial_obs=True,
            agent_view_size=7,
            actions_set=Actions,
            objects_set = World
    ):
        self.agents = agents

        # Does the agents have partial or full observation?
        self.partial_obs = partial_obs

        # Can't set both grid_size and width/height
        if grid_size:
            assert width == None and height == None
            width = grid_size
            height = grid_size

        # Action enumeration for this environment
        self.actions = actions_set

        # Actions are discrete integer values
        self.action_space = spaces.Discrete(len(self.actions.available))

        self.objects=objects_set

        if partial_obs:
            self.observation_space = spaces.Box(
                low=0,
                high=255,
                shape=(agent_view_size, agent_view_size, self.objects.encode_dim),
                dtype='uint8'
            )

        else:
            self.observation_space = spaces.Box(
                low=0,
                high=255,
                shape=(width, height, self.objects.encode_dim),
                dtype='uint8'
            )

        self.ob_dim = np.prod(self.observation_space.shape)
        self.ac_dim = self.action_space.n

        # Range of possible rewards
        self.reward_range = (0, 1)

        # Window to use for human rendering mode
        self.window = None

        # Environment configuration
        self.width = width
        self.height = height
        self.max_steps = max_steps
        self.see_through_walls = see_through_walls

        # Initialize the RNG
        self.seed(seed=seed)

        # Initialize the state
        self.reset()

    def reset(self):

        # Generate a new random grid at the start of each episode
        # To keep the same grid for each episode, call env.seed() with
        # the same seed before calling env.reset()
        self._gen_grid(self.width, self.height)

        # These fields should be defined by _gen_grid
        for a in self.agents:
            assert a.pos is not None
            assert a.dir is not None

        # Item picked up, being carried, initially nothing
        for a in self.agents:
            a.carrying = None

        # Step count since episode start
        self.step_count = 0

        # Return first observation
        if self.partial_obs:
            obs = self.gen_obs()
        else:
            obs = [self.grid.encode_for_agents(self.objects, self.agents[i].pos) for i in range(len(self.agents))]
        obs=[self.objects.normalize_obs*ob for ob in obs]
        return obs

    def seed(self, seed=1337):
        # Seed the random number generator
        self.np_random, _ = seeding.np_random(seed)
        return [seed]

    @property
    def steps_remaining(self):
        return self.max_steps - self.step_count

    def __str__(self):
        """
        Produce a pretty string of the environment's grid along with the agent.
        A grid cell is represented by 2-character string, the first one for
        the object and the second one for the color.
        """

        # Map of object types to short string
        OBJECT_TO_STR = {
            'wall': 'W',
            'floor': 'F',
            'door': 'D',
            'key': 'K',
            'ball': 'A',
            'box': 'B',
            'goal': 'G',
            'lava': 'V',
        }

        # Short string for opened door
        OPENDED_DOOR_IDS = '_'

        # Map agent's direction to short string
        AGENT_DIR_TO_STR = {
            0: '>',
            1: 'V',
            2: '<',
            3: '^'
        }

        str = ''

        for j in range(self.grid.height):

            for i in range(self.grid.width):
                if i == self.agent_pos[0] and j == self.agent_pos[1]:
                    str += 2 * AGENT_DIR_TO_STR[self.agent_dir]
                    continue

                c = self.grid.get(i, j)

                if c == None:
                    str += '  '
                    continue

                if c.type == 'door':
                    if c.is_open:
                        str += '__'
                    elif c.is_locked:
                        str += 'L' + c.color[0].upper()
                    else:
                        str += 'D' + c.color[0].upper()
                    continue

                str += OBJECT_TO_STR[c.type] + c.color[0].upper()

            if j < self.grid.height - 1:
                str += '\n'

        return str

    def _gen_grid(self, width, height):
        assert False, "_gen_grid needs to be implemented by each environment"

    def _handle_pickup(self, i, rewards, fwd_pos, fwd_cell):
        pass

    def _handle_build(self, i, rewards, fwd_pos, fwd_cell):
        pass

    def _handle_drop(self, i, rewards, fwd_pos, fwd_cell):
        pass

    def _handle_special_moves(self, i, rewards, fwd_pos, fwd_cell):
        pass
    
    def _can_push_objects(self, agent, start_pos):
        """
        Check if agent can push blocks/rocks starting at start_pos.
        Returns (can_push, num_objects, end_pos) where:
        - can_push: True if push is possible
        - num_objects: number of consecutive blocks/rocks
        - end_pos: position where the last object would move to
        """
        # Check if there are consecutive blocks/rocks in the direction the agent is facing
        direction = agent.dir_vec
        current_pos = np.array(start_pos)
        num_objects = 0
        
        # Count consecutive blocks/rocks
        while True:
            cell = self.grid.get(*current_pos)
            if cell is None:
                break
            if cell.type not in ['block', 'rock']:
                break
            # For rocks, check if agent can push this rock
            if cell.type == 'rock' and not cell.can_be_pushed_by(agent):
                return False, 0, None
            num_objects += 1
            current_pos = current_pos + direction
            
            # Bounds check
            if current_pos[0] < 0 or current_pos[0] >= self.grid.width or \
               current_pos[1] < 0 or current_pos[1] >= self.grid.height:
                return False, 0, None
        
        # current_pos is now the first empty cell after the objects
        # Check if this cell is empty or can be overlapped
        end_cell = self.grid.get(*current_pos)
        if end_cell is None or end_cell.can_overlap():
            return True, num_objects, current_pos
        else:
            return False, 0, None
    
    def _push_objects(self, agent, start_pos):
        """
        Push blocks/rocks starting at start_pos in the direction agent is facing.
        Returns True if push was successful.
        """
        can_push, num_objects, end_pos = self._can_push_objects(agent, start_pos)
        
        if not can_push or num_objects == 0:
            return False
        
        # Move objects from back to front to avoid overwriting
        direction = agent.dir_vec
        # Start from the end position and work backwards
        for j in range(num_objects):
            from_pos = end_pos - direction * (j + 1)
            to_pos = end_pos - direction * j
            obj = self.grid.get(*from_pos)
            self.grid.set(*to_pos, obj)
            if obj:
                obj.cur_pos = to_pos
        
        # Clear the original start position (now empty)
        self.grid.set(*start_pos, None)
        
        # Now agent can move into start_pos
        self.grid.set(*start_pos, agent)
        self.grid.set(*agent.pos, None)
        agent.pos = np.array(start_pos)
        
        return True

    def _handle_switch(self, i, rewards, fwd_pos, fwd_cell):
        pass

    def _reward(self, current_agent, rewards, reward=1):
        """
        Compute the reward to be given upon success
        """

        return 1 - 0.9 * (self.step_count / self.max_steps)

    def _rand_int(self, low, high):
        """
        Generate random integer in [low,high[
        """

        # Handle both old and new numpy random API
        if hasattr(self.np_random, 'randint'):
            return self.np_random.randint(low, high)
        else:
            return self.np_random.integers(low, high)

    def _rand_float(self, low, high):
        """
        Generate random float in [low,high[
        """

        return self.np_random.uniform(low, high)

    def _rand_bool(self):
        """
        Generate random boolean value
        """

        # Handle both old and new numpy random API
        if hasattr(self.np_random, 'randint'):
            return (self.np_random.randint(0, 2) == 0)
        else:
            return (self.np_random.integers(0, 2) == 0)

    def _rand_elem(self, iterable):
        """
        Pick a random element in a list
        """

        lst = list(iterable)
        idx = self._rand_int(0, len(lst))
        return lst[idx]

    def _rand_subset(self, iterable, num_elems):
        """
        Sample a random subset of distinct elements of a list
        """

        lst = list(iterable)
        assert num_elems <= len(lst)

        out = []

        while len(out) < num_elems:
            elem = self._rand_elem(lst)
            lst.remove(elem)
            out.append(elem)

        return out

    def _rand_color(self):
        """
        Generate a random color name (string)
        """

        return self._rand_elem(COLOR_NAMES)

    def _rand_pos(self, xLow, xHigh, yLow, yHigh):
        """
        Generate a random (x,y) position tuple
        """

        return (
            self.np_random.randint(xLow, xHigh),
            self.np_random.randint(yLow, yHigh)
        )

    def place_obj(self,
                  obj,
                  top=None,
                  size=None,
                  reject_fn=None,
                  max_tries=math.inf
                  ):
        """
        Place an object at an empty position in the grid

        :param top: top-left position of the rectangle where to place
        :param size: size of the rectangle where to place
        :param reject_fn: function to filter out potential positions
        """

        if top is None:
            top = (0, 0)
        else:
            top = (max(top[0], 0), max(top[1], 0))

        if size is None:
            size = (self.grid.width, self.grid.height)

        num_tries = 0

        while True:
            # This is to handle with rare cases where rejection sampling
            # gets stuck in an infinite loop
            if num_tries > max_tries:
                raise RecursionError('rejection sampling failed in place_obj')

            num_tries += 1

            pos = np.array((
                self._rand_int(top[0], min(top[0] + size[0], self.grid.width)),
                self._rand_int(top[1], min(top[1] + size[1], self.grid.height))
            ))

            # Don't place the object on top of another object
            if self.grid.get(*pos) != None:
                continue

            # Check if there is a filtering criterion
            if reject_fn and reject_fn(self, pos):
                continue

            break

        self.grid.set(*pos, obj)

        if obj is not None:
            obj.init_pos = pos
            obj.cur_pos = pos

        return pos

    def put_obj(self, obj, i, j):
        """
        Put an object at a specific position in the grid
        """

        self.grid.set(i, j, obj)
        obj.init_pos = (i, j)
        obj.cur_pos = (i, j)

    def place_agent(
            self,
            agent,
            top=None,
            size=None,
            rand_dir=True,
            max_tries=math.inf
    ):
        """
        Set the agent's starting point at an empty position in the grid
        """

        agent.pos = None
        pos = self.place_obj(agent, top, size, max_tries=max_tries)
        agent.pos = pos
        agent.init_pos = pos

        if rand_dir:
            agent.dir = self._rand_int(0, 4)

        agent.init_dir = agent.dir

        return pos

    def agent_sees(self, a, x, y):
        """
        Check if a non-empty grid position is visible to the agent
        """

        coordinates = a.relative_coords(x, y)
        if coordinates is None:
            return False
        vx, vy = coordinates

        obs = self.gen_obs()
        obs_grid, _ = Grid.decode(obs['image'])
        obs_cell = obs_grid.get(vx, vy)
        world_cell = self.grid.get(x, y)

        return obs_cell is not None and obs_cell.type == world_cell.type

    def step(self, actions):
        self.step_count += 1

        order = np.random.permutation(len(actions))

        rewards = np.zeros(len(actions))
        done = False

        for i in order:

            if self.agents[i].terminated or self.agents[i].paused or not self.agents[i].started or actions[i] == self.actions.still:
                continue

            # Get the position in front of the agent
            fwd_pos = self.agents[i].front_pos

            # Get the contents of the cell in front of the agent
            fwd_cell = self.grid.get(*fwd_pos)

            # Rotate left
            if actions[i] == self.actions.left:
                self.agents[i].dir -= 1
                if self.agents[i].dir < 0:
                    self.agents[i].dir += 4

            # Rotate right
            elif actions[i] == self.actions.right:
                self.agents[i].dir = (self.agents[i].dir + 1) % 4

            # Move forward
            elif actions[i] == self.actions.forward:
                moved = False
                # Check if forward cell contains a block or rock that can be pushed
                if fwd_cell is not None and fwd_cell.type in ['block', 'rock']:
                    # Try to push the object(s)
                    pushed = self._push_objects(self.agents[i], fwd_pos)
                    moved = pushed
                elif fwd_cell is not None:
                    if fwd_cell.type == 'goal':
                        done = True
                        self._reward(i, rewards, 1)
                        # Agent can still move onto goal
                        self.grid.set(*fwd_pos, self.agents[i])
                        self.grid.set(*self.agents[i].pos, None)
                        self.agents[i].pos = fwd_pos
                        moved = True
                    elif fwd_cell.type == 'switch':
                        self._handle_switch(i, rewards, fwd_pos, fwd_cell)
                        # Agent can move onto switch
                        self.grid.set(*fwd_pos, self.agents[i])
                        self.grid.set(*self.agents[i].pos, None)
                        self.agents[i].pos = fwd_pos
                        moved = True
                    elif fwd_cell.can_overlap():
                        self.grid.set(*fwd_pos, self.agents[i])
                        self.grid.set(*self.agents[i].pos, None)
                        self.agents[i].pos = fwd_pos
                        moved = True
                elif fwd_cell is None:
                    self.grid.set(*fwd_pos, self.agents[i])
                    self.grid.set(*self.agents[i].pos, None)
                    self.agents[i].pos = fwd_pos
                    moved = True
                self._handle_special_moves(i, rewards, fwd_pos, fwd_cell)

            elif 'build' in self.actions.available and actions[i]==self.actions.build:
                self._handle_build(i, rewards, fwd_pos, fwd_cell)

            # Pick up an object
            elif actions[i] == self.actions.pickup:
                self._handle_pickup(i, rewards, fwd_pos, fwd_cell)

            # Drop an object
            elif actions[i] == self.actions.drop:
                self._handle_drop(i, rewards, fwd_pos, fwd_cell)

            # Toggle/activate an object
            elif actions[i] == self.actions.toggle:
                if fwd_cell:
                    fwd_cell.toggle(self, fwd_pos)

            # Done action (not used by default)
            elif actions[i] == self.actions.done:
                pass

            else:
                assert False, "unknown action"

        if self.step_count >= self.max_steps:
            done = True

        if self.partial_obs:
            obs = self.gen_obs()
        else:
            obs = [self.grid.encode_for_agents(self.objects, self.agents[i].pos) for i in range(len(actions))]

        obs=[self.objects.normalize_obs*ob for ob in obs]

        return obs, rewards, done, {}

    def gen_obs_grid(self):
        """
        Generate the sub-grid observed by the agents.
        This method also outputs a visibility mask telling us which grid
        cells the agents can actually see.
        """

        grids = []
        vis_masks = []

        for a in self.agents:

            topX, topY, botX, botY = a.get_view_exts()

            grid = self.grid.slice(self.objects, topX, topY, a.view_size, a.view_size)

            for i in range(a.dir + 1):
                grid = grid.rotate_left()

            # Process occluders and visibility
            # Note that this incurs some performance cost
            if not self.see_through_walls:
                vis_mask = grid.process_vis(agent_pos=(a.view_size // 2, a.view_size - 1))
            else:
                vis_mask = np.ones(shape=(grid.width, grid.height), dtype=np.bool)

            grids.append(grid)
            vis_masks.append(vis_mask)

        return grids, vis_masks

    def gen_obs(self):
        """
        Generate the agent's view (partially observable, low-resolution encoding)
        """

        grids, vis_masks = self.gen_obs_grid()

        # Encode the partially observable view into a numpy array
        obs = [grid.encode_for_agents(self.objects, [grid.width // 2, grid.height - 1], vis_mask) for grid, vis_mask in zip(grids, vis_masks)]

        return obs

    def get_obs_render(self, obs, tile_size=TILE_PIXELS // 2):
        """
        Render an agent observation for visualization
        """

        grid, vis_mask = Grid.decode(obs)

        # Render the whole grid
        img = grid.render(
            self.objects,
            tile_size,
            highlight_mask=vis_mask
        )

        return img

    def render(self, mode='human', close=False, highlight=False, tile_size=TILE_PIXELS):
        """
        Render the whole-grid human view
        """

        if close:
            if self.window:
                self.window.close()
            return

        if mode == 'human' and not self.window:
            self.window = Window('gym_multigrid')
            self.window.show(block=False)

        if highlight:

            # Compute which cells are visible to the agent
            _, vis_masks = self.gen_obs_grid()

            highlight_masks = {(i, j): [] for i in range(self.width) for j in range(self.height)}

            for i, a in enumerate(self.agents):

                # Compute the world coordinates of the bottom-left corner
                # of the agent's view area
                f_vec = a.dir_vec
                r_vec = a.right_vec
                top_left = a.pos + f_vec * (a.view_size - 1) - r_vec * (a.view_size // 2)

                # Mask of which cells to highlight

                # For each cell in the visibility mask
                for vis_j in range(0, a.view_size):
                    for vis_i in range(0, a.view_size):
                        # If this cell is not visible, don't highlight it
                        if not vis_masks[i][vis_i, vis_j]:
                            continue

                        # Compute the world coordinates of this cell
                        abs_i, abs_j = top_left - (f_vec * vis_j) + (r_vec * vis_i)

                        if abs_i < 0 or abs_i >= self.width:
                            continue
                        if abs_j < 0 or abs_j >= self.height:
                            continue

                        # Mark this cell to be highlighted
                        highlight_masks[abs_i, abs_j].append(i)

        # Render the whole grid
        img = self.grid.render(
            self.objects,
            tile_size,
            highlight_masks=highlight_masks if highlight else None
        )

        if mode == 'human':
            self.window.show_img(img)

        return img

    def get_state(self):
        """
        Get the complete state of the environment.
        
        Returns a hashable dictionary containing everything needed to predict
        the consequences of possible actions:
        - Grid state (all objects and their properties)
        - Agent states (positions, directions, carrying items, status flags)
        - Step count (for timeout tracking)
        - Random number generator state
        
        Returns:
            tuple: A hashable representation of the complete environment state
        """
        # Serialize grid state
        grid_state = []
        for j in range(self.grid.height):
            for i in range(self.grid.width):
                cell = self.grid.get(i, j)
                if cell is None:
                    grid_state.append(None)
                else:
                    # Serialize object with all its properties
                    obj_data = {
                        'type': cell.type,
                        'color': cell.color,
                        'init_pos': tuple(cell.init_pos) if cell.init_pos is not None else None,
                        'cur_pos': tuple(cell.cur_pos) if cell.cur_pos is not None else None,
                    }
                    
                    # Add type-specific properties
                    if hasattr(cell, 'is_open'):
                        obj_data['is_open'] = cell.is_open
                    if hasattr(cell, 'is_locked'):
                        obj_data['is_locked'] = cell.is_locked
                    if hasattr(cell, 'contains'):
                        # Recursively serialize contained objects
                        if cell.contains is not None:
                            obj_data['contains'] = self._serialize_object(cell.contains)
                        else:
                            obj_data['contains'] = None
                    if hasattr(cell, 'index'):
                        obj_data['index'] = cell.index
                    if hasattr(cell, 'reward'):
                        obj_data['reward'] = cell.reward
                    if hasattr(cell, 'target_type'):
                        obj_data['target_type'] = cell.target_type
                    if hasattr(cell, 'pushable_by'):
                        # For rocks, serialize the pushable_by attribute
                        obj_data['pushable_by'] = cell.pushable_by
                    if isinstance(cell, Agent):
                        # For agents in grid, just store basic info (detailed agent state below)
                        obj_data['agent_index'] = cell.index
                    
                    grid_state.append(tuple(sorted(obj_data.items())))
        
        # Serialize agent states
        agents_state = []
        for agent in self.agents:
            agent_data = {
                'pos': tuple(agent.pos) if agent.pos is not None else None,
                'dir': agent.dir,
                'index': agent.index,
                'view_size': agent.view_size,
                'terminated': agent.terminated,
                'started': agent.started,
                'paused': agent.paused,
                'carrying': self._serialize_object(agent.carrying) if agent.carrying is not None else None,
            }
            agents_state.append(tuple(sorted(agent_data.items())))
        
        # Get RNG state - serialize in a way that works across numpy versions
        rng_state = self.np_random.bit_generator.state
        
        # Serialize RNG state in a version-agnostic way
        if isinstance(rng_state['state'], dict):
            # New numpy format (PCG64, etc.)
            rng_state_tuple = (
                rng_state['bit_generator'],
                tuple(sorted(rng_state['state'].items())),  # Serialize as tuple of items
                rng_state.get('has_uint32', 0),
                rng_state.get('uinteger', 0),
            )
        else:
            # Old numpy format (MT19937)
            rng_state_tuple = (
                rng_state['bit_generator'],
                tuple(rng_state['state']['key']),
                rng_state['state']['pos'],
                rng_state.get('has_gauss', 0),
                rng_state.get('gauss', 0.0),
            )
        
        # Create complete state tuple
        state = (
            ('grid', tuple(grid_state)),
            ('agents', tuple(agents_state)),
            ('step_count', self.step_count),
            ('rng_state', rng_state_tuple),
        )
        
        return state
    
    def _serialize_object(self, obj):
        """Helper method to serialize a WorldObj into a hashable structure."""
        if obj is None:
            return None
        
        obj_data = {
            'type': obj.type,
            'color': obj.color,
            'init_pos': tuple(obj.init_pos) if obj.init_pos is not None else None,
            'cur_pos': tuple(obj.cur_pos) if obj.cur_pos is not None else None,
        }
        
        # Add type-specific properties
        if hasattr(obj, 'is_open'):
            obj_data['is_open'] = obj.is_open
        if hasattr(obj, 'is_locked'):
            obj_data['is_locked'] = obj.is_locked
        if hasattr(obj, 'index'):
            obj_data['index'] = obj.index
        if hasattr(obj, 'reward'):
            obj_data['reward'] = obj.reward
        if hasattr(obj, 'target_type'):
            obj_data['target_type'] = obj.target_type
        if hasattr(obj, 'contains'):
            if obj.contains is not None:
                obj_data['contains'] = self._serialize_object(obj.contains)
            else:
                obj_data['contains'] = None
        if hasattr(obj, 'pushable_by'):
            # For rocks, serialize the pushable_by attribute
            obj_data['pushable_by'] = obj.pushable_by
        
        return tuple(sorted(obj_data.items()))
    
    def set_state(self, state):
        """
        Set the environment to a specific state.
        
        Args:
            state: A state tuple as returned by get_state()
        """
        # Convert state tuple back to dict for easier access
        state_dict = dict(state)
        
        # Restore step count
        self.step_count = state_dict['step_count']
        
        # Restore RNG state - handle different formats
        rng_info = state_dict['rng_state']
        bit_generator_name = rng_info[0]
        
        # Reconstruct the RNG state dict based on format
        if isinstance(rng_info[1], tuple) and len(rng_info[1]) > 0 and isinstance(rng_info[1][0], tuple):
            # New format with dict serialized as tuple of items
            state_dict_items = dict(rng_info[1])
            rng_state = {
                'bit_generator': bit_generator_name,
                'state': state_dict_items,
                'has_uint32': rng_info[2],
                'uinteger': rng_info[3],
            }
        else:
            # Old format or fallback
            rng_state = {
                'bit_generator': bit_generator_name,
                'state': {
                    'key': np.array(rng_info[1], dtype=np.uint32),
                    'pos': rng_info[2],
                },
                'has_gauss': rng_info[3],
                'gauss': rng_info[4],
            }
        
        self.np_random.bit_generator.state = rng_state
        
        # Restore grid
        grid_data = state_dict['grid']
        idx = 0
        for j in range(self.grid.height):
            for i in range(self.grid.width):
                cell_data = grid_data[idx]
                idx += 1
                
                if cell_data is None:
                    self.grid.set(i, j, None)
                else:
                    obj = self._deserialize_object(dict(cell_data))
                    self.grid.set(i, j, obj)
        
        # Restore agent states
        agents_data = state_dict['agents']
        for agent_idx, agent_state in enumerate(agents_data):
            agent_dict = dict(agent_state)
            agent = self.agents[agent_idx]
            
            agent.pos = np.array(agent_dict['pos']) if agent_dict['pos'] is not None else None
            agent.dir = agent_dict['dir']
            agent.terminated = agent_dict['terminated']
            agent.started = agent_dict['started']
            agent.paused = agent_dict['paused']
            
            if agent_dict['carrying'] is not None:
                agent.carrying = self._deserialize_object(dict(agent_dict['carrying']))
            else:
                agent.carrying = None
    
    def _deserialize_object(self, obj_data):
        """Helper method to deserialize a WorldObj from a dictionary."""
        obj_type = obj_data['type']
        color = obj_data['color']
        
        # Create appropriate object based on type
        if obj_type == 'wall':
            obj = Wall(self.objects, color)
        elif obj_type == 'floor':
            obj = Floor(self.objects, color)
        elif obj_type == 'lava':
            obj = Lava(self.objects)
        elif obj_type == 'door':
            obj = Door(self.objects, color, 
                      is_open=obj_data.get('is_open', False),
                      is_locked=obj_data.get('is_locked', False))
        elif obj_type == 'key':
            obj = Key(self.objects, color)
        elif obj_type == 'ball':
            obj = Ball(self.objects, 
                      index=obj_data.get('index', 0),
                      reward=obj_data.get('reward', 1))
        elif obj_type == 'box':
            contains = None
            if 'contains' in obj_data and obj_data['contains'] is not None:
                contains = self._deserialize_object(dict(obj_data['contains']))
            obj = Box(self.objects, color, contains=contains)
        elif obj_type == 'goal':
            obj = Goal(self.objects, 
                      index=obj_data.get('index', 0),
                      reward=obj_data.get('reward', 1),
                      color=self.objects.COLOR_TO_IDX.get(color))
        elif obj_type == 'objgoal':
            obj = ObjectGoal(self.objects,
                           index=obj_data.get('index', 0),
                           target_type=obj_data.get('target_type', 'ball'),
                           reward=obj_data.get('reward', 1),
                           color=self.objects.COLOR_TO_IDX.get(color))
        elif obj_type == 'switch':
            obj = Switch(self.objects)
        elif obj_type == 'block':
            obj = Block(self.objects)
        elif obj_type == 'rock':
            obj = Rock(self.objects, pushable_by=obj_data.get('pushable_by'))
        elif obj_type == 'agent':
            # For agents in the grid, find the agent by its index attribute (color)
            agent_color_idx = obj_data.get('agent_index', 0)
            # Find the agent in self.agents that has this color index
            obj = None
            for agent in self.agents:
                if agent.index == agent_color_idx:
                    obj = agent
                    break
            if obj is None:
                raise ValueError(f"Could not find agent with index {agent_color_idx}")
        else:
            raise ValueError(f"Unknown object type: {obj_type}")
        
        # Restore position information
        if obj_data['init_pos'] is not None:
            obj.init_pos = np.array(obj_data['init_pos'])
        if obj_data['cur_pos'] is not None:
            obj.cur_pos = np.array(obj_data['cur_pos'])
        
        return obj
    
    def transition_probabilities(self, state, actions):
        """
        Given a state and vector of actions, return possible transitions with exact probabilities.
        
        **When transitions are probabilistic vs deterministic:**
        
        Transitions in multigrid environments are **deterministic** EXCEPT in the following case:
        - When multiple agents are active (not terminated/paused/started=False) AND
        - At least 2+ agents choose non-"still" actions AND  
        - The order of action execution matters for the outcome
        
        The ONLY source of non-determinism is the random permutation of agent execution order
        at line 1257 of the step() function: `order = np.random.permutation(len(actions))`
        
        All individual actions (left, right, forward, pickup, drop, toggle, etc.) are 
        themselves deterministic - there is no randomness in their effects.
        
        **When order matters:**
        Order of execution can matter when:
        - Two agents try to move into the same cell
        - Two agents try to pick up the same object
        - Agents interact with each other (e.g., stealing carried objects)
        - An agent's action depends on the grid state that another agent modifies
        
        **Computing exact probabilities (optimized):**
        Since np.random.permutation creates a uniform distribution over all n! permutations,
        each permutation has probability 1/n! where n is the number of agents.
        
        **Optimization strategy:**
        Most of the time, most permutations lead to the same result because:
        1. Agents are often far apart and don't interact
        2. Only rotation/still actions are used (always commutative)
        3. Order only matters for conflicting actions
        
        We optimize by using **conflict block partitioning** (suggested by @mensch72):
        1. Early exit if ≤1 active agents (deterministic)
        2. Early exit if all actions are rotations (commutative)
        3. Partition agents into conflict blocks (agents competing for same resource)
        4. Compute Cartesian product of blocks instead of all k! permutations
        5. Each outcome has equal probability: 1 / product(block_sizes)
        
        This is MORE efficient than permutation enumeration:
        - If 2 blocks of 2 agents each: 2×2=4 outcomes instead of 4!=24 permutations
        - Most blocks are singletons (no conflicts), making this very fast
        - Only conflicting agents need to be considered in different orderings
        
        Args:
            state: A state tuple as returned by get_state()
            actions: List of action indices, one per agent
            
        Returns:
            list: List of (probability, successor_state) tuples describing all
                  possible transitions. Returns None if the state is terminal
                  or if any action is not feasible in the given state.
        """
        # Check if we're in a terminal state
        state_dict = dict(state)
        if state_dict['step_count'] >= self.max_steps:
            return None
        
        # Check if all actions are valid
        for action in actions:
            if action < 0 or action >= self.action_space.n:
                return None
        
        # Save current state to restore later
        original_state = self.get_state()
        
        try:
            # Restore to the query state
            self.set_state(state)
            
            num_agents = len(self.agents)
            
            # Identify which agents will actually act
            active_agents = []
            inactive_agents = []
            for i in range(num_agents):
                if (not self.agents[i].terminated and 
                    not self.agents[i].paused and 
                    self.agents[i].started and 
                    actions[i] != self.actions.still):
                    active_agents.append(i)
                else:
                    inactive_agents.append(i)
            
            # OPTIMIZATION 1: If ≤1 agents active, transition is deterministic
            if len(active_agents) <= 1:
                # Only one or zero agents acting - order doesn't matter
                successor_state = self._compute_successor_state(state, actions, tuple(range(num_agents)))
                return [(1.0, successor_state)]
            
            # OPTIMIZATION 2: Check if all but at most one actions are rotations (left/right)
            # Rotations never interfere with each other, so order doesn't matter
            n_non_rotations = sum(
                actions[i] not in [self.actions.left, self.actions.right] 
                for i in active_agents
            )
            if n_non_rotations < 2:
                # Rotations are commutative - result is deterministic
                successor_state = self._compute_successor_state(state, actions, tuple(range(num_agents)))
                return [(1.0, successor_state)]
            
            # OPTIMIZATION 3: Partition agents into conflict blocks
            # This is MORE efficient than permuting all active agents
            # Instead of k! permutations, we compute the Cartesian product of conflict blocks
            conflict_blocks = self._identify_conflict_blocks(state, actions, active_agents)
            
            # If all blocks are singletons (no conflicts), result is deterministic
            if all(len(block) == 1 for block in conflict_blocks):
                successor_state = self._compute_successor_state(state, actions, tuple(range(num_agents)))
                return [(1.0, successor_state)]
            
            # OPTIMIZATION 4: Compute outcomes via Cartesian product of conflict blocks
            # Each outcome has probability = 1 / product(block_sizes)
            
            # Generate all possible "winner" combinations (one per block)
            # Each tuple in the product represents which agent succeeds in each block
            block_sizes = [len(block) for block in conflict_blocks]
            total_outcomes = 1
            for size in block_sizes:
                total_outcomes *= size
            
            # Each outcome has equal probability
            outcome_probability = 1.0 / total_outcomes
            
            # Compute successor state for each outcome
            successor_states = {}
            
            for winner_indices in product(*[range(len(block)) for block in conflict_blocks]):
                # winner_indices[i] tells us which agent wins in block i
                winners = [conflict_blocks[i][winner_indices[i]] for i in range(len(conflict_blocks))]
                
                # Compute the successor state for this outcome
                # We need an ordering that respects the winners
                ordering = self._build_ordering_with_winners(num_agents, active_agents, 
                                                             conflict_blocks, winners)
                succ_state = self._compute_successor_state(state, actions, ordering)
                
                # Aggregate probabilities for identical successor states
                if succ_state not in successor_states:
                    successor_states[succ_state] = 0.0
                successor_states[succ_state] += outcome_probability
            
            # Convert to result list
            result = [(prob, state) for state, prob in successor_states.items()]
            
            # Sort by probability (descending) for consistency
            result.sort(key=lambda x: x[0], reverse=True)
            
            return result
            
        finally:
            # Always restore original state
            self.set_state(original_state)
    
    def _identify_conflict_blocks(self, state, actions, active_agents):
        """
        Partition active agents into conflict blocks where agents compete for resources.
        
        Agents are in the same block if they:
        - Try to move into the same cell (forward action to same position)
        - Try to pick up the same object
        - Interact with each other directly
        
        Args:
            state: Current state tuple
            actions: List of action indices
            active_agents: List of active agent indices
            
        Returns:
            list of lists: Each inner list is a conflict block of agent indices
        """
        # Constants for resource types
        RESOURCE_INDEPENDENT = 'independent'
        RESOURCE_CELL = 'cell'
        RESOURCE_PICKUP = 'pickup'
        RESOURCE_DROP_AGENT = 'drop_agent'
        
        # Restore to the query state to inspect agent positions and targets
        self.set_state(state)
        
        # Track which resource each agent targets
        agent_targets = {}  # agent_idx -> resource identifier
        
        for agent_idx in active_agents:
            action = actions[agent_idx]
            agent = self.agents[agent_idx]
            
            # Determine what resource this agent is targeting
            if action == self.actions.forward:
                # Check what the agent is trying to move into
                fwd_pos = agent.front_pos
                fwd_cell = self.grid.get(*fwd_pos)
                
                # If pushing blocks/rocks, the resource is the final cell where objects land
                if fwd_cell and fwd_cell.type in ['block', 'rock']:
                    can_push, num_objects, end_pos = self._can_push_objects(agent, fwd_pos)
                    if can_push:
                        # Agent will push objects and move into fwd_pos
                        # The contested resource is the end_pos where objects land
                        agent_targets[agent_idx] = (RESOURCE_CELL, tuple(end_pos))
                    else:
                        # Can't push, agent won't move - independent action
                        agent_targets[agent_idx] = (RESOURCE_INDEPENDENT, agent_idx)
                else:
                    # Normal movement - target is the cell they're moving into
                    target_pos = tuple(agent.front_pos)
                    agent_targets[agent_idx] = (RESOURCE_CELL, target_pos)
            
            elif action == self.actions.pickup:
                # Target is the object at the forward position
                fwd_pos = agent.front_pos
                fwd_cell = self.grid.get(*fwd_pos)
                if fwd_cell and fwd_cell.can_pickup():
                    # Use object position as identifier
                    agent_targets[agent_idx] = (RESOURCE_PICKUP, tuple(fwd_pos))
                else:
                    # No valid target, agent acts independently
                    agent_targets[agent_idx] = (RESOURCE_INDEPENDENT, agent_idx)
            
            elif action == self.actions.drop:
                # Check if dropping on another agent or specific location
                fwd_pos = agent.front_pos
                fwd_cell = self.grid.get(*fwd_pos)
                if fwd_cell and fwd_cell.type == 'agent':
                    # Interacting with another agent
                    agent_targets[agent_idx] = (RESOURCE_DROP_AGENT, tuple(fwd_pos))
                else:
                    # Dropping on ground, independent action
                    agent_targets[agent_idx] = (RESOURCE_INDEPENDENT, agent_idx)
            
            else:
                # Other actions (toggle, build, done) - typically independent
                agent_targets[agent_idx] = (RESOURCE_INDEPENDENT, agent_idx)
        
        # Group agents by their target resource
        resource_to_agents = {}
        for agent_idx, resource in agent_targets.items():
            if resource not in resource_to_agents:
                resource_to_agents[resource] = []
            resource_to_agents[resource].append(agent_idx)
        
        # Convert to list of blocks
        # Only resources with multiple agents form conflict blocks
        conflict_blocks = []
        for resource, agent_list in resource_to_agents.items():
            if len(agent_list) > 1 and resource[0] != RESOURCE_INDEPENDENT:
                # Multiple agents competing for same resource
                conflict_blocks.append(agent_list)
            else:
                # Each agent acts independently (singleton blocks)
                for agent_idx in agent_list:
                    conflict_blocks.append([agent_idx])
        
        return conflict_blocks
    
    def _build_ordering_with_winners(self, num_agents, active_agents, conflict_blocks, winners):
        """
        Build an agent ordering that respects conflict block winners.
        
        Winners go first in their blocks, giving them priority for contested resources.
        
        Args:
            num_agents: Total number of agents
            active_agents: List of active agent indices
            conflict_blocks: List of conflict blocks
            winners: List of winning agent indices (one per block)
            
        Returns:
            tuple: Full ordering of all agent indices
        """
        # Build ordering: winners first, then other agents in original order
        ordering = []
        
        # Track which agents are winners
        winner_set = set(winners)
        
        # Add winners first (in the order they win)
        for winner in winners:
            ordering.append(winner)
        
        # Add all inactive agents in their original order
        for i in range(num_agents):
            if i not in active_agents:
                ordering.append(i)
        
        # Add non-winning active agents
        for agent_idx in active_agents:
            if agent_idx not in winner_set:
                ordering.append(agent_idx)
        
        return tuple(ordering)
    
    def _build_full_ordering(self, num_agents, active_perm, inactive_agents):
        """
        Build a full agent ordering from a permutation of active agents.
        
        Inactive agents are placed in their original positions, active agents
        are placed in the order specified by active_perm.
        
        Args:
            num_agents: Total number of agents
            active_perm: Tuple of active agent indices in some order
            inactive_agents: List of inactive agent indices
            
        Returns:
            tuple: Full ordering of all agent indices
        """
        # Build ordering that respects relative order of active agents
        # but keeps inactive agents in their original positions
        ordering = []
        active_iter = iter(active_perm)
        
        for i in range(num_agents):
            if i in inactive_agents:
                ordering.append(i)
            else:
                ordering.append(next(active_iter))
        
        return tuple(ordering)
    
    def _compute_successor_state(self, state, actions, ordering):
        """
        Compute the exact successor state for given actions and agent ordering.
        
        This is a pure computation that doesn't rely on RNG or executing step().
        It replicates the deterministic logic of step() with a fixed ordering.
        
        Args:
            state: Current state tuple
            actions: List of action indices
            ordering: Tuple specifying the order in which agents act
            
        Returns:
            tuple: The successor state
        """
        # Start from the given state
        self.set_state(state)
        
        # Increment step counter
        self.step_count += 1
        
        # Execute each agent's action in the specified order
        rewards = np.zeros(len(actions))
        done = False
        
        for i in ordering:
            # Skip if agent shouldn't act
            if (self.agents[i].terminated or 
                self.agents[i].paused or 
                not self.agents[i].started or 
                actions[i] == self.actions.still):
                continue
            
            # Get the position in front of the agent
            fwd_pos = self.agents[i].front_pos
            
            # Get the contents of the cell in front of the agent
            fwd_cell = self.grid.get(*fwd_pos)
            
            # Execute the action
            if actions[i] == self.actions.left:
                # Rotate left
                self.agents[i].dir -= 1
                if self.agents[i].dir < 0:
                    self.agents[i].dir += 4
            
            elif actions[i] == self.actions.right:
                # Rotate right
                self.agents[i].dir = (self.agents[i].dir + 1) % 4
            
            elif actions[i] == self.actions.forward:
                # Move forward
                moved = False
                # Check if forward cell contains a block or rock that can be pushed
                if fwd_cell is not None and fwd_cell.type in ['block', 'rock']:
                    # Try to push the object(s)
                    pushed = self._push_objects(self.agents[i], fwd_pos)
                    moved = pushed
                elif fwd_cell is not None:
                    if fwd_cell.type == 'goal':
                        done = True
                        self._reward(i, rewards, 1)
                        # Agent can still move onto goal
                        self.grid.set(*fwd_pos, self.agents[i])
                        self.grid.set(*self.agents[i].pos, None)
                        self.agents[i].pos = fwd_pos
                        moved = True
                    elif fwd_cell.type == 'switch':
                        self._handle_switch(i, rewards, fwd_pos, fwd_cell)
                        # Agent can move onto switch
                        self.grid.set(*fwd_pos, self.agents[i])
                        self.grid.set(*self.agents[i].pos, None)
                        self.agents[i].pos = fwd_pos
                        moved = True
                    elif fwd_cell.can_overlap():
                        self.grid.set(*fwd_pos, self.agents[i])
                        self.grid.set(*self.agents[i].pos, None)
                        self.agents[i].pos = fwd_pos
                        moved = True
                elif fwd_cell is None:
                    self.grid.set(*fwd_pos, self.agents[i])
                    self.grid.set(*self.agents[i].pos, None)
                    self.agents[i].pos = fwd_pos
                    moved = True
                self._handle_special_moves(i, rewards, fwd_pos, fwd_cell)
            
            elif 'build' in self.actions.available and actions[i] == self.actions.build:
                self._handle_build(i, rewards, fwd_pos, fwd_cell)
            
            elif actions[i] == self.actions.pickup:
                # Pick up an object
                self._handle_pickup(i, rewards, fwd_pos, fwd_cell)
            
            elif actions[i] == self.actions.drop:
                # Drop an object
                self._handle_drop(i, rewards, fwd_pos, fwd_cell)
            
            elif actions[i] == self.actions.toggle:
                # Toggle/activate an object
                if fwd_cell:
                    fwd_cell.toggle(self, fwd_pos)
            
            elif actions[i] == self.actions.done:
                # Done action (not used by default)
                pass
            
            else:
                # Invalid action
                return None
        
        # Check if max steps reached
        if self.step_count >= self.max_steps:
            done = True
        
        # Return the resulting state
        return self.get_state()

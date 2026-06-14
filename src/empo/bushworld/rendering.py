"""
Rendering and movie generation for BushWorld (matplotlib based).

Color scheme (per the issue):
- empty cells (density 0)  -> black
- fully dense cells (B)    -> "satisfied" brown, linearly interpolated in between
- robots                   -> grey squares (labelled R0, R1, ...)
- humans                   -> yellow circles (labelled H0, H1, ...)
- goals                    -> dashed blue rectangles (with a line to the owner)

``render_frame`` returns an ``(H, W, 3)`` uint8 RGB array. ``save_movie`` writes a
list of such frames to an ``.mp4`` or ``.gif``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, TYPE_CHECKING

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.patches as patches  # noqa: E402

if TYPE_CHECKING:  # pragma: no cover
    from empo.bushworld.env import BushWorld

BROWN = np.array([139, 90, 43]) / 255.0
BLACK = np.array([0, 0, 0]) / 255.0
ROBOT_COLOR = np.array([100, 100, 100]) / 255.0
HUMAN_COLOR = np.array([255, 255, 0]) / 255.0
GOAL_COLOR = (0.0, 0.4, 1.0, 0.9)


def _density_color(d: int, B: int) -> np.ndarray:
    frac = 0.0 if B <= 0 else min(max(d / B, 0.0), 1.0)
    return (1.0 - frac) * BLACK + frac * BROWN


def _goal_bbox(goal) -> tuple:
    if hasattr(goal, "target_rect"):
        return tuple(int(c) for c in goal.target_rect)
    if hasattr(goal, "target_pos"):
        x, y = goal.target_pos
        return (int(x), int(y), int(x), int(y))
    if isinstance(goal, (tuple, list)):
        if len(goal) == 2:
            x, y = goal
            return (int(x), int(y), int(x), int(y))
        if len(goal) == 4:
            return tuple(int(c) for c in goal)
    raise ValueError(f"Unsupported goal for rendering: {goal!r}")


def render_frame(
    env: "BushWorld",
    tile_size: int = 48,
    annotation_text: Optional[Sequence[str]] = None,
    annotation_panel_width: int = 260,
    annotation_font_size: int = 11,
    goal_overlays: Optional[Dict[int, Any]] = None,
    dpi: int = 100,
) -> np.ndarray:
    """Render the current state of ``env`` to an RGB array."""
    width = env.width
    height = env.height
    state = env.get_state()
    _, positions, densities = state

    grid_px_w = width * tile_size
    grid_px_h = height * tile_size
    panel_px = annotation_panel_width if annotation_text is not None else 0
    fig_w = (grid_px_w + panel_px) / dpi
    fig_h = max(grid_px_h, 1) / dpi

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)

    # --- grid axes ---
    ax = fig.add_axes([0.0, 0.0, grid_px_w / (grid_px_w + panel_px), 1.0])
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)  # y grows downward
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")

    # bush density cells
    for y in range(height):
        for x in range(width):
            d = densities[env.cell_index(x, y)]
            color = _density_color(d, env.B)
            ax.add_patch(
                patches.Rectangle((x, y), 1, 1, facecolor=color, edgecolor=(0.25, 0.25, 0.25), linewidth=0.5)
            )
            if d > 0:
                ax.text(
                    x + 0.85,
                    y + 0.2,
                    str(d),
                    color=(0.85, 0.85, 0.85),
                    fontsize=annotation_font_size * 0.7,
                    ha="right",
                    va="top",
                )

    # goal overlays (dashed blue)
    if goal_overlays:
        for agent_idx, goal in goal_overlays.items():
            x1, y1, x2, y2 = _goal_bbox(goal)
            inset = 0.08
            rect = patches.Rectangle(
                (x1 + inset, y1 + inset),
                (x2 - x1 + 1) - 2 * inset,
                (y2 - y1 + 1) - 2 * inset,
                linewidth=2.0,
                edgecolor=GOAL_COLOR,
                facecolor="none",
                linestyle="--",
            )
            ax.add_patch(rect)
            if agent_idx < len(positions):
                ax_, ay_ = positions[agent_idx]
                cx = min(max(ax_ + 0.5, x1 + inset), x2 + 1 - inset)
                cy = min(max(ay_ + 0.5, y1 + inset), y2 + 1 - inset)
                ax.plot(
                    [ax_ + 0.5, cx],
                    [ay_ + 0.5, cy],
                    linestyle="--",
                    linewidth=1.5,
                    color=GOAL_COLOR[:3],
                    alpha=0.7,
                )

    # players
    for i, (x, y) in enumerate(positions):
        is_robot = i < env.num_robots
        if is_robot:
            ax.add_patch(
                patches.Rectangle(
                    (x + 0.15, y + 0.15), 0.7, 0.7, facecolor=ROBOT_COLOR, edgecolor="white", linewidth=1.0
                )
            )
            label = f"R{i}"
            text_color = "white"
        else:
            ax.add_patch(
                patches.Circle((x + 0.5, y + 0.5), 0.34, facecolor=HUMAN_COLOR, edgecolor="black", linewidth=1.0)
            )
            label = f"H{i - env.num_robots}"
            text_color = "black"
        ax.text(
            x + 0.5,
            y + 0.5,
            label,
            color=text_color,
            fontsize=annotation_font_size * 0.8,
            ha="center",
            va="center",
            fontweight="bold",
        )

    # --- annotation panel ---
    if annotation_text is not None:
        pax = fig.add_axes(
            [grid_px_w / (grid_px_w + panel_px), 0.0, panel_px / (grid_px_w + panel_px), 1.0]
        )
        pax.axis("off")
        pax.set_xlim(0, 1)
        pax.set_ylim(0, 1)
        lines = (
            annotation_text
            if isinstance(annotation_text, (list, tuple))
            else str(annotation_text).splitlines()
        )
        y_cursor = 0.98
        line_step = (annotation_font_size + 4) / (fig_h * dpi)
        for line in lines:
            color = "black"
            text = line
            if line.startswith(">"):
                color = "green"
                text = line[1:]
            elif line.startswith("!"):
                color = "red"
                text = line[1:]
            pax.text(
                0.02,
                y_cursor,
                text,
                color=color,
                fontsize=annotation_font_size,
                family="monospace",
                ha="left",
                va="top",
            )
            y_cursor -= line_step

    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return buf


def save_movie(frames: List[np.ndarray], path: str, fps: int = 4) -> str:
    """Save a list of RGB frames to ``path`` (``.mp4`` or ``.gif``).

    Frames are padded to a common size if necessary. Returns ``path``.
    """
    if not frames:
        raise ValueError("No frames to save")

    # Pad frames to common dimensions.
    max_h = max(f.shape[0] for f in frames)
    max_w = max(f.shape[1] for f in frames)
    padded = []
    for f in frames:
        if f.shape[0] != max_h or f.shape[1] != max_w:
            canvas = np.full((max_h, max_w, 3), 255, dtype=np.uint8)
            canvas[: f.shape[0], : f.shape[1]] = f
            padded.append(canvas)
        else:
            padded.append(f)

    import imageio.v3 as iio

    if path.endswith(".gif"):
        iio.imwrite(path, padded, duration=int(1000 / max(fps, 1)), loop=0)
    else:
        iio.imwrite(path, padded, fps=fps)
    return path

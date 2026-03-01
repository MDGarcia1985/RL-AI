"""
trail_viz.py

Plot agent trajectory (path) on a grid, with optional maze walls.
Used for "best maze run" and grid-world path visualization in main_panel.

Design intent:
- Single function plot_trail() consumed by main_panel for the "Best run (trail)" section.
- trajectory comes from train_runner (best_trajectory: list of (row, col)).
- When env is MazeEnv, walls are passed so the path is drawn over the wall layout.
- Grid convention: row 0 at top, same as GridEnv/MazeEnv and policy heatmaps.

Copyright (c) 2026 Michael Garcia, M&E Design
https://mandedesign.studio
CSC370 Spring 2026
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Trail plot (best run path)
# ---------------------------------------------------------------------------

def plot_trail(
    trajectory: List[Tuple[int, int]],
    rows: int,
    cols: int,
    walls: Optional[np.ndarray] = None,
    start_marker: str = "s",
    goal_marker: str = "*",
) -> plt.Figure:
    """
    Draw the agent's path on a grid.

    - trajectory: list of (row, col) positions in order (from train_runner best_trajectory).
    - rows, cols: grid dimensions (match env).
    - walls: optional (rows, cols) bool array; True = wall (drawn as dark). Pass env.walls for maze.
    - start_marker, goal_marker: matplotlib marker for first and last cell.

    Returns a matplotlib Figure (caller can pass to st.pyplot(fig)).
    """
    if not trajectory:
        # No successful run: show placeholder so the section still renders
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_xlim(0, max(cols, 1))
        ax.set_ylim(max(rows, 1), 0)
        ax.set_title("Best run (no trajectory)")
        return fig

    fig, ax = plt.subplots(figsize=(6, 6))

    # Background: walls if provided (row 0 at top).
    # extent = (left, right, bottom, top) so image aligns with (col, row) plot coords.
    if walls is not None and walls.shape[0] == rows and walls.shape[1] == cols:
        bg = np.where(walls, 0.3, 1.0)  # dark = wall, light = open
        ax.imshow(bg, cmap="gray", vmin=0, vmax=1, aspect="equal", extent=(0, cols, rows, 0), origin="upper")
    else:
        ax.set_facecolor("#f8f8f8")

    # Trail: we store (row, col); matplotlib x = col, y = row. Set ylim so row 0 at top.
    xs = [c for _r, c in trajectory]
    ys = [r for r, _c in trajectory]
    ax.plot(xs, ys, "b-", linewidth=2, alpha=0.8, label="Path")
    ax.scatter(xs, ys, c="blue", s=20, alpha=0.6, zorder=3)

    # Start and goal: first and last (row, col) in trajectory
    r0, c0 = trajectory[0]
    r1, c1 = trajectory[-1]
    ax.scatter([c0], [r0], marker=start_marker, s=200, c="green", edgecolors="darkgreen", linewidths=2, label="Start", zorder=4)
    ax.scatter([c1], [r1], marker=goal_marker, s=200, c="gold", edgecolors="orange", linewidths=2, label="Goal", zorder=4)

    ax.set_xlim(-0.5, cols - 0.5)
    ax.set_ylim(rows - 0.5, -0.5)  # row 0 at top (same as heatmaps / env convention)
    ax.set_aspect("equal")
    ax.set_title("Best run (trail)")
    # Ticks/grid only when readable (same threshold as main_panel heatmaps)
    if rows <= 30 and cols <= 30:
        ax.set_xticks(np.arange(cols))
        ax.set_yticks(np.arange(rows))
        ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    return fig

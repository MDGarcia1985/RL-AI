"""
test_maze_generation.py

Procedural maze generation tests.

Design intent:
- Same seed must produce identical wall grids.
- Different seeds should produce different wall grids.
- Walls grid must match expected shape and dtype.
"""

from __future__ import annotations

import numpy as np

from rc_agents.edge_ai.rcg_edge.runners.maze_runner import generate_dfs_maze_walls


def test_dfs_maze_same_seed_reproducible() -> None:
    rows, cols = 11, 11
    seed = 123

    rng1 = np.random.default_rng(seed)
    rng2 = np.random.default_rng(seed)

    walls1 = generate_dfs_maze_walls(rows, cols, rng=rng1)
    walls2 = generate_dfs_maze_walls(rows, cols, rng=rng2)

    expected_rows = rows if rows % 2 == 1 else rows + 1
    expected_cols = cols if cols % 2 == 1 else cols + 1

    assert walls1.shape == (expected_rows, expected_cols)
    assert walls1.dtype == np.bool_
    assert np.array_equal(walls1, walls2)


def test_dfs_maze_different_seed_likely_different() -> None:
    rows, cols = 11, 11

    rng1 = np.random.default_rng(123)
    rng2 = np.random.default_rng(999)

    walls1 = generate_dfs_maze_walls(rows, cols, rng=rng1)
    walls2 = generate_dfs_maze_walls(rows, cols, rng=rng2)

    assert not np.array_equal(walls1, walls2)


def test_dfs_maze_structure_invariants() -> None:
    rng = np.random.default_rng(123)
    walls = generate_dfs_maze_walls(11, 11, rng=rng)

    assert bool(walls[1, 1]) is False  # carved start cell
    assert bool(walls[0, :].all()) is True
    assert bool(walls[-1, :].all()) is True
    assert bool(walls[:, 0].all()) is True
    assert bool(walls[:, -1].all()) is True
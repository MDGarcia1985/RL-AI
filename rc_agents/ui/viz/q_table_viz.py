"""
q_table_viz.py

Utilities for turning a Q-table into human-readable tables and grids.
Designed to be UI-agnostic (Streamlit, Tkinter, CLI can all reuse it).
"""

from __future__ import annotations

from typing import Dict, Hashable, Tuple
import numpy as np

from ...edge_ai.rcg_edge.agents.base import Action

Obs = Tuple[int, int]


def q_table_to_matrix(
    q_table: Dict[Hashable, np.ndarray],
    rows: int,
    cols: int,
) -> np.ndarray:
    """
    Convert q_table dict into a dense matrix: shape (rows*cols, num_actions).

    Missing states are filled with zeros.
    State key is assumed to be (row, col) tuples for Phase 1.
    """
    num_actions = len(Action)
    mat = np.zeros((rows * cols, num_actions), dtype=float)

    for key, qvals in q_table.items():
        if not isinstance(key, tuple) or len(key) != 2:
            continue  # ignore non-grid states
        r, c = int(key[0]), int(key[1])
        if 0 <= r < rows and 0 <= c < cols:
            idx = r * cols + c
            mat[idx, :] = np.asarray(qvals, dtype=float)

    return mat


def state_value_grid(
    q_table: Dict[Hashable, np.ndarray],
    rows: int,
    cols: int,
) -> np.ndarray:
    """
    Returns a (rows, cols) grid of V(s) where V(s) = max_a Q(s,a)
    """
    mat = q_table_to_matrix(q_table, rows, cols)
    v = np.max(mat, axis=1)
    return v.reshape((rows, cols))

def greedy_policy_grid(
    q_table: Dict[Hashable, np.ndarray],
    rows: int,
    cols: int,
) -> np.ndarray:
    """
    Returns a (rows, cols) grid of best-action indices (0..num_actions-1)
    where best action = argmax_a Q(s,a)

    Missing states default to 0.
    """
    mat = q_table_to_matrix(q_table, rows, cols)
    best = np.argmax(mat, axis=1)
    return best.reshape((rows, cols))


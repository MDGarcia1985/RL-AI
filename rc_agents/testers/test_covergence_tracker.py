"""
test_convergence_tracker.py

ConvergenceTracker Tests
Verifies rolling-window behavior and convergence milestone detection.

Design intent:
- Tests are behavioral, not cosmetic.
- We verify correctness of:
  - Rolling window math (win-rate and avg steps on wins)
  - Perfect-window detection (100% wins across window)
  - Saturation detection (win-rate plateau for N windows)
  - Steps plateau detection (efficiency no longer improves)

Copyright (c) 2026 Michael Garcia, M&E Design
https://mandedesign.studio
michael@mandedesign.studio

CSC370 Spring 2026
"""
from __future__ import annotations

from rc_agents.edge_ai.rcg_edge.runners.convergence_tracker import ConvergenceTracker


def test_tracker_does_not_trigger_before_window_fills() -> None:
    """
    Until the rolling window is full, convergence metrics are not meaningful.

    We expect:
    - No milestones are set
    - last_window_win_rate remains its initial cached value
    """
    tracker = ConvergenceTracker(window=5, delta=0.1, patience=2)

    # Only 4 updates -> window not full.
    for _ in range(4):
        tracker.update(reached_goal=True, steps=10)

    s = tracker.summary()
    assert s.episode_first_perfect is None
    assert s.episode_first_saturation is None
    assert s.episode_first_steps_plateau is None


def test_perfect_window_is_detected_when_all_wins_in_window() -> None:
    """
    "Perfect" means 100% wins across a full rolling window.
    """
    tracker = ConvergenceTracker(window=5, delta=0.1, patience=2)

    for _ in range(5):
        tracker.update(reached_goal=True, steps=10)

    s = tracker.summary()
    assert s.last_window_win_rate == 1.0
    assert s.episode_first_perfect == 5


def test_win_rate_plateau_triggers_saturation_after_patience_windows() -> None:
    """
    Saturation is declared when win-rate fails to improve by delta for 'patience' windows.

    We simulate a stable (non-improving) win-rate once the window fills.

    Setup:
    - window=4
    - delta=0.25 (requires a big improvement to count)
    - patience=2 (two consecutive non-improving windows triggers saturation)

    Sequence:
    - First 4 episodes: 2 wins / 2 losses -> win_rate = 0.5
    - Next 2 episodes: keep win_rate at 0.5 (no improvement)
    - Expect saturation at episode 6 (the 2nd non-improving full window after best is established)
    """
    tracker = ConvergenceTracker(window=4, delta=0.25, patience=2)

    # Fill window to establish initial best.
    # wins pattern: 1,1,0,0 => 0.5
    tracker.update(reached_goal=True, steps=10)
    tracker.update(reached_goal=True, steps=10)
    tracker.update(reached_goal=False, steps=10)
    tracker.update(reached_goal=False, steps=10)

    # Two more episodes that keep rolling win-rate at 0.5
    # (Any mix that doesn't exceed 0.75, because delta=0.25)
    tracker.update(reached_goal=True, steps=10)
    tracker.update(reached_goal=False, steps=10)

    s = tracker.summary()
    assert s.episode_first_saturation == 6
    assert s.saturation_reason == "win_rate_plateau"


def test_avg_steps_win_is_over_wins_in_window_only() -> None:
    """
    avg_steps_win should be computed only over winning episodes in the window.

    Here:
    - window=4
    - wins have steps: 10 and 30
    - losses contribute 0 to steps and do not count as wins
    - avg_steps_win should be (10+30)/2 = 20
    """
    tracker = ConvergenceTracker(window=4)

    tracker.update(reached_goal=True, steps=10)
    tracker.update(reached_goal=False, steps=999)  # ignored for avg steps win
    tracker.update(reached_goal=True, steps=30)
    tracker.update(reached_goal=False, steps=999)  # ignored for avg steps win

    s = tracker.summary()
    assert s.last_window_win_rate == 0.5
    assert s.last_window_avg_steps_win == 20.0


def test_steps_plateau_triggers_when_efficiency_stops_improving() -> None:
    """
    Steps plateau triggers when avg steps-to-goal (wins in window) stops improving.

    Setup:
    - window=4
    - steps_delta=1.0
    - steps_patience=2

    We create windows where avg_steps_win improves once, then stops improving
    for two consecutive windows.
    """
    tracker = ConvergenceTracker(window=4, steps_delta=1.0, steps_patience=2)

    # Window 1: wins steps 40, 40 => avg = 40
    tracker.update(reached_goal=True, steps=40)
    tracker.update(reached_goal=False, steps=0)
    tracker.update(reached_goal=True, steps=40)
    tracker.update(reached_goal=False, steps=0)

    # Window 2: improve to avg 30 (wins 30, 30)
    tracker.update(reached_goal=True, steps=30)
    tracker.update(reached_goal=False, steps=0)
    tracker.update(reached_goal=True, steps=30)
    tracker.update(reached_goal=False, steps=0)

    # Window 3: no meaningful improvement (still avg 30)
    tracker.update(reached_goal=True, steps=30)
    tracker.update(reached_goal=False, steps=0)
    tracker.update(reached_goal=True, steps=30)
    tracker.update(reached_goal=False, steps=0)

    # Window 4: no meaningful improvement again -> should trigger plateau (patience=2)
    tracker.update(reached_goal=True, steps=30)
    tracker.update(reached_goal=False, steps=0)
    tracker.update(reached_goal=True, steps=30)
    tracker.update(reached_goal=False, steps=0)

    s = tracker.summary()
    assert s.episode_first_steps_plateau is not None
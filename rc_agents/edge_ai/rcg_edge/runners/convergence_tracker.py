"""
convergence_tracker.py

Convergence / Saturation Tracking Helper
Small, dependency-free rolling statistics tracker for training runs.

Purpose:
- Detect when an agent reaches:
  1) 100% goal acquisition (perfect win-rate) over a rolling window
  2) Saturation (win-rate stops improving meaningfully for multiple windows)
  3) Optional efficiency saturation (avg steps-to-goal stops improving)

Design intent:
- Keep training loop clean (runner owns episodes; UI only displays).
- Track "is it still learning?" without guessing.
- Provide stable, loggable milestones for engineering notebooks.

Usage (inside train_runner.run_training):
    tracker = ConvergenceTracker(window=200, delta=0.005, patience=5)

    for ep in range(1, cfg.episodes + 1):
        ... run episode ...
        tracker.update(reached_goal=reached_goal, steps=steps)

    summary = tracker.summary()

Then append to results metadata or print/log summary.

Copyright (c) 2026 Michael Garcia, M&E Design
https://mandedesign.studio
michael@mandedesign.studio

CSC370 Spring 2026
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Deque, Optional
from collections import deque


@dataclass(frozen=True)
class ConvergenceSummary:
    """
    ConvergenceSummary is a snapshot of training convergence signals.

    Notes:
    - Episodes are 1-indexed to match your runner convention.
    - "Perfect" means 100% wins across a full rolling window.
    - "Saturation" means win-rate improvement stays below delta for 'patience' windows.
    """
    window: int
    delta: float
    patience: int

    last_window_win_rate: float
    last_window_avg_steps_win: Optional[float]

    # Episode indices where conditions were first met.
    episode_first_perfect: Optional[int]
    episode_first_saturation: Optional[int]
    episode_first_steps_plateau: Optional[int]

    # Debug / interpretation helpers
    saturation_reason: Optional[str]


class ConvergenceTracker:
    """
    Tracks rolling window win-rate and avg steps-to-goal.

    Core signals:
    - Perfect (100% wins over window)
    - Saturation (win-rate no longer improves meaningfully)
    - Steps plateau (avg steps-to-goal no longer improves meaningfully)

    Why both win-rate and steps?
    - Win-rate can hit 100% while efficiency still improves.
    - Steps-to-goal can continue improving even after "always wins."

    NOTE (important):
    - win-rate is computed over the last W episodes.
    - avg_steps_win is computed over the wins *within* the last W episodes.
      (i.e., "wins in window", not "last W wins".)
    """

    def __init__(
        self,
        *,
        window: int = 200,
        delta: float = 0.005,
        patience: int = 5,
        steps_delta: float = 1.0,
        steps_patience: int = 5,
    ) -> None:
        # Rolling window size for metrics.
        self.window = int(window)

        # delta is "meaningful improvement threshold" for win-rate (absolute, not relative).
        # Example: delta=0.005 means "must improve by at least 0.5% to count."
        self.delta = float(delta)
        self.patience = int(patience)

        # Optional steps plateau tracking (wins only).
        # steps_delta is "meaningful improvement threshold" for avg steps-to-goal.
        self.steps_delta = float(steps_delta)
        self.steps_patience = int(steps_patience)

        # Deques store the most recent 'window' episode outcomes.
        # Both are capped at maxlen=window so expired episodes fall off automatically.
        self._wins: Deque[int] = deque(maxlen=self.window)

        # Steps aligned with episodes:
        # - store steps for wins, and 0 for non-wins
        # - track win_count_in_window so avg is only over wins
        self._steps_per_ep: Deque[int] = deque(maxlen=self.window)

        # Running totals mirror the deques above — updated on append/eviction
        # so win-rate and avg-steps are O(1) instead of O(window) per episode.
        self._wins_sum: int = 0
        self._steps_sum: int = 0
        self._win_count: int = 0

        # Episode counter (1-indexed externally; stored as int internally).
        self._episode_count: int = 0

        # Milestones (first time each condition is met).
        self._episode_first_perfect: Optional[int] = None
        self._episode_first_saturation: Optional[int] = None
        self._episode_first_steps_plateau: Optional[int] = None
        self._saturation_reason: Optional[str] = None

        # For win-rate plateau detection:
        # We store the best rolling win-rate we've seen, and a "no improvement streak".
        self._best_win_rate: float = -1.0
        self._no_improve_windows: int = 0

        # For steps plateau detection:
        # We store the best (lowest) avg steps-to-goal, and a "no improvement streak".
        self._best_avg_steps: Optional[float] = None
        self._no_improve_steps_windows: int = 0

        # Last computed rolling metrics (cached after window fills).
        self._last_win_rate: float = 0.0
        self._last_avg_steps_win: Optional[float] = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _evict_if_full(self) -> None:
        """
        If the rolling window is full, evict the oldest episode and keep running totals in sync.
        """
        if len(self._wins) < self.window:
            return

        # poplink:
        # `popleft()` is a method of collections.deque.
        # Official docs:
        # https://docs.python.org/3/library/collections.html#collections.deque.popleft
        #
        # deque = "double-ended queue".
        # `popleft()` removes AND returns the element from the left side
        # (the oldest element in the rolling window).
        #
        # Beginner explanation:
        # ../../../docs/FOR_BEGINNERS.md#rolling-windows-deque-and-popleft

        old_win = self._wins.popleft()
        old_steps = self._steps_per_ep.popleft()

        self._wins_sum -= old_win
        self._steps_sum -= old_steps
        if old_win == 1:
            self._win_count -= 1

    def _append_episode(self, *, win: int, steps_if_win: int) -> None:
        """
        Append one episode into the rolling window and update running totals.

        steps_if_win:
        - steps if win==1
        - 0 if win==0
        """
        self._evict_if_full()

        self._wins.append(win)
        self._steps_per_ep.append(steps_if_win)

        self._wins_sum += win
        self._steps_sum += steps_if_win
        if win == 1:
            self._win_count += 1

    def _check_plateau(
        self,
        current: float,
        best: float,
        delta: float,
        streak: int,
        patience: int,
        lower_is_better: bool = False,
    ) -> tuple[float, int, bool]:
        """
        Generic plateau detector.

        Returns (updated_best, updated_streak, triggered) where:
        - updated_best   — new best value (or unchanged)
        - updated_streak — new no-improvement streak count
        - triggered      — True if patience was just exceeded

        lower_is_better=True flips the comparison for steps-to-goal tracking.
        """
        if lower_is_better:
            improved = current < (best - delta)
        else:
            improved = current > (best + delta)

        if improved:
            return current, 0, False

        new_streak = streak + 1
        return best, new_streak, (new_streak >= patience)

    def _compute_rolling_metrics(self) -> tuple[float, Optional[float]]:
        """
        Compute current rolling win-rate and avg steps-to-goal from running totals.
        Only called once the window is full.
        """
        win_rate = self._wins_sum / float(self.window)

        if self._win_count > 0:
            avg_steps_win = self._steps_sum / float(self._win_count)
        else:
            # No wins recorded in the window yet — avg steps undefined.
            avg_steps_win = None

        return win_rate, avg_steps_win

    def _check_milestones(self, win_rate: float, avg_steps_win: Optional[float]) -> None:
        """
        Evaluate all three convergence conditions and record first-hit episodes.

        Separated from update() so each condition is easy to read, test, and extend
        without touching data ingestion logic.
        """
        # -------------------------------------------------------------------
        # Condition 1: Perfect window (100% goal acquisition)
        # -------------------------------------------------------------------
        if self._episode_first_perfect is None and self._wins_sum == self.window:
            self._episode_first_perfect = self._episode_count

        # -------------------------------------------------------------------
        # Condition 2: Win-rate saturation (plateau)
        # -------------------------------------------------------------------
        if self._episode_first_saturation is None:
            self._best_win_rate, self._no_improve_windows, triggered = self._check_plateau(
                current=win_rate,
                best=self._best_win_rate,
                delta=self.delta,
                streak=self._no_improve_windows,
                patience=self.patience,
            )
            if triggered:
                self._episode_first_saturation = self._episode_count
                self._saturation_reason = "win_rate_plateau"

        # -------------------------------------------------------------------
        # Condition 3 (optional): Steps-to-goal plateau (efficiency saturation)
        # -------------------------------------------------------------------
        if avg_steps_win is not None and self._episode_first_steps_plateau is None:
            # Seed best on first observation.
            if self._best_avg_steps is None:
                self._best_avg_steps = avg_steps_win
                self._no_improve_steps_windows = 0
            else:
                self._best_avg_steps, self._no_improve_steps_windows, triggered = self._check_plateau(
                    current=avg_steps_win,
                    best=self._best_avg_steps,
                    delta=self.steps_delta,
                    streak=self._no_improve_steps_windows,
                    patience=self.steps_patience,
                    lower_is_better=True,
                )
                if triggered:
                    self._episode_first_steps_plateau = self._episode_count

                    # If win-rate saturation wasn't set yet, steps plateau counts too.
                    if self._episode_first_saturation is None:
                        self._episode_first_saturation = self._episode_count
                        self._saturation_reason = "steps_plateau"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, *, reached_goal: bool, steps: int) -> None:
        """
        Update tracker with the latest episode outcome.

        reached_goal:
            True if the goal was reached (successful episode).
        steps:
            Steps taken in the episode (for efficiency tracking).
        """
        self._episode_count += 1

        win = 1 if reached_goal else 0
        steps_if_win = int(steps) if reached_goal else 0
        self._append_episode(win=win, steps_if_win=steps_if_win)

        # We only compute rolling signals once the window is full.
        if len(self._wins) < self.window:
            return

        win_rate, avg_steps_win = self._compute_rolling_metrics()
        self._last_win_rate = win_rate
        self._last_avg_steps_win = avg_steps_win

        self._check_milestones(win_rate, avg_steps_win)

    def summary(self) -> ConvergenceSummary:
        """
        Return a summary snapshot (safe to log/print/serialize).

        NOTE:
        - last_window_* metrics are only meaningful once the window is full.
        - If window has not filled yet, these will reflect partial state.
        """
        return ConvergenceSummary(
            window=self.window,
            delta=self.delta,
            patience=self.patience,
            last_window_win_rate=float(self._last_win_rate),
            last_window_avg_steps_win=self._last_avg_steps_win,
            episode_first_perfect=self._episode_first_perfect,
            episode_first_saturation=self._episode_first_saturation,
            episode_first_steps_plateau=self._episode_first_steps_plateau,
            saturation_reason=self._saturation_reason,
        )
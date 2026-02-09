"""
gui_main.py

Minimal Tkinter GUI to run training and display results.

Copyright (c) 2026 Michael Garcia, M&E Design
https://mandedesign.studio
michael@mandedesign.studio

CSC370 Spring 2026
"""

from __future__ import annotations

import numpy as np
# Reserved for future visualization work.
# import matplotlib.pyplot as plt
import tkinter as tk

# Uses the package imports from __init__ packages.
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import ttk, messagebox
from rc_agents.envs import GridEnv, GridConfig
from rc_agents.edge_ai.rcg_edge.agents import QAgent, QConfig
from rc_agents.edge_ai.rcg_edge.runners import run_training
from rc_agents.config import TrainingUIConfig


# TODO: Add a button to visualize the policy grid as arrows on top of the value grid.
# TODO: Add a button to save the trained agent's Q-table to disk.
# TODO: Add a button to load a pre-trained agent from disk.

# Trainer visual interface
# This GUI is a thin wrapper around the core training loop.
# In order, this class:
# 1. Builds a Tk window with training parameter inputs and a results panel.
# 2. Reads user inputs and converts them into a TrainingUIConfig instance.
# 3. Creates a GridEnv and QAgent from that config.
# 4. Runs the shared training loop (run_training).
# 5. Displays results in the text box and opens a separate matplotlib heatmap window
#    showing max_a Q(s,a) per grid cell (a quick sanity-check visualization).
class TrainerGUI:
    def _show_q_heatmap(self, env: GridEnv, agent: QAgent) -> None:
        """
        Visualize the learned Q-values as a grid.
        Each cell shows max_a Q(s,a) for that state.
        Unvisited states will display as NaN (blank/neutral).
        """
        rows = env.config.rows
        cols = env.config.cols

        # Build a grid of max Q-values per state.
        # Preallocate a grid matching the environment dimensions.
        # Initialize with NaN ("Not a Number") so unvisited states render as blank/neutral
        # in the heatmap (avoids implying confidence where no learning has occurred).
        # Force float dtype (data type descriptor) so NaN is supported and learned Q-values
        # retain fractional precision for correct matplotlib color scaling and allow
        # non-integer values.
        #
        # NOTE on NaN and dtype:
        # - "Not a Number" (NaN) represents a value that is undefined or meaningless
        #   as a numeric quantity.
        # - Examples of NaN include: 0.0 / 0.0, sqrt(-1) in the real number system,
        #   np.nan, or float("nan").
        # - NaN is a special floating-point value defined by the IEEE 754 standard.
        # - NaN cannot be reliably compared using standard equality operators
        #   (e.g., NaN == NaN is False).
        # - Using a float dtype is required for NaN support and ensures proper
        #   visualization of learned Q-values.
        # - dtype explicitly defines the in-memory numeric type of each array element;
        #   dtype=float enforces IEEE-754 floating-point semantics, allowing NaN
        #   and fractional Q-values to be represented and visualized correctly.
        q_values_grid = np.full((rows, cols), np.nan, dtype=float)

        for r in range(rows):
            for c in range(cols):
                s = (r, c)
                if s in agent.q_table:
                    # Heatmap value is max_a Q(s,a) for each state s (state-value proxy).
                    q_values_grid[r, c] = float(np.max(agent.q_table[s]))

        # Create a new Tk window for the plot
        win = tk.Toplevel(self.root)
        win.title("Learned Q-values (max over actions)")

        fig = Figure(figsize=(6, 6), dpi=100)
        ax = fig.add_subplot(111)

        im = ax.imshow(q_values_grid, cmap="coolwarm", interpolation="nearest")
        fig.colorbar(im, ax=ax, label="Q-value (max_a)")

        ax.set_title("Learned Q-values for Each State (max over actions)")
        ax.set_xticks(np.arange(cols))
        ax.set_yticks(np.arange(rows))
        ax.invert_yaxis()

        # Grid lines
        ax.set_xticks(np.arange(-0.5, cols, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, rows, 1), minor=True)
        ax.grid(which="minor", linestyle="-", linewidth=0.5)
        ax.tick_params(which="minor", bottom=False, left=False)

        # Annotate each cell
        for i in range(rows):
            for j in range(cols):
                v = q_values_grid[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", color="black")

        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    # Tkinter GUI setup
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("CSC370 Q-Learning Trainer")
        self.root.geometry("520x420")

        self.episodes_var = tk.StringVar(value="50")
        self.max_steps_var = tk.StringVar(value="200")
        self.epsilon_var = tk.StringVar(value="0.10")
        self.alpha_var = tk.StringVar(value="0.50")
        self.gamma_var = tk.StringVar(value="0.90")

        self._build()

    # Builds the main window widgets
    def _build(self) -> None:
        frm = ttk.Frame(self.root, padding=12)
        frm.pack(fill="both", expand=True)

        ttk.Label(frm, text="Training Settings", font=("Segoe UI", 14, "bold")).pack(anchor="w")

        grid = ttk.Frame(frm)
        grid.pack(fill="x", pady=(10, 10))

        def row(label: str, var: tk.StringVar):
            r = ttk.Frame(grid)
            r.pack(fill="x", pady=4)
            ttk.Label(r, text=label, width=14).pack(side="left")
            ttk.Entry(r, textvariable=var, width=12).pack(side="left")

        row("Episodes", self.episodes_var)
        row("Max steps", self.max_steps_var)
        row("Epsilon", self.epsilon_var)
        row("Alpha", self.alpha_var)
        row("Gamma", self.gamma_var)

        ttk.Button(frm, text="Run Training", command=self.run_training_clicked).pack(pady=10)

        ttk.Label(frm, text="Results", font=("Segoe UI", 12, "bold")).pack(anchor="w", pady=(10, 0))
        self.results_box = tk.Text(frm, height=12, width=60)
        self.results_box.pack(fill="both", expand=True, pady=(6, 0))

    # Event handler for "Run Training" button
    def run_training_clicked(self) -> None:
        try:
            episodes = int(self.episodes_var.get())
            max_steps = int(self.max_steps_var.get())
            epsilon = float(self.epsilon_var.get())
            alpha = float(self.alpha_var.get())
            gamma = float(self.gamma_var.get())
        #error handling
        except ValueError:
            messagebox.showerror("Input error", "Please enter valid numeric values.")
            return

        #pull values from config file
        # NOTE: Tk GUI currently uses a fixed 5x5 grid to keep the interface minimal.
        # Streamlit UI supports variable grid sizing.
        cfg = TrainingUIConfig(
            episodes=episodes,
            max_steps=max_steps,
            alpha=alpha,
            gamma=gamma,
            epsilon=epsilon,
            rows=5,
            cols=5,
            start=(0, 0),
            goal=(4, 4),
            seed=123,
        )

        env = GridEnv(cfg.to_grid_config())
        agent = QAgent(cfg.to_q_config(), seed=cfg.seed)

        results = run_training(env=env, agent=agent, cfg=cfg)

        self._show_q_heatmap(env, agent)

        wins = sum(1 for r in results if r.reached_goal)
        avg_steps = sum(r.steps for r in results) / len(results)
        avg_reward = sum(r.total_reward for r in results) / len(results)

        self.results_box.delete("1.0", tk.END)
        self.results_box.insert(tk.END, f"Episodes: {len(results)}\n")
        self.results_box.insert(tk.END, f"Reached goal: {wins}/{len(results)}\n")
        self.results_box.insert(tk.END, f"Avg steps: {avg_steps:.2f}\n")
        self.results_box.insert(tk.END, f"Avg total reward: {avg_reward:.2f}\n\n")

        # Show last few episodes as a quick review
        self.results_box.insert(tk.END, "Last 10 episodes:\n")
        for r in results[-10:]:
            self.results_box.insert(
                tk.END,
                f"  ep {r.episode:>3}: steps={r.steps:>3} reward={r.total_reward:>6.1f} goal={r.reached_goal}\n",
            )

    # Starts the GUI event loop
    def run(self) -> None:
        self.root.mainloop()

# Small test to make sure everything imports correctly.
if __name__ == "__main__":
    TrainerGUI().run()
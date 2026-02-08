"""
gui_main.py

Minimal Tkinter GUI to run training and display results.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import ttk, messagebox
from rc_agents.envs.grid_env import GridEnv, GridConfig
from rc_agents.edge_ai.rcg_edge.agents.q_agent import QAgent, QConfig
from rc_agents.edge_ai.rcg_edge.runners.train_runner import run_training
from rc_agents.config.ui_config import TrainingUIConfig

class TrainerGUI:
    def _show_q_heatmap(self, env: GridEnv, agent: QAgent) -> None:
        """
        Visualize the learned Q-values as a grid.
        Each cell shows max_a Q(s,a) for that state.
        Unvisited states will display as NaN (blank/neutral).
        """
        rows = env.config.rows
        cols = env.config.cols

        # Build a grid of max Q-values per state
        q_values_grid = np.full((rows, cols), np.nan, dtype=float)

        for r in range(rows):
            for c in range(cols):
                s = (r, c)
                if s in agent.q_table:
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

        # Optional: grid lines
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

    def run_training_clicked(self) -> None:
        try:
            episodes = int(self.episodes_var.get())
            max_steps = int(self.max_steps_var.get())
            epsilon = float(self.epsilon_var.get())
            alpha = float(self.alpha_var.get())
            gamma = float(self.gamma_var.get())
        except ValueError:
            messagebox.showerror("Input error", "Please enter valid numeric values.")
            return

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

        # Show last few episodes as a quick sanity view
        self.results_box.insert(tk.END, "Last 10 episodes:\n")
        for r in results[-10:]:
            self.results_box.insert(
                tk.END,
                f"  ep {r.episode:>3}: steps={r.steps:>3} reward={r.total_reward:>6.1f} goal={r.reached_goal}\n",
            )

    def run(self) -> None:
        self.root.mainloop()


if __name__ == "__main__":
    TrainerGUI().run()
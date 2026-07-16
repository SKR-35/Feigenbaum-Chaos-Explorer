"""Tkinter application for exploring Feigenbaum universality."""
from __future__ import annotations

import os
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import numpy as np

from .maps import MAPS, MapSpec
from .numerics import (
    FEIGENBAUM_ALPHA,
    FEIGENBAUM_DELTA,
    bifurcation_points,
    iterate_orbit,
    lyapunov_curve,
    scaling_estimates,
)

MODES = [
    "Bifurcation Diagram",
    "Lyapunov Spectrum",
    "Orbit & Cobweb",
    "Delta Convergence",
    "Alpha Scaling",
    "Universality Comparison",
]

PRESETS = {
    "Full range": None,
    "Period doubling": (3.0, 3.58),
    "Edge of chaos": (3.54, 3.58),
    "Period-3 window": (3.82, 3.86),
}


class FeigenbaumApp:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("Feigenbaum Chaos Explorer")
        self.root.geometry("1320x780")
        self.root.minsize(1000, 650)

        self.map_var = tk.StringVar(value="Logistic map")
        self.mode_var = tk.StringVar(value=MODES[0])
        self.preset_var = tk.StringVar(value="Full range")
        self.p_min_var = tk.DoubleVar(value=2.8)
        self.p_max_var = tk.DoubleVar(value=4.0)
        self.x0_var = tk.DoubleVar(value=0.2)
        self.parameter_var = tk.DoubleVar(value=3.5699456)
        self.burn_var = tk.IntVar(value=900)
        self.samples_var = tk.IntVar(value=220)
        self.levels_var = tk.IntVar(value=7)
        self.cmap_var = tk.StringVar(value="magma")
        self.status_var = tk.StringVar(value="Ready")

        self._build_ui()
        self.render()

    @property
    def spec(self) -> MapSpec:
        return MAPS[self.map_var.get()]

    def _build_ui(self) -> None:
        top = ttk.Frame(self.root, padding=6)
        top.pack(fill=tk.X)

        ttk.Label(top, text="View:").pack(side=tk.LEFT)
        ttk.Combobox(top, textvariable=self.mode_var, values=MODES, state="readonly", width=24).pack(side=tk.LEFT, padx=(4, 10))
        ttk.Label(top, text="Map:").pack(side=tk.LEFT)
        map_box = ttk.Combobox(top, textvariable=self.map_var, values=list(MAPS), state="readonly", width=17)
        map_box.pack(side=tk.LEFT, padx=(4, 10))
        map_box.bind("<<ComboboxSelected>>", lambda _e: self._on_map_change())

        ttk.Label(top, text="Preset:").pack(side=tk.LEFT)
        preset_box = ttk.Combobox(top, textvariable=self.preset_var, values=list(PRESETS), state="readonly", width=17)
        preset_box.pack(side=tk.LEFT, padx=(4, 10))
        preset_box.bind("<<ComboboxSelected>>", lambda _e: self._apply_preset())

        ttk.Button(top, text="Render", command=self.render).pack(side=tk.LEFT, padx=3)
        ttk.Button(top, text="Reset", command=self.reset).pack(side=tk.LEFT, padx=3)
        ttk.Button(top, text="Save PNG", command=self.save_png).pack(side=tk.LEFT, padx=3)

        ttk.Separator(top, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=7)
        ttk.Button(top, text="Zoom In (Z)", command=lambda: self._keyboard_zoom(1 / 1.25)).pack(side=tk.LEFT, padx=3)
        ttk.Button(top, text="Zoom Out (X)", command=lambda: self._keyboard_zoom(1.25)).pack(side=tk.LEFT, padx=3)
        ttk.Button(top, text="Home", command=self._home_view).pack(side=tk.LEFT, padx=3)

        body = ttk.Panedwindow(self.root, orient=tk.HORIZONTAL)
        body.pack(fill=tk.BOTH, expand=True)

        controls = ttk.Frame(body, padding=10)
        body.add(controls, weight=0)
        plot_frame = ttk.Frame(body)
        body.add(plot_frame, weight=1)

        row = 0
        ttk.Label(controls, text="Parameters", font=("TkDefaultFont", 11, "bold")).grid(row=row, column=0, columnspan=2, sticky="w", pady=(0, 8))
        row += 1
        for label, variable in [
            ("Parameter min", self.p_min_var),
            ("Parameter max", self.p_max_var),
            ("Selected parameter", self.parameter_var),
            ("Initial x₀", self.x0_var),
            ("Burn-in", self.burn_var),
            ("Samples", self.samples_var),
            ("Scaling levels", self.levels_var),
        ]:
            ttk.Label(controls, text=label).grid(row=row, column=0, sticky="w", pady=3)
            ttk.Entry(controls, textvariable=variable, width=14).grid(row=row, column=1, sticky="ew", pady=3)
            row += 1

        ttk.Label(controls, text="Colormap").grid(row=row, column=0, sticky="w", pady=3)
        ttk.Combobox(
            controls,
            textvariable=self.cmap_var,
            values=["magma", "viridis", "plasma", "inferno", "cividis", "turbo"],
            state="readonly",
            width=12,
        ).grid(row=row, column=1, sticky="ew", pady=3)
        row += 1

        ttk.Separator(controls).grid(row=row, column=0, columnspan=2, sticky="ew", pady=12)
        row += 1
        info = (
            "Zoom: mouse wheel or Z / X keys.\n"
            "Toolbar controls can zoom, pan, and restore Home.\n\n"
            "Delta measures parameter-space compression.\n"
            "Alpha measures state-space scaling near the critical point."
        )
        ttk.Label(controls, text=info, wraplength=220, justify=tk.LEFT).grid(row=row, column=0, columnspan=2, sticky="nw")
        controls.columnconfigure(1, weight=1)

        self.figure = Figure(figsize=(10, 7), dpi=100, constrained_layout=True)
        self.canvas = FigureCanvasTkAgg(self.figure, master=plot_frame)
        self.canvas.mpl_connect("scroll_event", self._on_scroll_zoom)

        # Keep the toolbar above the canvas; bottom controls can be clipped on
        # shorter desktop screens.
        self.toolbar = NavigationToolbar2Tk(self.canvas, plot_frame, pack_toolbar=False)
        self.toolbar.update()
        self.toolbar.pack(side=tk.TOP, fill=tk.X)
        canvas_widget = self.canvas.get_tk_widget()
        canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        canvas_widget.bind("<Button-1>", lambda _e: canvas_widget.focus_set(), add="+")

        ttk.Label(self.root, textvariable=self.status_var, anchor="w", padding=(6, 3)).pack(fill=tk.X)
        self.root.bind("<Return>", lambda _e: self.render())
        self.root.bind("<Control-s>", lambda _e: self.save_png())
        # bind_all makes the shortcuts reliable even after a Combobox, Entry,
        # toolbar button, or Matplotlib canvas has received keyboard focus.
        self.root.bind_all("<KeyPress-z>", self._zoom_in_key, add="+")
        self.root.bind_all("<KeyPress-Z>", self._zoom_in_key, add="+")
        self.root.bind_all("<KeyPress-x>", self._zoom_out_key, add="+")
        self.root.bind_all("<KeyPress-X>", self._zoom_out_key, add="+")
        self.root.bind("<Escape>", lambda _e: self.root.destroy())

    def _zoom_in_key(self, _event=None) -> str:
        self._keyboard_zoom(1 / 1.25)
        return "break"

    def _zoom_out_key(self, _event=None) -> str:
        self._keyboard_zoom(1.25)
        return "break"

    def _home_view(self) -> None:
        """Restore the full limits for the current view and parameters."""
        self.render()
        self.status_var.set("Home view restored | Z=in, X=out")

    def _keyboard_zoom(self, scale_factor: float) -> None:
        """Zoom every visible subplot around its current center."""
        axes = [ax for ax in self.figure.axes if ax.get_visible()]
        if not axes:
            return

        for ax in axes:
            x_left, x_right = ax.get_xlim()
            y_bottom, y_top = ax.get_ylim()
            x_center = (x_left + x_right) / 2.0
            y_center = (y_bottom + y_top) / 2.0
            half_width = (x_right - x_left) * scale_factor / 2.0
            half_height = (y_top - y_bottom) * scale_factor / 2.0
            ax.set_xlim(x_center - half_width, x_center + half_width)
            ax.set_ylim(y_center - half_height, y_center + half_height)

        self.canvas.draw_idle()
        action = "Zoom in" if scale_factor < 1.0 else "Zoom out"
        self.status_var.set(f"{action} | Z=in, X=out | toolbar Home restores full view")

    def _on_scroll_zoom(self, event) -> None:
        """Zoom the subplot under the cursor while keeping the cursor anchored."""
        ax = event.inaxes
        if ax is None or event.xdata is None or event.ydata is None:
            return

        # Matplotlib reports wheel-up as ``up`` and wheel-down as ``down``.
        scale_factor = 1 / 1.25 if event.button == "up" else 1.25
        x_left, x_right = ax.get_xlim()
        y_bottom, y_top = ax.get_ylim()

        x = float(event.xdata)
        y = float(event.ydata)
        new_width = (x_right - x_left) * scale_factor
        new_height = (y_top - y_bottom) * scale_factor

        rel_x = (x_right - x) / (x_right - x_left)
        rel_y = (y_top - y) / (y_top - y_bottom)
        ax.set_xlim(x - new_width * (1 - rel_x), x + new_width * rel_x)
        ax.set_ylim(y - new_height * (1 - rel_y), y + new_height * rel_y)
        self.canvas.draw_idle()
        self.status_var.set("Mouse-wheel zoom | use toolbar Home to restore the full view")

    def _on_map_change(self) -> None:
        lo, hi = self.spec.default_range
        self.p_min_var.set(lo)
        self.p_max_var.set(hi)
        self.parameter_var.set((lo + hi) / 2)
        self.preset_var.set("Full range")
        self.render()

    def _apply_preset(self) -> None:
        preset = PRESETS[self.preset_var.get()]
        if preset is None or self.spec.name != "Logistic map":
            lo, hi = self.spec.default_range
        else:
            lo, hi = preset
        self.p_min_var.set(lo)
        self.p_max_var.set(hi)
        self.parameter_var.set((lo + hi) / 2)
        self.render()

    def reset(self) -> None:
        self.map_var.set("Logistic map")
        self.mode_var.set(MODES[0])
        self.preset_var.set("Full range")
        self.p_min_var.set(2.8)
        self.p_max_var.set(4.0)
        self.parameter_var.set(3.5699456)
        self.x0_var.set(0.2)
        self.burn_var.set(900)
        self.samples_var.set(220)
        self.levels_var.set(7)
        self.cmap_var.set("magma")
        self.render()

    def render(self) -> None:
        started = time.perf_counter()
        try:
            self.figure.clear()
            mode = self.mode_var.get()
            if mode == "Bifurcation Diagram":
                self._plot_bifurcation()
            elif mode == "Lyapunov Spectrum":
                self._plot_lyapunov()
            elif mode == "Orbit & Cobweb":
                self._plot_orbit_cobweb()
            elif mode == "Delta Convergence":
                self._plot_delta()
            elif mode == "Alpha Scaling":
                self._plot_alpha()
            else:
                self._plot_universality()
            self.canvas.draw_idle()
            elapsed = time.perf_counter() - started
            self.status_var.set(f"{mode} | {self.spec.name} | render={elapsed:.2f}s")
        except Exception as exc:
            self.status_var.set(f"Render failed: {exc}")
            messagebox.showerror("Render failed", str(exc))

    def _plot_bifurcation(self) -> None:
        ax = self.figure.add_subplot(111)
        p, x = bifurcation_points(
            self.spec,
            self.p_min_var.get(),
            self.p_max_var.get(),
            width=1300,
            burn_in=max(50, self.burn_var.get()),
            samples=max(20, self.samples_var.get()),
        )
        ax.hexbin(p, x, gridsize=(900, 500), bins="log", mincnt=1, cmap=self.cmap_var.get(), linewidths=0)
        ax.set_title(f"{self.spec.name} — Bifurcation Diagram")
        ax.set_xlabel(self.spec.parameter_symbol)
        ax.set_ylabel("Long-run orbit value")
        ax.grid(alpha=0.15)

    def _plot_lyapunov(self) -> None:
        ax = self.figure.add_subplot(111)
        p, lam = lyapunov_curve(self.spec, self.p_min_var.get(), self.p_max_var.get())
        ax.plot(p, lam, linewidth=0.9)
        ax.axhline(0.0, linewidth=1.0, linestyle="--")
        ax.fill_between(p, 0, lam, where=lam > 0, alpha=0.25)
        ax.set_title(f"{self.spec.name} — Lyapunov Spectrum")
        ax.set_xlabel(self.spec.parameter_symbol)
        ax.set_ylabel("λ")
        ax.grid(alpha=0.25)

    def _plot_orbit_cobweb(self) -> None:
        ax1, ax2 = self.figure.subplots(1, 2)
        parameter = self.parameter_var.get()
        orbit = iterate_orbit(self.spec, parameter, self.x0_var.get(), 160)
        ax1.plot(np.arange(len(orbit)), orbit, marker=".", markersize=2, linewidth=0.8)
        ax1.set_title("Orbit time series")
        ax1.set_xlabel("Iteration")
        ax1.set_ylabel("xₙ")
        ax1.grid(alpha=0.25)

        if self.spec.name == "Quadratic map":
            lo, hi = -1.1, 1.1
        else:
            lo, hi = 0.0, 1.0
        xs = np.linspace(lo, hi, 1000)
        ax2.plot(xs, self.spec.function(xs, parameter), label="f(x)")
        ax2.plot(xs, xs, linestyle="--", label="y=x")
        cob = iterate_orbit(self.spec, parameter, self.x0_var.get(), 70)
        for a, b in zip(cob[:-1], cob[1:]):
            ax2.plot([a, a], [a, b], linewidth=0.55)
            ax2.plot([a, b], [b, b], linewidth=0.55)
        ax2.set_xlim(lo, hi)
        ax2.set_ylim(lo, hi)
        ax2.set_title(f"Cobweb at {self.spec.parameter_symbol}={parameter:.8g}")
        ax2.legend()
        ax2.grid(alpha=0.25)

    def _plot_delta(self) -> None:
        rows = scaling_estimates(self.spec, max(3, min(8, self.levels_var.get())))
        ax1, ax2 = self.figure.subplots(1, 2)
        valid = [r for r in rows if r.delta is not None]
        levels = [r.level for r in valid]
        values = [r.delta for r in valid]
        ax1.plot(levels, values, marker="o")
        ax1.axhline(FEIGENBAUM_DELTA, linestyle="--", label=f"δ={FEIGENBAUM_DELTA:.10f}")
        ax1.set_title("Feigenbaum δ convergence")
        ax1.set_xlabel("Level n")
        ax1.set_ylabel("δₙ")
        ax1.legend()
        ax1.grid(alpha=0.25)

        ax2.axis("off")
        table_data = [[r.level, r.period, f"{r.parameter:.12f}", "—" if r.delta is None else f"{r.delta:.9f}"] for r in rows]
        table = ax2.table(cellText=table_data, colLabels=["n", "Period", "Superstable parameter", "δₙ"], loc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.4)
        ax2.set_title("Period-doubling measurements", pad=20)

    def _plot_alpha(self) -> None:
        rows = scaling_estimates(self.spec, max(3, min(8, self.levels_var.get())))
        ax1, ax2 = self.figure.subplots(1, 2)
        valid = [r for r in rows if r.alpha is not None]
        ax1.plot([r.level for r in valid], [r.alpha for r in valid], marker="o")
        ax1.axhline(FEIGENBAUM_ALPHA, linestyle="--", label=f"α={FEIGENBAUM_ALPHA:.10f}")
        ax1.set_title("Feigenbaum α scaling")
        ax1.set_xlabel("Level n")
        ax1.set_ylabel("αₙ")
        ax1.legend()
        ax1.grid(alpha=0.25)

        ax2.semilogy([r.level for r in rows], [r.distance for r in rows], marker="o")
        ax2.set_title("Nearest critical-orbit distance")
        ax2.set_xlabel("Level n")
        ax2.set_ylabel("dₙ")
        ax2.grid(alpha=0.25)

    def _plot_universality(self) -> None:
        ax = self.figure.add_subplot(111)
        for name, spec in MAPS.items():
            rows = scaling_estimates(spec, max(4, min(7, self.levels_var.get())))
            valid = [r for r in rows if r.delta is not None]
            ax.plot([r.level for r in valid], [r.delta for r in valid], marker="o", label=name)
        ax.axhline(FEIGENBAUM_DELTA, linestyle="--", label="Universal δ")
        ax.set_title("Universality across nonlinear maps")
        ax.set_xlabel("Level n")
        ax.set_ylabel("δₙ")
        ax.legend()
        ax.grid(alpha=0.25)

    def save_png(self) -> None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        mode = self.mode_var.get().lower().replace(" ", "_").replace("&", "and")
        path = filedialog.asksaveasfilename(
            title="Save current figure",
            defaultextension=".png",
            filetypes=[("PNG image", "*.png")],
            initialfile=f"feigenbaum_{mode}_{timestamp}.png",
        )
        if not path:
            return
        self.figure.savefig(path, dpi=220, bbox_inches="tight")
        self.status_var.set(f"Saved: {os.path.abspath(path)}")

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    FeigenbaumApp().run()


if __name__ == "__main__":
    main()

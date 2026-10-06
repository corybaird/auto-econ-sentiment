"""Shared styling and output for every paper figure."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.research_paper.config import PaperConfig

logger = logging.getLogger(__name__)

FAMILY_KEY = "Solid: lexical    Dashed: transformer"


class FigureStyle:
    """Fonts, colors, reference lines and saving, taken from the ``figure`` config section."""

    def __init__(self, config: PaperConfig) -> None:
        self.figure = config["figure"]
        self.directory = config.path("figure_dir")
        self.directory.mkdir(parents=True, exist_ok=True)
        self.apply_theme()

    def apply_theme(self) -> None:
        font = self.figure["font"]
        sns.set_theme(style="white", context="paper")
        plt.rcParams.update({
            "axes.titlesize": font["subplot_title"],
            "axes.labelsize": font["axis_label"],
            "xtick.labelsize": font["tick_label"],
            "ytick.labelsize": font["tick_label"],
            "legend.fontsize": font["legend"],
            "figure.titlesize": font["title"],
        })

    def measure_color(self, column: str) -> str:
        return self.figure["measure_colors"][column]

    def family_linestyle(self, family: str) -> str:
        return self.figure["family_linestyles"][family]

    def grid(self, ax: plt.Axes, axis: str = "y") -> None:
        ax.grid(True, axis=axis, color=self.figure["grid_color"], linewidth=0.6, zorder=0)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    def zero_line(self, ax: plt.Axes, vertical: bool = False) -> None:
        line = ax.axvline if vertical else ax.axhline
        line(0, color=self.figure["reference_line_color"], linewidth=0.8, linestyle="--")

    def recessions(self, ax: plt.Axes) -> None:
        for band in self.figure["recessions"]:
            ax.axvspan(pd.Timestamp(band["start"]), pd.Timestamp(band["end"]), color="#c8c8c8", alpha=0.35, zorder=0)

    def time_axis(self, ax: plt.Axes, index: pd.DatetimeIndex) -> None:
        """Recession bands, grid, zero line and x limits for a time-series panel."""
        self.recessions(ax)
        self.grid(ax)
        self.zero_line(ax)
        ax.set_xlim(index.min(), index.max())
        ax.set_xlabel("")

    def family_legend(self, fig: plt.Figure, handles: list, bottom: float, top: float, ncol: int = 5, **adjust) -> None:
        """One legend below the figure, keyed by line style to the method family."""
        fig.legend(handles, [handle.get_label() for handle in handles], loc="lower center", ncol=ncol, frameon=True, bbox_to_anchor=(0.5, 0.0), title=FAMILY_KEY)
        fig.subplots_adjust(bottom=bottom, top=top, **adjust)

    def save(self, fig: plt.Figure, filename: str, tight: bool = True) -> Path:
        path = self.directory / filename
        if tight:
            fig.tight_layout()
        fig.savefig(path, dpi=self.figure["dpi"], bbox_inches="tight")
        plt.close(fig)
        logger.info("Wrote %s", path)
        return path

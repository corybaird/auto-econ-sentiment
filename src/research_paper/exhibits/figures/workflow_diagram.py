"""Figure 1: the package workflow as a vector flowchart."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from src.research_paper.config import PaperConfig
from src.research_paper.exhibits.style import FigureStyle


class WorkflowDiagram:
    """Figure 1: the package workflow as a vector flowchart."""

    # name: (x centre, y centre, width, title, subtitle)
    BOXES = {
        "config": (0.50, 0.90, 0.38, "YAML / Python API", "inputs, cleaning rules,\ndictionaries and models"),
        "load": (0.50, 0.69, 0.32, "1. Load text", "CSV, Excel, Parquet,\ntext or Markdown files"),
        "clean": (0.50, 0.48, 0.32, "2. Clean text", "normalize, tokenize, stem"),
        "lexical": (0.22, 0.27, 0.40, "3a. Lexical sentiment", "six dictionaries;\nPosNeg and All-Words"),
        "transformer": (0.78, 0.27, 0.40, "3b. Transformer sentiment", "sentence classification;\nPosNeg and All-Sentences"),
        "export": (0.50, 0.06, 0.44, "4. Export results", "document scores, matched words,\nsentence probabilities"),
    }
    HEIGHT, BAND = 0.15, 0.055
    DARK, LIGHT, EDGE = "#5f5f5f", "#f2f2f2", "#4a4a4a"

    def __init__(self, config: PaperConfig) -> None:
        self.style = FigureStyle(config)
        self.arrow = {"arrowstyle": "-|>", "color": self.EDGE, "linewidth": 1.3, "mutation_scale": 12}

    def run(self) -> list[Path]:
        fig, ax = plt.subplots(figsize=(7.2, 5.4))
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.03, 1.0)
        ax.axis("off")
        for box in self.BOXES.values():
            self._draw_box(ax, *box)
        self._draw_arrows(ax)
        return [self.style.save(fig, "pipeline_architecture.pdf", tight=False)]

    def _draw_box(self, ax: plt.Axes, x: float, y: float, width: float, title: str, subtitle: str) -> None:
        left, bottom = x - width / 2, y - self.HEIGHT / 2
        rounded = "round,pad=0,rounding_size=0.012"
        ax.add_patch(FancyBboxPatch((left, bottom), width, self.HEIGHT, boxstyle=rounded, facecolor=self.LIGHT, edgecolor=self.EDGE, linewidth=1.0))
        ax.add_patch(FancyBboxPatch((left, bottom + self.HEIGHT - self.BAND), width, self.BAND, boxstyle=rounded, facecolor=self.DARK, edgecolor=self.EDGE, linewidth=1.0))
        ax.text(x, bottom + self.HEIGHT - self.BAND / 2, title, ha="center", va="center", color="white", fontsize=10, fontweight="bold")
        ax.text(x, bottom + (self.HEIGHT - self.BAND) / 2, subtitle, ha="center", va="center", color="#222222", fontsize=8.5, linespacing=1.3)

    def _draw_arrows(self, ax: plt.Axes) -> None:
        for start, end in (("config", "load"), ("load", "clean")):
            ax.annotate("", xy=self._top(end), xytext=self._bottom(start), arrowprops=self.arrow)
        export_x, export_y, export_width = self.BOXES["export"][:3]
        for branch, side in (("lexical", -1), ("transformer", 1)):
            ax.annotate("", xy=self._top(branch), xytext=self._bottom("clean"), arrowprops={**self.arrow, "connectionstyle": "angle,angleA=0,angleB=90,rad=0"})
            ax.annotate("", xy=(export_x + side * export_width / 2, export_y), xytext=self._bottom(branch), arrowprops={**self.arrow, "connectionstyle": "angle,angleA=-90,angleB=0,rad=0"})

    def _top(self, name: str) -> tuple[float, float]:
        return self.BOXES[name][0], self.BOXES[name][1] + self.HEIGHT / 2

    def _bottom(self, name: str) -> tuple[float, float]:
        return self.BOXES[name][0], self.BOXES[name][1] - self.HEIGHT / 2

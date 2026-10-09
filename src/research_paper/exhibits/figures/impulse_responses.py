"""Figure 7: impulse responses to a sentiment shock, one VAR per measure."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.research_paper.analysis import LEXICAL, TRANSFORMER
from src.research_paper.config import PaperConfig
from src.research_paper.exhibits.style import FigureStyle
from src.research_paper.econometrics import ImpulseResponse


# Points added to the shared label, tick and legend sizes; the panels are small in print.
_FONT_INCREASE = 4


class ImpulseResponseFigure:
    """Figure 7: impulse responses to a sentiment shock, one VAR per measure."""

    def __init__(self, config: PaperConfig, responses: dict[str, ImpulseResponse]) -> None:
        self.var = config["var"]
        self.transformer_measures = config["measures"]["transformer"]
        self.style = FigureStyle(config)
        self.font = self.style.figure["font"]
        self.responses = responses

    def run(self) -> list[Path]:
        return [self.impulse_responses()]

    def impulse_responses(self) -> Path:
        variables = self.var["variables"]
        n_cols = 3
        n_rows = int(np.ceil((len(variables) - 1) / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6.0 * n_cols, 4.2 * n_rows), sharex=True)
        axes = np.atleast_1d(axes).ravel()
        for ax, variable in zip(axes, variables[1:]):
            self._plot_responses(ax, variables.index(variable))
            ax.set_title(self.var["response_labels"][variable], loc="left", fontsize=self.font["subplot_title"] + _FONT_INCREASE)
        for row in range(n_rows):
            axes[row * n_cols].set_ylabel("Response (SD)", fontsize=self.font["axis_label"] + _FONT_INCREASE)
        self._legend_in_spare_panel(fig, axes, n_panels=len(variables) - 1)
        fig.suptitle("VAR impulse responses to a one standard deviation sentiment shock", fontsize=self.font["title"] + _FONT_INCREASE)
        fig.subplots_adjust(bottom=0.08, top=0.89, hspace=0.42, wspace=0.22)
        return self.style.save(fig, "var_irf_usa.pdf", tight=False)

    def _plot_responses(self, ax: plt.Axes, variable: int) -> None:
        for measure, label in self.var["measures"].items():
            response = self.responses[measure]
            horizon = np.arange(response.point.shape[0])
            color = self.var["measure_colors"][measure]
            family = TRANSFORMER if measure in self.transformer_measures else LEXICAL
            ax.plot(horizon, response.point[:, variable], label=label, color=color, linestyle=self.style.family_linestyle(family), linewidth=2.0)
            ax.fill_between(horizon, response.lower[:, variable], response.upper[:, variable], color=color, alpha=0.10, linewidth=0)
        self.style.grid(ax)
        self.style.zero_line(ax)
        ax.set_xlabel("Months", fontsize=self.font["axis_label"] + _FONT_INCREASE)
        ax.set_xlim(0, self.var["irf_periods"])
        ax.tick_params(labelbottom=True, labelsize=self.font["tick_label"] + _FONT_INCREASE)

    def _legend_in_spare_panel(self, fig: plt.Figure, axes: np.ndarray, n_panels: int) -> None:
        handles, labels = axes[0].get_legend_handles_labels()
        spare = axes[n_panels:]
        for ax in spare:
            ax.set_visible(False)
        if len(spare):
            spare[0].set_visible(True)
            spare[0].axis("off")
            spare[0].legend(handles, labels, loc="center", frameon=True, fontsize=self.font["legend"] + _FONT_INCREASE + 2)
        else:
            fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=True, bbox_to_anchor=(0.5, 0.0))

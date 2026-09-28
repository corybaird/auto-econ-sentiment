"""Every figure in the paper, one class per figure or figure family."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import FancyBboxPatch

from src.research_paper.config import PaperConfig
from src.research_paper.econometrics import ImpulseResponse
from src.research_paper.exhibits.style import FigureStyle
from src.research_paper.measures import LEXICAL, TRANSFORMER, SentimentMeasures

FAMILIES = (LEXICAL, TRANSFORMER)


class MethodFigures:
    """Figures 3 to 6: how the measures compare across the 51-bank statement panel."""

    def __init__(self, config: PaperConfig, documents: pd.DataFrame) -> None:
        self.config = config
        self.figure = config["figure"]
        self.style = FigureStyle(config)
        self.documents = documents
        self.measures = SentimentMeasures(documents, config)
        self.n_banks = documents["Country"].nunique()

    def run(self) -> list[Path]:
        return [self.distributions(), self.correlations(), self.time_series(), self.country_comparison()]

    def distributions(self) -> Path:
        """Figure 3: the distribution of every measure, pooled across documents."""
        order = [self.measures.labels[column] for column in self.measures.columns]
        fig, ax = plt.subplots(figsize=(11, 6.5))
        sns.boxplot(data=self.measures.long_form(), x="value", y="Measure", hue="Family", order=order, palette=self.figure["family_colors"], dodge=False, fliersize=1.5, linewidth=0.9, ax=ax)
        self.style.grid(ax, axis="x")
        self.style.zero_line(ax, vertical=True)
        ax.set_xlabel("Net sentiment (positive share $-$ negative share)")
        ax.set_ylabel("")
        ax.legend(title="", loc="lower right", frameon=True)
        ax.set_title(f"Distribution of net sentiment by method, pooled across {self.n_banks} central banks")
        return self.style.save(fig, "method_distributions.pdf")

    def correlations(self) -> Path:
        """Figure 4: pooled document-level Pearson correlations, lexical block first."""
        correlation = self.measures.correlations()
        fig, ax = plt.subplots(figsize=(10, 8))
        mask = np.triu(np.ones_like(correlation, dtype=bool), k=1)
        sns.heatmap(correlation, mask=mask, cmap="RdBu_r", vmin=-1, vmax=1, center=0, annot=True, fmt=".2f", annot_kws={"size": 9}, linewidths=0.5, linecolor="white", cbar_kws={"shrink": 0.75, "label": "Pearson correlation"}, ax=ax)
        n_lexical = len(self.measures.columns_in(LEXICAL))
        ax.axhline(n_lexical, color="black", linewidth=1.4)
        ax.axvline(n_lexical, color="black", linewidth=1.4)
        ax.set_title(f"Document-level correlations across sentiment methods, pooled across {self.n_banks} central banks")
        plt.setp(ax.get_xticklabels(), rotation=40, ha="right")
        plt.setp(ax.get_yticklabels(), rotation=0)
        return self.style.save(fig, "method_correlation_levels.pdf")

    def time_series(self) -> Path:
        """Figure 5: rolling monthly means averaged across banks, one panel per family."""
        monthly = self._smoothed(self.measures.monthly_bank_average())
        fig, axes = plt.subplots(1, 2, figsize=(15, 6.5), sharey=True)
        for ax, family in zip(axes, FAMILIES):
            self._plot_family(ax, monthly, self.measures, family)
            ax.set_title(f"{family} methods", loc="left")
        axes[0].set_ylabel("Net sentiment")
        fig.suptitle(f"Net sentiment over time, averaged across {self.n_banks} central banks ({self.figure['rolling_window']}-month rolling mean)")
        self.style.family_legend(fig, self._measure_lines(axes, self.measures), bottom=0.24, top=0.88)
        return self.style.save(fig, "method_time_series.pdf", tight=False)

    def country_comparison(self) -> Path:
        """Figure 6: rolling monthly means for selected banks, one row per bank."""
        countries = self.figure["panel_countries"]
        fig, axes = plt.subplots(len(countries), len(FAMILIES), figsize=(15, 4.6 * len(countries)), sharey=True, sharex=True)
        axes = np.atleast_2d(axes)
        for row, (code, name) in zip(axes, countries.items()):
            bank = SentimentMeasures(self.documents[self.documents["Country"] == code], self.config)
            monthly = self._smoothed(bank.monthly())
            for ax, family in zip(row, FAMILIES):
                self._plot_family(ax, monthly, bank, family)
                ax.set_title(f"{name} — {family} methods", loc="left")
            row[0].set_ylabel("Net sentiment")
        fig.suptitle(f"Net sentiment by central bank and method family ({self.figure['rolling_window']}-month rolling mean)")
        self.style.family_legend(fig, self._measure_lines(axes[0], self.measures), bottom=0.16, top=0.92, hspace=0.28)
        return self.style.save(fig, "panel_country_comparison.pdf", tight=False)

    def _smoothed(self, monthly: pd.DataFrame) -> pd.DataFrame:
        """Rolling mean over ``figure.rolling_window`` months, from ``figure.time_series_start``."""
        rolled = monthly.rolling(self.figure["rolling_window"], min_periods=1).mean()
        return rolled.loc[rolled.index >= pd.Timestamp(self.figure["time_series_start"])]

    def _plot_family(self, ax: plt.Axes, monthly: pd.DataFrame, measures: SentimentMeasures, family: str) -> None:
        for column in measures.columns_in(family):
            ax.plot(monthly.index, monthly[column], label=measures.labels[column], color=self.style.measure_color(column), linestyle=self.style.family_linestyle(family), linewidth=1.5)
        self.style.time_axis(ax, monthly.index)

    @staticmethod
    def _measure_lines(axes, measures: SentimentMeasures) -> list:
        """The plotted measure lines, leaving out reference lines, for the shared legend."""
        return [line for ax in axes for line in ax.get_lines() if line.get_label() in measures.labels.values()]


class VarFigure:
    """Figure 7: impulse responses to a sentiment shock, one VAR per measure."""

    def __init__(self, config: PaperConfig, responses: dict[str, ImpulseResponse]) -> None:
        self.var = config["var"]
        self.transformer_measures = config["measures"]["transformer"]
        self.style = FigureStyle(config)
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
            ax.set_title(self.var["response_labels"][variable], loc="left")
        for row in range(n_rows):
            axes[row * n_cols].set_ylabel("Response (SD)")
        self._legend_in_spare_panel(fig, axes, n_panels=len(variables) - 1)
        fig.suptitle("VAR impulse responses to a one standard deviation sentiment shock, United States")
        fig.subplots_adjust(bottom=0.08, top=0.91, hspace=0.28, wspace=0.22)
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
        ax.set_xlabel("Months")
        ax.set_xlim(0, self.var["irf_periods"])
        ax.tick_params(labelbottom=True)

    def _legend_in_spare_panel(self, fig: plt.Figure, axes: np.ndarray, n_panels: int) -> None:
        handles, labels = axes[0].get_legend_handles_labels()
        spare = axes[n_panels:]
        for ax in spare:
            ax.set_visible(False)
        if len(spare):
            spare[0].set_visible(True)
            spare[0].axis("off")
            spare[0].legend(handles, labels, loc="center", frameon=True, fontsize=self.style.figure["font"]["legend"] + 2)
        else:
            fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=True, bbox_to_anchor=(0.5, 0.0))


class CleaningFigure:
    """Figure 2: lexical sentiment for clean FOMC statements and with a website header prepended."""

    def __init__(self, config: PaperConfig, clean: pd.DataFrame, contaminated: pd.DataFrame) -> None:
        self.cleaning = config["exhibits"]["cleaning"]
        self.dictionary_labels = config["exhibits"]["dictionary_labels"]
        self.method_labels = config["exhibits"]["method_labels"]
        self.style = FigureStyle(config)
        self.clean = self._since_start(clean)
        self.contaminated = self._since_start(contaminated)

    def run(self) -> list[Path]:
        columns = self.cleaning["columns"]
        sns.set_theme(style="whitegrid", context="paper")
        fig, axes = plt.subplots(len(columns), 1, figsize=(9.2, 2.6 * len(columns)), sharex=True)
        for ax, column in zip(np.atleast_1d(axes), columns):
            self._plot_column(ax, column)
        fig.suptitle("Boilerplate header contamination can shift lexical sentiment", y=0.995)
        fig.autofmt_xdate()
        path = self.style.save(fig, "sentiment_cleaning_comparison.pdf", tight=False)
        self.style.apply_theme()
        return [path]

    def _plot_column(self, ax: plt.Axes, column: str) -> None:
        self.style.grid(ax, axis="both")
        self.clean[column].plot(ax=ax, color="#245c7a", linewidth=1.4, label=self.cleaning["clean_label"])
        self.contaminated[column].plot(ax=ax, color="#c05621", linewidth=1.2, linestyle="--", label=self.cleaning["header_label"])
        ax.axhline(0, color=self.style.figure["reference_line_color"], linestyle=":", linewidth=0.9)
        ax.set_title(self._title(column), loc="left", fontsize=10)
        ax.set_xlabel("")
        ax.set_ylabel("Sentiment score")
        ax.legend(loc="best", frameon=True)

    def _title(self, column: str) -> str:
        """``correa_sentiment_posneg_net`` becomes ``Correa(PosNeg)``."""
        dictionary, rest = column.replace("_stem", "").split("_sentiment_", maxsplit=1)
        method = rest.split("_", maxsplit=1)[0]
        return f"{self.dictionary_labels.get(dictionary, dictionary)}({self.method_labels.get(method, method)})"

    def _since_start(self, documents: pd.DataFrame) -> pd.DataFrame:
        dated = documents.assign(date=pd.to_datetime(documents["date"], errors="coerce")).dropna(subset=["date"]).sort_values("date")
        return dated[dated["date"] >= pd.Timestamp(self.cleaning["start_date"])].set_index("date")


class PipelineDiagram:
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

"""Figures 3 to 6: how the measures compare across the 51-bank statement panel."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.research_paper.analysis import LEXICAL, TRANSFORMER, SentimentMeasures
from src.research_paper.config import PaperConfig
from src.research_paper.exhibits.style import FigureStyle


FAMILIES = (LEXICAL, TRANSFORMER)


class MethodComparisonFigures:
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
        """Figure 3: every measure's pooled distribution, one row per family and one column per normalization.

        All-Words values are an order of magnitude smaller than All-Sentences values, so the
        length-normalized column gives each family its own x-axis. The PosNeg column shares
        the full [-1, 1] axis, where saturation at the bounds is visible for both families.
        """
        normalizations = {
            "Statement Length Normalization\n(All-Words, All-Sentences)": self.measures,
            "Token Normalization\n(PosNeg)": SentimentMeasures(self.documents, self.config, posneg=True),
        }
        n_measures = [len(self.measures.columns_in(family)) for family in FAMILIES]
        fig, axes = plt.subplots(len(FAMILIES), len(normalizations), figsize=(14, 7.5), height_ratios=n_measures, sharey="row")
        for col, (title, measures) in enumerate(normalizations.items()):
            long_form = measures.long_form()
            for row, family in enumerate(FAMILIES):
                ax = axes[row, col]
                order = [measures.labels[column] for column in measures.columns_in(family)]
                subset = long_form[long_form["Family"] == family]
                sns.boxplot(data=subset, x="value", y="Measure", order=order, color=self.figure["family_colors"][family], fliersize=1.5, linewidth=0.9, ax=ax)
                self.style.grid(ax, axis="x")
                self.style.zero_line(ax, vertical=True)
                if col == 1:
                    ax.set_xlim(-1.05, 1.05)
                ax.set_xlabel("")
                ax.set_ylabel(f"{family}" if col == 0 else "")
                if row == 0:
                    ax.set_title(title)
            axes[-1, col].set_xlabel("Sentiment")
        fig.suptitle("Distribution of statement-level sentiment", fontsize=self.figure["font"]["title"] + 7)
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
        ax.set_title("Document-level correlations across sentiment methods")
        plt.setp(ax.get_xticklabels(), rotation=40, ha="right")
        plt.setp(ax.get_yticklabels(), rotation=0)
        return self.style.save(fig, "method_correlation_levels.pdf")

    def time_series(self) -> Path:
        """Figure 5: rolling monthly means averaged across banks, one panel per family.

        Each family has its own y-axis, since All-Words values are an order of magnitude
        smaller than All-Sentences values.
        """
        monthly = self._smoothed(self.measures.monthly_bank_average())
        fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
        for ax, family in zip(axes, FAMILIES):
            self._plot_family(ax, monthly, self.measures, family)
            ax.set_title(f"{family} methods", loc="left")
        for ax in axes:
            ax.set_ylabel("Sentiment")
        fig.suptitle(f"Sentiment averaged across {self.n_banks} central banks")
        self.style.family_legend(fig, self._measure_lines(axes, self.measures), bottom=0.28, top=0.88, fontsize=self.figure["font"]["legend"] + 3)
        return self.style.save(fig, "method_time_series.pdf", tight=False)

    def country_comparison(self) -> Path:
        """Figure 6: rolling monthly means for selected banks, one row per bank."""
        countries = self.figure["panel_countries"]
        fig, axes = plt.subplots(len(countries), len(FAMILIES), figsize=(15, 4.6 * len(countries)), sharey="col", sharex=True)
        axes = np.atleast_2d(axes)
        for row, (code, name) in zip(axes, countries.items()):
            bank = SentimentMeasures(self.documents[self.documents["Country"] == code], self.config)
            monthly = self._smoothed(bank.monthly())
            for ax, family in zip(row, FAMILIES):
                self._plot_family(ax, monthly, bank, family)
                ax.set_title(f"{name} — {family} methods", loc="left")
            row[0].set_ylabel("Sentiment")
        fig.suptitle(f"Sentiment by central bank and method family ({self.figure['rolling_window']}-month rolling mean)")
        self.style.family_legend(fig, self._measure_lines(axes[0], self.measures), bottom=0.18, top=0.92, hspace=0.28, fontsize=self.figure["font"]["legend"] + 3)
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

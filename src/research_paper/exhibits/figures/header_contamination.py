"""Figure 2: lexical sentiment for clean FOMC statements and with a website header prepended."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.research_paper.config import PaperConfig
from src.research_paper.exhibits.style import FigureStyle


class HeaderContaminationFigure:
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

"""The sentiment measures the paper compares, and the summaries built from them."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.research_paper.config import PaperConfig

LEXICAL, TRANSFORMER, CROSS = "Lexical", "Transformer", "Cross-family"


class SentimentMeasures:
    """Document-level scores for the configured measures, all on the shared [-1, 1] scale.

    Both families always share one normalization: All-Words and All-Sentences by
    default, or ``posneg=True`` for the PosNeg pair that divides by matched terms
    and sentiment-bearing sentences only.
    """

    def __init__(self, documents: pd.DataFrame, config: PaperConfig, posneg: bool = False) -> None:
        measures = config["measures"]
        suffix = "_posneg" if posneg else ""
        lexical, transformer = measures[f"lexical{suffix}"], measures[f"transformer{suffix}"]
        self.families = {**dict.fromkeys(lexical, LEXICAL), **dict.fromkeys(transformer, TRANSFORMER)}
        self.labels = {**lexical, **transformer}
        self.documents = self._dated(documents)
        self.columns = [column for column in self.labels if column in self.documents.columns]

    def values(self) -> pd.DataFrame:
        """One column per measure, one row per document."""
        return self.documents[self.columns]

    def columns_in(self, family: str) -> list[str]:
        return [column for column in self.columns if self.families[column] == family]

    def long_form(self) -> pd.DataFrame:
        """One row per document and measure, labelled with the measure and its family."""
        long = self._with_date(self.values()).melt(id_vars="date", var_name="column", value_name="value")
        long["Measure"] = long["column"].map(self.labels)
        long["Family"] = long["column"].map(self.families)
        return long.dropna(subset=["value"])

    def monthly(self) -> pd.DataFrame:
        """Monthly means, with gaps inside the sample interpolated."""
        monthly = self._with_date(self.values()).set_index("date").resample("MS").mean()
        return monthly.interpolate(limit_area="inside")

    def monthly_bank_average(self) -> pd.DataFrame:
        """Monthly means that weight every central bank equally, gaps interpolated."""
        by_bank = self._with_date(self.values()).assign(Country=self.documents["Country"].to_numpy())
        monthly = by_bank.set_index("date").groupby("Country")[self.columns].resample("MS").mean()
        return monthly.groupby(level="date").mean().interpolate(limit_area="inside")

    def correlations(self, method: str = "pearson", within_bank: bool = False) -> pd.DataFrame:
        """Pairwise correlations, pooled or after removing each central bank's mean."""
        values = self.values()
        if within_bank:
            values = values.groupby(self.documents["Country"].to_numpy()).transform(lambda column: column - column.mean())
        correlation = values.corr(method=method)
        return correlation.rename(index=self.labels, columns=self.labels)

    def family_averages(self, correlation: pd.DataFrame) -> dict[str, float]:
        """Mean off-diagonal correlation within each family and across the two."""
        family_of = {self.labels[column]: self.families[column] for column in self.columns}
        pairs: dict[str, list[float]] = {LEXICAL: [], TRANSFORMER: [], CROSS: []}
        labels = list(correlation.index)
        for i, row in enumerate(labels):
            for column in labels[i + 1:]:
                family = family_of[row] if family_of[row] == family_of[column] else CROSS
                pairs[family].append(correlation.loc[row, column])
        return {family: float(np.mean(values)) for family, values in pairs.items() if values}

    def _with_date(self, values: pd.DataFrame) -> pd.DataFrame:
        return values.assign(date=self.documents["date"].to_numpy())[["date", *values.columns]]

    @staticmethod
    def _dated(documents: pd.DataFrame) -> pd.DataFrame:
        dated = documents.assign(date=pd.to_datetime(documents["date"], errors="coerce"))
        return dated.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

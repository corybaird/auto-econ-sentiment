"""Figure 2: the same statements scored as scraped and with a website header prepended."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.research_paper.config import PaperConfig
from src.research_paper.scoring.package_scorer import PackageScorer, ScoredDocuments


class HeaderExperiment:
    """Figure 2: lexical scores for FOMC statements as scraped, and with a website header prepended."""

    def __init__(self, config: PaperConfig, force: bool = False) -> None:
        self.header = config["exhibits"]["cleaning"]["header"]
        self.source = config.path("fomc_sample")
        self.directory = config.data_path("cleaning_experiment")
        self.scorer = PackageScorer(config, transformers=False)
        self.force = force

    def run(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Document scores for the clean statements and for the contaminated ones."""
        return self._cached("clean", self.source), self._cached("header", self._contaminated_input())

    def _cached(self, name: str, source: Path) -> pd.DataFrame:
        if ScoredDocuments.exists(self.directory, name) and not self.force:
            return ScoredDocuments.read(self.directory, name).documents
        scored = self.scorer.score(source, self.directory / "clean" / name)
        scored.write(self.directory, name)
        return scored.documents

    def _contaminated_input(self) -> Path:
        statements = pd.read_parquet(self.source)
        statements["text"] = f"{self.header} " + statements["text"].fillna("")
        path = self.directory / "input" / "header.parquet.gzip"
        path.parent.mkdir(parents=True, exist_ok=True)
        statements.to_parquet(path, compression="gzip", index=False)
        return path

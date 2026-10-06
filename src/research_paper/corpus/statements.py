"""The Monetary Policy Statement Database: one folder of dated .txt files per central bank."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from auto_econ_sentiment import TextLoader
from src.research_paper.config import PaperConfig


class StatementCorpus:
    """The Monetary Policy Statement Database: one folder of dated .txt files per central bank."""

    def __init__(self, config: PaperConfig) -> None:
        self.directory = config.path("statement_dir")
        self.configured_countries = config["corpus"]["countries"]

    def countries(self) -> list[str]:
        """The configured central banks, or every folder in the corpus when none are listed."""
        if self.configured_countries:
            return list(self.configured_countries)
        return sorted(path.name for path in self.directory.iterdir() if path.is_dir())

    def country_directory(self, country: str) -> Path:
        directory = self.directory / country
        if not directory.is_dir():
            raise FileNotFoundError(f"No statement folder for {country} at {directory}.")
        return directory

    def archive(self) -> pd.DataFrame:
        """Every statement across all central banks, with its word count, for the corpus table."""
        statements = TextLoader(file_path=self.directory, group_column="Country").get_data()
        statements["n_words"] = statements["text"].str.split().str.len()
        return statements

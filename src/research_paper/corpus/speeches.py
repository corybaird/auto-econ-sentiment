"""The CBS central bank speeches: one parquet file per central bank."""

from __future__ import annotations

import logging
from functools import cached_property

import pandas as pd

from auto_econ_sentiment import TextCleaner
from src.research_paper.config import PaperConfig

logger = logging.getLogger(__name__)


class SpeechCorpus:
    """The CBS central bank speeches: one parquet file per central bank."""

    _COLUMNS = ["Date", "CentralBank", "Country", "text"]
    _US_NAMES = {"US", "USA", "UNITED STATES"}

    def __init__(self, config: PaperConfig) -> None:
        self.directory = config.path("speech_dir")
        self.clean_config = {**config["clean"], "tokenize": False, "stem": False}

    @cached_property
    def speeches(self) -> pd.DataFrame:
        """Every speech with a parsed ``date`` column."""
        paths = sorted(self.directory.glob("*.parquet.gzip"))
        if not paths:
            raise FileNotFoundError(f"No speech files in {self.directory}.")
        speeches = pd.concat([pd.read_parquet(path, columns=self._COLUMNS) for path in paths], ignore_index=True)
        speeches["date"] = pd.to_datetime(speeches.pop("Date"), errors="coerce")
        logger.info("Loaded %d speeches from %d central banks", len(speeches), speeches["CentralBank"].nunique())
        return speeches

    def archive(self) -> pd.DataFrame:
        """Every speech with cleaned text, for counting words in the corpus table."""
        return TextCleaner(df=self.speeches, text_column="text", clean_config=self.clean_config).run()

    def united_states(self) -> pd.DataFrame:
        """US speeches with text and a date, in date order, labelled ``Country = US``."""
        us = self.speeches[self.speeches["Country"].astype(str).str.upper().isin(self._US_NAMES)]
        us = us.dropna(subset=["date", "text"]).assign(text=lambda frame: frame["text"].astype(str).str.strip())
        us = us[us["text"].str.len() > 0].sort_values("date").reset_index(drop=True)
        return us.assign(Country="US")[["date", "CentralBank", "Country", "text"]]

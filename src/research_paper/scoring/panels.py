"""Score a corpus one central bank or one year at a time, caching every unit."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.research_paper.config import PaperConfig
from src.research_paper.corpus import SpeechCorpus, StatementCorpus
from src.research_paper.scoring.package_scorer import PackageScorer, ScoredDocuments

logger = logging.getLogger(__name__)


class CachedPanel:
    """Score a corpus unit by unit, caching every unit, then combine the units into a panel.

    Subclasses name the panel and say what the units are and how to score one.
    """

    name: str

    def __init__(self, config: PaperConfig, force: bool = False) -> None:
        self.config = config
        self.force = force
        self.scorer = PackageScorer(config)
        self.directory = config.data_path(self.name)

    def build(self) -> ScoredDocuments:
        """Score every unit (reusing cached units) and write the combined panel."""
        panel = ScoredDocuments.concat([self._cached_unit(unit) for unit in self._units()])
        panel.write(self.directory, "panel")
        logger.info("Wrote %s panel: %d documents, %d sentences", self.name, len(panel.documents), len(panel.sentences))
        return panel

    def load(self) -> ScoredDocuments:
        """The combined panel, built first if it does not exist yet."""
        if ScoredDocuments.exists(self.directory, "panel") and not self.force:
            return ScoredDocuments.read(self.directory, "panel")
        return self.build()

    def _cached_unit(self, unit: str) -> ScoredDocuments:
        units_dir = self.directory / "units"
        if ScoredDocuments.exists(units_dir, unit) and not self.force:
            logger.info("Reusing cached %s %s", self.name, unit)
            return ScoredDocuments.read(units_dir, unit)
        logger.info("Scoring %s %s", self.name, unit)
        scored = self._score_unit(unit)
        scored.write(units_dir, unit)
        return scored

    def _work_dir(self, unit: str) -> Path:
        return self.directory / "clean" / unit

    def _units(self) -> list[str]:
        raise NotImplementedError

    def _score_unit(self, unit: str) -> ScoredDocuments:
        raise NotImplementedError


class StatementPanel(CachedPanel):
    """Every central bank's policy statements since ``corpus.statement_start``, one unit per bank."""

    name = "statements"

    def __init__(self, config: PaperConfig, force: bool = False) -> None:
        super().__init__(config, force)
        self.corpus = StatementCorpus(config)
        self.start = pd.Timestamp(config["corpus"]["statement_start"])

    def _units(self) -> list[str]:
        return self.corpus.countries()

    def _score_unit(self, country: str) -> ScoredDocuments:
        scored = self.scorer.score(self.corpus.country_directory(country), self._work_dir(country))
        dated = pd.to_datetime(scored.documents["date"], errors="coerce") >= self.start
        return scored.restrict(dated).label(Country=country)


class SpeechPanel(CachedPanel):
    """US central bank speeches for the VAR, one unit per calendar year."""

    name = "speeches"

    def __init__(self, config: PaperConfig, force: bool = False) -> None:
        super().__init__(config, force)
        self.corpus = SpeechCorpus(config)

    def _units(self) -> list[str]:
        return [str(year) for year in sorted(self._speeches()["date"].dt.year.unique())]

    def _score_unit(self, year: str) -> ScoredDocuments:
        speeches = self._speeches()
        source = self._stage(speeches[speeches["date"].dt.year == int(year)], year)
        return self.scorer.score(source, self._work_dir(year)).label(Country="US")

    def _speeches(self) -> pd.DataFrame:
        return self.corpus.united_states()

    def _stage(self, speeches: pd.DataFrame, year: str) -> Path:
        # The speech files mix every central bank, so each year's US speeches are
        # written to their own file for the pipeline to load.
        path = self.directory / "input" / f"{year}.parquet.gzip"
        path.parent.mkdir(parents=True, exist_ok=True)
        speeches.to_parquet(path, compression="gzip", index=False)
        return path

"""Score the corpora with the released auto-econ-sentiment pipeline.

Every score in the paper comes from ``PackageScorer``, which runs the package exactly
as a user would: load, clean, lexical scoring and transformer scoring through
``AutoEconSentiment.run``. The panels below decide what to feed it and cache each
unit of work so an interrupted run resumes where it stopped.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from auto_econ_sentiment import AutoEconSentiment
from src.research_paper.config import PaperConfig
from src.research_paper.corpora import SpeechCorpus, StatementCorpus

logger = logging.getLogger(__name__)

# Lexical output also carries the matched words; the panel keeps scores and match counts.
_LEXICAL_COLUMNS = "sentiment|counttoken"


@dataclass
class ScoredDocuments:
    """Scores at two grains: one row per document and one row per sentence."""

    documents: pd.DataFrame
    sentences: pd.DataFrame

    @classmethod
    def concat(cls, parts: list[ScoredDocuments]) -> ScoredDocuments:
        return cls(
            documents=pd.concat([part.documents for part in parts], ignore_index=True),
            sentences=pd.concat([part.sentences for part in parts], ignore_index=True),
        )

    @classmethod
    def read(cls, directory: Path, name: str) -> ScoredDocuments:
        documents_path, sentences_path = cls._paths(directory, name)
        return cls(pd.read_parquet(documents_path), pd.read_parquet(sentences_path))

    @classmethod
    def exists(cls, directory: Path, name: str) -> bool:
        return all(path.exists() for path in cls._paths(directory, name))

    def write(self, directory: Path, name: str) -> None:
        for frame, path in zip((self.documents, self.sentences), self._paths(directory, name)):
            path.parent.mkdir(parents=True, exist_ok=True)
            frame.to_parquet(path, compression="gzip", index=False)

    def restrict(self, keep: pd.Series) -> ScoredDocuments:
        """Keep the documents where ``keep`` is true, and only their sentences."""
        documents = self.documents[keep.to_numpy()]
        return ScoredDocuments(documents, self.sentences[self.sentences["id_text"].isin(documents["id_text"])])

    def label(self, **columns: str) -> ScoredDocuments:
        """Add constant columns, such as ``Country``, to both grains."""
        return ScoredDocuments(self.documents.assign(**columns), self.sentences.assign(**columns))

    @staticmethod
    def _paths(directory: Path, name: str) -> tuple[Path, Path]:
        return directory / "documents" / f"{name}.parquet.gzip", directory / "sentences" / f"{name}.parquet.gzip"


class PackageScorer:
    """Run AutoEconSentiment over one input and collect its document and sentence scores."""

    def __init__(self, config: PaperConfig, transformers: bool = True) -> None:
        self.clean_config = config["clean"]
        self.dictionaries = config["lexical"]["dictionaries"]
        self.aggregation_methods = config["lexical"]["aggregation_methods"]
        self.transformer_config = config.transformer_settings() if transformers else None

    def score(self, source: Path, work_dir: Path) -> ScoredDocuments:
        """Score ``source``, a tabular file or a folder of .txt files; cleaned text lands in ``work_dir``."""
        analyzer = AutoEconSentiment(import_file_path=source, text_column="text", date_column="date", export_path=work_dir)
        analyzer.run(
            clean_config=self.clean_config,
            dictionaries=self.dictionaries,
            aggregation_methods=self.aggregation_methods,
            export_results=False,
            transformer_config=self.transformer_config,
        )
        return ScoredDocuments(self._documents(analyzer), self._sentences(analyzer))

    @staticmethod
    def _documents(analyzer: AutoEconSentiment) -> pd.DataFrame:
        lexical = analyzer.df_sent_lexical.filter(regex=_LEXICAL_COLUMNS)
        scores = [lexical.loc[:, ~lexical.columns.duplicated()]]
        if analyzer.df_sent_transformer is not None:
            scores.append(analyzer.df_sent_transformer)
        return analyzer.df_clean.set_index("id_text")[["date"]].join(scores, how="left").reset_index()

    @staticmethod
    def _sentences(analyzer: AutoEconSentiment) -> pd.DataFrame:
        sentences = analyzer.df_transformer_sentence_probabilities
        return pd.DataFrame(columns=["id_text"]) if sentences is None else sentences.reset_index()


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

"""Run the released auto-econ-sentiment pipeline and hold its scores."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from auto_econ_sentiment import AutoEconSentiment
from src.research_paper.config import PaperConfig


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

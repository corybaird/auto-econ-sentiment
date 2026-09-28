"""Tables 1, 2 and the corpus summary, written as LaTeX tabulars the paper inputs."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import yaml

from src.research_paper.config import PaperConfig

logger = logging.getLogger(__name__)


class PaperTables:
    """Build each table from the corpora and configuration and write it to ``paths.table_dir``."""

    def __init__(self, config: PaperConfig, statements: pd.DataFrame, speeches: pd.DataFrame) -> None:
        self.config = config
        self.exhibits = config["exhibits"]
        self.directory = config.path("table_dir")
        self.statements = statements
        self.speeches = speeches

    def run(self) -> list[Path]:
        return [self.corpus_summary(), self.dictionary_summary(), self.transformer_summary()]

    def corpus_summary(self) -> Path:
        """Size of the two corpora: statements by central bank folder, speeches by ``CentralBank``."""
        statements = _corpus_stats(self.statements, words=self.statements["n_words"], bank="Country")
        speeches = _corpus_stats(self.speeches, words=self.speeches["text_clean"].fillna("").str.split().str.len(), bank="CentralBank")
        rows = [[name, statements[name], speeches[name]] for name in statements]
        return self._write("corpus_summary.tex", "lrr", ["", "Policy statements", "Speeches"], rows)

    def dictionary_summary(self) -> Path:
        """Table 1: the built-in dictionaries and their word counts."""
        with self.config.path("dictionaries").open(encoding="utf-8") as file:
            words = yaml.safe_load(file)
        rows = []
        for item in self.exhibits["dictionaries"]:
            positive, negative = (len(words[item["key"]].get(side, [])) for side in ("positive", "negative"))
            rows.append([item["name"], _code(item["key"]), item["domain"], item["stemming"], _integer(positive), _integer(negative), _integer(positive + negative)])
        header = ["Dictionary", "Short name", "Domain", "Stemming", "Positive", "Negative", "Total"]
        return self._write("dictionary_summary.tex", "llllrrr", header, rows)

    def transformer_summary(self) -> Path:
        """Table 2: the supported transformer models."""
        rows = [
            [item["name"], _code(item["short_name"]), item["architecture"], item["domain"], item["labels"], _code(item["hf_id"].replace("_", "\\_"))]
            for item in self.exhibits["transformers"]
        ]
        header = ["Model", "Short name", "Base Model", "Domain", "Target Classes", "Hugging Face ID"]
        return self._write("transformer_summary.tex", "llllll", header, rows)

    def _write(self, filename: str, align: str, header: list[str], rows: list[list[str]]) -> Path:
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / filename
        path.write_text(_tabular(align, header, rows), encoding="utf-8")
        logger.info("Wrote %s", path)
        return path


def _corpus_stats(documents: pd.DataFrame, words: pd.Series, bank: str) -> dict[str, str]:
    dates = pd.to_datetime(documents["date"], errors="coerce")
    return {
        "Documents": _integer(len(documents)),
        "Central banks": _integer(documents[bank].nunique()),
        "Sample period": f"{dates.min():%Y-%m}--{dates.max():%Y-%m}",
        "Total words": _integer(words.sum()),
        "Median words per document": _integer(words.median()),
        "Median documents per central bank": _integer(documents.groupby(bank).size().median()),
    }


def _tabular(align: str, header: list[str], rows: list[list[str]]) -> str:
    """A booktabs tabular with one header row."""
    body = "\n    ".join(" & ".join(row) + " \\\\" for row in rows)
    return (
        f"\\begin{{tabular}}{{{align}}}\n"
        "    \\toprule\n"
        f"    {' & '.join(header).strip()} \\\\\n"
        "    \\midrule\n"
        f"    {body}\n"
        "    \\bottomrule\n"
        "\\end{tabular}\n"
    )


def _integer(value: float) -> str:
    return f"{int(value):,}"


def _code(text: str) -> str:
    return f"\\texttt{{{text}}}"

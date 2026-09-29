"""Every table in the paper, written as a LaTeX tabular that main.tex inputs."""

from __future__ import annotations

import logging
import re
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import pandas as pd
import yaml

from src.research_paper.config import PaperConfig
from src.research_paper.analysis.sentence_audit import AuditSentence, SentenceAudit

logger = logging.getLogger(__name__)


class PaperTables:
    """Build each table from the corpora and configuration and write it to ``paths.table_dir``."""

    def __init__(self, config: PaperConfig, statements: pd.DataFrame, speeches: pd.DataFrame, audit: SentenceAudit) -> None:
        self.config = config
        self.exhibits = config["exhibits"]
        self.directory = config.path("table_dir")
        self.statements = statements
        self.speeches = speeches
        self.audit = audit

    def run(self) -> list[Path]:
        return [self.corpus_summary(), self.dictionary_summary(), self.transformer_summary(), self.sentence_audit()]

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

    def sentence_audit(self) -> Path:
        """Table 3: the audited statement's sentences, dictionary matches and classes."""
        table = SentenceAuditTable(self.exhibits["sentence_audit"], self.audit)
        align = "".join(f"p{{{width}}}" for width in table.settings["column_widths"])
        return self._write("sentence_audit.tex", align, table.header(), table.rows(), footer=[table.document_row()])

    def _write(self, filename: str, align: str, header: list[str], rows: list[list[str]], footer: list[str] | None = None) -> Path:
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / filename
        path.write_text(_tabular(align, header, rows, footer), encoding="utf-8")
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


def _tabular(align: str, header: list[str], rows: list[list[str]], footer: list[str] | None = None) -> str:
    """A booktabs tabular with one header row and optional footer lines after a midrule."""
    body = "\n    ".join(" & ".join(row) + " \\\\" for row in rows)
    if footer:
        body += "\n    \\midrule\n    " + "\n    ".join(footer)
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


class SentenceAuditTable:
    """Format the sentence audit as Table 3: abridged sentences, grouped matches, classes."""

    def __init__(self, settings: dict, audit: SentenceAudit) -> None:
        self.settings = settings
        self.audit = audit
        self.models = settings["model_columns"]

    def header(self) -> list[str]:
        return ["\\#", "Sentence (abridged)", "Dictionary matches", *self.models.values()]

    def rows(self) -> list[list[str]]:
        shown = [sentence for sentence in self.audit.sentences if sentence.number in self.settings["rows"]]
        return [[str(sentence.number), self._sentence(sentence), self._matches(sentence), *self._classes(sentence)] for sentence in shown]

    def document_row(self) -> str:
        scores = self.audit.document_scores()
        cells = [_signed(scores[short]) for short in self.models]
        label = f"Document score, \\texttt{{All-Sentences}} ({len(self.audit.sentences)} sentences)"
        return f"\\multicolumn{{2}}{{l}}{{{label}}} & & " + " & ".join(cells) + " \\\\"

    def _sentence(self, sentence: AuditSentence) -> str:
        """The configured abridgement, checked against the scored text, or the sentence itself."""
        abridged = self.settings["abridged"].get(sentence.number)
        if abridged is None:
            return _escape(sentence.text)
        scored = re.sub(r"\s", "", sentence.text)
        for fragment in re.split(r"\\ldots\\?\s*", abridged):
            if re.sub(r"\s", "", fragment) not in scored:
                raise ValueError(f"Abridged sentence {sentence.number} contains text not in the scored sentence: {fragment!r}")
        return abridged

    def _matches(self, sentence: AuditSentence) -> str:
        """Matches grouped by dictionaries that found the same words, e.g. ``strong (+): HL, LM``."""
        groups: dict[tuple, list[str]] = {}
        for dictionary, words in sentence.matches.items():
            signature = tuple((sign, tuple(words[sign])) for sign in (1, -1) if words[sign])
            groups.setdefault(signature, []).append(self.settings["dictionary_abbreviations"][dictionary])
        parts = [f"{', '.join(_signed_words(sign, words) for sign, words in signature)}: {', '.join(dictionaries)}" for signature, dictionaries in groups.items()]
        return "; ".join(parts) or "none"

    def _classes(self, sentence: AuditSentence) -> list[str]:
        cells = []
        for short in self.models:
            found = sentence.classes[short]
            cells.append("none" if found is None else f"{self.settings['class_names'][short][found[0]]} ({_two_decimals(found[1])})")
        return cells


def _signed_words(sign: int, words: tuple[str, ...]) -> str:
    """``(1, ("achieving", "stability"))`` becomes ``\\textit{achieving}, \\textit{stability} ($+$)``."""
    italic = ", ".join(f"\\textit{{{word}}}" for word in words)
    return f"{italic} (${'+' if sign > 0 else '-'}$)"


def _two_decimals(value: float) -> str:
    """Round half up, so 0.125 prints as 0.13 as in the paper, not 0.12."""
    return str(Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def _signed(value: float) -> str:
    text = _two_decimals(abs(value))
    return f"$-{text}$" if value < 0 and text != "0.00" else text


def _escape(text: str) -> str:
    return re.sub(r"([%&#_$])", r"\\\1", text)

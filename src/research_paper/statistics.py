"""Every number the paper's text quotes, recomputed from the scored panels.

The output is a markdown file (``paths.statistics``) with one section per claim in the
text, so each figure quoted in main.tex can be checked against the run that made it.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from auto_econ_sentiment import SentimentLexical, TextCleaner
from src.research_paper.config import PaperConfig
from src.research_paper.econometrics import ImpulseResponse
from src.research_paper.measures import CROSS, LEXICAL, TRANSFORMER, SentimentMeasures
from src.research_paper.scoring import ScoredDocuments

logger = logging.getLogger(__name__)

_BOUND = 1 - 1e-9


class PaperStatistics:
    """Recompute the statistics quoted in the text and write them as markdown tables."""

    def __init__(self, config: PaperConfig, statements: ScoredDocuments, responses: dict[str, ImpulseResponse]) -> None:
        self.config = config
        self.documents = statements.documents
        self.sentences = statements.sentences
        self.responses = responses
        self.model_labels = config.model_labels()
        self.lexical = config["measures"]["lexical"]

    def run(self) -> list[Path]:
        sections = {
            "Transformer PosNeg coverage and saturation (Sections 2.3 and 3.2)": self.transformer_bounds(),
            "Lexical coverage and saturation (Sections 2.3 and 3.2)": self.lexical_bounds(),
            "Levels by measure, statement level (Section 3.2)": self.levels(),
            "Levels since 2006, monthly means weighting banks equally (Figure 5)": self.time_series_levels(),
            "Correlation family averages (Figure 4)": self.correlation_summary(),
            "VAR impulse responses (Section 3.3)": self.var_summary(),
            "Sentence audit (Table 3)": SentenceAudit(self.config, self.documents, self.sentences).table(),
        }
        return [self._write(sections)]

    def transformer_bounds(self) -> pd.DataFrame:
        rows = {}
        for short, label in self.model_labels.items():
            posneg = self.documents[f"{short}_sentiment_posneg_net"]
            allsentences = self.documents[f"{short}_sentiment_allsentences_net"]
            rows[label] = {
                "PosNeg undefined (%)": _percent(posneg.isna()),
                "PosNeg at a bound (% of statements)": _percent(posneg.abs() >= _BOUND),
                "All-Sentences at a bound (%)": _percent(allsentences.abs() >= _BOUND),
                "All-Sentences undefined": int(allsentences.isna().sum()),
            }
        return pd.DataFrame(rows).T

    def lexical_bounds(self) -> pd.DataFrame:
        rows = {}
        for column, label in self.lexical.items():
            rows[label] = {
                "No dictionary match (%)": _percent(self._matches(column) == 0),
                "PosNeg at a bound (%)": _percent(self.documents[column].abs() >= _BOUND),
            }
        return pd.DataFrame(rows).T

    def levels(self) -> pd.DataFrame:
        rows = {}
        for measures in (SentimentMeasures(self.documents, self.config), SentimentMeasures(self.documents, self.config, "transformer_posneg")):
            for column in measures.columns:
                name = f"{measures.labels[column]} ({_aggregation(column)})"
                rows[name] = {"Mean": measures.values()[column].mean(), "Share above zero": (measures.values()[column] > 0).mean()}
        return pd.DataFrame(rows).T

    def time_series_levels(self) -> pd.DataFrame:
        measures = SentimentMeasures(self.documents, self.config)
        monthly = measures.monthly_bank_average()
        monthly = monthly.loc[monthly.index >= pd.Timestamp(self.config["figure"]["time_series_start"])]
        return pd.DataFrame({"Mean": monthly.mean(), "Share of months above zero": (monthly > 0).mean()}).rename(index=measures.labels)

    def correlation_summary(self) -> pd.DataFrame:
        measures = SentimentMeasures(self.documents, self.config)
        rows = {}
        for within_bank in (False, True):
            for method in ("pearson", "spearman"):
                correlation = measures.correlations(method, within_bank)
                averages = measures.family_averages(correlation)
                row = {f"Within {LEXICAL.lower()}": averages[LEXICAL], f"Within {TRANSFORMER.lower()}": averages[TRANSFORMER], CROSS: averages[CROSS]}
                row.update({f"Correa, {label}": correlation.loc["Correa", label] for label in self.model_labels.values()})
                rows[f"{'Within-bank' if within_bank else 'Pooled'} {method}"] = row
        return pd.DataFrame(rows).T

    def var_summary(self) -> pd.DataFrame:
        variables = self.config["var"]["variables"]
        rows = []
        for measure, response in self.responses.items():
            for index, variable in enumerate(variables[1:], start=1):
                path = response.point[:, index]
                peak = int(np.argmax(np.abs(path)))
                rows.append({
                    "Measure": self.config["var"]["measures"][measure],
                    "Lags": response.lags,
                    "Response": self.config["var"]["response_labels"][variable],
                    "Peak": path[peak],
                    "Peak month": peak,
                    "Significant months": _month_ranges(response.significant_months(index)),
                })
        return pd.DataFrame(rows)

    def _matches(self, column: str) -> pd.Series:
        """Dictionary matches per statement, positive plus negative, for a lexical measure column."""
        dictionary = column.split("_")[0]
        stem = "_stem" if "_stem" in column else ""
        counts = [self.documents[f"{dictionary}_counttoken_{side}_posneg{stem}"] for side in ("positive", "negative")]
        return counts[0] + counts[1]

    def _write(self, sections: dict[str, pd.DataFrame]) -> Path:
        path = self.config.path("statistics")
        path.parent.mkdir(parents=True, exist_ok=True)
        header = f"# Paper statistics\n\nGenerated from {len(self.documents):,} statements across {self.documents['Country'].nunique()} central banks.\n"
        body = "\n".join(f"## {title}\n\n{_markdown(table)}\n" for title, table in sections.items())
        path.write_text(f"{header}\n{body}", encoding="utf-8")
        logger.info("Wrote %s", path)
        return path


class SentenceAudit:
    """Table 3: one statement sentence by sentence, dictionary matches next to classifier output."""

    def __init__(self, config: PaperConfig, documents: pd.DataFrame, sentences: pd.DataFrame) -> None:
        audit = config["exhibits"]["sentence_audit"]
        self.config = config
        self.cutoff = config["transformer"]["sentence_probability_cutoff"]
        self.models = {model["short_name"]: model for model in config["transformer"]["models"]}
        document = documents[(documents["Country"] == audit["country"]) & (pd.to_datetime(documents["date"]) == pd.Timestamp(audit["date"]))]
        if len(document) != 1:
            raise ValueError(f"Expected one {audit['country']} statement on {audit['date']}, found {len(document)}.")
        self.document = document.iloc[0]
        self.sentences = sentences[(sentences["Country"] == audit["country"]) & (sentences["id_text"] == self.document["id_text"])]

    def table(self) -> pd.DataFrame:
        rows = self._dictionary_matches(self.sentences["sentence_text"])
        for short, model in self.models.items():
            rows[model["label"]] = [self._classification(sentence, short, model["label_map"]) for _, sentence in self.sentences.iterrows()]
        rows.insert(0, "Sentence", self.sentences["sentence_text"].str.slice(0, 90).to_numpy())
        rows.insert(0, "#", self.sentences["sentence_number"].to_numpy())
        total = {"#": "", "Sentence": f"Document All-Sentences score ({len(rows)} sentences)", "Dictionary matches": ""}
        total.update({model["label"]: f"{self.document[f'{short}_sentiment_allsentences_net']:.3f}" for short, model in self.models.items()})
        return pd.concat([rows, pd.DataFrame([total])], ignore_index=True)

    def _classification(self, sentence: pd.Series, short: str, label_map: dict) -> str:
        """The class that clears the cutoff with its probability, or ``none``."""
        probabilities = {label: sentence[f"{short}_{label}"] for label in label_map}
        label, probability = max(probabilities.items(), key=lambda item: item[1])
        if probability < self.cutoff:
            return "none"
        return f"{label} ({label_map[label]:+d}, {probability:.2f})"

    def _dictionary_matches(self, texts: pd.Series) -> pd.DataFrame:
        """Matched positive and negative words per sentence for every dictionary."""
        clean_config = {**self.config["clean"], "tokenize": True, "stem": True}
        cleaned = TextCleaner(df=pd.DataFrame({"text": texts.to_numpy()}), text_column="text", clean_config=clean_config).run()
        scorer = SentimentLexical(df_input=cleaned)
        dictionaries = self.config["lexical"]["dictionaries"]
        matches = [[] for _ in range(len(cleaned))]
        for kind, text_column in (("unstemmed", "text_tokens_str"), ("stemmed", "text_stems")):
            for dictionary in dictionaries[kind]:
                result = scorer.sentiment_pipeline(dictionary, "posneg", text_column=text_column)
                for i, (positive, negative) in enumerate(zip(result[f"{dictionary}_words_positive_posneg"], result[f"{dictionary}_words_negative_posneg"])):
                    words = [f"+{word}" for word in positive] + [f"-{word}" for word in negative]
                    if words:
                        matches[i].append(f"{dictionary}: {' '.join(words)}")
        return pd.DataFrame({"Dictionary matches": ["; ".join(found) or "none" for found in matches]})


def _percent(flags: pd.Series) -> float:
    return round(100 * float(flags.mean()), 2)


def _aggregation(column: str) -> str:
    return "All-Sentences" if "allsentences" in column else "PosNeg"


def _month_ranges(months: list[int]) -> str:
    """``[0, 4, 5, 6]`` becomes ``0, 4-6``."""
    if not months:
        return "none"
    ranges, start = [], months[0]
    for previous, month in zip(months, months[1:] + [None]):
        if month != previous + 1:
            ranges.append(f"{start}" if start == previous else f"{start}-{previous}")
            start = month
    return ", ".join(ranges)


def _markdown(table: pd.DataFrame) -> str:
    """A pipe table, numbers to three decimals, without needing the tabulate package."""
    if not isinstance(table.index, pd.RangeIndex):
        table = table.rename_axis("").reset_index()
    cells = table.map(lambda value: f"{value:.3f}" if isinstance(value, float) else str(value))
    lines = [" | ".join(map(str, cells.columns)), " | ".join("---" for _ in cells.columns)]
    lines += [" | ".join(row) for row in cells.to_numpy().tolist()]
    return "\n".join(f"| {line} |" for line in lines)

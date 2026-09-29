"""Table 3: one statement read sentence by sentence, dictionary matches next to classifier output."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from auto_econ_sentiment import SentimentLexical, TextCleaner
from src.research_paper.config import PaperConfig
from src.research_paper.scoring import ScoredDocuments


@dataclass
class AuditSentence:
    """One sentence of the audited statement."""

    number: int
    text: str
    matches: dict[str, dict[int, list[str]]]      # dictionary -> {+1: words, -1: words}
    classes: dict[str, tuple[str, float] | None]   # model short name -> (class, probability), None below the cutoff


class SentenceAudit:
    """The sentences of the configured statement with their dictionary matches and classifications."""

    def __init__(self, config: PaperConfig, statements: ScoredDocuments) -> None:
        self.config = config
        self.settings = config["exhibits"]["sentence_audit"]
        self.cutoff = config["transformer"]["sentence_probability_cutoff"]
        self.models = {model["short_name"]: model for model in config["transformer"]["models"]}
        self.document = self._document(statements.documents)
        sentences = statements.sentences
        self.sentence_rows = sentences[(sentences["Country"] == self.settings["country"]) & (sentences["id_text"] == self.document["id_text"])]
        self.sentences = self._audit_sentences()

    def document_scores(self) -> dict[str, float]:
        """Each model's All-Sentences score for the whole statement."""
        return {short: float(self.document[f"{short}_sentiment_allsentences_net"]) for short in self.models}

    def markdown_table(self) -> pd.DataFrame:
        """Every sentence in full detail, for the statistics file."""
        rows = [
            {
                "#": sentence.number,
                "Sentence": sentence.text[:90],
                "Dictionary matches": _plain_matches(sentence.matches),
                **{self.models[short]["label"]: _plain_class(found, self.models[short]["label_map"]) for short, found in sentence.classes.items()},
            }
            for sentence in self.sentences
        ]
        total = {"#": "", "Sentence": f"Document All-Sentences score ({len(self.sentences)} sentences)", "Dictionary matches": ""}
        total.update({self.models[short]["label"]: f"{score:.3f}" for short, score in self.document_scores().items()})
        return pd.DataFrame([*rows, total])

    def _document(self, documents: pd.DataFrame) -> pd.Series:
        match = documents[(documents["Country"] == self.settings["country"]) & (pd.to_datetime(documents["date"]) == pd.Timestamp(self.settings["date"]))]
        if len(match) != 1:
            raise ValueError(f"Expected one {self.settings['country']} statement on {self.settings['date']}, found {len(match)}.")
        return match.iloc[0]

    def _audit_sentences(self) -> list[AuditSentence]:
        texts = self.sentence_rows["sentence_text"].tolist()
        matches = self._dictionary_matches(texts)
        return [
            AuditSentence(number=int(row["sentence_number"]), text=text, matches=found, classes=self._classes(row))
            for (_, row), text, found in zip(self.sentence_rows.iterrows(), texts, matches)
        ]

    def _classes(self, row: pd.Series) -> dict[str, tuple[str, float] | None]:
        """For each model, the class that clears the cutoff and its probability."""
        classes = {}
        for short, model in self.models.items():
            label, probability = max(((label, row[f"{short}_{label}"]) for label in model["label_map"]), key=lambda item: item[1])
            classes[short] = (label, float(probability)) if probability >= self.cutoff else None
        return classes

    def _dictionary_matches(self, texts: list[str]) -> list[dict[str, dict[int, list[str]]]]:
        """Positive and negative words each dictionary matches in each sentence."""
        cleaned = TextCleaner(df=pd.DataFrame({"text": texts}), text_column="text", clean_config={**self.config["clean"], "tokenize": True, "stem": True}).run()
        scorer = SentimentLexical(df_input=cleaned)
        matches: list[dict] = [{} for _ in texts]
        for kind, text_column in (("unstemmed", "text_tokens_str"), ("stemmed", "text_stems")):
            for dictionary in self.config["lexical"]["dictionaries"][kind]:
                result = scorer.sentiment_pipeline(dictionary, "posneg", text_column=text_column)
                positives, negatives = result[f"{dictionary}_words_positive_posneg"], result[f"{dictionary}_words_negative_posneg"]
                for found, positive, negative in zip(matches, positives, negatives):
                    if positive or negative:
                        found[dictionary] = {1: list(positive), -1: list(negative)}
        return matches


def _plain_matches(matches: dict[str, dict[int, list[str]]]) -> str:
    parts = [f"{dictionary}: " + " ".join([f"+{word}" for word in words[1]] + [f"-{word}" for word in words[-1]]) for dictionary, words in matches.items()]
    return "; ".join(parts) or "none"


def _plain_class(found: tuple[str, float] | None, label_map: dict) -> str:
    return "none" if found is None else f"{found[0]} ({label_map[found[0]]:+d}, {found[1]:.2f})"

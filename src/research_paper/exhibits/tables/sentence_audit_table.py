"""Table 3: the sentence audit formatted for the paper."""

from __future__ import annotations

import re
from decimal import ROUND_HALF_UP, Decimal

from src.research_paper.analysis import AuditSentence, SentenceAudit


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

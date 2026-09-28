import logging
import re
from typing import Callable, Optional

import pandas as pd

logger = logging.getLogger(__name__)

_ABBREVIATIONS = (
    r"U\.S|U\.K|E\.U|e\.g|i\.e|etc|vs|Mr|Mrs|Ms|Dr|Prof|Gov|Jr|Sr|Inc|Ltd|Corp|Co|No|Fig|Dept|"
    r"St|Ave|approx|p\.a|a\.m|p\.m|Sept|Oct|Nov|Dec|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep"
)
_ABBREV_PATTERN = re.compile(rf"\b({_ABBREVIATIONS})\.", re.IGNORECASE)
_INITIAL_PATTERN = re.compile(r"\b([A-Z])\.(?=\s*[,;]?\s*[A-Z0-9\"'(\[])")
_PLACEHOLDER_DOT = "\x00DOT\x00"

_NEWLINE_BOUNDARY = re.compile(
    r"(?<=[.!?])\s*\n+\s*|(?<=[a-z:])\s*\n+\s*(?=[A-Z0-9\"'(\[])|\n\s*\n+"
)
_FALLBACK_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")

_VERBS = re.compile(
    r"\b(is|are|was|were|be|been|being|has|have|had|do|does|did|"
    r"will|would|shall|should|can|could|may|might|must|"
    r"remains?|expects?|continues?|increases?|decreases?|rose|fell|declined?|"
    r"decided|voted|judges?|anticipates?|maintains?|raises?|lowers?|"
    r"sees?|saw|noted?|stated?|added?|believes?|agreed?|approved?|"
    r"announced?|indicated?|reaffirmed?|projected?|estimated?|"
    r"supports?|supported|adopted|confirmed|concluded|targeted?|targets?|"
    r"reiterated?|emphasized?|assessed?|weighed?|balanced?|"
    r"slowed?|accelerated?|grew|grown|expanded?|shrank|moderated?|"
    r"eased?|tightened?|held|holds?|holding|adjusted?|adjusts?|"
    r"provided?|provides?|providing|strengthened?|weakened?|"
    r"drop|dropped|dropping|gains?|gained|gaining|"
    r"cut|cuts|cutting|reduced?|reduces?|reducing|removes?|removed?|removing|"
    r"recovering|surpassed|exceeded|outpaced|stands?|stood|"
    r"acted|acting|acts|favored|opted|directed|instructed|authorized)\b",
    re.IGNORECASE,
)
_HEADER_FOOTER_NOISE = re.compile(
    r"^(For immediate release|Press releases|Contact Us|Last Update|Board of Governors|Federal Reserve)\b|^[|\s\-–—]+$",
    re.IGNORECASE,
)
_ROSTER_NOISE = re.compile(
    r"^(Voting (for|against)|Absent and not voting|Members voting|Voting were)\b|"
    r"\b(Vice Chairman|Chairman|Governor|President)\b.*(;|\band\b).*\b(Vice Chairman|Chairman|Governor|President)\b",
    re.IGNORECASE,
)


def _protect_abbreviations(text: str) -> str:
    """Replace dots in abbreviations and single initials with placeholders."""
    text = _ABBREV_PATTERN.sub(lambda m: m.group(0).replace(".", _PLACEHOLDER_DOT), text)
    text = _INITIAL_PATTERN.sub(rf"\1{_PLACEHOLDER_DOT}", text)
    return text


def _unprotect_abbreviations(text: str) -> str:
    """Restore dots from placeholder tokens."""
    return text.replace(_PLACEHOLDER_DOT, ".")


def _merge_incomplete_sentences(sentences: list[str]) -> list[str]:
    """Merge lowercase- or punctuation-initial fragments into the preceding sentence."""
    if not sentences:
        return []
    merged: list[str] = [sentences[0]]
    for piece in sentences[1:]:
        clean = piece.strip()
        if not clean:
            continue
        if clean.startswith(";") or clean.startswith(","):
            merged[-1] = f"{merged[-1]} {clean}"
        elif clean[0].islower():
            merged[-1] = f"{merged[-1]} {clean}"
        else:
            merged.append(clean)
    return merged


def _is_valid_sentence(sentence: str) -> bool:
    """Check if sentence contains meaningful verbal content and is not metadata garbage."""
    s = sentence.strip()
    if len(s) < 20:
        return False
    if _ROSTER_NOISE.search(s):
        return False
    if s.lower().startswith("for immediate release") and len(s) < 40:
        return False
    if _HEADER_FOOTER_NOISE.search(s) and not _VERBS.search(s):
        return False
    if not _VERBS.search(s):
        return False
    return True


class TextSegmenter:
    """Split documents into sentence-level rows for sentence-level scoring.

    Uses NLTK's ``sent_tokenize`` when available and falls back to a robust
    regex boundary rule when NLTK or its ``punkt`` data is missing.
    """

    def __init__(
        self,
        text_column: str,
        id_column: str = "id_text",
        min_chars: int = 20,
        sentence_number_column: str = "sentence_number",
        require_nltk: bool = False,
        drop_invalid: bool = False,
    ) -> None:
        self.text_column = text_column
        self.id_column = id_column
        self.min_chars = min_chars
        self.sentence_number_column = sentence_number_column
        self.require_nltk = require_nltk
        self.drop_invalid = drop_invalid

        try:
            self._sent_tokenize = self._load_tokenizer(require_nltk=require_nltk)
        except TypeError:
            self._sent_tokenize = self._load_tokenizer()

        self.tokenizer_name: str = (
            "nltk_punkt" if self._sent_tokenize is not None else "regex_fallback"
        )
        if self.require_nltk and self._sent_tokenize is None:
            raise RuntimeError("NLTK sentence tokenizer is required but unavailable.")
        logger.info("TextSegmenter initialized with tokenizer: %s", self.tokenizer_name)

    @staticmethod
    def _load_tokenizer(require_nltk: bool = False) -> Optional[Callable[[str], list[str]]]:
        try:
            from nltk.tokenize import sent_tokenize

            sent_tokenize("Probe sentence. Second probe.")
            logger.info("Using NLTK sentence tokenizer.")
            return sent_tokenize
        except (ImportError, LookupError) as exc:
            if require_nltk:
                raise exc
            logger.warning("NLTK sentence tokenizer unavailable; using regex sentence boundaries.")
            return None

    def split_text(self, text: str) -> list[str]:
        """Split a single document into sentences longer than ``min_chars``."""
        if not isinstance(text, str) or not text.strip():
            return []

        stripped = text.strip()
        protected = _protect_abbreviations(stripped)
        blocks = _NEWLINE_BOUNDARY.split(protected)

        raw_sentences: list[str] = []
        for block in blocks:
            clean_block = block.strip()
            if not clean_block:
                continue
            if self._sent_tokenize is not None:
                try:
                    pieces = self._sent_tokenize(clean_block)
                except LookupError:
                    pieces = _FALLBACK_BOUNDARY.split(clean_block)
            else:
                pieces = _FALLBACK_BOUNDARY.split(clean_block)

            for piece in pieces:
                unprotected = _unprotect_abbreviations(piece).strip()
                if unprotected:
                    raw_sentences.append(unprotected)

        merged = _merge_incomplete_sentences(raw_sentences)

        results: list[str] = []
        for sentence in merged:
            cleaned = re.sub(r"\s+", " ", sentence).strip()
            if len(cleaned) < self.min_chars:
                continue
            if self.drop_invalid and not _is_valid_sentence(cleaned):
                continue
            results.append(cleaned)

        return results

    def run(self, df_input: pd.DataFrame) -> pd.DataFrame:
        """Explode ``df_input`` into one row per sentence, preserving ``id_column``."""
        if self.id_column not in df_input.columns:
            raise ValueError(f"Sentence-level segmentation requires an '{self.id_column}' column.")
        if self.text_column not in df_input.columns:
            raise ValueError(f"Text column '{self.text_column}' not found in input DataFrame.")

        work = df_input.dropna(subset=[self.text_column]).copy()
        work["__sentences"] = work[self.text_column].apply(self.split_text)
        exploded = work.explode("__sentences").dropna(subset=["__sentences"])
        exploded = exploded[exploded["__sentences"].str.strip().str.len() > 0].copy()
        if exploded.empty:
            raise ValueError(f"No sentences produced from column '{self.text_column}'.")

        exploded[self.sentence_number_column] = exploded.groupby(self.id_column).cumcount() + 1
        exploded[self.text_column] = exploded["__sentences"]
        logger.info("Split %s documents into %s sentences.", df_input.shape[0], exploded.shape[0])
        return exploded.drop(columns=["__sentences"]).reset_index(drop=True)


class ParagraphSegmenter:
    """Split documents into paragraph-level rows for paragraph-level processing or scoring."""

    def __init__(
        self,
        text_column: str,
        id_column: str = "id_text",
        min_chars: int = 20,
        paragraph_number_column: str = "paragraph_number",
    ) -> None:
        self.text_column = text_column
        self.id_column = id_column
        self.min_chars = min_chars
        self.paragraph_number_column = paragraph_number_column

    def split_text(self, text: str) -> list[str]:
        """Split a single document into paragraphs longer than ``min_chars``."""
        if not isinstance(text, str) or not text.strip():
            return []
        stripped = text.strip()
        pieces = re.split(r"\n\s*\n+", stripped)
        results: list[str] = []
        for piece in pieces:
            cleaned = piece.strip()
            if len(cleaned) >= self.min_chars:
                results.append(cleaned)
        return results

    def run(self, df_input: pd.DataFrame) -> pd.DataFrame:
        """Explode ``df_input`` into one row per paragraph, preserving ``id_column``."""
        if self.id_column not in df_input.columns:
            raise ValueError(f"Paragraph-level segmentation requires an '{self.id_column}' column.")
        if self.text_column not in df_input.columns:
            raise ValueError(f"Text column '{self.text_column}' not found in input DataFrame.")

        work = df_input.dropna(subset=[self.text_column]).copy()
        work["__paragraphs"] = work[self.text_column].apply(self.split_text)
        exploded = work.explode("__paragraphs").dropna(subset=["__paragraphs"])
        exploded = exploded[exploded["__paragraphs"].str.strip().str.len() > 0].copy()
        if exploded.empty:
            raise ValueError(f"No paragraphs produced from column '{self.text_column}'.")

        exploded[self.paragraph_number_column] = (
            exploded.groupby(self.id_column).cumcount() + 1
        )
        exploded[self.text_column] = exploded["__paragraphs"]
        logger.info("Split %s documents into %s paragraphs.", df_input.shape[0], exploded.shape[0])
        return exploded.drop(columns=["__paragraphs"]).reset_index(drop=True)


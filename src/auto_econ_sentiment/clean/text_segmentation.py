import re
import logging
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

_FALLBACK_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])")


class TextSegmenter:
    """Split documents into sentence-level rows for sentence-level scoring.

    Uses NLTK's ``sent_tokenize`` when available and falls back to a regex
    boundary rule when NLTK or its ``punkt`` data is missing.
    """

    def __init__(self, text_column: str, id_column: str = "id_text", min_chars: int = 20, sentence_number_column: str = "sentence_number") -> None:
        self.text_column = text_column
        self.id_column = id_column
        self.min_chars = min_chars
        self.sentence_number_column = sentence_number_column
        self._sent_tokenize = self._load_tokenizer()

    @staticmethod
    def _load_tokenizer():
        try:
            from nltk.tokenize import sent_tokenize

            sent_tokenize("Probe sentence. Second probe.")
            return sent_tokenize
        except (ImportError, LookupError):
            logger.warning("NLTK sentence tokenizer unavailable; using regex sentence boundaries.")
            return None

    def split_text(self, text: str) -> list[str]:
        """Split a single document into sentences longer than ``min_chars``."""
        if not isinstance(text, str) or not text.strip():
            return []
        stripped = text.strip()
        if self._sent_tokenize is not None:
            try:
                pieces = self._sent_tokenize(stripped)
            except LookupError:
                pieces = _FALLBACK_BOUNDARY.split(stripped)
        else:
            pieces = _FALLBACK_BOUNDARY.split(stripped)
        return [piece.strip() for piece in pieces if len(piece.strip()) >= self.min_chars]

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

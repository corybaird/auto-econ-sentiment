import logging
from typing import Any, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from auto_econ_sentiment.models.sentiment_base import SentimentBase


logger = logging.getLogger(__name__)


class TransformerDependencyError(ImportError):
    """Raised when transformer support is requested without optional packages."""


def _import_transformer_dependencies():
    """Import heavy transformer dependencies only when they are actually used."""
    try:
        import torch
        import torch.nn.functional as functional
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:
        raise TransformerDependencyError(
            "Transformer sentiment requires optional dependencies. Install them with "
            "`uv sync --extra transformers` or `pip install 'auto-econ-sentiment[transformers]'`."
        ) from exc
    return torch, functional, AutoTokenizer, AutoModelForSequenceClassification


class SentimentTransformers(SentimentBase):
    """Sentiment scorer backed by Hugging Face sequence-classification models.

    Dependencies are imported lazily so lexical users do not need ``torch`` or
    ``transformers`` installed. Label direction is configured explicitly through
    ``label_map`` because Hugging Face models do not share a universal label
    ordering.
    """

    def __init__(
        self,
        model_name: str,
        model_name_short: str,
        label_map: dict[str, int | float],
        num_labels: Optional[int] = None,
        max_length: int = 512,
        batch_size: int = 16,
        output_schema: Optional[str] = None,
        net_sentiment_formula: str = "positive_minus_negative",
        df_input: Optional[pd.DataFrame] = None,
        text_column: Optional[str] = None,
        huggingface_token: str | bool | None = None,
        device: Optional[str] = None,
        tokenizer: Any = None,
        model: Any = None,
        log_level: int = logging.INFO,
    ) -> None:
        super().__init__(df_input, text_column)
        self.model_name = model_name
        self.model_name_short = model_name_short
        self.label_map = self._normalize_label_map(label_map)
        self.num_labels = num_labels
        self.max_length = max_length
        self.batch_size = batch_size
        self.output_schema = output_schema
        self.net_sentiment_formula = net_sentiment_formula
        self.huggingface_token = huggingface_token
        self.tokenizer = tokenizer
        self.model = model
        self.df_labels: Optional[pd.DataFrame] = None
        self.df_sentence_probabilities: Optional[pd.DataFrame] = None
        self.df_sentiment_output: Optional[pd.DataFrame] = None

        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.logger.setLevel(log_level)

        self._torch = None
        self._functional = None
        if self.tokenizer is None or self.model is None:
            self._prepare_model_and_tokenizer(device=device)
        else:
            self._torch, self._functional, _, _ = _import_transformer_dependencies()
            self.device = self._resolve_device(device)
            self.model.to(self.device)

        if self.num_labels is None:
            self.num_labels = self._infer_num_labels()

        self._validate_label_map()

    @staticmethod
    def _normalize_label_map(label_map: dict[str, int | float]) -> dict[str, float]:
        if not label_map:
            raise ValueError("label_map is required for transformer sentiment.")
        return {str(label): float(value) for label, value in label_map.items()}

    def _resolve_device(self, device: Optional[str] = None):
        if self._torch is None:
            self._torch, self._functional, _, _ = _import_transformer_dependencies()
        if device is not None:
            return self._torch.device(device)
        return self._torch.device("cuda" if self._torch.cuda.is_available() else "cpu")

    def _prepare_model_and_tokenizer(self, device: Optional[str] = None) -> None:
        self._torch, self._functional, auto_tokenizer, auto_model = _import_transformer_dependencies()
        self.device = self._resolve_device(device)
        self.logger.info("Loading transformer model: %s", self.model_name)
        self.tokenizer = auto_tokenizer.from_pretrained(self.model_name, token=self.huggingface_token)
        model_kwargs: dict[str, Any] = {"token": self.huggingface_token}
        if self.num_labels is not None:
            model_kwargs["num_labels"] = self.num_labels
        self.model = auto_model.from_pretrained(self.model_name, **model_kwargs)
        self.model.to(self.device)
        self.logger.info("Transformer model loaded on device: %s", self.device)

    def _infer_num_labels(self) -> int:
        id2label = getattr(getattr(self.model, "config", None), "id2label", None)
        if id2label:
            return len(id2label)
        raise ValueError("num_labels is required when the model config has no id2label mapping.")

    def _id2label(self) -> dict[int, str]:
        id2label = getattr(getattr(self.model, "config", None), "id2label", None)
        if not id2label:
            return {i: str(i) for i in range(int(self.num_labels))}
        return {int(label_id): str(label) for label_id, label in id2label.items()}

    def _validate_label_map(self) -> None:
        labels = set(self._id2label().values())
        missing = sorted(label for label in labels if label not in self.label_map)
        if missing:
            raise ValueError(
                "label_map is missing model labels: "
                f"{missing}. Available labels are {sorted(labels)}."
            )

    def _label_direction(self, label: str) -> float:
        try:
            return self.label_map[str(label)]
        except KeyError as exc:
            raise ValueError(f"label_map has no direction for predicted label: {label}") from exc

    @staticmethod
    def _direction_name(direction: float) -> str:
        if direction > 0:
            return "positive"
        if direction < 0:
            return "negative"
        return "neutral"

    def _add_harmonized_sentence_outputs(self, df_score: pd.DataFrame, id2label: dict[int, str]) -> pd.DataFrame:
        """Add positive/neutral/negative count, share and net sentiment columns.

        These columns make outputs comparable across models with different raw
        label names, including generic labels such as ``LABEL_0``.
        """
        harmonized = {
            "positive": [],
            "neutral": [],
            "negative": [],
        }
        for label in id2label.values():
            raw_count_col = f"{self.model_name_short}_{label}"
            if raw_count_col not in df_score.columns:
                continue
            direction_name = self._direction_name(self._label_direction(label))
            harmonized[direction_name].append(raw_count_col)

        for direction_name, columns in harmonized.items():
            count_col = f"{self.model_name_short}_count_{direction_name}"
            if columns:
                df_score[count_col] = df_score[columns].sum(axis=1)
            else:
                df_score[count_col] = 0

        count_cols = [
            f"{self.model_name_short}_count_positive",
            f"{self.model_name_short}_count_neutral",
            f"{self.model_name_short}_count_negative",
        ]
        denominator = df_score[count_cols].sum(axis=1).replace(0, np.nan)
        for direction_name in ("positive", "neutral", "negative"):
            count_col = f"{self.model_name_short}_count_{direction_name}"
            share_col = f"{self.model_name_short}_share_{direction_name}"
            df_score[share_col] = (df_score[count_col] / denominator).fillna(0)

        positive_share = df_score[f"{self.model_name_short}_share_positive"]
        negative_share = df_score[f"{self.model_name_short}_share_negative"]
        net_sentiment_formula = getattr(self, "net_sentiment_formula", "positive_minus_negative")
        if net_sentiment_formula == "negative_minus_positive":
            df_score[f"{self.model_name_short}_net_sentiment"] = negative_share - positive_share
        elif net_sentiment_formula == "positive_minus_negative":
            df_score[f"{self.model_name_short}_net_sentiment"] = positive_share - negative_share
        else:
            raise ValueError(
                "net_sentiment_formula must be either 'positive_minus_negative' "
                "or 'negative_minus_positive'."
            )
        return df_score

    def analyze_sentiment_single(
        self,
        texts: pd.Series | list[str],
        return_probabilities: bool = True,
    ) -> pd.DataFrame:
        """Run batched model inference for a list or Series of texts."""
        if isinstance(texts, pd.Series):
            texts = texts.fillna("").astype(str).tolist()
        else:
            texts = ["" if text is None else str(text) for text in texts]

        self.model.eval()
        all_probs: list[np.ndarray] = []

        for start in tqdm(range(0, len(texts), self.batch_size), desc=f"Transformer Sentiment ({self.model_name_short})"):
            batch_texts = texts[start : start + self.batch_size]
            encoded = self.tokenizer(
                batch_texts,
                max_length=self.max_length,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            inputs = {key: value.to(self.device) for key, value in encoded.items()}
            with self._torch.no_grad():
                outputs = self.model(**inputs)
                probs = self._functional.softmax(outputs.logits, dim=1)
            all_probs.extend(probs.detach().cpu().numpy())

            if getattr(self.device, "type", None) == "cuda":
                self._torch.cuda.empty_cache()

        return self._postprocess_predictions(all_probs, return_probabilities=return_probabilities)

    def _postprocess_predictions(
        self,
        all_probs: list[np.ndarray],
        return_probabilities: bool = True,
    ) -> pd.DataFrame:
        """Convert model probabilities into labels and directional scores."""
        df_probs = pd.DataFrame(all_probs)
        if df_probs.empty:
            return pd.DataFrame()

        id2label = self._id2label()
        predicted_labels = df_probs.idxmax(axis=1).astype(int)
        result: dict[str, Any] = {f"{self.model_name_short}_predicted_label": predicted_labels}

        for label_id in range(df_probs.shape[1]):
            result[f"{self.model_name_short}_probability_{label_id}"] = df_probs[label_id]

        result_df = pd.DataFrame(result)
        label_col = f"{self.model_name_short}_label"
        predicted_col = f"{self.model_name_short}_predicted_label"
        direction_col = f"{self.model_name_short}_label_sentiment"
        score_col = f"{self.model_name_short}_sentiment_byalltext"

        result_df[label_col] = result_df[predicted_col].map(id2label)
        result_df[direction_col] = result_df[label_col].map(self._label_direction)
        result_df[score_col] = result_df.apply(
            lambda row: row[direction_col] * row[f"{self.model_name_short}_probability_{int(row[predicted_col])}"],
            axis=1,
        )
        if not return_probabilities:
            result_df = result_df.drop(
                columns=[col for col in result_df.columns if "_probability_" in col],
                errors="ignore",
            )
        return result_df

    def analyze_sentiment(self) -> pd.DataFrame:
        """Analyze sentiment for ``self.input_df`` and return row-level labels."""
        if self.input_df is None or self.text_column is None:
            raise ValueError("Input DataFrame and text column must be set before analysis.")
        if self.text_column not in self.input_df.columns:
            raise ValueError(f"Text column '{self.text_column}' not found in input DataFrame.")

        predictions = self.analyze_sentiment_single(self.input_df[self.text_column])
        self.df_labels = pd.concat([self.input_df.reset_index(drop=True), predictions], axis=1)
        return self.df_labels

    def sentiment_bysentence(
        self,
        sentence_probability_cutoff: float = 0.7,
        sentence_probability_aggregation: str = "cutoff",
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Aggregate sentence-row classifications back to ``id_text``."""
        if self.df_labels is None:
            self.analyze_sentiment()
        assert self.df_labels is not None

        if "id_text" not in self.df_labels.columns:
            raise ValueError("Sentence-level aggregation requires an 'id_text' column.")

        agg_mode = sentence_probability_aggregation.lower()
        if agg_mode not in ("cutoff", "mean"):
            raise ValueError("sentence_probability_aggregation must be either 'cutoff' or 'mean'.")

        if agg_mode == "mean" and sentence_probability_cutoff != 0.7:
            logger.debug(
                "sentence_probability_cutoff (%s) is ignored when sentence_probability_aggregation='mean'",
                sentence_probability_cutoff,
            )

        id2label = self._id2label()
        probability_cols = [col for col in self.df_labels.columns if col.startswith(f"{self.model_name_short}_probability_")]
        df_prob = self.df_labels.set_index("id_text")[probability_cols]
        if agg_mode == "mean":
            df_score = df_prob.groupby("id_text").mean()
        else:
            df_score = df_prob.ge(sentence_probability_cutoff).astype(int).groupby("id_text").sum()
        rename_map = {
            f"{self.model_name_short}_probability_{label_id}": f"{self.model_name_short}_{label}"
            for label_id, label in id2label.items()
        }
        df_score = df_score.rename(columns=rename_map)
        df_sentence_prob = df_prob.rename(columns=rename_map)

        weighted_columns = []
        for label in id2label.values():
            col = f"{self.model_name_short}_{label}"
            if col in df_score.columns:
                direction = self._label_direction(label)
                weighted_col = f"__weighted_{label}"
                df_score[weighted_col] = df_score[col] * direction
                weighted_columns.append(weighted_col)

        count_cols = [f"{self.model_name_short}_{label}" for label in id2label.values() if f"{self.model_name_short}_{label}" in df_score.columns]
        denominator = df_score[count_cols].sum(axis=1).replace(0, np.nan)
        numerator = df_score[weighted_columns].sum(axis=1) if weighted_columns else 0
        score_column = (
            f"{self.model_name_short}_sentiment_bysentence_mean"
            if agg_mode == "mean"
            else f"{self.model_name_short}_sentiment_bysentence"
        )
        df_score[score_column] = (numerator / denominator).fillna(0)
        if getattr(self, "output_schema", None) == "shares":
            df_score = self._add_harmonized_sentence_outputs(df_score, id2label)
        df_score = df_score.drop(columns=weighted_columns, errors="ignore")
        label_prefix = (
            f"{self.model_name_short}_meanprobability_"
            if agg_mode == "mean"
            else f"{self.model_name_short}_countsentence_"
        )
        df_score = df_score.rename(
            columns={
                f"{self.model_name_short}_{label}": f"{label_prefix}{label}"
                for label in id2label.values()
            }
        )
        return df_score, df_sentence_prob

    def sentiment_pipeline(
        self,
        aggregation: str = "byalltext",
        sentence_probability_cutoff: float = 0.7,
        sentence_probability_aggregation: str = "cutoff",
    ) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
        """Run transformer sentiment and return document or sentence aggregates."""
        aggregation = aggregation.lower()
        self.analyze_sentiment()
        if aggregation == "byalltext":
            assert self.df_labels is not None
            self.df_sentiment_output = self.df_labels
            return self.df_sentiment_output
        if aggregation == "bysentence":
            df_agg, df_sentence_probabilities = self.sentiment_bysentence(
                sentence_probability_cutoff=sentence_probability_cutoff,
                sentence_probability_aggregation=sentence_probability_aggregation,
            )
            self.df_sentiment_output = df_agg
            self.df_sentence_probabilities = df_sentence_probabilities
            return df_agg, df_sentence_probabilities
        raise ValueError("aggregation must be either 'byalltext' or 'bysentence'.")

    def clear_gpu_memory(self) -> None:
        """Explicitly clear GPU memory when CUDA is active."""
        if self._torch is not None and getattr(self.device, "type", None) == "cuda":
            self._torch.cuda.empty_cache()

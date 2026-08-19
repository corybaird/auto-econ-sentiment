import numpy as np
import pandas as pd
import pytest

from auto_econ_sentiment.models.sentiment_transformers import SentimentTransformers
from auto_econ_sentiment.pipeline import AutoEconSentiment


class _FakeConfig:
    id2label = {0: "negative", 1: "neutral", 2: "positive"}


class _FakeModel:
    config = _FakeConfig()


def _transformer_shell() -> SentimentTransformers:
    analyzer = SentimentTransformers.__new__(SentimentTransformers)
    analyzer.model_name_short = "fake"
    analyzer.model = _FakeModel()
    analyzer.num_labels = 3
    analyzer.label_map = {"negative": -1.0, "neutral": 0.0, "positive": 1.0}
    analyzer.output_schema = None
    analyzer.net_sentiment_formula = "positive_minus_negative"
    analyzer.df_labels = None
    analyzer.df_sentence_probabilities = None
    analyzer.df_sentiment_output = None
    return analyzer


def _labels_frame(rows: list[tuple[str, float, float, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id_text": [row[0] for row in rows],
            "fake_probability_0": [row[1] for row in rows],
            "fake_probability_1": [row[2] for row in rows],
            "fake_probability_2": [row[3] for row in rows],
        }
    )


def test_sentence_aggregation_counts_every_confident_sentence():
    analyzer = _transformer_shell()
    analyzer.df_labels = _labels_frame(
        [
            ("a", 0.9, 0.05, 0.05),
            ("a", 0.85, 0.10, 0.05),
            ("a", 0.05, 0.05, 0.90),
            ("b", 0.05, 0.10, 0.85),
        ]
    )

    df_score, _ = analyzer.sentiment_bysentence(sentence_probability_cutoff=0.7)

    assert df_score.loc["a", "fake_countsentence_negative"] == 2
    assert df_score.loc["a", "fake_countsentence_positive"] == 1
    assert df_score.loc["a", "fake_sentiment_bysentence"] == pytest.approx(-1 / 3)
    assert df_score.loc["b", "fake_sentiment_bysentence"] == pytest.approx(1.0)


def test_sentence_counts_exceed_one_per_document_when_input_is_exploded():
    """Regression: unexploded input made every sentence count collapse to 0 or 1."""
    analyzer = _transformer_shell()
    analyzer.df_labels = _labels_frame(
        [
            ("a", 0.9, 0.05, 0.05),
            ("a", 0.88, 0.07, 0.05),
            ("a", 0.92, 0.04, 0.04),
        ]
    )

    df_score, _ = analyzer.sentiment_bysentence(sentence_probability_cutoff=0.7)

    assert df_score.loc["a", "fake_countsentence_negative"] == 3


def test_single_row_per_document_cannot_express_mixed_sentiment():
    """Documents scored as one row can only ever yield a single unit count."""
    analyzer = _transformer_shell()
    analyzer.df_labels = _labels_frame([("a", 0.9, 0.05, 0.05), ("b", 0.05, 0.05, 0.9)])

    df_score, _ = analyzer.sentiment_bysentence(sentence_probability_cutoff=0.7)

    count_cols = [col for col in df_score.columns if "countsentence" in col]
    assert df_score[count_cols].sum(axis=1).tolist() == [1, 1]


def test_sentences_below_cutoff_are_not_counted():
    analyzer = _transformer_shell()
    analyzer.df_labels = _labels_frame([("a", 0.5, 0.3, 0.2), ("a", 0.9, 0.05, 0.05)])

    df_score, _ = analyzer.sentiment_bysentence(sentence_probability_cutoff=0.7)

    assert df_score.loc["a", "fake_countsentence_negative"] == 1


def _base_transformer_config(**overrides) -> dict:
    config = {
        "enabled": True,
        "text_column_transformer": "text_clean",
        "aggregation_methods": ["bysentence"],
        "models": [
            {
                "name": "gtfintechlab/FOMC-RoBERTa",
                "short_name": "fomc",
                "label_mapping": {"LABEL_0": "positive", "LABEL_1": "negative"},
                "sentiment_values": {"positive": 1, "negative": -1},
            }
        ],
    }
    config.update(overrides)
    return config


def test_sentence_settings_default_when_absent_from_config():
    expanded = AutoEconSentiment._expand_transformer_model_configs(
        transformer_config=_base_transformer_config(),
        default_text_column="text_clean",
    )

    assert expanded[0]["min_sentence_chars"] == 20
    assert expanded[0]["sentence_probability_cutoff"] == 0.7


def test_sentence_settings_read_from_transformer_config():
    expanded = AutoEconSentiment._expand_transformer_model_configs(
        transformer_config=_base_transformer_config(
            min_sentence_chars=40,
            sentence_probability_cutoff=0.55,
        ),
        default_text_column="text_clean",
    )

    assert expanded[0]["min_sentence_chars"] == 40
    assert expanded[0]["sentence_probability_cutoff"] == 0.55


def test_per_model_sentence_settings_override_the_top_level_defaults():
    config = _base_transformer_config(min_sentence_chars=40, sentence_probability_cutoff=0.55)
    config["models"][0]["min_sentence_chars"] = 15
    config["models"][0]["sentence_probability_cutoff"] = 0.8

    expanded = AutoEconSentiment._expand_transformer_model_configs(
        transformer_config=config,
        default_text_column="text_clean",
    )

    assert expanded[0]["min_sentence_chars"] == 15
    assert expanded[0]["sentence_probability_cutoff"] == 0.8

import numpy as np
import pandas as pd
import pytest

from auto_econ_sentiment.models.sentiment_transformers import SentimentTransformers


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
    analyzer.df_labels = None
    analyzer.df_sentence_probabilities = None
    analyzer.df_sentiment_output = None
    return analyzer


def test_transformer_module_imports_without_optional_dependencies():
    import auto_econ_sentiment.models.sentiment_transformers as sentiment_transformers

    assert sentiment_transformers.SentimentTransformers is SentimentTransformers


def test_transformer_postprocess_predictions_uses_explicit_label_map():
    analyzer = _transformer_shell()
    result = analyzer._postprocess_predictions(
        [
            np.array([0.8, 0.1, 0.1]),
            np.array([0.1, 0.2, 0.7]),
            np.array([0.2, 0.6, 0.2]),
        ]
    )

    assert result["fake_label"].tolist() == ["negative", "positive", "neutral"]
    assert result["fake_label_sentiment"].tolist() == [-1.0, 1.0, 0.0]
    assert result["fake_sentiment_byalltext"].tolist() == [-0.8, 0.7, 0.0]


def test_transformer_validation_requires_all_model_labels():
    analyzer = _transformer_shell()
    analyzer.label_map = {"negative": -1.0, "positive": 1.0}

    with pytest.raises(ValueError, match="missing model labels"):
        analyzer._validate_label_map()


def test_transformer_sentence_aggregation_counts_confident_labels():
    analyzer = _transformer_shell()
    analyzer.df_labels = pd.DataFrame(
        {
            "id_text": [1, 1, 2],
            "fake_probability_0": [0.8, 0.1, 0.1],
            "fake_probability_1": [0.1, 0.1, 0.1],
            "fake_probability_2": [0.1, 0.9, 0.8],
        }
    )

    scores, probabilities = analyzer.sentiment_bysentence(sentence_probability_cutoff=0.7)

    assert scores.loc[1, "fake_countsentence_negative"] == 1
    assert scores.loc[1, "fake_countsentence_positive"] == 1
    assert scores.loc[1, "fake_sentiment_bysentence"] == 0
    assert scores.loc[2, "fake_countsentence_positive"] == 1
    assert scores.loc[2, "fake_sentiment_bysentence"] == 1
    assert "fake_positive" in probabilities.columns

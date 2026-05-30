import pandas as pd
import pytest

from auto_econ_sentiment.models.sentiment_transformers import SentimentTransformers


class _DummyConfig:
    id2label = {
        0: "LABEL_0",
        1: "LABEL_1",
        2: "LABEL_2",
    }


class _DummyModel:
    config = _DummyConfig()


def _transformer_without_dependencies() -> SentimentTransformers:
    transformer = SentimentTransformers.__new__(SentimentTransformers)
    transformer.model_name_short = "toy"
    transformer.label_map = {
        "LABEL_0": 1.0,
        "LABEL_1": -1.0,
        "LABEL_2": 0.0,
    }
    transformer.output_schema = "shares"
    transformer.net_sentiment_formula = "positive_minus_negative"
    transformer.model = _DummyModel()
    transformer.num_labels = 3
    transformer.df_labels = pd.DataFrame(
        {
            "id_text": [1, 1, 1, 2, 2],
            "toy_probability_0": [0.8, 0.1, 0.1, 0.9, 0.2],
            "toy_probability_1": [0.1, 0.8, 0.1, 0.0, 0.2],
            "toy_probability_2": [0.1, 0.1, 0.8, 0.1, 0.8],
        }
    )
    return transformer


def test_transformer_sentence_outputs_include_harmonized_shares():
    transformer = _transformer_without_dependencies()

    scores, probabilities = transformer.sentiment_bysentence(sentence_probability_cutoff=0.7)

    assert list(probabilities.columns) == ["toy_LABEL_0", "toy_LABEL_1", "toy_LABEL_2"]
    assert scores.loc[1, "toy_count_positive"] == 1
    assert scores.loc[1, "toy_count_negative"] == 1
    assert scores.loc[1, "toy_count_neutral"] == 1
    assert scores.loc[1, "toy_share_positive"] == pytest.approx(1 / 3)
    assert scores.loc[1, "toy_share_negative"] == pytest.approx(1 / 3)
    assert scores.loc[1, "toy_net_sentiment"] == pytest.approx(0)
    assert scores.loc[2, "toy_net_sentiment"] == pytest.approx(0.5)


def test_transformer_net_sentiment_formula_can_reverse_sign():
    transformer = _transformer_without_dependencies()
    transformer.net_sentiment_formula = "negative_minus_positive"

    scores, _ = transformer.sentiment_bysentence(sentence_probability_cutoff=0.7)

    assert scores.loc[1, "toy_net_sentiment"] == pytest.approx(0)
    assert scores.loc[2, "toy_net_sentiment"] == pytest.approx(-0.5)


def test_transformer_unknown_net_sentiment_formula_raises():
    transformer = _transformer_without_dependencies()
    transformer.net_sentiment_formula = "mystery_formula"

    with pytest.raises(ValueError, match="net_sentiment_formula"):
        transformer.sentiment_bysentence(sentence_probability_cutoff=0.7)

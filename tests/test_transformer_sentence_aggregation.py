import logging
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


def test_sentence_aggregation_mean_mode_arithmetic_and_column_names():
    analyzer = _transformer_shell()
    analyzer.df_labels = _labels_frame(
        [
            ("a", 0.9, 0.05, 0.05),
            ("a", 0.85, 0.10, 0.05),
            ("a", 0.05, 0.05, 0.90),
            ("b", 0.05, 0.10, 0.85),
        ]
    )

    df_score, _ = analyzer.sentiment_bysentence(sentence_probability_aggregation="mean")

    # Hand-computed expected values:
    # Doc "a":
    #   neg_mean = (0.9 + 0.85 + 0.05) / 3 = 1.80 / 3 = 0.6
    #   neu_mean = (0.05 + 0.10 + 0.05) / 3 = 0.20 / 3 = 1/15
    #   pos_mean = (0.05 + 0.05 + 0.90) / 3 = 1.00 / 3 = 1/3
    #   sentiment = (pos_mean * 1.0 + neg_mean * (-1.0)) / (neg_mean + neu_mean + pos_mean)
    #             = (1/3 - 0.6) / 1.0 = -4/15
    # Doc "b":
    #   neg_mean = 0.05, neu_mean = 0.10, pos_mean = 0.85
    #   sentiment = 0.85 * 1.0 + 0.05 * (-1.0) = 0.80
    assert df_score.loc["a", "fake_meanprobability_negative"] == pytest.approx(0.6)
    assert df_score.loc["a", "fake_meanprobability_neutral"] == pytest.approx(0.2 / 3)
    assert df_score.loc["a", "fake_meanprobability_positive"] == pytest.approx(1.0 / 3)
    assert df_score.loc["a", "fake_sentiment_bysentence_mean"] == pytest.approx(-4 / 15)

    assert df_score.loc["b", "fake_meanprobability_negative"] == pytest.approx(0.05)
    assert df_score.loc["b", "fake_meanprobability_neutral"] == pytest.approx(0.10)
    assert df_score.loc["b", "fake_meanprobability_positive"] == pytest.approx(0.85)
    assert df_score.loc["b", "fake_sentiment_bysentence_mean"] == pytest.approx(0.80)

    # Column names check
    expected_cols = {
        "fake_meanprobability_negative",
        "fake_meanprobability_neutral",
        "fake_meanprobability_positive",
        "fake_sentiment_bysentence_mean",
    }
    assert set(df_score.columns) == expected_cols
    assert not any("countsentence" in col for col in df_score.columns)
    assert "fake_sentiment_bysentence" not in df_score.columns


def test_sentences_below_cutoff_distinguish_cutoff_and_mean_modes():
    """Sentences all just below cutoff yield 0 in cutoff mode but non-zero lean in mean mode."""
    analyzer = _transformer_shell()
    analyzer.df_labels = _labels_frame(
        [
            ("a", 0.10, 0.30, 0.60),
            ("a", 0.05, 0.35, 0.60),
        ]
    )

    # Cutoff mode (cutoff=0.7): no sentence meets threshold -> counts and score are 0
    df_cutoff, _ = analyzer.sentiment_bysentence(
        sentence_probability_cutoff=0.7,
        sentence_probability_aggregation="cutoff",
    )
    assert df_cutoff.loc["a", "fake_countsentence_negative"] == 0
    assert df_cutoff.loc["a", "fake_countsentence_neutral"] == 0
    assert df_cutoff.loc["a", "fake_countsentence_positive"] == 0
    assert df_cutoff.loc["a", "fake_sentiment_bysentence"] == 0.0

    # Mean mode: raw probabilities are averaged -> retains the positive lean (0.525)
    # Hand computation: neg_mean = 0.075, neu_mean = 0.325, pos_mean = 0.60
    # score = (0.60 * 1.0 - 0.075 * 1.0) / 1.0 = 0.525
    df_mean, _ = analyzer.sentiment_bysentence(
        sentence_probability_aggregation="mean",
    )
    assert df_mean.loc["a", "fake_meanprobability_negative"] == pytest.approx(0.075)
    assert df_mean.loc["a", "fake_meanprobability_neutral"] == pytest.approx(0.325)
    assert df_mean.loc["a", "fake_meanprobability_positive"] == pytest.approx(0.60)
    assert df_mean.loc["a", "fake_sentiment_bysentence_mean"] == pytest.approx(0.525)


def test_mean_mode_with_harmonized_shares_output_schema():
    analyzer = _transformer_shell()
    analyzer.output_schema = "shares"
    analyzer.df_labels = _labels_frame(
        [
            ("a", 0.10, 0.30, 0.60),
            ("a", 0.05, 0.35, 0.60),
        ]
    )

    df_score, _ = analyzer.sentiment_bysentence(sentence_probability_aggregation="mean")

    # Means sum to 1.0, so count_* == share_*
    assert df_score.loc["a", "fake_count_positive"] == pytest.approx(0.60)
    assert df_score.loc["a", "fake_count_neutral"] == pytest.approx(0.325)
    assert df_score.loc["a", "fake_count_negative"] == pytest.approx(0.075)
    assert df_score.loc["a", "fake_share_positive"] == pytest.approx(0.60)
    assert df_score.loc["a", "fake_share_neutral"] == pytest.approx(0.325)
    assert df_score.loc["a", "fake_share_negative"] == pytest.approx(0.075)
    assert df_score.loc["a", "fake_net_sentiment"] == pytest.approx(0.525)
    assert df_score.loc["a", "fake_sentiment_bysentence_mean"] == pytest.approx(0.525)
    assert df_score.loc["a", "fake_meanprobability_positive"] == pytest.approx(0.60)
    assert df_score.loc["a", "fake_meanprobability_negative"] == pytest.approx(0.075)
    assert df_score.loc["a", "fake_meanprobability_neutral"] == pytest.approx(0.325)


def test_invalid_sentence_probability_aggregation_raises():
    analyzer = _transformer_shell()
    analyzer.df_labels = _labels_frame([("a", 0.5, 0.3, 0.2)])

    with pytest.raises(ValueError, match="sentence_probability_aggregation"):
        analyzer.sentiment_bysentence(sentence_probability_aggregation="median")


def test_mean_mode_logs_debug_on_non_default_cutoff(caplog):
    analyzer = _transformer_shell()
    analyzer.df_labels = _labels_frame([("a", 0.5, 0.3, 0.2)])

    with caplog.at_level(logging.DEBUG):
        analyzer.sentiment_bysentence(
            sentence_probability_cutoff=0.85,
            sentence_probability_aggregation="mean",
        )

    assert "ignored" in caplog.text


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
    assert expanded[0]["sentence_probability_aggregation"] == "cutoff"


def test_sentence_settings_read_from_transformer_config():
    expanded = AutoEconSentiment._expand_transformer_model_configs(
        transformer_config=_base_transformer_config(
            min_sentence_chars=40,
            sentence_probability_cutoff=0.55,
            sentence_probability_aggregation="mean",
        ),
        default_text_column="text_clean",
    )

    assert expanded[0]["min_sentence_chars"] == 40
    assert expanded[0]["sentence_probability_cutoff"] == 0.55
    assert expanded[0]["sentence_probability_aggregation"] == "mean"


def test_per_model_sentence_settings_override_the_top_level_defaults():
    config = _base_transformer_config(
        min_sentence_chars=40,
        sentence_probability_cutoff=0.55,
        sentence_probability_aggregation="cutoff",
    )
    config["models"][0]["min_sentence_chars"] = 15
    config["models"][0]["sentence_probability_cutoff"] = 0.8
    config["models"][0]["sentence_probability_aggregation"] = "mean"

    expanded = AutoEconSentiment._expand_transformer_model_configs(
        transformer_config=config,
        default_text_column="text_clean",
    )

    assert expanded[0]["min_sentence_chars"] == 15
    assert expanded[0]["sentence_probability_cutoff"] == 0.8
    assert expanded[0]["sentence_probability_aggregation"] == "mean"

import pytest

from auto_econ_sentiment.exceptions import SentimentAnalysisError
from auto_econ_sentiment.pipeline import AutoEconSentiment


def test_original_style_transformer_config_expands_to_internal_schema():
    config = {
        "enabled": True,
        "text_column_transformer": "text_clean",
        "aggregation_methods": ["sentence_pos", "full_text"],
        "output_schema": "shares",
        "models": [
            {
                "name": "gtfintechlab/FOMC-RoBERTa",
                "short_name": "fomc",
                "num_labels": 3,
                "label_mapping": {
                    "LABEL_0": "positive",
                    "LABEL_1": "negative",
                    "LABEL_2": "neutral",
                },
                "sentiment_values": {
                    "positive": 1,
                    "negative": -1,
                    "neutral": 0,
                },
            }
        ],
    }

    expanded = AutoEconSentiment._expand_transformer_model_configs(
        transformer_config=config,
        default_text_column="text_clean",
    )

    assert [model["aggregation"] for model in expanded] == ["bysentence", "byalltext"]
    assert expanded[0]["model_name"] == "gtfintechlab/FOMC-RoBERTa"
    assert expanded[0]["model_name_short"] == "fomc"
    assert expanded[0]["label_map"] == {"LABEL_0": 1, "LABEL_1": -1, "LABEL_2": 0}
    assert expanded[0]["output_schema"] == "shares"


def test_current_style_transformer_config_still_works():
    config = {
        "enabled": True,
        "models": [
            {
                "model_name": "ProsusAI/finbert",
                "model_name_short": "finbertpro",
                "num_labels": 3,
                "aggregation": "bysentence",
                "label_map": {
                    "positive": 1,
                    "negative": -1,
                    "neutral": 0,
                },
            }
        ],
    }

    expanded = AutoEconSentiment._expand_transformer_model_configs(
        transformer_config=config,
        default_text_column="text_clean",
    )

    assert len(expanded) == 1
    assert expanded[0]["aggregation"] == "bysentence"
    assert expanded[0]["label_map"]["positive"] == 1


def test_original_style_transformer_config_validates_sentiment_values():
    config = {
        "enabled": True,
        "models": [
            {
                "name": "example/model",
                "short_name": "bad",
                "label_mapping": {
                    "LABEL_0": "positive",
                    "LABEL_1": "negative",
                },
                "sentiment_values": {
                    "positive": 1,
                },
            }
        ],
    }

    with pytest.raises(SentimentAnalysisError, match="missing semantic labels"):
        AutoEconSentiment._expand_transformer_model_configs(
            transformer_config=config,
            default_text_column="text_clean",
        )

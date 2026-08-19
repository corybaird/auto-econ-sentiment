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


def test_flat_lexical_key_is_resolved():
    config = {"lexical": {"dictionaries": {"unstemmed": ["lm"]}, "aggregation_methods": ["posneg"]}}

    assert AutoEconSentiment.resolve_lexical_config(config)["aggregation_methods"] == ["posneg"]


def test_nested_lexical_key_still_resolves_for_legacy_configs():
    config = {"models": {"lexical": {"aggregation_methods": ["allwords"]}}}

    assert AutoEconSentiment.resolve_lexical_config(config)["aggregation_methods"] == ["allwords"]


def test_flat_lexical_key_takes_precedence_over_nested():
    config = {
        "lexical": {"aggregation_methods": ["posneg"]},
        "models": {"lexical": {"aggregation_methods": ["allwords"]}},
    }

    assert AutoEconSentiment.resolve_lexical_config(config)["aggregation_methods"] == ["posneg"]


def test_missing_lexical_key_resolves_to_empty_dict():
    assert AutoEconSentiment.resolve_lexical_config({}) == {}


def test_flat_transformer_key_is_resolved():
    config = {"transformer": {"enabled": True, "models": [{"name": "x", "short_name": "x"}]}}

    assert AutoEconSentiment.resolve_transformer_config(config)["enabled"] is True


def test_nested_transformer_key_still_resolves_for_legacy_configs():
    config = {"models": {"transformer": {"enabled": True, "output_schema": "shares"}}}

    assert AutoEconSentiment.resolve_transformer_config(config)["output_schema"] == "shares"


def test_legacy_transformers_list_form_still_resolves():
    config = {"models": {"transformers": [{"name": "x", "short_name": "x"}]}}
    resolved = AutoEconSentiment.resolve_transformer_config(config)

    assert resolved["enabled"] is True
    assert resolved["models"] == [{"name": "x", "short_name": "x"}]


def test_missing_transformer_key_resolves_to_empty_dict():
    assert AutoEconSentiment.resolve_transformer_config({}) == {}

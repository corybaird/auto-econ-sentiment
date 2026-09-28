import os
import unittest.mock as mock
import numpy as np
import pandas as pd
import pytest


from auto_econ_sentiment.exceptions import SentimentAnalysisError
from auto_econ_sentiment.models.sentiment_llm import SentimentLLM
from auto_econ_sentiment.pipeline import AutoEconSentiment


def _mock_llm_scorer(**kwargs) -> SentimentLLM:
    """Helper to create a SentimentLLM instance with default or custom kwargs."""
    defaults = {
        "model_name": "test-model",
        "model_name_short": "testllm",
        "provider": "ollama",
        "output_scale": "continuous",
    }
    defaults.update(kwargs)
    return SentimentLLM(**defaults)



def test_prompt_formatting_includes_input_text():
    scorer = _mock_llm_scorer()
    input_text = "Interest rates were held constant following the committee meeting."
    formatted = scorer.format_prompt(input_text)

    assert input_text in formatted
    assert "polarity" in formatted
    assert "confidence" in formatted


def test_custom_prompt_formatting():
    custom_template = "Score sentiment for: {text}\nFormat: JSON."
    scorer = _mock_llm_scorer(prompt_template=custom_template)
    input_text = "Inflation declined rapidly."
    formatted = scorer.format_prompt(input_text)

    assert formatted == "Score sentiment for: Inflation declined rapidly.\nFormat: JSON."


def test_continuous_score_calculation():
    scorer = _mock_llm_scorer(output_scale="continuous")
    # {"polarity": -1, "confidence": 0.9} -> -0.9
    with mock.patch.object(scorer, "_call_provider", return_value='{"polarity": -1, "confidence": 0.9}'):
        df = pd.DataFrame({"id_text": [1], "text": ["Economic outlook worsened significantly."]})
        scorer.input_df = df
        scorer.text_column = "text"
        result = scorer.analyze_sentiment()

    assert result.loc[0, "testllm_polarity"] == -1
    assert result.loc[0, "testllm_confidence"] == pytest.approx(0.9)
    assert result.loc[0, "testllm_sentiment_byalltext"] == pytest.approx(-0.9)


def test_discrete_score_calculation():
    scorer = _mock_llm_scorer(output_scale="discrete")
    # For polarity -1, discrete scale maps to 0
    with mock.patch.object(scorer, "_call_provider", return_value='{"polarity": -1, "confidence": 0.9}'):
        df = pd.DataFrame({"id_text": [1], "text": ["Recession fears escalate."]})
        scorer.input_df = df
        scorer.text_column = "text"
        result = scorer.analyze_sentiment()

    assert result.loc[0, "testllm_polarity"] == -1
    assert result.loc[0, "testllm_sentiment_byalltext"] == 0


def test_discrete_score_neutral_and_positive():
    scorer = _mock_llm_scorer(output_scale="discrete")
    # Neutral (0) -> 1, Positive (1) -> 2
    responses = [
        '{"polarity": 0, "confidence": 0.8}',
        '{"polarity": 1, "confidence": 0.95}',
    ]
    with mock.patch.object(scorer, "_call_provider", side_effect=responses):
        df = pd.DataFrame({"id_text": [1, 2], "text": ["Conditions unchanged.", "Growth surged."]})
        scorer.input_df = df
        scorer.text_column = "text"
        result = scorer.analyze_sentiment()

    assert result.loc[0, "testllm_sentiment_byalltext"] == 1
    assert result.loc[1, "testllm_sentiment_byalltext"] == 2


def test_json_wrapped_in_prose_fallback():
    scorer = _mock_llm_scorer()
    prose_response = (
        "Here is the economic sentiment evaluation you requested:\n"
        '```json\n{"polarity": 1, "confidence": 0.85}\n```\n'
        "I hope this helps."
    )
    with mock.patch.object(scorer, "_call_provider", return_value=prose_response):
        df = pd.DataFrame({"id_text": [1], "text": ["Labor market remains strong."]})
        scorer.input_df = df
        scorer.text_column = "text"
        result = scorer.analyze_sentiment()

    assert result.loc[0, "testllm_polarity"] == 1
    assert result.loc[0, "testllm_confidence"] == pytest.approx(0.85)
    assert result.loc[0, "testllm_sentiment_byalltext"] == pytest.approx(0.85)


def test_json_plain_prose_without_markdown():
    scorer = _mock_llm_scorer()
    prose_response = "The output is: {\"polarity\": -1, \"confidence\": 0.75} based on current analysis."
    with mock.patch.object(scorer, "_call_provider", return_value=prose_response):
        df = pd.DataFrame({"id_text": [1], "text": ["Inflation spike seen."]})
        scorer.input_df = df
        scorer.text_column = "text"
        result = scorer.analyze_sentiment()

    assert result.loc[0, "testllm_polarity"] == -1
    assert result.loc[0, "testllm_confidence"] == pytest.approx(0.75)
    assert result.loc[0, "testllm_sentiment_byalltext"] == pytest.approx(-0.75)


def test_malformed_reply_logs_warning_and_records_nan_batch_survives(caplog):
    scorer = _mock_llm_scorer()
    responses = [
        "I cannot provide JSON because I am just a language model.",
        '{"polarity": 1, "confidence": 0.9}',
    ]
    with mock.patch.object(scorer, "_call_provider", side_effect=responses):
        df = pd.DataFrame({"id_text": [1, 2], "text": ["First text.", "Second text."]})
        scorer.input_df = df
        scorer.text_column = "text"
        result = scorer.analyze_sentiment()

    # Batch should survive, row 0 NaN, row 1 scored
    assert np.isnan(result.loc[0, "testllm_polarity"])
    assert np.isnan(result.loc[0, "testllm_confidence"])
    assert np.isnan(result.loc[0, "testllm_sentiment_byalltext"])

    assert result.loc[1, "testllm_polarity"] == 1
    assert result.loc[1, "testllm_confidence"] == pytest.approx(0.9)
    assert result.loc[1, "testllm_sentiment_byalltext"] == pytest.approx(0.9)


def test_out_of_range_polarity_rejected():
    scorer = _mock_llm_scorer()
    responses = [
        '{"polarity": 2, "confidence": 0.8}',
        '{"polarity": -2, "confidence": 0.8}',
        '{"polarity": "positive", "confidence": 0.8}',
    ]
    with mock.patch.object(scorer, "_call_provider", side_effect=responses):
        df = pd.DataFrame({"id_text": [1, 2, 3], "text": ["Text A", "Text B", "Text C"]})
        scorer.input_df = df
        scorer.text_column = "text"
        result = scorer.analyze_sentiment()

    assert np.isnan(result.loc[0, "testllm_polarity"])
    assert np.isnan(result.loc[1, "testllm_polarity"])
    assert np.isnan(result.loc[2, "testllm_polarity"])


def test_out_of_range_confidence_rejected():
    scorer = _mock_llm_scorer()
    responses = [
        '{"polarity": 1, "confidence": 1.5}',
        '{"polarity": 1, "confidence": -0.2}',
        '{"polarity": 1, "confidence": "high"}',
    ]
    with mock.patch.object(scorer, "_call_provider", side_effect=responses):
        df = pd.DataFrame({"id_text": [1, 2, 3], "text": ["Text A", "Text B", "Text C"]})
        scorer.input_df = df
        scorer.text_column = "text"
        result = scorer.analyze_sentiment()

    assert np.isnan(result.loc[0, "testllm_confidence"])
    assert np.isnan(result.loc[1, "testllm_confidence"])
    assert np.isnan(result.loc[2, "testllm_confidence"])


def test_confidence_cutoff_filtering():
    scorer = _mock_llm_scorer(confidence_cutoff=0.8)
    responses = [
        '{"polarity": 1, "confidence": 0.7}',  # Below cutoff -> NaN score
        '{"polarity": 1, "confidence": 0.85}', # Above cutoff -> 0.85
    ]
    with mock.patch.object(scorer, "_call_provider", side_effect=responses):
        df = pd.DataFrame({"id_text": [1, 2], "text": ["Low cert text", "High cert text"]})
        scorer.input_df = df
        scorer.text_column = "text"
        result = scorer.analyze_sentiment()

    assert result.loc[0, "testllm_polarity"] == 1
    assert result.loc[0, "testllm_confidence"] == pytest.approx(0.7)
    assert np.isnan(result.loc[0, "testllm_sentiment_byalltext"])

    assert result.loc[1, "testllm_polarity"] == 1
    assert result.loc[1, "testllm_confidence"] == pytest.approx(0.85)
    assert result.loc[1, "testllm_sentiment_byalltext"] == pytest.approx(0.85)


def test_column_naming_and_metadata():
    scorer = _mock_llm_scorer(
        model_name="meta-llama/Llama-3-8b",
        model_name_short="llama3",
        provider="ollama",
        temperature=0.2,
        prompt_version="v2",
    )
    with mock.patch.object(scorer, "_call_provider", return_value='{"polarity": 0, "confidence": 0.95}'):
        df = pd.DataFrame({"id_text": [1], "text": ["Market remained balanced."]})
        scorer.input_df = df
        scorer.text_column = "text"
        result = scorer.analyze_sentiment()

    expected_cols = [
        "id_text",
        "text",
        "llama3_polarity",
        "llama3_confidence",
        "llama3_sentiment_byalltext",
        "llama3_provider",
        "llama3_model",
        "llama3_prompt_version",
        "llama3_temperature",
    ]
    for col in expected_cols:
        assert col in result.columns

    assert result.loc[0, "llama3_provider"] == "ollama"
    assert result.loc[0, "llama3_model"] == "meta-llama/Llama-3-8b"
    assert result.loc[0, "llama3_prompt_version"] == "v2"
    assert result.loc[0, "llama3_temperature"] == 0.2


def test_ollama_request_builder():
    scorer = _mock_llm_scorer(
        model_name="llama3:8b",
        provider="ollama",
        base_url="http://custom-ollama:11434",
        temperature=0.1,
    )
    url, headers, payload = scorer._build_ollama_request("Test prompt")

    assert url == "http://custom-ollama:11434/api/generate"
    assert headers["Content-Type"] == "application/json"
    assert payload["model"] == "llama3:8b"
    assert payload["prompt"] == "Test prompt"
    assert payload["stream"] is False
    assert payload["format"] == "json"
    assert payload["options"]["temperature"] == 0.1


def test_openai_request_builder():
    with mock.patch.dict(os.environ, {"TEST_KEY_ENV": "sk-mock-key-12345"}):
        scorer = _mock_llm_scorer(
            model_name="gpt-4o-mini",
            provider="openai",
            api_key_env="TEST_KEY_ENV",
            base_url="https://api.openai.com/v1",
            temperature=0.0,
        )
        url, headers, payload = scorer._build_openai_request("Analyze this text")

    assert url == "https://api.openai.com/v1/chat/completions"
    assert headers["Content-Type"] == "application/json"
    assert headers["Authorization"] == "Bearer sk-mock-key-12345"
    assert payload["model"] == "gpt-4o-mini"
    assert payload["messages"] == [{"role": "user", "content": "Analyze this text"}]
    assert payload["temperature"] == 0.0
    assert payload["response_format"] == {"type": "json_object"}


def test_openrouter_request_builder():
    with mock.patch.dict(os.environ, {"OPENROUTER_API_KEY": "sk-or-mock-key"}):
        scorer = _mock_llm_scorer(
            model_name="anthropic/claude-3.5-sonnet",
            provider="openai",
            api_key_env="OPENROUTER_API_KEY",
            base_url="https://openrouter.ai/api/v1",
            temperature=0.0,
        )
        url, headers, payload = scorer._build_openai_request("Analyze this text")

    assert url == "https://openrouter.ai/api/v1/chat/completions"
    assert headers["Authorization"] == "Bearer sk-or-mock-key"
    assert payload["model"] == "anthropic/claude-3.5-sonnet"


def test_sentiment_bysentence_aggregation():
    scorer = _mock_llm_scorer(output_scale="continuous")
    scorer.df_labels = pd.DataFrame(
        {
            "id_text": [1, 1, 1, 2, 2],
            "testllm_polarity": [1.0, -1.0, 0.0, 1.0, 0.0],
            "testllm_confidence": [0.9, 0.85, 0.95, 0.9, 0.8],
        }
    )

    df_score, df_details = scorer.sentiment_bysentence(confidence_cutoff=0.7)

    assert df_score.loc[1, "testllm_count_positive"] == 1
    assert df_score.loc[1, "testllm_count_negative"] == 1
    assert df_score.loc[1, "testllm_count_neutral"] == 1
    assert df_score.loc[1, "testllm_share_positive"] == pytest.approx(1 / 3)
    assert df_score.loc[1, "testllm_share_negative"] == pytest.approx(1 / 3)
    assert df_score.loc[1, "testllm_net_sentiment"] == pytest.approx(0.0)
    assert df_score.loc[1, "testllm_sentiment_bysentence"] == pytest.approx(0.0)

    assert df_score.loc[2, "testllm_count_positive"] == 1
    assert df_score.loc[2, "testllm_count_neutral"] == 1
    assert df_score.loc[2, "testllm_share_positive"] == pytest.approx(0.5)
    assert df_score.loc[2, "testllm_net_sentiment"] == pytest.approx(0.5)
    assert df_score.loc[2, "testllm_sentiment_bysentence"] == pytest.approx(0.5)


def test_sentiment_bysentence_net_sentiment_formula_reversal():
    scorer = _mock_llm_scorer(net_sentiment_formula="negative_minus_positive")
    scorer.df_labels = pd.DataFrame(
        {
            "id_text": [1, 2],
            "testllm_polarity": [1.0, -1.0],
            "testllm_confidence": [0.9, 0.9],
        }
    )

    df_score, _ = scorer.sentiment_bysentence(confidence_cutoff=0.7)

    # Positive sentence gives net sentiment = negative_share - positive_share = 0 - 1 = -1
    assert df_score.loc[1, "testllm_net_sentiment"] == pytest.approx(-1.0)
    # Negative sentence gives net sentiment = negative_share - positive_share = 1 - 0 = 1
    assert df_score.loc[2, "testllm_net_sentiment"] == pytest.approx(1.0)


def test_resolve_llm_config_flat_key():
    config = {
        "llm": {
            "enabled": True,
            "provider": "ollama",
            "models": [{"name": "llama3:8b", "short_name": "llama3"}],
        }
    }
    resolved = AutoEconSentiment.resolve_llm_config(config)
    assert resolved["enabled"] is True
    assert resolved["models"][0]["name"] == "llama3:8b"


def test_resolve_llm_config_nested_key():
    config = {
        "models": {
            "llm": {
                "enabled": True,
                "provider": "openai",
                "models": [{"name": "gpt-4o", "short_name": "gpt4"}],
            }
        }
    }
    resolved = AutoEconSentiment.resolve_llm_config(config)
    assert resolved["enabled"] is True
    assert resolved["provider"] == "openai"


def test_resolve_llm_config_legacy_list_form():
    config = {
        "models": {
            "llms": [{"name": "llama3:8b", "short_name": "llama3"}],
            "llms_config": {"provider": "ollama"},
        }
    }
    resolved = AutoEconSentiment.resolve_llm_config(config)
    assert resolved["enabled"] is True
    assert resolved["models"] == [{"name": "llama3:8b", "short_name": "llama3"}]


def test_resolve_llm_config_missing_returns_empty_dict():
    assert AutoEconSentiment.resolve_llm_config({}) == {}


def test_expand_llm_model_configs():
    config = {
        "enabled": True,
        "text_column_llm": "text_clean",
        "aggregation_methods": ["full_text", "sentence_pos"],
        "provider": "ollama",
        "models": [
            {
                "name": "llama3:8b",
                "short_name": "llama3",
            }
        ],
    }
    expanded = AutoEconSentiment._expand_llm_model_configs(
        llm_config=config,
        default_text_column="text_clean",
    )

    assert len(expanded) == 2
    assert [m["aggregation"] for m in expanded] == ["byalltext", "bysentence"]
    assert expanded[0]["model_name"] == "llama3:8b"
    assert expanded[0]["model_name_short"] == "llama3"
    assert expanded[0]["provider"] == "ollama"


def test_expand_llm_model_configs_missing_model_raises():
    config = {"enabled": True, "models": []}
    with pytest.raises(SentimentAnalysisError, match="no models are configured"):
        AutoEconSentiment._expand_llm_model_configs(
            llm_config=config,
            default_text_column="text_clean",
        )


def test_pipeline_runs_llm_scoring_with_mocked_provider(tmp_path):
    df_raw = pd.DataFrame(
        {
            "id_text": [1, 2],
            "text": [
                "Economic activity expanded at a robust pace.",
                "Inflation pressures intensified significantly across sectors.",
            ],
            "date": ["2024-01-15", "2024-02-15"],
        }
    )
    raw_path = tmp_path / "raw.csv"
    df_raw.to_csv(raw_path, index=False)

    pipeline = AutoEconSentiment(
        import_file_path=raw_path,
        text_column="text",
        date_column="date",
        export_path=tmp_path / "exports",
    )

    llm_config = {
        "enabled": True,
        "text_column_llm": "text_clean",
        "aggregation": "byalltext",
        "models": [
            {
                "model_name": "mock-llm",
                "model_name_short": "mockm",
                "provider": "ollama",
            }
        ],
    }

    mock_responses = [
        '{"polarity": 1, "confidence": 0.9}',
        '{"polarity": -1, "confidence": 0.85}',
    ]

    with mock.patch("auto_econ_sentiment.models.sentiment_llm.SentimentLLM._call_provider", side_effect=mock_responses):
        pipeline.load_data()
        pipeline.clean_data(clean_config={"clean_html": False, "tokenize": False, "stem": False})
        df_sent = pipeline.analyze_sentiment_llm(llm_config)

    assert "mockm_sentiment_byalltext" in df_sent.columns
    assert df_sent.loc[1, "mockm_sentiment_byalltext"] == pytest.approx(0.9)
    assert df_sent.loc[2, "mockm_sentiment_byalltext"] == pytest.approx(-0.85)


@pytest.mark.llm
@pytest.mark.skip(reason="Live LLM tests disabled by default; requires local Ollama service")
def test_live_ollama_integration():
    """Optional live integration test touching local Ollama instance."""
    scorer = SentimentLLM(
        model_name="llama3:8b",
        model_name_short="llama3",
        provider="ollama",
        output_scale="continuous",
    )
    df = pd.DataFrame({"id_text": [1], "text": ["Economic expansion continues at a steady pace."]})
    scorer.input_df = df
    scorer.text_column = "text"
    result = scorer.analyze_sentiment()

    assert not result.empty
    assert "llama3_sentiment_byalltext" in result.columns
    assert not np.isnan(result.loc[0, "llama3_sentiment_byalltext"])

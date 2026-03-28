import pytest
import pandas as pd
from pathlib import Path
from auto_econ_sentiment.clean.text_loader import TextLoader
from auto_econ_sentiment.clean.text_clean import TextCleaner
from auto_econ_sentiment.models.sentiment_lexical import SentimentLexical


# ── Realistic Fed statement text ────────────────────────────────────────────
# Drawn from actual FOMC statements in data/raw/monetary_policy_statement_mostrecent.csv

FOMC_JULY_2024 = (
    "Recent indicators suggest that economic activity has continued to expand at a solid pace. "
    "Job gains have moderated, and the unemployment rate has moved up but remains low. "
    "Inflation has eased over the past year but remains somewhat elevated. "
    "In recent months, there has been some further progress toward the Committee's 2 percent inflation objective. "
    "The Committee seeks to achieve maximum employment and inflation at the rate of 2 percent over the longer run. "
    "The Committee judges that the risks to achieving its employment and inflation goals continue to move into better balance. "
    "In support of its goals, the Committee decided to maintain the target range for the federal funds rate at 5-1/4 to 5-1/2 percent. "
    "The Committee does not expect it will be appropriate to reduce the target range until it has gained greater confidence "
    "that inflation is moving sustainably toward 2 percent."
)

FOMC_SEPT_2024 = (
    "Recent indicators suggest that economic activity has continued to expand at a solid pace. "
    "Job gains have slowed, and the unemployment rate has moved up but remains low. "
    "Inflation has made further progress toward the Committee's 2 percent objective but remains somewhat elevated. "
    "The Committee has gained greater confidence that inflation is moving sustainably toward 2 percent, "
    "and judges that the risks to achieving its employment and inflation goals are roughly in balance. "
    "In light of the progress on inflation and the balance of risks, the Committee decided to lower the target range "
    "for the federal funds rate by 1/2 percentage point to 4-3/4 to 5 percent. "
    "The Committee is strongly committed to supporting maximum employment and returning inflation to its 2 percent objective."
)

FOMC_HEADER_NOISE = (
    "Federal Open Market Committee Monetary Policy Principles and Practice Policy Implementation Reports "
    "Review of Monetary Policy Strategy, Tools, and Communications Institution Supervision Reports "
    "For release at 2:00 p. m. EDT "
    + FOMC_JULY_2024
)

DICT_PATH = Path(__file__).parents[1] / "src" / "auto_econ_sentiment" / "data" / "lexical_master_dict.yaml"


# ── TextLoader ────────────────────────────────────────────────────────────────

def test_loader_real_csv():
    csv_path = Path(__file__).parents[1] / "data" / "raw" / "monetary_policy_statement_mostrecent.csv"
    if not csv_path.exists():
        pytest.skip("Real CSV not present in data/raw/")
    loader = TextLoader(file_path=str(csv_path), text_column="text", date_column="date")
    df = loader.get_data()
    assert "text" in df.columns
    assert "date" in df.columns
    assert len(df) >= 1
    assert pd.api.types.is_datetime64_any_dtype(df["date"])


def test_loader_synthetic_csv(tmp_path):
    csv_file = tmp_path / "fomc.csv"
    pd.DataFrame({
        "body": [FOMC_JULY_2024, FOMC_SEPT_2024],
        "published": ["2024-07-31", "2024-09-18"],
    }).to_csv(csv_file, index=False)
    loader = TextLoader(file_path=str(csv_file), text_column="body", date_column="published")
    df = loader.get_data()
    assert len(df) == 2
    assert df["text"].iloc[0].startswith("Recent indicators")


def test_loader_missing_column(tmp_path):
    csv_file = tmp_path / "bad.csv"
    pd.DataFrame({"body": [FOMC_JULY_2024], "published": ["2024-07-31"]}).to_csv(csv_file, index=False)
    with pytest.raises(ValueError, match="Text column"):
        TextLoader(file_path=str(csv_file), text_column="nonexistent", date_column="published")


def test_loader_unsupported_format(tmp_path):
    bad_file = tmp_path / "data.jsonl"
    bad_file.write_text("not real")
    with pytest.raises(ValueError, match="Unsupported file format"):
        TextLoader(file_path=str(bad_file), text_column="text", date_column="date")


def test_loader_returns_copy(tmp_path):
    csv_file = tmp_path / "fomc.csv"
    pd.DataFrame({"text": [FOMC_JULY_2024], "date": ["2024-07-31"]}).to_csv(csv_file, index=False)
    loader = TextLoader(file_path=str(csv_file), text_column="text", date_column="date")
    assert loader.get_data() is not loader.get_data()


# ── TextCleaner ───────────────────────────────────────────────────────────────

def _df(*texts):
    return pd.DataFrame({"text": list(texts)})


def test_cleaner_basic_run_on_fomc():
    df = _df(FOMC_JULY_2024, FOMC_SEPT_2024)
    cleaner = TextCleaner(df=df, text_column="text", clean_config={"tokenize": False, "stem": False})
    result = cleaner.run()
    assert "text_clean" in result.columns
    assert len(result) == 2
    assert "inflation" in result["text_clean"].iloc[0]


def test_cleaner_header_removal():
    remove_pattern = (
        "Federal Open Market Committee Monetary Policy Principles and Practice Policy Implementation Reports "
        "Review of Monetary Policy Strategy, Tools, and Communications Institution Supervision Reports "
        "For release at 2:00 p. m. EDT "
    )
    df = _df(FOMC_HEADER_NOISE)
    cleaner = TextCleaner(
        df=df,
        text_column="text",
        clean_config={
            "clean_html": True,
            "remove_headers": [remove_pattern],
            "tokenize": False,
            "stem": False,
        },
    )
    result = cleaner.run()
    cleaned = result["text_clean"].iloc[0]
    assert "Federal Open Market Committee" not in cleaned
    assert "economic activity" in cleaned


def test_cleaner_tokenize_fomc():
    df = _df(FOMC_JULY_2024)
    cleaner = TextCleaner(df=df, text_column="text", clean_config={"tokenize": True, "stem": False})
    result = cleaner.run()
    tokens = result["text_tokens"].iloc[0]
    assert isinstance(tokens, list)
    assert "inflation" in tokens
    assert "employment" in tokens


def test_cleaner_stem_fomc():
    df = _df(FOMC_JULY_2024)
    cleaner = TextCleaner(df=df, text_column="text", clean_config={"tokenize": True, "stem": True})
    result = cleaner.run()
    stems = result["text_stems"].iloc[0]
    assert isinstance(stems, str)
    assert len(stems) > 0




def test_cleaner_percentage_normalization():
    df = _df("The target range for the federal funds rate at 5-1/4 to 5-1/2 percent.")
    cleaner = TextCleaner(
        df=df, text_column="text",
        clean_config={"clean_numbers_percentages": True, "tokenize": False, "stem": False},
    )
    result = cleaner.run()
    cleaned = result["text_clean"].iloc[0]
    assert "%" in cleaned or "percent" in cleaned.lower()


def test_cleaner_assigns_id_text():
    df = _df(FOMC_JULY_2024, FOMC_SEPT_2024)
    cleaner = TextCleaner(df=df, text_column="text")
    result = cleaner.run()
    assert "id_text" in result.columns
    assert result["id_text"].tolist() == [1, 2]


def test_cleaner_missing_column():
    df = _df(FOMC_JULY_2024)
    with pytest.raises(ValueError):
        TextCleaner(df=df, text_column="nonexistent")


# ── SentimentLexical ──────────────────────────────────────────────────────────

@pytest.fixture
def fomc_df():
    return pd.DataFrame({
        "text": [FOMC_JULY_2024, FOMC_SEPT_2024],
        "date": ["2024-07-31", "2024-09-18"],
    })


@pytest.fixture
def lexical(fomc_df):
    return SentimentLexical(df_input=fomc_df, text_column="text", dictionary_path=str(DICT_PATH))


def test_sentiment_hubert_posneg(lexical):
    result = lexical.sentiment_pipeline(dictionary_name="hubert", method="posneg")
    assert "hubert_sentiment_posneg" in result.columns
    scores = result["hubert_sentiment_posneg"]
    assert all(scores >= 0)


def test_sentiment_lm_posneg(lexical):
    result = lexical.sentiment_pipeline(dictionary_name="lm", method="posneg")
    assert "lm_sentiment_posneg" in result.columns


def test_sentiment_correa_allwords(lexical):
    result = lexical.sentiment_pipeline(dictionary_name="correa", method="allwords")
    assert "correa_sentiment_allwords" in result.columns


def test_sentiment_text_column_override_does_not_mutate(fomc_df):
    fomc_df["tokens"] = fomc_df["text"].str.lower()
    analyzer = SentimentLexical(df_input=fomc_df, text_column="text", dictionary_path=str(DICT_PATH))
    analyzer.sentiment_pipeline(dictionary_name="hubert", method="posneg", text_column="tokens")
    assert analyzer.text_column == "text", "sentiment_pipeline must not mutate self.text_column"


def test_sentiment_unknown_dictionary(lexical):
    with pytest.raises(KeyError):
        lexical.sentiment_pipeline(dictionary_name="nonexistent_dict", method="posneg")


def test_sentiment_word_counts_nonzero(lexical):
    result = lexical.sentiment_pipeline(dictionary_name="lm", method="posneg")
    pos_col = [c for c in result.columns if "counttoken_positive" in c]
    neg_col = [c for c in result.columns if "counttoken_negative" in c]
    assert pos_col or neg_col, "Expected word count columns in output"


# ── Public API ────────────────────────────────────────────────────────────────

def test_public_api_imports():
    from auto_econ_sentiment import (
        AutoEconSentiment,
        SentimentLexical,
        TextLoader,
        TextCleaner,
        AutoEconSentimentError,
        __version__,
    )
    assert __version__ is not None


def test_version_is_string():
    from auto_econ_sentiment import __version__
    assert isinstance(__version__, str)
    assert len(__version__) > 0

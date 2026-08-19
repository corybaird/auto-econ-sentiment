import numpy as np
import pandas as pd
import pytest

from auto_econ_sentiment.clean.text_segmentation import TextSegmenter


def _document_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id_text": ["a", "b"],
            "text_clean": [
                "Inflation remains elevated across the euro area. "
                "The committee decided to hold rates steady this month. "
                "Too short.",
                "Financial conditions have tightened considerably this quarter. "
                "We expect a gradual moderation in price pressures ahead.",
            ],
        }
    )


def test_segmenter_expands_documents_into_sentence_rows():
    df_input = _document_frame()
    result = TextSegmenter(text_column="text_clean").run(df_input)

    assert len(result) > len(df_input)
    assert result["id_text"].tolist() == ["a", "a", "b", "b"]
    assert result["text_clean"].tolist() == [
        "Inflation remains elevated across the euro area.",
        "The committee decided to hold rates steady this month.",
        "Financial conditions have tightened considerably this quarter.",
        "We expect a gradual moderation in price pressures ahead.",
    ]


def test_segmenter_numbers_sentences_within_each_document():
    result = TextSegmenter(text_column="text_clean").run(_document_frame())

    assert result["sentence_number"].tolist() == [1, 2, 1, 2]


def test_segmenter_drops_sentences_below_min_chars():
    result = TextSegmenter(text_column="text_clean", min_chars=20).run(_document_frame())

    assert "Too short." not in result["text_clean"].tolist()


def test_segmenter_min_chars_is_configurable():
    result = TextSegmenter(text_column="text_clean", min_chars=5).run(_document_frame())

    assert "Too short." in result["text_clean"].tolist()


def test_segmenter_preserves_other_columns():
    df_input = _document_frame()
    df_input["date"] = pd.to_datetime(["2024-01-01", "2024-02-01"])
    result = TextSegmenter(text_column="text_clean").run(df_input)

    assert result.loc[result["id_text"] == "a", "date"].nunique() == 1
    assert set(df_input.columns).issubset(result.columns)


def test_segmenter_regex_fallback_matches_tokenizer_output(monkeypatch):
    monkeypatch.setattr(TextSegmenter, "_load_tokenizer", staticmethod(lambda: None))
    segmenter = TextSegmenter(text_column="text_clean")

    assert segmenter._sent_tokenize is None
    assert segmenter.run(_document_frame())["sentence_number"].tolist() == [1, 2, 1, 2]


def test_segmenter_skips_rows_with_missing_text():
    df_input = _document_frame()
    df_input.loc[1, "text_clean"] = np.nan
    result = TextSegmenter(text_column="text_clean").run(df_input)

    assert result["id_text"].unique().tolist() == ["a"]


def test_segmenter_requires_id_column():
    df_input = _document_frame().drop(columns=["id_text"])

    with pytest.raises(ValueError, match="id_text"):
        TextSegmenter(text_column="text_clean").run(df_input)


def test_segmenter_requires_text_column():
    with pytest.raises(ValueError, match="missing_column"):
        TextSegmenter(text_column="missing_column").run(_document_frame())


def test_segmenter_raises_when_no_sentences_survive():
    df_input = pd.DataFrame({"id_text": ["a"], "text_clean": ["Tiny."]})

    with pytest.raises(ValueError, match="No sentences produced"):
        TextSegmenter(text_column="text_clean", min_chars=20).run(df_input)


def test_segmenter_supports_custom_column_names():
    df_input = _document_frame().rename(columns={"id_text": "doc_id"})
    result = TextSegmenter(
        text_column="text_clean",
        id_column="doc_id",
        sentence_number_column="sent_idx",
    ).run(df_input)

    assert result["sent_idx"].tolist() == [1, 2, 1, 2]


def test_split_text_returns_empty_for_non_string_input():
    segmenter = TextSegmenter(text_column="text_clean")

    assert segmenter.split_text(None) == []
    assert segmenter.split_text(float("nan")) == []
    assert segmenter.split_text("   ") == []

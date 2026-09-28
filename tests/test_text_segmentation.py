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


def test_segmenter_tokenizer_name_property(monkeypatch):
    monkeypatch.setattr(TextSegmenter, "_load_tokenizer", staticmethod(lambda *args, **kwargs: None))
    segmenter_fallback = TextSegmenter(text_column="text_clean")
    assert segmenter_fallback.tokenizer_name == "regex_fallback"

    monkeypatch.setattr(TextSegmenter, "_load_tokenizer", staticmethod(lambda *args, **kwargs: lambda text: [text]))
    segmenter_nltk = TextSegmenter(text_column="text_clean")
    assert segmenter_nltk.tokenizer_name == "nltk_punkt"


def test_segmenter_require_nltk_raises_when_unavailable(monkeypatch):
    def mock_load(require_nltk=False):
        if require_nltk:
            raise ImportError("No module named 'nltk'")
        return None

    monkeypatch.setattr(TextSegmenter, "_load_tokenizer", staticmethod(mock_load))

    with pytest.raises(ImportError, match="nltk"):
        TextSegmenter(text_column="text_clean", require_nltk=True)


def test_regression_abbreviation_protection_st_louis(monkeypatch):
    for force_fallback in [True, False]:
        if force_fallback:
            monkeypatch.setattr(TextSegmenter, "_load_tokenizer", staticmethod(lambda *args, **kwargs: None))
        text = "The Federal Reserve Banks of Chicago, St. Louis, and Kansas City approved the action."
        segmenter = TextSegmenter(text_column="text_clean")
        sentences = segmenter.split_text(text)
        assert len(sentences) == 1
        assert sentences[0] == "The Federal Reserve Banks of Chicago, St. Louis, and Kansas City approved the action."


def test_regression_middle_initial_protection_susan_s_bies(monkeypatch):
    for force_fallback in [True, False]:
        if force_fallback:
            monkeypatch.setattr(TextSegmenter, "_load_tokenizer", staticmethod(lambda *args, **kwargs: None))
        text = "Voting for the action were Susan S. Bies, Timothy F. Geithner, and Ben S. Bernanke."
        segmenter = TextSegmenter(text_column="text_clean")
        sentences = segmenter.split_text(text)
        assert len(sentences) == 1
        assert sentences[0] == "Voting for the action were Susan S. Bies, Timothy F. Geithner, and Ben S. Bernanke."


def test_regression_newline_header_boundary(monkeypatch):
    for force_fallback in [True, False]:
        if force_fallback:
            monkeypatch.setattr(TextSegmenter, "_load_tokenizer", staticmethod(lambda *args, **kwargs: None))
        text = (
            "For immediate release\n"
            "Chairman Alan Greenspan announced today that the Committee decided to hold rates."
        )
        segmenter = TextSegmenter(text_column="text_clean")
        sentences = segmenter.split_text(text)
        assert len(sentences) == 2
        assert sentences[0] == "For immediate release"
        assert (
            sentences[1]
            == "Chairman Alan Greenspan announced today that the Committee decided to hold rates."
        )


def test_segmenter_drop_invalid_filters_garbage_and_rosters():
    text = (
        "For immediate release\n"
        "Inflation remains elevated across the euro area. "
        "The committee decided to hold rates steady this month.\n"
        "Voting for the FOMC monetary policy action were: Alan Greenspan, Chairman; Timothy F. Geithner, Vice Chairman; Ben S. Bernanke.\n"
        "|\nPress releases\n|\nContact Us"
    )
    df_input = pd.DataFrame({"id_text": ["doc1"], "text_clean": [text]})

    res_keep = TextSegmenter(text_column="text_clean", drop_invalid=False).run(df_input)
    assert len(res_keep) >= 3

    res_clean = TextSegmenter(text_column="text_clean", drop_invalid=True).run(df_input)
    assert res_clean["text_clean"].tolist() == [
        "Inflation remains elevated across the euro area.",
        "The committee decided to hold rates steady this month.",
    ]


def test_segmenter_merges_incomplete_sentence_fragments(monkeypatch):
    monkeypatch.setattr(TextSegmenter, "_load_tokenizer", staticmethod(lambda *args, **kwargs: None))
    text = (
        "The Committee decided to maintain the target range for the federal funds rate.\n"
        "and will continue to monitor the implications of incoming information.\n"
        "; with one member dissenting."
    )
    segmenter = TextSegmenter(text_column="text_clean")
    sentences = segmenter.split_text(text)
    assert len(sentences) == 1
    assert (
        sentences[0]
        == "The Committee decided to maintain the target range for the federal funds rate. and will continue to monitor the implications of incoming information. ; with one member dissenting."
    )


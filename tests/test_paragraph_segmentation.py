import numpy as np
import pandas as pd
import pytest

from auto_econ_sentiment.clean.text_segmentation import ParagraphSegmenter, TextSegmenter


def _document_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id_text": ["a", "b"],
            "text_clean": [
                "Inflation remains elevated across the euro area.\n\n"
                "The committee decided to hold rates steady this month.\n\n"
                "Short.",
                "Financial conditions have tightened considerably this quarter.\n\n"
                "We expect a gradual moderation in price pressures ahead.",
            ],
        }
    )


def test_paragraph_segmenter_expands_documents_into_paragraph_rows():
    df_input = _document_frame()
    result = ParagraphSegmenter(text_column="text_clean").run(df_input)

    assert len(result) > len(df_input)
    assert result["id_text"].tolist() == ["a", "a", "b", "b"]
    assert result["text_clean"].tolist() == [
        "Inflation remains elevated across the euro area.",
        "The committee decided to hold rates steady this month.",
        "Financial conditions have tightened considerably this quarter.",
        "We expect a gradual moderation in price pressures ahead.",
    ]


def test_paragraph_segmenter_numbers_paragraphs_within_each_document():
    result = ParagraphSegmenter(text_column="text_clean").run(_document_frame())

    assert result["paragraph_number"].tolist() == [1, 2, 1, 2]


def test_paragraph_segmenter_drops_paragraphs_below_min_chars():
    result = ParagraphSegmenter(text_column="text_clean", min_chars=20).run(_document_frame())

    assert "Short." not in result["text_clean"].tolist()


def test_paragraph_segmenter_min_chars_is_configurable():
    result = ParagraphSegmenter(text_column="text_clean", min_chars=5).run(_document_frame())

    assert "Short." in result["text_clean"].tolist()


def test_paragraph_segmenter_preserves_other_columns():
    df_input = _document_frame()
    df_input["date"] = pd.to_datetime(["2024-01-01", "2024-02-01"])
    result = ParagraphSegmenter(text_column="text_clean").run(df_input)

    assert result.loc[result["id_text"] == "a", "date"].nunique() == 1
    assert set(df_input.columns).issubset(result.columns)


def test_paragraph_segmenter_skips_rows_with_missing_text():
    df_input = _document_frame()
    df_input.loc[1, "text_clean"] = np.nan
    result = ParagraphSegmenter(text_column="text_clean").run(df_input)

    assert result["id_text"].unique().tolist() == ["a"]


def test_paragraph_segmenter_requires_id_column():
    df_input = _document_frame().drop(columns=["id_text"])

    with pytest.raises(ValueError, match="id_text"):
        ParagraphSegmenter(text_column="text_clean").run(df_input)


def test_paragraph_segmenter_requires_text_column():
    with pytest.raises(ValueError, match="missing_column"):
        ParagraphSegmenter(text_column="missing_column").run(_document_frame())


def test_paragraph_segmenter_raises_when_no_paragraphs_survive():
    df_input = pd.DataFrame({"id_text": ["a"], "text_clean": ["Tiny."]})

    with pytest.raises(ValueError, match="No paragraphs produced"):
        ParagraphSegmenter(text_column="text_clean", min_chars=20).run(df_input)


def test_paragraph_segmenter_supports_custom_column_names():
    df_input = _document_frame().rename(columns={"id_text": "doc_id"})
    result = ParagraphSegmenter(
        text_column="text_clean",
        id_column="doc_id",
        paragraph_number_column="para_idx",
    ).run(df_input)

    assert result["para_idx"].tolist() == [1, 2, 1, 2]


def test_paragraph_split_text_returns_empty_for_non_string_input():
    segmenter = ParagraphSegmenter(text_column="text_clean")

    assert segmenter.split_text(None) == []
    assert segmenter.split_text(float("nan")) == []
    assert segmenter.split_text("   ") == []


def test_paragraph_and_sentence_segmenter_composition():
    df_input = pd.DataFrame(
        {
            "id_text": ["doc1"],
            "text_clean": [
                "First paragraph first sentence. First paragraph second sentence.\n\n"
                "Second paragraph first sentence. Second paragraph second sentence."
            ],
        }
    )

    df_paragraphs = ParagraphSegmenter(text_column="text_clean").run(df_input)
    assert len(df_paragraphs) == 2
    assert df_paragraphs["paragraph_number"].tolist() == [1, 2]

    df_sentences = TextSegmenter(text_column="text_clean").run(df_paragraphs)
    assert len(df_sentences) == 4
    assert df_sentences["paragraph_number"].tolist() == [1, 1, 2, 2]
    assert df_sentences["sentence_number"].tolist() == [1, 2, 3, 4]


import logging
from pathlib import Path
import pandas as pd
import pytest
from auto_econ_sentiment.clean.text_loader import TextLoader


def test_loader_txt_flat_directory(tmp_path: Path):
    (tmp_path / "2024-01-15_statement.txt").write_text("First statement text", encoding="utf-8")
    (tmp_path / "2024-02-20_speech.txt").write_text("Second speech text", encoding="utf-8")
    (tmp_path / "2024-03-25_minutes.txt").write_text("Third minutes text", encoding="utf-8")

    loader = TextLoader(file_path=tmp_path)
    df = loader.get_data()

    assert len(df) == 3
    assert set(df["id_text"]) == {
        "2024-01-15_statement",
        "2024-02-20_speech",
        "2024-03-25_minutes",
    }
    assert pd.api.types.is_datetime64_any_dtype(df["date"])

    sorted_df = df.sort_values("date").reset_index(drop=True)
    assert sorted_df.loc[0, "text"] == "First statement text"
    assert sorted_df.loc[0, "date"] == pd.Timestamp("2024-01-15")
    assert sorted_df.loc[1, "text"] == "Second speech text"
    assert sorted_df.loc[1, "date"] == pd.Timestamp("2024-02-20")
    assert sorted_df.loc[2, "text"] == "Third minutes text"
    assert sorted_df.loc[2, "date"] == pd.Timestamp("2024-03-25")


def test_loader_txt_unparseable_date_retained_with_warning(tmp_path: Path, caplog: pytest.LogCaptureFixture):
    (tmp_path / "2024-01-15_valid.txt").write_text("Valid date text", encoding="utf-8")
    (tmp_path / "undated_speech.txt").write_text("No date in filename", encoding="utf-8")

    with caplog.at_level(logging.WARNING):
        loader = TextLoader(file_path=tmp_path)

    df = loader.get_data()
    assert len(df) == 2

    undated_row = df[df["id_text"] == "undated_speech"]
    assert len(undated_row) == 1
    assert pd.isna(undated_row["date"].iloc[0])
    assert undated_row["text"].iloc[0] == "No date in filename"

    valid_row = df[df["id_text"] == "2024-01-15_valid"]
    assert len(valid_row) == 1
    assert valid_row["date"].iloc[0] == pd.Timestamp("2024-01-15")

    assert "Could not parse dates from 1 file(s)" in caplog.text


def test_loader_txt_group_column_subdirectories(tmp_path: Path):
    us_dir = tmp_path / "US"
    ea_dir = tmp_path / "EA"
    us_dir.mkdir()
    ea_dir.mkdir()

    (us_dir / "2024-01-01_statement.txt").write_text("US Jan statement", encoding="utf-8")
    (us_dir / "2024-02-01_statement.txt").write_text("US Feb statement", encoding="utf-8")
    (ea_dir / "2024-01-01_statement.txt").write_text("EA Jan statement", encoding="utf-8")

    loader = TextLoader(file_path=tmp_path, group_column="Country")
    df = loader.get_data()

    assert len(df) == 3
    assert "Country" in df.columns
    assert set(df["Country"]) == {"US", "EA"}
    assert set(df["id_text"]) == {
        "US_2024-01-01_statement",
        "US_2024-02-01_statement",
        "EA_2024-01-01_statement",
    }

    us_rows = df[df["Country"] == "US"]
    assert len(us_rows) == 2
    ea_rows = df[df["Country"] == "EA"]
    assert len(ea_rows) == 1


def test_loader_txt_recursive_flag(tmp_path: Path):
    (tmp_path / "2024-01-01_root.txt").write_text("Root statement", encoding="utf-8")
    sub_dir = tmp_path / "sub"
    deep_dir = sub_dir / "deep"
    deep_dir.mkdir(parents=True)

    (sub_dir / "2024-02-01_sub.txt").write_text("Sub statement", encoding="utf-8")
    (deep_dir / "2024-03-01_deep.txt").write_text("Deep statement", encoding="utf-8")

    loader_flat = TextLoader(file_path=tmp_path, recursive=False)
    assert len(loader_flat.get_data()) == 1

    loader_rec = TextLoader(file_path=tmp_path, recursive=True)
    assert len(loader_rec.get_data()) == 3


def test_loader_txt_empty_directory(tmp_path: Path):
    empty_dir = tmp_path / "empty_dir"
    empty_dir.mkdir()

    with pytest.raises(ValueError, match="No .txt files found"):
        TextLoader(file_path=empty_dir)


def test_loader_txt_skip_date_parsing(tmp_path: Path, caplog: pytest.LogCaptureFixture):
    (tmp_path / "some_file.txt").write_text("Sample text", encoding="utf-8")

    with caplog.at_level(logging.WARNING):
        loader = TextLoader(file_path=tmp_path, filename_date_pattern=None)

    df = loader.get_data()
    assert len(df) == 1
    assert pd.isna(df["date"].iloc[0])
    assert caplog.text == ""


def test_loader_txt_custom_id_and_date_pattern(tmp_path: Path):
    (tmp_path / "statement_20240501.txt").write_text("May statement", encoding="utf-8")

    loader = TextLoader(
        file_path=tmp_path,
        id_column="doc_id",
        filename_date_pattern=r"statement_(\d{4})(\d{2})(\d{2})",
    )
    df = loader.get_data()

    assert "doc_id" in df.columns
    assert df["doc_id"].iloc[0] == "statement_20240501"
    assert df["date"].iloc[0] == pd.Timestamp("2024-05-01")


def test_loader_txt_non_default_column_debug_log(tmp_path: Path, caplog: pytest.LogCaptureFixture):
    (tmp_path / "2024-01-01_doc.txt").write_text("Doc text", encoding="utf-8")

    with caplog.at_level(logging.DEBUG):
        loader = TextLoader(file_path=tmp_path, text_column="custom_text", date_column="custom_date")

    df = loader.get_data()
    assert "text" in df.columns
    assert "date" in df.columns
    assert "text_column and date_column arguments are unused for directory/txt input" in caplog.text


def test_loader_txt_lossy_encoding(tmp_path: Path):
    bad_file = tmp_path / "2024-01-01_lossy.txt"
    bad_file.write_bytes(b"Good prefix \xff\xfe bad bytes and suffix")

    loader = TextLoader(file_path=tmp_path)
    df = loader.get_data()

    assert len(df) == 1
    assert "Good prefix" in df["text"].iloc[0]
    assert "bad bytes and suffix" in df["text"].iloc[0]


def test_loader_txt_get_summary_stats(tmp_path: Path):
    us_dir = tmp_path / "US"
    uk_dir = tmp_path / "UK"
    us_dir.mkdir()
    uk_dir.mkdir()

    (us_dir / "2024-01-01_speech1.txt").write_text("US first speech", encoding="utf-8")
    (us_dir / "2024-02-01_speech2.txt").write_text("US second speech", encoding="utf-8")
    (uk_dir / "2024-03-01_speech3.txt").write_text("UK speech", encoding="utf-8")

    loader = TextLoader(file_path=tmp_path, group_column="Country")
    stats = loader.get_summary_stats("Country")

    assert stats is not None
    counts, text_stats, time_span, date_ranges = stats
    assert len(counts) == 2
    assert len(text_stats) == 2
    assert len(date_ranges) == 2

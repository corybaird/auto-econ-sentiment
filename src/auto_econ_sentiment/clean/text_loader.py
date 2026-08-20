import logging
from pathlib import Path
import re
from typing import Optional, Tuple, Union
import pandas as pd

logger = logging.getLogger(__name__)


class TextLoader:
    """Load a tabular (CSV, Excel, Parquet) corpus or directory of raw .txt files,
    validate columns/files, and standardize text and parsed dates.

    Parameters
    ----------
    file_path : str or Path
        Path to a single tabular file (.csv, .xlsx/.xls, .parquet/.parquet.gzip/.parquet.gz)
        or a directory containing .txt document files.
    text_column : str, default "text"
        Name of the column containing document text (used for tabular inputs; ignored for
        directory inputs).
    date_column : str, default "date"
        Name of the column containing dates (used for tabular inputs; ignored for directory
        inputs).
    id_column : str, default "id_text"
        Column name to store document identifier generated from filename stems for directory
        inputs. When ``group_column`` is set and subdirectories exist, identifiers are
        prefixed with the group name (e.g., ``f"{group}_{stem}"``).
    filename_date_pattern : str or None, default r"^(\\d{4})[-_](\\d{2})[-_](\\d{2})"
        Regex pattern with three capture groups (year, month, day) matched against the filename
        stem to extract dates for directory inputs. Pass ``None`` to skip date parsing.
        Files with unparseable dates are retained with ``date = NaT`` and a warning is logged
        (departure from scripts that silently drop rows).
    group_column : str or None, default None
        If provided and the directory contains subdirectories, each top-level subdirectory name
        populates this column for its files.
    recursive : bool, default False
        Whether to recursively scan for .txt files within subdirectories.
    """

    def __init__(
        self,
        file_path: Union[str, Path],
        text_column: str = "text",
        date_column: str = "date",
        id_column: str = "id_text",
        filename_date_pattern: Optional[str] = r"^(\d{4})[-_](\d{2})[-_](\d{2})",
        group_column: Optional[str] = None,
        recursive: bool = False,
    ) -> None:
        self.file_path = str(file_path)
        self.text_column = text_column
        self.date_column = date_column
        self.id_column = id_column
        self.filename_date_pattern = filename_date_pattern
        self.group_column = group_column
        self.recursive = recursive
        self.data: pd.DataFrame = self._load_and_process()

    def _parse_filename_date(self, stem: str) -> Tuple[Optional[pd.Timestamp], bool]:
        if self.filename_date_pattern is None:
            return pd.NaT, False

        match = re.search(self.filename_date_pattern, stem)
        if match and len(match.groups()) >= 3:
            date_str = f"{match.group(1)}-{match.group(2)}-{match.group(3)}"
            try:
                parsed = pd.to_datetime(date_str)
                return parsed, False
            except Exception:
                return pd.NaT, True
        return pd.NaT, True

    def _load_directory(self, dir_path: Path) -> pd.DataFrame:
        if self.text_column != "text" or self.date_column != "date":
            logger.debug(
                "text_column and date_column arguments are unused for directory/txt input; "
                "outputs are standardized to 'text' and 'date'."
            )

        records = []
        unparseable_count = 0

        subdirs = sorted([d for d in dir_path.iterdir() if d.is_dir()]) if dir_path.exists() else []

        if self.group_column is not None and len(subdirs) > 0:
            for subdir in subdirs:
                group_name = subdir.name
                txt_files = (
                    sorted(subdir.rglob("*.txt"))
                    if self.recursive
                    else sorted(subdir.glob("*.txt"))
                )
                for txt_file in txt_files:
                    text_content = txt_file.read_text(encoding="utf-8", errors="ignore")
                    parsed_date, is_unparseable = self._parse_filename_date(txt_file.stem)
                    if is_unparseable:
                        unparseable_count += 1
                    doc_id = f"{group_name}_{txt_file.stem}"
                    record = {
                        self.id_column: doc_id,
                        self.group_column: group_name,
                        "text": text_content,
                        "date": parsed_date,
                    }
                    records.append(record)

            top_level_files = sorted(dir_path.glob("*.txt"))
            for txt_file in top_level_files:
                text_content = txt_file.read_text(encoding="utf-8", errors="ignore")
                parsed_date, is_unparseable = self._parse_filename_date(txt_file.stem)
                if is_unparseable:
                    unparseable_count += 1
                doc_id = txt_file.stem
                record = {
                    self.id_column: doc_id,
                    self.group_column: None,
                    "text": text_content,
                    "date": parsed_date,
                }
                records.append(record)
        else:
            txt_files = (
                sorted(dir_path.rglob("*.txt"))
                if self.recursive
                else sorted(dir_path.glob("*.txt"))
            )
            for txt_file in txt_files:
                text_content = txt_file.read_text(encoding="utf-8", errors="ignore")
                parsed_date, is_unparseable = self._parse_filename_date(txt_file.stem)
                if is_unparseable:
                    unparseable_count += 1
                doc_id = txt_file.stem
                record = {
                    self.id_column: doc_id,
                    "text": text_content,
                    "date": parsed_date,
                }
                if self.group_column is not None:
                    record[self.group_column] = None
                records.append(record)

        if not records:
            raise ValueError(f"No .txt files found in directory: {self.file_path}")

        if self.filename_date_pattern is not None and unparseable_count > 0:
            logger.warning(
                f"Could not parse dates from {unparseable_count} file(s) matching pattern "
                f"'{self.filename_date_pattern}'. Dates set to NaT."
            )

        df = pd.DataFrame(records)
        df["date"] = pd.to_datetime(df["date"])
        return df

    def _load_and_process(self) -> pd.DataFrame:
        path = Path(self.file_path)

        if path.is_dir():
            return self._load_directory(path)
        elif self.file_path.endswith(".csv"):
            df = pd.read_csv(self.file_path)
        elif self.file_path.endswith(".xlsx") or self.file_path.endswith(".xls"):
            df = pd.read_excel(self.file_path)
        elif (
            self.file_path.endswith(".parquet")
            or self.file_path.endswith(".parquet.gzip")
            or self.file_path.endswith(".parquet.gz")
        ):
            df = pd.read_parquet(self.file_path)
        else:
            raise ValueError("Unsupported file format. Please use CSV, Excel, or Parquet files.")

        if self.text_column not in df.columns:
            raise ValueError(f"Text column '{self.text_column}' not found in the file.")
        if self.date_column not in df.columns:
            raise ValueError(f"Date column '{self.date_column}' not found in the file.")

        try:
            df[self.date_column] = pd.to_datetime(df[self.date_column])
        except Exception as e:
            raise ValueError(f"Error converting date column: {e}") from e

        df = df.rename(columns={self.text_column: "text", self.date_column: "date"})
        return df

    def get_data(self) -> pd.DataFrame:
        """Return a defensive copy of the normalized DataFrame."""
        return self.data.copy()

    def get_summary_stats(
        self, group_by_column: str
    ) -> Optional[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
        """Return ``(group_counts, text_stats, time_span, date_ranges)`` summarizing
        the corpus by ``group_by_column``. Returns ``None`` if the data is empty."""
        if self.data.empty:
            return None

        if group_by_column not in self.data.columns:
            raise ValueError(f"Column '{group_by_column}' not found in the data.")

        self.data["text_length"] = self.data["text"].str.len()

        df_group_counts = (
            self.data[group_by_column]
            .value_counts()
            .reset_index()
            .rename(columns={"index": group_by_column, group_by_column: "count"})
        )

        df_text_stats = (
            self.data.groupby(group_by_column)["text_length"]
            .agg(["mean", "min", "max", "count"])
            .round(2)
            .reset_index()
        )

        min_date = self.data["date"].min()
        max_date = self.data["date"].max()
        start_date_str = min_date.strftime("%Y-%m-%d") if pd.notna(min_date) else None
        end_date_str = max_date.strftime("%Y-%m-%d") if pd.notna(max_date) else None
        total_days = (
            (max_date - min_date).days if pd.notna(min_date) and pd.notna(max_date) else None
        )

        df_time_span = pd.DataFrame({
            "metric": ["start_date", "end_date", "total_days", "total_records"],
            "value": [
                start_date_str,
                end_date_str,
                total_days,
                len(self.data),
            ],
        })

        date_ranges = (
            self.data.groupby(group_by_column)
            .agg({"date": ["min", "max"]})
            .round(2)
        )
        date_ranges.columns = ["start_date", "end_date"]
        df_date_ranges = date_ranges.reset_index().assign(
            start_date=lambda x: x["start_date"].dt.strftime("%Y-%m-%d"),
            end_date=lambda x: x["end_date"].dt.strftime("%Y-%m-%d"),
        )

        return df_group_counts, df_text_stats, df_time_span, df_date_ranges
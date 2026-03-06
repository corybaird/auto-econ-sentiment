import pandas as pd


class TextLoader:
    def __init__(self, file_path, text_column, date_column):
        self.file_path = file_path
        self.text_column = text_column
        self.date_column = date_column
        self.data = self._load_and_process()

    def _load_and_process(self):
        if self.file_path.endswith(".csv"):
            df = pd.read_csv(self.file_path)
        elif self.file_path.endswith(".xlsx") or self.file_path.endswith(".xls"):
            df = pd.read_excel(self.file_path)
        elif self.file_path.endswith(".parquet") or self.file_path.endswith(".parquet.gzip") or self.file_path.endswith(".parquet.gz"):
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

    def get_data(self):
        return self.data.copy()

    def get_summary_stats(self, group_by_column):
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

        df_time_span = pd.DataFrame({
            "metric": ["start_date", "end_date", "total_days", "total_records"],
            "value": [
                self.data["date"].min().strftime("%Y-%m-%d"),
                self.data["date"].max().strftime("%Y-%m-%d"),
                (self.data["date"].max() - self.data["date"].min()).days,
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
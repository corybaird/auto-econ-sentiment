# Data And Outputs

Raw datasets and generated outputs are intentionally kept out of version control. Use the scripts in `src/data/` to download or create local data.

## Inputs

| Path | Description |
| --- | --- |
| `data/raw/basic_tests/monetary_policy_statement.parquet.gzip` | FOMC monetary policy statements for quick local validation. |
| `data/raw/basic_tests/statements_speeches.parquet.gzip` | Small mixed sample of statements and speeches. |
| `data/raw/speeches/CBNAME.parquet.gzip` | Per-central-bank files generated from the CBS speeches dataset. |
| `data/raw/statements/` | Directory of per-document `.txt` files (flat or organized by subdirectory). |

Input files can be single tabular files (`.csv`, `.xlsx`/`.xls`, `.parquet`/`.parquet.gzip`) containing configured text and date columns, or a directory of raw `.txt` files.

### Directory of Text Files (`TextLoader`)

`TextLoader` supports loading corpora directly from a directory of `.txt` files with automatic date parsing and optional categorization:

```python
TextLoader(
    file_path="data/raw/statements/",
    text_column="text",        # ignored for txt/dir input
    date_column="date",        # ignored for txt/dir input
    id_column="id_text",       # document ID column (default: "id_text")
    filename_date_pattern=r"^(\d{4})[-_](\d{2})[-_](\d{2})",  # date regex pattern (None to skip)
    group_column="Country",    # optional; populates column from 1-level subdirectories
    recursive=False,           # recursively search for .txt files
)
```

- **Date parsing**: Filename stems are matched against `filename_date_pattern` (extracting year, month, day). Files with unparseable dates are retained with `date = NaT` and a warning is logged with the count. Pass `filename_date_pattern=None` to skip date parsing entirely.
- **Group columns & IDs**: When `group_column` is provided and the directory contains subdirectories, `group_column` is populated with the subdirectory name, and `id_text` is prefixed with the group name (`f"{group}_{stem}"`) to ensure uniqueness across groups.
- **Encoding**: Text files are read using UTF-8 with `errors="ignore"` to handle lossy encodings gracefully.


## Outputs

| Path | Description |
| --- | --- |
| `cleaned.parquet.gzip` | Cleaned text, tokens, stems, and document IDs. |
| `sentiment_lexical.parquet.gzip` | Lexical counts, matched words, and sentiment scores. |
| `sentiment_transformer.parquet.gzip` | Optional transformer labels, probabilities, counts, shares, and scores. |
| `sentiment_transformer_sentence_probabilities.parquet.gzip` | Optional sentence-level transformer probabilities. |
| `sentiment_all_results.parquet.gzip` | Combined output used by notebooks and paper scripts. |

## Lexical Columns

Lexical columns follow this pattern:

```text
{dictionary}_counttoken_positive_{method}
{dictionary}_counttoken_negative_{method}
{dictionary}_counttoken_total_{method}
{dictionary}_words_positive_{method}
{dictionary}_words_negative_{method}
{dictionary}_sentiment_{method}
```

## Transformer Columns

Transformer columns are prefixed by `model_name_short`, for example:

```text
fomc_label
fomc_probability_0
fomc_sentiment_byalltext
fomc_countsentence_LABEL_0
fomc_meanprobability_LABEL_0
fomc_sentiment_bysentence
fomc_sentiment_bysentence_mean
fomc_count_positive
fomc_share_negative
fomc_net_sentiment
```

`bysentence` aggregation supports two modes via `sentence_probability_aggregation`:
- `cutoff` (default): first records raw per-sentence probabilities, then counts labels that pass the configured `sentence_probability_cutoff` and aggregates those counts back to `id_text` (producing columns like `{model}_countsentence_{label}` and `{model}_sentiment_bysentence`).
- `mean`: averages raw per-sentence probabilities across sentences for each document (producing columns like `{model}_meanprobability_{label}` and `{model}_sentiment_bysentence_mean`), preserving confidence magnitude.

When `output_schema: shares` is enabled, the model-specific labels are also converted into harmonized positive, neutral, and negative counts and shares across both modes.

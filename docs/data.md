# Data And Outputs

Raw datasets and generated outputs are intentionally kept out of version control. Use the scripts in `src/data/` to download or create local data.

## Inputs

| Path | Description |
| --- | --- |
| `data/raw/basic_tests/monetary_policy_statement.parquet.gzip` | FOMC monetary policy statements for quick local validation. |
| `data/raw/basic_tests/statements_speeches.parquet.gzip` | Small mixed sample of statements and speeches. |
| `data/raw/speeches/CBNAME.parquet.gzip` | Per-central-bank files generated from the CBS speeches dataset. |
| `data/raw/statements/` | Directory of per-document `.txt` files (flat or organized by subdirectory). |

Input files can be single tabular files (`.csv`, `.xlsx`/`.xls`, `.parquet`/`.parquet.gzip`) containing configured text and date columns, or a directory of raw `.txt` and Markdown (`.md`, `.markdown`) files.

### Directory of Text Files (`TextLoader`)

`TextLoader` supports loading corpora directly from a directory of `.txt` and Markdown files, one document per file, with automatic date parsing and optional categorization:

```python
TextLoader(
    file_path="data/raw/statements/",
    text_column="text",        # ignored for txt/dir input
    date_column="date",        # ignored for txt/dir input
    id_column="id_text",       # document ID column (default: "id_text")
    filename_date_pattern=r"^(\d{4})[-_](\d{2})[-_](\d{2})",  # date regex pattern (None to skip)
    group_column="Country",    # optional; populates column from 1-level subdirectories
    recursive=False,           # recursively search for .txt and .md files
)
```

- **Date parsing**: Filename stems are matched against `filename_date_pattern` (extracting year, month, day). Files with unparseable dates are retained with `date = NaT` and a warning is logged with the count. Pass `filename_date_pattern=None` to skip date parsing entirely.
- **Group columns & IDs**: When `group_column` is provided and the directory contains subdirectories, `group_column` is populated with the subdirectory name, and `id_text` is prefixed with the group name (`f"{group}_{stem}"`) to ensure uniqueness across groups.
- **Markdown**: `.md` and `.markdown` files are read as plain text, so headings, emphasis and links reach the cleaner as written.
- **Encoding**: Text files are read using UTF-8 with `errors="ignore"` to handle lossy encodings gracefully.


## Outputs

| Path | Description |
| --- | --- |
| `sentiment_lexical.parquet.gzip` | Lexical counts, matched words, and sentiment scores. |
| `sentiment_transformer.parquet.gzip` | Optional transformer labels, probabilities, counts, shares, and scores. |
| `sentiment_transformer_sentence_probabilities.parquet.gzip` | Optional sentence-level transformer probabilities, one row per sentence with `id_text`, `sentence_number` and `sentence_text`. |
| `sentiment_llm.parquet.gzip` | Optional LLM polarities, confidences, scores, and metadata. |
| `sentiment_llm_sentence_probabilities.parquet.gzip` | Optional sentence-level LLM polarities and confidences. |
| `sentiment_all_results.parquet.gzip` | Combined output uniting cleaned text with lexical, transformer and LLM scores, one row per document. |
| `cleaned.parquet.gzip` | Original text, cleaned text, tokens, stems, and document IDs. Written on every pipeline run, and by `TextCleaner.export_data()`. |

## Lexical Columns

Lexical columns follow this pattern:

```text
{dictionary}_counttoken_positive_{method}
{dictionary}_counttoken_negative_{method}
{dictionary}_counttoken_total_{method}      # allwords only
{dictionary}_words_positive_{method}
{dictionary}_words_negative_{method}
{dictionary}_sentiment_{method}
{dictionary}_sentiment_{method}_net
```

Dictionaries matched against stemmed text (`stemmed` in `params.yaml`) append `_stem` to the method, so every column carries it, for example `bn_counttoken_positive_posneg_stem`, `bn_sentiment_posneg_stem` and `bn_sentiment_posneg_stem_net`. The `allwords` token total excludes English stop words.

## Score Scales

Every column ending in `_net` is on $[-1, 1]$ with zero as neutral, so lexical and transformer scores can be compared directly.

| Column | Range | Neutral | Notes |
|---|---|---|---|
| `{dictionary}_sentiment_{method}` | $[0, 2]$ | 1 | Apel-Blix Grimaldi convention: the net score plus one. |
| `{dictionary}_sentiment_{method}_net` | $[-1, 1]$ | 0 | The same score recentered on zero. |
| `{model}_sentiment_bysentence` | $[-1, 1]$ | 0 | Net count over sentences that clear the probability cutoff. |
| `{model}_sentiment_bysentence_mean` | $[-1, 1]$ | 0 | Direction-weighted mean of sentence probabilities (`sentence_probability_aggregation: mean`). |
| `{model}_sentiment_byalltext` | $[-1, 1]$ | 0 | Label direction times the predicted class probability. |
| `{model}_sentiment_posneg_net` | $[-1, 1]$ | 0 | Net count over sentences that clear the cutoff in either direction; null when none do. Cutoff mode with `output_schema: shares`. |
| `{model}_sentiment_allsentences_net` | $[-1, 1]$ | 0 | Net count over every segmented sentence. Cutoff mode with `output_schema: shares`. |
| `{model}_net_sentiment` | $[-1, 1]$ | 0 | Positive share minus negative share; needs `output_schema: shares`. |
| `{llm}_sentiment_byalltext` | $[-1, 1]$ or $\{0, 1, 2\}$ | 0 or 1 | Polarity times confidence with `output_scale: continuous`; polarity mapped to 0, 1, 2 with `discrete`. |
| `{llm}_sentiment_bysentence`, `{llm}_net_sentiment` | $[-1, 1]$ | 0 | Positive share minus negative share over sentences that clear `confidence_cutoff`. |

Transformer and LLM columns are prefixed by the model's `short_name`; `{model}` and `{llm}` stand for it above. The `_net` suffix marks the scores designed to be compared across methods: lexical `posneg` and `allwords` pair with transformer `posneg` and `allsentences`, which share their numerator and differ only in the denominator.

For example, a document with three positive and one negative Hubert-Labondance match has `hubert_sentiment_posneg = 1.5` and `hubert_sentiment_posneg_net = 0.5`.

### Paper Notation

The accompanying paper names the aggregations rather than the columns. Each maps to one column:

| Paper | Formula | Column |
|---|---|---|
| Lexical `PosNeg` | $(P - N) / (P + N)$ | `{dictionary}_sentiment_posneg_net` (`_posneg_stem_net` for stemmed dictionaries) |
| Lexical `All-Words` | $(P - N) / T$ | `{dictionary}_sentiment_allwords_net` |
| Transformer `PosNeg` | $(N^{+} - N^{-}) / (N^{+} + N^{-})$ | `{model}_sentiment_posneg_net` |
| Transformer `All-Sentences` | $(N^{+} - N^{-}) / S$ | `{model}_sentiment_allsentences_net` |
| Net share over classified sentences | $(N^{+} - N^{-}) / (N^{+} + N^{0} + N^{-})$ | `{model}_net_sentiment` |

$P$ and $N$ count positive and negative dictionary matches, $T$ the tokens left after English stop words are removed, $N^{c}$ the sentences whose class $c$ clears `sentence_probability_cutoff`, and $S$ every segmented sentence.

`{model}_sentiment_posneg_net` is missing when no sentence is classified positive or negative. When no sentence clears the cutoff at all, `{model}_net_sentiment` and `{model}_sentiment_bysentence` are 0 rather than missing. Filter on `{model}_count_positive + {model}_count_neutral + {model}_count_negative > 0` to treat those documents as unmeasured.

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
fomc_count_sentences
fomc_sentiment_posneg_net
fomc_sentiment_allsentences_net
```

`bysentence` aggregation supports two modes via `sentence_probability_aggregation`:
- `cutoff` (default): first records raw per-sentence probabilities, then counts labels that pass the configured `sentence_probability_cutoff` and aggregates those counts back to `id_text` (producing columns like `{model}_countsentence_{label}` and `{model}_sentiment_bysentence`).
- `mean`: averages raw per-sentence probabilities across sentences for each document (producing columns like `{model}_meanprobability_{label}` and `{model}_sentiment_bysentence_mean`), preserving confidence magnitude.

When `output_schema: shares` is enabled, the model-specific labels are also converted into harmonized positive, neutral, and negative counts and shares across both modes.

## LLM Columns

LLM columns are prefixed by `model_name_short`, for example:

```text
llama3_polarity
llama3_confidence
llama3_sentiment_byalltext
llama3_sentiment_bysentence
llama3_count_positive
llama3_count_neutral
llama3_count_negative
llama3_countsentence_positive   # same counts as count_*, kept for parity with transformers
llama3_countsentence_neutral
llama3_countsentence_negative
llama3_share_positive
llama3_share_neutral
llama3_share_negative
llama3_net_sentiment
llama3_provider
llama3_model
llama3_prompt_version
llama3_temperature
```

In document-level scoring (`byalltext`), the derived score is computed as `polarity * confidence` (continuous scale) or `{0, 1, 2}` mapped from polarity (discrete scale). In sentence-level scoring (`bysentence`), sentences meeting the `confidence_cutoff` are aggregated into harmonized counts, shares, and net sentiment. Metadata columns record provider, model name, prompt version, and temperature for governance and reproducibility.


# Data And Outputs

Raw datasets and generated outputs are intentionally kept out of version control. Use the scripts in `src/data/` to download or create local data.

## Inputs

| Path | Description |
| --- | --- |
| `data/raw/basic_tests/monetary_policy_statement.parquet.gzip` | FOMC monetary policy statements for quick local validation. |
| `data/raw/basic_tests/statements_speeches.parquet.gzip` | Small mixed sample of statements and speeches. |
| `data/raw/speeches/CBNAME.parquet.gzip` | Per-central-bank files generated from the CBS speeches dataset. |

Input files must contain the configured text and date columns from `params.yaml`.

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
fomc_count_positive
fomc_share_negative
fomc_net_sentiment
```

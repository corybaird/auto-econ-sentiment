# Examples

## Run From YAML

Edit `params.yaml`, then run:

```bash
uv run python -m src.auto_econ_sentiment.pipeline
```

The default config keeps transformer models disabled. Set `models.transformer.enabled: true` after installing the transformer extra.

## Run With Python

```python
from auto_econ_sentiment.pipeline import AutoEconSentiment

analyzer = AutoEconSentiment(
    import_file_path="data/raw/basic_tests/monetary_policy_statement.parquet.gzip",
    text_column="text",
    date_column="date",
    export_path="data/sentiment/basic_tests/",
)

analyzer.run(
    clean_config={"tokenize": True, "stem": True},
    dictionaries={"unstemmed": ["correa", "hubert", "lm"], "stemmed": ["ap", "bn"]},
    aggregation_methods=["posneg", "allwords"],
    export_results=True,
)
```

## Run The CBS Speeches Demo

```bash
uv run python -m src.data.cb_speeches_download
uv run python -m src.data.cb_speeches_clean
```

Then open:

```text
notebooks/demo_cb_speechs.ipynb
```

## Transformer Example

```bash
uv sync --extra transformers
```

```yaml
transformer:
  enabled: true
  text_column_transformer: text_clean
  aggregation_methods: [bysentence]
  output_schema: shares
  min_sentence_chars: 20
  sentence_probability_cutoff: 0.7
  models:
    - name: ProsusAI/finbert
      short_name: finbertpro
      num_labels: 3
      label_mapping:
        neutral: neutral
        positive: positive
        negative: negative
      sentiment_values:
        neutral: 0
        positive: 1
        negative: -1
```

Then run the YAML-configured pipeline:

```bash
uv run python -m src.auto_econ_sentiment.pipeline
```

For sentence-level aggregation, the pipeline writes both `sentiment_transformer.parquet.gzip` and `sentiment_transformer_sentence_probabilities.parquet.gzip`.

## Text and Paragraph Segmentation

The `clean` module provides `TextSegmenter` for sentence-level splitting and `ParagraphSegmenter` for paragraph-level splitting.

### Sentence Segmentation with `TextSegmenter`

`TextSegmenter` uses NLTK's `sent_tokenize` when available and falls back to a robust regex boundary tokenizer with abbreviation and single-initial protection when NLTK or `punkt` data is unavailable.

```python
import pandas as pd
from auto_econ_sentiment.clean import TextSegmenter

df = pd.DataFrame({
    "id_text": ["doc1"],
    "text_clean": [
        "Inflation remains elevated across the euro area. "
        "The committee decided to hold rates steady this month.\n"
        "Voting for the FOMC monetary policy action were: Alan Greenspan, Chairman; Timothy F. Geithner, Vice Chairman; Ben S. Bernanke."
    ],
})

# Standard segmentation
segmenter = TextSegmenter(text_column="text_clean")
print(f"Tokenizer used: {segmenter.tokenizer_name}")  # 'nltk_punkt' or 'regex_fallback'
df_sentences = segmenter.run(df)
```

#### Tokenizer Visibility and Fail-Fast

- `tokenizer_name`: Exposes whether `'nltk_punkt'` or `'regex_fallback'` is active.
- `require_nltk=True`: Ensures production workflows fail fast with an `ImportError` or `LookupError` rather than silently using the regex fallback:

```python
segmenter = TextSegmenter(text_column="text_clean", require_nltk=True)
```

#### Filtering Non-Content Fragments (`drop_invalid`)

Pass `drop_invalid=True` to filter out non-sentential noise such as header/footer metadata, navigation links, and voting rosters:

```python
segmenter = TextSegmenter(text_column="text_clean", drop_invalid=True)
df_clean_sentences = segmenter.run(df)
```

### Paragraph Segmentation with `ParagraphSegmenter`

`ParagraphSegmenter` splits documents on blank lines (`\n\s*\n+`) and tracks `paragraph_number`.

```python
from auto_econ_sentiment.clean import ParagraphSegmenter, TextSegmenter

# 1. Split documents into paragraphs
p_segmenter = ParagraphSegmenter(text_column="text_clean")
df_paragraphs = p_segmenter.run(df)

# 2. Split paragraphs into sentences while preserving paragraph numbers
s_segmenter = TextSegmenter(text_column="text_clean")
df_sentences = s_segmenter.run(df_paragraphs)
# df_sentences contains id_text, paragraph_number, sentence_number, text_clean
```


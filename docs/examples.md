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
models:
  transformer:
    enabled: true
    text_column_transformer: text_clean
    aggregation_methods: [bysentence]
    output_schema: shares
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

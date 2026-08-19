# AutoEconSentiment

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

`auto-econ-sentiment` is a reproducible pipeline for measuring economic sentiment in central bank and financial text. The core package is lexical-first: it cleans text, scores it with established economic dictionaries, and exports comparable sentiment outputs. Transformer sentiment is available as an optional extension.

## Quick Start

Install the base package:

```bash
uv sync
```

Run the default YAML-configured pipeline:

```bash
uv run python -m src.auto_econ_sentiment.pipeline
```

Or call the Python API:

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
    dictionaries={"unstemmed": ["correa", "hubert", "lm", "hiv"], "stemmed": ["ap", "bn"]},
    aggregation_methods=["posneg", "allwords"],
    export_results=True,
)
```

## Transformer Quick Start

Transformer support is optional so lexical users do not need to install `torch` or download Hugging Face models.

```bash
uv sync --extra transformers
```

Then enable transformer models in `params.yaml`:

```yaml
transformer:
  enabled: true
  text_column_transformer: text_clean
  aggregation_methods: [bysentence]
  output_schema: shares
  min_sentence_chars: 20
  sentence_probability_cutoff: 0.7
  models:
    - name: gtfintechlab/FOMC-RoBERTa
      short_name: fomc
      num_labels: 3
      label_mapping:
        LABEL_0: positive
        LABEL_1: negative
        LABEL_2: neutral
      sentiment_values:
        positive: 1
        negative: -1
        neutral: 0
```

The transformer examples in `params.yaml` show the supported model-list format. More experimental transformer notes live locally in `docs/feedback/` when present.

Transformer runs export `sentiment_transformer.parquet.gzip` and, for sentence-level aggregation, `sentiment_transformer_sentence_probabilities.parquet.gzip` alongside the existing lexical and combined output tables.

## What It Does

- Loads CSV or parquet text data.
- Cleans and normalizes economic text.
- Scores text with multiple central bank and financial dictionaries.
- Optionally scores text with transformer classifiers using explicitly configured label mappings.
- Exports cleaned text, matched words, counts, probabilities, and sentiment scores.
- Makes dictionary and model disagreement visible for research workflows.

## Documentation

- [Architecture](docs/architecture.md)
- [Data and Outputs](docs/data.md)
- [Examples](docs/examples.md)
- [Roadmap](docs/ROADMAP.md)
- [Transformer notebook](notebooks/autoecon_transformers.ipynb)

## CBS Speeches Demo

Download the CBS central bank speeches dataset and run the sentiment pipeline:

```bash
uv run python -m src.data.cb_speeches_download
uv run python -m src.data.cb_speeches_clean
```

Then open `notebooks/demo_cb_speechs.ipynb` to explore the outputs.

## Citations

Dataset:

- Campiglio, E., Deyris, J., Romelli, D., & Scalisi, G. (2025). Warning words in a warming world: Central bank communication and climate change. *European Economic Review*, 105101.

Lexical dictionaries:

- Loughran, T. and B. McDonald (2011). When Is a Liability Not a Liability? Textual Analysis, Dictionaries, and 10-Ks. *The Journal of Finance* 66, 35-65.
- Correa, R., K. Garud, J. Londono, and N. Mislang (2017). Sentiment in Central Bank as Financial Stability Reports. International Finance Discussion Paper 1203.
- Hubert, P. and F. Labondance (2021). The signaling effects of central bank tone. *European Economic Review* 133, 103684.
- Stone, P. J., D. C. Dunphy, and M. S. Smith (1966). *The General Inquirer: A Computer Approach to Content Analysis*.
- Apel, M. and M. Blix Grimaldi (2014). How Informative Are Central Bank Minutes? *Review of Economics* 65(1), 53-76.
- Bennani, H. and M. Neuenkirch (2017). The (Home) Bias of European Central Bankers. *Applied Economics* 49(11), 1114-1131.

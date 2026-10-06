# AutoEconSentiment

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

`auto-econ-sentiment` is a reproducible pipeline for measuring economic sentiment in central bank and financial text.

## What It Does

- **Load:** CSV, Excel, Parquet or a directory of `.txt` and Markdown files, with dates parsed from filenames
- **Clean:** normalize economic text, tokenize, stem, split into sentences or paragraphs
- **Score:** six central bank and financial dictionaries, plus optional transformer and LLM scoring with explicit label mappings
- **Compare:** every `_net` column lies in $[-1, 1]$ with zero as neutral, so methods share one scale
- **Export:** cleaned text, matched words, counts, probabilities and sentiment scores
- **Audit:** dictionary and model disagreement stays visible for research workflows

## Table of Contents

- [What It Does](#what-it-does)
- [Documentation](#documentation)
- [Quick Start](#quick-start)
- [Transformer Quick Start](#transformer-quick-start)
- [LLM Quick Start](#llm-quick-start)
- [CBS Speeches Demo](#cbs-speeches-demo)
- [Citations](#citations)

## Documentation

- [Architecture](docs/architecture.md)
  - pipeline stages and components
  - config-driven design, tests, directory tree
- [Data and Outputs](docs/data.md)
  - input formats, directory loader
  - output files, column names, score scales
- [Examples](docs/examples.md)
  - YAML and Python runs
  - segmentation, method comparison, transformer setup
- [LLM Scoring](docs/llm_scoring.md)
  - Ollama and OpenAI-compatible providers
  - polarity x confidence, prompts, output columns
- [Roadmap](docs/roadmap.md)
  - v1.0.1 scope, planned features
  - release and tagging steps
- [Transformer notebook](notebooks/autoecon_transformers.ipynb)
  - walkthrough of transformer scoring



## Quick Start

Install from PyPI, adding the extras you need:

```bash
pip install auto-econ-sentiment                 # lexical dictionaries only
pip install 'auto-econ-sentiment[transformers]' # + Hugging Face classifiers
pip install 'auto-econ-sentiment[llm]'          # + Ollama / OpenAI-compatible LLMs
```

Or, working from a clone of this repository:

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
      # FOMC-RoBERTa classifies stance: LABEL_0 dovish, LABEL_1 hawkish,
      # LABEL_2 neutral. Hawkish maps to +1.
      label_mapping:
        LABEL_0: negative
        LABEL_1: positive
        LABEL_2: neutral
      sentiment_values:
        positive: 1
        negative: -1
        neutral: 0
```

The transformer examples in `params.yaml` show the supported model-list format, including each model's label mapping.

Transformer runs export `sentiment_transformer.parquet.gzip` and, for sentence-level aggregation, `sentiment_transformer_sentence_probabilities.parquet.gzip`, which carries each sentence's number and text next to its class probabilities.

## LLM Quick Start

```bash
uv sync --extra llm
```

Set `llm.enabled: true` in `params.yaml` and point it at a local Ollama model or any OpenAI-compatible API. See [LLM Scoring](docs/llm_scoring.md) for providers, prompting and output columns.

## CBS Speeches Demo

Download the CBS central bank speeches dataset and run the sentiment pipeline:

```bash
uv run python -m src.data.cb_speeches_download
uv run python -m src.data.cb_speeches_clean
```

Then open `notebooks/demo_cb_speechs.ipynb` to explore the outputs.

## Citations

To cite `auto-econ-sentiment` itself, use the **Cite this repository** button on GitHub, which reads [CITATION.cff](CITATION.cff). Contributions are welcome; see [CONTRIBUTING](.github/CONTRIBUTING.md).

Dataset:

- Campiglio, E., Deyris, J., Romelli, D., & Scalisi, G. (2025). Warning words in a warming world: Central bank communication and climate change. *European Economic Review*, 105101.

Lexical dictionaries:

- Loughran, T. and B. McDonald (2011). When Is a Liability Not a Liability? Textual Analysis, Dictionaries, and 10-Ks. *The Journal of Finance* 66, 35-65.
- Correa, R., K. Garud, J. Londono, and N. Mislang (2017). Sentiment in Central Bank as Financial Stability Reports. International Finance Discussion Paper 1203.
- Hubert, P. and F. Labondance (2021). The signaling effects of central bank tone. *European Economic Review* 133, 103684.
- Stone, P. J., D. C. Dunphy, and M. S. Smith (1966). *The General Inquirer: A Computer Approach to Content Analysis*.
- Apel, M. and M. Blix Grimaldi (2014). How Informative Are Central Bank Minutes? *Review of Economics* 65(1), 53-76.
- Bennani, H. and M. Neuenkirch (2017). The (Home) Bias of European Central Bankers. *Applied Economics* 49(11), 1114-1131.

Transformer models:

- Pfeifer, M. and V. P. Marohl (2023). CentralBankRoBERTa: A fine-tuned large language model for central bank communications. *The Journal of Finance and Data Science* 9, 100114.
- Shah, A., S. Paturi, and S. Chava (2023). Trillion Dollar Words: A New Financial Dataset, Task & Market Analysis. *Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics*, 6664-6679.
- Araci, D. (2019). FinBERT: Financial Sentiment Analysis with Pre-trained Language Models. arXiv:1908.10063.
- Huang, A. H., H. Wang, and Y. Yang (2023). FinBERT: A Large Language Model for Extracting Information from Financial Text. *Contemporary Accounting Research* 40(2), 806-841.

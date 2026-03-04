# AutoEconSentiment

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

A streamlined, production-ready pipeline for extracting and analyzing economic sentiment from textual data. Focused on high-performance lexical sentiment analysis using established financial dictionaries.

## Features

- **Robust Data Cleaning**: Automatically handles HTML, unicode normalization, special character encodings, percent fixes, header/footer stripping, tokenization and stemming.
- **Lexical Sentiment Execution**: Computes word-level and sentence-level sentiment counts and aggregates (methods: `pos`, `alt`) across multiple dictionaries:
  - Correa (Financial Stability)
  - Hubert (Central Bank Tone)
  - Loughran-McDonald (LM)
  - General Inquirer (HIV)
- **Pipeline Orchestration**: Simplified 1-file pipeline orchestrator (`AutoEconSentiment`) managed by YAML configuration.

## Installation

Ensure you have `uv` installed, then synchronize the environment:

```bash
uv sync
```

## Quick Start

### 1. Run Tests & Synthetic Data

You can verify the pipeline functionality using built-in synthetic test data:

```bash
uv run python -m auto_econ_sentiment.pipeline --test
```

Or execute the unit test suite with `pytest`:

```bash
uv run pytest
```

### 2. Run Pipeline from YAML

Configure your data inputs, cleaning rules, and target dictionaries in `params.yaml`, then run the entire analysis pipeline:

```bash
uv run python -m auto_econ_sentiment.pipeline
```

## Pipeline Components

- **Clean & Load**: `src.clean.text_loader`, `src.clean.text_clean`
- **Sentiment Models**: `src.models.sentiment_lexical`, `src.models.modules.sentiment_lexical_gpt`
- **Main Entrypoint**: `src/auto_econ_sentiment/pipeline.py`


## Citations

### Lexical Dictionaries
- **Loughran-McDonald (LM)**: Loughran, T. and B. Mcdonald (2011). When Is a Liability Not a Liability? Textual Analysis, Dictionaries, and 10-Ks. *The Journal of Finance* 66, 35–65.
- **Correa**: Correa, R., K. Garud, J. Londono, and N. Mislang (2017). Sentiment in Central Bank as Financial Stability Reports. *Board of Governors of the Federal Reserve System Research Series*. International Finance Discussion Paper 1203.
- **Hubert**: Hubert, P. and F. Labondance (2021). The signaling effects of central bank tone. *European Economic Review* 133, 103684.
- **General Inquirer (HIV)**: 
  - Stone, Philip J., Dexter C. Dunphy, and Marshall S. Smith. "The general inquirer: A computer approach to content analysis." (1966).
  - Lasswell, Harold Dwight, and Nathan Constantin Leites. "Language of politics: Studies in quantitative semantics." (1966).

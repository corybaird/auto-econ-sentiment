# arXiv Paper Plan

This document sketches a short, publishable first paper for `auto-econ-sentiment`, followed by a clear research and software roadmap for custom dictionaries and transformer-based updates.

## Working Title

**AutoEconSentiment: Reproducible Dictionary-Based Sentiment Analysis for Central Bank Communication**

## Core Thesis

Economic sentiment estimates are highly sensitive to the lexical dictionary and aggregation method used. `auto-econ-sentiment` makes this variation visible, reproducible, and extensible by running multiple established economic and central-bank dictionaries through one configuration-driven pipeline.

The first arXiv version should stay intentionally narrow: publish the lexical package, document the empirical variation across dictionaries, and show that these measurement choices materially affect downstream tasks. A later version can add transformer models as an example of how computational social science software can be updated transparently instead of frozen at publication.

## Paper Versioning Strategy

### v1: Lexical Methods Paper

Goal: publish a compact, credible first version using only the current lexical system.

Main contributions:

1. A reusable Python package for cleaning, scoring, and exporting economic text sentiment.
2. A unified implementation of several central-bank and financial sentiment dictionaries.
3. Evidence that lexical sentiment estimates vary substantially across dictionaries and scoring formulas.
4. A downstream-task demonstration showing that dictionary choice changes empirical conclusions.
5. A reproducible workflow using YAML configs, tests, and notebooks.

Keep this version tight. Avoid promising transformer results in the main empirical contribution. Mention transformers only as planned extensibility.

### v2: Transformer Extension Paper

Goal: update the same paper/software line with transformer sentiment models.

Main additions:

1. Optional Hugging Face transformer backend.
2. Sentence-level and full-document transformer scoring.
3. Comparison of lexical and transformer scores on the same corpus.
4. Stability analysis: when do lexical and transformer methods agree, and where do they diverge?
5. Documentation showing how the research artifact evolves across versions.

Seed implementation:

`/Users/cory/Desktop/github/Econ_Text_Algos/src/models/sentiment_transformers.py`

Target location:

`src/auto_econ_sentiment/models/sentiment_transformers.py`

## Suggested Paper Outline

### 1. Introduction

Problem: economic text sentiment is widely used, but measurements often depend on hidden preprocessing, one-off dictionary choices, and hard-to-reproduce scripts.

Argument: the package is useful not because it claims one dictionary is universally correct, but because it exposes the measurement uncertainty created by dictionary choice.

Short framing:

- Central bank communication is a high-value test case because tone is often linked to financial markets, policy expectations, inflation narratives, and uncertainty.
- Lexical methods remain common because they are transparent, fast, and easy to audit.
- Transparency does not eliminate ambiguity: dictionaries encode different concepts of positive and negative economic language.

### 2. Package Design

Describe the current pipeline:

1. Load text from CSV or parquet.
2. Clean and normalize text.
3. Tokenize and optionally stem.
4. Score each document across configured dictionaries.
5. Export cleaned text, dictionary counts, matched words, and final sentiment scores.

Reference implementation files:

- `src/auto_econ_sentiment/pipeline.py`
- `src/auto_econ_sentiment/models/sentiment_lexical.py`
- `src/auto_econ_sentiment/data/lexical_master_dict.yaml`
- `params.yaml`
- `docs/architecture.md`

Emphasize design principles:

- Configuration-driven analysis.
- Multiple dictionaries in one run.
- Inspectable word counts and matched terms.
- Reproducible export paths.
- Tests for loader, cleaner, and scorer behavior.

### 3. Data

Primary demonstration dataset:

- CBS Central Bank Speeches Dataset.
- Demo notebook: `notebooks/demo_cb_speechs.ipynb`.
- Current demo scale: about 35K speeches across 143 central banks.

Possible compact v1 sample:

- Use the full central-bank speech corpus for headline descriptive results.
- Use a smaller reproducible FOMC or central-bank subset for downstream-task examples if runtime or data distribution is a concern.

### 4. Lexical Dictionaries and Scores

Document the built-in dictionaries:

- Loughran-McDonald (`lm`)
- Correa et al. (`correa`)
- Hubert-Labondance (`hubert`)
- General Inquirer / Harvard IV (`hiv`)
- Apel-Blix Grimaldi (`ap`)
- Bennani-Neuenkirch (`bn`)

Document scoring methods:

- `posneg`: sentiment normalized by matched positive and negative terms.
- `allwords`: sentiment normalized by total document tokens.

Core empirical question:

How much does estimated economic sentiment change when the same text is scored with different dictionaries and aggregation methods?

Recommended figures:

1. Distribution of sentiment scores by dictionary and method.
2. Pairwise correlation heatmap across all dictionary-method combinations.
3. Rolling central-bank sentiment over time for several dictionaries.
4. Rank disagreement plot: which speeches/countries/months look most different depending on dictionary?
5. Matched-word coverage by dictionary.

The existing `demo_cb_speechs.ipynb` already contains useful building blocks for descriptive statistics, distributions, and time-series plots.

### 5. Downstream Task Demonstrations

The downstream section should be simple and convincing. The goal is to show that dictionary variation is not cosmetic.

Candidate tasks:

1. **Nowcasting or forecasting macro variables**
   - Outcome examples: inflation, unemployment, industrial production, policy rates, or GDP growth.
   - Model: lagged outcome baseline vs. baseline plus sentiment.
   - Compare performance across dictionary-method combinations.

2. **Policy communication classification**
   - Outcome examples: rate hike/cut/hold, hawkish/dovish labels, tightening/easing periods.
   - Model: logistic regression or regularized classifier.
   - Show that dictionary choice changes accuracy, coefficients, or selected episodes.

3. **Market reaction regression**
   - Outcome examples: short-window bond yield changes, equity index moves, exchange-rate changes.
   - Model: event-study regression around speeches or statements.
   - Show coefficient sign, size, and significance vary across sentiment measures.

Recommended v1 path:

Start with one lightweight downstream task that can be reproduced from available local or easily downloadable data. A compact regression/classification table is enough for the first arXiv version.

Suggested table:

| Sentiment Measure | Outcome | Model | Coefficient / Score | Direction | Notes |
| --- | --- | --- | --- | --- | --- |
| `lm_sentiment_allwords` | downstream target | baseline + sentiment | TBD | TBD | lexical finance dictionary |
| `hubert_sentiment_allwords` | downstream target | baseline + sentiment | TBD | TBD | central-bank tone dictionary |
| `correa_sentiment_allwords` | downstream target | baseline + sentiment | TBD | TBD | financial stability dictionary |

### 6. User Extensibility

This should be a software contribution in v1 and a documentation priority before release.

Add a user-facing guide:

`docs/custom_dictionaries.md`

Guide contents:

1. Required dictionary schema.
2. How to add a dictionary to `lexical_master_dict.yaml`.
3. Difference between stemmed and unstemmed dictionaries.
4. How to enable the dictionary in `params.yaml`.
5. Expected output columns.
6. Minimal test showing the custom dictionary produces non-empty counts.

Proposed dictionary schema:

```yaml
my_dictionary:
  positive:
    - growth
    - resilient
  negative:
    - recession
    - fragile
```

Recommended follow-up API:

- Add a loader for external dictionary YAML files so users do not need to edit the packaged master dictionary.
- Add validation errors for missing `positive` or `negative` keys.
- Add documentation examples using a tiny custom dictionary and a tiny text fixture.

### 7. Transformer Roadmap

Transformers should be documented as the next research update, not the core v1 claim.

Implementation plan:

1. Move the old transformer class into the package namespace.
2. Replace old imports such as `from src.models.sentiment_base import SentimentBase` with `from auto_econ_sentiment.models.sentiment_base import SentimentBase`.
3. Add optional dependencies, for example `auto-econ-sentiment[transformers]`.
4. Add YAML config support for model name, short name, number of labels, batch size, max length, cutoff, and inference mode.
5. Add sentence-level and full-document scoring methods to the main pipeline.
6. Add tests that mock or use a tiny local model to avoid heavyweight CI downloads.
7. Add documentation showing how transformer outputs differ from lexical outputs.

Candidate config shape:

```yaml
models:
  transformer:
    enabled: false
    model_name: Moritz-Pfeifer/CentralBankRoBERTa-sentiment-classifier
    model_name_short: cbroberta
    num_labels: 2
    max_length: 512
    batch_size: 8
    sentiment_by_sentence: true
    sentiment_by_sentence_cutoff: 0.7
```

## Minimum Repo Work Before v1 Paper

1. Create a reproducible analysis script that generates the paper tables and figures. Because `reports/overleaf/` is a separate git repository connected to Overleaf, keep the script in the main repo at `src/research_paper/make_figures.py` and write final assets into `reports/overleaf/figures/` so Overleaf can render them.
2. Clean up `notebooks/demo_cb_speechs.ipynb` so paths work from the repo root and outputs are not stale.
3. Add `docs/custom_dictionaries.md`.
4. Add a small paper-oriented config file for central-bank speeches or the chosen reproducible subset.
5. Add an Overleaf asset convention: rendered paper figures live in `reports/overleaf/figures/`, while the code that creates them lives in `src/research_paper/`.
6. Add a citation file (`CITATION.cff`) before arXiv submission.
7. Confirm package install and tests from a clean environment.

## Lean arXiv v1 Checklist

Paper:

- 6 to 8 pages excluding references.
- One architecture diagram.
- One dictionary summary table.
- Two lexical-variation figures.
- One downstream-task table.
- Short limitations section.
- Clear software availability statement.

Repository:

- Tagged release, ideally `v0.1.x`.
- README quickstart works.
- `uv run pytest` passes.
- Demo notebook can be re-run.
- Paper figures can be regenerated from scripts or notebooks.

## Limitations to State Clearly

- Lexical scores are transparent but not context-aware.
- Dictionary coverage varies by corpus, country, and time period.
- Stemming changes matching behavior and should be reported.
- Cross-lingual speeches require careful handling; the current lexical dictionaries are English-centered unless translated/preprocessed.
- The package reports measurement variation; it does not declare a single universally correct sentiment measure.

## Proposed One-Paragraph Abstract

This paper introduces `auto-econ-sentiment`, a reproducible Python package for dictionary-based sentiment analysis of economic and central-bank text. The package standardizes loading, cleaning, lexical scoring, and export workflows while allowing users to compare multiple established financial and monetary-policy dictionaries in a single configuration-driven pipeline. Applying the package to central-bank speeches, we show that estimated sentiment varies substantially across lexical dictionaries and aggregation methods, and that this variation can affect downstream empirical tasks. The package is designed as an extensible research artifact: the initial release focuses on transparent lexical methods, while future releases add user-defined dictionaries and transformer-based sentiment models under the same reproducible interface.

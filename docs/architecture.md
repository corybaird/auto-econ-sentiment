# AutoEconSentiment System Architecture and File Structure

`auto-econ-sentiment` is a modular, configuration-driven pipeline for extracting and analyzing economic sentiment from text data. It keeps lexical scoring as the stable baseline while allowing optional transformer sentiment models to run through the same load, clean, score, and export workflow.

## 1. High-Level Logic Flow

The library operates through a sequence of well-defined stages, orchestrated by the `AutoEconSentiment` pipeline class. Raw text is loaded and cleaned once, then can be scored by lexical dictionaries, optional transformer classifiers, or both.

```mermaid
graph TD
    A[params.yaml] --> B[AutoEconSentiment Orchestrator]
    B --> C[Stage: Load Data]
    C --> D[TextLoader]
    D --> E[Raw DataFrame]
    E --> F[Stage: Clean Text]
    F --> G[TextCleaner]
    G --> H[Cleaned & Tokenized DataFrame]
    H --> I[Stage: Sentiment Analysis]
    I --> J[SentimentLexical Models]
    I --> N[Optional SentimentTransformers Models]
    J --> K[Lexical Scores]
    N --> O[Transformer Labels, Probabilities, Shares]
    K --> P[Combined Sentiment Tables]
    O --> P
    P --> L[Stage: Export]
    L --> M[reports/ / data/sentiment/]
```

## 2. Component Tree (Architecture Tree)

The system is organized into specialized layers governed by the pipeline.

```mermaid
graph TD
    Pipeline[AutoEconSentiment Pipeline] --> DataLayer[Data Layer]
    Pipeline --> CleanLayer[Cleaning Layer]
    Pipeline --> ModelLayer[Modeling Layer]

    DataLayer --> TextLoader
    
    CleanLayer --> TextCleaner
    CleanLayer --> TextSegmenter[Text Segmenter]
    CleanLayer --> TextViz[Text Visualizer]

    ModelLayer --> SentimentBase[Base Sentiment Model]
    ModelLayer --> SentimentLexical[Lexical Sentiment Scorer]
    ModelLayer --> SentimentTransformers[Optional Transformer Sentiment Scorer]
```

## 3. Library Components (`src/auto_econ_sentiment/`)

### 3.1 `pipeline.py` (Main Orchestrator)
The `AutoEconSentiment` class is the primary entry point. It orchestrates loading, cleaning, lexical scoring, optional transformer scoring, and exports via its `run()` method. It accepts `import_file_path`, `text_column`, `date_column`, and `export_path`. It can also be invoked from the command line with `--test` for a built-in synthetic data run.

Pipeline state is kept explicit:

- `df_raw`
- `df_clean`
- `df_sent_lexical`
- `df_sent_transformer`
- `df_transformer_sentence_probabilities`

The transformer config parser accepts both the current package format and the older `Econ_Text_Algos` model-list format using `name`, `short_name`, `label_mapping`, and `sentiment_values`.

### 3.2 `clean/text_loader.py` (Data Loader)
`TextLoader` handles loading input data from `csv`, `parquet`, and `parquet.gzip` formats. It ensures structural requirements are met (e.g., verifying `text_column` and `date_column` are present) and returns a clean copy of the DataFrame.

### 3.3 `clean/text_clean.py` (Text Cleaner)
`TextCleaner` applies a configurable multi-step cleaning pipeline:
- HTML stripping and unicode normalization
- British-to-American English conversion (`clean/references/british_2_american.py`)
- Number and percentage normalization
- Configurable header/boilerplate removal
- Word tokenization (splits text into token lists)
- Porter stemming (reduces tokens to root forms for stemmed dictionaries)

Cleaned text is assigned a unique `id_text` to maintain alignment.

### 3.4 `clean/text_segmentation.py` (Text Segmenter)
`TextSegmenter` splits documents into sentence-level rows for sentence-level transformer scoring. It utilizes NLTK `sent_tokenize` with a regex sentence-boundary fallback when NLTK or `punkt` data is unavailable. Configurable `min_chars` drops sentence fragments below a specified length, and document indices are preserved across exploded sentence rows.

### 3.5 `models/sentiment_lexical.py` (Lexical Sentiment Model)
Computes bag-of-words sentiment against multiple central bank and financial dictionaries. Employs user-selected aggregation methods:
*   **`posneg`**: Normalizes sentiment by the total count of matched sentiment words. 
    $$ \text{Sentiment}_{\text{posneg}} = 1 + \frac{N_{\text{pos}} - N_{\text{neg}}}{N_{\text{pos}} + N_{\text{neg}}} $$
*   **`allwords`**: Normalizes sentiment by the total number of tokens in the entire document.
    $$ \text{Sentiment}_{\text{allwords}} = 1 + \frac{N_{\text{pos}} - N_{\text{neg}}}{N_{\text{total}}} $$

### 3.6 `models/sentiment_transformers.py` (Optional Transformer Sentiment Model)
`SentimentTransformers` wraps Hugging Face sequence-classification models behind optional dependencies. `torch` and `transformers` are imported lazily so the base package can still run lexical sentiment without installing or downloading transformer models.

Transformer features:

- explicit `label_map` validation,
- batched model inference,
- document-level scoring with `aggregation: byalltext`,
- sentence-level scoring with `aggregation: bysentence`,
- probability exports,
- sentence aggregation by `id_text`,
- harmonized positive/neutral/negative counts, shares, and net sentiment when `output_schema: shares` is enabled.

The transformer path treats labels as model-specific. Generic model labels such as `LABEL_0` are mapped through configuration rather than hard-coded in the model class.

### 3.7 `models/sentiment_base.py` (Abstract Base)
`SentimentBase` is the abstract base class for sentiment models, providing shared input DataFrame handling and the `text_column` interface.

### 3.8 `data/lexical_master_dict.yaml` (Dictionary Definitions)
Master YAML file containing the positive/negative word lists for all 6 supported dictionaries: `hubert`, `lm`, `hiv`, `correa`, `bn`, `ap`.

### 3.9 `exceptions.py` (Custom Exceptions)
Defines structured error classes throughout the pipeline:
- `AutoEconSentimentError`: Base exception for the package.
- `ConfigurationError`: Raised for invalid or conflicting pipeline configurations.
- `DataLoadError`: Raised when input data cannot be loaded or validated.
- `SentimentAnalysisError`: Raised during failures in model execution or scoring.

### 3.10 `utils/load_yaml.py` (YAML Config Loader)
`load_yaml_config()` loads and validates pipeline configuration from a YAML file using `yaml.safe_load()`.

### 3.11 `utils/paths.py` (Path Utilities)
Shared path resolution helpers.

### 3.12 `clean/text_viz.py` (Cleaning Visualizer)
Utilities for visualizing text before and after cleaning (for exploratory and debugging use).

## 4. Tests (`tests/`)

The test suite covers the full pipeline across six dedicated modules in `tests/`:

- `tests/test_pipeline.py`: Full-pipeline integration, `TextLoader` validation, `TextCleaner` normalization, lexical scoring, and package public API imports.
- `tests/test_sentiment_transformers.py`: Lazy imports without optional dependencies, explicit `label_map` validation, and prediction post-processing.
- `tests/test_text_segmentation.py`: Sentence splitting via NLTK/regex, minimum character threshold handling, and index alignment across sentence rows.
- `tests/test_transformer_config.py`: Configuration resolution across modern flat keys and legacy nested YAML schemas.
- `tests/test_transformer_sentence_aggregation.py`: Sentence aggregation arithmetic, confident label counting, and per-model cutoff overrides.
- `tests/test_transformers.py`: Isolated model test doubles, harmonized count/share outputs, and formula verification.

Run the test suite with:

```bash
uv run --extra dev pytest
```

## 5. Project Directory Tree

The repository maintains strict boundaries between source code, reference configurations, original datasets, and generated outputs.

```text
.
├── data/                          # Immutable and derived data (gitignored)
│   ├── raw/                       # Original, untouched source files
│   │   ├── basic_tests/           # Datasets for sanity checks and unit tests
│   │   └── speeches/              # Large downloaded datasets (e.g., CBS Speeches)
│   └── sentiment/                 # Generated sentiment output tables and cleaned text
├── docs/                          # Architectural and user documentation
├── notebooks/                     # Exploratory analysis and pipeline demonstrations
├── references/                    # CONFIGURATION CENTER
│   └── configs/                   # YAML configuration files (e.g., params_cb_speeches.yaml)
├── reports/                       # Automated model outputs and pipeline logs
├── src/                           # SOURCE CODE
│   ├── auto_econ_sentiment/       # Core Python library package
│   │   ├── clean/                 # Data loading, text cleaning, and segmentation logic
│   │   ├── data/                  # Built-in master dictionaries and configuration
│   │   ├── models/                # Lexical and optional transformer models
│   │   ├── utils/                 # Path handling and YAML parsing helpers
│   │   └── pipeline.py            # Main pipeline orchestrator
│   └── data/                      # Data fetching and ingestion scripts
└── tests/                         # Unit and integration test suite
```

## 6. Configuration-Driven Design

The pipeline relies heavily on the `params.yaml` construct to guarantee reproducibility. This allows you to:
1.  Swap out target text or date columns rapidly.
2.  Enable/disable specific cleaning procedures (e.g., `stem`, `tokenize`, `clean_numbers_percentages`).
3.  Designate specific lexical dictionaries (`unstemmed` vs. `stemmed`) and aggregation methods (`posneg`, `allwords`).
4.  Enable optional transformer models with `transformer.enabled` (top-level key; `models.transformer.enabled` supported via backward-compatible fallback).
5.  List multiple transformer models using either the package-native `model_name` / `model_name_short` / `label_map` format or the original `name` / `short_name` / `label_mapping` / `sentiment_values` format.
6.  Choose transformer aggregation modes (`byalltext`, `bysentence`, or aliases such as `full_text` and `sentence_pos`).
7.  Run entirely different datasets without modifying the core `pipeline.py` Python logic.

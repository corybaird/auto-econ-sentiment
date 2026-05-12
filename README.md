# AutoEconSentiment

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

A streamlined, production-ready pipeline for extracting and analyzing economic sentiment from text data using established central bank and financial dictionaries (lexical methods).

---

## Q. Quick Start

### Q.1 Install

Ensure you have `uv` installed, then synchronize the environment:

```bash
uv sync
```

### Q.2 Run on Your Own Data (Python API)

```python
from auto_econ_sentiment.pipeline import AutoEconSentiment

analyzer = AutoEconSentiment(
    import_file_path="data/raw/basic_tests/monetary_policy_statement.parquet.gzip",
    text_column="text",
    date_column="date",
    export_path="data/sentiment/basic_tests/"
)
analyzer.run(
    clean_config={"tokenize": True, "stem": True},
    dictionaries={"unstemmed": ["correa", "hubert", "lm", "hiv"], "stemmed": ["ap", "bn"]},
    aggregation_methods=["posneg", "allwords"],
    export_results=True
)
```

### Q.3 Run from YAML Config

Configure inputs, cleaning rules, and dictionaries in `params.yaml`, then run:

```bash
uv run python -m src.auto_econ_sentiment.pipeline
```

### Q.4 Run the CBS Speeches Demo

Download ~35K central bank speeches and run sentiment analysis across all 143 central banks:

```bash
# 1. Download the CBS dataset and split by central bank
uv run python -m src.data.cb_speeches_download

# 2. Run the sentiment pipeline over all banks
uv run python -m src.data.cb_speeches_clean
```

Then open `notebooks/demo_cb_speechs.ipynb` to explore the results interactively.

---

## 1. Key Features

- 🧹 **Robust NLP Preprocessing**: Out-of-the-box text normalization including HTML stripping, unicode correction, configurable boilerplate/header removal, tokenization, and Porter stemming.
- 📊 **Domain-Specific Sentiment**: Built-in support for 6 specialized lexical dictionaries (e.g., Loughran-McDonald, Hubert, Correa) tailored specifically for macroeconomic and central bank text analysis.
- ⚙️ **Reproducible Orchestration**: Fully YAML-driven pipeline (`params.yaml`) ensuring zero hard-coded values, enabling seamless swapping of datasets, cleaning rules, and target models.
- 🚀 **Scalable Architecture**: Designed to efficiently ingest, clean, and score tens of thousands of documents, as demonstrated by the built-in end-to-end demo processing 35K+ speeches across 143 central banks.
- 🔌 **Modular & Extensible**: Decoupled `Loader`, `Cleaner`, and `Scorer` interfaces allowing researchers to easily plug in custom datasets or proprietary sentiment algorithms.

---

## 2. Data

### 2.1 Input Data (`data/raw/`)

Contains immutable, original input data. Never modified directly. **Note: Raw datasets are large and excluded from version control (`.gitignore`). You must download or generate them locally using the scripts in `src/data/`.**

| Path                                                          | Description                                                                                                                                                                                                                                                                                                                                                                                                   |
| ------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `data/raw/basic_tests/monetary_policy_statement.parquet.gzip` | FOMC monetary policy statements. Used as the primary test and demo dataset for `params.yaml` and `--test` mode. Columns: `text`, `date`.                                                                                                                                                                                                                                                                      |
| `data/raw/basic_tests/statements_speeches.parquet.gzip`       | A small mixed sample of central bank statements and speeches. Used for quick pipeline validation.                                                                                                                                                                                                                                                                                                             |
| `data/raw/speeches/CBNAME.parquet.gzip`                       | The full [CBS Central Bank Speeches Dataset](https://www.cbspeeches.com/) (~35K speeches, 143 central banks, 1986-2023), split into one file per central bank (see citation below). Generated by `src/data/cb_speeches_download.py`. Columns: `URL`, `PDF`, `Title`, `Subtitle`, `Date`, `Authorname`, `Role`, `Gender`, `CentralBank`, `Country`, `text`, `text_original`, `Filename`, `Language`, `Source`. |

### 2.2 Sentiment Outputs (`data/sentiment/`)

Contains all outputs generated by the `AutoEconSentiment` pipeline.

| Path | Description |
|------|-------------|
| `basic_tests/cleaned.parquet.gzip` | Cleaned and tokenized text from the basic test dataset. |
| `basic_tests/sentiment_all_results.csv` | Combined sentiment results for the basic test dataset across all dictionaries and methods. |
| `cb_speeches/CBNAME/cleaned.parquet.gzip` | Cleaned and tokenized speeches for each central bank. |
| `cb_speeches/CBNAME/sentiment_all_results.csv` | Final sentiment scores per speech for each central bank, with columns for each `{dictionary}_{method}_sentiment` combination. |

#### Output Structure (`sentiment_all_results.csv`)

The final output DataFrame contains the following core columns for each configured dictionary (e.g., `hubert`):

| Column Pattern               | Description                                                                         |
| ---------------------------- | ----------------------------------------------------------------------------------- |
| `{dict}_counttoken_positive` | Total count of positive words identified by the dictionary.                         |
| `{dict}_counttoken_negative` | Total count of negative words identified by the dictionary.                         |
| `{dict}_counttoken_total`    | Total tokens in the document (only present if using `allwords` method).             |
| `{dict}_words_positive`      | JSON-like mapping of the specific positive words found and their counts.            |
| `{dict}_words_negative`      | JSON-like mapping of the specific negative words found and their counts.            |
| `{dict}_sentiment_{method}`  | Final computed sentiment score using the specified method (`posneg` or `allwords`). |

### 2.3 Configuration (`references/configs/`)

The pipeline relies on YAML configuration files to define inputs, cleaning rules, and target dictionaries. Importantly, these configuration files dictate exactly where the final sentiment data is saved, directly linking the process to the outputs described in Section 2.2.

| File | Description | Input Location (`file_path`) | Output Location (`export_path`) |
|------|-------------|------------------------------|---------------------------------|
| `params.yaml` | Main pipeline configuration for the basic FOMC test dataset. | `data/raw/basic_tests/monetary_policy_statement.parquet.gzip` | `data/sentiment/basic_tests/` |
| `references/configs/params_cb_speeches.yaml` | Pipeline configuration for the CBS central bank speeches demo. | `data/raw/cb_speeches.parquet.gzip` | `data/sentiment/cb_speeches/` |

**Example configuration structure from `params.yaml`:**

```yaml
input:
  file_path: "data/raw/basic_tests/monetary_policy_statement.parquet.gzip"
  text_column: text
  date_column: date

output:
  export_path: data/sentiment/basic_tests/
  export_results: true

models:
  lexical:
    #text_column_lexical: 'text_clean'
    dictionaries:
      unstemmed: [correa, hubert, lm, hiv]
      stemmed: [bn, ap]
    aggregation_methods: [posneg, allwords]
```

---

## 3. Architecture and Documentation

For detailed information about the inner workings of the library, the individual components, and the test suite, please refer to our architectural documentation:
- [System Architecture & File Structure](docs/architecture.md)

---

## 4. Citations

### Datasets

- **CBS Central Bank Speeches Dataset**: Campiglio, E., Deyris, J., Romelli, D., & Scalisi, G. (2025). Warning words in a warming world: Central bank communication and climate change. European Economic Review, 105101. Available at [www.cbspeeches.com](https://www.cbspeeches.com/).

### Lexical Dictionaries

- **Loughran-McDonald (LM)**: Loughran, T. and B. Mcdonald (2011). When Is a Liability Not a Liability? Textual Analysis, Dictionaries, and 10-Ks. *The Journal of Finance* 66, 35–65.
- **Correa**: Correa, R., K. Garud, J. Londono, and N. Mislang (2017). Sentiment in Central Bank as Financial Stability Reports. *Board of Governors of the Federal Reserve System Research Series*. International Finance Discussion Paper 1203.
- **Hubert**: Hubert, P. and F. Labondance (2021). The signaling effects of central bank tone. *European Economic Review* 133, 103684.
- **General Inquirer (HIV)**:
  - Stone, Philip J., Dexter C. Dunphy, and Marshall S. Smith. "The general inquirer: A computer approach to content analysis." (1966).
  - Lasswell, Harold Dwight, and Nathan Constantin Leites. "Language of politics: Studies in quantitative semantics." (1966).
- **Apel-Blix Grimaldi (AP)**: Apel, M. and M. Blix Grimaldi (2014). How Informative Are Central Bank Minutes? *Review of Economics* 65(1), 53-76.
- **Bennani-Neuenkirch (BN)**: Bennani, H. and M. Neuenkirch (2017). The (Home) Bias of European Central Bankers: New Evidence Based on Speeches. *Applied Economics* 49(11), 1114-1131.

---

## 5. Scripts and Notebooks

### 5.1 `src/data/` (Data Pipelines & Orchestration)
The `src/data/` folder contains orchestration scripts used to fetch external datasets and orchestrate sentiment analysis runs. These scripts function as standalone execution entry points.
- `cb_speeches_download.py`: Ingests the central bank speeches dataset from cbsspeeches.org and partitions the data into `data/raw/`.
- `cb_speeches_clean.py`: Orchestrates the `AutoEconSentiment` pipeline specifically for the CBS speeches dataset, producing the sentiment outputs locally.

### 5.2 `notebooks/` (Exploration & Demos)
The `notebooks/` folder contains exploratory data analysis (EDA) and demonstration Jupyter Notebooks. These notebooks consume the processed data generated by the `src/data/` orchestration scripts. 
- `autoecon_demo.ipynb`: A general walkthrough demo.
- `demo_cb_speechs.ipynb`: An interactive output visualization notebook showcasing results processed by `cb_speeches_clean.py`.

### 5.3 `src/auto_econ_sentiment/` (Core Library)
The core library contains all the data loading, text cleaning, and sentiment analysis modules. For a full breakdown of these components, see the [Architecture Document](docs/architecture.md).

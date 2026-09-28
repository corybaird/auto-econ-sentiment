# Replicating the Paper

This folder rebuilds every figure, table and statistic in *AutoEconSentiment: Reproducible Infrastructure for Sentiment Analysis* from raw text, using the released `auto-econ-sentiment` package (v1.0.1).

```bash
uv sync --extra transformers --extra research
uv run python -m src.research_paper
```

A full run scores 6,693 policy statements and 6,607 US speeches with three transformer models. It takes about an hour on a single modern GPU. Every central bank and every year is cached as soon as it finishes, so an interrupted run resumes where it stopped.

## Inputs

| Input | Location | How to obtain |
| --- | --- | --- |
| Monetary Policy Statement Database | `data/raw/statements/{bank}/{YYYY-MM-DD}_*.txt` | Baird (2026) |
| CBS central bank speeches | `data/raw/speeches/*.parquet.gzip` | `uv run python -m src.data.cb_speeches_download` then `uv run python -m src.data.cb_speeches_clean` |
| FOMC statement sample (Figure 2) | `data/raw/basic_tests/monetary_policy_statement.parquet.gzip` | Included in the repository |
| FRED macro series (VAR) | `data/research/fred_macro_monthly.parquet.gzip` | Fetched on first run; needs `FRED_API_KEY` in the environment or `.env` |

FOMC-RoBERTa (`gtfintechlab/FOMC-RoBERTa`) is a gated model on Hugging Face. Request access on its model page and log in with `hf auth login` before the first run. Once every model is in the local cache, `HF_HUB_OFFLINE=1` runs without network access.

## Stages

`--stages` runs a subset, in this order:

| Stage | Produces | Reads |
| --- | --- | --- |
| `statements` | Lexical and transformer scores for every statement since 1994 | the statement database |
| `speeches` | The same scores for every US speech | the speech corpus |
| `figures` | Figures 1 to 6 | statement scores, FOMC sample |
| `var` | Figure 7 | speech scores, FRED |
| `tables` | Corpus summary and Tables 1 and 2 | both raw corpora |
| `statistics` | `reports/paper_statistics.md`: every number quoted in the text, and the Table 3 sentence audit | statement scores, the VAR |

The scoring stages reuse cached units; the later stages only read the cached panels, so after one full run

```bash
uv run python -m src.research_paper --stages figures,var,tables,statistics
```

rebuilds the exhibits in a few minutes. `--force` rescores everything.

## Outputs

- Figures go to `reports/overleaf/figures/` and tables to `reports/overleaf/tables/`, where `main.tex` includes them.
- Scores and cleaned text go to `data/research/`. Deleting that folder, apart from the FRED file if you want to keep the same vintage of macro data, forces a clean run.

## Code

| Module | Responsibility |
| --- | --- |
| `paper_configuration.yaml` | Every setting: corpora, scoring, measures, figures, VAR, exhibits |
| `config.py` | `PaperConfig`: section access, path resolution, the package's transformer settings |
| `corpora.py` | `StatementCorpus`, `SpeechCorpus`: reading the raw text |
| `scoring.py` | `PackageScorer` runs the package; `StatementPanel` and `SpeechPanel` cache it per bank and per year; `HeaderExperiment` for Figure 2 |
| `measures.py` | `SentimentMeasures`: the compared columns, monthly series and correlations |
| `econometrics.py` | `FredMacro`, `MacroPanel`, `ImpulseResponses` |
| `exhibits/` | `FigureStyle`, the figure classes and `PaperTables` |
| `statistics.py` | `PaperStatistics` and `SentenceAudit` |
| `pipeline.py` | `PaperPipeline`: the stages above |

All scoring goes through `PackageScorer`, which calls `AutoEconSentiment.run` exactly as a user of the package would; the paper code adds no scoring logic of its own.

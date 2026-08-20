# Changelog

All notable changes to this project will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added
- Added `ParagraphSegmenter` in `clean/text_segmentation.py` to split documents into paragraph-level rows by blank lines, with configurable `paragraph_number` column.
- Added `tokenizer_name` attribute on `TextSegmenter` to expose whether `nltk_punkt` or `regex_fallback` is active.
- Added `require_nltk` parameter on `TextSegmenter` to fail fast when NLTK is unavailable.
- Added `drop_invalid` parameter on `TextSegmenter` to filter non-sentential fragments (rosters, headers, no-verb items).
- Added abbreviation and single-initial protection to the regex fallback tokenizer, preventing false splits on names like `St. Louis`, `Susan S. Bies`.
- Added newline-boundary detection to both tokenizer paths, fixing 420 missed boundaries across `\n` after `.!?` or between lowercase/`:` and a capital.
- Added fragment merging (`_merge_incomplete_sentences`) to rejoin lowercase- and punctuation-initial fragments into the preceding sentence.

### Changed
- Changed `TextSegmenter.split_text` to apply newline pre-splitting, abbreviation protection, and fragment merging in a unified pipeline before both NLTK and regex tokenizers.
- Expanded `_FALLBACK_BOUNDARY` regex to include opening quote/bracket in the lookahead character class.
- Added directory-of-`.txt` corpus loading in `TextLoader`, supporting flat directories as well as categorized subdirectories with `group_column`.
- Added regex-based filename date parsing (`filename_date_pattern`), retaining unparseable dates as `NaT` with a logged warning.
- Added `recursive` flag for directory traversal and automatic unique `id_column` generation for text corpora.
- Added unit test suite for `TextLoader` covering directory loading, date parsing, group column attribution, recursive scanning, and edge cases in `tests/test_text_loader.py`.
- Updated `docs/data.md` documentation for directory-of-txt loading and constructor parameters.

## [0.3.0] - 2026-08-19

### Added
- Added `TextSegmenter` in `clean/text_segmentation.py` to split documents into sentence-level rows, using NLTK `sent_tokenize` with a regex boundary fallback when NLTK or its `punkt` data is unavailable.
- Added `min_sentence_chars` and `sentence_probability_cutoff` as transformer configuration keys, resolvable at the transformer level with per-model overrides.
- Added `resolve_lexical_config` and `resolve_transformer_config` helpers that read flat top-level config keys and fall back to the legacy nested layouts.
- Added tests covering sentence segmentation, sentence-level aggregation arithmetic, and config resolution.

### Changed
- Changed `params.yaml` to promote `lexical` and `transformer` to top-level keys, replacing the `models` wrapper, with labeled section comments. Existing nested configs continue to work through the resolver fallbacks.
- Updated README and `docs/examples.md` transformer examples to the flat configuration layout.

### Fixed
- Fixed transformer `bysentence` aggregation scoring whole documents instead of sentences. `sentiment_bysentence` aggregates rows by `id_text`, but the pipeline passed `df_clean` through unsplit, so each document was a single row. Sentence counts collapsed to 0 or 1 and the resulting score could only ever be `+1`, `-1`, or `0`, silently returning document-level classification under sentence-level column names.
- Fixed `sentence_probability_cutoff` being unreachable from YAML. It was read from the model config but never populated, so the 0.7 cutoff was effectively hard-coded.

## [0.2.0] - 2026-06-02

### Added
- Added optional Hugging Face transformer sentiment support behind the `transformers` extra so the base lexical package remains lightweight.
- Added `SentimentTransformers` with lazy `torch`/`transformers` imports, explicit model label mapping, batched inference, and configurable device selection.
- Added YAML-driven transformer configuration in `params.yaml`, including support for both package-native config keys and original-style `name`/`short_name`/`label_mapping`/`sentiment_values` model lists.
- Added document-level (`byalltext`) and sentence-level (`bysentence`) transformer aggregation, including harmonized positive, neutral, negative, share, and net-sentiment outputs.
- Added transformer parquet exports for model outputs and sentence probabilities.
- Added `notebooks/autoecon_transformers.ipynb` for optional transformer workflows.
- Added transformer-focused tests covering optional imports, config coercion, label-map behavior, sentence aggregation, harmonized outputs, and no-download model test doubles.

### Changed
- Updated the pipeline to run lexical and optional transformer sentiment through the same load, clean, score, and export workflow.
- Updated README and docs to split architecture, data/output, examples, and roadmap content into focused documentation pages.

### Fixed
- Fixed lexical `allwords` scoring so it uses the active text column override instead of falling back to the default token column.

## [0.1.2] - 2026-05-15

### Added
- Created `docs/architecture.md` combining architecture and file structure details.
- Added LaTeX formulas for `posneg` and `allwords` aggregation in the architecture documentation.
- Added input file paths to the configuration overview table in `README.md`.

### Changed
- Improved the Features section of the `README.md` to match industry standards, highlighting scalability and modularity.
- Migrated detailed library components and testing descriptions from the `README.md` to `docs/architecture.md`.
- Updated `notebooks/autoecon_demo.ipynb` to improve the walkthrough presentation.
- Updated `src/auto_econ_sentiment/pipeline.py` and `src/data/cb_speeches_clean.py` to export results as compressed parquet files instead of CSV.
- Updated `notebooks/demo_cb_speechs.ipynb` and `README.md` to reference `.parquet.gzip` output files.

---

## [0.1.1] - 2026-04-26

### Added
- `authors` field in `pyproject.toml` so PyPI displays maintainer info correctly.
- Docstrings and type hints on the public API (`AutoEconSentiment`, `SentimentLexical`, `TextLoader`, `TextCleaner`) so the existing `py.typed` marker is fully usable downstream.

### Changed
- Tightened upper bounds on `matplotlib` (`<4`) and `seaborn` (`<1`) to match the `viz` extra and prevent surprise major-version breaks.
- Removed redundant `[dependency-groups]` block and empty `[tool.hatch.build.targets.wheel.shared-data]` table from `pyproject.toml`.

---

## [0.1.0] - 2026-03-04

### Added
- Initial release of `auto-econ-sentiment`.
- Lexical sentiment analysis pipeline supporting Correa, Hubert, LM, and HIV dictionaries.
- `TextLoader` for CSV/Excel ingestion.
- `TextCleaner` with HTML cleaning, unicode normalization, stemming, and sentence tokenization.
- `SentimentLexical` with `posneg` and `allwords` aggregation methods.
- `AutoEconSentiment` orchestration pipeline with cleanly decoupled data loading, cleaning, and lexical execution into extensible class methods.
- GitHub Actions CI workflow running pytest across Python 3.10, 3.11, and 3.12.
- GitHub Actions PyPI publish workflow using OIDC Trusted Publisher.
- GitHub Actions release workflow to auto-create GitHub Releases on tag push.

### Security
- Excluded raw `.xlsx` and `.csv` dictionary source files from the sdist to avoid publishing proprietary data.
- Secure HTML stripping parser implementation avoiding ReDoS vulnerabilities.

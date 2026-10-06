# Changelog

All notable changes to this project will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added
- Added `src/research_paper/`, which rebuilds every figure, table and statistic in the accompanying paper from raw text. It sits outside the package and is not shipped to PyPI; `uv sync --extra research` installs its dependencies.

### Removed
- Removed the WCB stance model (`gtfintechlab/model_WCB_stance_label`) from the example models in `params.yaml`. It is gated on Hugging Face, was never validated in this repository, and is not used in the paper. It can still be added back to `transformer.models`.

## [1.0.2] - 2026-10-06

### Added
- Added `python-dotenv` so API keys for OpenAI-compatible providers can be read from a `.env` file in the working directory; variables already set in the shell take precedence. `.env.example` lists the keys.

### Removed
- Removed the `API_OLLAMA` environment variable. Set `base_url` to point Ollama at a remote host.

## [1.0.1] - 2026-09-28

The v1.0.0 tag was cut before these two changes reached `main`, so the 1.0.0 package on PyPI does not include them.

### Changed
- Changed the default LLM prompt so polarity follows the package sign convention: positive or hawkish text is +1 and negative or dovish text is -1, matching the FOMC-RoBERTa and WCB stance mappings. The default `prompt_version` is now `v2`; the earlier `v1` prompt scored hawkish as -1.

### Fixed
- Fixed the pipeline failing with a missing-column error whenever `text_column` was not `text`. `TextLoader` renames the configured column to `text`, and the cleaning and lexical stages now read it under that name.

## [1.0.0] - 2026-09-28

### Added
- Added `CITATION.cff`, which GitHub uses for its Cite this repository button, and a contributing guide in `.github/CONTRIBUTING.md`.
- Added directory-of-`.txt` corpus loading in `TextLoader`, supporting flat directories as well as categorized subdirectories via `group_column`.
- Added regex-based filename date parsing (`filename_date_pattern`) to `TextLoader`, retaining unparseable dates as `NaT` with a logged warning instead of silently dropping rows.
- Added Markdown (`.md`, `.markdown`) files to `TextLoader` directory input, read as plain text alongside `.txt` files.
- Added a `recursive` flag for directory traversal and automatic unique `id_column` generation for text corpora.
- Added `ParagraphSegmenter` in `clean/text_segmentation.py` to split documents into paragraph-level rows by blank lines, with a configurable `paragraph_number` column. It composes with `TextSegmenter`, so sentence rows inherit their paragraph number.
- Added `tokenizer_name` attribute on `TextSegmenter` exposing whether `nltk_punkt` or `regex_fallback` is active, so a degraded run is identifiable after the fact.
- Added `require_nltk` parameter on `TextSegmenter` to fail fast when NLTK is unavailable rather than silently degrading.
- Added `drop_invalid` parameter on `TextSegmenter` to filter non-sentential fragments (voting rosters, headers, no-verb items). Defaults to `False`.
- Added abbreviation and single-initial protection to the regex fallback tokenizer, preventing false splits on names such as `St. Louis` and `Susan S. Bies`.
- Added newline-boundary detection to both tokenizer paths, covering `\n` after `.!?` and between a lowercase character or `:` and a capital.
- Added fragment merging (`_merge_incomplete_sentences`) to rejoin lowercase- and punctuation-initial fragments into the preceding sentence.
- Added `sentence_probability_aggregation` parameter (`cutoff` | `mean`) to transformer `bysentence` aggregation, threaded through `sentiment_pipeline` and pipeline config resolution.
- Added a mean-probability aggregation mode that averages raw sentence probabilities per document instead of thresholding, preserving confidence magnitude and producing a continuous document score.
- Added LLM single-shot sentiment scoring via `SentimentLLM`, with Ollama and OpenAI-compatible providers (OpenAI, OpenRouter, Together, Groq, vLLM) behind a new optional `llm` extra (`httpx` only, no provider SDK).
- Added `polarity` x `confidence` scoring, deliberately mirroring the transformer's existing `direction * probability` shape so LLM output is comparable in the same harmonized columns. Selectable continuous (`-1..1`) or discrete (`{0,1,2}`) presentation via `output_scale`.
- Added `analyze_sentiment_llm` pipeline method with `resolve_llm_config`, `_expand_llm_model_configs`, and `_coerce_llm_model_config` helpers mirroring the transformer config pattern.
- Added `docs/llm_scoring.md` covering installation, configuration, output columns, provider setup, prompting, and caveats.
- Added a `@pytest.mark.llm` marker for optional LLM integration tests, skipped by default.
- Added tests for `TextLoader` directory loading, segmentation edge cases, paragraph splitting, mean-mode aggregation arithmetic, and `SentimentLLM` (prompt formatting, JSON parsing, continuous/discrete scoring, prose fallback, batch resilience, out-of-range rejection, confidence cutoff, column naming, request building, and pipeline integration).
- Added a `{dictionary}_sentiment_{method}_net` column to lexical output that recenters each score on $[-1, 1]$, the scale transformer scores use. The existing `{dictionary}_sentiment_{method}` column keeps the $[0, 2]$ Apel-Blix Grimaldi convention.
  Stemmed dictionaries name it `{dictionary}_sentiment_{method}_stem_net`, so `_net` is always the last suffix.
- Added a score-scale reference table to `docs/data.md`.
- Added `{model}_sentiment_posneg_net` and `{model}_sentiment_allsentences_net` to cutoff-mode sentence-level transformer output, dividing the net sentence count by the sentences that clear the cutoff and by every segmented sentence, mirroring the lexical `posneg` and `allwords` pair. `{model}_count_sentences` records the segmented sentence count.
- Added `sentence_number` and `sentence_text` columns to the sentence-level transformer probability export, so each row can be read against the sentence it scored.

### Changed
- Changed the package classifier from `3 - Alpha` to `5 - Production/Stable` for the first stable release.
- Renamed `docs/ROADMAP.md` to `docs/roadmap.md` and reviewed every docs page for v1.0.0. The README example for FOMC-RoBERTa now uses the corrected label mapping, the architecture page covers the LLM scorer and paragraph segmenter, and the paragraph example runs on the original text, since the cleaner collapses newlines.
- Changed `TextSegmenter.split_text` to apply newline pre-splitting, abbreviation protection, and fragment merging in a unified pipeline before both the NLTK and regex tokenizers.
- Expanded `_FALLBACK_BOUNDARY` to include opening quotes and brackets in its lookahead character class.
- Changed `src/auto_econ_sentiment/__init__.py` and `models/__init__.py` to export `SentimentLLM`.
- Updated `docs/data.md` with directory-of-txt loading, constructor parameters, mean-aggregation output columns, and LLM output columns.

### Fixed
- Fixed the default `params.yaml` label mappings for the two stance models. FOMC-RoBERTa had `LABEL_0` (dovish) and `LABEL_1` (hawkish) swapped, so its scores carried the wrong sign. The WCB stance model mapped `LABEL_0` (neutral) and `LABEL_2` (dovish) the wrong way round and then inverted the sign. Both now map hawkish to +1 and dovish to -1.
- Fixed `TextSegmenter` silently degrading to a substantially weaker regex tokenizer when NLTK or its `punkt` data was unavailable, logging only a warning. Measured over 230 FOMC statements, the fallback produced 4,245 sentences of which 37.9% were non-sentential fragments, against 3,302 sentences and 21.5% for `punkt` -- so the same corpus could differ by 16 percentage points of garbage with no visible error. The two paths now converge (3,631 vs 3,633 sentences, 28.6% both), and `drop_invalid=True` brings the garbage share to 7.2%.
- Fixed name rosters being shredded into fragments by the regex fallback: sentences ending on a single initial went from 1,023 (24.1%) to 0, false abbreviation splits from 113 to 0, and missed newline boundaries from 420 to 0.

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

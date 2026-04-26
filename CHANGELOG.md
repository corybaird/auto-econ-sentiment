# Changelog

All notable changes to this project will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

# Changelog

All notable changes to this project will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.1.0] - 2026-03-04

### Added
- Initial release of `auto-econ-sentiment`.
- Lexical sentiment analysis pipeline supporting Correa, Hubert, LM, and HIV dictionaries.
- `TextLoader` for CSV/Excel ingestion.
- `TextCleaner` with HTML cleaning, unicode normalization, stemming, and sentence tokenization.
- `SentimentLexical` with `pos` and `alt` aggregation methods.
- `AutoEconSentiment` pipeline orchestrator with YAML-based configuration.
- GitHub Actions CI workflow running pytest across Python 3.10, 3.11, and 3.12.
- GitHub Actions PyPI publish workflow using OIDC Trusted Publisher.
- GitHub Actions release workflow to auto-create GitHub Releases on tag push.

### Security
- Excluded raw `.xlsx` and `.csv` dictionary source files from the sdist to avoid publishing proprietary data.
- Secure HTML stripping parser implementation avoiding ReDoS vulnerabilities.

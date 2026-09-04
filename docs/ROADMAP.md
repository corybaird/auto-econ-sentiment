# Product Roadmap & Release Management

This document outlines the planned future features for `auto-econ-sentiment` and the standardized procedures for versioning, releasing, and publishing via GitHub and PyPI.

Research-planning notes, paper feedback, and transformer refactor scratch docs can live locally under `docs/feedback/`. That directory is ignored by git so exploratory notes can evolve without becoming release documentation.

## Current Release: `v0.3.0` (Sentence Segmentation and Config Overhaul)
The `v0.3.0` release introduces robust sentence-level segmentation for transformer models and modernizes configuration handling.

### Implemented Scope (`v0.3.0`)
1. `TextSegmenter` in `src/auto_econ_sentiment/clean/text_segmentation.py` using NLTK `sent_tokenize` with a regex sentence-boundary fallback.
2. Direct document-to-sentence explosion during transformer sentence-level scoring (`bysentence`).
3. Top-level `lexical` and `transformer` configuration keys in `params.yaml` with backward-compatible resolvers for legacy nested keys.
4. Per-model overrides for `min_sentence_chars` and `sentence_probability_cutoff`.
5. Expanded multi-module test coverage across six test suites in `tests/`.

### Upcoming Roadmap (In Active Development on `dev`)
Key features currently merged into the `dev` integration branch for the next major pipeline release:
1. **Directory-of-Text Ingestion (`PR #12`):** Direct support in `TextLoader` for loading entire directories of raw `.txt` files, extracting document identifiers and dates automatically from filenames.
2. **Multilevel Text Segmentation (`PR #13`):** Enhanced chunking supporting both paragraph-level blocks and robust sentence boundary detection for complex financial abbreviations.
3. **Continuous Probability Aggregation (`PR #14`):** Support for `sentence_probability_aggregation: mean` to calculate document sentiment via continuous probability averaging across sentences alongside existing count and share metrics.
4. **LLM Single-Shot Sentiment Scoring (`PR #15`):** Provider-neutral LLM scoring interface for prompt-based economic sentiment analysis with structured schema enforcement.

---

## Release & Tagging Process

This repository adheres strictly to **Semantic Versioning** (`vMAJOR.MINOR.PATCH`). 

### How GitHub Releases and PyPI Publishing Are Managed
Releases are automated via GitHub Actions (`.github/workflows/release.yml` and `.github/workflows/publish.yml`) triggered by annotated Git tags.

When deploying a release, follow these steps after merging the relevant PR into `main`:

#### Step 1: Verify Version in Codebase
1. Verify `pyproject.toml` contains the target version (e.g. `version = "0.3.0"`).
2. Ensure all changes are documented in `CHANGELOG.md` under the version header.
3. Commit version updates following the commit convention:
   `UPDATE project versions for v0.3.0 release`
4. Merge the final update into `main`.

#### Step 2: Cut the Release via Git Tags
Create an annotated Git tag locally and push it to GitHub:

```bash
# 1. Fetch latest main and ensure local branch is synchronized
git checkout main
git pull origin main

# 2. Create an annotated tag (-a creates annotation, -m sets message)
git tag -a v0.3.0 -m "Release v0.3.0 - Transformer sentence segmentation overhaul and docs"

# 3. Push the tag to GitHub
git push origin v0.3.0
```

#### Step 3: Automation Workflow Execution
Once the tag is pushed to GitHub:

1. **GitHub Release:** `.github/workflows/release.yml` detects the tag and drafts the corresponding release.
2. **PyPI Publishing:** `.github/workflows/publish.yml` detects the release, builds package artifacts with Hatch, and publishes them securely to PyPI using OIDC trusted publishing.

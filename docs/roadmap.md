# Product Roadmap & Release Management

This document outlines the planned future features for `auto-econ-sentiment` and the standardized procedures for versioning, releasing, and publishing via GitHub and PyPI.

Research-planning notes, paper feedback, and transformer refactor scratch docs can live locally under `docs/feedback/`. That directory is ignored by git so exploratory notes can evolve without becoming release documentation.

## Current Release: `v1.0.0` (Stable Release)
`v1.0.0` is the first stable release. `v1.0.1` adds two fixes that missed the v1.0.0 tag, the custom text-column fix and the hawkish = +1 LLM prompt, and is the version the accompanying paper's results are produced with. Together they add directory input, paragraph segmentation, mean aggregation and LLM scoring, and put every scoring method on a comparable net-sentiment scale.

### Implemented Scope (`v1.0.0`)
1. **Directory-of-Text Ingestion (`PR #12`):** `TextLoader` reads directories of `.txt` and Markdown files, one document per file, with dates parsed from filenames.
2. **Multilevel Text Segmentation (`PR #13`):** `ParagraphSegmenter`, plus sentence splitting that is robust to financial abbreviations and consistent whether or not NLTK is available.
3. **Continuous Probability Aggregation (`PR #14`):** `sentence_probability_aggregation: mean` for a continuous document score alongside the count and share metrics.
4. **LLM Single-Shot Sentiment Scoring (`PR #15`):** A provider-neutral LLM scoring interface behind the optional `llm` extra.
5. **Net-Sentiment Columns (`PR #17`, `PR #18`):** Every score ending in `_net` is on $[-1, 1]$ with zero as neutral. Lexical scores gain `{dictionary}_sentiment_{method}_net`, and sentence-level transformer scores gain `{model}_sentiment_posneg_net` and `{model}_sentiment_allsentences_net`.
6. **Sentence Audit Output (`PR #18`):** The sentence-level export carries each sentence's number and text.
7. **Stance Model Labels (`PR #19`):** Corrected FOMC-RoBERTa and WCB stance label mappings in `params.yaml`.
8. **Markdown Input (`PR #20`):** Directory input reads `.md` and `.markdown` files alongside `.txt`.

### Planned
1. **Lemmatization:** A lemmatizer in `TextCleaner` producing a `text_lemmas` column that dictionaries can match against, alongside `text_tokens_str` and `text_stems`.
2. **Paragraph-Level Scoring:** `ParagraphSegmenter` splits documents into paragraphs, but the pipeline segments only by sentence. A segmentation setting in the transformer and LLM config would let models score paragraphs as the unit between the sentence and the whole document. `TextCleaner` currently collapses newlines, so this also needs a cleaning option that keeps paragraph breaks in `text_clean`.
3. **Sentence-Level Lexical Matches:** Dictionary matches are reported per document. Reporting them per sentence, aligned with the transformer sentence export, would let the matched words be read against each sentence's class probabilities without a separate script.
4. **LLM Net-Sentiment Parity:** Sentence-level LLM scoring reports `{llm}_net_sentiment` over classified sentences. Adding `{llm}_sentiment_posneg_net` and `{llm}_sentiment_allsentences_net` would put LLMs on the same denominators as the lexical and transformer scores.

---

## Release & Tagging Process

This repository adheres strictly to **Semantic Versioning** (`vMAJOR.MINOR.PATCH`). 

### How GitHub Releases and PyPI Publishing Are Managed
Releases are automated via GitHub Actions (`.github/workflows/release.yml` and `.github/workflows/publish.yml`) triggered by annotated Git tags.

When deploying a release, follow these steps after merging the relevant PR into `main`:

#### Step 1: Verify Version in Codebase
1. Verify `pyproject.toml` contains the target version (e.g. `version = "1.0.0"`).
2. Ensure all changes are documented in `CHANGELOG.md` under the version header.
3. Commit version updates following the commit convention:
   `UPDATE project versions for v1.0.0 release`
4. Merge the final update into `main`.

#### Step 2: Cut the Release via Git Tags
Tag only once every pull request meant for the release is merged into `main`. Pushing the tag publishes to PyPI straight away, and PyPI never lets a published version be replaced, so a missed change needs a new patch version (as happened with `v1.0.1`).

Create an annotated Git tag locally and push it to GitHub:

```bash
# 1. Fetch latest main and ensure local branch is synchronized
git checkout main
git pull origin main

# 2. Create an annotated tag (-a creates annotation, -m sets message)
git tag -a v1.0.0 -m "Release v1.0.0 - Stable release"

# 3. Push the tag to GitHub
git push origin v1.0.0
```

#### Step 3: Automation Workflow Execution
Once the tag is pushed to GitHub:

1. **GitHub Release:** `.github/workflows/release.yml` detects the tag and drafts the corresponding release.
2. **PyPI Publishing:** `.github/workflows/publish.yml` detects the release, builds package artifacts with Hatch, and publishes them securely to PyPI using OIDC trusted publishing.

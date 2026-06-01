# Product Roadmap & Release Management

This document outlines the planned future features for `auto-econ-sentiment` and the standardized procedures for versioning, releasing, and publishing via GitHub and PyPI.

Research-planning notes, paper feedback, and transformer refactor scratch docs can live locally under `docs/feedback/`. That directory is ignored by git so exploratory notes can evolve without becoming release documentation.

## Next Release Candidate: `v0.2.0` (Optional Transformer Extension)
The current transformer feature line expands the lexical pipeline to include optional Hugging Face transformer models. The release should upgrade model coverage while maintaining the existing YAML-driven architecture and keeping transformer dependencies optional.

### Implemented Scope
The transformer extension currently includes:

1. `transformers` and `torch` as optional dependencies under the `transformers` extra.
2. `SentimentTransformers` in `src/auto_econ_sentiment/models/sentiment_transformers.py`.
3. Pipeline integration through `AutoEconSentiment.analyze_sentiment_transformer()`.
4. YAML model configuration in `params.yaml`, with support for original-style `name`, `short_name`, `label_mapping`, and `sentiment_values` keys.
5. Document-level and sentence-level aggregation modes.
6. Harmonized positive, neutral, negative, share, and net-sentiment columns for comparable transformer outputs.
7. Parquet exports for transformer sentiment tables and sentence probabilities.

### Branching Note
The optional transformer work should live on `feat/optional-transformer-extension`. Documentation-only branches stacked on top of it should target that branch as their PR base so transformer notebook and model changes do not appear as unrelated docs changes.

---

## Release & Tagging Process

This repository adheres strictly to **Semantic Versioning** (`vMAJOR.MINOR.PATCH`). 

### How GitHub Releases and PyPI Publishing Are Managed
Currently, in `v0.1.0`, your repository is configured with a GitHub Actions workflow to auto-publish releases. This means the process is partially automated, but triggered by a manual Git action.

When deploying a new version, follow these exact steps after reviewing and merging the relevant PRs into `main`:

#### Step 1: Prepare the Codebase for Release
1. Update `pyproject.toml`: Change `version = "0.2.0"`.
2. Update `CHANGELOG.md`: 
   - Change `[Unreleased]` to `[0.2.0] - YYYY-MM-DD`.
   - Ensure all feature notes from the `add/transformer` branches are documented under `### Added`.
3. Commit the bumped files using the commit-agent pattern:
   `UPDATE project versions for v0.2.0 release`
4. Merge this final update into the `main` branch.

#### Step 2: Cut the Release via Git Tags (Terminal/CLI)
Because you have a GitHub Actions Release workflow, the cleanest way to handle this is by creating an **annotated Git Tag** locally and pushing it to GitHub.

```bash
# 1. Fetch latest main and ensure you are on it
git checkout main
git pull

# 2. Create an annotated tag (the -a flag is critical, -m adds the message)
git tag -a v0.2.0 -m "Release v0.2.0 - Transformer Model Integration"

# 3. Push the tag to GitHub
git push origin v0.2.0
```

#### Step 3: GitHub Automation Takes Over
Once you run `git push origin v0.2.0`:

1. **GitHub Release:** Your GitHub action workflow (often located at `.github/workflows/release.yml`) spots the new tag. It will automatically draft and publish a GitHub Release on the repository page.
2. **PyPI Publishing:** The PyPI publish workflow (using your OIDC Trusted Publisher configuration) will recognize the release/tag, build your package artifacts (`sdist` and `wheel` through `hatch`), and push them to PyPI seamlessly. 

*(Note: Alternatively, if you did not want to push tags via the terminal, you can go to the **GitHub UI > Releases > Draft a New Release**, set the title to `v0.2.0`, and click Publish. The GitHub workflows will then handle the PyPI publishing exactly the same way.)*

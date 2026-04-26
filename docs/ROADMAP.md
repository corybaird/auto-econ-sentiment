# Product Roadmap & Release Management

This document outlines the planned future features for `auto-econ-sentiment` and the standardized procedures for versioning, releasing, and publishing via GitHub and PyPI.

## Future Plans: `v0.2.0` (Transformer Integration)
The next major objective is expanding the purely lexical pipeline to include Hugging Face Transformer models. This release will significantly upgrade capabilities while maintaining the existing YAML-driven architecture.

### Development Plan
The implementation should follow these atomic branches (aligning with `workflows/branch-agent.md`):

1. **`add/transformer-dependencies`**
   - Goal: Safely introduce `transformers`, `torch`, etc., to `pyproject.toml` (likely under an `[project.optional-dependencies]` group so users aren't forced to download heavy torch binaries unless needed).
2. **`add/transformer-models-class`**
   - Goal: Introduce `SentimentTransformersPipeline` into a new `src/models/sentiment_transformers.py` file.
3. **`update/pipeline-transformer-integration`**
   - Goal: Update `src/auto_econ_sentiment/pipeline.py` with `analyze_sentiment_transformer_sentence` and `analyze_sentiment_transformer_fulltext` methods.
4. **`update/transformer-configs`**
   - Goal: Update `params.yaml` structure to support transformer model specifications, configurations, huggingface tags, and cloud inference flags.

---

## Release & Tagging Process

This repository adheres strictly to **Semantic Versioning** (`vMAJOR.MINOR.PATCH`). 

### How GitHub Releases and PyPI Publishing Are Managed
Currently, in `v0.1.0`, your repository is configured with a GitHub Actions workflow to auto-publish releases. This means the process is partially automated, but triggered by a manual Git action.

When deploying a new version (e.g., the upcoming `v0.2.0`), follow these exact steps:

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

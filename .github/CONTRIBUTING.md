# Contributing

Thanks for your interest in `auto-econ-sentiment`. Bug reports, dictionary suggestions and pull requests are welcome.

## Reporting Issues

Open an issue with the package version (`auto_econ_sentiment.__version__`), the relevant part of your `params.yaml`, and the full error or the output you did not expect. For scoring questions, a short text that reproduces the result helps most.

## Development Setup

```bash
git clone https://github.com/corybaird/auto-econ-sentiment.git
cd auto-econ-sentiment
uv sync --all-extras
uv run pytest
```

Tests that call a live LLM provider are marked `llm` and skipped by default. CI runs the suite on Python 3.10, 3.11 and 3.12.

## Branches

- Branch from `main` and open pull requests against it. `main` should always pass the tests.
- Name branches by type: `feat/...` for features, `fix/...` for bug fixes, `docs/...` for documentation.
- A release is one pull request, `release/vX.Y.Z`, that bumps the version and dates the changelog. Publishing happens only when the tag is pushed; see [docs/roadmap.md](../docs/roadmap.md) for the steps.

## Commits

Keep each commit to one change, with a subject that starts with an uppercase verb:

```text
ADD Markdown files to TextLoader directory input
FIX pipeline failing when the text column is not named text
UPDATE changelog for the LLM prompt direction change
DOCS describe the lexical _net column and its tests
REFACTOR config resolution into reusable helpers
```

Use the body to explain why the change was needed.

## Pull Requests

- Add tests for new behaviour and for the bug a fix addresses.
- Add a line under `## [Unreleased]` in `CHANGELOG.md` for anything a user would notice. Version numbers change only in release pull requests.
- Update the pages in `docs/` that describe what you changed.
- Describe the change and how you tested it in the pull request.

## Conventions Worth Knowing

- **Sign convention:** positive tone and hawkish stance are `+1`. Transformer label mappings in `params.yaml` and the default LLM prompt follow it.
- **Score scales:** every column ending in `_net` lies in $[-1, 1]$ with zero as neutral. New scoring methods should provide one. See [docs/data.md](../docs/data.md#score-scales).
- **Optional dependencies:** transformer and LLM support sit behind the `transformers` and `llm` extras and are imported lazily, so the lexical package must keep working without them.

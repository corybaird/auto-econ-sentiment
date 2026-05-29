# Transformer Refactor Roadmap

This document describes the refactor needed to add transformer sentiment models to `auto-econ-sentiment` while preserving the current lexical pipeline as the stable v1 baseline.

The original transformer implementation lives at:

`/Users/cory/Desktop/github/Econ_Text_Algos/src/models/sentiment_transformers.py`

The target package location is:

`src/auto_econ_sentiment/models/sentiment_transformers.py`

## Goal

Add transformer-based sentiment analysis as an optional backend that can be run through the same package conventions as lexical methods:

1. configuration-driven execution,
2. reproducible exports,
3. sentence-level and document-level outputs,
4. clear model metadata,
5. optional heavy dependencies,
6. tests that do not require large model downloads in normal CI.

The transformer path should not replace the lexical paper baseline. It should extend it in a later software and paper version so the project can demonstrate how empirical text research can be updated instead of treated as static.

## Refactor Principles

1. Keep lexical behavior unchanged.
2. Keep transformer dependencies optional.
3. Avoid model downloads during import.
4. Make output columns predictable.
5. Store model configuration and model identity in exported metadata.
6. Separate scoring mechanics from pipeline orchestration.
7. Make CPU execution possible for small examples, while allowing GPU acceleration when available.

## Current Source Assessment

The old transformer class already provides useful pieces:

- Hugging Face tokenizer/model loading.
- Batched inference with `torch.utils.data.DataLoader`.
- GPU detection.
- Probability outputs.
- Label-to-direction mapping.
- Sentence aggregation by `id_text`.
- Model save/load helpers.

It needs package-level cleanup before it can land here:

- Replace old imports such as `from src.models.sentiment_base import SentimentBase` with `from auto_econ_sentiment.models.sentiment_base import SentimentBase`.
- Rename the class to match package conventions, likely `SentimentTransformers`.
- Remove demo-only code from the module body.
- Move examples into notebooks, tests, or docs.
- Make label mapping configurable rather than hard-coded for a specific model.
- Avoid requiring `torch` and `transformers` unless the transformer feature is requested.

## Proposed Package Structure

```text
src/auto_econ_sentiment/
├── models/
│   ├── sentiment_base.py
│   ├── sentiment_lexical.py
│   └── sentiment_transformers.py
├── pipeline.py
└── data/
```

Optional future helpers:

```text
src/auto_econ_sentiment/models/transformer_utils.py
src/auto_econ_sentiment/models/label_maps.py
```

Start with one file unless complexity genuinely requires helpers.

## Dependency Plan

Add transformer dependencies as an optional extra in `pyproject.toml`:

```toml
[project.optional-dependencies]
transformers = [
    "torch",
    "transformers",
]
```

Potential additional dependencies, only if needed:

- `accelerate` for device placement or larger models.
- `sentencepiece` for tokenizers that require it.

Avoid making these required dependencies for lexical users.

Expected install pattern:

```bash
uv sync --extra transformers
```

or, after publication:

```bash
pip install "auto-econ-sentiment[transformers]"
```

## Config Design

Add transformer configuration under `models.transformer` without changing the existing lexical config.

```yaml
models:
  lexical:
    dictionaries:
      unstemmed: [correa, hubert, lm, hiv]
      stemmed: [bn, ap]
    aggregation_methods: [posneg, allwords]

  transformer:
    enabled: false
    text_column_transformer: text_clean
    models:
      - model_name: Moritz-Pfeifer/CentralBankRoBERTa-sentiment-classifier
        model_name_short: cbroberta
        num_labels: 2
        max_length: 512
        batch_size: 8
        aggregation: byalltext
        label_map:
          positive: 1
          negative: -1
      - model_name: gtfintechlab/FOMC-RoBERTa
        model_name_short: fomc_roberta
        num_labels: 3
        max_length: 512
        batch_size: 8
        aggregation: bysentence
        sentence_probability_cutoff: 0.7
        label_map:
          LABEL_0: 1
          LABEL_1: -1
          LABEL_2: 0
```

Important design choices:

- `enabled` keeps transformer execution opt-in.
- `models` is a list so users can compare several transformer models.
- `model_name_short` controls output column prefixes.
- `label_map` keeps model-specific sentiment direction explicit.
- `aggregation` can be `byalltext` or `bysentence`.

## API Design

Add a transformer model class with a narrow public surface:

```python
pipe = SentimentTransformers(
    df_input=df,
    text_column="text_clean",
    model_name="Moritz-Pfeifer/CentralBankRoBERTa-sentiment-classifier",
    model_name_short="cbroberta",
    label_map={"positive": 1, "negative": -1},
)

df_scores = pipe.sentiment_pipeline(aggregation="byalltext")
```

The model class should expose:

- `analyze_sentiment()`
- `sentiment_pipeline()`
- `clear_gpu_memory()`

Optional later methods:

- `save_model()`
- `load_model()`

Do not expose too many inference internals until the core behavior is stable.

## Pipeline Integration

Add transformer methods to `AutoEconSentiment`:

```python
analyze_sentiment_transformer(...)
```

The main `run()` method should preserve the lexical path and conditionally run transformers only when requested:

```python
if transformer_config.get("enabled", False):
    self.analyze_sentiment_transformer(...)
```

Pipeline state should remain explicit:

- `self.df_sent_lexical`
- `self.df_sent_transformer`
- `self.df_sentiment_all`

Export files:

```text
sentiment_lexical.parquet.gzip
sentiment_transformer.parquet.gzip
sentiment_all_results.parquet.gzip
```

If sentence probabilities are generated, export them separately:

```text
sentiment_transformer_sentence_probabilities.parquet.gzip
```

## Output Column Conventions

Transformer document-level outputs:

```text
{model_short}_predicted_label
{model_short}_label
{model_short}_probability_{label_id}
{model_short}_label_sentiment
{model_short}_sentiment_byalltext
```

Transformer sentence-level outputs:

```text
{model_short}_countsentence_positive
{model_short}_countsentence_negative
{model_short}_sentiment_bysentence
```

Metadata columns to consider:

```text
{model_short}_model_name
{model_short}_aggregation
{model_short}_max_length
```

Avoid ambiguous columns such as plain `{model_short}_sentiment` when both sentence and document aggregation are available.

## Label Mapping Refactor

The old implementation includes hard-coded mapping for specific labels:

```python
sentiment_map = {
    "positive": 1,
    "negative": -1,
    "LABEL_2": 0,
    "LABEL_1": -1,
    "LABEL_0": 1,
}
```

This should become a required or validated model-specific config. The package cannot assume that every Hugging Face model orders labels in the same way.

Validation rules:

- Every non-neutral model label must appear in `label_map`.
- Neutral labels can map to `0`.
- Raise a clear error if a predicted label is missing from the map.
- Include the label map in logs or exported metadata.

## Sentence-Level Refactor

The old class expects repeated `id_text` values for sentence aggregation. The package currently stores one row per document. Sentence-level transformer support therefore needs a sentence-splitting step or an explicit sentence-input mode.

Two possible approaches:

1. **Input already has one row per sentence**
   - Require `id_text` to group sentences back to documents.
   - Simplest first implementation.

2. **Package splits documents into sentences**
   - Add a sentence splitter.
   - Preserve original document IDs.
   - Export sentence-level probabilities.

Recommended first version:

- Support explicit sentence-row input first.
- Document the expected schema.
- Add package-owned sentence splitting later.

## Testing Strategy

Avoid downloading large Hugging Face models in default tests.

Test layers:

1. Unit-test config parsing and validation.
2. Unit-test label mapping with synthetic probability arrays.
3. Unit-test output column naming.
4. Mock tokenizer/model inference for fast CI.
5. Add one optional integration test behind a marker:

```bash
uv run pytest -m transformers
```

Potential test markers:

```python
@pytest.mark.transformers
@pytest.mark.slow
```

CI default should skip network/model-download tests.

## Documentation Work

Add:

- `docs/transformers.md`
- README optional install snippet.
- YAML config example.
- Expected output schema.
- Short warning about label-map responsibility.
- Example comparing lexical and transformer outputs.

The docs should make clear that transformer outputs are model-dependent and not automatically more correct than lexical outputs.

## Research Paper Integration

For the arXiv project, transformer support should be versioned as a follow-up:

- v1 paper: lexical methods only.
- v2 paper/update: lexical vs transformer comparison.

Suggested v2 figures:

1. Lexical-transformer correlation heatmap.
2. Time-series comparison for selected central banks.
3. Agreement/disagreement table for recession periods or policy episodes.
4. Downstream task comparison using lexical and transformer scores.

Core research question:

When does a context-aware transformer measure agree with transparent lexical dictionaries, and when does it materially revise the empirical interpretation?

## Branch Sequence

Recommended implementation branches:

1. `add/transformer-optional-dependencies`
2. `add/transformer-model-class`
3. `add/transformer-config-validation`
4. `update/pipeline-transformer-integration`
5. `add/transformer-tests`
6. `docs/transformer-usage`
7. `paper/transformer-comparison-v2`

Keep each branch small enough to review independently.

## Acceptance Criteria

Transformer support is ready when:

1. Lexical tests still pass unchanged.
2. Package import works without `torch` or `transformers` installed.
3. Transformer extra install works.
4. A tiny mocked transformer test produces expected columns.
5. A documented Hugging Face example runs locally when the extra is installed.
6. Outputs can be merged with lexical scores by `id_text`.
7. Label mappings are explicit and validated.
8. The README and docs explain optional dependencies and model-specific caveats.

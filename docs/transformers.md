# Transformer Sentiment

Transformer sentiment is optional. The core package remains lexical-first and does not require `torch` or `transformers` unless this feature is explicitly installed and enabled.

## Install

For local development with `uv`:

```bash
uv sync --extra transformers
```

After package publication:

```bash
pip install "auto-econ-sentiment[transformers]"
```

The regular install still supports lexical sentiment without transformer dependencies.

## Configuration

Transformer models are configured under `models.transformer` and are disabled by default.

```yaml
models:
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
```

Set `enabled: true` only after installing the optional dependencies.

## Label Maps

Every transformer model can use different label names and label ordering. The package therefore requires an explicit `label_map`.

Examples:

```yaml
label_map:
  positive: 1
  neutral: 0
  negative: -1
```

For models that use generic labels:

```yaml
label_map:
  LABEL_0: 1
  LABEL_1: -1
  LABEL_2: 0
```

Check the model card before choosing a label map. A transformer score is only meaningful if labels are mapped to sentiment directions correctly.

## Aggregation Modes

`byalltext` scores each row as one text:

```yaml
aggregation: byalltext
```

Outputs include:

```text
{model_short}_predicted_label
{model_short}_label
{model_short}_probability_{label_id}
{model_short}_label_sentiment
{model_short}_sentiment_byalltext
```

`bysentence` expects sentence-level rows with an `id_text` column that groups sentences back to documents:

```yaml
aggregation: bysentence
sentence_probability_cutoff: 0.7
```

Outputs include:

```text
{model_short}_countsentence_{label}
{model_short}_sentiment_bysentence
```

Sentence-level probabilities are exported separately when `export_results` is enabled.

## Caveats

- Transformer models may require large downloads.
- Some models require additional tokenizer packages.
- GPU acceleration is used when available, but CPU execution should work for small examples.
- Transformer outputs are model-dependent and should be compared with lexical baselines rather than treated as automatically superior.

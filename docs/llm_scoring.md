# LLM Sentiment Scoring

`auto-econ-sentiment` provides a provider-neutral interface for single-shot Large Language Model (LLM) sentiment scoring via `SentimentLLM`. It supports local Ollama endpoints as well as any OpenAI-compatible `/v1/chat/completions` API (OpenAI, OpenRouter, Together, Groq, vLLM).

## Installation

LLM sentiment scoring requires optional dependencies:

```bash
uv sync --extra llm
# or
pip install 'auto-econ-sentiment[llm]'
```

Both backends use plain HTTP communication (`httpx` or standard library HTTP), eliminating vendor SDK lock-in and keeping the dependency footprint minimal.

## Configuration

Enable LLM scoring in `params.yaml` by configuring the `llm` block:

```yaml
# ------------------------------------------------------------------------------
# LLM SENTIMENT
# Optional single-shot LLM scoring via Ollama or OpenAI-compatible APIs.
# Set `enabled: true` to activate. Settings here apply to every model in the
# `models` list below, and any model may override them individually.
# ------------------------------------------------------------------------------
llm:
  enabled: true
  text_column_llm: text_clean
  aggregation_methods: [byalltext]  # or [bysentence]
  provider: ollama                  # 'ollama' or 'openai'
  output_scale: continuous          # 'continuous' (-1..1) or 'discrete' (0, 1, 2)
  temperature: 0
  confidence_cutoff: 0.7
  models:
    - name: llama3:8b
      short_name: llama3
      provider: ollama
      temperature: 0
      output_scale: continuous
```

### Configuration Resolution

`AutoEconSentiment.resolve_llm_config(config)` resolves the active configuration. It supports the flat top-level `llm:` block with fallback to legacy `models.llm` nested dictionaries and `models.llms` list formats.

## Output Design: Polarity x Confidence

Rather than requesting a single opaque score from the model, `SentimentLLM` prompts the model for two structured fields:

```text
score = polarity * confidence        (e.g., -1 * 0.9 = -0.9)
```

- **`polarity`**: Direction in `{-1, 0, 1}` (where `-1` is hawkish/negative, `0` is neutral, and `1` is dovish/positive).
- **`confidence`**: Certainty float in `[0.0, 1.0]`.

This mirrors the probability $\times$ direction calculation used in `SentimentTransformers`, allowing LLM outputs to produce harmonized columns (`{short}_count_*`, `{short}_share_*`, `{short}_net_sentiment`) directly comparable with transformer and lexical scores.

### Output Scales

The `output_scale` parameter controls presentation:

- **`continuous`** (default): The signed product $\text{polarity} \times \text{confidence} \in [-1, 1]$.
- **`discrete`**: Mapped from polarity alone into $\{0, 1, 2\}$ (0 for negative, 1 for neutral, 2 for positive), matching the `LABEL_0/1/2` conventions in transformer models.

### Output Columns

Scoring adds the following columns to output tables:

- **Document scoring (`byalltext`)**:
  - `{short}_polarity`: Raw polarity ($-1, 0, 1$).
  - `{short}_confidence`: Raw certainty ($0.0 \dots 1.0$).
  - `{short}_sentiment_byalltext`: Continuous score or discrete category.
- **Sentence scoring (`bysentence`)**:
  - `{short}_count_positive`, `{short}_count_neutral`, `{short}_count_negative`: Confident sentence counts.
  - `{short}_share_positive`, `{short}_share_neutral`, `{short}_share_negative`: Sentence shares.
  - `{short}_net_sentiment`: Net sentiment (`share_positive - share_negative`).
  - `{short}_sentiment_bysentence`: Aggregated sentence sentiment score.
- **Metadata columns**:
  - `{short}_provider`: Provider name (e.g. `ollama`, `openai`).
  - `{short}_model`: Model identifier.
  - `{short}_prompt_version`: Prompt version identifier.
  - `{short}_temperature`: Generation temperature.

## Providers

### 1. Ollama (Default)

Used for local inference without API fees or external network calls:

- **Default Endpoint**: `http://localhost:11434/api/generate`
- **Host Override**: Read from `.env`'s `API_OLLAMA` variable or specified via `base_url`.
- **Payload**: JSON request specifying `model`, `prompt`, `stream: false`, and `temperature`.

```yaml
llm:
  enabled: true
  provider: ollama
  models:
    - name: llama3:8b
      short_name: llama3
```

### 2. OpenAI-Compatible APIs (OpenAI, OpenRouter, Together, Groq, vLLM)

Any endpoint following the OpenAI `/v1/chat/completions` specification is supported by configuring `provider: openai`, `base_url`, and `api_key_env`.

#### OpenRouter Example

```yaml
llm:
  enabled: true
  provider: openai
  base_url: https://openrouter.ai/api/v1
  api_key_env: OPENROUTER_API_KEY
  models:
    - name: anthropic/claude-3.5-sonnet
      short_name: claude35
      temperature: 0
```

#### vLLM / Local OpenAI-Compatible Server Example

```yaml
llm:
  enabled: true
  provider: openai
  base_url: http://localhost:8000/v1
  models:
    - name: meta-llama/Meta-Llama-3-8B-Instruct
      short_name: llama3
      temperature: 0
```

## Prompting and Parsing

`SentimentLLM` requests strict JSON:

```json
{
  "polarity": -1,
  "confidence": 0.9
}
```

- **JSON Fallback**: If the model encloses the response in markdown code fences (````json ... ````) or introductory prose, regex extractors recover the JSON payload.
- **Validation**: `polarity` must be an integer in `{-1, 0, 1}`, and `confidence` must be a float in `[0, 1]`.

## Caveats and Fault Tolerance

- **Batch Resilience**: Unparseable or out-of-range outputs log a warning and record `NaN` for that row without raising or terminating the batch run.
- **Confidence Cutoff**: The `confidence_cutoff` setting filters out low-certainty predictions, setting document scores below the threshold to `NaN` and excluding unconfident sentences from aggregate counts.
- **Determinism**: Setting `temperature: 0` ensures reproducible results across runs.

# MILA Feedback

This note compares the Bundesbank technical paper **"Monetary-Intelligent Language Agent (MILA)"** with the current `auto-econ-sentiment` research and software plan.

Source: https://www.bundesbank.de/resource/blob/855186/89fae2a6abdc3ea6de36abe12147269e/472B63F073F071307366337C94F8C870/2025-01-technical-paper-data.pdf

## Bottom Line

MILA is adjacent to this repository, but it is not doing the same thing.

`auto-econ-sentiment` is currently strongest as a reproducible, dictionary-first package for comparing established economic and central-bank sentiment measures. MILA is an LLM-based monetary-policy analysis framework that uses prompt engineering, granular classification, macroeconomic context and deterministic aggregation to produce sentiment and hawkish/dovish indicators.

The paper is therefore best treated as a benchmark for a later v2 or v3 extension, not as a reason to reposition the lexical v1 paper.

## What MILA Does

MILA analyses ECB monetary policy statements and Executive Board speeches. It breaks communication into human-readable units, classifies those units with LLM prompts and aggregates the labels into document-level indicators.

Core features:

1. LLM-based classification rather than dictionary matching.
2. Role-based prompting, prompt chaining and few-shot examples.
3. Sentence, paragraph or category-level classification depending on the task.
4. Context-aware labels, including inflation context and previous text segments.
5. Explanations attached to granular classifications.
6. Deterministic statistical aggregation after classification.
7. Two major indicator families: sentiment and Hawk-O-Meter measures.

The key methodological point is that MILA avoids asking the LLM to directly produce final numerical document scores. Instead, it uses the LLM for bounded classification tasks and calculates final indicators outside the model.

## Comparison With This Repo

| Dimension | `auto-econ-sentiment` | MILA |
| --- | --- | --- |
| Primary method | Dictionary-based lexical scoring, with optional transformer backend | Prompt-engineered LLM agent |
| Main contribution | Reproducible measurement across dictionaries and scoring formulas | Context-aware monetary-policy classification and aggregation |
| Text unit | Document/token level; optional sentence rows for transformer mode | Sentence, paragraph or semantic category |
| Context handling | Mostly preprocessing and model/dictionary choice | Inflation context, previous sentence and document context |
| Transparency | Matched terms, counts and formula-based scores | Segment labels and generated classification reasoning |
| Indicators | Positive/negative sentiment | Sentiment and hawkish/dovish indicators |
| Empirical focus | Central bank speeches/statements across configurable corpora | ECB monetary policy statements and speeches |
| Best repo fit | v1 baseline | Future LLM/prompt-agent extension |

## Implications For The Paper Plan

The existing v1 framing should stay narrow:

1. Present the package as a transparent lexical baseline.
2. Show that dictionary and aggregation choices materially affect sentiment measurement.
3. Emphasize reproducibility, inspectable word matches and configuration-driven analysis.
4. Mention transformer and LLM methods as planned extensions rather than as the main claim.

MILA strengthens the case for this sequence. It shows that leading central-bank work is moving toward granular AI classification, but it also makes clear that the lexical baseline remains valuable because it is deterministic, cheap, auditable and easy to reproduce.

## Potential V2/V3 Extensions Inspired By MILA

1. Add a sentence segmentation stage that preserves `id_text` and sentence order.
2. Add prompt-based classifiers behind a provider-neutral interface.
3. Store prompt version, model version, temperature and schema metadata with outputs.
4. Add typed classification schemas for sentiment, hawkish/dovish direction, topic relevance and time orientation.
5. Aggregate labels using deterministic formulas rather than generated numeric values.
6. Export evidence tables linking document scores back to sentence-level labels.
7. Compare lexical, transformer and prompt-based outputs on the same corpus.

## Suggested Positioning Language

`auto-econ-sentiment` provides a reproducible lexical foundation for economic text sentiment analysis. Recent LLM-based systems such as MILA show the value of granular, context-aware classification for central bank communication. Rather than treating these approaches as substitutes, the package is designed to make measurement choices explicit and to support future comparisons between dictionary, transformer and prompt-based methods.

## Caution

MILA is more sophisticated, but not automatically a better substitute for every use case. Its output depends on model choice, prompt design, inference settings, governance constraints and access to LLM infrastructure. For a reusable open-source package, lexical methods remain a strong first release because they are deterministic and low-friction.

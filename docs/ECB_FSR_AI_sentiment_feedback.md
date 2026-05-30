# ECB FSR AI Sentiment Feedback

This note summarises and positions the ECB Financial Stability Review special feature **"From dictionaries to AI: a new era in sentiment analysis for financial stability"** for the `auto-econ-sentiment` roadmap.

Source: https://www.ecb.europa.eu/press/financial-stability-publications/fsr/special/html/ecb.fsrart202605_01~fe2f4ea541.en.html

## Bottom Line

This ECB article is highly aligned with the repo's research direction. It explicitly compares three text-based approaches:

1. Dictionary-based word counting.
2. Transformer-based sentence classification, using FinBERT.
3. Prompt-based generative AI classification.

The article's most useful lesson is not that AI replaces dictionaries. It argues that the methods measure related but different objects and should be used as complements and cross-checks. That is almost exactly the right framing for `auto-econ-sentiment`.

## What The ECB Article Does

The article studies sentiment in 43 ECB Financial Stability Review issues from 2004 to 2025. It asks how financial stability communication can be measured over time and how text-based tools can support risk monitoring and drafting consistency.

Key methodological points:

1. Dictionary models provide a transparent, deterministic baseline.
2. FinBERT adds sentence-level context and can capture negation and qualification.
3. Prompt-based AI can filter for sentences that contain explicit financial stability risk assessments.
4. A common net sentiment score is built from negative, neutral and positive shares.
5. The methods broadly co-move, but divergences are informative.
6. GPT-based assessment filtering tends to create sharper signals during stress episodes because it focuses on explicit risk judgements.
7. The article cautions that GPT-based results are not a proven accuracy benchmark.

The article also includes a SPOT indicator box. That system uses a structured three-stage LLM prompting process on Financial Times articles to identify potential trigger events, classify their probability and severity, and build a forward-looking financial stability risk indicator.

## Fit With This Repo

| ECB article | Current or planned repo capability |
| --- | --- |
| Dictionary-based financial stability sentiment | Current lexical pipeline |
| Deterministic baseline and reproducibility | Core v1 package claim |
| FinBERT sentence-level classification | Optional transformer backend |
| Prompt-based AI for explicit risk assessments | Future LLM extension |
| Common net sentiment score across methods | Candidate harmonised comparison layer |
| Chapter/sector-level dashboards | Future grouping and reporting layer |
| Drafting consistency checks | Future internal communication-analysis use case |
| SPOT trigger extraction from news | Future event/risk extraction module, outside v1 scope |

## Implications For The Paper Plan

This article supports the current v1-to-v2 structure in `docs/arxiv_paper_plan.md`.

For v1, the repo should emphasize:

1. Dictionary methods are still useful because they are transparent, deterministic and cheap.
2. Running multiple dictionaries side by side is valuable because no dictionary captures the full concept.
3. Variation across dictionaries is not a nuisance; it is evidence about measurement uncertainty.
4. Lexical outputs can serve as a baseline for later transformer and LLM systems.

For v2, the repo can add:

1. Sentence-level transformer scoring.
2. Harmonised negative/neutral/positive share outputs.
3. Comparisons between lexical and transformer sentiment on the same documents.
4. Stability analysis showing when methods agree or diverge.
5. Group-level dashboards by central bank, country, chapter, topic or period.

For a later LLM extension, the article suggests:

1. A relevance filter for assessment-bearing sentences.
2. A time-orientation classifier for backward-looking/current/forward-looking assessments.
3. A risk-direction classifier for improving/deteriorating/balanced assessments.
4. Structured extraction of trigger events, severity, probability, source and horizon.
5. Strict metadata and governance around model, prompt, temperature, provider and data retention.

## Recommended Repo Additions

1. Add an optional harmonised sentiment schema:
   - `count_negative`
   - `count_neutral`
   - `count_positive`
   - `share_negative`
   - `share_neutral`
   - `share_positive`
   - `net_sentiment = share_positive - share_negative` or a configurable sign convention

2. Add a method-comparison report:
   - lexical dictionary scores
   - transformer labels and probabilities
   - pairwise correlations
   - disagreement examples
   - coverage and confidence diagnostics

3. Add grouping hooks:
   - `date`
   - `central_bank`
   - `country`
   - `chapter`
   - `topic`
   - `speaker`

4. Add a future `assessment_bearing` classifier:
   - `assessment_bearing`: true/false
   - `risk_direction`: positive/neutral/negative
   - `time_orientation`: past/current/forward-looking
   - `risk_topic`: banking/markets/NBFI/macro/credit/geopolitical/other

## Suggested Positioning Language

Recent ECB work on financial stability communication shows that dictionary, transformer and prompt-based AI methods should be treated as complementary measurement tools. `auto-econ-sentiment` follows this logic by first providing a reproducible lexical baseline and then extending toward transformer and AI-assisted sentence-level analysis. The package is designed to make methodological disagreement visible rather than hide it behind a single sentiment score.

## Caution

The ECB article is about financial stability reviews, not only monetary policy statements or speeches. Its concepts are especially relevant for risk communication, vulnerability monitoring and financial stability text. The repo can borrow the method-comparison framing immediately, but SPOT-style trigger extraction should remain outside the lexical v1 scope.

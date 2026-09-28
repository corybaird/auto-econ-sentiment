# Paper verification results (51-bank run)

Line numbers refer to the current working copy of `reports/overleaf/main.tex`, which includes the uncommitted Overleaf edits. They differ from the handoff's numbers: handoff line 200 is now 195, 263 is 254, and 276 is 266.

Every number below comes from cached data on this machine, and no model was re-run to produce the panel figures. The only inference was the FOMC-RoBERTa label check in task 6, run on 3,381 labelled sentences on the GPU.

- Scripts: `src/temp/paper_verification.py` (tasks 1–5) and `src/temp/fomc_label_check.py` (task 6).
- Raw tables: `src/temp/paper_verification_tables.md`.

## Summary of verdicts

| Line | Claim | Verdict |
|---|---|---|
| 195 | PosNeg undefined for 3.4–9.4% of statements | **matches** (3.41 / 9.37 / 7.88) |
| 195 | Those documents are reported as missing, not zero | **matches** for `_posneg`; the legacy `net_sentiment` column still stores them as 0 |
| 195 | All-Sentences is defined for every document | **matches** (S_i > 0 for all 6,678) |
| 254 | Transformer PosNeg saturates at 7.5, 11.2, 11.6% | **matches** (FinBERT 7.46, CBRoBERTa 11.19, FOMC 11.58) |
| 254 | Dictionaries span 4.1–27.7% | **matches** (GI 4.09, Apel-BG 27.67) |
| 254 | All-Sentences reaches a bound on at most 1.4% | **matches** (CBRoBERTa 1.35 max) |
| 252–254 | The dispersion gap comes from the denominator, not the method family | **matches** (transformer PosNeg sd 0.40–0.54 vs lexical PosNeg sd 0.37–0.55) |
| 256 | Several dictionaries sit positive; CBRoBERTa sits negative | **matches**, with one caveat: LM also sits negative |
| 266 | Pearson averages: 0.28 lexical / 0.30 transformer / 0.29 cross | **matches** (0.281 / 0.297 / 0.290, All-Sentences, pooled) |
| 266 | Correa–CBRoBERTa 0.47, Correa–FinBERT 0.43 | **matches** (0.465 / 0.435) |
| 266 | Robust to within-bank Pearson and Spearman | **matches** (all family averages between 0.28 and 0.31) |
| 275 | Transformer measures "stay near zero" | **partly contradicts**: CBRoBERTa averages −0.24 and is below zero in every month |
| 224 vs 250 | 6,693 statements from 1990 vs 6,678 scored | **inconsistent**: 15 pre-1994 AU statements are dropped by `start_date` |
| 224 | 35,487 speeches, 143 banks, 1986–2023, ~94.8m words | **matches** (94.5m by whitespace split) |
| VAR (299–322) | Monthly US speech series 2000–2023 | **matches**: no empty months, so interpolation never triggers |
| Table 3 (`transformer_summary.tex`) | FOMC-RoBERTa classes "Pos, Neu, Neg" | **contradicts**: the classes are Dovish / Hawkish / Neutral |
| 315, 317 | VAR significance, signs and magnitudes | **contradicts** (see VAR section; text rewritten) |
| 322 | AIC chooses four lags for both measures | **matches** |

---

## Task 1: 51-bank panel inventory

| Item | Value |
|---|---|
| Banks with a `2_clean` file | 51 |
| Banks with a sentence-probability file (`3_sentiment/sentiment_sentences`) | 51 |
| Banks where re-segmentation aligns row for row with the cache | 51 |
| Documents in `2_clean` / in panel | 6,678 / 6,678 |
| Cached sentences | 245,781 |
| Raw `.txt` statements | 6,693 |

**Verdict:** the sentence-level files exist for all 51 banks, so no GPU rescore is needed. The cache was written without `id_text`, but re-segmenting `2_clean` reproduces the exact sentence count for every bank, which recovers the document link.

**Doc-count gap (lines 224 vs 250):** all 15 missing statements are Australian statements dated 1990–1993, excluded by `corpus.start_date: 1994-01-01`. Line 224 says "6,693 statements … between January 1990", but the scored sample starts in 1994. Either change line 224 to 6,678 and 1994, or say that 15 pre-1994 statements are excluded.

## Task 2: All-Sentences and PosNeg recomputed from sentence probabilities

I counted N⁺, N⁻, N⁰ and S_i independently from the cached probabilities (τ = 0.7, paper label maps) and compared them with the stored columns.

| | CentralBankRoBERTa | FinBERT | FOMC-RoBERTa |
|---|---|---|---|
| max \|recomputed − stored `_posneg`\| | 0 | 0 | 0 |
| max \|recomputed − stored `_allsentences`\| | 0 | 0 | 0 |
| max \|counted-denominator − stored `net_sentiment`\| | 1e-16 | 1e-16 | 1e-16 |
| **PosNeg undefined (% of docs)** | **3.41** | **9.37** | **7.88** |
| Legacy `net_sentiment` = 0 on those docs | 228/228 | 626/626 | 526/526 |
| **PosNeg at ±1 (% of docs)** | **11.19** | **7.46** | **11.58** |
| **All-Sentences at ±1 (% of docs)** | **1.35** | **0.00** | **0.30** |
| Legacy `net_sentiment` at ±1 (%) | 11.19 | 0.10 | 0.34 |

- **Line 195, matches.** "3.4 to 9.4 percent" is exactly the range. The "reported as missing" claim holds for `*_net_sentiment_posneg` (NaN), but the legacy `*_net_sentiment` column still fills those 228–626 documents with 0, so the handoff's "undefined recorded as neutral" finding remains true for that column. The paper no longer uses it, but it is still in the panel and the speech file.
- **Line 254, matches.** The three percentages are correct, but the text lists them in ascending order (7.5, 11.2, 11.6), not in the order the models are introduced. Naming the model beside each number would avoid the implied CBRoBERTa → 7.5.
- For the two-class CBRoBERTa, the legacy `net_sentiment` is identical to PosNeg (11.19% saturated). This confirms the footnote on line 192: the counted-sentences measure "coincides with PosNeg for a two-class model".

## Task 3: Pooled vs within-bank correlations (line 266)

Family averages use the mean of the off-diagonal entries. Lexical measures are recentered with offset 1.

| Transformer column | Centering | Method | Within lexical | Within transformer | Cross-family | Correa–CBRoBERTa | Correa–FinBERT |
|---|---|---|---|---|---|---|---|
| `net_sentiment` (legacy) | pooled | Pearson | 0.281 | 0.285 | 0.292 | 0.435 | 0.438 |
| `net_sentiment` (legacy) | pooled | Spearman | 0.283 | 0.291 | 0.296 | 0.468 | 0.462 |
| `net_sentiment` (legacy) | within-bank | Pearson | 0.286 | 0.296 | 0.292 | 0.445 | 0.405 |
| `net_sentiment` (legacy) | within-bank | Spearman | 0.294 | 0.301 | 0.298 | 0.485 | 0.419 |
| **All-Sentences** | **pooled** | **Pearson** | **0.281** | **0.297** | **0.290** | **0.465** | **0.435** |
| All-Sentences | pooled | Spearman | 0.283 | 0.291 | 0.301 | 0.502 | 0.462 |
| All-Sentences | within-bank | Pearson | 0.286 | 0.308 | 0.289 | 0.484 | 0.402 |
| All-Sentences | within-bank | Spearman | 0.294 | 0.302 | 0.299 | 0.508 | 0.420 |
| PosNeg | pooled | Pearson | 0.281 | 0.292 | 0.305 | 0.441 | 0.429 |
| PosNeg | within-bank | Pearson | 0.286 | 0.316 | 0.304 | 0.451 | 0.397 |

- **Matches.** The pooled Pearson row under All-Sentences (the figure's configuration) gives 0.28 / 0.30 / 0.29, and 0.47 / 0.43 for Correa. The robustness sentence also holds: every centering and method combination keeps the three family averages within 0.28–0.32.
- The paper's numbers come from **All-Sentences**, not the legacy `net_sentiment`. Under the legacy column the within-transformer average is 0.29 (0.285), and FinBERT, not CBRoBERTa, is Correa's top match (0.438 vs 0.435). The current text is consistent with the current figure.
- **Caveat for the interpretation:** the within-transformer average of 0.30 hides a split. CBRoBERTa–FOMC is −0.02, while FOMC–Hubert is 0.52 and FOMC–BN is 0.49. FOMC-RoBERTa lines up with the hawkish/dovish dictionaries (Hubert, BN, Apel-BG), and CBRoBERTa lines up with the financial-tone ones (Correa, LM). That supports "domain drives correlation", but the domain split is stance vs tone rather than central-bank vs general.

## Task 4: Lexical PosNeg vs All-Words, and documents with no matches

I recounted the match counts from `2_clean` with the package's `SentimentLexical`. The recount reproduces the stored PosNeg exactly (max diff 0). A document with no matches is scored 1 (0 after recentering), not NaN, so it cannot be told apart from a balanced document by its score alone.

| Dictionary | No matches (%) | Median matches | PosNeg at ±1 (%) | ±1 excl. no-match (%) | PosNeg sd | All-Words sd | All-Words max \|x\| | corr(PosNeg, AllWords) |
|---|---|---|---|---|---|---|---|---|
| Correa | 11.19 | 8 | 12.62 | 14.21 | 0.490 | 0.015 | 0.077 | 0.87 |
| Hubert-Labondance | 9.55 | 7 | 19.99 | 22.10 | 0.511 | 0.016 | 0.125 | 0.82 |
| Loughran-McDonald | 8.89 | 14 | 6.20 | 6.80 | 0.396 | 0.020 | 0.114 | 0.88 |
| General Inquirer | 2.28 | 18 | 4.09 | 4.18 | 0.370 | 0.023 | 0.129 | 0.92 |
| Bennani-Neuenkirch | 8.60 | 10 | 15.36 | 16.81 | 0.477 | 0.016 | 0.122 | 0.85 |
| Apel-Blix Grimaldi | 12.77 | 5 | 27.67 | 31.73 | 0.551 | 0.013 | 0.110 | 0.78 |

- **Lines 252–254, match.** Saturation tracks median matches: Apel-BG has 5 median matches and 27.7% saturation, while GI has 18 and 4.1%. All-Words never exceeds |0.13|.
- **This is new and not in the paper:** 2–13% of statements have **zero** dictionary matches and are silently scored as neutral (0). That is the lexical counterpart of the transformer "undefined" issue on line 195, but the lexical side treats them as zero rather than missing. A sentence alongside line 195 would make the two families symmetric. Alternatively, lexical PosNeg could return NaN when there are no matches, which is a package change (`sentiment_lexical.py:114-118`).
- **Line 284, matches.** All-Words correlates 0.78–0.92 with PosNeg but its sd is roughly 30× smaller, so it keeps variation where PosNeg is pinned at ±1.

## Levels (lines 256 and 275)

Document-level means, plus the figure's weighting (bank-equal monthly means since 2006):

| Measure | Doc mean | Share > 0 | Fig. monthly mean | Fig. monthly min / max |
|---|---|---|---|---|
| Hubert-Labondance | 0.288 | 0.65 | 0.312 | −0.35 / 0.76 |
| Apel-Blix Grimaldi | 0.321 | 0.63 | 0.342 | −0.18 / 0.79 |
| Bennani-Neuenkirch | 0.241 | 0.63 | 0.259 | −0.40 / 0.64 |
| Correa | 0.005 | 0.41 | 0.017 | −0.53 / 0.43 |
| General Inquirer | −0.038 | 0.39 | −0.046 | −0.32 / 0.23 |
| Loughran-McDonald | −0.196 | 0.22 | −0.200 | −0.53 / 0.17 |
| CentralBankRoBERTa (All-Sent) | −0.243 | 0.14 | −0.240 | −0.47 / **−0.02** |
| FinBERT (All-Sent) | 0.107 | 0.62 | 0.110 | −0.15 / 0.24 |
| FOMC-RoBERTa (All-Sent) | 0.038 | 0.50 | 0.038 | −0.39 / 0.37 |

- **Line 256, matches:** Hubert, BN and Apel-BG sit positive, and CBRoBERTa sits negative. LM also sits clearly negative (−0.20), so "lexical positive vs CBRoBERTa negative" is not a clean family split.
- **Line 275, partly contradicts.** FinBERT and FOMC-RoBERTa do stay near zero, but CBRoBERTa averages −0.24 and its monthly cross-bank mean never goes above −0.02. The S_i denominator explains the *narrower range*, since All-Sentences = PosNeg × the share of sentiment-bearing sentences. It does not explain CBRoBERTa's *level*, which is already −0.38 under PosNeg. The sentence also conflicts with line 256, which correctly says CBRoBERTa "sits persistently negative". Suggested fix: say the transformer measures move within a narrower band because of the S_i denominator, with FinBERT and FOMC-RoBERTa near zero and CBRoBERTa persistently below it.

## Task 5: US speech monthly coverage for the VAR

Source: `data/sentiment/cb_speeches/sentiment_all_results.parquet.gzip`, which holds 6,607 US speeches from 1986 on and already carries the lexical and All-Sentences transformer columns. The file configured as `speech_transformer_scores` (`speech_transformer_us.parquet.gzip`) does not exist, so `SpeechScoreLoader` logs a warning and uses the main file, which already contains `cbroberta_net_sentiment_allsentences`.

| Measure | Speeches since 2000-01 | Span | Months | Months with ≥1 speech | Empty months | Median speeches/month |
|---|---|---|---|---|---|---|
| `hubert_sentiment_posneg` | 4,699 | 2000-01 → 2023-12 | 288 | 288 | 0 | 17 |
| `cbroberta_net_sentiment_allsentences` | 4,699 | 2000-01 → 2023-12 | 288 | 288 | 0 | 17 |

FRED macro covers 1990-01 → 2026-08.

**Verdict, matches:** every month from 2000 to 2023 has speeches (no month has just one), so the handoff's "interpolates rather than forward-fills" point is moot for the VAR. `interpolate(limit_area="inside")` never fills anything. It *does* matter for the statement time-series figure (line 280 caption, `SentimentPanel.monthly`), where the caption correctly says months are interpolated.

## Task 6: FOMC-RoBERTa label semantics and sign

The `config.json` shipped with the model only has generic `LABEL_0/1/2` names. To pin the labels down, I ran the model offline from the local cache on 3,381 human-annotated sentences (`gtfintechlab/all_annotated_sentences_25000` test splits, up to 1,000 per class) and cross-tabulated predictions against the human stance labels (row %):

| Human label | LABEL_0 | LABEL_1 | LABEL_2 |
|---|---|---|---|
| dovish | **49.6** | 13.9 | 36.5 |
| hawkish | 14.7 | **57.4** | 27.9 |
| neutral | 12.5 | 12.8 | **74.7** |
| irrelevant | 3.1 | 2.1 | 94.8 |

**LABEL_0 = dovish, LABEL_1 = hawkish, LABEL_2 = neutral.**

| Config | LABEL_0 (dovish) | LABEL_1 (hawkish) | Implied convention |
|---|---|---|---|
| `src/research_paper/paper_configuration.yaml` | −1 | +1 | hawkish = positive |
| `params.yaml` | positive | negative | dovish = positive |

- The configs are opposite, as the handoff said. The **paper config is the consistent one**. With hawkish = +1, FOMC-RoBERTa correlates positively with the dictionaries that read strong-economy or tightening language as positive (Hubert 0.52, BN 0.49, Apel-BG 0.43), and with FinBERT (0.35). Flipping to `params.yaml`'s sign would make all of those negative. **Recommendation:** align `params.yaml` to the paper config, or document why the package default differs.
- **`tables/transformer_summary.tex` contradicts the model.** It lists FOMC-RoBERTa's target classes as "Pos, Neu, Neg (3)", but they are Hawkish, Dovish and Neutral. The paper should say it maps hawkish → +1 and dovish → −1, and that this is a stance, not tone, which is also what explains the CBRoBERTa–FOMC correlation of −0.02 (task 3).

## VAR results (lines 315–322), checked 2026-09-28

Source: `src/temp/var_check.py`, running the committed `MacroPanel` and `ImpulseResponses` on the current speech and FRED caches. The sample is 288 months, 2000-01 to 2023-12. Significance means outside the 90% band.

| Response | Hubert peak (h) | CBRoBERTa peak (h) | Hubert sig. horizons | CBRoBERTa sig. horizons |
|---|---|---|---|---|
| Industrial production growth | +0.066 (1) | +0.111 (1) | none | 1, 4, 5 |
| CPI inflation | +0.101 (1) | +0.116 (3) | 1 | 3, 4, 7 |
| Unemployment rate | −0.055 (7) | −0.082 (9) | none | 0, 5–24 |
| Short-term rate | +0.073 (14) | ≈0, max +0.022 at 24 | 0, 4–21 | none |
| Credit spread | −0.027 (0), then slightly > 0 | −0.094 (5) | 0 only | 3–8 |

AIC selects 4 lags for both measures, which matches the caption.

The previous text said industrial production and unemployment were significant under both measures, that the short rate cleared the band only under CBRoBERTa, that the credit spread cleared it under Hubert for 9 of 12 months, and that the peaks differed by a factor of seven. None of that holds. The short-rate and credit-spread results were attributed to the wrong measures. The regenerated `var_irf_usa.pdf` is byte-size identical to the working copy's, so the figure was right and the prose misread it. None of the alternative aggregations (legacy `net_sentiment`, PosNeg, Hubert All-Words) reproduce the old text either. I rewrote lines 315 and 317 to match the table.

## Not checked

- **Corpus word counts:** the speech corpus comes to 94.5m words by whitespace split against the paper's "roughly 94.8m". I treated this as consistent.

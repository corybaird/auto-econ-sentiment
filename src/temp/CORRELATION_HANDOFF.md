# Handoff: correlation estimator for sentiment measures

## Task
Add a within-country correlation panel alongside the existing pooled one, and
recompute the family-average numbers quoted in the paper.

## Problem
`main.tex:274` reads the pooled correlation matrix as evidence about *methods*
("0.28 within lexical, 0.29 within transformer, 0.29 cross-family... methods are
not the primary axis of disagreement"). A pooled correlation cannot support that
claim: it stacks all documents with the country label discarded, so it contains
both method agreement AND between-country level differences. If two measures both
score the RBNZ low and the Riksbank high, they correlate even with zero agreement
on any single statement.

## Three estimators (NOT interchangeable)
| Estimator | Operation | N | Contains |
|---|---|---|---|
| Pooled | `.corr()` on stacked rows, country ignored | n_docs | method agreement + country levels |
| Within-country | subtract each bank's own mean, then `.corr()` | n_docs (unchanged) | method agreement only |
| Averaged series | collapse countries per month, then `.corr()` | n_months | shared global cycle only |

Key point: within-country demeaning is a **centering choice**, not a different
estimator. Pearson already subtracts a mean; pooled uses one grand mean per
column, within-country uses each bank's own mean. Nothing is averaged and no rows
are dropped.

## Demo (synthetic, verified)
`src/temp/corr_walkthrough.py` — run: `uv run python -m src.temp.corr_walkthrough`
2 banks x 20 quarterly statements, FED baseline +0.45 / ECB -0.45 on both measures.

    Pooled                      0.547   <- inflated ~8x by country levels
    Within-country Pearson      0.066
    Within-country Spearman     0.060
    Averaged series             0.246

Saturation test (on a deliberately high-agreement pair, since clipping only bites
where there is agreement to destroy; 11 of 40 statements pinned at the bound):

    Pearson   0.955 -> 0.816   (drop 0.139)
    Spearman  0.908 -> 0.892   (drop 0.017)

=> Spearman is saturation-robust. A large **Pearson/Spearman gap** is the
diagnostic that bound-pinning is driving a result. This matters because saturation
is the paper's central argument (`main.tex:261`, `:299`).

## Files to change
- `src/research_paper/pipes/overleaf/figures.py:129` — `SentimentPanel.correlations()`
  currently `self.comparable().corr(method="pearson")`. `comparable()` (line 102)
  recenters lexical by `recenter_offset` onto [-1,1]. It returns measure columns
  ONLY, so the country column must be carried in for grouping.
- `figures.py:198` — `make_level_correlations()`; reuse `_heatmap()` (line 186)
  for the new panels.
- `figures.py:273` — register new figures in the render list.

Add roughly:

    def correlations_within(self, method="pearson"):
        values = self.comparable()
        values["country"] = self.df[<country_col>]
        dm = values.groupby("country")[self.measure_columns].transform(lambda s: s - s.mean())
        return dm.corr(method=method)

## Data
- Config: `src/research_paper/paper_configuration.yaml`
- Panel scores: `data/sentiment/research_paper/panel_sentiment_all.parquet.gzip`
  (key `paths.panel_scores`) — **does not exist yet**; produced by `PanelScorer`
  (`src/research_paper/pipes/sentiment/sentiment.py:181`), per-country cache under
  `data/sentiment/research_paper/3_sentiment`. Statements: `data/raw/statements`.
- Confirm the country column name in the scored panel before grouping; not verified.

## Deliverables
1. Within-country Pearson + Spearman matrices beside the pooled one.
2. Recompute the three family averages at `main.tex:274` on demeaned data. If they
   fall, the paper's claim inverts into a stronger finding (measures agree only
   about which banks are gloomy, not about what any statement says).
3. Per-measure **mean off-diagonal correlation** (use absolute values) to rank
   "which measure is most representative". Report within-family and cross-family
   separately.

## Cautions
- Do not present within-country as the "correct" one. It deletes genuine
  cross-bank agreement by construction. Report both; the GAP is the finding.
- Demeaning removes level shifts only, not scale/saturation differences that vary
  within a bank. Spearman covers that; Pearson alone does not.
- FOMC-RoBERTa is hawkish/dovish, not sentiment (hawkish mapped to positive,
  `main.tex:277`). Its ranking is not comparable; flag it rather than listing it last.
- Pooled is statement-weighted, so prolific banks dominate. `main.tex:291` claims
  equal country weighting, but `figures.py:124` (`resample("MS").mean()`) does not
  do this. Fix the text or the code.
- Figures in `reports/overleaf/figures/` are STALE: they render 5 banks and include
  FinBERT-Tone, which commit 0595c7b disabled. Text says 51 banks / 3 transformers.
  Regenerating will change bank counts; confirm intent first.
- Synthetic magnitudes above are illustrative, not predictions for the real corpus.
  With only 2 banks the averaged estimate fell BELOW pooled; with 51 it typically
  runs high. Do not generalize from the demo's ordering.

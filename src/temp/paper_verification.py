"""Recompute the paper's sentiment statistics from the cached 51-bank run, with no model inference.

Writes intermediate tables to src/temp/paper_verification_tables.md.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.auto_econ_sentiment.clean.text_segmentation import TextSegmenter
from src.auto_econ_sentiment.models.sentiment_lexical import SentimentLexical
from src.research_paper.pipes import load_config, resolve_path

logging.basicConfig(level=logging.WARNING)
OUT = Path(__file__).with_name("paper_verification_tables.md")
TOL = 1e-9


class Verification:
    def __init__(self) -> None:
        self.config = load_config()
        transformer = self.config["transformer"]
        self.cutoff = transformer["sentence_probability_cutoff"]
        self.segmenter = TextSegmenter(text_column="text_clean", min_chars=transformer["min_sentence_chars"])
        self.clean_dir = resolve_path(self.config["paths"]["clean_dir"])
        cache = resolve_path(self.config["paths"]["panel_cache_dir"])
        self.sentence_dir = cache / "sentiment_sentences"
        self.panel = pd.read_parquet(resolve_path(self.config["paths"]["panel_scores"]))
        self.models = {m["short_name"]: m["label_map"] for m in transformer["models"]}
        self.labels = {m["short_name"]: m["label"] for m in transformer["models"]}
        self.lexical = self.config["lexical"]["columns"]
        self.sections: list[str] = []

    def emit(self, title: str, body: str | pd.DataFrame) -> None:
        text = body.to_markdown() if isinstance(body, pd.DataFrame) else body
        self.sections.append(f"## {title}\n\n{text}\n")
        print(f"\n## {title}\n{text}")

    # ---------------------------------------------------------------- task 1 + 2
    def sentence_counts(self) -> pd.DataFrame:
        inventory, frames = [], []
        for path in sorted(self.clean_dir.glob("*.parquet.gzip")):
            country = path.name.split(".")[0]
            clean = pd.read_parquet(path)
            sentence_path = self.sentence_dir / path.name
            row = {"Country": country, "docs_clean": len(clean), "docs_panel": int((self.panel.Country == country).sum()),
                   "sentence_file": sentence_path.exists()}
            if not sentence_path.exists():
                inventory.append(row)
                continue
            probs = pd.read_parquet(sentence_path)
            segmented = self.segmenter.run(clean)
            row.update(sentences_cached=len(probs), sentences_resegmented=len(segmented), aligned=len(probs) == len(segmented))
            inventory.append(row)
            if len(probs) != len(segmented):
                continue
            probs["id_text"] = segmented["id_text"].to_numpy()
            counts = probs.groupby("id_text").size().to_frame("S")
            for short, label_map in self.models.items():
                for direction, sign in (("pos", 1), ("neg", -1), ("neu", 0)):
                    columns = [f"{short}_{label}" for label, value in label_map.items() if value == sign]
                    counts[f"{short}_N{direction}"] = (probs[columns].ge(self.cutoff).any(axis=1).astype(int)
                                                       .groupby(probs["id_text"]).sum() if columns else 0)
            frames.append(counts.reset_index().assign(Country=country))
        self.inventory = pd.DataFrame(inventory).set_index("Country")
        return pd.concat(frames, ignore_index=True)

    def task1(self) -> None:
        inv = self.inventory
        summary = pd.DataFrame({
            "banks with 2_clean file": [len(inv)],
            "banks with sentence file": [int(inv.sentence_file.sum())],
            "banks re-segmentation aligned": [int(inv.aligned.sum())],
            "docs in 2_clean": [int(inv.docs_clean.sum())],
            "docs in panel": [int(inv.docs_panel.sum())],
            "sentences cached": [int(inv.sentences_cached.sum())],
            "raw .txt statements": [len(list(resolve_path("data/raw/statements").rglob("*.txt")))],
        }).T.rename(columns={0: "value"})
        self.emit("Task 1: 51-bank inventory", summary)
        bad = inv[(~inv.aligned) | (inv.docs_clean != inv.docs_panel)]
        self.emit("Task 1: banks with misalignment", bad if len(bad) else "None: every bank aligns row for row.")

    def task2(self, counts: pd.DataFrame) -> pd.DataFrame:
        df = self.panel.merge(counts, on=["Country", "id_text"], how="left", validate="one_to_one")
        rows = []
        for short in self.models:
            S = df["S"]
            pos, neg, neu = df[f"{short}_Npos"], df[f"{short}_Nneg"], df[f"{short}_Nneu"]
            net = pos - neg
            posneg = net / (pos + neg).replace(0, np.nan)
            allsent = net / S.replace(0, np.nan)
            counted = net / (pos + neg + neu).replace(0, np.nan)
            df[f"{short}_re_posneg"], df[f"{short}_re_allsent"] = posneg, allsent
            undefined = posneg.isna()
            n = len(df)
            rows.append({
                "model": self.labels[short],
                "max |re_posneg - stored posneg|": (posneg - df[f"{short}_net_sentiment_posneg"]).abs().max(),
                "max |re_allsent - stored allsent|": (allsent - df[f"{short}_net_sentiment_allsentences"]).abs().max(),
                "max |counted.fillna(0) - stored net_sentiment|": (counted.fillna(0) - df[f"{short}_net_sentiment"]).abs().max(),
                "stored net_sentiment NaN": int(df[f"{short}_net_sentiment"].isna().sum()),
                "stored net_sentiment == 0 on PosNeg-undefined docs": f"{int((df.loc[undefined, f'{short}_net_sentiment'] == 0).sum())}/{int(undefined.sum())}",
                "PosNeg undefined % (all docs)": 100 * undefined.mean(),
                "PosNeg at +-1 % of all docs": 100 * posneg.abs().ge(1 - TOL).sum() / n,
                "PosNeg at +-1 % of defined docs": 100 * posneg.abs().ge(1 - TOL).mean(),
                "AllSent at +-1 % of all docs": 100 * allsent.abs().ge(1 - TOL).sum() / n,
                "old net_sentiment at +-1 %": 100 * df[f"{short}_net_sentiment"].abs().ge(1 - TOL).mean(),
                "docs with S == 0": int((S.fillna(0) == 0).sum()),
            })
        self.emit("Task 2: transformer recomputation, undefined and saturation (lines 195, 254)",
                  pd.DataFrame(rows).set_index("model").T.round(4))
        return df

    # ---------------------------------------------------------------- lexical counts (task 4)
    def lexical_counts(self) -> pd.DataFrame:
        frames = []
        dictionaries = self.config["lexical"]["dictionaries"]
        for path in sorted(self.clean_dir.glob("*.parquet.gzip")):
            clean = pd.read_parquet(path).dropna(subset=["text_clean"]).reset_index(drop=True)
            out = clean[["Country", "id_text"]].copy()
            scorer = SentimentLexical(df_input=clean, text_column="text_tokens_str")
            for kind, column in (("unstemmed", "text_tokens_str"), ("stemmed", "text_stems")):
                for name in dictionaries[kind]:
                    result = scorer.sentiment_pipeline(name, "posneg", text_column=column)
                    out[f"{name}_pos"] = result[f"{name}_counttoken_positive_posneg"].to_numpy()
                    out[f"{name}_neg"] = result[f"{name}_counttoken_negative_posneg"].to_numpy()
                    out[f"{name}_posneg_re"] = result[f"{name}_sentiment_posneg"].to_numpy()
            frames.append(out)
        return pd.concat(frames, ignore_index=True)

    def task4(self, df: pd.DataFrame) -> None:
        lex = self.lexical_counts()
        merged = df.merge(lex, on=["Country", "id_text"], how="left", validate="one_to_one")
        rows = []
        for column, label in self.lexical.items():
            name = column.split("_")[0]
            allwords = column.replace("posneg", "allwords")
            posneg = merged[column] - 1
            aw = merged[allwords] - 1
            matches = merged[f"{name}_pos"] + merged[f"{name}_neg"]
            rows.append({
                "dictionary": label,
                "max |recount posneg - stored|": (merged[f"{name}_posneg_re"] - merged[column]).abs().max(),
                "no matches %": 100 * (matches == 0).mean(),
                "median matches": matches.median(),
                "PosNeg at +-1 %": 100 * posneg.abs().ge(1 - TOL).mean(),
                "PosNeg at +-1 % (excl. no-match)": 100 * posneg[matches > 0].abs().ge(1 - TOL).mean(),
                "PosNeg sd": posneg.std(),
                "AllWords sd": aw.std(),
                "PosNeg IQR": posneg.quantile(.75) - posneg.quantile(.25),
                "AllWords IQR": aw.quantile(.75) - aw.quantile(.25),
                "AllWords max |x|": aw.abs().max(),
                "PosNeg mean": posneg.mean(),
                "corr(PosNeg, AllWords)": posneg.corr(aw),
            })
        self.emit("Task 4: lexical PosNeg vs All-Words (lines 252-254, 284)", pd.DataFrame(rows).set_index("dictionary").T.round(4))

    # ---------------------------------------------------------------- task 3
    def correlation_set(self, df: pd.DataFrame, transformer_columns: dict[str, str]) -> dict:
        labels = {**self.lexical, **transformer_columns}
        values = df[list(labels)].copy()
        values[list(self.lexical)] -= self.config["lexical"]["recenter_offset"]
        results = {}
        for centering in ("pooled", "within-bank"):
            v = values.copy()
            if centering == "within-bank":
                v = v.groupby(df["Country"]).transform(lambda c: c - c.mean())
            for method in ("pearson", "spearman"):
                corr = v.corr(method=method)
                corr.index = corr.columns = [labels[c] for c in corr.columns]
                results[(centering, method)] = corr
        return results

    def family_summary(self, corr: pd.DataFrame) -> dict:
        lex = list(self.lexical.values())
        trf = [c for c in corr.columns if c not in lex]
        upper = lambda block: block.where(np.triu(np.ones(block.shape, bool), 1)).stack().mean()
        return {
            "within lexical": upper(corr.loc[lex, lex]),
            "within transformer": upper(corr.loc[trf, trf]),
            "cross-family": corr.loc[lex, trf].to_numpy().mean(),
            "Correa-CentralBankRoBERTa": corr.loc["Correa", "CentralBankRoBERTa"],
            "Correa-FinBERT": corr.loc["Correa", "FinBERT"],
            "Correa-FOMC-RoBERTa": corr.loc["Correa", "FOMC-RoBERTa"],
            "top transformer for Correa": corr.loc["Correa", trf].idxmax(),
        }

    def task3(self, df: pd.DataFrame) -> None:
        variants = {
            "net_sentiment (old)": {f"{s}_net_sentiment": self.labels[s] for s in self.models},
            "All-Sentences": {f"{s}_net_sentiment_allsentences": self.labels[s] for s in self.models},
            "PosNeg": {f"{s}_net_sentiment_posneg": self.labels[s] for s in self.models},
        }
        rows = {}
        for variant, columns in variants.items():
            for key, corr in self.correlation_set(df, columns).items():
                rows[(variant, *key)] = self.family_summary(corr)
                if variant == "All-Sentences" and key == ("pooled", "pearson"):
                    self.emit("Task 3: pooled Pearson matrix, All-Sentences (the figure's configuration)", corr.round(2))
        table = pd.DataFrame(rows).T
        table.index.names = ["transformer aggregation", "centering", "method"]
        self.emit("Task 3: correlation family averages (line 266)", table.map(lambda x: round(x, 3) if isinstance(x, float) else x))

    # ---------------------------------------------------------------- levels (lines 256, 275)
    def levels(self, df: pd.DataFrame) -> None:
        rows = {}
        for column, label in self.lexical.items():
            v = df[column] - self.config["lexical"]["recenter_offset"]
            rows[f"Lexical: {label}"] = {"mean": v.mean(), "median": v.median(), "sd": v.std(), "share > 0": (v > 0).mean()}
        for short in self.models:
            for suffix in ("allsentences", "posneg"):
                v = df[f"{short}_net_sentiment_{suffix}"]
                rows[f"{self.labels[short]} ({suffix})"] = {"mean": v.mean(), "median": v.median(), "sd": v.std(), "share > 0": (v > 0).mean()}
        self.emit("Levels: central tendency by measure (lines 256, 275)", pd.DataFrame(rows).T.round(3))

    # ---------------------------------------------------------------- task 5
    def task5(self) -> None:
        speeches = pd.read_parquet(resolve_path(self.config["paths"]["speech_scores"]))
        speeches["date"] = pd.to_datetime(speeches["date"], errors="coerce")
        speeches = speeches[speeches.Country == "US"].dropna(subset=["date"])
        start = pd.Timestamp(self.config["var"]["sample_start"])
        rows = {}
        for measure in self.config["var"]["measures"]:
            s = speeches.dropna(subset=[measure]).set_index("date")[measure].resample("ME").agg(["mean", "size"])
            s = s[s.index >= start]
            months = pd.date_range(start + pd.offsets.MonthEnd(0), s.index.max(), freq="ME")
            per_month = s["size"].reindex(months, fill_value=0)
            empty = per_month[per_month == 0]
            rows[measure] = {
                "speeches (since start)": int(per_month.sum()), "first month": months.min().date(), "last month": months.max().date(),
                "months in span": len(months), "months with >=1 speech": int((per_month > 0).sum()),
                "empty months (interpolated)": len(empty),
                "empty month list": ", ".join(d.strftime("%Y-%m") for d in empty.index[:20]) or "none",
                "median speeches/month": per_month.median(), "months with 1 speech": int((per_month == 1).sum()),
            }
        macro = pd.read_parquet(resolve_path(self.config["paths"]["macro_data"]))
        rows["FRED macro"] = {"first month": pd.to_datetime(macro.date).min().date(), "last month": pd.to_datetime(macro.date).max().date(),
                              "months in span": len(macro)}
        self.emit("Task 5: US speech monthly coverage for the VAR", pd.DataFrame(rows).T)

    def run(self) -> None:
        counts = self.sentence_counts()
        self.task1()
        df = self.task2(counts)
        self.task3(df)
        self.levels(df)
        self.task4(df)
        self.task5()
        OUT.write_text("# Paper verification tables (generated)\n\n" + "\n".join(self.sections))


if __name__ == "__main__":
    Verification().run()

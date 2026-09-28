import numpy as np
import pandas as pd

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)


class CorrelationWalkthrough:
    def __init__(self, seed: int = 7, n_per_country: int = 20) -> None:
        self.seed = seed
        self.n_per_country = n_per_country
        self.measures = ["lex_correa", "trf_cbroberta"]

    def _build(self) -> pd.DataFrame:
        rng = np.random.default_rng(self.seed)
        n = self.n_per_country
        dates = pd.date_range("2019-01-01", periods=n, freq="QS")

        # Shared quarterly cycle both measures partly track (e.g. 2020 shock).
        cycle = np.sin(np.linspace(0, 3 * np.pi, n)) * 0.30

        frames = []
        # country_level: each bank's baseline tone. FED sits high, ECB sits low,
        # on BOTH measures. This is the between-country component.
        for country, country_level in [("FED", 0.45), ("ECB", -0.45)]:
            # Statement-specific signal BOTH measures partly see. Weighted low
            # (0.25) against large independent noise, so the two measures agree
            # only weakly about any single statement -- which is the empirically
            # realistic case (the paper's pooled correlations run around 0.3).
            shock = rng.normal(0, 0.30, n)
            lex = country_level + cycle + 0.25 * shock + rng.normal(0, 0.45, n)
            trf = country_level + cycle + 0.25 * shock + rng.normal(0, 0.45, n)
            frames.append(pd.DataFrame({"date": dates, "country": country, "lex_correa": lex, "trf_cbroberta": trf}))

        df = pd.concat(frames, ignore_index=True)
        df[self.measures] = df[self.measures].clip(-1, 1)
        return df.sort_values(["date", "country"]).reset_index(drop=True)

    def _pooled(self, df: pd.DataFrame) -> pd.DataFrame:
        return df[self.measures].corr()

    def _demeaned(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        # Subtract each country's OWN mean from each of its rows.
        out[self.measures] = df.groupby("country")[self.measures].transform(lambda s: s - s.mean())
        return out

    def _build_agreeing(self) -> pd.DataFrame:
        # Same structure as _build but the two measures share most of their
        # statement-level variation, so there is real agreement to destroy.
        rng = np.random.default_rng(self.seed)
        n = self.n_per_country
        dates = pd.date_range("2019-01-01", periods=n, freq="QS")
        cycle = np.sin(np.linspace(0, 3 * np.pi, n)) * 0.30
        frames = []
        for country, country_level in [("FED", 0.45), ("ECB", -0.45)]:
            shock = rng.normal(0, 0.35, n)
            lex = country_level + cycle + shock + rng.normal(0, 0.08, n)
            trf = country_level + cycle + shock + rng.normal(0, 0.08, n)
            frames.append(pd.DataFrame({"date": dates, "country": country, "lex_correa": lex, "trf_cbroberta": trf}))
        return pd.concat(frames, ignore_index=True)

    def _within_spearman(self, df: pd.DataFrame) -> pd.DataFrame:
        # Rank WITHIN each country, then correlate. Ranks survive saturation:
        # a measure pinned at +1 still orders the statements below the bound.
        ranked = df.copy()
        ranked[self.measures] = df.groupby("country")[self.measures].rank()
        centered = ranked.copy()
        centered[self.measures] = ranked.groupby("country")[self.measures].transform(lambda s: s - s.mean())
        return centered[self.measures].corr()

    def _saturate(self, df: pd.DataFrame, bound: float = 0.55) -> pd.DataFrame:
        # Mimic lexical saturation: the dictionary measure pins at the bound.
        out = df.copy()
        out["lex_correa"] = out["lex_correa"].clip(-bound, bound)
        return out

    def _monthly(self, df: pd.DataFrame) -> pd.DataFrame:
        # Collapse countries within each date to one row.
        return df.groupby("date")[self.measures].mean()

    def run(self) -> None:
        df = self._build()

        print("=" * 100)
        print("STEP 0 -- RAW PANEL (one row per statement). N =", len(df))
        print("=" * 100)
        print(df.head(6).to_string(index=False))
        print("...")
        print(df.tail(4).to_string(index=False))

        print("\n" + "=" * 100)
        print("STEP 1 -- COUNTRY BASELINES (the between-country component)")
        print("=" * 100)
        print(df.groupby("country")[self.measures].agg(["mean", "std"]).round(3).to_string())

        print("\n" + "=" * 100)
        print("STEP 2a -- POOLED: country label DISCARDED, all rows stacked. N =", len(df))
        print("=" * 100)
        grand = df[self.measures].mean()
        print("Grand means subtracted by Pearson (one per COLUMN, not per country):")
        print(grand.round(3).to_string())
        pooled_dev = df[self.measures] - grand
        show = df[["country"]].join(pooled_dev.round(3).add_suffix("_dev"))
        print("\nDeviations from the GRAND mean (first 4 FED, first 4 ECB):")
        print(pd.concat([show[show.country == "FED"].head(4), show[show.country == "ECB"].head(4)]).to_string(index=False))
        print("\n-> Note both FED rows are mostly POSITIVE on both measures, both ECB rows mostly NEGATIVE.")
        print("   That shared sign is the country level, and it inflates the correlation.")
        pooled = self._pooled(df)
        print("\nPOOLED correlation:")
        print(pooled.round(3).to_string())

        print("\n" + "=" * 100)
        print("STEP 2b -- WITHIN-COUNTRY: subtract each country's OWN mean. N =", len(df), "(unchanged)")
        print("=" * 100)
        dm = self._demeaned(df)
        print("Each country's mean is now exactly zero by construction:")
        print(dm.groupby("country")[self.measures].mean().round(6).to_string())
        show2 = dm[["country"]].join(dm[self.measures].round(3).add_suffix("_dm"))
        print("\nResiduals (first 4 FED, first 4 ECB):")
        print(pd.concat([show2[show2.country == "FED"].head(4), show2[show2.country == "ECB"].head(4)]).to_string(index=False))
        print("\n-> Signs are now MIXED within each country. The level is gone; only")
        print("   'is this statement hot relative to its own bank?' remains.")
        demeaned = self._pooled(dm)
        print("\nWITHIN-COUNTRY correlation:")
        print(demeaned.round(3).to_string())

        print("\n" + "=" * 100)
        print("STEP 2c -- MONTHLY AVERAGE: collapse countries, THEN correlate")
        print("=" * 100)
        monthly = self._monthly(df)
        print(f"Rows destroyed: N goes {len(df)} -> {len(monthly)}")
        print(monthly.head(5).round(3).to_string())
        averaged = self._pooled(monthly)
        print("\nAVERAGED-SERIES correlation:")
        print(averaged.round(3).to_string())

        print("\n" + "=" * 100)
        print("STEP 3 -- WITHIN-COUNTRY SPEARMAN (rank-based, saturation-robust)")
        print("=" * 100)
        spearman = self._within_spearman(df)
        print("Within-country Pearson :", round(demeaned.iloc[0, 1], 3))
        print("Within-country Spearman:", round(spearman.iloc[0, 1], 3))

        print("\n-- SATURATION TEST --")
        print("Clipping only bites where there IS agreement to destroy. At r=0.07")
        print("above there is no signal to lose, so we run this step on a")
        print("HIGH-AGREEMENT pair (seed 7, shared shock) to isolate the effect.")
        agree = self._build_agreeing()
        base_p = self._pooled(self._demeaned(agree)).iloc[0, 1]
        base_s = self._within_spearman(agree).iloc[0, 1]
        sat = self._saturate(agree)
        n_pinned = int((sat["lex_correa"].abs() >= 0.549).sum())
        sat_pearson = self._pooled(self._demeaned(sat)).iloc[0, 1]
        sat_spearman = self._within_spearman(sat).iloc[0, 1]
        print(f"\nStatements pinned at the bound: {n_pinned} of {len(sat)}")
        print(f"Pearson : {round(base_p, 3)} -> {round(sat_pearson, 3)}  (drop {round(base_p - sat_pearson, 3)})")
        print(f"Spearman: {round(base_s, 3)} -> {round(sat_spearman, 3)}  (drop {round(base_s - sat_spearman, 3)})")
        print("\n-> Spearman retains more than Pearson: clipping destroys magnitudes")
        print("   but preserves ordering among the unpinned statements. A large")
        print("   Pearson/Spearman GAP is the diagnostic that saturation is biting.")

        print("\n" + "=" * 100)
        print("SUMMARY -- same data, same pair of measures, five estimates")
        print("=" * 100)
        summary = pd.DataFrame({
            "estimate": ["Pooled (country ignored)", "Within-country Pearson", "Within-country Spearman", "[agreeing pair] Pearson, saturated", "[agreeing pair] Spearman, saturated", "Averaged series"],
            "N": [len(df), len(dm), len(dm), len(dm), len(dm), len(monthly)],
            "corr(lex, trf)": [pooled.iloc[0, 1], demeaned.iloc[0, 1], spearman.iloc[0, 1], sat_pearson, sat_spearman, averaged.iloc[0, 1]],
            "what it contains": ["method agreement + country levels", "method agreement only", "ORDER agreement only", "magnitudes destroyed by clipping", "ordering survives clipping", "shared cycle only"],
        })
        print(summary.round(3).to_string(index=False))


if __name__ == "__main__":
    CorrelationWalkthrough().run()

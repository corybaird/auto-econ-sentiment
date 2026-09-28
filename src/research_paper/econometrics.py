"""The VAR case study: FRED macro data, the monthly panel, and impulse responses."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR

from src.research_paper.config import PROJECT_ROOT, PaperConfig

logger = logging.getLogger(__name__)

_Z_SCORES = {0.90: 1.645, 0.95: 1.96}


class FredMacro:
    """The monthly US macro block, fetched from FRED once and cached."""

    def __init__(self, config: PaperConfig) -> None:
        self.macro = config["macro"]
        self.path = config.path("macro")

    def load(self, force: bool = False) -> pd.DataFrame:
        if self.path.exists() and not force:
            return pd.read_parquet(self.path)
        macro = self._fetch()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        macro.to_parquet(self.path, compression="gzip", index=False)
        logger.info("Wrote %s: %d months", self.path, len(macro))
        return macro

    def _fetch(self) -> pd.DataFrame:
        from fredapi import Fred

        fred = Fred(api_key=_fred_api_key())
        # Daily series (the rate and the spread) are averaged to month start.
        series = {
            name: fred.get_series(series_id, observation_start=self.macro["start"]).resample("MS").mean()
            for name, series_id in self.macro["series"].items()
        }
        return pd.DataFrame(series).rename_axis("date").reset_index()


class MacroPanel:
    """Monthly sentiment joined to the macro variables the VAR uses."""

    def __init__(self, config: PaperConfig) -> None:
        self.var = config["var"]
        self.macro = config["macro"]
        self.fred = FredMacro(config)

    def build(self, speeches: pd.DataFrame) -> pd.DataFrame:
        """One row per month from ``var.sample_start``, gaps interpolated, incomplete rows dropped."""
        panel = self._macro_variables().join(self._monthly_sentiment(speeches), how="inner")
        panel = panel[panel.index >= pd.Timestamp(self.var["sample_start"])]
        panel = panel.asfreq("ME").interpolate(limit_area="inside").dropna()
        logger.info("VAR panel: %d months from %s to %s", len(panel), panel.index.min().date(), panel.index.max().date())
        return panel

    def _macro_variables(self) -> pd.DataFrame:
        macro = self.fred.load().assign(date=lambda frame: pd.to_datetime(frame["date"]) + pd.offsets.MonthEnd(0))
        macro = macro.sort_values("date").set_index("date")
        growth = {name: 100 * np.log(macro[column]).diff() for column, name in self.macro["log_growth"].items()}
        levels = {name: macro[column] for column, name in self.macro["levels"].items()}
        return pd.DataFrame({**growth, **levels})[self.var["variables"][1:]]

    def _monthly_sentiment(self, speeches: pd.DataFrame) -> pd.DataFrame:
        dated = speeches.assign(date=pd.to_datetime(speeches["date"], errors="coerce")).dropna(subset=["date"])
        return dated.set_index("date")[list(self.var["measures"])].resample("ME").mean()


@dataclass
class ImpulseResponse:
    """Orthogonalized responses to a one standard deviation sentiment shock, with bands."""

    point: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    lags: int

    def significant_months(self, variable: int) -> list[int]:
        """Horizons at which the band excludes zero for the response of ``variable``."""
        excludes_zero = (self.lower[:, variable] > 0) | (self.upper[:, variable] < 0)
        return [int(month) for month in np.flatnonzero(excludes_zero)]


class ImpulseResponses:
    """Fit one recursive VAR per sentiment measure, with sentiment ordered first."""

    def __init__(self, config: PaperConfig) -> None:
        self.var = config["var"]
        self.z_score = _Z_SCORES[self.var["confidence"]]

    def fit(self, panel: pd.DataFrame) -> dict[str, ImpulseResponse]:
        responses = {}
        for measure in self.var["measures"]:
            data = self._standardized(panel, measure)
            if len(data) < self.var["min_observations"]:
                logger.warning("Skipping %s: only %d observations", measure, len(data))
                continue
            responses[measure] = self._impulse_response(data)
            logger.info("Fitted VAR for %s with %d lags", measure, responses[measure].lags)
        return responses

    def _standardized(self, panel: pd.DataFrame, measure: str) -> pd.DataFrame:
        data = panel[self.var["variables"][1:]].copy()
        data.insert(0, "sentiment", panel[measure])
        return (data - data.mean()) / data.std(ddof=0)

    def _impulse_response(self, data: pd.DataFrame) -> ImpulseResponse:
        # Lag length is chosen by AIC, capped at maxlags and at a tenth of the sample.
        max_lags = min(self.var["maxlags"], max(1, len(data) // 10))
        results = VAR(data.reset_index(drop=True)).fit(maxlags=max_lags, ic="aic")
        irf = results.irf(self.var["irf_periods"])
        point = irf.orth_irfs[:, :, 0]
        band = self.z_score * irf.stderr(orth=True)[:, :, 0]
        return ImpulseResponse(point=point, lower=point - band, upper=point + band, lags=results.k_ar)


def _fred_api_key() -> str:
    """The FRED key from FRED_API_KEY or API_FRED, in the environment or a local .env file."""
    names = ("FRED_API_KEY", "API_FRED")
    for name in names:
        if os.environ.get(name):
            return os.environ[name].strip()
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            name, separator, value = line.partition("=")
            if separator and name.strip() in names:
                return value.strip().strip("'\"")
    raise RuntimeError(f"No FRED API key in {' or '.join(names)}, or in {env_path}.")

"""One recursive VAR per sentiment measure, and its impulse responses with bands."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR

from src.research_paper.config import PaperConfig

logger = logging.getLogger(__name__)


_Z_SCORES = {0.90: 1.645, 0.95: 1.96}


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

"""Recompute the VAR claims in paper Section 4.3 from the configured speech panel."""
import logging
import numpy as np
import pandas as pd
from src.research_paper.pipes import load_config
from src.research_paper.pipes.econometric.stats_transformer import ImpulseResponses, MacroPanel

logging.basicConfig(level=logging.WARNING)
config = load_config()
panel = MacroPanel(config).run()
print("panel", len(panel), panel.index.min().date(), panel.index.max().date())
results = ImpulseResponses(config["var"]).run(panel)
names = config["var"]["variables"]
rows = []
for measure, r in results.items():
    print(measure, "lags", r["lags"])
    for j, var in enumerate(names[1:], start=1):
        p, lo, hi = r["point"][:, j], r["lower"][:, j], r["upper"][:, j]
        sig = (lo > 0) | (hi < 0)
        k = int(np.abs(p).argmax())
        rows.append({"measure": measure.split("_")[0], "response": var, "peak h": k, "peak": p[k],
                     "sig h1-12": int(sig[1:13].sum()), "sig h0-12": int(sig[:13].sum()), "sig h0-24": int(sig.sum()),
                     "sig hs (0-24)": ",".join(map(str, np.flatnonzero(sig)))})
df = pd.DataFrame(rows)
print(df.round(3).to_string())
wide = df.pivot(index="response", columns="measure", values="peak")
wide["ratio cb/hub"] = wide["cbroberta"] / wide["hubert"]
print(wide.round(3))

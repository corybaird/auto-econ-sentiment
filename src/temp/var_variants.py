import logging, copy
import numpy as np
from src.research_paper.pipes import load_config
from src.research_paper.pipes.econometric.stats_transformer import ImpulseResponses, MacroPanel
from src.research_paper.pipes.sentiment.corpus import SpeechScoreLoader
logging.basicConfig(level=logging.ERROR)
config = load_config()
alts = ["cbroberta_net_sentiment", "cbroberta_net_sentiment_posneg", "cbroberta_net_sentiment_allsentences", "hubert_sentiment_posneg", "hubert_sentiment_allwords"]
config["var"]["measures"] = {m: m for m in alts}
scores = SpeechScoreLoader(config).load()
panel = MacroPanel(config).run(scores)
for m, r in ImpulseResponses(config["var"]).run(panel).items():
    out = []
    for j, v in enumerate(config["var"]["variables"][1:], 1):
        p, lo, hi = r["point"][:, j], r["lower"][:, j], r["upper"][:, j]
        sig = (lo > 0) | (hi < 0); k = int(np.abs(p).argmax())
        out.append(f"{v[:6]}:{p[k]:+.3f}@{k} s{int(sig[1:13].sum())}")
    print(f"{m:40s} lags={r['lags']} " + " | ".join(out))

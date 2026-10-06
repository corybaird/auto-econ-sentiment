"""Score the corpora with the released auto-econ-sentiment pipeline.

Every score in the paper comes from ``PackageScorer``, which runs the package exactly
as a user would: load, clean, lexical scoring and transformer scoring through
``AutoEconSentiment.run``. The panels decide what to feed it and cache each unit of
work so an interrupted run resumes where it stopped.
"""

from src.research_paper.scoring.header_experiment import HeaderExperiment
from src.research_paper.scoring.package_scorer import PackageScorer, ScoredDocuments
from src.research_paper.scoring.panels import CachedPanel, SpeechPanel, StatementPanel

__all__ = ["CachedPanel", "HeaderExperiment", "PackageScorer", "ScoredDocuments", "SpeechPanel", "StatementPanel"]

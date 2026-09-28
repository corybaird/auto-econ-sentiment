"""Build every figure, table and statistic in reports/overleaf/main.tex."""

from __future__ import annotations

import logging
from functools import cached_property
from pathlib import Path

from src.research_paper.config import PaperConfig
from src.research_paper.corpora import SpeechCorpus, StatementCorpus
from src.research_paper.econometrics import ImpulseResponse, ImpulseResponses, MacroPanel
from src.research_paper.exhibits.figures import CleaningFigure, MethodFigures, PipelineDiagram, VarFigure
from src.research_paper.exhibits.tables import PaperTables
from src.research_paper.scoring import HeaderExperiment, ScoredDocuments, SpeechPanel, StatementPanel
from src.research_paper.statistics import PaperStatistics

logger = logging.getLogger(__name__)


class PaperPipeline:
    """Run the paper's stages in dependency order.

    The two scoring stages are the expensive ones: they run the transformer models
    and cache every central bank and every year, so a rerun only scores what is
    missing. Later stages load the cached panels, which makes
    ``--stages figures,var,tables,statistics`` a rebuild from existing scores.
    ``force`` rescores everything.
    """

    STAGES = ("statements", "speeches", "figures", "var", "tables", "statistics")

    def __init__(self, config: PaperConfig | None = None, force: bool = False) -> None:
        self.config = config or PaperConfig()
        self.force = force

    def run(self, stages: list[str] | None = None) -> list[Path]:
        outputs: list[Path] = []
        for stage in self._resolve(stages):
            logger.info("Running stage: %s", stage)
            outputs.extend(getattr(self, f"_stage_{stage}")())
        logger.info("Produced %d files", len(outputs))
        return outputs

    # -- stages ---------------------------------------------------------------

    def _stage_statements(self) -> list[Path]:
        self.__dict__["statements"] = StatementPanel(self.config, self.force).build()
        return []

    def _stage_speeches(self) -> list[Path]:
        self.__dict__["speeches"] = SpeechPanel(self.config, self.force).build()
        return []

    def _stage_figures(self) -> list[Path]:
        clean, contaminated = HeaderExperiment(self.config, self.force).run()
        return [
            *PipelineDiagram(self.config).run(),
            *CleaningFigure(self.config, clean, contaminated).run(),
            *MethodFigures(self.config, self.statements.documents).run(),
        ]

    def _stage_var(self) -> list[Path]:
        return VarFigure(self.config, self.impulse_responses).run()

    def _stage_tables(self) -> list[Path]:
        statements = StatementCorpus(self.config).archive()
        speeches = SpeechCorpus(self.config).archive()
        return PaperTables(self.config, statements, speeches).run()

    def _stage_statistics(self) -> list[Path]:
        return PaperStatistics(self.config, self.statements, self.impulse_responses).run()

    # -- shared inputs, loaded once and reused by every stage that needs them --

    @cached_property
    def statements(self) -> ScoredDocuments:
        return StatementPanel(self.config).load()

    @cached_property
    def speeches(self) -> ScoredDocuments:
        return SpeechPanel(self.config).load()

    @cached_property
    def impulse_responses(self) -> dict[str, ImpulseResponse]:
        panel = MacroPanel(self.config).build(self.speeches.documents)
        return ImpulseResponses(self.config).fit(panel)

    def _resolve(self, stages: list[str] | None) -> list[str]:
        if not stages:
            return list(self.STAGES)
        unknown = sorted(set(stages) - set(self.STAGES))
        if unknown:
            raise ValueError(f"Unknown stage(s) {unknown}. Valid stages: {list(self.STAGES)}")
        return [stage for stage in self.STAGES if stage in stages]

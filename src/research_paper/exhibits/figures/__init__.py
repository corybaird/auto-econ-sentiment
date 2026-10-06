"""Every figure in the paper, one module per figure or figure family."""

from src.research_paper.exhibits.figures.header_contamination import HeaderContaminationFigure
from src.research_paper.exhibits.figures.impulse_responses import ImpulseResponseFigure
from src.research_paper.exhibits.figures.method_comparison import MethodComparisonFigures
from src.research_paper.exhibits.figures.workflow_diagram import WorkflowDiagram

__all__ = ["HeaderContaminationFigure", "ImpulseResponseFigure", "MethodComparisonFigures", "WorkflowDiagram"]

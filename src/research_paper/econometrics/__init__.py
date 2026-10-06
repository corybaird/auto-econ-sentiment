"""The VAR case study: FRED macro data, the monthly panel, and impulse responses."""

from src.research_paper.econometrics.macro import FredMacro, MacroPanel
from src.research_paper.econometrics.var import ImpulseResponse, ImpulseResponses

__all__ = ["FredMacro", "ImpulseResponse", "ImpulseResponses", "MacroPanel"]

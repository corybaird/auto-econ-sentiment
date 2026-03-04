from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("auto_econ_sentiment")
except PackageNotFoundError:
    __version__ = "unknown"

from auto_econ_sentiment.pipeline import AutoEconSentiment
from auto_econ_sentiment.models.sentiment_lexical import SentimentLexical
from auto_econ_sentiment.clean.text_loader import TextLoader
from auto_econ_sentiment.clean.text_clean import TextCleaner
from auto_econ_sentiment.exceptions import (
    AutoEconSentimentError,
    ConfigurationError,
    DataLoadError,
    SentimentAnalysisError,
)

__all__ = [
    "AutoEconSentiment",
    "SentimentLexical",
    "TextLoader",
    "TextCleaner",
    "AutoEconSentimentError",
    "ConfigurationError",
    "DataLoadError",
    "SentimentAnalysisError",
    "__version__",
]

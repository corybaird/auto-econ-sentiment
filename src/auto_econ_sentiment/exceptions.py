class AutoEconSentimentError(Exception):
    pass


class ConfigurationError(AutoEconSentimentError):
    pass


class DataLoadError(AutoEconSentimentError):
    pass


class SentimentAnalysisError(AutoEconSentimentError):
    pass

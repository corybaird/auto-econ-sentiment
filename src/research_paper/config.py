"""Read paper_configuration.yaml and resolve its paths against the project root."""

from __future__ import annotations

from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = Path(__file__).with_name("paper_configuration.yaml")


class PaperConfig:
    """Section access to the paper configuration, plus the settings derived from it."""

    def __init__(self, path: Path = CONFIG_PATH) -> None:
        with Path(path).open(encoding="utf-8") as file:
            self._sections: dict = yaml.safe_load(file)

    def __getitem__(self, section: str) -> dict:
        return self._sections[section]

    def path(self, key: str) -> Path:
        """A path from the ``paths`` section, absolute under the project root."""
        return _resolve(self._sections["paths"][key])

    def data_path(self, *parts: str) -> Path:
        """A location for generated data under ``paths.data_dir``."""
        return self.path("data_dir").joinpath(*parts)

    def transformer_settings(self) -> dict:
        """The ``transformer`` block in the form AutoEconSentiment.run expects."""
        transformer = self._sections["transformer"]
        return {
            "enabled": True,
            "text_column_transformer": "text_clean",
            "aggregation": "bysentence",
            "models": [self._package_model(model) for model in transformer["models"]],
            **{key: transformer[key] for key in _SHARED_TRANSFORMER_KEYS},
        }

    def model_labels(self) -> dict[str, str]:
        """Transformer short name to display label, e.g. ``cbroberta`` to ``CentralBankRoBERTa``."""
        return {model["short_name"]: model["label"] for model in self._sections["transformer"]["models"]}

    @staticmethod
    def _package_model(model: dict) -> dict:
        # ``label`` is a display name for the paper; the package does not accept it.
        return {key: value for key, value in model.items() if key != "label"}


_SHARED_TRANSFORMER_KEYS = (
    "min_sentence_chars",
    "sentence_probability_cutoff",
    "output_schema",
    "net_sentiment_formula",
)


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path

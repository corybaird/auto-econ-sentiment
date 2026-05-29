import os
import logging
import argparse
from typing import Optional, Union
import pandas as pd
from tqdm import tqdm
from pathlib import Path

from auto_econ_sentiment.utils.load_yaml import load_yaml_config
from auto_econ_sentiment.clean.text_loader import TextLoader
from auto_econ_sentiment.clean.text_clean import TextCleaner
from auto_econ_sentiment.models.sentiment_lexical import SentimentLexical
from auto_econ_sentiment.exceptions import DataLoadError, SentimentAnalysisError

logger = logging.getLogger(__name__)


class AutoEconSentiment:
    """End-to-end pipeline that loads a text corpus, cleans it, and scores it
    with one or more lexical sentiment dictionaries."""

    def __init__(
        self,
        import_file_path: Union[str, Path],
        text_column: str,
        date_column: str,
        export_path: Union[str, Path],
    ) -> None:
        self.import_file = import_file_path
        self.export_path = export_path
        self.text_column = text_column
        self.date_column = date_column
        os.makedirs(self.export_path, exist_ok=True)
        self.df_raw: Optional[pd.DataFrame] = None
        self.df_clean: Optional[pd.DataFrame] = None
        self.df_sent_lexical: Optional[pd.DataFrame] = None
        self.df_sent_transformer: Optional[pd.DataFrame] = None
        self.df_transformer_sentence_probabilities: Optional[pd.DataFrame] = None
        logger.info("AutoEconSentiment initialized successfully")

    def load_data(self) -> pd.DataFrame:
        """Load the input file via :class:`TextLoader` and return the raw DataFrame."""
        logger.info("Loading data...")
        try:
            loader = TextLoader(
                file_path=self.import_file,
                text_column=self.text_column,
                date_column=self.date_column,
            )
            self.df_raw = loader.get_data()
        except Exception as e:
            raise DataLoadError(f"Failed to load data from {self.import_file}: {e}") from e
        logger.info(f"Data loaded successfully. Shape: {self.df_raw.shape}")
        return self.df_raw

    def clean_data(self, clean_config: Optional[dict] = None) -> pd.DataFrame:
        """Run :class:`TextCleaner` on the loaded data and return the cleaned DataFrame."""
        logger.info("Cleaning data...")
        cleaner = TextCleaner(
            df=self.df_raw,
            text_column=self.text_column,
            export_path=self.export_path,
            clean_config=clean_config,
        )
        self.df_clean = cleaner.run()
        return self.df_clean

    def analyze_sentiment_lexical(
        self,
        dictionaries: Union[dict, list],
        aggregation_methods: list,
    ) -> pd.DataFrame:
        """Score the cleaned text against each dictionary × aggregation method
        combination and return the concatenated results."""
        logger.info("Analyzing sentiment using lexical methods...")
        pipe_lexical = SentimentLexical(df_input=self.df_clean.dropna(subset=[self.text_column]))
        df_sent_lexical = []

        unstemmed_dicts = dictionaries.get("unstemmed", []) if isinstance(dictionaries, dict) else dictionaries
        stemmed_dicts = dictionaries.get("stemmed", []) if isinstance(dictionaries, dict) else []
        
        dict_text_map = []
        for d in unstemmed_dicts:
            dict_text_map.append((d, "text_tokens_str"))
        for d in stemmed_dicts:
            dict_text_map.append((d, "text_stems"))

        for sentiment, text in tqdm(dict_text_map, desc="Lexical Methods"):
            for aggregation_method in aggregation_methods:
                try:
                    df_sent = (
                        pipe_lexical
                        .sentiment_pipeline(
                            dictionary_name=sentiment,
                            text_column=text,
                            method=aggregation_method,
                        )
                        .set_index("id_text")
                        .filter(regex=f"{sentiment}|count|words")
                    )
                    if text == "text_stems":
                        df_sent = df_sent.rename(
                            lambda col_name: f"{col_name}_stem" if sentiment in col_name else col_name,
                            axis="columns",
                        )
                    df_sent_lexical.append(df_sent)
                    logger.info(f"Completed lexical analysis for {sentiment} with {aggregation_method}")
                except Exception as e:
                    raise SentimentAnalysisError(
                        f"Error in lexical analysis for {sentiment} with {aggregation_method}: {e}"
                    ) from e

        self.df_sent_lexical = pd.concat(df_sent_lexical, axis=1)
        logger.info("Lexical sentiment analysis complete.")
        return self.df_sent_lexical

    def analyze_sentiment_transformer(
        self,
        transformer_config: dict,
    ) -> pd.DataFrame:
        """Score cleaned text with one or more optional transformer models."""
        if self.df_clean is None:
            raise SentimentAnalysisError("Clean data before transformer sentiment analysis.")

        if not transformer_config.get("enabled", False):
            logger.info("Skipping transformer sentiment analysis: disabled in config.")
            return pd.DataFrame()

        try:
            from auto_econ_sentiment.models.sentiment_transformers import SentimentTransformers
        except ImportError as e:
            raise SentimentAnalysisError(str(e)) from e

        model_configs = transformer_config.get("models", [])
        if isinstance(model_configs, dict):
            model_configs = [model_configs]
        if not model_configs:
            raise SentimentAnalysisError("Transformer config is enabled but no models are configured.")

        default_text_column = transformer_config.get("text_column_transformer", "text_clean")
        df_sent_transformer = []
        df_sentence_probabilities = []

        for model_config in model_configs:
            model_short = model_config["model_name_short"]
            text_column = model_config.get("text_column_transformer", default_text_column)
            aggregation = model_config.get("aggregation", "byalltext")
            if text_column not in self.df_clean.columns:
                raise SentimentAnalysisError(f"Transformer text column '{text_column}' not found.")

            try:
                pipe_transformer = SentimentTransformers(
                    df_input=self.df_clean.dropna(subset=[text_column]),
                    text_column=text_column,
                    model_name=model_config["model_name"],
                    model_name_short=model_short,
                    label_map=model_config["label_map"],
                    num_labels=model_config.get("num_labels"),
                    max_length=model_config.get("max_length", 512),
                    batch_size=model_config.get("batch_size", 16),
                    huggingface_token=model_config.get("huggingface_token"),
                    device=model_config.get("device"),
                )
                result = pipe_transformer.sentiment_pipeline(
                    aggregation=aggregation,
                    sentence_probability_cutoff=model_config.get("sentence_probability_cutoff", 0.7),
                )
            except Exception as e:
                raise SentimentAnalysisError(f"Error in transformer analysis for {model_short}: {e}") from e

            if aggregation == "bysentence":
                df_agg, df_prob = result
                df_sent_transformer.append(df_agg)
                df_sentence_probabilities.append(df_prob)
            else:
                df_model = (
                    result
                    .set_index("id_text")
                    .filter(regex=f"^{model_short}_")
                )
                df_sent_transformer.append(df_model)

            logger.info("Completed transformer analysis for %s", model_short)

        self.df_sent_transformer = pd.concat(df_sent_transformer, axis=1)
        if df_sentence_probabilities:
            self.df_transformer_sentence_probabilities = pd.concat(df_sentence_probabilities, axis=1)
        logger.info("Transformer sentiment analysis complete.")
        return self.df_sent_transformer

    def run(
        self,
        clean_config: Optional[dict],
        dictionaries: Union[dict, list, None],
        aggregation_methods: Optional[list],
        export_results: bool,
        transformer_config: Optional[dict] = None,
    ) -> tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """Run the full pipeline (load → clean → score → optional export).

        Returns a ``(df_raw, df_clean, df_sent_lexical)`` tuple. Any stage that is
        skipped (e.g. no dictionaries supplied) yields ``None`` in its slot.
        """
        logger.info("Starting AutoEconSentiment pipeline...")
        self.load_data()
        self.clean_data(clean_config=clean_config)

        if dictionaries and aggregation_methods:
            self.analyze_sentiment_lexical(
                dictionaries=dictionaries,
                aggregation_methods=aggregation_methods,
            )
        else:
            logger.warning("Skipping lexical sentiment analysis: no dictionaries or aggregation methods provided.")

        if transformer_config and transformer_config.get("enabled", False):
            self.analyze_sentiment_transformer(transformer_config=transformer_config)

        if export_results:
            logger.info("Exporting results...")
            dataframes_to_concat = []
            if self.df_clean is not None:
                dataframes_to_concat.append(self.df_clean.set_index("id_text"))
            if self.df_sent_lexical is not None:
                self.df_sent_lexical.to_parquet(f"{self.export_path}/sentiment_lexical.parquet.gzip", compression="gzip", index=False)
                dataframes_to_concat.append(
                    self.df_sent_lexical.reset_index().set_index("id_text").filter(regex="sentiment")
                )
            if self.df_sent_transformer is not None:
                self.df_sent_transformer.reset_index().to_parquet(
                    f"{self.export_path}/sentiment_transformer.parquet.gzip",
                    compression="gzip",
                    index=False,
                )
                dataframes_to_concat.append(
                    self.df_sent_transformer.reset_index().set_index("id_text").filter(regex="sentiment|label|probability")
                )
            if self.df_transformer_sentence_probabilities is not None:
                self.df_transformer_sentence_probabilities.reset_index().to_parquet(
                    f"{self.export_path}/sentiment_transformer_sentence_probabilities.parquet.gzip",
                    compression="gzip",
                    index=False,
                )

            if dataframes_to_concat:
                df_sentiment_all = pd.concat(dataframes_to_concat, axis=1)
                df_sentiment_all.drop(
                    ["Unnamed: 0", "text", "id_text"], axis=1, errors="ignore"
                ).to_parquet(f"{self.export_path}/sentiment_all_results.parquet.gzip", compression="gzip", index=False)
                logger.info(f"ALL sentiment results exported to: {self.export_path}/sentiment_all_results.parquet.gzip")

        logger.info("AutoEconSentiment pipeline finished.")
        return self.df_raw, self.df_clean, self.df_sent_lexical


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

    parser = argparse.ArgumentParser(description="Run AutoEconSentiment pipeline")
    parser.add_argument("--test", action="store_true", help="Run with synthetic test data.")
    args = parser.parse_args()

    if args.test:
        logger.info("Running pipeline with dummy test data.")
        data = {
            "text": [
                "Our commitment continues. By altering rates, we aim to maintain balance. The resilience of our economy is bolstered.",
                "The central bank's deliberations are influenced by trade environment.",
            ],
            "date": ["2024-01-15", "2024-03-20"],
        }
        Path("tests/fixtures/synthetic_input").mkdir(parents=True, exist_ok=True)
        csv_path = "tests/fixtures/synthetic_input/test_onerow_onestatement.csv"
        pd.DataFrame(data).to_csv(csv_path, index=False)

        analyzer = AutoEconSentiment(
            import_file_path=csv_path,
            text_column="text",
            date_column="date",
            export_path="data/sentiment/basic_tests",
        )
        analyzer.run(
            clean_config={"clean_html": True, "clean_numbers_percentages": True, "remove_headers": [], "tokenize": True},
            dictionaries={"unstemmed": ["correa", "hubert", "lm", "hiv"], "stemmed": []},
            aggregation_methods=["posneg", "allwords"],
            export_results=True,
        )
        logger.info("Dummy data pipeline run completed.")
    else:
        logger.info("Running pipeline with configuration from params.yaml.")
        config = load_yaml_config(config_path="params.yaml")
        analyzer = AutoEconSentiment(
            import_file_path=config["input"]["file_path"],
            text_column=config["input"]["text_column"],
            date_column=config["input"]["date_column"],
            export_path=config["output"]["export_path"],
        )
        lexical_config = config.get("models", {}).get("lexical", {})
        analyzer.run(
            clean_config=config.get("cleaning", {}),
            dictionaries=lexical_config.get("dictionaries", {}),
            aggregation_methods=lexical_config.get("aggregation_methods", []),
            export_results=config["output"].get("export_results", True),
            transformer_config=config.get("models", {}).get("transformer", {}),
        )
        logger.info("Pipeline run with params.yaml configuration completed.")

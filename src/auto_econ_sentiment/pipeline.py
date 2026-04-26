import os
import logging
import argparse
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
    def __init__(self, import_file_path, text_column, date_column, export_path):
        self.import_file = import_file_path
        self.export_path = export_path
        self.text_column = text_column
        self.date_column = date_column
        os.makedirs(self.export_path, exist_ok=True)
        self.df_raw = None
        self.df_clean = None
        self.df_sent_lexical = None
        logger.info("AutoEconSentiment initialized successfully")

    def load_data(self):
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

    def clean_data(self, clean_config=None):
        logger.info("Cleaning data...")
        cleaner = TextCleaner(
            df=self.df_raw,
            text_column=self.text_column,
            export_path=self.export_path,
            clean_config=clean_config,
        )
        self.df_clean = cleaner.run()
        return self.df_clean

    def analyze_sentiment_lexical(self, dictionaries, aggregation_methods):
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

    def run(self, clean_config, dictionaries, aggregation_methods, export_results):
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

        if export_results:
            logger.info("Exporting results...")
            dataframes_to_concat = []
            if self.df_clean is not None:
                dataframes_to_concat.append(self.df_clean.set_index("id_text"))
            if self.df_sent_lexical is not None:
                self.df_sent_lexical.to_csv(f"{self.export_path}/sentiment_lexical.csv")
                dataframes_to_concat.append(
                    self.df_sent_lexical.reset_index().set_index("id_text").filter(regex="sentiment")
                )

            if dataframes_to_concat:
                df_sentiment_all = pd.concat(dataframes_to_concat, axis=1)
                df_sentiment_all.drop(
                    ["Unnamed: 0", "text", "id_text"], axis=1, errors="ignore"
                ).to_csv(f"{self.export_path}/sentiment_all_results.csv")
                logger.info(f"ALL sentiment results exported to: {self.export_path}/sentiment_all_results.csv")

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
        )
        logger.info("Pipeline run with params.yaml configuration completed.")
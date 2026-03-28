import logging
import re
import yaml
import os
from pathlib import Path

import pandas as pd
from auto_econ_sentiment.pipeline import AutoEconSentiment

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

CONFIG_PATH = Path("references/configs/params_cb_speeches.yaml")


class CBSpeechesSentiment:
    def __init__(self, config_path=CONFIG_PATH):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

    def process_all_banks(self):
        speeches_dir = Path("data/raw/speeches")
        if not speeches_dir.exists():
            logger.error(
                f"{speeches_dir} does not exist. Run download_cb_speeches.py first."
            )
            return

        all_files = list(speeches_dir.glob("*.parquet.gzip"))
        logger.info(f"Found {len(all_files)} central bank files to process.")

        text_col = self.config["input"]["text_column"]

        for file_path in all_files:
            cb_name = file_path.name.replace(".parquet.gzip", "")
            clean_cb_name = (
                re.sub(r"[<>:\"/\\|?*\s]+", "_", cb_name).strip("_.").lower()
            )
            logger.info(f"--- Processing {clean_cb_name} ---")

            df = pd.read_parquet(file_path)
            # Drop empty rows to prevent CountVectorizer crashes
            orig_len = len(df)
            df = df.dropna(subset=[text_col]).reset_index(drop=True)
            df = df[df[text_col].str.strip().astype(bool)].reset_index(drop=True)
            if orig_len > len(df):
                logger.info(f"Dropped {orig_len - len(df)} rows with missing text.")

            if len(df) == 0:
                logger.warning(f"No valid text rows for {cb_name}, skipping.")
                continue

            # Save cleaned dataset temporarily for the pipeline
            temp_path = Path(f"data/raw/temp_{clean_cb_name}.parquet.gzip")
            df.to_parquet(str(temp_path), compression="gzip", index=False)

            # Use unique export path for each bank
            base_export = Path(self.config["output"]["export_path"])
            export_path = base_export / clean_cb_name
            os.makedirs(export_path, exist_ok=True)

            analyzer = AutoEconSentiment(
                import_file_path=str(temp_path),
                text_column=self.config["input"]["text_column"],
                date_column=self.config["input"]["date_column"],
                export_path=str(export_path),
            )

            lexical_config = self.config.get("models", {}).get("lexical", {})
            try:
                analyzer.run(
                    clean_config=self.config.get("cleaning", {}),
                    dictionaries=lexical_config.get("dictionaries", {}),
                    aggregation_methods=lexical_config.get("aggregation_methods", []),
                    export_results=self.config["output"].get("export_results", True),
                )
            except Exception as e:
                logger.error(f"Failed to process {clean_cb_name}: {e}")

            # Cleanup temp file
            if temp_path.exists():
                temp_path.unlink()

    def run(self):
        self.process_all_banks()
        logger.info("CBS speeches sentiment pipeline for all banks complete.")


if __name__ == "__main__":
    pipeline = CBSpeechesSentiment()
    pipeline.run()

import os
import logging
import pandas as pd

logger = logging.getLogger(__name__)


class SentimentBase:
    def __init__(self, df_input, text_column):
        self.input_df = df_input
        self.text_column = text_column
        self.df_final = None

    def export_results(self, export_path, index=False):
        if self.df_final is None:
            raise RuntimeError("No results to export. Run analysis first.")
        os.makedirs(os.path.dirname(export_path), exist_ok=True)
        self.df_final.to_csv(export_path, index=index)
        logger.info(f"Results exported to {export_path}")
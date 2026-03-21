import logging
import urllib.request
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DROPBOX_URL = (
    "https://www.dropbox.com/scl/fi/n2tsmswnk4ymm3zz0wteg/"
    "CBS_dataset_v1.0.csv?rlkey=sdxk6b9xpid0xe7w7s4nm6s87&dl=1"
)

RAW_CSV_PATH = Path("data/raw/cb_speeches.csv")
RAW_PARQUET_PATH = Path("data/raw/cb_speeches.parquet.gzip")


class CBSpeechesDownloader:
    def __init__(self, url=DROPBOX_URL, csv_path=RAW_CSV_PATH, parquet_path=RAW_PARQUET_PATH):
        self.url = url
        self.csv_path = csv_path
        self.parquet_path = parquet_path

    def download_csv(self):
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        if self.csv_path.exists():
            logger.info(f"CSV already exists at {self.csv_path}, skipping download.")
            return
        logger.info(f"Downloading CBS dataset from Dropbox...")
        urllib.request.urlretrieve(self.url, str(self.csv_path))
        logger.info(f"Downloaded to {self.csv_path}")

    def convert_to_parquet(self):
        logger.info(f"Reading CSV from {self.csv_path}...")
        df = pd.read_csv(self.csv_path, low_memory=False)
        logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
        
        speeches_dir = Path("data/raw/speeches")
        speeches_dir.mkdir(parents=True, exist_ok=True)
        
        for cb, group in df.groupby('CentralBank'):
            clean_name = str(cb).replace('/', '_').replace('\\', '_').replace(' ', '_')
            out_file = speeches_dir / f"{clean_name}.parquet.gzip"
            group.to_parquet(str(out_file), compression="gzip", index=False)
            
        logger.info(f"Saved {df['CentralBank'].nunique()} individual central bank files to {speeches_dir}")

    def run(self):
        self.download_csv()
        self.convert_to_parquet()
        logger.info("CBS speeches download complete.")


if __name__ == "__main__":
    downloader = CBSpeechesDownloader()
    downloader.run()

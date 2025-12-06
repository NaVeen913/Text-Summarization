import os
import urllib.request
import zipfile
from pathlib import Path

from textsummarization.entity.config_entity import DataIngestionConfig
from textsummarization.logging import logger


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_data(self) -> None:
        """
        Download the dataset zip file from config.data_url
        to config.local_data_file.
        """
        if self.config.local_data_file.exists():
            logger.info(f"File already exists at {self.config.local_data_file}")
            return

        logger.info(f"Downloading data from {self.config.data_url}")
        urllib.request.urlretrieve(
            self.config.data_url, str(self.config.local_data_file)
        )
        logger.info(f"Downloaded to {self.config.local_data_file}")

    def extract_zip(self) -> None:
        """
        Extract the downloaded zip file into config.unzip_dir.
        """
        logger.info(f"Extracting {self.config.local_data_file} to {self.config.unzip_dir}")
        with zipfile.ZipFile(self.config.local_data_file, "r") as zip_ref:
            zip_ref.extractall(self.config.unzip_dir)
        logger.info("Extraction completed")

    def run(self) -> None:
        logger.info(">>> Running Data Ingestion...")
        self.download_data()
        self.extract_zip()
        logger.info("Data Ingestion completed.")




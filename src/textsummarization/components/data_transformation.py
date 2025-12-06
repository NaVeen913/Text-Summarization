from pathlib import Path

import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer

from textsummarization.entity.config_entity import DataTransformationConfig
from textsummarization.logging import logger


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)

    def _load_csv_as_dataset(self, path: Path) -> Dataset:
        logger.info(f"Loading CSV from {path}")
        df = pd.read_csv(path)
        # expects columns "article" and "highlights"
        return Dataset.from_pandas(df[["article", "highlights"]])

    def _preprocess_function(self, batch):
        inputs = batch["article"]
        targets = batch["highlights"]

        model_inputs = self.tokenizer(
            inputs,
            max_length=self.config.max_input_length,
            truncation=True,
        )

        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                targets,
                max_length=self.config.max_target_length,
                truncation=True,
            )

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def run(self) -> None:
        logger.info(">>> Running Data Transformation...")

        # 1. load raw CSV as datasets
        train_ds = self._load_csv_as_dataset(self.config.train_path)
        val_ds = self._load_csv_as_dataset(self.config.val_path)
        test_ds = self._load_csv_as_dataset(self.config.test_path)

        # 2. tokenize
        logger.info("Tokenizing train/val/test datasets...")
        train_tok = train_ds.map(
            self._preprocess_function,
            batched=True,
            remove_columns=train_ds.column_names,
        )
        val_tok = val_ds.map(
            self._preprocess_function,
            batched=True,
            remove_columns=val_ds.column_names,
        )
        test_tok = test_ds.map(
            self._preprocess_function,
            batched=True,
            remove_columns=test_ds.column_names,
        )

        # 3. save to disk
        logger.info("Saving tokenized datasets to disk...")
        train_tok.save_to_disk(self.config.transformed_train_path)
        val_tok.save_to_disk(self.config.transformed_val_path)
        test_tok.save_to_disk(self.config.transformed_test_path)

        logger.info("Data Transformation completed.")


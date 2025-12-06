from pathlib import Path

from textsummarization.utils.common import read_yaml, create_directories
from textsummarization.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH
from textsummarization.entity.config_entity import (
    DataIngestionConfig,
    DataTransformationConfig,
    ModelTrainerConfig,
)


class ConfigurationManager:
    def __init__(
        self,
        config_filepath: Path = CONFIG_FILE_PATH,
        params_filepath: Path = PARAMS_FILE_PATH,
    ):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([Path(self.config.artifacts_root)])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([Path(config.root_dir)])

        return DataIngestionConfig(
            root_dir=Path(config.root_dir),
            data_url=config.data_url,
            local_data_file=Path(config.local_data_file),
            unzip_dir=Path(config.unzip_dir),
        )

    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config.data_transformation

        create_directories(
            [
                Path(config.root_dir),
                Path(config.transformed_train_path),
                Path(config.transformed_val_path),
                Path(config.transformed_test_path),
            ]
        )

        return DataTransformationConfig(
            root_dir=Path(config.root_dir),
            train_path=Path(config.train_path),
            val_path=Path(config.val_path),
            test_path=Path(config.test_path),
            transformed_train_path=Path(config.transformed_train_path),
            transformed_val_path=Path(config.transformed_val_path),
            transformed_test_path=Path(config.transformed_test_path),
            tokenizer_name=config.tokenizer_name,
            max_input_length=int(config.max_input_length),
            max_target_length=int(config.max_target_length),
        )

    def get_model_trainer_config(self) -> ModelTrainerConfig:
        config = self.config.model_trainer
        params = self.params.model_trainer

        create_directories(
            [
                Path(config.root_dir),
                Path(config.output_dir),
            ]
        )

        return ModelTrainerConfig(
            root_dir=Path(config.root_dir),
            model_ckpt=config.model_ckpt,
            train_dataset_path=Path(config.train_dataset_path),
            val_dataset_path=Path(config.val_dataset_path),
            num_train_epochs=int(params.num_train_epochs),
            per_device_train_batch_size=int(params.per_device_train_batch_size),
            per_device_eval_batch_size=int(params.per_device_eval_batch_size),
            learning_rate=float(params.learning_rate),
            weight_decay=float(params.weight_decay),
            logging_steps=int(params.logging_steps),
            predict_with_generate=bool(params.predict_with_generate),
            output_dir=Path(config.output_dir),
        )


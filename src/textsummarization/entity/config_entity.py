from dataclasses import dataclass
from pathlib import Path


@dataclass
class DataIngestionConfig:
    root_dir: Path
    data_url: str
    local_data_file: Path
    unzip_dir: Path


@dataclass
class DataTransformationConfig:
    root_dir: Path
    train_path: Path
    val_path: Path
    test_path: Path
    transformed_train_path: Path
    transformed_val_path: Path
    transformed_test_path: Path
    tokenizer_name: str
    max_input_length: int
    max_target_length: int


@dataclass
class ModelTrainerConfig:
    root_dir: Path
    model_ckpt: str
    train_dataset_path: Path
    val_dataset_path: Path
    num_train_epochs: int
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    learning_rate: float
    weight_decay: float
    logging_steps: int
    predict_with_generate: bool
    output_dir: Path


from dataclasses import dataclass
from pathlib import Path

from src.handwritten_digit_classifier.utils.transformer import Transformer


@dataclass
class DataIngestionConfig:
    # these are the inputs to the data ingestion pipeline
    data_root_dir: Path
    transformer: Transformer

@dataclass
class DataValidationConfig:
    # these are the inputs to the data validation pipeline
    data_root_dir: Path
    data_mnist_dir: Path
    STATUS_FILE: str
    mnist_file_count: int

@dataclass
class DataTransformationConfig:
    # these are the inputs to the data transformation pipeline
    data_root_dir: Path
    data_mnist_dir: Path
    data_train_size: float
    data_val_size: float
    data_random_state: int
    data_batch_size: int
    data_shuffle: bool
    transformer: Transformer
    data_preprocessed_train_file: str
    data_preprocessed_val_file: str
    data_preprocessed_test_file: str

@dataclass
class ModelTrainerConfig:
    # these are the inputs to the model training pipeline
    data_root_dir: Path
    data_train_file: Path
    data_val_file: Path
    adam_learning_rate: float
    model_file: Path

@dataclass
class ModelEvaluationConfig:
    # these are the inputs to the model evaluation pipeline
    data_root_dir: Path
    data_test_file: Path
    model_file: Path
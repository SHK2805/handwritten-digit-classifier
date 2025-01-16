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
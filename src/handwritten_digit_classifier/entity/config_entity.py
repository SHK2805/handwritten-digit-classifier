from dataclasses import dataclass
from pathlib import Path

from src.handwritten_digit_classifier.utils.transformer import Transformer


@dataclass
class DataIngestionConfig:
    # these are the inputs to the data ingestion pipeline
    data_root_dir: Path
    mnist_params: dict
    transformer: Transformer
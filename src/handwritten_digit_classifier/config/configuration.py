import os
from pathlib import Path

from src.handwritten_digit_classifier.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH
from src.handwritten_digit_classifier.entity.config_entity import DataIngestionConfig, DataValidationConfig
from src.handwritten_digit_classifier.logger.logger_config import logger
from src.handwritten_digit_classifier.utils.common import read_yaml, create_directories
from src.handwritten_digit_classifier.utils.transformer import Transformer


class ConfigurationManager:
    def __init__(self, config_file_path: Path = CONFIG_FILE_PATH, params_file_path: Path = PARAMS_FILE_PATH):
        self.class_name = self.__class__.__name__
        self.config_file_path: Path = config_file_path
        self.config = read_yaml(config_file_path)
        self.params = read_yaml(params_file_path)
        # create the artifacts directory
        self.artifacts_dir = self.config.artifacts_root
        logger.info(f"Artifacts directory: {self.artifacts_dir}")
        create_directories([os.path.join(self.artifacts_dir)])

    def get_mnist_params(self):
        return self.params.MNIST

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        tag: str = f"{self.class_name}::get_data_ingestion_config::"
        config = self.config.data_ingestion
        logger.info(f"{tag}Data ingestion configuration obtained from the config file")

        # create the data directory
        data_dir = config.data_root_dir
        logger.info(f"{tag}Data directory: {data_dir} obtained from the config file")

        create_directories([data_dir])
        logger.info(f"{tag}Data directory created: {data_dir}")

        transformer = Transformer().get_transform()

        data_ingestion_config: DataIngestionConfig = DataIngestionConfig(
            data_root_dir = Path(config.data_root_dir),
            transformer = transformer
        )
        logger.info(f"{tag}Data ingestion configuration created")
        return data_ingestion_config

    def get_data_validation_config(self) -> DataValidationConfig:
        tag: str = f"{self.class_name}::get_data_validation_config::"
        config = self.config.data_validation
        logger.info(f"{tag}Data validation configuration obtained from the config file")

        # create the data directory
        data_dir = config.data_root_dir
        logger.info(f"{tag}Data directory: {data_dir} obtained from the config file")

        create_directories([data_dir])
        logger.info(f"{tag}Data directory created: {data_dir}")

        data_validation_config: DataValidationConfig = DataValidationConfig(
            data_root_dir=Path(config.data_root_dir),
            data_mnist_dir=Path(config.data_mnist_dir),
            STATUS_FILE=config.STATUS_FILE,
            mnist_file_count=8
        )
        logger.info(f"{tag}Data validation configuration created")
        return data_validation_config


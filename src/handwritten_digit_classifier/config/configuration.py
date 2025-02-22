import os
from pathlib import Path

from src.handwritten_digit_classifier.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH
from src.handwritten_digit_classifier.entity.config_entity import DataIngestionConfig, DataValidationConfig, \
    DataTransformationConfig, ModelTrainerConfig, ModelEvaluationConfig
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

    def get_adam_params(self):
        tag: str = f"{self.class_name}::get_adam_params::"
        return self.params.Adam

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

    def get_data_transformation_config(self) -> DataTransformationConfig:
        tag: str = f"{self.class_name}::get_data_transformation_config::"
        config = self.config.data_transformation
        logger.info(f"{tag}Data transformation configuration obtained from the config file")

        # create the data directory
        data_dir = config.data_root_dir
        logger.info(f"{tag}Data directory: {data_dir} obtained from the config file")
        create_directories([data_dir])
        logger.info(f"{tag}Data directory created: {data_dir}")

        transformer = Transformer().get_transform()

        data_transformation_config: DataTransformationConfig = DataTransformationConfig(
            data_root_dir=Path(config.data_root_dir),
            data_mnist_dir=Path(config.data_mnist_dir),
            data_train_size=config.data_train_size,
            data_val_size=config.data_val_size,
            data_random_state=config.data_random_state,
            data_batch_size=config.data_batch_size,
            data_shuffle=config.data_shuffle,
            transformer=transformer,
            data_preprocessed_train_file=config.data_preprocessed_train_file,
            data_preprocessed_val_file=config.data_preprocessed_val_file,
            data_preprocessed_test_file=config.data_preprocessed_test_file
        )

        return data_transformation_config

    def get_model_trainer_config(self) -> ModelTrainerConfig:
        tag: str = f"{self.class_name}::get_model_training_config::"
        config = self.config.model_trainer
        logger.info(f"{tag}Model training configuration obtained from the config file")

        # create the data directory
        data_dir = config.data_root_dir
        logger.info(f"{tag}Data directory: {data_dir} obtained from the config file")
        create_directories([data_dir])
        logger.info(f"{tag}Data directory created: {data_dir}")

        params = self.get_adam_params()
        logger.info(f"{tag}Model parameters obtained for Adam optimizer from the params file")

        model_trainer_config: ModelTrainerConfig = ModelTrainerConfig(
            data_root_dir=Path(config.data_root_dir),
            data_train_file=Path(config.data_train_file),
            data_val_file=Path(config.data_val_file),
            adam_learning_rate=params.learning_rate,
            model_file=Path(config.model_file)
        )

        return model_trainer_config

    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        tag: str = f"{self.class_name}::get_model_evaluation_config::"
        config = self.config.model_evaluation
        logger.info(f"{tag}Model evaluation configuration obtained from the config file")

        # create the data directory
        data_dir = config.data_root_dir
        logger.info(f"{tag}Data directory: {data_dir} obtained from the config file")
        # create_directories([data_dir])
        # logger.info(f"{tag}Data directory created: {data_dir}")

        model_evaluation_config: ModelEvaluationConfig = ModelEvaluationConfig(
            data_root_dir=Path(config.data_root_dir),
            data_test_file=Path(config.data_test_file),
            model_file=Path(config.model_file)
        )

        return model_evaluation_config



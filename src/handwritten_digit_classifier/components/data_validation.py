import os.path
from pathlib import Path

from src.handwritten_digit_classifier.entity.config_entity import DataValidationConfig
from src.handwritten_digit_classifier.logger.logger_config import logger
from src.handwritten_digit_classifier.utils.common import write_data_to_file


class DataValidation:
    def __init__(self, config: DataValidationConfig):
        self.class_name = self.__class__.__name__
        self.config = config

    def validate(self):
        tag: str = f"{self.class_name}::validate::"
        validation_status = False
        try:
            logger.info(f"{tag}Validating the data")
            # check if the data directory exists
            if not os.path.exists(self.config.data_root_dir):
                logger.error(f"{tag}Data directory does not exist: {self.config.data_root_dir}")
                return validation_status
            logger.info(f"{tag}Data directory exists: {self.config.data_root_dir}")

            # check if the MNIST data directory exists
            data_mnist_dir = self.config.data_mnist_dir
            if not os.path.exists(data_mnist_dir):
                logger.error(f"{tag}MNIST data directory does not exist: {data_mnist_dir}")
                return validation_status
            logger.info(f"{tag}MNIST data directory exists: {data_mnist_dir}")

            # check if 8 files exists in the MNIST data directory
            files = os.listdir(data_mnist_dir)
            if len(files) != self.config.mnist_file_count:
                logger.error(f"{tag}Number of files in the MNIST data directory is not {self.config.mnist_file_count}")
                return validation_status
            logger.info(f"{tag}Number of files in the MNIST data directory is {self.config.mnist_file_count}")
            validation_status = True

            logger.info(f"{tag}Writing the validation status {validation_status} to: {self.config.STATUS_FILE}")
            write_data_to_file(Path(os.path.join(self.config.STATUS_FILE)), str(validation_status))
            return validation_status
        except Exception as e:
            logger.error(f"{tag}Error reading the data: {e}")
            logger.info(f"{tag}Writing the validation status {validation_status} to: {self.config.STATUS_FILE}")
            write_data_to_file(Path(os.path.join(self.config.STATUS_FILE)), str(validation_status))
            raise e


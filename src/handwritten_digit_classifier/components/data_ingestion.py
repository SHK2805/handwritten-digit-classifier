from torchvision import datasets

from src.handwritten_digit_classifier.entity.config_entity import DataIngestionConfig
from src.handwritten_digit_classifier.logger.logger_config import logger


class DataIngestion:
    def __init__(self,config: DataIngestionConfig):
        self.class_name = self.__class__.__name__
        self.config: DataIngestionConfig = config

    def download_data(self):
        tag: str = f"{self.class_name}::download_data::"
        # download MNIST data
        train_data_set = datasets.MNIST(root=self.config.data_root_dir, train=True, transform=None, download=True)
        test_data_set = datasets.MNIST(root=self.config.data_root_dir, train=False, transform=None, download=True)
        logger.info(f"{tag}Data downloaded into the directory: {self.config.data_root_dir}. New data loaded")

        return train_data_set, test_data_set




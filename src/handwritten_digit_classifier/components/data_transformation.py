import os

import torch
from torch.utils.data import random_split, DataLoader
from torchvision import datasets

from src.handwritten_digit_classifier.entity.config_entity import DataTransformationConfig
from src.handwritten_digit_classifier.logger.logger_config import logger


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.class_name = self.__class__.__name__
        self.config = config

    def transform(self):
        tag: str = f"{self.class_name}::transform::"
        try:
            # Check if the data is present
            if not os.path.exists(self.config.data_mnist_dir):
                logger.error(f"{tag}Data file {self.config.data_mnist_dir} does not exist")
                raise FileNotFoundError(f"{tag}Data file {self.config.data_mnist_dir} does not exist")
            # read the MNIST data
            logger.info(f"{tag}Reading the MNIST data from {self.config.data_mnist_dir}")
            train_data_set = datasets.MNIST(root=self.config.data_mnist_dir, train=True, transform=self.config.transformer, download=False)
            test_data_set = datasets.MNIST(root=self.config.data_mnist_dir, train=False, transform=self.config.transformer, download=False)

            # split the data into train and validation
            train_size = int(self.config.data_train_size * len(train_data_set))
            val_size = len(train_data_set) - train_size
            train_data, val_data = random_split(train_data_set, [train_size, val_size])

            # create the data loaders
            train_loader = DataLoader(train_data, batch_size=self.config.data_batch_size,shuffle=self.config.data_shuffle)
            val_loader = DataLoader(val_data, batch_size=self.config.data_batch_size, shuffle=False)
            test_loader = DataLoader(test_data_set, batch_size=self.config.data_batch_size, shuffle=False)

            # save the data loaders into files
            logger.info(f"{tag}Saving the data loaders into files")

            torch.save(train_loader, self.config.data_preprocessed_train_file)
            logger.info(f"{tag}Saved the train data into {self.config.data_preprocessed_train_file}")

            torch.save(val_loader, self.config.data_preprocessed_val_file)
            logger.info(f"{tag}Saved the validation data into {self.config.data_preprocessed_val_file}")

            torch.save(test_loader, self.config.data_preprocessed_test_file)
            logger.info(f"{tag}Saved the test data into {self.config.data_preprocessed_test_file}")

            logger.info(f"{tag}Data transformation completed")
            return train_loader, val_loader, test_loader
        except Exception as e:
            logger.error(f"{tag}Error transforming the data: {e}")
            raise e
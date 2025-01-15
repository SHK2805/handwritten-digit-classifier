import os
import shutil

from torchvision import datasets
from src.transformation.transformer import Transformer

class DataIngestion:
    def __init__(self,data_dir, transformer: Transformer=None):
        # load MNIST data
        self.transformer = transformer
        if self.transformer is None:
            self.transformer = Transformer()

        # the data id downloaded into this directory
        self.data_dir = data_dir
        self.clean()
        self.train_data = datasets.MNIST(root=data_dir, train=True, transform=transformer.get_transform(), download=True)
        self.test_data = datasets.MNIST(root=data_dir, train=False, transform=transformer.get_transform(), download=True)
        self.classes = self.train_data.classes

    def get_train_data(self):
        return self.train_data

    def get_test_data(self):
        return self.test_data

    def get_classes(self):
        return self.classes

    def get_transformer(self):
        return self.transformer

    def get_data_dir(self):
        return self.data_dir

    def clean(self):
        # remove downloaded data
        if os.path.exists(self.data_dir):
            shutil.rmtree(self.data_dir)

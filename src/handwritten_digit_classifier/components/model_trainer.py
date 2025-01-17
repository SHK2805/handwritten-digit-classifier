import torch
from torch import nn

from src.handwritten_digit_classifier.entity.config_entity import ModelTrainerConfig
from src.handwritten_digit_classifier.logger.logger_config import logger
from src.handwritten_digit_classifier.utils.digit_classifier import DigitClassifier


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.class_name = self.__class__.__name__
        self.config = config

    def train(self):
        tag: str = f"{self.class_name}::train_model::"
        try:
            # check if the data exists
            if not self.config.data_train_file.exists():
                logger.error(f"{tag}Training data file does not exist: {self.config.data_train_file}")
                raise FileNotFoundError(f"Training data file does not exist: {self.config.data_train_file}")

            if not self.config.data_val_file.exists():
                logger.error(f"{tag}Validation data file does not exist: {self.config.data_val_file}")
                raise FileNotFoundError(f"Validation data file does not exist: {self.config.data_val_file}")

            if not self.config.data_test_file.exists():
                logger.error(f"{tag}Test data file does not exist: {self.config.data_test_file}")
                raise FileNotFoundError(f"Test data file does not exist: {self.config.data_test_file}")

            # load the data
            logger.info(f"{tag}Loading the data from files")
            train_data = torch.load(self.config.data_train_file)
            val_data = torch.load(self.config.data_val_file)
            test_data = torch.load(self.config.data_test_file)
            logger.info(f"{tag}Data loaded from files: {self.config.data_train_file}, {self.config.data_val_file}, {self.config.data_test_file}")

            learning_rate = self.config.adam_learning_rate
            logger.info(f"{tag}Learning rate: {learning_rate}")
            # get the model
            # check if GPU is available
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"{tag}Device: {device}")
            model = DigitClassifier().to(device)
            # we have more than one output class so we are using CrossEntropyLoss
            # this will apply softmax to the output layer automatically
            # the output will be a probability distribution over the classes
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            number_of_epochs = 1

            # train
            for epoch in range(number_of_epochs):
                train_loss, train_accuracy = self.train_model(model, train_data, criterion, optimizer, device)
                logger.info(f"{tag}Epoch: {epoch + 1}, Train Loss: {train_loss:.4f}")
                logger.info(f"{tag}Train Accuracy: {train_accuracy:.4f}")

            # save the model
            torch.save(model, self.config.model_file)
            logger.info(f"{tag}Model saved to: {self.config.data_root_dir}")

            logger.info(f"{tag}Training the model using Adam optimizer")
        except Exception as e:
            logger.error(f"{tag}Exception occurred during model training: {str(e)}")
            raise e

    def train_model(self, model, loader, criterion, optimizer, device):
        tag: str = f"{self.class_name}::train::"
        try:
            # the model is in training mode
            # the weights and biases are updated
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                # statistics
                # calculate the loss for each image in the batch
                running_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)

                correct += (predicted.eq(labels).sum().item())
            epoch_loss = running_loss / total
            accuracy = correct / total
            return epoch_loss, accuracy
        except Exception as e:
            logger.error(f"{tag}Exception occurred during model training: {str(e)}")
            raise e
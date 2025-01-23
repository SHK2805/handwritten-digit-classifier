import torch
from torch import nn

from src.handwritten_digit_classifier.entity.config_entity import ModelEvaluationConfig
from src.handwritten_digit_classifier.logger.logger_config import logger
from src.handwritten_digit_classifier.utils.common import get_model, get_device
from src.handwritten_digit_classifier.utils.digit_classifier import DigitClassifier


class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.class_name = self.__class__.__name__
        self.config = config

    def evaluate(self):
        tag: str = f"{self.class_name}::evaluate::"
        try:
            # check if the data exists
            if not self.config.data_test_file.exists():
                message: str = f"{tag}Testing data file does not exist: {self.config.data_test_file}"
                logger.error(message)
                raise FileNotFoundError(message)

            # check if the model exists
            if not self.config.model_file.exists():
                message: str = f"{tag}Model file does not exist: {self.config.model_file}"
                logger.error(message)
                raise FileNotFoundError(message)

            # load the data and model
            # check if GPU is available
            device = get_device()
            logger.info(f"{tag}Device: {device}")
            model = get_model(device)
            model.load_state_dict(torch.load(self.config.model_file, weights_only=True))
            logger.info(f"{tag}Loading the data from files")
            test_data = torch.load(self.config.data_test_file)
            logger.info(f"{tag}Data loaded from files: {self.config.data_test_file}, {self.config.model_file}")


            logger.info(f"{tag}Device: {device}")

            # evaluate the model on the test data
            criterion = nn.CrossEntropyLoss()
            test_loss, test_accuracy = self.evaluate_model(model, test_data, criterion, device)
            logger.info(f"{tag}Test Loss: {test_loss:.4f}")
            logger.info(f"{tag}Test Accuracy: {test_accuracy:.4f}")


        except Exception as e:
            logger.info(f"{tag}Error evaluating metrics: {e}")
            raise e

    def evaluate_model(self, model, loader, criterion, device):
        tag: str = f"{self.class_name}::evaluate_model::"
        try:
            # the model is set to evaluation mode
            # the weights and biases are not updated
            model.eval()
            running_loss = 0.0
            correct = 0
            total = 0
            # we do not need to calculate the gradients
            # no backpropagation
            with torch.no_grad():
                for images, labels in loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    running_loss += loss.item() # we can multiply this by a factor like 100 if you want a percentage or larger number by a factor
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += (predicted.eq(labels).sum().item())
            epoch_loss = running_loss / total
            accuracy = correct / total
            return epoch_loss, accuracy
        except Exception as e:
            logger.error(f"{tag}Exception occurred during model evaluation: {str(e)}")
            raise e
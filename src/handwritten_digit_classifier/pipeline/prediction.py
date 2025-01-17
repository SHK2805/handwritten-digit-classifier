from pathlib import Path

import cv2
import torch

from src.handwritten_digit_classifier.logger.logger_config import logger
from src.handwritten_digit_classifier.utils.common import preprocess_image
from src.handwritten_digit_classifier.utils.digit_classifier import DigitClassifier


class PredictionPipeline:
    def __init__(self):
        self.class_name: str = self.__class__.__name__
        # 'artifacts/model_trainer/model.pth'
        self.model_dir = Path("artifacts/model_trainer")
        self.model_file_path = f"{self.model_dir}/model.pth"

    def get_model_file_path(self):
        return self.model_file_path

    def load_model(self):
        """
        Load the model from the model file
        :return: Loaded model
        """
        tag: str = f"{self.class_name}::load_model::"
        # Load the model from the file
        # check if GPU is available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = DigitClassifier().to(device)
        model.load_state_dict(torch.load(self.model_file_path))
        model.eval()
        logger.info(f"{tag}Model loaded from file: {self.model_file_path} for device: {device}")
        return model



    def predict(self,image, device):
        """
        Predict the class label for the input data
        :param image:  the image needed for prediction
        :param device: the device used for prediction
        :return: Predicted class label
        """

        tag: str = f"{self.class_name}::predict::"

        # Load the model
        model = self.load_model()
        model.eval()
        with torch.no_grad():
            image = image.to(device)
            output = model(image)
            # get the predicted label with maximum probability
            _, predicted = torch.max(output, 1)
            logger.info(f"{tag}Predicted label: {predicted.item()}")
            return predicted.item()


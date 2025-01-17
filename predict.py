from pathlib import Path

import cv2
import torch

from src.handwritten_digit_classifier.logger.logger_config import logger
from src.handwritten_digit_classifier.pipeline.prediction import PredictionPipeline
from src.handwritten_digit_classifier.utils.common import preprocess_image, check_file_exists


def predict(image_path: str):
    pipeline = PredictionPipeline()
    input_image = cv2.imread(image_path)
    preprocessed_image = preprocess_image(input_image)
    input_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predicted_label = pipeline.predict(preprocessed_image, input_device)
    logger.info(f"Predicted label: {predicted_label}")

if __name__ == "__main__":
    image_file_path: Path = Path("test_images/image.png")
    if not image_file_path:
        raise ValueError("Please provide the path to the image file")

    if not check_file_exists(image_file_path):
        raise FileNotFoundError(f"File not found: {image_file_path}")

    predict(str(image_file_path))
import os
from pathlib import Path
import cv2

import yaml
from box import ConfigBox
from box.exceptions import BoxValueError
from ensure import ensure_annotations
from torchvision import transforms

from src.handwritten_digit_classifier.logger.logger_config import logger


@ensure_annotations
def read_yaml(file_path: Path) -> ConfigBox:
    """
    Read the YAML file and return the ConfigBox object
    ConfigBox:
    ConfigBox is a powerful tool that simplifies configuration management in Python projects.
    By centralizing, organizing, and providing easy access to your configuration settings,
    ConfigBox helps you build more maintainable and flexible applications


    :param file_path: Path to the YAML file
    :return: ConfigBox object
    """
    try:
        with open(file_path, "r") as file:
            config = yaml.safe_load(file)
            logger.info(f"The YAML file: {file_path} has been read successfully")
            return ConfigBox(config)
    except BoxValueError as e:
        logger.error(f"Error reading the YAML file: {e}")
        raise ValueError(f"Error reading the YAML file: {e}")
    except Exception as e:
        logger.error(f"Error reading the YAML file: {e}")
        raise e

def create_directories(dirs: list, verbose=True) -> None:
    """
    Create directories if they do not exist

    :param verbose:
    :param dirs: List of directories to create
    :return: None
    """
    logger.info(f"Creating directories: {dirs}")
    for my_dir in dirs:
        if not os.path.exists(my_dir):
            os.makedirs(my_dir, exist_ok=True)
            if verbose:
                logger.info(f"Directory: {my_dir} has been created")
        else:
            if verbose:
                logger.info(f"Directory: {my_dir} already exists")
    logger.info(f"Directories have been created")

# function to check if the file exists
def check_file_exists(file_path: Path) -> bool:
    """
    Check if the file exists

    :param file_path: Path to the file
    :return: True if the file exists, False otherwise
    """
    return file_path.exists()

# function to write the data to the file
def write_data_to_file(file_path: Path, data: str) -> None:
    """
    Write the data to the file

    :param file_path: Path to the file
    :param data: Data to write
    :return: None
    """
    with open(file_path, "w") as file:
        file.write(data)

    logger.info(f"Data has been written to the file: {file_path}")

def preprocess_image(image):
    """
    Preprocess the image

    :param image: Image to preprocess
    :return: Preprocessed image
    """
    # we need the image to have black background and white digits
    # if the input image is of white background and black digits use the following code
    # image = cv2.bitwise_not(image)
    # convert the image to grayscale
    transform = transforms.Compose([
        transforms.ToPILImage(), # convert to PIL image
        transforms.Grayscale(num_output_channels=1), # convert to grayscale
        transforms.Resize((28, 28)), # resize the image to 28x28 to match the MNIST dataset
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)) # Normalize the image data to [-1 , 1] i.e. (1, 1, 28, 28
    ])
    preprocessed_image = transform(image)

    return preprocessed_image

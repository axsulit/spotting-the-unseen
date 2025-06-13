"""
This module provides utility functions for directory creation, image resizing,
and loading a facial landmark predictor model for the project.
"""

import os
import cv2
from pathlib import Path
import dlib

def create_directory(path):
    """
    Create a directory if it does not already exist.

    Args:
        path (str or Path): The directory path to create.
    """
    os.makedirs(path, exist_ok=True)


def resize(image, target_size=(256, 256)):
    """
    Resize an image to the target size using linear interpolation.

    Args:
        image (numpy.ndarray): The image to resize.
        target_size (tuple, optional): Target size as (width, height). Defaults to (256, 256).

    Returns:
        numpy.ndarray: The resized image.
    """
    target_w, target_h = target_size
    resized_image = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    return resized_image


DEFAULT_LANDMARK_MODEL = "preprocess/utils/shape_predictor_68_face_landmarks.dat"


def load_predictor(model_path=DEFAULT_LANDMARK_MODEL):
    """
    Load a dlib shape predictor model from the specified path.

    Args:
        model_path (str or Path): Path to the .dat model file.

    Returns:
        dlib.shape_predictor: Loaded shape predictor model.

    Raises:
        FileNotFoundError: If the model file does not exist at the given path.
    """
    if not Path(model_path).exists():
        raise FileNotFoundError(f"❌ Landmark model not found at: {model_path}")
    return dlib.shape_predictor(model_path)

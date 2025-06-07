# This file contains utility functions for the project.
import os
import cv2
from pathlib import Path
import dlib

# Function to create a directory if it does not exist
def create_directory(path):
    """
    Creates a directory if it does not exist.
    """
    os.makedirs(path, exist_ok=True)

# Function to resize image while maintaining aspect ratio and adding padding
def resize(image, target_size=(256, 256)):
    target_w, target_h = target_size
    resized_image = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    return resized_image

DEFAULT_LANDMARK_MODEL = "preprocess/utils/shape_predictor_68_face_landmarks.dat"

def load_predictor(model_path=DEFAULT_LANDMARK_MODEL):
    if not Path(model_path).exists():
        raise FileNotFoundError(f"❌ Landmark model not found at: {model_path}")
    return dlib.shape_predictor(model_path)
"""
This module filters out non-frontal face images based on estimated yaw angle
using dlib's facial landmark predictor and frontal face detector.
"""

import cv2
import dlib
import numpy as np
from pathlib import Path
import shutil


def load_predictor(model_path):
    """
    Load the dlib shape predictor model.

    Args:
        model_path (str or Path): Path to the dlib .dat model file.

    Returns:
        dlib.shape_predictor: Loaded shape predictor model.
    """
    return dlib.shape_predictor(model_path)


def get_yaw_angle(landmarks):
    """
    Estimate the yaw angle of a face based on facial landmarks.

    Args:
        landmarks (dlib.full_object_detection): Facial landmarks.

    Returns:
        float: Estimated yaw angle in degrees.
    """
    left_eye = landmarks.part(36).x
    right_eye = landmarks.part(45).x
    nose_tip = landmarks.part(30).x
    yaw_angle = np.arctan2(nose_tip - (left_eye + right_eye) / 2, right_eye - left_eye) * 180 / np.pi
    return yaw_angle


def filter_non_frontal_faces(folder, model_path, yaw_thresh=10):
    """
    Move images with non-frontal faces (based on yaw angle) to a separate folder.

    Args:
        folder (str or Path): Path to the folder containing face images.
        model_path (str or Path): Path to the dlib shape predictor model (.dat file).
        yaw_thresh (float, optional): Threshold for yaw angle to consider a face frontal. Defaults to 10.

    Returns:
        None
    """
    predictor = load_predictor(model_path)
    detector = dlib.get_frontal_face_detector()
    dump_dir = Path(folder).parent / "non_frontal"
    dump_dir.mkdir(parents=True, exist_ok=True)

    for img_path in Path(folder).glob("*"):
        if img_path.suffix.lower() not in [".jpg", ".png", ".jpeg"]:
            continue

        image = cv2.imread(str(img_path))
        if image is None:
            continue

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        for face in faces:
            landmarks = predictor(gray, face)
            yaw = get_yaw_angle(landmarks)
            if abs(yaw) > yaw_thresh:
                print(f"🛑 Moving {img_path.name} (Yaw: {yaw:.2f})")
                shutil.move(str(img_path), dump_dir / img_path.name)
                break

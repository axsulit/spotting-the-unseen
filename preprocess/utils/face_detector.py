"""
This module contains the core logic for detecting faces and determining 
if they are frontal using dlib and OpenCV.
"""

import cv2
import dlib
import numpy as np

# Load dlib's face detector
detector = dlib.get_frontal_face_detector()

# 3D model points for head pose estimation
MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0),           # Nose tip
    (0.0, -330.0, -65.0),      # Chin
    (-225.0, 170.0, -135.0),   # Left eye left corner
    (225.0, 170.0, -135.0),    # Right eye right corner
    (-150.0, -150.0, -125.0),  # Left mouth corner
    (150.0, -150.0, -125.0)    # Right mouth corner
], dtype=np.float32)

# Assumed camera matrix for pose estimation
CAMERA_MATRIX = np.array([
    [1.0, 0, 0],
    [0, 1.0, 0],
    [0, 0, 1]
], dtype=np.float32)

# No distortion coefficients assumed
DIST_COEFFS = np.zeros((4, 1))


def is_frontal_face(landmarks, yaw_threshold, pitch_threshold, roll_threshold):
    """
    Determine if the face is frontal based on head pose estimation.

    Args:
        landmarks (list or array): Array of facial landmarks (68 points).
        yaw_threshold (float): Maximum allowed yaw angle in degrees.
        pitch_threshold (float): Maximum allowed pitch angle in degrees.
        roll_threshold (float): Maximum allowed roll angle in degrees.

    Returns:
        bool: True if face is within the threshold limits (frontal), False otherwise.
    """
    image_points = np.array([
        landmarks[30],  # Nose tip
        landmarks[8],   # Chin
        landmarks[36],  # Left eye left corner
        landmarks[45],  # Right eye right corner
        landmarks[48],  # Left mouth corner
        landmarks[54],  # Right mouth corner
    ], dtype=np.float32)

    success, rotation_vector, _ = cv2.solvePnP(
        MODEL_POINTS, image_points, CAMERA_MATRIX, DIST_COEFFS, flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not success:
        return False

    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    yaw = np.degrees(np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0]))
    pitch = np.degrees(np.arcsin(-rotation_matrix[2, 0]))
    roll = np.degrees(np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2]))

    return abs(yaw) < yaw_threshold and abs(pitch) < pitch_threshold and abs(roll) < roll_threshold


def detect_faces(frame, scale_factor=2.0):
    """
    Detect faces in the input frame using dlib's frontal face detector.

    Args:
        frame (numpy.ndarray): Input image in BGR format.
        scale_factor (float, optional): Scale factor for resizing image to improve detection. Defaults to 2.0.

    Returns:
        list: List of dlib.rectangle objects representing detected face bounding boxes.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized_gray = cv2.resize(gray, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)
    faces = detector(resized_gray, 0)

    scaled_faces = [
        dlib.rectangle(
            int(face.left() / scale_factor),
            int(face.top() / scale_factor),
            int(face.right() / scale_factor),
            int(face.bottom() / scale_factor)
        ) for face in faces
    ]
    return scaled_faces

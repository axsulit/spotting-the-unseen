# This file contains the core logic for detecting faces and determining if they are frontal

import cv2
import dlib
import numpy as np
import yaml

# Load configuration
with open("config.yaml", "r") as ymlfile:
    config = yaml.safe_load(ymlfile)

# Load dlib's face detector and shape predictor (edit predictor path in config.yaml)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(config["landmark_model_path"])

# Define 3D model points (used for head pose estimation).  These are standard model points.
MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0),        # Nose tip
    (0.0, -330.0, -65.0),   # Chin
    (-225.0, 170.0, -135.0),  # Left eye left corner
    (225.0, 170.0, -135.0),  # Right eye right corner
    (-150.0, -150.0, -125.0),  # Left mouth corner
    (150.0, -150.0, -125.0)   # Right mouth corner
], dtype=np.float32)

# Camera matrix for head pose estimation (assuming focal length of 1.0)
FOCAL_LENGTH = 1.0
CAMERA_MATRIX = np.array([
    [FOCAL_LENGTH, 0, 0],
    [0, FOCAL_LENGTH, 0],
    [0, 0, 1]
], dtype=np.float32)

DIST_COEFFS = np.zeros((4, 1))  # No lens distortion assumed

# Function to determine if a face is frontal
# args: 
#   landmarks: facial landmarks (numpy array)
#   yaw_threshold: threshold for yaw angle (degrees)
#   pitch_threshold: threshold for pitch angle (degrees)
#   roll_threshold: threshold for roll angle (degrees)
# returns: true if frontal, false otherwise
def is_frontal_face(landmarks, yaw_threshold, pitch_threshold, roll_threshold):
    # Define image points corresponding to the 3D model points
    image_points = np.array([
        landmarks[30],  # Nose tip
        landmarks[8],   # Chin
        landmarks[36],  # Left eye left corner
        landmarks[45],  # Right eye right corner
        landmarks[48],  # Left mouth corner
        landmarks[54]   # Right mouth corner
    ], dtype=np.float32)

    # Solve PnP to estimate head pose
    success, rotation_vector, _ = cv2.solvePnP(
        MODEL_POINTS, image_points, CAMERA_MATRIX, DIST_COEFFS, flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not success:
        return False

    # Convert rotation vector to rotation matrix
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

    # Extract yaw, pitch, and roll angles from the rotation matrix
    yaw = np.degrees(np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0]))   # Left-Right rotation
    pitch = np.degrees(np.arcsin(-rotation_matrix[2, 0]))  # Up-Down rotation
    roll = np.degrees(np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2]))  # Tilt

    # Check if yaw, pitch, and roll angles are within the thresholds
    return abs(yaw) < yaw_threshold and abs(pitch) < pitch_threshold and abs(roll) < roll_threshold


# Function to detect faces using dlib's face detector. Scales down the frame for faster detection. 
# args: 
#   frame: the input frame (BGR image)
#   scale_factor: factor to scale down the frame for faster detection
# returns: list of dlib rectangles representing the detected faces (scaled back to original size)
def detect_faces(frame, scale_factor=2.0):

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
     # Resize frame to detect smaller faces
    resized_gray = cv2.resize(gray, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)
    
    # Detect faces with lower confidence threshold
    faces = detector(resized_gray, 0)  # Lower threshold (detects more faces)

    # Scale back detected face coordinates
    scaled_faces = []
    for face in faces:
        scaled_faces.append(
            dlib.rectangle(
                int(face.left() / scale_factor),
                int(face.top() / scale_factor),
                int(face.right() / scale_factor),
                int(face.bottom() / scale_factor)
            )
        )
    
    return scaled_faces

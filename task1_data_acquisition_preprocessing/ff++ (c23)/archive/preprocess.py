import cv2
import os
import yaml
import dlib
import numpy as np

# Load configuration
with open("config.yaml", "r") as ymlfile:
    config = yaml.safe_load(ymlfile)

dataset_root = config["dataset_root"]
base_output_dir = config["output_dir"]

# Define specific subset and manipulation type
subset = "manipulated"
manipulation_types = [
    "DeepFakeDetection",
    "Deepfakes",
    "Face2Face",
    "FaceShifter",
    "FaceSwap",
    "NeuralTextures",
]

# Load dlib's face detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(config["landmark_model_path"])  # Ensure the .dat file path is set in config.yaml

# Define 3D model points of facial landmarks (assuming a standard face model)
MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0),        # Nose tip
    (0.0, -330.0, -65.0),   # Chin
    (-225.0, 170.0, -135.0),  # Left eye left corner
    (225.0, 170.0, -135.0),  # Right eye right corner
    (-150.0, -150.0, -125.0),  # Left mouth corner
    (150.0, -150.0, -125.0)   # Right mouth corner
], dtype=np.float32)

# Camera matrix (assumed values, should be calibrated for real-world applications)
FOCAL_LENGTH = 1.0
CAMERA_MATRIX = np.array([
    [FOCAL_LENGTH, 0, 0],
    [0, FOCAL_LENGTH, 0],
    [0, 0, 1]
], dtype=np.float32)

DIST_COEFFS = np.zeros((4, 1))  # Assuming no lens distortion

# Function to check if a face is frontal based on head pose estimation
def is_frontal_face(landmarks, yaw_threshold=20, pitch_threshold=20, roll_threshold=20):
    """
    Determines if the face is frontal based on head pose estimation.

    yaw_threshold, pitch_threshold, roll_threshold: Maximum allowed head rotation in degrees.
    """
    # Extract 2D landmark points corresponding to 3D model points
    image_points = np.array([
        landmarks[30],  # Nose tip
        landmarks[8],   # Chin
        landmarks[36],  # Left eye left corner
        landmarks[45],  # Right eye right corner
        landmarks[48],  # Left mouth corner
        landmarks[54]   # Right mouth corner
    ], dtype=np.float32)

    # Solve PnP to get rotation vector (head pose)
    success, rotation_vector, _ = cv2.solvePnP(
        MODEL_POINTS, image_points, CAMERA_MATRIX, DIST_COEFFS, flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not success:
        return False  # Could not determine head pose

    # Convert rotation vector to rotation matrix
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

    # Convert rotation matrix to Euler angles (yaw, pitch, roll)
    yaw = np.degrees(np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0]))   # Left-Right rotation
    pitch = np.degrees(np.arcsin(-rotation_matrix[2, 0]))  # Up-Down rotation
    roll = np.degrees(np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2]))  # Tilt

    # Check if the face is within frontal thresholds
    return abs(yaw) < yaw_threshold and abs(pitch) < pitch_threshold and abs(roll) < roll_threshold

# Loop through manipulation types
for manipulation_type in manipulation_types:
    print(f"Processing {manipulation_type}...")

    # Construct the video directory path (this is to access the different manipulated)
    if subset == "original":
        video_dir = os.path.join(dataset_root, "original_sequences", "videos")
    elif subset == "manipulated":
        video_dir = os.path.join(dataset_root, manipulation_type, "c23", "videos")
    else:
        raise ValueError("Invalid subset specified. Must be 'original' or 'manipulated'.")

    # Verify directory existence
    if not os.path.exists(video_dir):
        print(f"Skipping {manipulation_type}, directory not found: {video_dir}")
        continue  # Skip this manipulation type if directory doesn't exist

    # Create the output directory for this manipulation type
    output_dir = os.path.join(base_output_dir, manipulation_type)
    os.makedirs(output_dir, exist_ok=True)

    # List video files
    video_files = [f for f in os.listdir(video_dir) if f.endswith(('.mp4', '.avi', '.mov'))]

    # Process each video for the current manipulation type
    if video_files:
        for video_file in video_files:
            video_path = os.path.join(video_dir, video_file)
            print(f"Processing video: {video_path}")

            # Open video file
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Error: Could not open video file: {video_path}")
                continue  # Skip this file if it can't be opened

            frame_count = 0
            saved_frame_count = 0

            # Loop through each frame in the video
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break  # Stop if no frame is read

                frame_count += 1

                # Process every 10th frame
                if frame_count % 10 == 0:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = detector(gray)

                    for face in faces:
                        landmarks = predictor(gray, face)
                        landmarks = np.array([[p.x, p.y] for p in landmarks.parts()])

                        if is_frontal_face(landmarks):
                            # Crop the face region
                            x, y, w, h = face.left(), face.top(), face.width(), face.height()
                            cropped_face = frame[max(0, y):y+h, max(0, x):x+w]

                            # Resize face to 128x128 for uniformity
                            cropped_face = cv2.resize(cropped_face, (128, 128))

                            # Save cropped face
                            output_filename = f"{video_file[:-4]}_face_{frame_count}.jpg"
                            output_path = os.path.join(output_dir, output_filename)
                            cv2.imwrite(output_path, cropped_face)
                            print(f"Saved frontal face from frame {frame_count} to {output_path}")
                            saved_frame_count += 1

            cap.release()
            cv2.destroyAllWindows()
            print(f"Video {video_file} processed. Total frames: {frame_count}, Saved Faces: {saved_frame_count}")

    else:
        print(f"No video files found for {manipulation_type} in {video_dir}")

import cv2
import dlib
import numpy as np
import os
import shutil

# Paths
# input_folder = "datasets\celebdf-preprocessed-cropped\Celeb-synthesis"  # Folder containing input images
# output_folder = "synthesis dump"  # Folder to move non-frontal images
input_folder = r"D:\ACADEMICS\THESIS\Datasets\WDF\WildDeepfake\wdf_restructured\real_test"  # Folder containing input images
output_folder = r"D:\ACADEMICS\THESIS\Datasets\WDF\WildDeepfake\synthesis_dump\real_test2"  # Folder to move non-frontal images

# Ensure output folder exists
os.makedirs(output_folder, exist_ok=True)

# Load dlib face detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("task1_data_acquisition_preprocessing\shape_predictor_68_face_landmarks.dat") 

def get_yaw_angle(landmarks):
    """Calculate yaw (left-right rotation) based on facial landmarks."""
    left_eye = landmarks.part(36).x  # Left eye outer corner
    right_eye = landmarks.part(45).x  # Right eye outer corner
    nose_tip = landmarks.part(30).x  # Nose tip (center)

    yaw_angle = np.arctan2(nose_tip - (left_eye + right_eye) / 2, right_eye - left_eye) * 180 / np.pi
    return yaw_angle

def process_images():
    """Process images and move non-frontal faces to output folder."""
    for filename in os.listdir(input_folder):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            image_path = os.path.join(input_folder, filename)
            image = cv2.imread(image_path)

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = detector(gray)

            for face in faces:
                landmarks = predictor(gray, face)
                yaw = get_yaw_angle(landmarks)

                
                if abs(yaw) > 7:  # If yaw is beyond ±10 degrees, move image
                    print(f"Moving {filename} (Yaw: {yaw:.2f})")
                    shutil.move(image_path, os.path.join(output_folder, filename))
                    break  # Only move once per image

if __name__ == "__main__":
    process_images()
    print("Processing complete.")
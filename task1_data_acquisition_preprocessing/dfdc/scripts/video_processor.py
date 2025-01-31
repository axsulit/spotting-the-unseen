# This file contains the logic for processing individual video files.

import cv2
import os
import yaml
import numpy as np
from face_detector import detect_faces, is_frontal_face, predictor
from utils import create_directory

# Load configuration
with open("config.yaml", "r") as ymlfile:
    config = yaml.safe_load(ymlfile)

# Function to resize image while maintaining aspect ratio and adding padding
# args:
#  image: input image to be resized
#  target_size: desired output size (width, height)
# returns:
#  resized image with padding applied

# Function to resize an image to fit within a target size (default 720p - 1280x720) while maintaining aspect ratio
def resize_with_padding(image, target_size=(1280, 720)):
    # Get image dimensions
    h, w = image.shape[:2]
    target_w, target_h = target_size

    # Compute scaling factor
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)

    # Resize while keeping aspect ratio
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Create a black background (change to white with np.ones() * 255 if needed)
    padded_image = np.zeros((target_h, target_w, 3), dtype=np.uint8)

    # Compute padding offsets to center the resized image
    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2

    # Place the resized face in the center
    padded_image[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

    return padded_image

# Function to process a video file, detect faces, check for frontal faces, and save as a cropped frontal face image
# args:
#  video_path: path to the video file
#  output_dir: directory to save the cropped frontal face images
#  frame_interval: interval to process frames
#  yaw_threshold: threshold for yaw angle (degrees)
#  pitch_threshold: threshold for pitch angle (degrees)
#  roll_threshold: threshold for roll angle (degrees)
def process_video(video_path, output_dir, frame_interval, yaw_threshold, pitch_threshold, roll_threshold):

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file: {video_path}")
        return

    # Initialize frame count and saved frame count
    frame_count = 0 
    saved_frame_count = 0

    # Loop through the video frames
    while cap.isOpened():
        ret, frame = cap.read() # Read a frame
        if not ret: # If frame is not read, end the loop
            break

        frame_count += 1 # Else increment frame count

        # Process frame every frame_interval
        if frame_count % frame_interval == 0:
            # Detect faces in the current frame
            faces = detect_faces(frame)

            # Loop through the detected faces
            for face in faces:
                # Predict facial landmarks for the current face
                landmarks = predictor(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), face)
                landmarks = np.array([[p.x, p.y] for p in landmarks.parts()])

                # Check if the face is frontal based on head pose estimation.
                if is_frontal_face(landmarks, yaw_threshold, pitch_threshold, roll_threshold):
                    # Get the bounding box coordinates of the face.
                    x, y, w, h = face.left(), face.top(), face.width(), face.height()
                    
                    # Crop the face region from the frame
                    cropped_face = frame[max(0, y):y+h, max(0, x):x+w]

                    # Resize while maintaining aspect ratio
                    cropped_face_resized = resize_with_padding(cropped_face, (1280, 720))

                    # Save the cropped frontal face image
                    output_filename = f"{os.path.basename(video_path)[:-4]}_face_{frame_count}.jpg"
                    output_path = os.path.join(output_dir, output_filename)
                    cv2.imwrite(output_path, cropped_face_resized)
                    print(f"Saved frontal face from frame {frame_count} to {output_path}")
                    saved_frame_count += 1

    # Release the video capture and close all windows
    cap.release() 
    cv2.destroyAllWindows()
    print(f"Video {video_path} processed. Saved Faces: {saved_frame_count}")

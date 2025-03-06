# This file contains the logic for processing individual video files.

import cv2
import os
import yaml
import numpy as np
from face_detector import detect_faces, is_frontal_face, predictor, detect_faces_lenient
from utils import create_directory

# Load configuration
with open("config.yaml", "r") as ymlfile: # TODO: Replace with the correct path
    config = yaml.safe_load(ymlfile)

# Function to resize image while maintaining aspect ratio and adding padding
# args:
#  image: input image to be resized
#  target_size: desired output size (width, height)
# returns:
#  resized image with padding applied

# Function to resize an image to fit within a target size (default 720p - 1280x720) while maintaining aspect ratio
def resize(image, target_size=(512, 512)):
    target_w, target_h = target_size
    resized_image = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
   
    return resized_image

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
                    
                    # Make the face region square
                    if w != h:
                        h = w = max(w, h)
                        
                    # Crop the face region from the frame
                    cropped_face = frame[max(0, y):y+h, max(0, x):x+w]
                    
                    # Resize while maintaining aspect ratio
                    cropped_face_resized = resize(cropped_face, (512, 512))

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

def get_head_pose_angles(landmarks):
    """
    Calculate head pose angles (yaw, pitch, roll) from facial landmarks.
    Returns angles in degrees.
    """
    # Define image points corresponding to the 3D model points
    image_points = np.array([
        landmarks[30],  # Nose tip
        landmarks[8],   # Chin
        landmarks[36],  # Left eye left corner
        landmarks[45],  # Right eye right corner
        landmarks[48],  # Left mouth corner
        landmarks[54]   # Right mouth corner
    ], dtype=np.float32)
    
    # Get camera matrix and model points from face_detector
    from face_detector import MODEL_POINTS, CAMERA_MATRIX, DIST_COEFFS
    
    # Solve PnP to estimate head pose
    success, rotation_vector, _ = cv2.solvePnP(
        MODEL_POINTS, image_points, CAMERA_MATRIX, DIST_COEFFS, flags=cv2.SOLVEPNP_ITERATIVE
    )
    
    if not success:
        return 0, 0, 0  # Default values if estimation fails
    
    # Convert rotation vector to rotation matrix
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    
    # Extract yaw, pitch, and roll angles from the rotation matrix
    yaw = np.degrees(np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0]))
    pitch = np.degrees(np.arcsin(-rotation_matrix[2, 0]))
    roll = np.degrees(np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2]))
    
    return yaw, pitch, roll

def process_image(image_path, output_dir, frame_interval, yaw_threshold, pitch_threshold, roll_threshold):
    # Check if the path is an image
    if image_path.lower().endswith(('.jpg', '.jpeg', '.png')):
        # Read the image
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"Error: Could not open image file: {image_path}")
            return False
        
        # Try multiple detection strategies
        faces = []
        
        # Strategy 1: Standard detection
        faces = detect_faces(frame)
        
        # Strategy 2: If no faces found, try with different scale factor
        if len(faces) == 0:
            faces = detect_faces(frame, scale_factor=1.5)
        
        # Strategy 3: If still no faces, try with more lenient parameters
        if len(faces) == 0:
            faces = detect_faces_lenient(frame)
        
        # If no faces detected at all, return False
        if len(faces) == 0:
            print(f"No faces detected in {image_path}")
            return False
        
        # Flag to track if any frontal face was found
        found_frontal_face = False
        best_face = None
        best_score = float('-inf')
        best_landmarks = None
        best_angles = None
        
        # Loop through the detected faces to find the best one
        for face in faces:
            try:
                # Predict facial landmarks for the current face
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                landmarks = predictor(gray, face)
                landmarks = np.array([[p.x, p.y] for p in landmarks.parts()])
                
                # Get head pose angles
                yaw, pitch, roll = get_head_pose_angles(landmarks)
                
                # Calculate a quality score based on angles and face size
                face_size = face.width() * face.height()
                image_size = frame.shape[0] * frame.shape[1]
                size_ratio = face_size / image_size  # Relative size of face in image
                
                # Improved scoring: prioritize faces that are more frontal and larger
                angle_penalty = (abs(yaw)/yaw_threshold + abs(pitch)/pitch_threshold + abs(roll)/roll_threshold) / 3.0
                angle_score = 1.0 - min(angle_penalty, 1.0)  # Higher is better (1.0 is perfect)
                
                # Combined score: 70% angle quality, 30% size
                quality_score = (0.7 * angle_score + 0.3 * min(size_ratio * 10, 1.0)) * 100
                
                print(f"Face in {os.path.basename(image_path)}: yaw={yaw:.1f}, pitch={pitch:.1f}, roll={roll:.1f}, score={quality_score:.1f}")
                
                # Keep track of the best face
                if quality_score > best_score:
                    best_score = quality_score
                    best_face = face
                    best_landmarks = landmarks
                    best_angles = (yaw, pitch, roll)
                
                # Check if this face meets the strict frontal criteria
                if abs(yaw) < yaw_threshold and abs(pitch) < pitch_threshold and abs(roll) < roll_threshold:
                    found_frontal_face = True
            except Exception as e:
                print(f"Error processing face in {image_path}: {e}")
                continue
        
        # If we found a good face, process it
        if best_face is not None:
            yaw, pitch, roll = best_angles
            
            # Adaptive threshold based on image quality
            # For higher quality images (larger faces), we can be more strict
            face_size = best_face.width() * best_face.height()
            image_size = frame.shape[0] * frame.shape[1]
            size_ratio = face_size / image_size
            
            # Adjust relaxation factor based on face size
            # Smaller faces get more relaxed thresholds
            base_relaxation = 1.5
            size_adjustment = max(0, 0.5 - size_ratio * 2)  # Smaller faces get more relaxation
            relaxed_threshold = base_relaxation + size_adjustment
            
            # Accept face if it meets relaxed criteria or has a good quality score
            if ((abs(yaw) < yaw_threshold * relaxed_threshold and 
                 abs(pitch) < pitch_threshold * relaxed_threshold and 
                 abs(roll) < roll_threshold * relaxed_threshold) or 
                best_score > 60):  # Accept if score is above 60
                
                # Get the bounding box coordinates of the face
                x, y, w, h = best_face.left(), best_face.top(), best_face.width(), best_face.height()
                
                # Make the face region square with some margin
                margin = int(max(w, h) * 0.1)  # 10% margin
                center_x, center_y = x + w//2, y + h//2
                side = max(w, h) + 2 * margin
                
                # Calculate new coordinates with margin
                new_x = max(0, center_x - side//2)
                new_y = max(0, center_y - side//2)
                new_right = min(frame.shape[1], new_x + side)
                new_bottom = min(frame.shape[0], new_y + side)
                
                # Adjust to maintain square aspect ratio
                final_width = new_right - new_x
                final_height = new_bottom - new_y
                final_side = min(final_width, final_height)
                
                # Recenter if needed
                if final_width > final_side:
                    new_x += (final_width - final_side) // 2
                if final_height > final_side:
                    new_y += (final_height - final_side) // 2
                
                # Crop the face region from the frame
                cropped_face = frame[new_y:new_y+final_side, new_x:new_x+final_side]
                
                if cropped_face.size > 0:  # Make sure we have a valid crop
                    # Resize while maintaining aspect ratio
                    cropped_face_resized = resize(cropped_face, (512, 512))
                    
                    # Save the cropped frontal face image
                    output_filename = os.path.basename(image_path)
                    output_path = os.path.join(output_dir, output_filename)
                    cv2.imwrite(output_path, cropped_face_resized)
                    print(f"Saved face to {output_path} (angles: yaw={yaw:.1f}, pitch={pitch:.1f}, roll={roll:.1f}, score={best_score:.1f})")
                    found_frontal_face = True
                else:
                    print(f"Invalid crop for {image_path}")
            else:
                print(f"Best face in {image_path} exceeds angle thresholds: yaw={yaw:.1f}, pitch={pitch:.1f}, roll={roll:.1f}, score={best_score:.1f}")
        else:
            print(f"No suitable faces found in {image_path}")
                
        return found_frontal_face
    
    return False  # Default return if not an image or no frontal face found

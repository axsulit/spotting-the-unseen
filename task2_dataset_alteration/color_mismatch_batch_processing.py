import cv2
import dlib
import os
import numpy as np
import random

# Load dlib's pre-trained face detector & shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("task2_dataset_alteration/shape_predictor_68_face_landmarks.dat")  

def apply_partial_color_distortion(image, regions, intensity=50):
    """
    Apply random color distortions to specific regions.
    
    Parameters:
        image (numpy.ndarray): The input image.
        regions (list): A list of regions (each as [(x1, y1), (x2, y2), ...]).
        intensity (int): Maximum color distortion intensity.

    Returns:
        numpy.ndarray: The altered image.
    """
    image = image.astype(np.int16) 

    for region in regions:
        # Compute bounding box around selected region
        x_min = min(point[0] for point in region)
        x_max = max(point[0] for point in region)
        y_min = min(point[1] for point in region)
        y_max = max(point[1] for point in region)

        # Apply color distortion to this region
        for channel in range(3):  # BGR format
            shift = random.randint(-intensity, intensity)
            image[y_min:y_max, x_min:x_max, channel] = np.clip(image[y_min:y_max, x_min:x_max, channel] + shift, 0, 255)

    return image.astype(np.uint8)

def get_relevant_facial_regions(landmarks):
    """
    Define specific **forgery-prone** facial sub-regions using Dlib's 68-point landmarks.

    Parameters:
        landmarks (dlib.full_object_detection): Facial landmark object.
    
    Returns:
        list: List of regions (as point lists).
    """
    points = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(68)]

    # Prioritize relevant areas for face forgery mismatches
    regions = {
        "left_cheek": points[1:5],   # Extended Left cheek (landmarks 1 to 4)
        "right_cheek": points[12:16],  # Extended Right cheek (landmarks 12 to 15)
        "nose": points[27:36],  # Nose area (landmarks 27 to 35)
        "upper_forehead": points[17:27],  # Full Forehead (landmarks 17 to 26)
        "mid_forehead": points[19:24],  # Center Forehead (landmarks 19 to 23)
        "chin": points[5:12],  # Chin & Lower Jaw (landmarks 5 to 11)
        "mouth_area": points[48:61],  # Mouth surroundings (landmarks 48 to 60)
        "jawline": points[0:17],  # Full Jawline (landmarks 0 to 16)
    }

    # Select 1-2 facial regions randomly to distort
    selected_regions = random.sample(list(regions.values()), random.randint(1, 2))
    return selected_regions

def apply_random_image_distortion(image, intensity=50):
    """
    Apply a color mismatch to a random part of the image when no face is detected.
    
    Parameters:
        image (numpy.ndarray): The input image.
        intensity (int): Maximum color distortion intensity.

    Returns:
        numpy.ndarray: The altered image.
    """
    height, width = image.shape[:2]

    # Define a random rectangular region
    region_width = random.randint(width // 6, width // 3)
    region_height = random.randint(height // 6, height // 3)
    x = random.randint(0, width - region_width)
    y = random.randint(0, height - region_height)

    image = image.astype(np.int16)

    # Apply color distortion
    for channel in range(3):  # BGR format
        shift = random.randint(-intensity, intensity)
        image[y:y+region_height, x:x+region_width, channel] = np.clip(
            image[y:y+region_height, x:x+region_width, channel] + shift, 0, 255
        )

    # Convert back to uint8
    image = image.astype(np.uint8)

    print(f"Applied random distortion at: (x={x}, y={y}, w={region_width}, h={region_height})")
    return image

def process_images_with_color_mismatch(input_folder, output_folder, intensity=50):
    """
    Process images, detect faces using Dlib, and apply color mismatches to **selected facial areas**.
    If no face is detected, apply a random mismatch to another part of the image.

    Parameters:
        input_folder (str): Path to the input folder containing images.
        output_folder (str): Path to the output folder to save altered images.
        intensity (int): Maximum color distortion intensity.
    """
    if not os.path.exists(input_folder):
        print(f"Error: Input folder '{input_folder}' does not exist.")
        return

    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        file_path = os.path.join(input_folder, filename)

        if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            print(f"Skipping non-image file: {filename}")
            continue

        image = cv2.imread(file_path)
        if image is None:
            print(f"Error: Could not load image '{filename}'. Skipping.")
            continue

        # Convert image to grayscale for detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect face(s)
        faces = detector(gray)

        if len(faces) == 0:
            print(f"No face detected in {filename}. Applying random mismatch.")
            altered_image = apply_random_image_distortion(image, intensity)
        else:
            # Process only the first detected face
            face = faces[0]
            landmarks = predictor(gray, face)

            # Get the relevant face regions for distortion
            facial_regions = get_relevant_facial_regions(landmarks)

            # Apply localized color distortion
            altered_image = apply_partial_color_distortion(image, facial_regions, intensity)

        # Modify filename to indicate color mismatch
        name, ext = os.path.splitext(filename)
        new_filename = f"{name}_color_mismatch{ext}"
        output_path = os.path.join(output_folder, new_filename)

        cv2.imwrite(output_path, altered_image)
        print(f"Altered image saved: {output_path}")

    print("Processing complete.")

# Define input and output folder paths
input_base_folder = 'path_to_input_folder'  # Replace
output_base_folder = 'path_to_output_folder'  # Replace
resolutions = ["64x64","128x128", "256x256"]

# Process images across all resolutions
for res in resolutions:
    input_folder = os.path.join(input_base_folder, res)
    output_folder = os.path.join(output_base_folder, res)
    process_images_with_color_mismatch(input_folder, output_folder, intensity=50)

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
        x_min = min(point[0] for point in region)
        x_max = max(point[0] for point in region)
        y_min = min(point[1] for point in region)
        y_max = max(point[1] for point in region)

        for channel in range(3):  # BGR format
            shift = random.randint(-intensity, intensity)
            image[y_min:y_max, x_min:x_max, channel] = np.clip(image[y_min:y_max, x_min:x_max, channel] + shift, 0, 255)

    return image.astype(np.uint8)

def get_facial_regions(landmarks):
    """
    Define specific **forgery-prone** facial sub-regions using Dlib's 68-point landmarks.

    Parameters:
        landmarks (dlib.full_object_detection): Facial landmark object.

    Returns:
        list: List of regions (as point lists).
    """
    points = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(68)]

    regions = {
        "left_cheek": points[1:5],   
        "right_cheek": points[12:16],  
        "nose": points[27:36],  
        "upper_forehead": points[17:27],  
        "mid_forehead": points[19:24],  
        "chin": points[5:12],  
        "mouth_area": points[48:61],  
        "jawline": points[0:17],  
    }

    selected_regions = random.sample(list(regions.values()), random.randint(1, 2))
    return selected_regions

def process_images(input_folder, output_folder, intensity=50):
    """
    Process images in the 256x256 resolution folder and save them directly in the output folder.

    Parameters:
        input_folder (str): Path to the input folder.
        output_folder (str): Path to the output folder.
        intensity (int): Color distortion intensity.
    """
    if not os.path.exists(input_folder):
        print(f"Error: Input folder '{input_folder}' does not exist.")
        return

    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            continue

        # Load image
        input_path = os.path.join(input_folder, filename)
        image = cv2.imread(input_path)
        if image is None:
            continue

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        face_regions = None  # To store face landmark regions
        random_region = None  # To store a random region if no face is found

        if len(faces) > 0:
            face = faces[0]
            landmarks = predictor(gray, face)
            face_regions = get_facial_regions(landmarks)
        else:
            # Generate a random region if no face is detected
            h, w = image.shape[:2]
            region_w = random.randint(w // 6, w // 3)
            region_h = random.randint(h // 6, h // 3)
            x = random.randint(0, w - region_w)
            y = random.randint(0, h - region_h)
            random_region = [[(x, y), (x + region_w, y), (x + region_w, y + region_h), (x, y + region_h)]]

        if face_regions:
            altered_image = apply_partial_color_distortion(image, face_regions, intensity)
        else:
            altered_image = apply_partial_color_distortion(image, random_region, intensity)

        name, ext = os.path.splitext(filename)
        new_filename = f"{name}_color_mismatch{ext}"
        output_path = os.path.join(output_folder, new_filename)

        cv2.imwrite(output_path, altered_image)
        print(f"Saved: {output_path}")

    print("Processing complete.")

# Define input and output folder paths
input_folder = 'datasets/2 celebdf-resized/Celeb-synthesis/256x256'
output_folder = 'datasets/3.2 celebdf-color_mismatch/Celeb-synthesis'

# Process images and save directly in the output folder
process_images(input_folder, output_folder, intensity=50)

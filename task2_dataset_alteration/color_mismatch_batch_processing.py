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

def scale_regions(regions, original_size, target_size):
    """
    Scale facial regions from one resolution to another.
    
    Parameters:
        regions (list): List of regions with facial points.
        original_size (tuple): Original image size (width, height).
        target_size (tuple): Target image size (width, height).

    Returns:
        list: Scaled regions for the target resolution.
    """
    orig_w, orig_h = original_size
    target_w, target_h = target_size

    scale_x = target_w / orig_w
    scale_y = target_h / orig_h

    scaled_regions = []
    for region in regions:
        scaled_region = [(int(x * scale_x), int(y * scale_y)) for x, y in region]
        scaled_regions.append(scaled_region)

    return scaled_regions

def process_images_across_resolutions(input_base_folder, output_base_folder, resolutions, intensity=50):
    """
    Process images at multiple resolutions, ensuring color mismatches are consistent.
    
    Parameters:
        input_base_folder (str): Path to the base input folder.
        output_base_folder (str): Path to the base output folder.
        resolutions (list): List of resolutions to process (e.g., ["64x64", "128x128", "256x256"])
        intensity (int): Color distortion intensity.
    """
    high_res = resolutions[-1]  # Highest resolution for landmark detection
    high_res_folder = os.path.join(input_base_folder, high_res)
    
    if not os.path.exists(high_res_folder):
        print(f"Error: High-resolution folder '{high_res_folder}' does not exist.")
        return

    for filename in os.listdir(high_res_folder):
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            continue

        face_regions = None  # To store face landmark regions
        random_region = None  # To store a random region if no face is found

        # Load high-res image
        high_res_path = os.path.join(high_res_folder, filename)
        high_res_image = cv2.imread(high_res_path)
        if high_res_image is None:
            continue

        gray = cv2.cvtColor(high_res_image, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        if len(faces) > 0:
            face = faces[0]
            landmarks = predictor(gray, face)
            face_regions = get_facial_regions(landmarks)
        else:
            # Generate a random region if no face is detected
            h, w = high_res_image.shape[:2]
            region_w = random.randint(w // 6, w // 3)
            region_h = random.randint(h // 6, h // 3)
            x = random.randint(0, w - region_w)
            y = random.randint(0, h - region_h)
            random_region = [[(x, y), (x + region_w, y), (x + region_w, y + region_h), (x, y + region_h)]]

        for res in resolutions:
            input_folder = os.path.join(input_base_folder, res)
            output_folder = os.path.join(output_base_folder, res)
            os.makedirs(output_folder, exist_ok=True)

            input_path = os.path.join(input_folder, filename)
            if not os.path.exists(input_path):
                continue

            image = cv2.imread(input_path)
            if image is None:
                continue

            original_size = high_res_image.shape[:2][::-1]  # (width, height)
            target_size = image.shape[:2][::-1]  # (width, height)

            if face_regions:
                scaled_regions = scale_regions(face_regions, original_size, target_size)
                altered_image = apply_partial_color_distortion(image, scaled_regions, intensity)
            else:
                scaled_random_region = scale_regions(random_region, original_size, target_size)
                altered_image = apply_partial_color_distortion(image, scaled_random_region, intensity)

            name, ext = os.path.splitext(filename)
            new_filename = f"{name}_color_mismatch{ext}"
            output_path = os.path.join(output_folder, new_filename)

            cv2.imwrite(output_path, altered_image)
            print(f"Saved: {output_path}")

    print("Processing complete.")

# Define input and output folder paths
input_base_folder = 'datasets/2 celebdf-resized/Celeb-real'
output_base_folder = 'datasets/3.2 celebdf-color_mismatch/Celeb-real'
resolutions = ["64x64", "128x128", "256x256"]

# Process images with consistent mismatched regions across all resolutions
process_images_across_resolutions(input_base_folder, output_base_folder, resolutions, intensity=50)

import cv2
import os
import numpy as np
import random

def apply_color_distortion(image, region, intensity=50):
    """
    Apply random color distortions to a specific region of the image.

    Parameters:
        image (numpy.ndarray): The input image.
        region (tuple): The region to alter (x, y, width, height).
        intensity (int): Maximum color distortion intensity.

    Returns:
        numpy.ndarray: The altered image.
    """
    x, y, w, h = region

    # Clip region to ensure it's within image bounds
    x = max(0, x)
    y = max(0, y)
    w = min(w, image.shape[1] - x)
    h = min(h, image.shape[0] - y)

    # Extract the region
    region_of_interest = image[y:y+h, x:x+w]

    # Apply random color distortions
    for channel in range(3):  # Assuming the image is in BGR format
        random_shift = random.randint(-intensity, intensity)
        region_of_interest[:, :, channel] = cv2.add(region_of_interest[:, :, channel], random_shift)

    # Replace the region in the original image
    image[y:y+h, x:x+w] = region_of_interest
    return image

def process_images_with_color_mismatch(input_folder, output_folder, intensity=50):
    """
    Process multiple images in a folder to apply random color mismatches to random regions.

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

        # Define random region for distortion
        height, width = image.shape[:2]
        region_width = random.randint(width // 10, width // 4)
        region_height = random.randint(height // 10, height // 4)
        x = random.randint(0, width - region_width)
        y = random.randint(0, height - region_height)

        # Apply color distortion
        altered_image = apply_color_distortion(image, (x, y, region_width, region_height), intensity)

        # Save the altered image
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, altered_image)
        print(f"Altered image saved: {output_path}")

    print("Processing complete.")

# Define input and output folder paths
input_folder = 'path_to_input_folder'  # Replace
output_folder = 'path_to_output_folder'  # Replace

# Process images with color mismatches
process_images_with_color_mismatch(input_folder, output_folder, intensity=50)

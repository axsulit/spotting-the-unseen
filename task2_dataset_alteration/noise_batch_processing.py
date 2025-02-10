import cv2
import os
import numpy as np
import random
from collections import defaultdict

def add_gaussian_noise(image, noise_level):
    """Apply Gaussian noise to the image."""
    noise = np.random.normal(0, noise_level, image.shape).astype(np.int16)
    noisy_image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return noisy_image

def add_salt_and_pepper_noise(image, amount=0.02):
    """Apply Salt & Pepper noise to the image."""
    noisy_image = image.copy()
    num_salt = int(amount * image.size * 0.5)
    num_pepper = int(amount * image.size * 0.5)

    # Add Salt (white) noise
    coords = [np.random.randint(0, i - 1, num_salt) for i in image.shape[:2]]
    noisy_image[coords[0], coords[1], :] = 255

    # Add Pepper (black) noise
    coords = [np.random.randint(0, i - 1, num_pepper) for i in image.shape[:2]]
    noisy_image[coords[0], coords[1], :] = 0

    return noisy_image

def add_speckle_noise(image, noise_level=0.1):
    """Apply Speckle noise to the image."""
    noise = np.random.randn(*image.shape).astype(np.float32) * noise_level
    noisy_image = np.clip(image.astype(np.float32) + image.astype(np.float32) * noise, 0, 255).astype(np.uint8)
    return noisy_image

def apply_noise(image, noise_type, noise_level):
    """Apply a selected noise type to the image."""
    if noise_type == "gaussian":
        return add_gaussian_noise(image, noise_level)
    elif noise_type == "salt_pepper":
        return add_salt_and_pepper_noise(image, amount=noise_level / 300)
    elif noise_type == "speckle":
        return add_speckle_noise(image, noise_level / 50)
    else:
        return image

def process_images_with_fixed_noise(input_base_folder, output_base_folder, resolutions):
    """
    Process images with fixed noise levels and noise types per video ID across all resolutions.

    Parameters:
        input_base_folder (str): Base input folder containing images.
        output_base_folder (str): Base output folder to save altered images.
        resolutions (list): List of resolutions to process.
    """
    video_groups = defaultdict(list)

    # Collect all files from the first resolution (e.g., "64x64") for consistency
    first_res = resolutions[0]
    first_res_folder = os.path.join(input_base_folder, first_res)

    if not os.path.exists(first_res_folder):
        print(f"Error: Input folder '{first_res_folder}' does not exist.")
        return
    
    # TODO: FIX SPLIT CODE
    # Group filenames by (celebrity ID + video ID) using the first resolution as reference
    for filename in os.listdir(first_res_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            parts = filename.split("_")
            if len(parts) < 3:
                print(f"Skipping invalid filename format: {filename}")
                continue

            celeb_id = parts[0]   # e.g., "id0"
            video_id = parts[1]   # e.g., "0008"
            video_key = f"{celeb_id}_{video_id}"  # Unique key: "id0_0008"
            
            video_groups[video_key].append(filename)

    # Assign a fixed noise level & noise type per video ID (CONSISTENT ACROSS RESOLUTIONS)
    noise_types = ["gaussian", "salt_pepper", "speckle"]
    noise_configs = {
        video_key: {
            "noise_level": random.randint(10, 40),  # Fixed noise level per video
            "noise_type": random.choice(noise_types)  # Fixed noise type per video
        }
        for video_key in video_groups
    }

    # Process each resolution
    for res in resolutions:
        input_folder = os.path.join(input_base_folder, res)
        output_folder = os.path.join(output_base_folder, res)

        if not os.path.exists(input_folder):
            print(f"Error: Input folder '{input_folder}' does not exist.")
            continue

        os.makedirs(output_folder, exist_ok=True)

        for video_key, filenames in video_groups.items():
            base_noise = noise_configs[video_key]["noise_level"]
            noise_type = noise_configs[video_key]["noise_type"]
            print(f"Processing video {video_key} at {res} with noise type: {noise_type}, base noise level: {base_noise}")

            for filename in filenames:
                file_path = os.path.join(input_folder, filename)
                image = cv2.imread(file_path)
                if image is None:
                    print(f"Error: Could not load image '{filename}'. Skipping.")
                    continue

                # Apply the **same** noise level per frame across resolutions
                altered_image = apply_noise(image, noise_type, base_noise)

                # Modify filename to include noise type and noise level
                name, ext = os.path.splitext(filename)
                new_filename = f"{name}_noise{base_noise}_{noise_type}{ext}"
                output_path = os.path.join(output_folder, new_filename)

                cv2.imwrite(output_path, altered_image)
                print(f"Saved {new_filename} at {res} with noise type {noise_type} and noise level {base_noise}")

    print("Processing complete.")

# Define input and output folder paths
input_base_folder = 'datasets/2 celebdf-resized/Celeb-real'
output_base_folder = 'datasets/3.4 celebdf-noise_addition/Celeb-real'
resolutions = ["64x64", "128x128", "256x256"]

# Process images with fixed noise levels across resolutions
process_images_with_fixed_noise(input_base_folder, output_base_folder, resolutions)

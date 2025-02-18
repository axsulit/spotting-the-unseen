import cv2
import os
import numpy as np
import random

def add_random_noise(image, noise_level=25):
    """
    Add random noise to an image to simulate artifacts.

    Parameters:
        image (numpy.ndarray): The input image.
        noise_level (int): The standard deviation of the Gaussian noise.

<<<<<<< Updated upstream
    Returns:
        numpy.ndarray: The altered image with added noise.
    """
    noise = np.random.normal(0, noise_level, image.shape).astype(np.int16)
    noisy_image = cv2.add(image.astype(np.int16), noise, dtype=cv2.CV_8U)
    return noisy_image

def process_images_with_noise(input_folder, output_folder, noise_level=25):
    """
    Process multiple images in a folder to add random noise.

    Parameters:
        input_folder (str): Path to the input folder containing images.
        output_folder (str): Path to the output folder to save altered images.
        noise_level (int): The standard deviation of the Gaussian noise.
    """
    if not os.path.exists(input_folder):
        print(f"Error: Input folder '{input_folder}' does not exist.")
        return

    os.makedirs(output_folder, exist_ok=True)
=======
    # Define noise types and intensity levels
    noise_types = ["salt_pepper", "gaussian", "speckle"] # Primary experiment
    # noise_types = ["gaussian", "speckle"] # Secondary experiment
    noise_levels = [10,20,30,40,50]  # Fixed intensity levels
>>>>>>> Stashed changes

    for filename in os.listdir(input_folder):
        file_path = os.path.join(input_folder, filename)

        if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            print(f"Skipping non-image file: {filename}")
            continue

        image = cv2.imread(file_path)
        if image is None:
            print(f"Error: Could not load image '{filename}'. Skipping.")
            continue

        # Add random noise
        altered_image = add_random_noise(image, noise_level)

        # Save the altered image
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, altered_image)
        print(f"Altered image with noise saved: {output_path}")

    print("Processing complete.")

# Define input and output folder paths
<<<<<<< Updated upstream
input_folder = 'path_to_input_folder'  # Replace
output_folder = 'path_to_output_folder'  # Replace
=======
input_folder = r'D:\ACADEMICS\THESIS\Datasets\FF\c40\preprocessed_frames\originalFrames\alterations\Resized\youtube\256x256'
output_base_folder = r'D:\ACADEMICS\THESIS\Datasets\FF\c40\preprocessed_frames\originalFrames\alterations\Noise\youtube'
>>>>>>> Stashed changes

# Process images with random noise
process_images_with_noise(input_folder, output_folder, noise_level=25)
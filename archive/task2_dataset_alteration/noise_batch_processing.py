import cv2
import os
import numpy as np

def add_gaussian_noise(image, noise_level):
    """Apply Gaussian noise to the image."""
    noise = np.random.normal(0, noise_level, image.shape).astype(np.int16)
    noisy_image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return noisy_image

def add_salt_and_pepper_noise(image, amount):
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

def add_speckle_noise(image, noise_level):
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
    return image

def process_images_with_fixed_noise(input_folder, output_base_folder):
    """
    Process images with fixed noise levels and noise types across all resolutions.

    Parameters:
        input_folder (str): Base input folder containing images.
        output_base_folder (str): Base output folder to save altered images.
    """

    # Define noise types and intensity levels
    noise_types = ["salt_pepper"] # Primary experiment
    # noise_types = ["gaussian", "speckle"] # Secondary experiment
    noise_levels = [10,20,30,40,50]  # Fixed intensity levels

    for filename in os.listdir(input_folder):
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            print(f"Skipping non-image file: {filename}")
            continue

        input_path = os.path.join(input_folder, filename)
        image = cv2.imread(input_path)
        if image is None:
            print(f"Error: Could not load image '{filename}'. Skipping.")
            continue

        # Apply each noise type at different intensities
        for noise_type in noise_types:
            for noise_level in noise_levels:
                noisy_image = apply_noise(image, noise_type, noise_level)

                # Define folder structure: output_base_folder/noise_type/intensity/
                intensity_folder = os.path.join(output_base_folder, noise_type, str(noise_level))
                os.makedirs(intensity_folder, exist_ok=True)

                output_filename = f"{os.path.splitext(filename)[0]}_{noise_type}_{noise_level}.jpg"
                output_path = os.path.join(intensity_folder, output_filename)

                cv2.imwrite(output_path, noisy_image)
                print(f"Saved: {output_path}")

    print("Processing complete.")

# Define input and output folder paths
input_folder = 'datasets/2 celebdf-resized/YouTube-real/256x256'
output_base_folder = 'datasets/3.4 celebdf-noise_addition/YouTube-real'

# Apply noise transformations
process_images_with_fixed_noise(input_folder, output_base_folder)

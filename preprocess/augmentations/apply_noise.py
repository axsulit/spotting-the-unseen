import cv2
import os
import numpy as np
from pathlib import Path

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

def batch_apply_noise(input_folder, output_folder):
    """Applies salt & pepper noise to all images in input_folder and saves them in output_folder."""
    input_folder = Path(input_folder)
    noise_type = "salt_pepper"
    noise_levels = [10, 20, 30, 40, 50]

    for level in noise_levels:
        level_folder = Path(output_folder) / f"{noise_type}_{level}"
        level_folder.mkdir(parents=True, exist_ok=True)

        for image_file in input_folder.glob("*"):
            if image_file.suffix.lower() not in [".png", ".jpg", ".jpeg"]:
                continue

            image = cv2.imread(str(image_file))
            if image is None:
                print(f"❌ Skipping unreadable: {image_file}")
                continue

            noisy_image = apply_noise(image, noise_type, level)

            out_name = f"{image_file.stem}_{noise_type}_{level}{image_file.suffix}"
            out_path = level_folder / out_name
            cv2.imwrite(str(out_path), noisy_image)
            print(f"✅ Saved: {out_path}")

             
import cv2
import os
import numpy as np
import random

def apply_blur(image, blur_type, kernel_size):
    """Applies a specified blur type and kernel size to an image."""
    if blur_type == "gaussian":
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    
    elif blur_type == "median":
        return cv2.medianBlur(image, kernel_size)

    elif blur_type == "bilateral":
        d, sigma_color, sigma_space = kernel_size
        return cv2.bilateralFilter(image, d, sigma_color, sigma_space)
    
    return image  # Default (no change)

def blur_images_across_resolutions(input_base_folder, output_base_folder, resolutions):
    """Ensures the same blur type is applied across different resolutions."""
    for filename in os.listdir(os.path.join(input_base_folder, "64x64")):
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            print(f"Skipping non-image file: {filename}")
            continue

        blur_type = random.choice(["gaussian", "median", "bilateral"])

        if blur_type in ["gaussian", "median"]:
            kernel_size = random.choice(range(3, 16, 2))  # Odd kernel sizes
        else:  # Bilateral filter parameters
            kernel_size = (random.randint(5, 15), random.randint(50, 150), random.randint(50, 150))

        for res in resolutions:
            input_folder = os.path.join(input_base_folder, res)
            output_folder = os.path.join(output_base_folder, res)

            os.makedirs(output_folder, exist_ok=True)

            input_path = os.path.join(input_folder, filename)
            output_filename = f"{os.path.splitext(filename)[0]}_{blur_type}_{kernel_size}{os.path.splitext(filename)[1]}"
            output_path = os.path.join(output_folder, output_filename)

            image = cv2.imread(input_path)
            if image is None:
                print(f"Error: Could not load image '{input_path}'. Skipping.")
                continue

            # Apply the same blur settings across resolutions
            blurred_image = apply_blur(image, blur_type, kernel_size)
            cv2.imwrite(output_path, blurred_image)
            print(f"Blurred image saved ({blur_type}, {kernel_size}): {output_path}")

    print("Batch processing complete.")

# Define input and output folder structures
input_base_folder = 'datasets/2 celebdf-resized/Celeb-real'
output_base_folder = 'datasets/3.1 celebdf-blurring/Celeb-real'
resolutions = ["64x64", "128x128", "256x256"]

# Apply consistent blurring across resolutions
blur_images_across_resolutions(input_base_folder, output_base_folder, resolutions)

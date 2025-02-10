import cv2
import os
import numpy as np
import random

def apply_random_blur(image):
    blur_type = random.choice(["gaussian", "median", "bilateral"])
    
    if blur_type == "gaussian":
        kernel_size = random.choice(range(3, 16, 2))  # Random odd kernel size
        blurred_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    
    elif blur_type == "median":
        kernel_size = random.choice(range(5, 16, 2))
        blurred_image = cv2.medianBlur(image, kernel_size)

    elif blur_type == "bilateral":
        d = random.randint(5, 15)  # Diameter of pixel neighborhood
        sigma_color = random.randint(50, 150)  # Color space filtering
        sigma_space = random.randint(50, 150)  # Spatial filtering
        blurred_image = cv2.bilateralFilter(image, d, sigma_color, sigma_space)
    
    return blurred_image, blur_type

def blur_images(input_folder, output_folder):
    if not os.path.exists(input_folder):
        print(f"Error: Input folder '{input_folder}' does not exist.")
        return

    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder)[:20]:
        file_path = os.path.join(input_folder, filename)

        if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            print(f"Skipping non-image file: {filename}")
            continue

        image = cv2.imread(file_path)
        if image is None:
            print(f"Error: Could not load image '{filename}'. Skipping.")
            continue

        # Apply random blur type
        blurred_image, blur_type = apply_random_blur(image)

        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, blurred_image)
        print(f"Blurred image saved ({blur_type}): {output_path}")

    print("Processing complete.")

# Define input and output folder paths
input_folder = 'path_to_input_folder'  # Replace
output_folder = 'path_to_output_folder'  # Replace

# Apply different types of blurring
blur_images(input_folder, output_folder)

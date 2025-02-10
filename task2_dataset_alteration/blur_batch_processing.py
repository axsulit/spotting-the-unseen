import cv2
import os
import numpy as np
import random

def apply_random_blur(image):
    blur_type = random.choice(["gaussian", "median", "bilateral"])
    kernel_size = None  # To store kernel size info

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
        kernel_size = f"{d}_{sigma_color}_{sigma_space}"  # Represent bilateral filter params

    return blurred_image, blur_type, kernel_size

def blur_images(input_folder, output_folder):
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

        # Apply random blur type
        blurred_image, blur_type, kernel_size = apply_random_blur(image)

        # Modify filename to include blur type and kernel size
        name, ext = os.path.splitext(filename)
        new_filename = f"{name}_{blur_type}_{kernel_size}{ext}"
        output_path = os.path.join(output_folder, new_filename)

        cv2.imwrite(output_path, blurred_image)
        print(f"Blurred image saved: {output_path}")

    print("Processing complete.")

# Define input and output folder paths
input_folder = 'datasets\\2 celebdf-resized\\Celeb-real\\64x64'
output_folder = 'datasets\\3.1 celebdf-blurring\\Celeb-real\\64x64'

# Apply different types of blurring
blur_images(input_folder, output_folder)

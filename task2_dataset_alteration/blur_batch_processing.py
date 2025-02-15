import cv2
import os
import numpy as np

def apply_blur(image, blur_type, intensity):
    """Applies a specific blur type and intensity to an image."""
    if blur_type == "gaussian":
        kernel_size = max(3, int(intensity * min(image.shape[:2]) / 100))
        if kernel_size % 2 == 0:  # Ensure kernel size is odd
            kernel_size += 1
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    
    elif blur_type == "median":
        kernel_size = max(3, int(intensity * min(image.shape[:2]) / 100))
        if kernel_size % 2 == 0:
            kernel_size += 1
        return cv2.medianBlur(image, kernel_size)

    elif blur_type == "bilateral":
        d = max(5, int(intensity * min(image.shape[:2]) / 100))
        sigma_color = int(intensity * 2.5)
        sigma_space = int(intensity * 2.5)
        return cv2.bilateralFilter(image, d, sigma_color, sigma_space)

    return image  

def blur_images(input_folder, output_base_folder):
    """Applies different blurs at various intensities and organizes output into subfolders."""
    
    # Define intensity levels
    primary_gaussian_intensities = [5, 10, 15, 20, 25]  # Used for Gaussian blur
    secondary_intensities = [5, 10, 15, 20, 25]  # Used for both Gaussian & Other blurs
    
    blur_types_secondary = ["median", "bilateral"]

    # Create primary and secondary output folders
    primary_output_folder = os.path.join(output_base_folder, "primary")
    secondary_output_folder = os.path.join(output_base_folder, "secondary")
    
    os.makedirs(primary_output_folder, exist_ok=True)
    os.makedirs(secondary_output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            print(f"Skipping non-image file: {filename}")
            continue

        input_path = os.path.join(input_folder, filename)
        image = cv2.imread(input_path)
        if image is None:
            print(f"Error: Could not load image '{input_path}'. Skipping.")
            continue

        # Apply Gaussian Blur for primary & secondary sets
        for intensity in primary_gaussian_intensities:
            blurred_image = apply_blur(image, "gaussian", intensity)

            output_filename = f"{os.path.splitext(filename)[0]}_gaussian_{intensity}.jpg"
            output_path = os.path.join(primary_output_folder, output_filename)
            cv2.imwrite(output_path, blurred_image)
            print(f"Saved (Primary): {output_path}")

            # Save 5, 10, 15, 25 Gaussian blur in the secondary set as well
            if intensity in secondary_intensities:
                secondary_output_path = os.path.join(secondary_output_folder, output_filename)
                cv2.imwrite(secondary_output_path, blurred_image)
                print(f"Saved (Secondary): {secondary_output_path}")

        # Apply Median & Bilateral Blurs for secondary set
        for blur_type in blur_types_secondary:
            for intensity in secondary_intensities:
                blurred_image = apply_blur(image, blur_type, intensity)

                output_filename = f"{os.path.splitext(filename)[0]}_{blur_type}_{intensity}.jpg"
                output_path = os.path.join(secondary_output_folder, output_filename)
                cv2.imwrite(output_path, blurred_image)
                print(f"Saved (Secondary): {output_path}")

    print("Batch processing complete.")

# Define input and output folder paths
input_folder = 'datasets/2 celebdf-resized/Celeb-real/256x256'
output_base_folder = 'datasets/3.1 celebdf-blurring/Celeb-real'

# Apply blurring transformations
blur_images(input_folder, output_base_folder)

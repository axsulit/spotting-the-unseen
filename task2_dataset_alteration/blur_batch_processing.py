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
    """Applies different blurs at various intensities and organizes output into structured subfolders."""
    
    # Define intensity levels (Same for all blur types)
    intensities = [5, 10, 15, 20, 25]  

    # Define blur types
    blur_types = ["gaussian"] # primary experiment
    # blur_types = ["median", "bilateral"] # secondary experiment

    # Process each split (train/test/val)
    for split in ['train', 'test', 'val']:
        split_input_folder = os.path.join(input_folder, split)
        if not os.path.exists(split_input_folder):
            print(f"Warning: {split_input_folder} does not exist. Skipping.")
            continue

        for filename in os.listdir(split_input_folder):
            if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                print(f"Skipping non-image file: {filename}")
                continue

            input_path = os.path.join(split_input_folder, filename)
            image = cv2.imread(input_path)
            if image is None:
                print(f"Error: Could not load image '{input_path}'. Skipping.")
                continue

            # Apply all blur types and save them in organized subfolders
            for blur_type in blur_types:
                for intensity in intensities:
                    blurred_image = apply_blur(image, blur_type, intensity)

                    # Define folder structure: output_base_folder/blur_intensity/split/
                    output_folder = os.path.join(output_base_folder, f"blur_{intensity}", split)
                    os.makedirs(output_folder, exist_ok=True)

                    output_filename = f"{os.path.splitext(filename)[0]}_{blur_type}_{intensity}.jpg"
                    output_path = os.path.join(output_folder, output_filename)

                    cv2.imwrite(output_path, blurred_image)
                    print(f"Saved: {output_path}")

    print("Batch processing complete.")

# Define input and output folder paths
# input_folder = r'D:\.THESIS\WildDeepfake\wdf_final_fake\01_wdf_fake_unaltered'
# output_base_folder = r'D:\.THESIS\WildDeepfake\wdf_final_fake'
input_folder = r'D:\ACADEMICS\THESIS\Datasets\WDF\WildDeepfake\wdf_final_real\01_wdf_real_unaltered'
output_base_folder = r'D:\ACADEMICS\THESIS\Datasets\WDF\WildDeepfake\wdf_final_real'

# Apply blurring transformations
blur_images(input_folder, output_base_folder)

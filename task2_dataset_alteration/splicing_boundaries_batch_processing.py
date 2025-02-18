import cv2
import os
import numpy as np
import random


def add_splicing_boundaries(image, num_boundaries=3, boundary_thickness=2, intensity=255):
    """
    Add faint boundary lines to simulate poorly blended regions in the image.

    Parameters:
        image (numpy.ndarray): The input image.
        num_boundaries (int): Number of random boundary lines to add.
        boundary_thickness (int): Thickness of the boundary lines.
        intensity (int): Brightness of the boundary lines (0-255).

    Returns:
        numpy.ndarray: The altered image.
    """
    height, width = image.shape[:2]

    for _ in range(num_boundaries):
        # Randomly choose horizontal or vertical boundary
        if random.choice([True, False]):  # Horizontal boundary
            y = random.randint(0, height - 1)
            cv2.line(image, (0, y), (width, y), (intensity, intensity, intensity), thickness=boundary_thickness)
        else:  # Vertical boundary
            x = random.randint(0, width - 1)
            cv2.line(image, (x, 0), (x, height), (intensity, intensity, intensity), thickness=boundary_thickness)

    return image


def process_images_with_splicing_boundaries(input_folder, output_folder, num_boundaries=3, boundary_thickness=2,
                                            intensity=255):
    """
    Process multiple images in a folder to add splicing boundaries.

    Parameters:
        input_folder (str): Path to the input folder containing images.
        output_folder (str): Path to the output folder to save altered images.
        num_boundaries (int): Number of splicing boundaries to add to each image.
        boundary_thickness (int): Thickness of the splicing boundaries.
        intensity (int): Brightness of the splicing boundaries (0-255).
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

        # Add splicing boundaries
        altered_image = add_splicing_boundaries(image, num_boundaries, boundary_thickness, intensity)

        # Save the altered image
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, altered_image)
        print(f"Altered image with splicing boundaries saved: {output_path}")

    print("Processing complete.")


# Define input and output folder paths
<<<<<<< Updated upstream
input_folder = 'path_to_input_folder'  # Replace
output_folder = 'path_to_output_folder'  # Replace
=======
input_folder = r'D:\ACADEMICS\THESIS\Datasets\FF\c40\preprocessed_frames\originalFrames\alterations\Resized\youtube\256x256'
output_base_folder = r'D:\ACADEMICS\THESIS\Datasets\FF\c40\preprocessed_frames\originalFrames\alterations\SplicingBoundaries\youtube'
>>>>>>> Stashed changes

# Process images with splicing boundaries
process_images_with_splicing_boundaries(input_folder, output_folder, num_boundaries=3, boundary_thickness=2,
                                        intensity=255)

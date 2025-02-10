import cv2
import dlib
import os
import numpy as np
import random

# Load Dlib face detector
detector = dlib.get_frontal_face_detector()

def apply_canny_edge_detection(image, edge_opacity, edge_threshold1=50, edge_threshold2=150):
    """
    Apply Canny Edge Detection with a fixed opacity for consistency across resolutions.

    Parameters:
        image (numpy.ndarray): Input image.
        edge_opacity (float): Fixed opacity for edge blending.
        edge_threshold1 (int): First threshold for Canny edge detection.
        edge_threshold2 (int): Second threshold for Canny edge detection.

    Returns:
        numpy.ndarray: Image with transparent edges.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, edge_threshold1, edge_threshold2)

    # Convert edges to 3-channel image
    edge_overlay = np.zeros_like(image)
    edge_overlay[edges > 0] = (255, 255, 255)  # White edges

    # Apply Gaussian blur for smooth blending
    edge_overlay = cv2.GaussianBlur(edge_overlay, (3, 3), 0)

    # Blend edge overlay with the original image using fixed opacity
    transparent_edges = cv2.addWeighted(image, 1, edge_overlay, edge_opacity, 0)

    return transparent_edges

def sharpen_and_brighten_edges(image, edge_opacity):
    """
    If no face is detected, apply sharpening and brighten the edges with a fixed opacity.

    Parameters:
        image (numpy.ndarray): Input image.
        edge_opacity (float): Fixed opacity for edge blending.

    Returns:
        numpy.ndarray: Image with sharpened and brightened edges.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect edges using Canny
    edges = cv2.Canny(gray, 50, 150)

    # Convert edges to 3-channel image
    edge_mask = cv2.merge([edges, edges, edges])

    # Apply Gaussian blur for smoothness
    edge_mask = cv2.GaussianBlur(edge_mask, (3, 3), 0)

    # Erode edges slightly to keep them thin
    kernel = np.ones((2, 2), np.uint8)
    edge_mask = cv2.erode(edge_mask, kernel, iterations=1)

    # Blend edges with the original image using fixed opacity
    transparent_edges = cv2.addWeighted(image, 1, edge_mask, edge_opacity, 0)

    return transparent_edges

def process_images_with_canny(input_base_folder, output_base_folder, resolutions, edge_threshold1=50, edge_threshold2=150):
    """
    Process images by applying Canny edge detection with fixed opacity.
    If no face is detected, sharpen and brighten edges with the same opacity.

    Parameters:
        input_base_folder (str): Path to the base input folder.
        output_base_folder (str): Path to the base output folder.
        resolutions (list): List of resolutions to process.
        edge_threshold1 (int): First threshold for Canny edge detection.
        edge_threshold2 (int): Second threshold for Canny edge detection.
    """
    for filename in os.listdir(os.path.join(input_base_folder, resolutions[0]))[:50]:
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            continue

        # Generate a fixed opacity value per image
        

        for res in resolutions:
            input_folder = os.path.join(input_base_folder, res)
            output_folder = os.path.join(output_base_folder, res)

            if not os.path.exists(input_folder):
                print(f"Error: Input folder '{input_folder}' does not exist.")
                continue

            os.makedirs(output_folder, exist_ok=True)

            file_path = os.path.join(input_folder, filename)
            image = cv2.imread(file_path)
            if image is None:
                continue

            # Convert image to grayscale for face detection
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Detect faces
            faces = detector(gray)

            if len(faces) > 0:
                edge_opacity = random.uniform(0.2, 0.5)  # Adjust range as needed
                # Apply Canny edge detection with fixed opacity
                processed_image = apply_canny_edge_detection(image, edge_opacity, edge_threshold1, edge_threshold2)
            else:
                edge_opacity = random.uniform(0.4, 0.7)  # Adjust range as needed
                # Apply sharpening and edge brightening with fixed opacity
                processed_image = sharpen_and_brighten_edges(image, edge_opacity)

            # Modify filename to indicate edge detection
            name, ext = os.path.splitext(filename)
            new_filename = f"{name}_canny{ext}"
            output_path = os.path.join(output_folder, new_filename)

            cv2.imwrite(output_path, processed_image)
            print(f"Saved processed image: {output_path}")

    print("Processing complete.")

# Define input and output folder paths
input_base_folder = 'datasets/2 celebdf-resized/Celeb-real'
output_base_folder = 'datasets/3.3 celebdf-boundary_splicing/Celeb-real'
resolutions = ["64x64", "128x128", "256x256"]

# Process images with consistent opacity across resolutions
process_images_with_canny(input_base_folder, output_base_folder, resolutions, edge_threshold1=50, edge_threshold2=150)

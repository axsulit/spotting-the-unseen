import cv2
import dlib
import os
import numpy as np

# Load Dlib face detector
detector = dlib.get_frontal_face_detector()

def apply_canny_edge_detection(image, edge_opacity, edge_threshold1=50, edge_threshold2=150):
    """
    Apply Canny Edge Detection with a fixed opacity for consistency.

    Parameters:
        image (numpy.ndarray): Input image.
        edge_opacity (float): Opacity for edge blending.
        edge_threshold1 (int): First threshold for Canny edge detection.
        edge_threshold2 (int): Second threshold for Canny edge detection.

    Returns:
        numpy.ndarray: Image with blended edge detection.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, edge_threshold1, edge_threshold2)

    # Convert edges to 3-channel image
    edge_overlay = np.zeros_like(image)
    edge_overlay[edges > 0] = (255, 255, 255)  # White edges

    # Apply Gaussian blur for smooth blending
    edge_overlay = cv2.GaussianBlur(edge_overlay, (3, 3), 0)

    # Blend edge overlay with the original image using fixed opacity
    processed_image = cv2.addWeighted(image, 1, edge_overlay, edge_opacity, 0)

    return processed_image

def sharpen_and_brighten_edges(image, edge_opacity):
    """
    If no face is detected, apply sharpening and brighten the edges.

    Parameters:
        image (numpy.ndarray): Input image.
        edge_opacity (float): Opacity for edge blending.

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
    processed_image = cv2.addWeighted(image, 1, edge_mask, edge_opacity, 0)

    return processed_image

def process_images(input_folder, output_base_folder, edge_opacity_levels, edge_threshold1=50, edge_threshold2=150):
    """
    Process images by applying Canny edge detection with fixed intensity levels.

    Parameters:
        input_folder (str): Path to the input folder.
        output_base_folder (str): Path to the output folder.
        edge_opacity_levels (list): List of fixed opacity levels for boundary splicing.
        edge_threshold1 (int): First threshold for Canny edge detection.
        edge_threshold2 (int): Second threshold for Canny edge detection.
    """
    if not os.path.exists(input_folder):
        print(f"Error: Input folder '{input_folder}' does not exist.")
        return

    for filename in os.listdir(input_folder):
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            continue

        # Load image
        input_path = os.path.join(input_folder, filename)
        image = cv2.imread(input_path)
        if image is None:
            continue

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = detector(gray)

        for opacity in edge_opacity_levels:
            opacity_scaled = opacity / 100.0  # Convert to float (e.g., 20 → 0.2)

            if len(faces) > 0:
                # Apply Canny edge detection
                processed_image = apply_canny_edge_detection(image, opacity_scaled, edge_threshold1, edge_threshold2)
            else:
                # Apply sharpening and edge brightening
                processed_image = sharpen_and_brighten_edges(image, opacity_scaled)

            # Define folder structure: output_base_folder/intensity_level/
            intensity_folder = os.path.join(output_base_folder, str(opacity))
            os.makedirs(intensity_folder, exist_ok=True)

            # Modify filename to indicate boundary splicing intensity
            name, ext = os.path.splitext(filename)
            new_filename = f"{name}_canny_{opacity}.jpg"
            output_path = os.path.join(intensity_folder, new_filename)

            cv2.imwrite(output_path, processed_image)
            print(f"Saved: {output_path}")

    print("Processing complete.")

# Define input and output folder paths
input_folder = 'datasets/2 celebdf-resized/Celeb-synthesis/256x256'
output_base_folder = 'datasets/3.3 celebdf-boundary_splicing/Celeb-synthesis'

# Define fixed intensity levels (in percentage form)
edge_opacity_levels = [20, 40, 60, 80]

# Process images and save them in intensity-based folders
process_images(input_folder, output_base_folder, edge_opacity_levels, edge_threshold1=50, edge_threshold2=150)

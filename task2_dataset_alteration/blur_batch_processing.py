import cv2
import os

def blur_images(input_folder, output_folder, kernel_size=(15, 15)):
    # Check if input folder exists
    if not os.path.exists(input_folder):
        print(f"Error: Input folder '{input_folder}' does not exist.")
        return

<<<<<<< Updated upstream
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
=======
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
    blur_types = ["gaussian", "median", "bilateral"] # primary experiment
    # blur_types = ["median", "bilateral"] # secondary experiment
>>>>>>> Stashed changes

    # Process each file in the input folder
    for filename in os.listdir(input_folder):
        file_path = os.path.join(input_folder, filename)

        # Check if the file is an image
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            print(f"Skipping non-image file: {filename}")
            continue

        # Load the image
        image = cv2.imread(file_path)
        if image is None:
            print(f"Error: Could not load image '{filename}'. Skipping.")
            continue

        # Apply Gaussian blur
        blurred_image = cv2.GaussianBlur(image, kernel_size, 0)

        # Save the blurred image to the output folder
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, blurred_image)
        print(f"Blurred image saved: {output_path}")

    print("Processing complete.")

# Define input and output folder paths
<<<<<<< Updated upstream
input_folder = 'path_to_input_folder'  # Replace
output_folder = 'path_to_output_folder'  # Replace
=======
input_folder = r'D:\ACADEMICS\THESIS\Datasets\FF\c40\preprocessed_frames\originalFrames\alterations\Resized\youtube\256x256'
output_base_folder = r'D:\ACADEMICS\THESIS\Datasets\FF\c40\preprocessed_frames\originalFrames\alterations\Blur\youtube'
>>>>>>> Stashed changes

# Call the function to blur images
blur_images(input_folder, output_folder, kernel_size=(15, 15))
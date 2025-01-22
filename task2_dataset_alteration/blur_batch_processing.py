import cv2
import os

def blur_images(input_folder, output_folder, kernel_size=(15, 15)):
    # Check if input folder exists
    if not os.path.exists(input_folder):
        print(f"Error: Input folder '{input_folder}' does not exist.")
        return

    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

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
input_folder = 'path_to_input_folder'  # Replace
output_folder = 'path_to_output_folder'  # Replace

# Call the function to blur images
blur_images(input_folder, output_folder, kernel_size=(15, 15))
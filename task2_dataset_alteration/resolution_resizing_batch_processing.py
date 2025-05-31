import cv2
import os

def resize_images(input_folder, output_folder, sizes=[(512, 512), (256, 256), (128, 128), (64, 64)]):
    # Check if input folder exists
    if not os.path.exists(input_folder):
        print(f"Error: Input folder '{input_folder}' does not exist.")
        return

    # Create separate folders for each size
    for size in sizes:
        size_folder = os.path.join(output_folder, f"{size[0]}x{size[1]}")
        os.makedirs(size_folder, exist_ok=True)

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

        # Check if the input image is 512x512
        # if image.shape[:2] != (512, 512):
        #     print(f"Skipping '{filename}' - not 512x512")
        #     continue

        # Resize and save the images in their respective folders
        for size in sizes:
            resized_image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
            size_folder = os.path.join(output_folder, f"{size[0]}x{size[1]}")
            output_path = os.path.join(size_folder, filename)
            cv2.imwrite(output_path, resized_image)
            print(f"Resized {filename} to {size} and saved in {size_folder}")

    print("Processing complete.")

# Define input and output folder paths
<<<<<<< Updated upstream
input_folder = r'datasets\celebdf-preprocessed-cropped\YouTube-real'  # Replace with your actual input folder path
output_folder = r'datasets\celebdf-resized'  # Replace with your actual output folder path
=======
input_folder = r'D:\ACADEMICS\THESIS\Datasets\WDF\WildDeepfake\final_split\train\real'  # Replace with your actual input folder path
output_folder = r'D:\ACADEMICS\THESIS\Datasets\WDF\WildDeepfake\alterations\resized\train'  # Replace with your actual output folder path
>>>>>>> Stashed changes

# Call the function to resize images
resize_images(input_folder, output_folder)

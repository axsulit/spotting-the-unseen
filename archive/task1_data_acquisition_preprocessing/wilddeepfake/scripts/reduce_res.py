import cv2
import os

def resize_images(input_folders, output_base_folder, sizes=[(512, 512), (256, 256), (128, 128), (64, 64)]):
    # Process each input folder (test/train/val)
    for input_folder in input_folders:
        # Get the split name (test/train/val) from the input folder path
        split_name = os.path.basename(input_folder)
        print(f"\nProcessing {split_name} split...")
        
        # Check if input folder exists
        if not os.path.exists(input_folder):
            print(f"Error: Input folder '{input_folder}' does not exist.")
            continue
        
        # Process fake and real subfolders
        for subfolder in ['fake', 'real']:
            subfolder_path = os.path.join(input_folder, subfolder)
            if not os.path.exists(subfolder_path):
                print(f"Warning: {subfolder} folder not found in {split_name}")
                continue
                
            print(f"Processing {subfolder} images...")
            
            # Create output folders for each size
            for size in sizes:
                size_folder = os.path.join(output_base_folder, f"{size[0]}x{size[1]}", split_name, subfolder)
                os.makedirs(size_folder, exist_ok=True)

            # Process each file in the subfolder
            for filename in os.listdir(subfolder_path):
                file_path = os.path.join(subfolder_path, filename)

                # Check if the file is an image
                if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    print(f"Skipping non-image file: {filename}")
                    continue

                # Load the image
                image = cv2.imread(file_path)
                if image is None:
                    print(f"Error: Could not load image '{filename}'. Skipping.")
                    continue

                # # Check if the input image is 512x512
                # if image.shape[:2] != (512, 512):
                #     print(f"Skipping '{filename}' - not 512x512")
                #     continue

                # Resize and save the images in their respective folders
                for size in sizes:
                    resized_image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
                    size_folder = os.path.join(output_base_folder, f"{size[0]}x{size[1]}", split_name, subfolder)
                    output_path = os.path.join(size_folder, filename)
                    cv2.imwrite(output_path, resized_image)
                    print(f"Resized {filename} to {size} and saved in {size_folder}")

    print("\nProcessing complete.")

# Define input and output folder paths
base_input_path = r'D:\.THESIS\WildDeepfake\final_split'
input_folders = [
    os.path.join(base_input_path, 'test'),
    os.path.join(base_input_path, 'train'),
    os.path.join(base_input_path, 'val')
]
output_base_folder = r'D:\.THESIS\WildDeepfake\01_wdf_fake_final_resize'

# Create the main output folder if it doesn't exist
os.makedirs(output_base_folder, exist_ok=True)

# Call the function to resize images
resize_images(input_folders, output_base_folder)

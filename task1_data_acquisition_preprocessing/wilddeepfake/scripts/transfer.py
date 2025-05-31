import os
import shutil
from pathlib import Path

def transfer_images(source_folder, destination_folder, file_extensions=('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
    """
    Transfer images from source folder to destination folder, maintaining the train/test/val structure
    and only moving images from the fake subfolders.
    
    Args:
        source_folder (str): Path to the source folder containing images
        destination_folder (str): Path to the destination folder
        file_extensions (tuple): Tuple of file extensions to transfer (default: common image formats)
    """
    # Convert paths to Path objects for better handling
    source_path = Path(source_folder)
    dest_path = Path(destination_folder)
    
    # Create destination folder if it doesn't exist
    dest_path.mkdir(parents=True, exist_ok=True)
    
    # Check if source folder exists
    if not source_path.exists():
        print(f"Error: Source folder '{source_folder}' does not exist.")
        return
    
    # Counter for transferred files
    total_transferred = 0
    
    # Process each subfolder (train/test/val)
    for split in ['train', 'test', 'val']:
        split_source = source_path / split
        split_dest = dest_path / split
        
        # Skip if source split folder doesn't exist
        if not split_source.exists():
            print(f"Warning: {split} folder not found in source directory. Skipping...")
            continue
            
        # Create destination split folder
        split_dest.mkdir(parents=True, exist_ok=True)
        
        print(f"\nProcessing {split} split...")
        split_transferred = 0
        
        # Process fake folder
        fake_source = split_source / 'fake'
        if not fake_source.exists():
            print(f"Warning: fake folder not found in {split} directory. Skipping...")
            continue
            
        # Process each file in the fake folder
        for file_path in fake_source.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in file_extensions:
                try:
                    # Create destination file path
                    dest_file = split_dest / file_path.name
                    
                    # Move the file
                    shutil.move(str(file_path), str(dest_file))
                    split_transferred += 1
                    print(f"Moved: {split}/fake/{file_path.name}")
                    
                except Exception as e:
                    print(f"Error moving {split}/fake/{file_path.name}: {str(e)}")
        
        total_transferred += split_transferred
        print(f"Completed {split} split. Moved {split_transferred} files from fake folder.")
    
    print(f"\nTransfer complete. Total files moved: {total_transferred}")

if __name__ == "__main__":
    # Example usage
    source_folder = r"D:\.THESIS\WildDeepfake\01_wdf_fake_final_resize\512x512"  # Replace with your source folder path
    destination_folder = r"D:\.THESIS\WildDeepfake\wdf_final_fake\04_wdf_fake_512x512"  # Replace with your destination folder path
    
    transfer_images(source_folder, destination_folder) 
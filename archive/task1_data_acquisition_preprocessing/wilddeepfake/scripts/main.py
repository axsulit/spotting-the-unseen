# This file serves as the entry point for the face extraction process. It processes videos from the dataset and extracts frontal faces from them.

import os
import yaml
import shutil
from video_processor import process_image
from video_processor import process_video
from utils import create_directory

# Load configuration
with open("config.yaml", "r") as ymlfile:
    config = yaml.safe_load(ymlfile)

# Get configuration parameters (edit config.yaml)
dataset_root = config["dataset_root"]
base_output_dir = config["output_dir"]
rejects_dir = config["rejects_dir"]
process_category = config["process_category"]  # 'real' or 'fake'
process_subset = config["process_subset"]  # 'train' or 'test'
frame_interval = config["frame_interval"]
yaw_threshold = config["yaw_threshold"]
pitch_threshold = config["pitch_threshold"]
roll_threshold = config["roll_threshold"]

# Construct paths
input_dir = os.path.join(dataset_root, process_category, f"{process_category}_{process_subset}")
output_dir = os.path.join(base_output_dir, process_category, f"{process_category}_{process_subset}")
reject_output_dir = os.path.join(rejects_dir, process_category)

print(f"Processing {process_category}_{process_subset} data...")

# Check if the input directory exists
if not os.path.exists(input_dir):
    print(f"Error: Input directory not found: {input_dir}")
    exit(1)

# Create output directories
create_directory(output_dir)
create_directory(reject_output_dir)

# Function to process images in a directory and its subdirectories
def process_directory(current_dir, current_output_dir, current_reject_dir):
    # Create corresponding output directory structure
    create_directory(current_output_dir)
    
    # Get all items in the current directory
    items = os.listdir(current_dir)
    
    for item in items:
        item_path = os.path.join(current_dir, item)
        
        # If it's a directory, process it recursively
        if os.path.isdir(item_path):
            new_output_dir = os.path.join(current_output_dir, item)
            new_reject_dir = os.path.join(current_reject_dir, item)
            process_directory(item_path, new_output_dir, new_reject_dir)
        
        # If it's an image file, process it with process_image
        elif item.lower().endswith(('.jpg', '.jpeg', '.png')):
            print(f"Processing image: {item_path}")
            
            # Process the image using the image processor function
            is_frontal = process_image(item_path, current_output_dir, frame_interval, 
                                      yaw_threshold, pitch_threshold, roll_threshold)
            
            # If not frontal, simply log and continue to next image
            if not is_frontal:
                print(f"Image rejected: {item_path}")
                
        # If it's a video file, process it with process_video
        elif item.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            print(f"Processing video: {item_path}")
            process_video(item_path, current_output_dir, frame_interval, 
                         yaw_threshold, pitch_threshold, roll_threshold)

# Start processing from the root directory
process_directory(input_dir, output_dir, reject_output_dir)

print(f"Processing complete. Processed images saved to: {output_dir}")
print(f"Rejected images saved to: {reject_output_dir}")

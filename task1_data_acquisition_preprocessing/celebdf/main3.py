# This file serves as the entry point for the face extraction process. It processes videos from the dataset and extracts frontal faces from them.

import os
import yaml
from video_processor import process_video
from utils import create_directory

# Load configuration
with open("task1_data_acquisition_preprocessing/celebdf/config.yaml", "r") as ymlfile:
    config = yaml.safe_load(ymlfile)

# Get configuration parameters (edit config.yaml)
dataset_root = "datasets\\celebdf-v2-unaltered\\YouTube-real"
base_output_dir = "datasets\\celebdf-preprocessed-cropped\\YouTube-real"
subset = "real"
# manipulation_types = ["DeepFakeDetection", "Deepfakes", "Face2Face", "FaceShifter", "FaceSwap", "NeuralTextures",]
manipulation_type = "celeb-real"
frame_interval = config["frame_interval"]
yaw_threshold = config["yaw_threshold"]
pitch_threshold = config["pitch_threshold"]
roll_threshold = config["roll_threshold"]

print(f"Processing {manipulation_type}...")

# Construct the path to the video directory (edit nyo nalang for your own dataset)
#video_dir = os.path.join(dataset_root, manipulation_type, "celeb", "videos")
video_dir = os.path.join(dataset_root)

# Check if the video directory exists
if not os.path.exists(video_dir):
    print(f"Skipping {manipulation_type}, directory not found: {video_dir}")

# Create output directory
output_dir = os.path.join(base_output_dir, manipulation_type)
create_directory(output_dir)

# Get list of video files in the directory
video_files = [f for f in os.listdir(video_dir) if f.endswith(('.mp4', '.avi', '.mov'))]

# Loop through each video file and process it
for video_file in video_files:
    video_path = os.path.join(video_dir, video_file)
    print(f"Processing video: {video_path}")
    process_video(video_path, output_dir, frame_interval, yaw_threshold, pitch_threshold, roll_threshold)

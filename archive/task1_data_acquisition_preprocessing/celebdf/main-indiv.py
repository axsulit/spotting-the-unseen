import os
import yaml
from video_processor import process_video
from utils import create_directory

# Load configuration
with open("task1_data_acquisition_preprocessing/celebdf/config.yaml", "r") as ymlfile:
    config = yaml.safe_load(ymlfile)

base_output_dir = config["output_dir"]

# Enter the full video path manually
video_path = r"datasets\celebdf-v2-unaltered\Celeb-real\id0_0000.mp4"

# Define frame processing parameters
frame_interval = config["frame_interval"]
yaw_threshold = config["yaw_threshold"]
pitch_threshold = config["pitch_threshold"]
roll_threshold = config["roll_threshold"]

# Check if the video file exists
if not os.path.exists(video_path):
    print(f"Error: The specified video file does not exist: {video_path}")
    exit()

# Extract manipulation type from the path (optional, for organized output)
manipulation_type = os.path.basename(os.path.dirname(os.path.dirname(video_path)))  # Extracts "DeepFakeDetection"
output_dir = os.path.join(base_output_dir, manipulation_type)

# Create output directory
create_directory(output_dir)

# Process the specified video
print(f"Processing single video: {video_path}")
process_video(video_path, output_dir, frame_interval, yaw_threshold, pitch_threshold, roll_threshold)

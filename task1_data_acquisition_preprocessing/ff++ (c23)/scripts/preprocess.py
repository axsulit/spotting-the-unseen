import cv2
import os
import yaml

# Load configuration
with open("config.yaml", "r") as ymlfile:
    config = yaml.safe_load(ymlfile)

dataset_root = config["dataset_root"]
output_dir = config["output_dir"]

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Define specific subset and manipulation type
subset = "manipulated"
manipulation_types = [
    "DeepFakeDetection",
    "Deepfakes",
    "Face2Face",
    "FaceShifter",
    "FaceSwap",
    "NeuralTextures",
]

# Loop through manipulation types
for manipulation_type in manipulation_types:
    print(f"Processing {manipulation_type}...")

    # Construct the video directory path (this is to access the different manipulated)
    if subset == "original":
        video_dir = os.path.join(dataset_root, "original_sequences", "videos")
    elif subset == "manipulated":
        video_dir = os.path.join(dataset_root, manipulation_type, "c23", "videos")
    else:
        raise ValueError("Invalid subset specified. Must be 'original' or 'manipulated'.")

# Verify directory existence
    if not os.path.exists(video_dir):
        raise FileNotFoundError(f"Directory not found: {video_dir}. Check config.yaml: {video_dir}")

    # List video files
    video_files = [f for f in os.listdir(video_dir) if f.endswith(('.mp4', '.avi', '.mov'))]

    # Process each video for the current manipulation type
    if video_files:
        for video_file in video_files:
            video_path = os.path.join(video_dir, video_file)
            print(f"Processing video: {video_path}")

            # Open video file
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Error: Could not open video file: {video_path}")
            
            # Initialize counter for the number of frames processed
            frame_count = 0

            # Loop through each frame in the video
            while cap.isOpened():
                ret, frame = cap.read() # Read a frame, ret is a boolean value (true if frame is read correctly)
                
                # If a frame is read, increment the frame count
                if ret:
                    frame_count += 1

                    # Example: Save a frame (every 100th frame)
                    # if frame_count % 100 == 0:
                    #     frame_name = f"{manipulation_type}_{video_file[:-4]}_frame_{frame_count}.jpg" # Include manipulation type
                    #     frame_path = os.path.join(output_dir, frame_name)
                    #     cv2.imwrite(frame_path, frame)
                    #     print(f"Saved frame: {frame_path}")
                    

                    # ... Your processing code here ...
                    # Example: Print frame dimensions
                    height, width, channels = frame.shape
                    print(f"Frame {frame_count}: {width}x{height}x{channels}")

                # If no frame is read, break the loop
                else:
                    break

            # Release the video capture object and close all windows
            cap.release()
            cv2.destroyAllWindows()
            print(f"Video {video_file} processed. Total frames: {frame_count}")

    # If no video files found
    else:
        print(f"No video files found for {manipulation_type} in {video_dir}")



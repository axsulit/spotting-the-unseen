"""
Universal face extractor script for processing datasets containing videos 
or image folders. Extracts and saves frontal faces using facial landmark 
estimation and pose filtering.

Supported formats:
- Videos: .mp4, .avi, .mov, .mkv
- Images: .png, .jpg, .jpeg
"""

import argparse
from pathlib import Path
from utils.video_processor import process_video, process_image_folder
from utils.utils import create_directory
from utils.filter_frontal_faces import filter_non_frontal_faces

# Default parameters for face filtering
DEFAULT_FRAME_INTERVAL = 10
DEFAULT_YAW_THRESHOLD = 30
DEFAULT_PITCH_THRESHOLD = 30
DEFAULT_ROLL_THRESHOLD = 30

VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv")
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")

# Argument parsing
parser = argparse.ArgumentParser(description="Universal face extractor for video or image-folder datasets.")
parser.add_argument("dataset_root", type=str, help="Path to dataset root")
parser.add_argument("output_root", type=str, help="Path to output root")
args = parser.parse_args()

dataset_root = Path(args.dataset_root)
output_root = Path(args.output_root)

# Validate input path
if not dataset_root.exists():
    print(f"❌ Dataset root does not exist: {dataset_root}")
    exit(1)

print(f"📂 Scanning: {dataset_root}")

# Walk through dataset and process directories
for folder in dataset_root.rglob("*"):
    if not folder.is_dir():
        continue

    files = list(folder.iterdir())
    video_files = [f for f in files if f.suffix.lower() in VIDEO_EXTENSIONS][:5]
    image_files = [f for f in files if f.suffix.lower() in IMAGE_EXTENSIONS][:5]

    # Skip empty folders
    if not video_files and not image_files:
        continue

    # Prepare corresponding output directory
    rel_path = folder.relative_to(dataset_root)
    output_dir = output_root / rel_path
    create_directory(output_dir)

    # Process videos in folder
    if video_files:
        for video_path in video_files:
            print(f"🎬 Processing video: {video_path}")
            process_video(
                str(video_path),
                str(output_dir),
                frame_interval=DEFAULT_FRAME_INTERVAL,
                yaw_threshold=DEFAULT_YAW_THRESHOLD,
                pitch_threshold=DEFAULT_PITCH_THRESHOLD,
                roll_threshold=DEFAULT_ROLL_THRESHOLD,
            )

    # Process image folders (e.g., WildDeepFake)
    elif image_files:
        print(f"🖼️ Processing image folder: {folder}")
        process_image_folder(
            image_folder=str(folder),
            output_folder=str(output_dir),
            frame_interval=DEFAULT_FRAME_INTERVAL,
            yaw_threshold=DEFAULT_YAW_THRESHOLD,
            pitch_threshold=DEFAULT_PITCH_THRESHOLD,
            roll_threshold=DEFAULT_ROLL_THRESHOLD,
        )

    # Filter out non-frontal faces from output
    filter_non_frontal_faces(output_dir, "preprocess/utils/shape_predictor_68_face_landmarks.dat")

print("✅ Done extracting faces.")

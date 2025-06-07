import argparse
from pathlib import Path
from utils.video_processor import process_video
from utils.video_processor import process_image_folder
from utils.utils import create_directory
from utils.filter_frontal_faces import filter_non_frontal_faces

DEFAULT_FRAME_INTERVAL = 10
DEFAULT_YAW_THRESHOLD = 30
DEFAULT_PITCH_THRESHOLD = 30
DEFAULT_ROLL_THRESHOLD = 30
VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv")
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")

parser = argparse.ArgumentParser(description="Universal face extractor for video or image-folder datasets.")
parser.add_argument("dataset_root", type=str, help="Path to dataset root")
parser.add_argument("output_root", type=str, help="Path to output root")
args = parser.parse_args()

dataset_root = Path(args.dataset_root)
output_root = Path(args.output_root)

if not dataset_root.exists():
    print(f"❌ Dataset root does not exist: {dataset_root}")
    exit(1)

print(f"📂 Scanning: {dataset_root}")

for folder in dataset_root.rglob("*"):
    if not folder.is_dir():
        continue

    files = list(folder.iterdir())
    video_files = [f for f in files if f.suffix.lower() in VIDEO_EXTENSIONS][:5]
    image_files = [f for f in files if f.suffix.lower() in IMAGE_EXTENSIONS][:5]

    # Skip empty folders
    if not video_files and not image_files:
        continue

    # Output directory mirrors input structure
    rel_path = folder.relative_to(dataset_root)
    output_dir = output_root / rel_path
    create_directory(output_dir)

    if video_files:
        # Process each video file in the folder
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
            
    elif image_files:
        # Treat folder as a sequence of frames (WDF)
        print(f"🖼️ Processing image folder: {folder}")
        process_image_folder(
            image_folder=str(folder),
            output_folder=str(output_dir),
            frame_interval=DEFAULT_FRAME_INTERVAL,
            yaw_threshold=DEFAULT_YAW_THRESHOLD,
            pitch_threshold=DEFAULT_PITCH_THRESHOLD,
            roll_threshold=DEFAULT_ROLL_THRESHOLD,
        )

    filter_non_frontal_faces(output_dir, "preprocess/utils/shape_predictor_68_face_landmarks.dat")
print("✅ Done extracting faces.")
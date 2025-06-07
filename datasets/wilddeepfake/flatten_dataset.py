"""
WildDeepfake Dataset Flattening Script

This script flattens the hierarchical structure of the WildDeepfake dataset into a more manageable
format. It processes the dataset by organizing images from various splits (real/fake, train/test)
into a flat directory structure while maintaining unique identifiers for each image.

The script expects a source directory containing the original WildDeepfake dataset structure
and creates a new destination directory with a flattened organization of all images.

Dataset Structure:
    Source: /path/to/source/
        ├── real_train/
        │   └── extracted/
        │       └── [tar_folders]/
        │           └── [sequence_folders]/
        │               └── [frame_images].png
        ├── real_test/
        ├── fake_train/
        └── fake_test/

    Destination: /path/to/destination/
        ├── real_train/
        │   └── [tar_name]_[sequence_name]_[frame_number].png
        ├── real_test/
        ├── fake_train/
        └── fake_test/

Usage:
    python flatten_dataset.py /path/to/source /path/to/destination

Author: [Your Name]
Date: [Current Date]
"""

import shutil
from pathlib import Path
import argparse

# Configure command line argument parser
parser = argparse.ArgumentParser(
    description="Flatten WildDeepfake dataset structure into a more manageable format.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument(
    "src_root",
    type=str,
    help="Path to the source dataset root folder containing the original WildDeepfake structure"
)
parser.add_argument(
    "dst_root",
    type=str,
    help="Path to the destination root folder where the flattened dataset will be created"
)
args = parser.parse_args()

# Define constants
SRC_ROOT = Path(args.src_root)
DST_ROOT = Path(args.dst_root)
SPLITS = ("real_train", "real_test", "fake_train", "fake_test")  # Dataset splits to process

# Create destination root directory if it doesn't exist
DST_ROOT.mkdir(parents=True, exist_ok=True)

# Process each split (real/fake, train/test)
for split in SPLITS:
    # Construct paths for source and destination
    src_extracted = SRC_ROOT / split / "extracted"
    dst_split = DST_ROOT / split
    dst_split.mkdir(parents=True, exist_ok=True)

    # Process each tar folder in the extracted directory
    for tar_folder in src_extracted.iterdir():
        if not tar_folder.is_dir():
            continue

        # Find the inner directory containing the actual content
        subdirs = [d for d in tar_folder.iterdir() if d.is_dir()]
        if not subdirs:
            continue

        # Handle different directory structures
        if len(subdirs) == 1:
            inner = subdirs[0]
        else:
            # Try to find a directory matching the tar folder name
            candidate = tar_folder / tar_folder.stem
            inner = candidate if candidate.exists() else subdirs[0]

        # Navigate to the content directory (real or fake)
        content_dir = inner / split.split("_")[0]  # "real" or "fake"
        if not content_dir.is_dir():
            content_dir = inner

        # Process each sequence folder
        for seq_folder in content_dir.iterdir():
            if not seq_folder.is_dir():
                continue

            # Process each image in the sequence
            for img in seq_folder.iterdir():
                if img.suffix.lower() != ".png":
                    continue

                try:
                    # Generate unique filename: tar_name_sequence_name_frame_number.png
                    frame_num = int(img.stem)
                    new_name = f"{tar_folder.name}_{seq_folder.name}_{frame_num:06d}{img.suffix}"
                    # Copy image to destination with new name
                    shutil.copy2(img, dst_split / new_name)
                except ValueError:
                    # Skip files that don't have numeric names
                    continue

print("✅ Dataset flattening complete. Output directory:", DST_ROOT)

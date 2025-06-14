"""
add_alterations.py

Applies a selected visual alteration (e.g. blur, noise, color mismatch) to all images
in the input directory and saves results to the output directory.

Usage:
    python add_alterations.py --type blur --input_dir ./dataset/original --output_dir ./dataset/blurred
"""

import argparse
from pathlib import Path
from preprocess.augmentations.apply_blur import batch_apply_blur 
from preprocess.augmentations.apply_noise import batch_apply_noise
from preprocess.augmentations.apply_splice import batch_apply_splice
from preprocess.augmentations.apply_resize import batch_apply_resize
from preprocess.augmentations.apply_color_mismatch import batch_apply_color_mismatch

ALTERATION_FUNCS = {
    "blur": batch_apply_blur,
    "noise": batch_apply_noise,
    "splice": batch_apply_splice,
    "resize": batch_apply_resize,
    "color": batch_apply_color_mismatch,
}

def process_images(alteration_type, input_dir, output_dir):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    alter_fn = ALTERATION_FUNCS.get(alteration_type)
    if alter_fn is None:
        raise ValueError(f"Unsupported alteration: {alteration_type}")

    if alteration_type == 'blur':
        alter_fn(input_dir, output_dir)
    elif alteration_type == 'noise':
        alter_fn(input_dir, output_dir)
    elif alteration_type == 'splice':
        alter_fn(input_dir, output_dir)
    elif alteration_type == 'resize':
        alter_fn(input_dir, output_dir)
    elif alteration_type == 'color':
        alter_fn(input_dir, output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply a visual alteration to a folder of images.")
    parser.add_argument("--type", required=True, choices=ALTERATION_FUNCS.keys(), help="Type of alteration")
    parser.add_argument("--input_dir", required=True, help="Input directory of images")
    parser.add_argument("--output_dir", required=True, help="Where to save altered images")
    args = parser.parse_args()

    process_images(args.type, args.input_dir, args.output_dir)


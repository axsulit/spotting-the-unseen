import cv2
import os
import numpy as np
from pathlib import Path

def apply_blur(image, blur_type, intensity):
    """Applies a specific blur type and intensity to an image."""
    if blur_type == "gaussian":
        kernel_size = max(3, int(intensity * min(image.shape[:2]) / 100))
        if kernel_size % 2 == 0:
            kernel_size += 1
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

    elif blur_type == "median":
        kernel_size = max(3, int(intensity * min(image.shape[:2]) / 100))
        if kernel_size % 2 == 0:
            kernel_size += 1
        return cv2.medianBlur(image, kernel_size)

    elif blur_type == "bilateral":
        d = max(5, int(intensity * min(image.shape[:2]) / 100))
        sigma_color = int(intensity * 2.5)
        sigma_space = int(intensity * 2.5)
        return cv2.bilateralFilter(image, d, sigma_color, sigma_space)

    return image

def batch_apply_blur(input_folder, output_folder):
    """Applies blur to all images in input_folder and saves them in output_folder."""
    blur_levels = [5, 10, 15, 20, 25]
    blur_type="gaussian"

    for ksize in blur_levels:
        level_folder = os.path.join(output_folder, f"blur_{ksize}")
        os.makedirs(level_folder, exist_ok=True)

        for image_file in input_folder.glob("*"):
            if image_file.suffix.lower() not in [".png", ".jpg", ".jpeg"]:
                continue

            image = cv2.imread(str(image_file))
            if image is None:
                print(f"❌ Skipping unreadable: {image_file}")
                continue

            blurred = apply_blur(image, blur_type, ksize)

            out_name = f"{image_file.stem}_{blur_type}_{ksize}{image_file.suffix}"
            out_path = os.path.join(level_folder, out_name)
            cv2.imwrite(str(out_path), blurred)
            print(f"✅ Saved: {out_path}")
from pathlib import Path
import cv2
import dlib
import numpy as np
import os

# Load Dlib face detector
detector = dlib.get_frontal_face_detector()

def apply_canny_edge_detection(image, edge_opacity, edge_threshold1=50, edge_threshold2=150):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, edge_threshold1, edge_threshold2)

    edge_overlay = np.zeros_like(image)
    edge_overlay[edges > 0] = (255, 255, 255)

    edge_overlay = cv2.GaussianBlur(edge_overlay, (3, 3), 0)

    return cv2.addWeighted(image, 1, edge_overlay, edge_opacity, 0)

def sharpen_and_brighten_edges(image, edge_opacity):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    edge_mask = cv2.merge([edges, edges, edges])
    edge_mask = cv2.GaussianBlur(edge_mask, (3, 3), 0)
    edge_mask = cv2.erode(edge_mask, np.ones((2, 2), np.uint8), iterations=1)

    return cv2.addWeighted(image, 1, edge_mask, edge_opacity, 0)

def batch_apply_splice(input_folder, output_folder):
    input_folder = Path(input_folder)
    edge_opacity_levels = [20, 40, 60, 80]

    for image_file in input_folder.glob("*"):
        if image_file.suffix.lower() not in [".png", ".jpg", ".jpeg"]:
            continue

        image = cv2.imread(str(image_file))
        if image is None:
            print(f"❌ Skipping unreadable: {image_file}")
            continue

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        for opacity in edge_opacity_levels:
            opacity_scaled = opacity / 100.0

            if len(faces) > 0:
                processed_image = apply_canny_edge_detection(image, opacity_scaled)
            else:
                processed_image = sharpen_and_brighten_edges(image, opacity_scaled)

            level_folder = Path(output_folder) / f"splice_{opacity}"
            level_folder.mkdir(parents=True, exist_ok=True)

            out_name = f"{image_file.stem}_canny_{opacity}{image_file.suffix}"
            out_path = level_folder / out_name

            cv2.imwrite(str(out_path), processed_image)
            print(f"✅ Saved: {out_path}")

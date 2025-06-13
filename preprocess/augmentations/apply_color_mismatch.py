from pathlib import Path
import cv2
import dlib
import numpy as np
import random
from preprocess.utils.utils import load_predictor

# Load Dlib face detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = load_predictor()

def apply_partial_color_distortion(image, regions, intensity=50):
    image = image.astype(np.int16)
    for region in regions:
        x_min = min(point[0] for point in region)
        x_max = max(point[0] for point in region)
        y_min = min(point[1] for point in region)
        y_max = max(point[1] for point in region)

        for channel in range(3):  # BGR
            shift = random.randint(-intensity, intensity)
            image[y_min:y_max, x_min:x_max, channel] = np.clip(image[y_min:y_max, x_min:x_max, channel] + shift, 0, 255)

    return image.astype(np.uint8)

def get_facial_regions(landmarks):
    points = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(68)]

    regions = {
        "left_cheek": points[1:5],
        "right_cheek": points[12:16],
        "nose": points[27:36],
        "upper_forehead": points[17:27],
        "mid_forehead": points[19:24],
        "chin": points[5:12],
        "mouth_area": points[48:61],
        "jawline": points[0:17],
    }

    selected = random.sample(list(regions.values()), random.randint(1, 2))
    return selected

def batch_apply_color_mismatch(input_folder, output_folder, intensity=50):
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    target_folder = output_folder / "color_mismatch"
    target_folder.mkdir(parents=True, exist_ok=True)

    for image_file in input_folder.glob("*"):
        if image_file.suffix.lower() not in [".png", ".jpg", ".jpeg", ".bmp", ".tiff"]:
            continue

        image = cv2.imread(str(image_file))
        if image is None:
            print(f"❌ Skipping unreadable: {image_file.name}")
            continue

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        if len(faces) > 0:
            landmarks = predictor(gray, faces[0])
            regions = get_facial_regions(landmarks)
        else:
            h, w = image.shape[:2]
            region_w = random.randint(w // 6, w // 3)
            region_h = random.randint(h // 6, h // 3)
            x = random.randint(0, w - region_w)
            y = random.randint(0, h - region_h)
            regions = [[(x, y), (x + region_w, y), (x + region_w, y + region_h), (x, y + region_h)]]

        altered = apply_partial_color_distortion(image, regions, intensity)

        new_name = f"{image_file.stem}_color_mismatch{image_file.suffix}"
        out_path = target_folder / new_name
        cv2.imwrite(str(out_path), altered)
        print(f"✅ Saved: {out_path}")

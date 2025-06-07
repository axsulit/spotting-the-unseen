"""
This module contains logic for processing video and image files to detect 
and extract frontal faces using facial landmark estimation.
"""

import cv2
import os
import numpy as np
from pathlib import Path
from utils.face_detector import detect_faces, is_frontal_face
from utils.utils import resize, load_predictor


def process_video(
    video_path,
    output_dir,
    frame_interval,
    yaw_threshold,
    pitch_threshold,
    roll_threshold,
    predictor_path="preprocess/utils/shape_predictor_68_face_landmarks.dat"
):
    """
    Process a video file to extract and save frontal face images at specified frame intervals.

    Args:
        video_path (str or Path): Path to the input video file.
        output_dir (str or Path): Directory where cropped frontal faces will be saved.
        frame_interval (int): Process every nth frame (e.g., every 5th frame).
        yaw_threshold (float): Maximum allowed yaw angle in degrees.
        pitch_threshold (float): Maximum allowed pitch angle in degrees.
        roll_threshold (float): Maximum allowed roll angle in degrees.
        predictor_path (str or Path, optional): Path to the dlib shape predictor model. 
            Defaults to 'preprocess/utils/shape_predictor_68_face_landmarks.dat'.

    Returns:
        None
    """
    predictor = load_predictor(predictor_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Error: Could not open video file: {video_path}")
        return

    frame_count = 0
    saved_frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        if frame_count % frame_interval != 0:
            continue

        faces = detect_faces(frame)

        for face in faces:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            landmarks = predictor(gray, face)
            landmarks = np.array([[p.x, p.y] for p in landmarks.parts()])

            if is_frontal_face(landmarks, yaw_threshold, pitch_threshold, roll_threshold):
                x, y, w, h = face.left(), face.top(), face.width(), face.height()
                w = h = max(w, h)  # Ensure square crop

                cropped = frame[max(0, y):y + h, max(0, x):x + w]
                resized = resize(cropped, (256, 256))

                filename = f"{os.path.basename(video_path)[:-4]}_face_{frame_count}.jpg"
                out_path = os.path.join(output_dir, filename)
                cv2.imwrite(out_path, resized)
                print(f"✅ Saved frontal face from frame {frame_count} → {out_path}")
                saved_frame_count += 1

    cap.release()
    cv2.destroyAllWindows()
    print(f"🎞️ Finished {video_path} — Faces saved: {saved_frame_count}")


def process_image_folder(image_folder, output_folder, frame_interval, yaw_threshold, pitch_threshold, roll_threshold):
    """
    Process a folder of images to extract and save frontal faces at specified intervals.

    Args:
        image_folder (str or Path): Path to the folder containing input images.
        output_folder (str or Path): Directory to save the output cropped face images.
        frame_interval (int): Process every nth image (e.g., every 5th image).
        yaw_threshold (float): Maximum allowed yaw angle in degrees.
        pitch_threshold (float): Maximum allowed pitch angle in degrees.
        roll_threshold (float): Maximum allowed roll angle in degrees.

    Returns:
        None
    """
    import cv2
    from utils.utils import is_frontal, extract_face

    image_paths = sorted(
        [p for p in Path(image_folder).glob("*") if p.suffix.lower() in [".png", ".jpg"]],
        key=lambda p: int(p.stem) if p.stem.isdigit() else p.name
    )

    for idx, img_path in enumerate(image_paths):
        if idx % frame_interval != 0:
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        face = extract_face(img)
        if face is None:
            continue

        if not is_frontal(face, yaw_threshold, pitch_threshold, roll_threshold):
            continue

        out_name = f"{idx:06d}.png"
        out_path = Path(output_folder) / out_name
        cv2.imwrite(str(out_path), face)

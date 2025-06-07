# This file contains the logic for processing individual video files.

import cv2
import os
import numpy as np
from pathlib import Path
from utils.face_detector import detect_faces, is_frontal_face
from utils.utils import create_directory, resize, load_predictor

# Function to process a video file, detect frontal faces, and save them
def process_video(
    video_path,
    output_dir,
    frame_interval,
    yaw_threshold,
    pitch_threshold,
    roll_threshold,
    predictor_path="preprocess/utils/shape_predictor_68_face_landmarks.dat"
):
    # Load the shape predictor once per video
    predictor = load_predictor(predictor_path)

    # Open video
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
                w = h = max(w, h)  # Make it square

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
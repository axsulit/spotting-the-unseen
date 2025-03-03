import os
import random
import shutil
import albumentations as A
import cv2

# 🚀 Define dataset paths
real_types = {
    "actors": r"D:\.THESIS\datasets\0_unaltered\real\actors",
    "youtube": r"D:\.THESIS\datasets\0_unaltered\real\youtube",
}

fake_types = {
    "deepfakedetection": r"D:\.THESIS\datasets\0_unaltered\fake\DeepFakeDetection",
    "deepfakes": r"D:\.THESIS\datasets\0_unaltered\fake\Deepfakes",
    "face2face": r"D:\.THESIS\datasets\0_unaltered\fake\Face2Face",
    "faceshifter": r"D:\.THESIS\datasets\0_unaltered\fake\FaceShifter",
    "faceswap": r"D:\.THESIS\datasets\0_unaltered\fake\FaceSwap",
    "neuraltextures": r"D:\.THESIS\datasets\0_unaltered\fake\NeuralTextures",
}

output_dir = r"D:\.THESIS\datasets\final\01_ffc23_final_unaltered"

print("🚀 Starting dataset split...\n")

# 📌 Create output directories
for split in ["train", "test", "val"]:
    os.makedirs(os.path.join(output_dir, split, "real"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, split, "fake"), exist_ok=True)

# 📌 Function to group images by video
def group_by_video(type_dict):
    video_dict = {}
    for type_name, folder in type_dict.items():
        for file in os.listdir(folder):
            if file.endswith((".jpg", ".png")):
                video_name = "_".join(file.split("_")[:-2])  
                if video_name not in video_dict:
                    video_dict[video_name] = {"type": type_name, "files": []}
                video_dict[video_name]["files"].append(os.path.join(folder, file))
    return video_dict

# 📌 Group images by video
real_videos = group_by_video(real_types)
fake_videos = group_by_video(fake_types)

# 📌 Count total images
def count_images(video_dict):
    return sum(len(v["files"]) for v in video_dict.values())

real_image_count = count_images(real_videos)
fake_image_count = count_images(fake_videos)

print(f"📊 Initial Dataset Summary:")
print(f"   🔹 Real Images: {real_image_count}")
print(f"   🔹 Fake Images: {fake_image_count}\n")

# 📌 Function to move images with strict limit enforcement
def move_images(video_list, video_dict, label, split, target_count):
    dataset_path = os.path.join(output_dir, split, label)
    os.makedirs(dataset_path, exist_ok=True)

    total_images = 0
    total_files = min(target_count, sum(len(video_dict[video]["files"]) for video in video_list if video in video_dict))

    print(f"📂 Moving {total_files} {label} images to {split}...")

    for video in video_list:
        if video in video_dict:
            for image_path in video_dict[video]["files"]:
                if total_images >= target_count:
                    break  # 🚨 Hard stop when `target_count` is reached
                
                shutil.copy(image_path, os.path.join(dataset_path, os.path.basename(image_path)))
                total_images += 1

                # ✅ Ensure real-time correct progress tracking
                if total_images % 500 == 0 or total_images == target_count:
                    print(f"   → {total_images}/{target_count} {label} images moved to {split}")

    print(f"✅ Completed moving {total_images}/{target_count} {label} images to {split}.\n")

# 📌 Function to enforce the 70%-15%-15% split
def enforce_strict_split():
    train_ratio, test_ratio, val_ratio = 0.7, 0.15, 0.15
    train_size = int(fake_image_count * train_ratio)
    test_size = int(fake_image_count * test_ratio)
    val_size = int(fake_image_count * val_ratio)

    for split, target_count in zip(["train", "test", "val"], [train_size, test_size, val_size]):
        real_folder = os.path.join(output_dir, split, "real")
        fake_folder = os.path.join(output_dir, split, "fake")

        real_files = sorted(os.listdir(real_folder))
        fake_files = sorted(os.listdir(fake_folder))

        min_count = min(len(real_files), len(fake_files), target_count)

        real_files = real_files[:min_count]
        fake_files = fake_files[:min_count]

        for excess in os.listdir(real_folder):
            if excess not in real_files:
                os.remove(os.path.join(real_folder, excess))

        for excess in os.listdir(fake_folder):
            if excess not in fake_files:
                os.remove(os.path.join(fake_folder, excess))

# 🚀 Move files with strict image limits
for split, target_count in zip(["train", "test", "val"], 
                               [int(fake_image_count * 0.7), int(fake_image_count * 0.15), int(fake_image_count * 0.15)]):
    move_images(real_videos.keys(), real_videos, "real", split, target_count)
    move_images(fake_videos.keys(), fake_videos, "fake", split, target_count)

# 🚀 Enforce strict balance
enforce_strict_split()

# 📌 Print final dataset summary
def count_final_images():
    final_counts = {}
    for split in ["train", "test", "val"]:
        real_count = len(os.listdir(os.path.join(output_dir, split, "real")))
        fake_count = len(os.listdir(os.path.join(output_dir, split, "fake")))
        final_counts[split] = (real_count, fake_count)
    return final_counts

final_counts = count_final_images()
print("\n✅ Final Balanced Split Counts (Train 70% / Test 15% / Val 15%):")
for split, (real_count, fake_count) in final_counts.items():
    print(f"   {split.capitalize()} - Real: {real_count}, Fake: {fake_count}")

print("\n✅ Dataset split completed successfully! 🎉")

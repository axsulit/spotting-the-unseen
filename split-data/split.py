import os
import random
import shutil
import albumentations as A
import cv2

# 🚀 Define dataset paths
real_types = {
    "actors": r"D:\.THESIS\datasets\sample_data\real\actors",
    "youtube": r"D:\.THESIS\datasets\sample_data\real\youtube",
}

fake_types = {
    "deepfakedetection": r"D:\.THESIS\datasets\sample_data\fake\DeepFakeDetection",
    "deepfakes": r"D:\.THESIS\datasets\sample_data\fake\Deepfakes",
    "face2face": r"D:\.THESIS\datasets\sample_data\fake\Face2Face",
    "faceshifter": r"D:\.THESIS\datasets\sample_data\fake\FaceShifter",
    "faceswap": r"D:\.THESIS\datasets\sample_data\fake\FaceSwap",
    "neuraltextures": r"D:\.THESIS\datasets\sample_data\fake\NeuralTextures",
}

output_dir = r"D:\.THESIS\datasets\final\balanced_dataset"  # Final dataset directory

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
                video_name = "_".join(file.split("_")[:-2])  # Extract video identifier
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
print(f"   🔹 Fake Images: {fake_image_count}")

# 📌 Augmentation pipeline for oversampling
augmentation = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.GaussianBlur(p=0.2),
    A.HueSaturationValue(p=0.2),
    A.ISONoise(p=0.2),
])

# 📌 Function to apply augmentation
def augment_image(image_path):
    image = cv2.imread(image_path)
    return augmentation(image=image)["image"] if image is not None else None

# 📌 Oversampling real images to exactly match fake images
def oversample_real_images(real_videos, target_count):
    all_real_images = [img for v in real_videos.values() for img in v["files"]]
    real_images_needed = target_count - len(all_real_images)

    if real_images_needed <= 0:
        print("✅ No oversampling needed. Real dataset is already balanced.")
        return real_videos

    print(f"⚡ Oversampling: Need {real_images_needed} more real images.")
    extra_real_images = random.choices(all_real_images, k=real_images_needed)

    for i, img_path in enumerate(extra_real_images):
        video_name = "_".join(os.path.basename(img_path).split("_")[:-2])  
        new_filename = f"{video_name}_oversample_{i}.png"
        new_image_path = os.path.join(os.path.dirname(img_path), new_filename)

        augmented_image = augment_image(img_path)
        if augmented_image is not None:
            cv2.imwrite(new_image_path, augmented_image)
        else:
            shutil.copy(img_path, new_image_path)

        for v in real_videos.values():
            if img_path in v["files"]:
                v["files"].append(new_image_path)
                break

    return real_videos

# 🚀 Oversample real dataset to match fake dataset exactly
real_videos = oversample_real_images(real_videos, fake_image_count)

# 📌 Function to strictly enforce 70%-15%-15% split AND exact parity
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

        # Trim or copy files to match exact counts
        real_files = real_files[:min_count]
        fake_files = fake_files[:min_count]

        # Remove excess files
        for excess in os.listdir(real_folder):
            if excess not in real_files:
                os.remove(os.path.join(real_folder, excess))

        for excess in os.listdir(fake_folder):
            if excess not in fake_files:
                os.remove(os.path.join(fake_folder, excess))

# 📌 Move images to respective folders
def move_images(video_list, video_dict, label, split):
    dataset_path = os.path.join(output_dir, split, label)
    os.makedirs(dataset_path, exist_ok=True)

    for video in video_list:
        if video in video_dict:
            for image_path in video_dict[video]["files"]:
                filename = os.path.basename(image_path)
                shutil.copy(image_path, os.path.join(dataset_path, filename))

# ✅ Move files and enforce strict balance
for split in ["train", "test", "val"]:
    move_images(real_videos.keys(), real_videos, "real", split)
    move_images(fake_videos.keys(), fake_videos, "fake", split)

# 🚀 Enforce strict real vs. fake balance
enforce_strict_split()

# 📌 Print final actual file count summary
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

print("\n✅ Dataset split completed successfully!")

import os
import random
import shutil
import argparse
from pathlib import Path
import albumentations as A
import cv2

# ------------------------- ARGUMENT PARSING -------------------------

parser = argparse.ArgumentParser(description="Split and balance real/fake datasets under an alteration.")
parser.add_argument("--alteration_dir", required=True, help="Path to root folder of an altered dataset (with real/ and fake/)")
parser.add_argument("--output_dir", required=True, help="Where to write the split result (train/test/val)")
args = parser.parse_args()

alteration_root = Path(args.alteration_dir)
output_dir = Path(args.output_dir)

# ------------------------- DIRECTORY PREP -------------------------

for split in ["train", "test", "val"]:
    os.makedirs(output_dir / split / "real", exist_ok=True)
    os.makedirs(output_dir / split / "fake", exist_ok=True)

# ------------------------- DISCOVERY -------------------------

def discover_subfolders(root):
    subfolders = {}
    for dirpath, _, filenames in os.walk(root):
        if any(f.lower().endswith((".png", ".jpg", ".jpeg")) for f in filenames):
            key = Path(dirpath).relative_to(root).as_posix()
            subfolders[key] = dirpath
    return subfolders

real_types = discover_subfolders(alteration_root / "real")
fake_types = discover_subfolders(alteration_root / "fake")

# ------------------------- GROUP IMAGES -------------------------

def group_by_video(type_dict):
    video_dict = {}
    for type_name, folder in type_dict.items():
        for file in os.listdir(folder):
            if file.endswith((".jpg", ".png")):
                video_name = "_".join(file.split("_")[:-2])  # remove _frame_000123
                if video_name not in video_dict:
                    video_dict[video_name] = {"type": type_name, "files": []}
                video_dict[video_name]["files"].append(os.path.join(folder, file))
    return video_dict

real_videos = group_by_video(real_types)
fake_videos = group_by_video(fake_types)

def count_images(video_dict):
    return sum(len(v["files"]) for v in video_dict.values())

real_image_count = count_images(real_videos)
fake_image_count = count_images(fake_videos)

print(f"\n📊 Initial Counts — Real: {real_image_count}, Fake: {fake_image_count}\n")

# ------------------------- AUGMENTATION -------------------------

augmentation = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.GaussianBlur(p=0.2),
    A.HueSaturationValue(p=0.2),
    A.ISONoise(p=0.2),
])

def augment_image(image_path):
    image = cv2.imread(image_path)
    return augmentation(image=image)["image"] if image is not None else None

def oversample_real_images(real_videos, target_count):
    all_real_images = [img for v in real_videos.values() for img in v["files"]]
    needed = target_count - len(all_real_images)
    if needed <= 0:
        print("✅ No oversampling needed.\n")
        return real_videos

    print(f"⚡ Oversampling {needed} real images...")
    extras = random.choices(all_real_images, k=needed)

    for i, img_path in enumerate(extras):
        video_name = "_".join(os.path.basename(img_path).split("_")[:-2])
        new_name = f"{video_name}_aug_{i}.png"
        new_path = os.path.join(os.path.dirname(img_path), new_name)

        aug_img = augment_image(img_path)
        if aug_img is not None:
            cv2.imwrite(new_path, aug_img)
        else:
            shutil.copy(img_path, new_path)

        for v in real_videos.values():
            if img_path in v["files"]:
                v["files"].append(new_path)
                break

        if (i + 1) % 500 == 0 or (i + 1) == needed:
            print(f"   → {i+1}/{needed} real images oversampled.")

    print("✅ Oversampling complete!\n")
    return real_videos

real_videos = oversample_real_images(real_videos, fake_image_count)

# ------------------------- MOVE FUNCTION -------------------------

def move_images(video_list, video_dict, label, split, target_count):
    target_path = output_dir / split / label
    os.makedirs(target_path, exist_ok=True)

    total_images = 0
    max_images = min(target_count, sum(len(video_dict[v]["files"]) for v in video_list if v in video_dict))

    print(f"📂 Moving {max_images} {label} images to {split}...")

    for video in video_list:
        if video not in video_dict:
            continue
        for image_path in video_dict[video]["files"]:
            if total_images >= target_count:
                break
            shutil.copy(image_path, target_path / Path(image_path).name)
            total_images += 1
            if total_images % 500 == 0 or total_images == target_count:
                print(f"   → {total_images}/{target_count} {label} images moved")

    print(f"✅ Completed moving {total_images} {label} images to {split}.\n")

# ------------------------- ENFORCE BALANCE -------------------------

def enforce_strict_split():
    for split in ["train", "test", "val"]:
        real_folder = output_dir / split / "real"
        fake_folder = output_dir / split / "fake"

        real_files = sorted(os.listdir(real_folder))
        fake_files = sorted(os.listdir(fake_folder))
        min_count = min(len(real_files), len(fake_files))

        for excess in real_files[min_count:]:
            os.remove(real_folder / excess)
        for excess in fake_files[min_count:]:
            os.remove(fake_folder / excess)

# ------------------------- PERFORM SPLIT -------------------------

ratios = {"train": 0.7, "test": 0.15, "val": 0.15}
for split, ratio in ratios.items():
    count = int(fake_image_count * ratio)
    move_images(real_videos.keys(), real_videos, "real", split, count)
    move_images(fake_videos.keys(), fake_videos, "fake", split, count)

enforce_strict_split()

# ------------------------- SUMMARY -------------------------

def count_final_images():
    counts = {}
    for split in ["train", "test", "val"]:
        r = len(os.listdir(output_dir / split / "real"))
        f = len(os.listdir(output_dir / split / "fake"))
        counts[split] = (r, f)
    return counts

final_counts = count_final_images()
print("\n✅ Final Balanced Counts:")
for split, (r, f) in final_counts.items():
    print(f"   {split.capitalize()} - Real: {r}, Fake: {f}")

print("\n✅ Dataset split completed successfully! 🎉")

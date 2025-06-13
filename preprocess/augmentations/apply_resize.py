from pathlib import Path
import cv2

def batch_apply_resize(input_folder, output_folder):
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    sizes = [(512, 512), (128, 128), (64, 64)]

    for size in sizes:
        (output_folder / f"{size[0]}x{size[1]}").mkdir(parents=True, exist_ok=True)

    for image_file in input_folder.glob("*"):
        if image_file.suffix.lower() not in [".png", ".jpg", ".jpeg", ".bmp", ".tiff"]:
            print(f"⏭️ Skipping non-image file: {image_file.name}")
            continue

        image = cv2.imread(str(image_file))
        if image is None:
            print(f"❌ Could not load image: {image_file.name}")
            continue

        if image.shape[:2] != (512, 512):
            print(f"⚠️ Skipping {image_file.name} — not 512x512")
            continue

        for size in sizes:
            resized = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
            size_folder = output_folder / f"{size[0]}x{size[1]}"
            out_path = size_folder / image_file.name
            cv2.imwrite(str(out_path), resized)
            print(f"✅ {image_file.name} → {size[0]}x{size[1]}")

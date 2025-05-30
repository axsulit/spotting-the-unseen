import shutil
from pathlib import Path

# ─── CONFIG ────────────────────────────────────────────────────────────────
SRC_ROOT = Path(r"D:\ACADEMICS\THESIS\Datasets\WDF\WildDeepfake\deepfake_in_the_wild")
DST_ROOT = Path(r"D:\ACADEMICS\THESIS\Datasets\WDF\WildDeepfake\wdf_restructured")
DST_ROOT.mkdir(parents=True, exist_ok=True)

SPLITS = ("real_train", "real_test")
# ────────────────────────────────────────────────────────────────────────────

for split in SPLITS:
    src_extracted = SRC_ROOT / split 
    dst_split    = DST_ROOT / split
    dst_split.mkdir(parents=True, exist_ok=True)

    for tar_folder in src_extracted.iterdir():
        if not tar_folder.is_dir():
            continue  # skip files, only process folders

        # find the one subfolder under tar_folder that represents the extracted content
        subdirs = [d for d in tar_folder.iterdir() if d.is_dir()]
        if not subdirs:
            continue
        # if there's exactly one, that’s your “inner” folder
        if len(subdirs) == 1:
            inner = subdirs[0]
        else:
            # otherwise try matching the stem of tar_folder (without “.tar”)
            candidate = tar_folder / tar_folder.stem
            inner = candidate if candidate.exists() else subdirs[0]

        # now look for the “fake” folder inside it (or assume inner is already fake)
        fake_dir = inner / "fake"
        if not fake_dir.is_dir():
            fake_dir = inner

        # copy & rename
        for seq_folder in fake_dir.iterdir():
            if not seq_folder.is_dir():
                continue
            for img in seq_folder.iterdir():
                if img.suffix.lower() != ".png":
                    continue

                stem = img.stem
                frame_num = int(stem)  # original frame number
                new_name  = f"{tar_folder.name}_{seq_folder.name}_{frame_num:06d}{img.suffix}"
                shutil.copy2(img, dst_split / new_name)

print("✅ Done copying into:", DST_ROOT)

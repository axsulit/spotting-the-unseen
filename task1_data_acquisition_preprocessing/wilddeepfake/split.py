import os
import shutil
from glob import glob
from sklearn.model_selection import train_test_split

# ─────────────────────────────────────────────────────────────────────────────
# 1. CONFIGURATION: SOURCE & TARGET DIRECTORIES
# ─────────────────────────────────────────────────────────────────────────────

OLD_TRAIN_DIR = r"D:\.THESIS\WildDeepfake\wdf_restructured\fake_train"
OLD_TEST_DIR  = r"D:\.THESIS\WildDeepfake\wdf_restructured\fake_test\archive"

NEW_BASE_DIR  = r"D:\.THESIS\WildDeepfake\final_split"
NEW_TRAIN_DIR = os.path.join(NEW_BASE_DIR, "train")
NEW_VAL_DIR   = os.path.join(NEW_BASE_DIR, "val")
NEW_TEST_DIR  = os.path.join(NEW_BASE_DIR, "test")

# Create subdirectories for fake and real images
for base_dir in [NEW_TRAIN_DIR, NEW_VAL_DIR, NEW_TEST_DIR]:
    os.makedirs(os.path.join(base_dir, "fake"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "real"), exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 2. DESIRED SPLIT SIZES
# ─────────────────────────────────────────────────────────────────────────────

# Direct split sizes
n_train_desired = 67895  # 70% of total
n_val_desired   = 14549  # 15% of total
n_test_desired  = 14549  # 15% of total

total_desired = n_train_desired + n_val_desired + n_test_desired
print(f"Desired split sizes:")
print(f" - Train:      {n_train_desired}")
print(f" - Validation: {n_val_desired}")
print(f" - Test:       {n_test_desired}")
print(f" - Total:      {total_desired}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. LIST FILES IN ORIGINAL FOLDERS
# ─────────────────────────────────────────────────────────────────────────────

train0_files = glob(os.path.join(OLD_TRAIN_DIR, "*.*"))
test0_files  = glob(os.path.join(OLD_TEST_DIR,  "*.*"))

n_train0 = len(train0_files)
n_test0  = len(test0_files)

print(f"\nOriginal counts:")
print(f" - fake_train: {n_train0} images")
print(f" - fake_test:  {n_test0} images")

# ─────────────────────────────────────────────────────────────────────────────
# 4. SPLIT fake_train INTO TRAIN & VAL BASED ON DESIRED SIZES
# ─────────────────────────────────────────────────────────────────────────────

# Calculate fraction of fake_train images to hold out as val
val_frac = n_val_desired / n_train0  # val images out of available train images

# Sanity check val_frac: must be <=1
if val_frac > 1.0:
    raise ValueError(f"Validation fraction {val_frac:.2f} > 1.0: Not enough train images to satisfy val size")

# Split fake_train into train and val
train_paths, val_paths = train_test_split(
    train0_files,
    test_size=val_frac,
    shuffle=True,
    random_state=42
)

print(f"\nSplit fake_train into:")
print(f" - Train:      {len(train_paths)} images")
print(f" - Validation: {len(val_paths)} images")

# ─────────────────────────────────────────────────────────────────────────────
# 5. SUBSAMPLE TEST SET FROM fake_test BASED ON DESIRED SIZE
# ─────────────────────────────────────────────────────────────────────────────

# Subsample n_test_desired images from fake_test (without replacement)
if n_test_desired > n_test0:
    raise ValueError(f"Test desired {n_test_desired} > available fake_test images {n_test0}")

test_paths = train_test_split(
    test0_files,
    train_size=n_test_desired,
    shuffle=True,
    random_state=42
)[0]

print(f"Selected {len(test_paths)} images for test set (from fake_test)")

# ─────────────────────────────────────────────────────────────────────────────
# 6. SUMMARY OF FINAL COUNTS & UNUSED IMAGES
# ─────────────────────────────────────────────────────────────────────────────

total_used = len(train_paths) + len(val_paths) + len(test_paths)
print(f"\nFinal dataset counts:")
print(f" - Train set size:      {len(train_paths)}")
print(f" - Validation set size: {len(val_paths)}")
print(f" - Test set size:       {len(test_paths)}")
print(f" - Total images used:   {total_used}")
print(f" - Total images unused: {n_train0 + n_test0 - total_used} (leftover in original folders)")

# ─────────────────────────────────────────────────────────────────────────────
# 7. COPY FILES INTO NEW FOLDER STRUCTURE
# ─────────────────────────────────────────────────────────────────────────────

def scatter(file_list, target_dir, is_fake=True):
    """
    Copy files to the target directory, organizing them into fake/real subfolders.
    
    Args:
        file_list: List of file paths to copy
        target_dir: Base target directory
        is_fake: Boolean indicating if these are fake images (True) or real images (False)
    """
    subfolder = "fake" if is_fake else "real"
    target_subdir = os.path.join(target_dir, subfolder)
    os.makedirs(target_subdir, exist_ok=True)
    
    for src in file_list:
        filename = os.path.basename(src)
        dst = os.path.join(target_subdir, filename)
        print(f"Moving {filename} from {os.path.dirname(src)} to {target_subdir}...")
        shutil.copy2(src, dst)

print("\nCopying train images...")
scatter(train_paths, NEW_TRAIN_DIR, is_fake=True)

print("Copying validation images...")
scatter(val_paths, NEW_VAL_DIR, is_fake=True)

print("Copying test images...")
scatter(test_paths, NEW_TEST_DIR, is_fake=True)

print("\nAll done!")
print(f"Train images folder:      {NEW_TRAIN_DIR}")
print(f"Validation images folder: {NEW_VAL_DIR}")
print(f"Test images folder:       {NEW_TEST_DIR} (subset of original fake_test)")
print("\nEach folder contains 'fake' and 'real' subfolders for organized storage.")

# preprocess

This folder contains the main data preparation scripts used in our pipeline. Each file is modular and handles a specific stage of preprocessing — from face extraction to dataset augmentation and splitting.

---

## 🧩 Files

### `extract_face_images.py`

Extracts **frontal faces** from:
- video files (e.g. CelebDF, FF++)
- image folders (e.g. WildDeepfake)

Uses head pose estimation (yaw, pitch, roll) to filter non-frontal faces.

**Inputs:**  
- Root folder with videos or images  
- Output directory

**Usage:**
```bash
python extract_face_images.py --dataset_root /path/to/dataset --output_root /path/to/output
```

---

### `add_alterations.py`

Applies visual alterations to images in a folder.

**Supported types:**
- blur
- noise
- splice
- resize
- color

**Inputs:**  
- Type of alteration  
- Input and output directories

**Usage:**
```bash
python add_alterations.py --type blur --input_dir /path/to/images --output_dir /path/to/altered
```

---

### `split_altered_dataset.py`

Splits an altered dataset into train, val, and test sets with:
- 70% training
- 15% validation
- 15% testing

Class balance enforced via oversampling (real to match fake).

**Expected structure:**
```
<alteration_dir>/
├── real/
│   ├── actors/ or real_train/
│   └── ...
└── fake/
    ├── deepfakes/ or fake_test/
    └── ...
```

**Usage:**
```bash
python split_altered_dataset.py --alteration_dir /path/to/blur --output_dir /path/to/split
```

---

## 🛠 Notes

- Make sure all images are `.png` or `.jpg`.
- `real/` and `fake/` folders can contain arbitrary subgroups (e.g. actors, youtube, real_train).
- This structure is compatible with all datasets used in this project.


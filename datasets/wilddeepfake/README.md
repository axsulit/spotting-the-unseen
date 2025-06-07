# WildDeepfake Dataset Processing

This directory contains scripts and utilities for processing and managing the WildDeepfake dataset, a large-scale dataset for deepfake detection research.

## Dataset Overview

The WildDeepfake dataset is a comprehensive collection of real and fake videos, designed for deepfake detection research. The dataset contains various splits for training and testing purposes:

- `real_train`: Real videos for training
- `real_test`: Real videos for testing
- `fake_train`: Fake videos for training
- `fake_test`: Fake videos for testing

## Directory Structure

### Expected Dataset Structure
```
wilddeepfake/
├── real_train/
│   └── extracted/
│       └── [tar_folders]/
│           └── [sequence_folders]/
│               └── [frame_images].png
├── real_test/
├── fake_train/
└── fake_test/
```

### Flattened Dataset Structure
After processing, the dataset is organized in a more manageable flat structure:
```
processed_wilddeepfake/
├── real_train/
│   └── [tar_name]_[sequence_name]_[frame_number].png
├── real_test/
├── fake_train/
└── fake_test/
```

## Processing Scripts

### `flatten_dataset.py`

This script converts the hierarchical dataset structure into a flat directory structure, making it easier to work with the dataset.

#### Features
- Maintains unique identifiers for each image
- Preserves the train/test and real/fake splits
- Handles various directory structures
- Generates consistent file naming patterns

#### Usage
```bash
python flatten_dataset.py /path/to/source /path/to/destination
```

#### Arguments
- `src_root`: Path to the source dataset root folder containing the original WildDeepfake structure
- `dst_root`: Path to the destination root folder where the flattened dataset will be created

#### Output
- Creates a new directory structure with flattened organization
- Images are renamed using the pattern: `[tar_name]_[sequence_name]_[frame_number].png`
- Maintains the original image quality and metadata

## Notes

- The script preserves the original image quality
- All images are in PNG format
- Frame numbers are zero-padded to 6 digits for consistent sorting
- Non-numeric image names are skipped during processing
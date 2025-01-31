## Description

This folder contains scripts and files related to acquiring ff++ (c23) dataset and preprocessing them.

## Contents

- **`face_detector.py`**: Script file for face detection
- **`main.py`**: Main script file. Load this to preprocess data
- **`utils.py`**: Script file for other utility functions
- **`video_processor.py`**: Script file for processing individual video files
- **`main-indiv.py`**: Alternative main file for testing out in one video only
- **`config.yaml`**: Configuration file
- **`requirements.txt`**: Required python packages
- **`README.md`**: Project documentation

## Running Preprocessing

1. Make sure that you have successfully downloaded your dataset.
2. Create a virtual environment (recommended): `python3 -m venv .venv` and activate it: `.venv\Scripts\activate` (Windows).
3. Install required packages: `pip install -r requirements.txt`
4. Modify `config.yaml` and `main.py` based on your dataset configurations.
5. Run script: `python scripts/main.py`

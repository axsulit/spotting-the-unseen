## Description

This folder contains scripts and files related to acquiring ff++ (c23) dataset and preprocessing them.

## Contents

- **`preprocess.py`**: Script file for preprocessing
- **`config.yaml`**: Configuration file
- **`requirements.txt`**: Required python packages
- **`README.md`**: Project documentation

## Data Setup

1. Download the FaceForensics++ dataset here https://github.com/ondyari/FaceForensics.

2. Place the extracted dataset at the path specified in `config.yaml`

## Running Preprocessing

1. Create a virtual environment (recommended): `python3 -m venv .venv` and activate it: `.venv\Scripts\activate` (Windows).

2. Install required packages: `pip install -r requirements.txt`

3. Run the preprocessing script: `python scripts/preprocess.py`

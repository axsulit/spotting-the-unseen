# Preprocessing Utilities

This directory contains utility scripts for preprocessing video and image data, with a focus on face detection, extraction, and quality control. These utilities are designed to prepare data for deepfake detection research.

## Overview

The preprocessing pipeline consists of several components that work together to:
- Detect faces in videos and images
- Extract frontal faces
- Ensure consistent image quality and dimensions
- Process both video files and image sequences

## Components

### 1. Video Processor (`video_processor.py`)

Processes video files to extract high-quality frontal faces.

#### Key Features
- Frame-by-frame processing with configurable intervals
- Frontal face detection using facial landmarks
- Automatic face cropping and resizing
- Quality control through pose estimation

#### Usage
```python
def process_video(
    video_path: Union[str, Path],
    output_dir: Union[str, Path],
    frame_interval: int = 30,
    yaw_threshold: float = 20.0,
    pitch_threshold: float = 20.0,
    roll_threshold: float = 20.0
) -> None:
    """Process a video file to extract frontal faces.

    Args:
        video_path: Path to the input video file
        output_dir: Directory to save extracted faces
        frame_interval: Process every Nth frame
        yaw_threshold: Maximum allowed yaw angle in degrees
        pitch_threshold: Maximum allowed pitch angle in degrees
        roll_threshold: Maximum allowed roll angle in degrees

    Returns:
        None

    Raises:
        FileNotFoundError: If video file doesn't exist
        ValueError: If frame_interval is less than 1
    """
```

### 2. Face Detector (`face_detector.py`)

Core face detection and pose estimation functionality.

#### Features
- Face detection using dlib's HOG-based detector
- Head pose estimation using facial landmarks
- Configurable pose thresholds for quality control
- Efficient face detection with optional scaling

#### API Reference
```python
def detect_faces(frame: np.ndarray, scale_factor: float = 2.0) -> List[dlib.rectangle]:
    """Detect faces in an image using dlib's HOG-based detector.

    Args:
        frame: Input image in BGR format (OpenCV default)
        scale_factor: Factor to scale the image for detection

    Returns:
        List[dlib.rectangle]: List of detected face rectangles
    """

def is_frontal_face(
    landmarks: dlib.full_object_detection,
    yaw_threshold: float,
    pitch_threshold: float,
    roll_threshold: float
) -> bool:
    """Determine if a face is frontal based on head pose angles.

    Args:
        landmarks: dlib facial landmarks object
        yaw_threshold: Maximum allowed yaw angle in degrees
        pitch_threshold: Maximum allowed pitch angle in degrees
        roll_threshold: Maximum allowed roll angle in degrees

    Returns:
        bool: True if face is frontal, False otherwise
    """
```

### 3. Utilities (`utils.py`)

Common utility functions used across the preprocessing pipeline.

#### Features
- Directory management
- Image resizing with aspect ratio preservation
- Model loading and management
- Common image processing operations

#### API Reference
```python
def create_directory(path: Union[str, Path]) -> Path:
    """Create directory if it doesn't exist.

    Args:
        path: Directory path to create

    Returns:
        Path: Path object of created directory
    """

def resize(
    image: np.ndarray,
    target_size: Tuple[int, int] = (256, 256)
) -> np.ndarray:
    """Resize image while preserving aspect ratio.

    Args:
        image: Input image
        target_size: Target dimensions (width, height)

    Returns:
        np.ndarray: Resized image
    """

def load_predictor(model_path: Union[str, Path]) -> dlib.shape_predictor:
    """Load dlib's facial landmark predictor.

    Args:
        model_path: Path to the predictor model file

    Returns:
        dlib.shape_predictor: Loaded predictor model

    Raises:
        FileNotFoundError: If model file doesn't exist
    """
```

### 4. Frontal Face Filter (`filter_frontal_faces.py`)

Filters out non-frontal faces from a directory of face images using yaw angle estimation.

#### Features
- Batch processing of face images
- Yaw angle estimation using facial landmarks
- Automatic separation of non-frontal faces
- Configurable yaw threshold

#### API Reference
```python
def filter_non_frontal_faces(
    folder: Union[str, Path],
    model_path: Union[str, Path],
    yaw_thresh: float = 10.0
) -> None:
    """Filter non-frontal faces from a directory.

    Args:
        folder: Directory containing face images
        model_path: Path to facial landmark predictor model
        yaw_thresh: Maximum allowed yaw angle in degrees

    Returns:
        None

    Note:
        Non-frontal faces are moved to a 'non_frontal' subdirectory
    """
```

## Dependencies

- OpenCV (cv2) >= 4.5.0
- dlib >= 19.22.0
- NumPy >= 1.19.0
- Python >= 3.6

## Required Models

The face detection pipeline requires the following model:
- `shape_predictor_68_face_landmarks.dat`: dlib's facial landmark predictor
  - Default location: `preprocess/utils/shape_predictor_68_face_landmarks.dat`
  - Download from: [dlib's official model repository](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)

## Processing Parameters

### Face Detection
- `scale_factor`: Controls the image scaling for face detection (default: 2.0)
  - Higher values increase detection speed but may miss smaller faces
  - Recommended range: 1.0 to 4.0

### Pose Estimation
- `yaw_threshold`: Maximum allowed yaw angle (default: 20 degrees)
- `pitch_threshold`: Maximum allowed pitch angle (default: 20 degrees)
- `roll_threshold`: Maximum allowed roll angle (default: 20 degrees)
- Adjust these thresholds based on your quality requirements
  - Stricter thresholds (lower values) ensure more frontal faces
  - Relaxed thresholds (higher values) allow more non-frontal faces

### Image Processing
- Output image size: 256x256 pixels
- Format: JPG for video frames, PNG for image sequences
- Face cropping: Square crops centered on detected faces

### Frontal Face Filtering
- `yaw_thresh`: Maximum allowed yaw angle (default: 10 degrees)
  - Higher values allow more non-frontal faces
  - Lower values ensure stricter frontal face selection
  - Recommended range: 5 to 30 degrees

## Error Handling

The utilities include comprehensive error handling for:
- Missing model files (FileNotFoundError)
- Invalid video files (cv2.error)
- Failed face detections (returns empty list)
- Image processing errors (ValueError)
- Invalid parameters (TypeError, ValueError)

## Notes

- All face crops are square and resized to 256x256 pixels
- The pipeline prioritizes frontal faces for better quality
- Non-frontal faces are automatically filtered out
- Frame numbers are zero-padded for consistent sorting
- Non-frontal faces are preserved in a separate directory for potential reuse
- All functions include type hints for better IDE support and code quality
- Documentation follows PEP 257 (Docstring Conventions) and PEP 484 (Type Hints)

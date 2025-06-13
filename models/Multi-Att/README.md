# Multiple Attention Model for Deepfake Detection

This repository is a modified version of the [Multiple Attention Model](https://github.com/yoctta/multiple-attention) implementation, adapted for thesis requirements. The original implementation was created by [yoctta](https://github.com/yoctta).

## About

This project implements a Multiple Attention Model (MAT) for deepfake detection, featuring advanced attention mechanisms and AGDA (Adversarial Gradient Descent Augmentation) for improved performance. The implementation is based on the original work by yoctta, with modifications and enhancements for SPOTTING THE UNSEEN: A COMPREHENSIVE ANALYSIS OF FACE FORGERY DETECTION MODELS.
.

## Features

- Multiple attention mechanisms for enhanced feature extraction
- AGDA (Adversarial Gradient Descent Augmentation) for robust training
- Support for various backbone networks (Xception, EfficientNet)
- Distributed training support
- Comprehensive evaluation metrics
- Flexible configuration system

## Requirements

- Python 3.6+
- PyTorch
- CUDA-compatible GPU
- OpenCV
- NumPy

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/multiple-attention-master.git
cd multiple-attention-master
```

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Project Structure

- `models/`: Contains the model architecture implementations
- `datasets/`: Dataset handling and processing code
- `config.py`: Configuration management
- `train.py`: Training script
- `test.py`: Testing script
- `evaluation.py`: Evaluation metrics and utilities
- `utils.py`: Utility functions
- `AGDA.py`: Adversarial Gradient Descent Augmentation implementation

## Usage

### Training

To train the model, use the following command:

```bash
python train.py --config your_config.yaml
```

### Testing

To evaluate the model:

```bash
python test.py --config your_config.yaml --checkpoint path_to_checkpoint
```

### Configuration

The model can be configured through the `config.py` file. Key parameters include:

- Network architecture settings
- Training hyperparameters
- Dataset configurations
- AGDA parameters
- Loss weights

## Model Architecture

The Multiple Attention Model (MAT) consists of:

- A backbone network (Xception or EfficientNet)
- Multiple attention layers
- Feature extraction and classification heads
- AGDA for adversarial training

## Original Repository

This project is based on the original implementation by yoctta:

- Original Repository: [multiple-attention](https://github.com/yoctta/multiple-attention)
- Original Author: [yoctta](https://github.com/yoctta)

## Thesis Modifications

This version includes modifications and enhancements made for thesis research purposes, including:

- Custom dataset handling
- Modified training pipeline
- Enhanced evaluation metrics
- Additional configuration options

## Acknowledgments

- Original implementation by [yoctta](https://github.com/yoctta)
- All contributors to the original repository
- Thesis advisors and research team

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

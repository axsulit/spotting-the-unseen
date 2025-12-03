# FreqNet Sample Test Script

This script (`freqnet_sample_test.py`) is a modified version of the FreqNet test script designed to visualize broken cross-scale consistency in FreqNet's frequency domain processing. It demonstrates how the model's multi-resolution dependencies and frequency band interactions are affected by image degradation.

## Purpose

The script addresses the research finding that frequency models like FreqNet deteriorate more rapidly at higher blur levels due to broken cross-scale consistency. It visualizes this phenomenon by:

1. **Sampling Images**: Collects random samples from each prediction category (TP, TN, FP, FN)
2. **Frequency Analysis**: Shows how FreqNet's frequency domain operations break down
3. **Cross-Scale Visualization**: Demonstrates the disruption of multi-resolution dependencies

## Key Features

- **Category-based Sampling**: Saves samples organized by prediction accuracy (TP/TN/FP/FN)
- **Frequency Domain Visualization**: Shows magnitude spectra at different processing stages
- **Cross-Scale Analysis**: Visualizes how HFRI, HFRFC, and FCL operations affect frequency content
- **No Metrics Calculation**: Focuses purely on visualization, not performance metrics

## Usage

```bash
python freqnet_sample_test.py \
    --model_path /path/to/freqnet/checkpoint.pth \
    --dataroot /path/to/test/dataset \
    --output_dir freqnet_samples \
    --num_samples 5 \
    --save_frequency_analysis \
    --device cuda
```

### Arguments

- `--model_path`: Path to FreqNet model checkpoint (required)
- `--dataroot`: Path to test dataset root directory (required)
- `--output_dir`: Directory to save sample images (default: 'freqnet_samples')
- `--num_samples`: Number of samples per category (default: 5)
- `--save_frequency_analysis`: Save frequency domain visualizations (flag)
- `--device`: Device to use - 'cuda' or 'cpu' (default: 'cuda')

## Output Structure

```
freqnet_samples/
├── TP/                    # True Positives
│   ├── TP_sample_01_true_1_pred_1_original.png
│   └── ...
├── TN/                    # True Negatives
│   ├── TN_sample_01_true_0_pred_0_original.png
│   └── ...
├── FP/                    # False Positives
│   ├── FP_sample_01_true_0_pred_1_original.png
│   └── ...
├── FN/                    # False Negatives
│   ├── FN_sample_01_true_1_pred_0_original.png
│   └── ...
└── frequency_analysis/     # Frequency domain visualizations
    ├── TP_freq_analysis_01_true_1_pred_1.png
    ├── TN_freq_analysis_01_true_0_pred_0.png
    ├── FP_freq_analysis_01_true_0_pred_1.png
    └── FN_freq_analysis_01_true_1_pred_0.png
```

## Frequency Analysis Visualization

The frequency analysis images show:

1. **Original Image**: The input face image
2. **After HFRI**: High-frequency removal in spatial domain
3. **After HFRFC1**: High-frequency removal in channel domain
4. **After FCL1**: Frequency convolution layer output

Each visualization displays the magnitude spectrum in log scale, showing how frequency content changes through FreqNet's processing pipeline. This helps identify where cross-scale consistency breaks down.

## Research Context

This script supports the research finding that:

> "Frequency models performed slightly lower than spatial, but still higher than the models with attention mechanisms. Despite having a stronger initial performance (F1-score of 71.74% for FreqNet and 74.94% for Hifi-FD), both models deteriorate more rapidly at higher blur levels, likely due to broken cross-scale consistency."

The visualizations demonstrate how blurring disrupts the multi-resolution dependencies and frequency band interactions that FreqNet exploits for accurate classification.

## Requirements

- PyTorch
- NumPy
- Matplotlib
- PIL (Pillow)
- tqdm
- scikit-learn (for metrics, though not used in this script)

## Notes

- The script processes images one at a time for detailed analysis
- Frequency visualizations are saved as high-resolution PNG files
- The script stops early once enough samples are collected from all categories
- No performance metrics are calculated - this is purely for visualization purposes

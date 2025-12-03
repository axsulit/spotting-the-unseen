# FreqNet Cross-Scale Consistency Visualization

This script (`test_visualize_freqnet.py`) provides comprehensive visualization of FreqNet's cross-scale consistency breakdown, demonstrating how blur disrupts the model's frequency-based feature consistency.

## Purpose

The script addresses the research finding that FreqNet's performance deteriorates under Gaussian blur due to broken cross-scale consistency. It provides detailed visual analysis of:

- **HFRI Block Output**: High-frequency removal in spatial domain with FFT magnitude spectrum
- **HFRF Block Output**: Multi-scale feature maps showing spatial and channel frequency processing
- **Cross-Scale Correlation**: Heatmaps showing consistency between different scales
- **FCL Block Output**: Frequency convolution layer outputs with amplitude/phase spectrum

## Key Features

### Multi-Panel Visualization
Each analysis creates a comprehensive 20-panel figure showing:

1. **Input Image**: Original (potentially blurred) face image
2. **HFRI Analysis**: Spatial domain output and FFT magnitude spectrum
3. **HFRF Multi-Scale**: Feature maps at three different scales
4. **Cross-Scale Correlation**: Pearson correlation heatmap between scales
5. **FCL Analysis**: Amplitude/phase spectrum and spatial output
6. **Individual Channels**: Detailed channel-wise visualizations
7. **Summary Statistics**: Quantitative analysis of cross-scale consistency

### Category-Based Organization
- **TP (True Positive)**: Correctly identified fake images
- **TN (True Negative)**: Correctly identified real images  
- **FP (False Positive)**: Incorrectly identified as fake
- **FN (False Negative)**: Incorrectly identified as real

## Usage

```bash
python test_visualize_freqnet.py \
    --model_path /path/to/freqnet/checkpoint.pth \
    --dataroot /path/to/test/dataset \
    --output_dir freqnet_crossscale_analysis \
    --num_samples 3 \
    --device cuda \
    --seed 42
```

### Arguments

- `--model_path`: Path to FreqNet model checkpoint (required)
- `--dataroot`: Path to test dataset root directory (required)
- `--output_dir`: Directory to save visualizations (default: 'freqnet_crossscale_analysis')
- `--num_samples`: Number of samples per category (default: 3)
- `--device`: Device to use - 'cuda' or 'cpu' (default: 'cuda')
- `--seed`: Random seed for reproducible sampling (default: 42)

## Output Structure

```
freqnet_crossscale_analysis/
├── TP/                    # True Positives
│   ├── TP_sample_01_true_1_pred_1_original.png
│   ├── TP_crossscale_analysis_01_true_1_pred_1.png
│   └── ...
├── TN/                    # True Negatives
│   ├── TN_sample_01_true_0_pred_0_original.png
│   ├── TN_crossscale_analysis_01_true_0_pred_0.png
│   └── ...
├── FP/                    # False Positives
│   ├── FP_sample_01_true_0_pred_1_original.png
│   ├── FP_crossscale_analysis_01_true_0_pred_1.png
│   └── ...
└── FN/                    # False Negatives
    ├── FN_sample_01_true_1_pred_0_original.png
    ├── FN_crossscale_analysis_01_true_1_pred_0.png
    └── ...
```

## Visualization Components

### 1. HFRI Block Analysis
- **Spatial Output**: Shows high-frequency removal effect in spatial domain
- **FFT Magnitude**: Log-scale frequency spectrum showing frequency content

### 2. HFRF Multi-Scale Features
- **Scale 1**: After first HFRI + Conv operation
- **Scale 2**: After HFRFC + FCL operation  
- **Scale 3**: After second HFRI + Conv (stride=2)
- **Channel Selection**: Shows representative channels (0, 15, 31)

### 3. Cross-Scale Correlation Heatmap
- **Pearson Correlation**: Computed between flattened feature vectors
- **Color Coding**: Red (high correlation), Blue (low correlation)
- **Interpretation**: 
  - High correlations (>0.7): Strong cross-scale consistency
  - Medium correlations (0.3-0.7): Moderate consistency
  - Low correlations (<0.3): Broken cross-scale consistency

### 4. FCL Block Analysis
- **Amplitude Spectrum**: Shows learned amplitude responses
- **Phase Spectrum**: Shows learned phase relationships
- **Spatial Output**: Final spatial domain representation

### 5. Summary Statistics
- Average cross-scale correlation
- Min/Max correlation values
- Category interpretation guide

## Research Context

This visualization supports the research finding:

> "Frequency models performed slightly lower than spatial, but still higher than the models with attention mechanisms. Despite having a stronger initial performance (F1-score of 71.74% for FreqNet and 74.94% for Hifi-FD), both models deteriorate more rapidly at higher blur levels, likely due to broken cross-scale consistency."

### Expected Observations

**Original Images (Sharp)**:
- Strong cross-scale correlations (>0.7)
- Clear frequency patterns in HFRI/HFRF outputs
- Distinct amplitude/phase relationships in FCL

**Blurred Images**:
- Collapsed cross-scale correlations (<0.3)
- Smoothed frequency patterns
- Weakened amplitude/phase separability

## Technical Implementation

### FreqNet Pipeline Visualization
The script traces through FreqNet's processing pipeline:

1. **HFRI**: `hfreqWH()` - High-frequency removal in spatial domain
2. **Conv1**: First learned convolution with learned weights
3. **HFRFC**: `hfreqC()` - High-frequency removal in channel domain
4. **FCL1**: Frequency convolution layer with real/imaginary convolutions
5. **HFRI2**: Second spatial frequency removal
6. **Conv2**: Second learned convolution (stride=2)
7. **HFRFC2**: Second channel frequency removal
8. **FCL2**: Second frequency convolution layer

### Correlation Computation
Cross-scale correlation is computed using Pearson correlation coefficient:
```python
corr, _ = pearsonr(feat1.flatten(), feat2.flatten())
```

### Visualization Techniques
- **Log Scale**: Applied to frequency magnitudes for better visualization
- **Color Maps**: 
  - `hot`: Frequency magnitudes
  - `plasma`: Feature maps
  - `RdBu_r`: Correlation heatmaps
  - `hsv`: Phase spectra
- **High Resolution**: 150 DPI for publication-quality figures

## Requirements

- PyTorch
- NumPy
- Matplotlib
- Seaborn
- PIL (Pillow)
- tqdm
- scikit-learn
- scipy

## Notes

- The script processes images one at a time for detailed analysis
- Random sampling ensures diverse examples from each category
- Reproducible results with seed control
- No performance metrics calculated - pure visualization focus
- Comprehensive logging of predictions and confidence scores

## Interpretation Guide

### Cross-Scale Consistency Indicators

**Strong Consistency (Original Images)**:
- Correlation values > 0.7
- Clear frequency patterns across scales
- Distinct amplitude/phase relationships

**Broken Consistency (Blurred Images)**:
- Correlation values < 0.3
- Smoothed/blurred frequency patterns
- Weakened amplitude/phase separability

This visualization provides clear evidence of how Gaussian blur disrupts FreqNet's multi-resolution dependencies and frequency band interactions, explaining the performance drop from 71.74% to 67.25% F1-score at kernel size 25.

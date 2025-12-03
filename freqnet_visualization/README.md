# FreqNet Cross-Scale Consistency Visualization

This folder contains scripts to visualize how Gaussian blur affects the cross-scale consistency in FreqNet's frequency domain processing. The visualization demonstrates why FreqNet's performance degrades with increasing blur levels.

## What is Cross-Scale Consistency?

Cross-scale consistency refers to the relationships between different frequency bands and scales that FreqNet exploits for accurate deepfake detection. When images are blurred, these relationships break down, causing performance degradation.

## FreqNet Operations Visualized

1. **HFRI (High-Frequency Removal in spatial domain)**: `hfreqWH()` - removes high frequencies using 2D FFT
2. **HFRFC (High-Frequency Removal in channel domain)**: `hfreqC()` - removes high frequencies using 1D FFT along channels  
3. **FCL (Frequency Convolution Layers)**: Complex convolution in frequency domain

## Files

- `freqnet_cross_scale_visualizer.py`: Main visualization script
- `run_visualization.py`: Simple runner script for your specific dataset
- `demo_visualization.py`: Demo script that works without PyTorch installation
- `requirements.txt`: Required Python packages
- `README.md`: This file

## Installation

1. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

### Option 1: Run the demo (no PyTorch required)
```bash
python demo_visualization.py
```

This creates synthetic frequency domain features to demonstrate the concept.

### Option 2: Use the simple runner script
```bash
python run_visualization.py
```

This will automatically process images from your dataset:
`D:\ACADEMICS\THESIS\Datasets\final\09_ff40_blur_gaussian_25\train\real`

### Option 3: Use the main script with custom parameters
```bash
python freqnet_cross_scale_visualizer.py --input_dir "D:\ACADEMICS\THESIS\Datasets\final\05_ff40_blur_gaussian_5\train\real" --output_dir "./output" --blur_kernel 25 --max_images 10
```

## Parameters

- `--input_dir`: Directory containing blurred images
- `--output_dir`: Output directory for visualizations (default: `./freqnet_visualizations`)
- `--blur_kernel`: Blur kernel size (5, 10, 15, 20, 25) (default: 25)
- `--max_images`: Maximum number of images to process (default: 10)
- `--device`: Device to use (cpu or cuda) (default: cpu)

## Output Visualizations

For each processed image, two visualization files are generated:

1. **`{filename}_freqnet_analysis_k{blur_kernel}.png`**: Comprehensive frequency domain analysis showing:
   - Original vs blurred images
   - Frequency spectra comparison
   - HFRI and HFRFC processing steps
   - Cross-scale disruption visualization
   - Consistency score

2. **`{filename}_cross_scale_analysis_k{blur_kernel}.png`**: Detailed cross-scale consistency breakdown showing:
   - Multi-scale frequency differences
   - Channel-wise consistency analysis
   - Consistency scores across different scales
   - Overall consistency summary

## Understanding the Results

The visualizations show:

- **Red highlighting**: Areas where cross-scale consistency is broken
- **Consistency scores**: Lower scores indicate more disruption
- **Frequency domain differences**: How blur affects different frequency bands
- **Channel interactions**: How blur disrupts channel-wise frequency relationships

## Why This Matters

FreqNet relies on multi-resolution dependencies and frequency band interactions for accurate classification. When images are blurred:

1. High-frequency details are lost
2. Cross-scale relationships break down
3. Channel-wise frequency interactions are disrupted
4. The model's ability to detect subtle artifacts is compromised

This explains why FreqNet's F1-score drops from 71.74% to 67.25% at kernel size 25, as mentioned in your research.

## Example Output

The visualizations will show frequency domain representations that look like heat maps, with:
- **Hot colors (red/yellow)**: High frequency content
- **Cool colors (blue/purple)**: Low frequency content
- **Red highlighting**: Areas of broken consistency
- **Consistency scores**: Quantitative measures of disruption

This provides visual evidence of how blur disrupts the frequency domain features that FreqNet depends on for accurate deepfake detection.
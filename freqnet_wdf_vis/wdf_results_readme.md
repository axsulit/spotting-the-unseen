# WDF Frequency-Domain Visualization Results

## Instructions

This directory contains visualizations generated for WildDeepFake (WDF) frequency-domain evidence analysis using FreqNet.

## Input Images

### Real Images Loaded:
1. real1.png
2. real10.png
3. real2.png

### Fake Images Loaded:
1. fake1.png
2. fake10.png
3. fake2.png

## Processing Parameters

- **Resize**: 256×256 pixels
- **Center Fraction**: 0.5 (removes central 50% of width & height in frequency domain)
- **Grayscale Conversion**: Applied before processing
- **FFT Normalization**: Log-magnitude normalized to [0, 1]
- **High-Pass Filter**: Zero-out central region, then iFFT reconstruction

## Output Files

1. **wdf_fft_grid.png**: 3 rows × 4 columns showing Real | FFT(Real) | Fake | FFT(Fake)
2. **wdf_hp_grid.png**: 3 rows × 6 columns showing Real | HF(Real) | Real-Diff | Fake | HF(Fake) | Fake-Diff
3. **wdf_fig_captions.txt**: Captions for all generated figures
4. **wdf_results_readme.md**: This file

## High-Frequency Energy Statistics

### Real Images:
- Mean: 6.04e+05
- Std: 6.73e+05
- Min: 5.72e+04
- Max: 2.38e+06

### Fake Images:
- Mean: 9.62e+05
- Std: 4.82e+05
- Min: 2.69e+05
- Max: 2.14e+06

## Notes

- All FFT visualizations use identical color scaling across subplots for fair comparison
- High-pass reconstructions are normalized to [0, 1] for visualization
- Difference images show absolute difference between original and HF reconstruction
- Statistics computed over all images in the input directories (not just visualized samples)

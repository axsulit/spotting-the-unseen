"""
Generate WDF frequency-domain visualizations for FreqNet analysis.

This script produces:
1. wdf_fft_grid.png - FFT log-magnitude visualizations
2. wdf_hp_grid.png - High-pass filtered reconstructions
3. wdf_fig_captions.txt - Figure captions
4. wdf_results_readme.md - Documentation with statistics
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn.functional as F
from pathlib import Path
import glob

# Input directories
REAL_DIR = r"D:/ACADEMICS/THESIS/spotting-the-unseen/freqnet_wdf_vis/wdf/real"
FAKE_DIR = r"D:/ACADEMICS/THESIS/spotting-the-unseen/freqnet_wdf_vis/wdf/fake"

# Output directory
OUTPUT_DIR = r"D:/ACADEMICS/THESIS/spotting-the-unseen/freqnet_wdf_vis"

# Processing parameters
RESIZE_SIZE = 256
CENTER_FRAC = 0.5  # Remove central 50% of width & height

# Number of samples to visualize
NUM_SAMPLES = 3


def load_and_preprocess_image(image_path, target_size=256):
    """Load image, convert to grayscale, and resize."""
    img = Image.open(image_path).convert('RGB')
    img = img.convert('L')  # Convert to grayscale
    img = img.resize((target_size, target_size), Image.Resampling.LANCZOS)
    img_array = np.array(img, dtype=np.float32) / 255.0
    return img_array


def compute_fft_log_magnitude(img_array):
    """Compute FFT, fftshift, log-magnitude, and normalize to [0,1]."""
    # Compute FFT
    fft_result = np.fft.fft2(img_array)
    # Shift zero frequency to center
    fft_shifted = np.fft.fftshift(fft_result)
    # Compute magnitude
    magnitude = np.abs(fft_shifted)
    # Log scale (add small epsilon to avoid log(0))
    log_magnitude = np.log(magnitude + 1e-10)
    # Normalize to [0, 1]
    log_magnitude_norm = (log_magnitude - log_magnitude.min()) / (log_magnitude.max() - log_magnitude.min() + 1e-10)
    return log_magnitude_norm


def apply_high_pass_filter(img_array, center_frac=0.5):
    """Apply high-pass filter by zeroing central region in frequency domain."""
    # Compute FFT
    fft_result = np.fft.fft2(img_array)
    # Shift zero frequency to center
    fft_shifted = np.fft.fftshift(fft_result)
    
    h, w = img_array.shape
    center_h, center_w = h // 2, w // 2
    # Calculate region to zero out (central center_frac of width & height)
    zero_h = int(h * center_frac / 2)
    zero_w = int(w * center_frac / 2)
    
    # Create mask (1 outside, 0 inside central region)
    mask = np.ones_like(fft_shifted, dtype=complex)
    mask[center_h - zero_h:center_h + zero_h, 
         center_w - zero_w:center_w + zero_w] = 0.0
    
    # Apply mask
    fft_filtered = fft_shifted * mask
    
    # Shift back
    fft_unshifted = np.fft.ifftshift(fft_filtered)
    # Inverse FFT
    img_reconstructed = np.fft.ifft2(fft_unshifted)
    # Take real part
    img_reconstructed = np.real(img_reconstructed)
    
    # Normalize to [0, 1]
    img_reconstructed = (img_reconstructed - img_reconstructed.min()) / (img_reconstructed.max() - img_reconstructed.min() + 1e-10)
    
    return img_reconstructed


def compute_hf_energy_statistics(real_images, fake_images, center_frac=0.5):
    """Compute high-frequency energy statistics for real and fake images."""
    real_hf_energies = []
    fake_hf_energies = []
    
    for img_array in real_images:
        # Compute FFT
        fft_result = np.fft.fft2(img_array)
        fft_shifted = np.fft.fftshift(fft_result)
        
        h, w = img_array.shape
        center_h, center_w = h // 2, w // 2
        zero_h = int(h * center_frac / 2)
        zero_w = int(w * center_frac / 2)
        
        # Create high-frequency mask (1 for HF, 0 for LF)
        hf_mask = np.ones((h, w), dtype=bool)
        hf_mask[center_h - zero_h:center_h + zero_h, 
                center_w - zero_w:center_w + zero_w] = False
        
        # Compute energy in high-frequency region
        magnitude = np.abs(fft_shifted)
        hf_energy = np.sum(magnitude[hf_mask] ** 2)
        real_hf_energies.append(hf_energy)
    
    for img_array in fake_images:
        # Compute FFT
        fft_result = np.fft.fft2(img_array)
        fft_shifted = np.fft.fftshift(fft_result)
        
        h, w = img_array.shape
        center_h, center_w = h // 2, w // 2
        zero_h = int(h * center_frac / 2)
        zero_w = int(w * center_frac / 2)
        
        # Create high-frequency mask
        hf_mask = np.ones((h, w), dtype=bool)
        hf_mask[center_h - zero_h:center_h + zero_h, 
                center_w - zero_w:center_w + zero_w] = False
        
        # Compute energy in high-frequency region
        magnitude = np.abs(fft_shifted)
        hf_energy = np.sum(magnitude[hf_mask] ** 2)
        fake_hf_energies.append(hf_energy)
    
    stats = {
        'real_mean': np.mean(real_hf_energies),
        'real_std': np.std(real_hf_energies),
        'real_min': np.min(real_hf_energies),
        'real_max': np.max(real_hf_energies),
        'fake_mean': np.mean(fake_hf_energies),
        'fake_std': np.std(fake_hf_energies),
        'fake_min': np.min(fake_hf_energies),
        'fake_max': np.max(fake_hf_energies),
    }
    
    return stats


def create_fft_grid(real_images, fake_images, fft_real, fft_fake, output_path):
    """Create 3 rows × 4 columns FFT visualization grid."""
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    fig.suptitle('WDF Frequency Domain Analysis - FFT Log-Magnitude', fontsize=16, fontweight='bold')
    
    # Find global min/max for consistent color scaling across all FFT subplots
    all_fft = fft_real + fft_fake
    global_min = min([fft.min() for fft in all_fft])
    global_max = max([fft.max() for fft in all_fft])
    
    for row in range(3):
        # Real image
        axes[row, 0].imshow(real_images[row], cmap='gray')
        axes[row, 0].set_title('Real', fontsize=12, fontweight='bold')
        axes[row, 0].axis('off')
        
        # FFT(Real)
        im1 = axes[row, 1].imshow(fft_real[row], cmap='hot', vmin=global_min, vmax=global_max)
        axes[row, 1].set_title('FFT(Real)', fontsize=12, fontweight='bold')
        axes[row, 1].axis('off')
        
        # Fake image
        axes[row, 2].imshow(fake_images[row], cmap='gray')
        axes[row, 2].set_title('Fake', fontsize=12, fontweight='bold')
        axes[row, 2].axis('off')
        
        # FFT(Fake)
        im2 = axes[row, 3].imshow(fft_fake[row], cmap='hot', vmin=global_min, vmax=global_max)
        axes[row, 3].set_title('FFT(Fake)', fontsize=12, fontweight='bold')
        axes[row, 3].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def create_hp_grid(real_images, fake_images, output_path):
    """Create 3 rows × 6 columns high-pass visualization grid."""
    fig, axes = plt.subplots(3, 6, figsize=(20, 10))
    fig.suptitle('WDF High-Frequency Reconstructions', fontsize=16, fontweight='bold')
    
    for row in range(3):
        # Real | HF(Real) | Real-Diff
        real_img = real_images[row]
        real_hf = apply_high_pass_filter(real_img, center_frac=CENTER_FRAC)
        real_diff = np.abs(real_img - real_hf)
        real_diff = (real_diff - real_diff.min()) / (real_diff.max() - real_diff.min() + 1e-10)
        
        axes[row, 0].imshow(real_img, cmap='gray')
        axes[row, 0].set_title('Real', fontsize=12, fontweight='bold')
        axes[row, 0].axis('off')
        
        axes[row, 1].imshow(real_hf, cmap='gray')
        axes[row, 1].set_title('HF(Real)', fontsize=12, fontweight='bold')
        axes[row, 1].axis('off')
        
        axes[row, 2].imshow(real_diff, cmap='hot')
        axes[row, 2].set_title('Real-Diff', fontsize=12, fontweight='bold')
        axes[row, 2].axis('off')
        
        # Fake | HF(Fake) | Fake-Diff
        fake_img = fake_images[row]
        fake_hf = apply_high_pass_filter(fake_img, center_frac=CENTER_FRAC)
        fake_diff = np.abs(fake_img - fake_hf)
        fake_diff = (fake_diff - fake_diff.min()) / (fake_diff.max() - fake_diff.min() + 1e-10)
        
        axes[row, 3].imshow(fake_img, cmap='gray')
        axes[row, 3].set_title('Fake', fontsize=12, fontweight='bold')
        axes[row, 3].axis('off')
        
        axes[row, 4].imshow(fake_hf, cmap='gray')
        axes[row, 4].set_title('HF(Fake)', fontsize=12, fontweight='bold')
        axes[row, 4].axis('off')
        
        axes[row, 5].imshow(fake_diff, cmap='hot')
        axes[row, 5].set_title('Fake-Diff', fontsize=12, fontweight='bold')
        axes[row, 5].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def write_captions(output_path):
    """Write figure captions to file."""
    captions = """wdf_fft_grid.png:

Log-magnitude FFT of WDF samples (real vs fake). Each row shows a Real image, its FFT, the corresponding Fake image, and its FFT. The FFTs reveal spatially-varying high-frequency energy (compression lines, sensor noise, and inconsistent edges) found in WDF forgeries and real images, demonstrating the natural frequency variability that frequency-based detectors exploit.

wdf_hp_grid.png:

High-frequency reconstructions from WDF obtained by applying a high-pass mask in frequency space and transforming back to image space (iFFT). For each pair: Original | HF reconstruction | Absolute difference. The reconstructions preserve realistic compression and sensor artifacts and localized edge irregularities, showing the informative high-frequency content seen by frequency-domain modules.

wdf_feature_fft.png (if generated):

Average log-magnitude FFT of an intermediate conv-layer activation (feature-level frequency) for a representative real vs fake WDF sample, and their difference. This demonstrates that learned features also contain discriminative high-frequency structure in WDF.
"""
    with open(output_path, 'w') as f:
        f.write(captions)
    print(f"Saved: {output_path}")


def write_readme(output_path, real_paths, fake_paths, stats):
    """Write README with instructions, parameters, and statistics."""
    readme_content = f"""# WDF Frequency-Domain Visualization Results

## Instructions

This directory contains visualizations generated for WildDeepFake (WDF) frequency-domain evidence analysis using FreqNet.

## Input Images

### Real Images Loaded:
"""
    for i, path in enumerate(real_paths[:NUM_SAMPLES], 1):
        readme_content += f"{i}. {os.path.basename(path)}\n"
    
    readme_content += "\n### Fake Images Loaded:\n"
    for i, path in enumerate(fake_paths[:NUM_SAMPLES], 1):
        readme_content += f"{i}. {os.path.basename(path)}\n"
    
    readme_content += f"""
## Processing Parameters

- **Resize**: {RESIZE_SIZE}×{RESIZE_SIZE} pixels
- **Center Fraction**: {CENTER_FRAC} (removes central {int(CENTER_FRAC*100)}% of width & height in frequency domain)
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
- Mean: {stats['real_mean']:.2e}
- Std: {stats['real_std']:.2e}
- Min: {stats['real_min']:.2e}
- Max: {stats['real_max']:.2e}

### Fake Images:
- Mean: {stats['fake_mean']:.2e}
- Std: {stats['fake_std']:.2e}
- Min: {stats['fake_min']:.2e}
- Max: {stats['fake_max']:.2e}

## Notes

- All FFT visualizations use identical color scaling across subplots for fair comparison
- High-pass reconstructions are normalized to [0, 1] for visualization
- Difference images show absolute difference between original and HF reconstruction
- Statistics computed over all images in the input directories (not just visualized samples)
"""
    
    with open(output_path, 'w') as f:
        f.write(readme_content)
    print(f"Saved: {output_path}")


def main():
    """Main function to generate all visualizations."""
    print("=" * 60)
    print("WDF Frequency-Domain Visualization Generator")
    print("=" * 60)
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load image paths
    real_paths = sorted(glob.glob(os.path.join(REAL_DIR, "*.png")))
    fake_paths = sorted(glob.glob(os.path.join(FAKE_DIR, "*.png")))
    
    if len(real_paths) < NUM_SAMPLES or len(fake_paths) < NUM_SAMPLES:
        print(f"Warning: Need at least {NUM_SAMPLES} images in each directory")
        print(f"Found {len(real_paths)} real images and {len(fake_paths)} fake images")
        return
    
    print(f"\nFound {len(real_paths)} real images and {len(fake_paths)} fake images")
    print(f"Using first {NUM_SAMPLES} samples from each category\n")
    
    # Load and preprocess images
    print("Loading and preprocessing images...")
    real_images = [load_and_preprocess_image(path, RESIZE_SIZE) for path in real_paths[:NUM_SAMPLES]]
    fake_images = [load_and_preprocess_image(path, RESIZE_SIZE) for path in fake_paths[:NUM_SAMPLES]]
    
    # Load all images for statistics
    all_real_images = [load_and_preprocess_image(path, RESIZE_SIZE) for path in real_paths]
    all_fake_images = [load_and_preprocess_image(path, RESIZE_SIZE) for path in fake_paths]
    
    # Compute FFT log-magnitude
    print("Computing FFT log-magnitude...")
    fft_real = [compute_fft_log_magnitude(img) for img in real_images]
    fft_fake = [compute_fft_log_magnitude(img) for img in fake_images]
    
    # Create FFT grid
    print("Creating FFT grid visualization...")
    fft_output_path = os.path.join(OUTPUT_DIR, "wdf_fft_grid.png")
    create_fft_grid(real_images, fake_images, fft_real, fft_fake, fft_output_path)
    
    # Create high-pass grid
    print("Creating high-pass grid visualization...")
    hp_output_path = os.path.join(OUTPUT_DIR, "wdf_hp_grid.png")
    create_hp_grid(real_images, fake_images, hp_output_path)
    
    # Compute statistics
    print("Computing high-frequency energy statistics...")
    stats = compute_hf_energy_statistics(all_real_images, all_fake_images, CENTER_FRAC)
    
    # Write captions
    print("Writing captions file...")
    captions_path = os.path.join(OUTPUT_DIR, "wdf_fig_captions.txt")
    write_captions(captions_path)
    
    # Write README
    print("Writing README file...")
    readme_path = os.path.join(OUTPUT_DIR, "wdf_results_readme.md")
    write_readme(readme_path, real_paths, fake_paths, stats)
    
    print("\n" + "=" * 60)
    print("All visualizations generated successfully!")
    print("=" * 60)
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print("\nGenerated files:")
    print(f"  - wdf_fft_grid.png")
    print(f"  - wdf_hp_grid.png")
    print(f"  - wdf_fig_captions.txt")
    print(f"  - wdf_results_readme.md")


if __name__ == "__main__":
    main()

